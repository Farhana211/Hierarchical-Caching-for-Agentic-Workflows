#!/usr/bin/env python3
import numpy as np
import scipy.stats as sp_stats
import logging
from typing import Dict, List
from datetime import datetime
import json
import hashlib

# Import your system modules
# Ensure these files are in the same directory or PYTHONPATH
from comprehensive_evaluation_system import (
    ExperimentConfig, run_single_experiment_impl
)
from enhanced_benchmark_loader import EnhancedBenchmarkLoader
from fixed_agent_implementation import FixedAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_configs() -> Dict[str, ExperimentConfig]:
    """
    Create configurations for the comparison.
    Explicitly sets adaptive_ttl_enabled=False to avoid TypeError.
    """
    configs = {}
    
    # 1. No Cache (Baseline)
    configs['no_cache'] = ExperimentConfig(
        name="no_cache",
        tool_cache_enabled=False,
        workflow_cache_enabled=False,
        adaptive_ttl_enabled=False,  # Explicitly set
        description="No caching (baseline)"
    )
    
    # 2. Plan Cache Equivalent (Workflow Only)
    # Simulates plan caching by enabling workflow cache (saves reasoning/planning)
    # but disabling tool cache (executes tools every time, even on plan hit)
    # Note: In our system, workflow cache usually returns results. 
    # To strictly simulate 'Plan Cache' where tools are re-executed, 
    # we would need a special agent mode. 
    # HOWEVER, based on your paper's context, you are comparing against 
    # "Plan Caching" which saves reasoning time. 
    # The closest approximation in your current architecture without writing new agent logic
    # is to run with Workflow=True, Tool=False. 
    # *Crucially*, if your Workflow Cache stores *results* (not just plans), 
    # then this is actually a "Result Cache".
    # To be strictly accurate to "Plan Cache" (Huang et al), it should NOT return results.
    # If your agent returns cached *results* for workflow hits, you should clarify this 
    # in the paper: "We compare against a workflow-level result cache..." 
    # OR, if you want to simulate pure Plan Cache (re-execute tools), 
    # you might need to modify the agent to invalidate results on workflow hit.
    # Assuming for this script we use the standard agent behavior:
    configs['plan_cache'] = ExperimentConfig(
        name="plan_cache",
        tool_cache_enabled=False,
        workflow_cache_enabled=True,
        adaptive_ttl_enabled=False,
        description="Workflow/Plan cache (saves reasoning)"
    )
    
    # 3. Tool Cache Only
    configs['tool_cache'] = ExperimentConfig(
        name="tool_only",
        tool_cache_enabled=True,
        workflow_cache_enabled=False,
        adaptive_ttl_enabled=False,
        description="Tool cache only (ablation)"
    )
    
    # 4. Full Hierarchical System
    configs['full_system'] = ExperimentConfig(
        name="full",
        tool_cache_enabled=True,
        workflow_cache_enabled=True,
        adaptive_ttl_enabled=False,
        description="Hierarchical (tool+workflow)"
    )
    
    return configs

def compute_stats(data: list, n: int) -> Dict:
    """
    Compute mean and 95% Confidence Interval.
    Uses ddof=1 for sample standard deviation.
    Uses dynamic t-distribution critical value based on n.
    """
    if len(data) == 0:
        return {'mean': np.nan, 'ci95': np.nan}
    
    # Standard Error of the Mean (SEM)
    sem = sp_stats.sem(data, ddof=1)
    
    # Critical t-value for 95% CI (two-tailed)
    # df = n - 1
    t_crit = sp_stats.t.ppf(0.975, df=n-1)
    
    return {
        'mean': np.mean(data),
        'ci95': t_crit * sem
    }

def run_comparison(num_runs: int = 15) -> Dict:
    """
    Run the comparison experiment n times.
    Uses unique session IDs to prevent cache contamination.
    """
    logger.info(f"Starting Plan Cache vs Hierarchical Comparison (n={num_runs})")
    
    # Generate one standard benchmark set for all runs (controlled workload)
    # Or generate fresh per run? Paper says "15 independent runs". 
    # Usually better to vary seed per run.
    
    configs = create_configs()
    
    # storage for raw data
    raw_results = {cfg: {
        'total_latency': [], 
        'tool_hit_rate': [], 
        'workflow_hit_rate': [], 
        'efficiency': []
    } for cfg in configs}
    
    for run_id in range(num_runs):
        seed = 42 + run_id * 100  # Unique seed per run
        logger.info(f"--- Run {run_id+1}/{num_runs} (seed={seed}) ---")
        
        # Create loader/questions for this run
        loader = EnhancedBenchmarkLoader(num_questions=15000, seed=seed)
        questions = loader.generate_full_benchmark(total_samples=15000)
        
        for cfg_name, config in configs.items():
            # Unique session ID for strict isolation
            # Format: plancomp_r{run_id}_{config_name}
            session_id = f"plancomp_r{run_id}_{cfg_name}"
            
            # Initialize Agent
            # explicitly set use_redis=True to capture real overhead
            agent = FixedAgent(
                use_real_apis=False,
                use_redis=True, 
                enable_tool_cache=config.tool_cache_enabled,
                enable_workflow_cache=config.workflow_cache_enabled,
                enable_adaptive_ttl=config.adaptive_ttl_enabled
            )
            
            # Run Experiment
            # Note: run_single_experiment_impl usually returns a dict
            result = run_single_experiment_impl(agent, questions, config, seed, session_id)
            
            # Store Metrics
            # Handle potential missing keys safely
            lat = result.get('total_time', result.get('totallatency', np.nan))
            raw_results[cfg_name]['total_latency'].append(lat)
            
            raw_results[cfg_name]['tool_hit_rate'].append(result.get('tool_cache_hit_rate', 0.0))
            raw_results[cfg_name]['workflow_hit_rate'].append(result.get('workflow_cache_hit_rate', 0.0))
            raw_results[cfg_name]['efficiency'].append(result.get('overall_caching_efficiency', 0.0))
            
            # Clean up agent to free connections
            del agent

    # --- Post-Processing & Stats ---
    final_analysis = {}
    
    # Get baseline latencies for speedup calc
    baseline_lats = np.array(raw_results['no_cache']['total_latency'])
    
    for cfg_name, data in raw_results.items():
        # 1. Latency Stats
        lats = np.array(data['total_latency'])
        lat_stats = compute_stats(lats, num_runs)
        
        # 2. Speedup Stats (Relative to baseline run-by-run)
        # Avoid div by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            speedups = baseline_lats / lats
            # Filter NaNs or Infs if any run failed
            valid_speedups = speedups[np.isfinite(speedups)]
        
        # Use length of VALID speedups for proper CI
        speedup_stats = compute_stats(valid_speedups, len(valid_speedups))
        
        # 3. Hit Rate & Efficiency Means
        mean_tool_hit = np.mean(data['tool_hit_rate'])
        mean_workflow_hit = np.mean(data['workflow_hit_rate'])
        mean_efficiency = np.mean(data['efficiency'])
        
        # 4. Significance Test (Paired t-test vs No Cache)
        p_value = np.nan
        if cfg_name != 'no_cache':
            # paired t-test requires same length
            if len(lats) == len(baseline_lats):
                t_stat, p_value = sp_stats.ttest_rel(baseline_lats, lats)
        
        final_analysis[cfg_name] = {
            'latency': lat_stats,
            'speedup': speedup_stats,
            'tool_hit_rate': mean_tool_hit,
            'workflow_hit_rate': mean_workflow_hit,
            'efficiency': mean_efficiency,
            'p_vs_baseline': p_value,
            'n': num_runs
        }

    return final_analysis

if __name__ == "__main__":
    # Ensure Redis env var is set if needed (though passed in script execution usually)
    # os.environ["REDIS_DB"] = "3" 
    
    results = run_comparison(num_runs=15)
    
    # Save to JSON
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"plan_cache_final_{timestamp}.json"
    
    # Convert numpy types to native python for JSON serialization
    def default_serializer(obj):
        if isinstance(obj, np.float64): return float(obj)
        if isinstance(obj, np.int64): return int(obj)
        raise TypeError
        
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2, default=default_serializer)
        
    print(f"\nResults saved to {filename}")
    
    # Print Publication-Ready Table
    print("\n" + "="*120)
    print(f"PLAN CACHE vs HIERARCHICAL COMPARISON (n=15, 95% CI)")
    print("="*120)
    headers = f"{'Configuration':<20} {'Latency (s)':<18} {'Speedup (x)':<18} {'Efficiency %':<15} {'p-value'}"
    print(headers)
    print("-" * 120)
    
    for cfg in ['no_cache', 'plan_cache', 'tool_cache', 'full_system']:
        if cfg not in results: continue
        res = results[cfg]
        
        lat_str = f"{res['latency']['mean']:.2f} ± {res['latency']['ci95']:.2f}"
        
        if cfg == 'no_cache':
            spd_str = "1.00"
            p_str = "-"
        else:
            spd_str = f"{res['speedup']['mean']:.2f} ± {res['speedup']['ci95']:.2f}"
            p_val = res['p_vs_baseline']
            p_str = "< 0.001" if p_val < 0.001 else f"{p_val:.4f}"
            
        eff_str = f"{res['efficiency']:.1f}%"
        
        print(f"{cfg:<20} {lat_str:<18} {spd_str:<18} {eff_str:<15} {p_str}")
        
    print("="*120)
    print("\nMetrc Definitions:")
    print("Efficiency: % of tool executions avoided (Eq. 1)")
    print("Speedup: Ratio of No-Cache Latency to Config Latency")
