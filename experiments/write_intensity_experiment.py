import logging
from comprehensive_evaluation_system import (
    ComprehensiveEvaluator, ExperimentConfig,
    create_agent_with_baseline, run_single_experiment_impl
)
from enhanced_benchmark_loader import EnhancedBenchmarkLoader
from fixed_agent_implementation import FixedAgent
from baseline_systems import get_all_baseline_systems
import json
from datetime import datetime
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_single_config_helper(config_name, questions, session_id):
    """Helper to run single configuration"""
    
    if config_name == "baseline_no_cache":
        config = ExperimentConfig(
            name="baseline_no_cache",
            tool_cache_enabled=False,
            workflow_cache_enabled=False,
            adaptive_ttl_enabled=False,
            description="No caching",
            baseline_system="no_cache"
        )
    elif config_name == "tool_cache_only":
        config = ExperimentConfig(
            name="tool_cache_only",
            tool_cache_enabled=True,
            workflow_cache_enabled=False,
            adaptive_ttl_enabled=False,
            description="Tool cache only"
        )
    elif config_name == "full_system":
        config = ExperimentConfig(
            name="full_system",
            tool_cache_enabled=True,
            workflow_cache_enabled=True,
            adaptive_ttl_enabled=False,
            description="Full system"
        )
    else:
        raise ValueError(f"Unknown config: {config_name}")
    
    # Create agent
    if config.baseline_system:
        baseline_systems = get_all_baseline_systems()
        agent = create_agent_with_baseline(baseline_systems[config.baseline_system])
    else:
        agent = FixedAgent(
            use_real_apis=False,
            enable_tool_cache=config.tool_cache_enabled,
            enable_workflow_cache=config.workflow_cache_enabled,
            enable_adaptive_ttl=config.adaptive_ttl_enabled,
            use_redis=True,
            enable_detailed_logging=False
        )
    
    # Run
    result = run_single_experiment_impl(agent, questions, config, 0, session_id)
    del agent
    return result

def run_write_intensity_experiment():
    """Test caching architecture under different write ratios"""
    
    write_ratios = [0.04, 0.10, 0.20, 0.30]  # 4%, 10%, 20%, 30%
    num_runs = 3  # Multiple runs for statistical validity
    
    results = {}
    baseline_efficiency = None
    
    for write_ratio in write_ratios:
        logger.info(f"\n{'='*70}")
        logger.info(f"Testing write ratio: {write_ratio*100:.0f}%")
        logger.info(f"{'='*70}")
        
        # Generate workload
        benchmark_loader = EnhancedBenchmarkLoader(
            num_questions=15000,
            write_ratio=write_ratio,
            seed=42
        )
        
        questions = benchmark_loader.generate_full_benchmark(total_samples=15000)
        logger.info(f"Generated {len(questions)} queries")
        
        # Count write operations
        write_count = sum(1 for q in questions if q.get('is_write', False))
        actual_write_ratio = write_count / len(questions)
        logger.info(f"Actual write ratio: {actual_write_ratio*100:.1f}%")
        
        # Test configurations
        configs_to_test = ["baseline_no_cache", "tool_cache_only", "full_system"]
        config_results = {}
        
        # Multiple runs per config
        for config_name in configs_to_test:
            logger.info(f"\n  Evaluating: {config_name} ({num_runs} runs)")
            
            runs = []
            for run_id in range(num_runs):
                result = run_single_config_helper(
                    config_name=config_name,
                    questions=questions,
                    session_id=f"write_{int(write_ratio*100)}pct_{config_name}_run{run_id}"
                )
                runs.append(result)
                logger.info(f"    Run {run_id+1}/{num_runs}: Efficiency={result.get('overall_caching_efficiency', 0.0):.1f}%")
            
            # Robust metric extraction with fallbacks
            efficiencies = [r.get('overall_caching_efficiency', 0.0) for r in runs]
            tool_hits = [r.get('tool_cache_hit_rate', 0.0) for r in runs]
            workflow_hits = [r.get('workflow_cache_hit_rate', 0.0) for r in runs]
            speedups = [r.get('speedup', 1.0) for r in runs]
            
            # Invalidation metrics with robust key lookup
            inv_totals = [
                r.get('invalidations_total', r.get('total_invalidations', 0))
                for r in runs
            ]
            inv_per_1k = [
                r.get('invalidations_per_1k', r.get('invalidations_per_1000', 0))
                for r in runs
            ]
            inv_times = [
                r.get('avg_invalidation_time_ms', r.get('avg_invalidation_time', 0))
                for r in runs
            ]
            
            config_results[config_name] = {
                "overall_efficiency": float(np.mean(efficiencies)),
                "overall_efficiency_std": float(np.std(efficiencies, ddof=1)) if num_runs > 1 else 0.0,
                "tool_hit_rate": float(np.mean(tool_hits)),
                "tool_hit_rate_std": float(np.std(tool_hits, ddof=1)) if num_runs > 1 else 0.0,
                "workflow_hit_rate": float(np.mean(workflow_hits)),
                "workflow_hit_rate_std": float(np.std(workflow_hits, ddof=1)) if num_runs > 1 else 0.0,
                "speedup": float(np.mean(speedups)),
                "speedup_std": float(np.std(speedups, ddof=1)) if num_runs > 1 else 0.0,
                "invalidations_total": float(np.mean(inv_totals)),
                "invalidations_per_1k": float(np.mean(inv_per_1k)),
                "avg_invalidation_time_ms": float(np.mean(inv_times))
            }
        
        # Efficiency degradation with correct sign
        if write_ratio == 0.04:
            baseline_efficiency = config_results['full_system']['overall_efficiency']
            efficiency_degradation = 0.0
        else:
            # Positive degradation = performance got worse
            efficiency_degradation = (
                baseline_efficiency - config_results['full_system']['overall_efficiency']
            )
        
        results[f"write_{int(write_ratio*100)}pct"] = {
            "write_ratio": write_ratio,
            "actual_write_ratio": actual_write_ratio,
            "write_count": write_count,
            "num_runs": num_runs,
            "configs": config_results,
            "efficiency_degradation_pp": efficiency_degradation
        }
        
        logger.info(f"\n  Full System Results (mean ± std over {num_runs} runs):")
        logger.info(f"    Efficiency: {config_results['full_system']['overall_efficiency']:.1f}% ± {config_results['full_system']['overall_efficiency_std']:.1f}%")
        logger.info(f"    Degradation: {efficiency_degradation:+.1f}pp")
    
    output_file = f"write_intensity_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n Write-intensity experiment complete")
    logger.info(f"Results saved to: {output_file}")
    
    # Print summary
    print("WRITE-INTENSITY IMPACT SUMMARY")
    print(f"{'Write %':<12} {'Efficiency':<20} {'Tool Hit':<20} {'WF Hit':<20} {'Degradation':<15}")
    for key in sorted(results.keys()):
        data = results[key]
        fs = data['configs']['full_system']
        print(f"{data['write_ratio']*100:>5.0f}%     "
              f"{fs['overall_efficiency']:>6.1f}% ± {fs['overall_efficiency_std']:>4.1f}  "
              f"{fs['tool_hit_rate']:>6.1f}% ± {fs['tool_hit_rate_std']:>4.1f}  "
              f"{fs['workflow_hit_rate']:>6.1f}% ± {fs['workflow_hit_rate_std']:>4.1f}  "
              f"{data['efficiency_degradation_pp']:>+6.1f}pp")
    print("="*90)
    print(f"\nNote: Results averaged over {num_runs} runs with standard deviation")
    
    return results

if __name__ == "__main__":
    run_write_intensity_experiment()
