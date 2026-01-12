import os
import sys
import json
import logging
import time
import random
from datetime import datetime, timedelta
from enhanced_benchmark_loader import EnhancedBenchmarkLoader
from fixed_agent_implementation import FixedAgent
from adaptive_ttl import AdaptiveTTLManager
import scipy.stats as stats
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(f"24hour_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def simulate_diurnal_pattern(hour: int) -> int:
    """
    Simulate workload variation across 24 hours.
    Peak: 8am-6pm (200 queries/batch)
    Off-peak: 6pm-8am (50 queries/batch)
    """
    if 8 <= hour < 18:  # Business hours
        return 2000  # Peak load
    else:
        return 500   # Off-peak load


# Staleness injection helper function
def inject_staleness_events(hour: int, agent_fixed_ttl, agent_adaptive_ttl):
    """
    Simulate real-world data freshness changes by invalidating cache entries.
    This creates staleness events that adaptive TTL can learn from.
    
    Triggered at hours: 4, 8, 12, 16, 20 (5 times per day)
    """
    staleness_hours = [4, 8, 12, 16, 20]
    
    if hour not in staleness_hours:
        return
    
    logger.info(f" STALENESS EVENT at hour {hour}: Simulating data freshness change")

    invalidation_count_fixed = 0
    invalidation_count_adaptive = 0
    
    # Invalidate weather cache (simulate weather actually changing)
    for agent, agent_name in [(agent_fixed_ttl, "fixed_ttl"), (agent_adaptive_ttl, "adaptive_ttl")]:
        if agent.tool_cache and hasattr(agent.tool_cache, 'redis_cache'):
            try:
                if agent.tool_cache.redis_cache and agent.tool_cache.redis_cache.using_redis:
                    # Invalidate weather cache entries
                    weather_pattern = f"*24h_test_{agent_name}*get_weather*"
                    weather_keys = agent.tool_cache.redis_cache.redis_client.keys(weather_pattern)
                    
                    if weather_keys:
                        agent.tool_cache.redis_cache.redis_client.delete(*weather_keys)
                        if agent_name == "fixed_ttl":
                            invalidation_count_fixed += len(weather_keys)
                        else:
                            invalidation_count_adaptive += len(weather_keys)
                    
                    # Invalidate location cache entries (simulate users moving)
                    location_pattern = f"*24h_test_{agent_name}*get_user_location*"
                    location_keys = agent.tool_cache.redis_cache.redis_client.keys(location_pattern)
                    
                    if location_keys:
                        # Only invalidate 20% of location entries (not all users move)
                        keys_to_invalidate = random.sample(list(location_keys), k=max(1, len(location_keys) // 5))
                        agent.tool_cache.redis_cache.redis_client.delete(*keys_to_invalidate)
                        if agent_name == "fixed_ttl":
                            invalidation_count_fixed += len(keys_to_invalidate)
                        else:
                            invalidation_count_adaptive += len(keys_to_invalidate)
                    
            except Exception as e:
                logger.warning(f"Error during staleness injection for {agent_name}: {e}")
    
    logger.info(f"  Fixed TTL: Invalidated {invalidation_count_fixed} cache entries")
    logger.info(f"  Adaptive TTL: Invalidated {invalidation_count_adaptive} cache entries")
def run_24hour_adaptive_ttl_experiment():
    """
    Run 24 SIMULATED hours (not 24 real hours!)
    
    Simulates 24 hours by running 24 sequential batches.
    Each batch represents one hour of workload.
    Expected runtime: 20-40 minutes (not 24 hours!)
    
    NEW: Includes staleness injection at hours 4, 8, 12, 16, 20
    """
    
    logger.info("ADAPTIVE TTL VALIDATION EXPERIMENT")
    logger.info("Configuration:")
    logger.info("  - Workflow TTL: 150s (reduced from 300s)")
    logger.info("  - Tool TTLs: 150-1800s (50% reduction)")
    logger.info("  - Duration: 24 SIMULATED hours (24 sequential batches)")
    logger.info("  - Workload: Diurnal pattern (peak 8am-6pm)")
    logger.info("  - Redis backend: ENABLED (persistent cache)")
    logger.info("  - Adaptive TTL: Redis-persisted state")
    logger.info("  - Staleness Injection: ENABLED at hours 4,8,12,16,20")
    logger.info("  - Expected runtime: 20-40 minutes")
    
    #  Create agents ONCE (persist across all 24 hours)
    logger.info("\n Creating persistent agents (Redis-backed)...")
    
    agent_fixed_ttl = FixedAgent(
        use_real_apis=False,
        enable_tool_cache=True,
        enable_workflow_cache=True,
        enable_adaptive_ttl=False,  # Baseline: fixed TTL
        use_redis=True
    )
    
    agent_adaptive_ttl = FixedAgent(
        use_real_apis=False,
        enable_tool_cache=True,
        enable_workflow_cache=True,
        enable_adaptive_ttl=True,  # Test: adaptive TTL
        use_redis=True
    )
    
    logger.info(" Agents created with Redis backend")
    
    # Verify adaptive TTL is connected to Redis
    if agent_adaptive_ttl.adaptive_manager:
        logger.info(f"✓ Adaptive TTL manager initialized with Redis persistence")
        logger.info(f"  Tools already tracked: {len(agent_adaptive_ttl.adaptive_manager.tool_history)}")
    
    # Prepare benchmark loader
    loader = EnhancedBenchmarkLoader(seed=42)
    
    # Results tracking
    results = {
        "fixed_ttl": {"hourly": [], "cumulative": {}},
        "adaptive_ttl": {"hourly": [], "cumulative": {}},
        "staleness_events": []  #Track when staleness events occurred
    }
    
    start_time = time.time()
    
    # Run 24 iterations (each = 1 simulated hour)
    for hour in range(24):
        logger.info(f"\n{'='*70}")
        logger.info(f"SIMULATED HOUR {hour + 1}/24 - {datetime.now().strftime('%H:%M:%S')}")
        logger.info(f"{'='*70}")
        
        # NEW: Inject staleness events BEFORE running queries
        inject_staleness_events(hour, agent_fixed_ttl, agent_adaptive_ttl)
        if hour in [4, 8, 12, 16, 20]:
            results["staleness_events"].append({
                "hour": hour,
                "timestamp": datetime.now().isoformat(),
                "description": "Weather and location cache invalidated"
            })
        
        # Determine workload for this hour
        queries_this_batch = simulate_diurnal_pattern(hour)
        
        logger.info(f"Generating {queries_this_batch} queries for hour {hour+1}...")
        questions = loader.generate_full_benchmark(queries_this_batch)
        
        # Run both agents
        for agent_name, agent in [("fixed_ttl", agent_fixed_ttl), ("adaptive_ttl", agent_adaptive_ttl)]:
            batch_start = time.time()
            
            # Reset per-batch metrics (but keep cache and adaptive state!)
            # Store old values
            old_tool_calls = agent.metrics.tool_calls_attempted
            old_tool_hits = agent.metrics.tool_cache_hits
            old_workflow_queries = agent.metrics.workflow_queries
            old_workflow_hits = agent.metrics.workflow_cache_hits
            old_tools_saved = agent.metrics.tools_saved_by_workflow
            
            # Log adaptive TTL state BEFORE this hour
            if agent_name == "adaptive_ttl" and agent.adaptive_manager:
                before_stats = agent.adaptive_manager.get_stats()
                logger.info(f"{agent_name} BEFORE hour {hour+1}:")
                logger.info(f"  Tools tracked: {before_stats.get('_summary', {}).get('total_tools_tracked', 0)}")
                logger.info(f"  Total adjustments: {before_stats.get('_summary', {}).get('total_adjustments', 0)}")
            
            # Run queries for this hour
            for q in questions:
                try:
                    agent.run_agent(q["question"], session_id=f"24h_test_{agent_name}")
                except Exception as e:
                    logger.error(f"Query error in {agent_name}: {e}")
            
            batch_time = time.time() - batch_start
            
            # Calculate metrics for THIS HOUR ONLY (delta)
            tool_calls_this_hour = agent.metrics.tool_calls_attempted - old_tool_calls
            tool_hits_this_hour = agent.metrics.tool_cache_hits - old_tool_hits
            workflow_queries_this_hour = agent.metrics.workflow_queries - old_workflow_queries
            workflow_hits_this_hour = agent.metrics.workflow_cache_hits - old_workflow_hits
            tools_saved_this_hour = agent.metrics.tools_saved_by_workflow - old_tools_saved
            
            # Calculate rates for this hour
            tool_hit_rate = (tool_hits_this_hour / tool_calls_this_hour * 100) if tool_calls_this_hour > 0 else 0.0
            workflow_hit_rate = (workflow_hits_this_hour / workflow_queries_this_hour * 100) if workflow_queries_this_hour > 0 else 0.0
            
            total_work_this_hour = tool_calls_this_hour + tools_saved_this_hour
            total_saved_this_hour = tool_hits_this_hour + tools_saved_this_hour
            overall_efficiency = (total_saved_this_hour / total_work_this_hour * 100) if total_work_this_hour > 0 else 0.0
            
            hourly_metrics = {
                "hour": hour,
                "timestamp": datetime.now().isoformat(),
                "queries": queries_this_batch,
                "batch_time": batch_time,
                "tool_hit_rate": tool_hit_rate,
                "workflow_hit_rate": workflow_hit_rate,
                "overall_efficiency": overall_efficiency,
                "staleness_event": hour in [4, 8, 12, 16, 20]  # NEW: Mark hours with staleness
            }
            
            # Track adaptive TTL adjustments (only for adaptive agent)
            if agent_name == "adaptive_ttl" and agent.adaptive_manager:
                after_stats = agent.adaptive_manager.get_stats()
                hourly_metrics["adaptive_ttl_adjustments"] = after_stats.get("_summary", {}).get("total_adjustments", 0)
                hourly_metrics["tools_with_adjustments"] = after_stats.get("_summary", {}).get("tools_with_adjustments", 0)
                
                logger.info(f"{agent_name} AFTER hour {hour+1}:")
                logger.info(f"  Total adjustments: {hourly_metrics['adaptive_ttl_adjustments']}")
                logger.info(f"  Tools with adjustments: {hourly_metrics['tools_with_adjustments']}")
            
            results[agent_name]["hourly"].append(hourly_metrics)
            
            logger.info(f"{agent_name}: Hit rate {hourly_metrics['tool_hit_rate']:.1f}%, "
                       f"Workflow {hourly_metrics['workflow_hit_rate']:.1f}%, "
                       f"Efficiency {hourly_metrics['overall_efficiency']:.1f}%")
        
        # Save checkpoint every 6 hours
        if (hour + 1) % 6 == 0:
            checkpoint_file = f"24hour_checkpoint_hour{hour+1}.json"
            with open(checkpoint_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)  # FIXED: Added default=str for bool serialization
            logger.info(f" Checkpoint saved: {checkpoint_file}")
    
    # Final statistics
    total_time = time.time() - start_time
    logger.info(f"Actual runtime: {total_time/60:.1f} minutes")
    logger.info(f"Staleness events triggered: {len(results['staleness_events'])} times")
    
    # Aggregate results
    for agent_name in ["fixed_ttl", "adaptive_ttl"]:
        hourly_data = results[agent_name]["hourly"]
        
        avg_tool_hit = np.mean([h["tool_hit_rate"] for h in hourly_data])
        avg_workflow_hit = np.mean([h["workflow_hit_rate"] for h in hourly_data])
        avg_efficiency = np.mean([h["overall_efficiency"] for h in hourly_data])
        
        results[agent_name]["cumulative"] = {
            "avg_tool_hit_rate": avg_tool_hit,
            "avg_workflow_hit_rate": avg_workflow_hit,
            "avg_overall_efficiency": avg_efficiency,
            "total_hours": len(hourly_data),
        }
        
        logger.info(f"\n{agent_name.upper()} Results:")
        logger.info(f"  Avg Tool Hit Rate: {avg_tool_hit:.1f}%")
        logger.info(f"  Avg Workflow Hit Rate: {avg_workflow_hit:.1f}%")
        logger.info(f"  Avg Efficiency: {avg_efficiency:.1f}%")
        
        if agent_name == "adaptive_ttl":
            total_adjustments = sum(h.get("adaptive_ttl_adjustments", 0) for h in hourly_data)
            logger.info(f"  Total TTL Adjustments: {total_adjustments}")
    
    # Statistical comparison
    logger.info("\n" + "="*70)
    logger.info("STATISTICAL VALIDATION")
    logger.info("="*70)
    
    fixed_efficiencies = [h["overall_efficiency"] for h in results["fixed_ttl"]["hourly"]]
    adaptive_efficiencies = [h["overall_efficiency"] for h in results["adaptive_ttl"]["hourly"]]
    
    if len(fixed_efficiencies) >= 2 and len(adaptive_efficiencies) >= 2:
        t_stat, p_value = stats.ttest_ind(adaptive_efficiencies, fixed_efficiencies)
        mean_diff = np.mean(adaptive_efficiencies) - np.mean(fixed_efficiencies)
        
        logger.info(f"Two-sample t-test:")
        logger.info(f"  Mean difference: {mean_diff:+.2f}%")
        logger.info(f"  t-statistic: {t_stat:.4f}")
        logger.info(f"  p-value: {p_value:.4f}")
        logger.info(f"  Significant at 95%: {p_value < 0.05}")
        logger.info(f"  Significant at 99%: {p_value < 0.01}")
        
        results["statistical_test"] = {
            "mean_difference": float(mean_diff),
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "significant_95": bool(p_value < 0.05),  # Explicit bool conversion
            "significant_99": bool(p_value < 0.01)   # Explicit bool conversion
        }
    
    # Save final results
    final_file = f"24hour_adaptive_ttl_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(final_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)  # Added default=str
    
    logger.info(f"\n✓ Final results saved: {final_file}")
    
    # Interpretation
    fixed_efficiency = results["fixed_ttl"]["cumulative"]["avg_overall_efficiency"]
    adaptive_efficiency = results["adaptive_ttl"]["cumulative"]["avg_overall_efficiency"]
    improvement = adaptive_efficiency - fixed_efficiency
    
    logger.info(f"CONCLUSION")
    logger.info(f"Adaptive TTL Improvement: {improvement:+.2f}% (efficiency)")
    
    if "statistical_test" in results:
        p_val = results["statistical_test"]["p_value"]
        if p_val < 0.05:
            logger.info(f" Adaptive TTL shows statistically significant benefit (p={p_val:.4f})")
            logger.info(f"   Staleness injection enabled adaptive TTL to learn optimal TTLs")
        else:
            logger.info(f" Adaptive TTL improvement not statistically significant (p={p_val:.4f})")
            logger.info(f"   Possible reasons:")
            logger.info(f"   - Base TTLs already well-tuned for this workload")
            logger.info(f"   - Staleness events may not be frequent enough")
            logger.info(f"   - Would benefit more from dynamic production workloads")
    
    return results


if __name__ == "__main__":
    try:
        results = run_24hour_adaptive_ttl_experiment()
        print("\n 24-hour adaptive TTL experiment completed successfully")
    except KeyboardInterrupt:
        print("\n Experiment interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
