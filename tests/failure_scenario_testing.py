import time
import random
import logging
from typing import Dict, List, Any
from dataclasses import dataclass
import numpy as np

from tools import set_failure_mode
from fixed_agent_implementation import FixedAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FailureScenario:
    name: str
    description: str
    failure_type: str
    failure_rate: float
    duration_seconds: float


class FailureScenarioTester:   
    def __init__(self):
        self.results = {}
    
    def test_tool_failures(
        self,
        agent: FixedAgent,
        questions: List[Dict],
        failure_rate: float = 0.05
    ) -> Dict[str, Any]:
        
        logger.info(f"\nTesting tool failures (rate={failure_rate*100}%)...")
        
        set_failure_mode(True, failure_rate)
        
        start_time = time.time()
        successful_queries = 0
        failed_queries = 0
        recovery_times = []
        
        for i, q_data in enumerate(questions):
            try:
                result = agent.run_agent(q_data["question"], session_id="failure_test")
                successful_queries += 1
                
            except Exception as e:
                failed_queries += 1
                logger.warning(f"Query {i} failed: {e}")
                
                recovery_start = time.time()
                try:
                    result = agent.run_agent(q_data["question"], session_id="failure_test")
                    recovery_time = time.time() - recovery_start
                    recovery_times.append(recovery_time)
                    successful_queries += 1
                except:
                    pass
        
        set_failure_mode(False, 0.0)
        
        total_time = time.time() - start_time
        
        return {
            "scenario": "tool_failures",
            "failure_rate": failure_rate,
            "total_queries": len(questions),
            "successful_queries": successful_queries,
            "failed_queries": failed_queries,
            "success_rate": successful_queries / len(questions) * 100,
            "total_time": total_time,
            "avg_recovery_time": np.mean(recovery_times) if recovery_times else 0,
            "median_recovery_time": np.median(recovery_times) if recovery_times else 0,
            "recovery_attempts": len(recovery_times)
        }
    
    def test_cache_corruption(
        self,
        agent: FixedAgent,
        questions: List[Dict],
        corruption_interval: int = 100
    ) -> Dict[str, Any]:
        
        logger.info(f"\nTesting cache corruption (every {corruption_interval} queries)...")
        
        start_time = time.time()
        cache_corruptions = 0
        cache_hits_after = []
        
        for i, q_data in enumerate(questions):
            if i > 0 and i % corruption_interval == 0:
                cache_corruptions += 1
                hits_before = agent.metrics.tool_cache_hits
                
                if hasattr(agent.tool_cache, 'cache') and agent.tool_cache.cache:
                    with agent.tool_cache._lock:
                        keys_to_delete = random.sample(
                            list(agent.tool_cache.cache.keys()),
                            min(10, len(agent.tool_cache.cache))
                        )
                        for key in keys_to_delete:
                            del agent.tool_cache.cache[key]
                    
                    logger.info(f"Corrupted cache: deleted {len(keys_to_delete)} entries")
                
                for j in range(min(20, len(questions) - i)):
                    if i + j < len(questions):
                        agent.run_agent(questions[i + j]["question"], session_id="corruption_test")
                
                hits_after = agent.metrics.tool_cache_hits
                cache_hits_after.append(hits_after - hits_before)
            else:
                agent.run_agent(q_data["question"], session_id="corruption_test")
        
        total_time = time.time() - start_time
        
        return {
            "scenario": "cache_corruption",
            "corruption_events": cache_corruptions,
            "total_time": total_time,
            "avg_hit_rate_degradation": np.mean(cache_hits_after) if cache_hits_after else 0,
            "cache_resilience": "good" if len(cache_hits_after) > 0 and np.mean(cache_hits_after) > 5 else "poor"
        }
    
    def run_all_failure_scenarios(
        self,
        questions: List[Dict],
        save_path: str = "failure_scenario_results.json"
    ) -> Dict[str, Any]:
        
        logger.info("\n" + "="*70)
        logger.info("COMPREHENSIVE FAILURE SCENARIO TESTING")
        logger.info("="*70)
        
        results = {}
        
        agent1 = FixedAgent(use_real_apis=False)
        results["tool_failures_5pct"] = self.test_tool_failures(
            agent1, questions[:200], failure_rate=0.05
        )
        
        agent2 = FixedAgent(use_real_apis=False)
        results["tool_failures_10pct"] = self.test_tool_failures(
            agent2, questions[:200], failure_rate=0.10
        )
        
        agent3 = FixedAgent(use_real_apis=False)
        results["cache_corruption"] = self.test_cache_corruption(
            agent3, questions[:500], corruption_interval=100
        )

        import json
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"\n Failure scenario results saved to {save_path}")
        
        self._print_failure_summary(results)
        
        return results
    
    def _print_failure_summary(self, results: Dict):
        
        print("\n" + "="*70)
        print("FAILURE SCENARIO TESTING SUMMARY")
        print("="*70)
        
        for scenario_name, result in results.items():
            print(f"\n{scenario_name.upper()}")
            print("-"*70)
            
            if "success_rate" in result:
                print(f"  Success Rate: {result['success_rate']:.1f}%")
                print(f"  Recovery Attempts: {result.get('recovery_attempts', 0)}")
                print(f"  Avg Recovery Time: {result.get('avg_recovery_time', 0):.3f}s")
            
            if "cache_resilience" in result:
                print(f"  Cache Resilience: {result['cache_resilience']}")
                print(f"  Corruption Events: {result.get('corruption_events', 0)}")
        
        print("\n" + "="*70)


if __name__ == "__main__":
    from enhanced_benchmark_loader import EnhancedBenchmarkLoader
    
    loader = EnhancedBenchmarkLoader(seed=42)
    questions = loader.generate_full_benchmark(1000)
    
    tester = FailureScenarioTester()
    results = tester.run_all_failure_scenarios(questions)
    
    print("\n Failure scenario testing complete!")