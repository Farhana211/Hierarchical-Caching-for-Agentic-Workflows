import numpy as np
import logging
import random
import hashlib
from comprehensive_evaluation_system import (
    ComprehensiveEvaluator, ExperimentConfig,
    create_agent_with_baseline, run_single_experiment_impl
)
from enhanced_benchmark_loader import EnhancedBenchmarkLoader
from fixed_agent_implementation import FixedAgent
from baseline_systems import get_all_baseline_systems
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_user_personas(num_users=15):
    """Create 5 distinct user personas"""
    
    persona_templates = {
        "accountant": {
            "description": "Repeats financial queries - high repetition",
            "preferred_tools": ["db_query_user", "db_aggregation", "db_join_query"],
            "preferred_entities": {
                "user_ids": [1, 2, 3, 5, 8],
                "departments": ["Finance", "Sales"]
            },
            "concentration": 0.85,
            "repeat_probability": 0.70,
            "expected_efficiency": "85-92%"
        },
        "traveler": {
            "description": "Travel-focused queries - moderate repetition",
            "preferred_tools": ["get_weather", "get_user_location", "calculate_distance"],
            "preferred_entities": {
                "cities": ["New York", "London", "Paris", "Tokyo", "Berlin"],
                "user_ids": [10, 20, 30]
            },
            "concentration": 0.70,
            "repeat_probability": 0.50,
            "expected_efficiency": "70-80%"
        },
        "researcher": {
            "description": "Exploratory queries - minimal repetition",
            "preferred_tools": ["external_slow", "compute_fibonacci", "fs_read_file"],
            "preferred_entities": {
                "search_terms": [f"query_{i}" for i in range(100)],
                "numbers": list(range(1, 50))
            },
            "concentration": 0.25,
            "repeat_probability": 0.10,
            "expected_efficiency": "30-45%"
        },
        "customer_service": {
            "description": "Mixed customer queries - balanced",
            "preferred_tools": ["db_query_user", "get_user_location", "db_query_products"],
            "preferred_entities": {
                "user_ids": list(range(1, 30)),
                "categories": ["Electronics", "Clothing"]
            },
            "concentration": 0.55,
            "repeat_probability": 0.35,
            "expected_efficiency": "55-70%"
        },
        "monitoring_bot": {
            "description": "Extreme repetition - monitoring specific entities",
            "preferred_tools": ["get_weather", "db_query_user", "fs_read_config"],
            "preferred_entities": {
                "cities": ["New York", "London", "Tokyo"],
                "user_ids": [1, 2],
                "config_names": ["app_config"]
            },
            "concentration": 0.95,
            "repeat_probability": 0.85,
            "expected_efficiency": "90-98%"
        }
    }
    
    user_configs = []
    persona_keys = list(persona_templates.keys())
    
    for user_id in range(num_users):
        persona_key = persona_keys[user_id % len(persona_keys)]
        persona = persona_templates[persona_key]
        
        # UNIQUE SEED PER USER using hash
        user_seed = 42 + int(hashlib.md5(
            f"user_{user_id}_{persona_key}".encode()
        ).hexdigest()[:8], 16) % 10000
        
        user_configs.append({
            "user_id": user_id,
            "session_id": f"user_{user_id:03d}",
            "persona_type": persona_key,
            "persona": persona,
            "seed": user_seed  # UNIQUE SEED!
        })
    
    return user_configs

def generate_persona_workload(user_config, num_queries=15000):
    """Generate user-specific workload based on persona"""
    persona = user_config['persona']
    seed = user_config['seed']
    
    np.random.seed(seed)
    random.seed(seed)
    
    # Create benchmark loader
    loader = EnhancedBenchmarkLoader(
        num_questions=num_queries,
        seed=seed,
        distribution="zipfian",
        zipf_alpha=1.5
    )
    
    # Override popular entities
    if 'cities' in persona['preferred_entities']:
        loader.popular_cities = persona['preferred_entities']['cities']
    
    if 'user_ids' in persona['preferred_entities']:
        loader.popular_users = persona['preferred_entities']['user_ids']
    
    # Generate base queries
    questions = loader.generate_full_benchmark(total_samples=num_queries)
    
    # Apply persona-specific modifications
    modified_questions = []
    recent_queries = []
    intra_session_repeats = 0  # Track intra-session repetitions
    
    for i, q in enumerate(questions):
        rand = random.random()
        
        # Repeat recent query
        if rand < persona['repeat_probability'] and len(recent_queries) > 0:
            q_to_use = random.choice(recent_queries[-10:])
            q_to_use = q_to_use.copy()
            q_to_use['id'] = f"persona_repeat_{i}"
            q_to_use['is_persona_repeat'] = True
            q_to_use['is_intra_session_repeat'] = True
            modified_questions.append(q_to_use)
            recent_queries.append(q_to_use)
            intra_session_repeats += 1
            
        # Use preferred tools
        elif rand < persona['concentration']:
            if any(tool in q.get('expected_tools', []) for tool in persona['preferred_tools']):
                modified_questions.append(q)
                recent_queries.append(q)
            else:
                modified_questions.append(q)
                recent_queries.append(q)
        
        # Keep original
        else:
            modified_questions.append(q)
            recent_queries.append(q)
        
        if len(recent_queries) > 50:
            recent_queries.pop(0)
    
    return modified_questions, intra_session_repeats

def evaluate_single_user(user_config, cache_mode="isolated"):
    """
    Evaluate caching performance for single user
    
    Args:
        user_config: User configuration dict
        cache_mode: "isolated" (per-user cache) or "shared" (global cache)
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluating User {user_config['user_id']} ({user_config['persona_type']}) - {cache_mode.upper()} mode")
    logger.info(f"Seed: {user_config['seed']}")
    logger.info(f"Description: {user_config['persona']['description']}")
     
    
    # Generate persona-specific workload
    questions, intra_repeats = generate_persona_workload(user_config, num_queries=15000)
    logger.info(f"Generated {len(questions)} queries ({intra_repeats} intra-session repeats)")
    
    # Create config
    config = ExperimentConfig(
        name="full_system",
        tool_cache_enabled=True,
        workflow_cache_enabled=True,
        adaptive_ttl_enabled=False,
        description="Full system"
    )
    
    # Determine session_id based on cache mode
    if cache_mode == "isolated":
        session_id = user_config['session_id']  # user_000, user_001, etc.
    else:  # shared
        session_id = "shared_population"  # All users share same cache
    
    # Create agent
    agent = FixedAgent(
        use_real_apis=False,
        enable_tool_cache=config.tool_cache_enabled,
        enable_workflow_cache=config.workflow_cache_enabled,
        enable_adaptive_ttl=False,
        use_redis=True,
        enable_detailed_logging=False
    )
    
    # Run evaluation
    result = run_single_experiment_impl(
        agent, 
        questions, 
        config, 
        0, 
        session_id  # Use cache_mode-dependent session_id
    )
    
    del agent
    
    # Calculate intra-session metrics
    total_queries = len(questions)
    intra_session_hit_rate = (intra_repeats / total_queries * 100) if total_queries > 0 else 0
    
    return {
        "user_id": user_config['user_id'],
        "session_id": session_id,
        "cache_mode": cache_mode,
        "seed": user_config['seed'],
        "persona_type": user_config['persona_type'],
        "persona_description": user_config['persona']['description'],
        "expected_efficiency": user_config['persona']['expected_efficiency'],
        "observed_efficiency": result.get('overall_caching_efficiency', 0.0),
        "tool_hit_rate": result.get('tool_cache_hit_rate', 0.0),
        "workflow_hit_rate": result.get('workflow_cache_hit_rate', 0.0),
        "total_queries": total_queries,
        "intra_session_repeats": intra_repeats,
        "intra_session_hit_rate": intra_session_hit_rate
    }

def compute_inter_session_gain(isolated_results, shared_results):
    """
    Compute TRUE inter-session gain:
    Inter-session gain = efficiency(shared) - efficiency(isolated)
    """
    
    # Group by persona type
    isolated_by_persona = {}
    shared_by_persona = {}
    
    for r in isolated_results:
        ptype = r['persona_type']
        if ptype not in isolated_by_persona:
            isolated_by_persona[ptype] = []
        isolated_by_persona[ptype].append(r)
    
    for r in shared_results:
        ptype = r['persona_type']
        if ptype not in shared_by_persona:
            shared_by_persona[ptype] = []
        shared_by_persona[ptype].append(r)
    
    inter_session_analysis = {}
    
    for ptype in isolated_by_persona.keys():
        isolated = isolated_by_persona[ptype]
        shared = shared_by_persona[ptype]
        
        isolated_efficiencies = [r['observed_efficiency'] for r in isolated]
        shared_efficiencies = [r['observed_efficiency'] for r in shared]
        
        # TRUE inter-session gain
        inter_session_gains = [
            shared_efficiencies[i] - isolated_efficiencies[i]
            for i in range(len(isolated))
        ]
        
        inter_session_analysis[ptype] = {
            "num_users": len(isolated),
            "mean_isolated_efficiency": np.mean(isolated_efficiencies),
            "mean_shared_efficiency": np.mean(shared_efficiencies),
            "mean_inter_session_gain": np.mean(inter_session_gains),
            "std_inter_session_gain": np.std(inter_session_gains),
            "interpretation": (
                "Inter-session gain represents TRUE benefit from shared cache "
                "across users with similar patterns"
            )
        }
    
    return inter_session_analysis

def run_personalized_memory_experiment():
    """Main experiment with TWO RUNS: isolated vs shared cache"""
    logger.info("PERSONALIZED MEMORY EFFECTS EXPERIMENT")
    
    # Create user personas with UNIQUE SEEDS
    user_configs = create_user_personas(num_users=15)
    
    # Display persona distribution
    logger.info("\nUser Persona Distribution:")
    persona_counts = {}
    for uc in user_configs:
        ptype = uc['persona_type']
        persona_counts[ptype] = persona_counts.get(ptype, 0) + 1
    
    for ptype, count in persona_counts.items():
        logger.info(f"  {ptype}: {count} users")
    
    # Display seed diversity
    seeds = [uc['seed'] for uc in user_configs]
    logger.info(f"\nSeed diversity: {len(set(seeds))} unique seeds (min={min(seeds)}, max={max(seeds)})")
    
    # RUN 1: ISOLATED CACHE (no sharing between users)
    logger.info("\n" + "="*70)
    logger.info("RUN 1: ISOLATED CACHE (per-user session_id)")
    logger.info("="*70)
    isolated_results = []
    
    for user_config in user_configs:
        result = evaluate_single_user(user_config, cache_mode="isolated")
        isolated_results.append(result)
        
        logger.info(f"  User {result['user_id']} complete: "
                   f"Efficiency={result['observed_efficiency']:.1f}%, "
                   f"Intra-session={result['intra_session_hit_rate']:.1f}%")
    
    # RUN 2: SHARED CACHE (all users share cache)
    logger.info("RUN 2: SHARED CACHE (session_id = shared_population)")
    shared_results = []
    
    for user_config in user_configs:
        result = evaluate_single_user(user_config, cache_mode="shared")
        shared_results.append(result)
        
        logger.info(f"  User {result['user_id']} complete: "
                   f"Efficiency={result['observed_efficiency']:.1f}%, "
                   f"Intra-session={result['intra_session_hit_rate']:.1f}%")
    
    # Compute TRUE INTER-SESSION GAIN
    logger.info("\nComputing inter-session gain (shared - isolated)...")
    inter_session_metrics = compute_inter_session_gain(isolated_results, shared_results)
    
    # Group by persona
    isolated_by_persona = {}
    shared_by_persona = {}
    
    for result in isolated_results:
        ptype = result['persona_type']
        if ptype not in isolated_by_persona:
            isolated_by_persona[ptype] = []
        isolated_by_persona[ptype].append(result)
    
    for result in shared_results:
        ptype = result['persona_type']
        if ptype not in shared_by_persona:
            shared_by_persona[ptype] = []
        shared_by_persona[ptype].append(result)
    
    # Compute statistics
    persona_stats = {}
    for ptype in isolated_by_persona.keys():
        isolated = isolated_by_persona[ptype]
        shared = shared_by_persona[ptype]
        
        isolated_efficiencies = [r['observed_efficiency'] for r in isolated]
        shared_efficiencies = [r['observed_efficiency'] for r in shared]
        intra_rates = [r['intra_session_hit_rate'] for r in isolated]
        
        persona_stats[ptype] = {
            "num_users": len(isolated),
            "description": isolated[0]['persona_description'],
            "expected_range": isolated[0]['expected_efficiency'],
            "mean_isolated_efficiency": np.mean(isolated_efficiencies),
            "std_isolated_efficiency": np.std(isolated_efficiencies),
            "mean_shared_efficiency": np.mean(shared_efficiencies),
            "std_shared_efficiency": np.std(shared_efficiencies),
            "mean_intra_session_rate": np.mean(intra_rates),
            "std_intra_session_rate": np.std(intra_rates)
        }
    
    # Save results
    output = {
        "summary": persona_stats,
        "inter_session_metrics": inter_session_metrics,
        "isolated_results": isolated_results,
        "shared_results": shared_results,
        "metadata": {
            "total_users": len(user_configs),
            "queries_per_user": 15000,
            "total_queries": 15000 * len(user_configs),
            "unique_seeds": len(set(seeds)),
            "experiment_design": "Two-run: isolated vs shared cache",
            "experiment_date": datetime.now().isoformat()
        }
    }
    
    output_file = f"personalized_memory_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(output, indent=2, fp=f)
    
    logger.info(f"\n Personalized memory experiment complete")
    logger.info(f"Results saved to: {output_file}")
    
    # Print summary with TRUE INTER-SESSION GAIN
    print("PERSONALIZED MEMORY EFFECTS - ISOLATED vs SHARED CACHE ANALYSIS")
    print(f"{'Persona':<20} {'Users':<8} {'Isolated':<20} {'Shared':<20} {'Intra-Session':<20} {'Inter-Gain':<20}")
    print("-"*120)
    
    for ptype, stats in persona_stats.items():
        inter_stats = inter_session_metrics[ptype]
        print(f"{ptype:<20} {stats['num_users']:<8} "
              f"{stats['mean_isolated_efficiency']:>6.1f}% ± {stats['std_isolated_efficiency']:.1f}  "
              f"{stats['mean_shared_efficiency']:>6.1f}% ± {stats['std_shared_efficiency']:.1f}  "
              f"{stats['mean_intra_session_rate']:>6.1f}% ± {stats['std_intra_session_rate']:.1f}  "
              f"{inter_stats['mean_inter_session_gain']:>+6.1f}% ± {inter_stats['std_inter_session_gain']:.1f}")
    
    # Variance analysis
    all_isolated = [r['observed_efficiency'] for r in isolated_results]
    all_shared = [r['observed_efficiency'] for r in shared_results]
    print(f"\nIsolated Cache - Mean: {np.mean(all_isolated):.1f}%, Std: {np.std(all_isolated):.1f}%")
    print(f"Shared Cache   - Mean: {np.mean(all_shared):.1f}%, Std: {np.std(all_shared):.1f}%")
    print(f"Overall Inter-Session Gain: {np.mean(all_shared) - np.mean(all_isolated):+.1f}%")
    
    return output

if __name__ == "__main__":
    run_personalized_memory_experiment()
