#!/usr/bin/env python3
"""
Multi-tenant Heterogeneous Workload Test
Addresses Reviewer Point 2: "Individualized memory effects"
"""
import hashlib
import numpy as np
import logging
from comprehensive_evaluation_system import run_single_experiment_impl, ExperimentConfig
from enhanced_benchmark_loader import EnhancedBenchmarkLoader
from fixed_agent_implementation import FixedAgent
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_heterogeneous_tenants(num_tenants=10):
    """
    Create tenants with DIFFERENT access patterns (heterogeneous)
    Each tenant has unique seed and query distribution
    """
    tenant_profiles = [
        {"type": "accountant", "alpha": 1.9, "preferred_category": "database"},
        {"type": "traveler", "alpha": 1.5, "preferred_category": "api"},
        {"type": "researcher", "alpha": 1.1, "preferred_category": "external"},
        {"type": "customer_service", "alpha": 1.6, "preferred_category": "database"},
        {"type": "monitoring", "alpha": 2.0, "preferred_category": "api"}
    ]
    
    tenants = []
    for tenant_id in range(num_tenants):
        profile = tenant_profiles[tenant_id % len(tenant_profiles)]
        
        # UNIQUE SEED per tenant
        tenant_seed = 42 + int(hashlib.md5(
            f"tenant_{tenant_id}_{profile['type']}".encode()
        ).hexdigest()[:8], 16) % 10000
        
        tenants.append({
            "tenant_id": tenant_id,
            "profile_type": profile['type'],
            "zipf_alpha": profile['alpha'],
            "preferred_category": profile['preferred_category'],
            "seed": tenant_seed
        })
    
    return tenants

def generate_tenant_workload(tenant_config, num_queries=5000):
    """Generate workload specific to tenant's profile"""
    loader = EnhancedBenchmarkLoader(
        num_questions=num_queries,
        seed=tenant_config['seed'],  # UNIQUE SEED!
        distribution="zipfian",
        zipf_alpha=tenant_config['zipf_alpha']
    )
    
    questions = loader.generate_full_benchmark(total_samples=num_queries)
    
    # Filter to preferred category (if specified)
    preferred = tenant_config.get('preferred_category')
    if preferred:
        filtered = [q for q in questions if q.get('category', '').startswith(preferred)]
        # Mix with 30% other categories
        others = [q for q in questions if not q.get('category', '').startswith(preferred)]
        mixed = filtered[:int(num_queries * 0.7)] + others[:int(num_queries * 0.3)]
        questions = mixed[:num_queries]
    
    return questions

def run_multitenant_heterogeneous_test():
    """Test with heterogeneous tenant workloads"""
    logger.info("="*70)
    logger.info("MULTI-TENANT HETEROGENEOUS WORKLOAD TEST")
    logger.info("="*70)
    
    # Create heterogeneous tenants
    tenants = create_heterogeneous_tenants(num_tenants=10)
    
    logger.info(f"\nCreated {len(tenants)} heterogeneous tenants:")
    for t in tenants:
        logger.info(f"  Tenant {t['tenant_id']}: {t['profile_type']} "
                   f"(α={t['zipf_alpha']}, seed={t['seed']})")
    
    # Run evaluation
    config = ExperimentConfig(
        name="full_system",
        tool_cache_enabled=True,
        workflow_cache_enabled=True,
        adaptive_ttl_enabled=True,
        description="Multi-tenant"
    )
    
    results = []
    
    for tenant in tenants:
        logger.info(f"\nProcessing Tenant {tenant['tenant_id']} ({tenant['profile_type']})...")
        
        # Generate tenant-specific workload
        questions = generate_tenant_workload(tenant, num_queries=5000)
        
        # Run evaluation
        agent = FixedAgent(
            use_real_apis=False,
            enable_tool_cache=True,
            enable_workflow_cache=True,
            enable_adaptive_ttl=True,
            use_redis=True,
            enable_detailed_logging=False
        )
        
        result = run_single_experiment_impl(
            agent, questions, config, 0, f"tenant_{tenant['tenant_id']}"
        )
        del agent
        
        results.append({
            "tenant_id": tenant['tenant_id'],
            "profile_type": tenant['profile_type'],
            "seed": tenant['seed'],
            "alpha": tenant['zipf_alpha'],
            "efficiency": result.get('overall_caching_efficiency', 0.0),
            "tool_hit_rate": result.get('tool_cache_hit_rate', 0.0),
            "workflow_hit_rate": result.get('workflow_cache_hit_rate', 0.0)
        })
        
        logger.info(f"  Efficiency: {results[-1]['efficiency']:.1f}%")
    
    # Analyze heterogeneity
    efficiencies = [r['efficiency'] for r in results]
    variance = np.var(efficiencies)
    
    print("HETEROGENEOUS TENANT RESULTS")
    print(f"Mean Efficiency: {np.mean(efficiencies):.1f}%")
    print(f"Std Dev: {np.std(efficiencies):.1f}%")
    print(f"Range: {np.min(efficiencies):.1f}% - {np.max(efficiencies):.1f}%")
    print(f"Variance: {variance:.1f}")
    # Save
    output = {"tenants": results, "summary": {
        "mean": np.mean(efficiencies),
        "std": np.std(efficiencies),
        "variance": variance
    }}
    
    output_file = f"multitenant_heterogeneous_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)  
    
    logger.info(f"\n Results saved to {output_file}")
    return output

if __name__ == "__main__":
    run_multitenant_heterogeneous_test()
