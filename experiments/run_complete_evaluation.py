import os
import sys
import json
import logging
import argparse
from datetime import datetime
import time

from comprehensive_evaluation_system import ComprehensiveEvaluator
from advanced_visualization import JournalFigureGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def quick_test():
    logger.info("="*70)
    logger.info("QUICK TEST MODE (100 questions, 3 runs, parallel)")
    logger.info("="*70)
    
    evaluator = ComprehensiveEvaluator(num_runs=3, seed=42, parallel=True)
    
    results_dir = f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    summary = evaluator.run_ablation_study(
        num_questions=100,
        save_dir=results_dir
    )
    
    print("\n" + "="*70)
    print("QUICK TEST RESULTS")
    print("="*70)
    
    if "full_system" in summary:
        full = summary["full_system"]
        print(f"Full System Tool Hit Rate: {full['tool_cache_hit_rate']['mean']:.1f}%")
        print(f"Full System Workflow Hit Rate: {full['workflow_cache_hit_rate']['mean']:.1f}%")
        print(f"Overall Efficiency: {full['overall_caching_efficiency']['mean']:.1f}%")
        
        if "std" in full["tool_cache_hit_rate"]:
            print(f"Standard Deviation: {full['tool_cache_hit_rate']['std']:.2f}%")
        
        if "pairwise_comparisons" in summary:
            comps = summary["pairwise_comparisons"]
            if "workflow_cache_contribution" in comps:
                wc = comps["workflow_cache_contribution"]
                print(f"Workflow Contribution: {wc['improvement']:.1f}%")
    
    print(f"\nResults saved to: {results_dir}/")
    print("="*70)
    
    return True


def run_full_evaluation(args):
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = args.output or f"results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    if args.clean:
        checkpoint_dir = f"{results_dir}/checkpoints"
        if os.path.exists(checkpoint_dir):
            import shutil
            logger.info(f"Cleaning checkpoint directory: {checkpoint_dir}")
            shutil.rmtree(checkpoint_dir)
            logger.info("Checkpoints cleaned")
    
    logger.info("="*70)
    logger.info("COMPREHENSIVE JOURNAL EVALUATION v2.4")
    logger.info("="*70)
    logger.info(f"Configuration:")
    logger.info(f"  Questions: {args.questions}")
    logger.info(f"  Runs per config: {args.runs}")
    logger.info(f"  Parallel execution: {not args.sequential}")
    logger.info(f"  Results directory: {results_dir}")
    logger.info("="*70)
    
    total_start = time.time()
    
    logger.info("\nSTEP 1: Running comprehensive evaluation...")
    
    estimated_time = args.runs * 9 * 2 if not args.sequential else args.runs * 9 * 10
    logger.info(f"Estimated time: {estimated_time / 60:.0f} minutes")
    
    evaluator = ComprehensiveEvaluator(
        num_runs=args.runs,
        seed=42,
        failure_mode=False,
        parallel=not args.sequential
    )
    
    summary = evaluator.run_ablation_study(
        num_questions=args.questions,
        save_dir=results_dir
    )
    
    logger.info("\nSTEP 2: Generating publication figures...")
    
    summary_files = sorted([f for f in os.listdir(results_dir) if f.startswith("summary_")], reverse=True)
    if summary_files:
        summary_file = os.path.join(results_dir, summary_files[0])
        figures_dir = f"{results_dir}/figures"
        os.makedirs(figures_dir, exist_ok=True)
        
        try:
            viz_generator = JournalFigureGenerator(summary_file)
            viz_generator.generate_all_figures(figures_dir)
        except Exception as e:
            logger.error(f"Visualization generation failed: {e}")
    
    total_time = time.time() - total_start
    
    logger.info("\n" + "="*70)
    logger.info("EVALUATION COMPLETE!")
    logger.info("="*70)
    logger.info(f"Total time: {total_time/3600:.2f} hours")
    logger.info(f"\nResults saved to: {results_dir}/")
    logger.info("\n" + "="*70)
    
    return results_dir


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive evaluation for journal submission v2.4",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test (5 minutes, parallel)
  python run_complete_evaluation.py --test

  # Full journal evaluation (2-3 hours, parallel)
  python run_complete_evaluation.py --full

  # Custom configuration (parallel by default)
  python run_complete_evaluation.py --questions 5000 --runs 10
  
  # Sequential execution (slower but more stable)
  python run_complete_evaluation.py --questions 1000 --runs 10 --sequential
  
  # Clean checkpoints and start fresh
  python run_complete_evaluation.py --full --clean
        """
    )
    
    parser.add_argument("--test", action="store_true",
                       help="Quick test (100 questions, 3 runs, ~5 min)")
    parser.add_argument("--full", action="store_true",
                       help="Full evaluation (10K questions, 10 runs, 2-3 hours)")
    
    parser.add_argument("--questions", type=int, default=10000,
                       help="Number of questions (default: 10000)")
    parser.add_argument("--runs", type=int, default=10,
                       help="Runs per configuration (default: 10)")
    parser.add_argument("--output", type=str,
                       help="Output directory (default: auto-generated)")
    parser.add_argument("--sequential", action="store_true",
                       help="Run sequentially instead of parallel (slower but more stable)")
    parser.add_argument("--clean", action="store_true",
                       help="Clean checkpoint files before starting")
    
    args = parser.parse_args()
    
    try:
        if args.test:
            success = quick_test()
            sys.exit(0 if success else 1)
        
        elif args.full or args.questions or args.runs:
            results_dir = run_full_evaluation(args)
            
            print(f"\n✓ All results saved to: {results_dir}/")
            print("\nRecommended next steps:")
            print("1. Review the paper summary for key claims")
            print("2. Check figures/ for publication-ready visualizations")
            print("3. Verify variance exists (std > 0)")
            print("4. Check workflow contribution is measured correctly")
            
        else:
            parser.print_help()
    
    except KeyboardInterrupt:
        logger.info("\n\nEvaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error during evaluation: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()