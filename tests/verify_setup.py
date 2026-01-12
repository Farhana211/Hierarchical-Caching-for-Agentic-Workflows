import sys
import subprocess
import importlib

def check_python_version():
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        print(f" Python 3.9+ required, found {version.major}.{version.minor}")
        return False
    print(f"Python {version.major}.{version.minor}.{version.micro}")
    return True

def check_package(package_name, import_name=None):
    if import_name is None:
        import_name = package_name
    
    try:
        mod = importlib.import_module(import_name)
        version = getattr(mod, '__version__', 'unknown')
        print(f"{package_name}: {version}")
        return True
    except ImportError:
        print(f" {package_name} not installed")
        return False

def check_redis():

    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, db=0)
        r.ping()
        print("✓ Redis connection successful")
        
        info = r.info()
        print(f"  Redis version: {info.get('redis_version', 'unknown')}")
        print(f"  Memory: {info.get('used_memory_human', 'unknown')}")
        return True
    except Exception as e:
        print(f"Redis connection failed: {e}")
        print("  Start Redis with: redis-server --port 6379")
        return False

def check_file_structure():
    import os
    
    required_files = [
        'requirements.txt',
        'tools.py',
        'enhanced_benchmark_loader.py',
        'comprehensive_evaluation_system.py',
        'advanced_visualization.py',
        'baseline_systems.py',
        'fixed_agent_implementation.py',
        'run_complete_evaluation.py',
        'README.md'
    ]
    
    all_exist = True
    for file in required_files:
        if os.path.exists(file):
            print(f" {file}")
        else:
            print(f"{file} missing")
            all_exist = False
    
    return all_exist

def check_disk_space():
    import shutil
    
    total, used, free = shutil.disk_usage(".")
    free_gb = free // (2**30)
    
    if free_gb < 10:
        print(f"Low disk space: {free_gb} GB free (10 GB recommended)")
        return False
    else:
        print(f"Disk space: {free_gb} GB free")
        return True

def check_memory():
    try:
        import psutil
        mem = psutil.virtual_memory()
        total_gb = mem.total // (2**30)
        available_gb = mem.available // (2**30)
        
        if total_gb < 8:
            print(f"Low RAM: {total_gb} GB total (8 GB minimum)")
            return False
        else:
            print(f"RAM: {total_gb} GB total, {available_gb} GB available")
            return True
    except ImportError:
        print("Cannot check memory (psutil not installed)")
        return True

def main():
    
    print("="*70)
    print("EVALUATION SETUP VERIFICATION")
    print("="*70)
    
    checks = []
    
    print("\n1. Checking Python version...")
    checks.append(check_python_version())
    
    print("\n2. Checking required packages...")
    packages = [
        ('redis', 'redis'),
        ('numpy', 'numpy'),
        ('scipy', 'scipy'),
        ('matplotlib', 'matplotlib'),
        ('seaborn', 'seaborn'),
        ('pandas', 'pandas'),
        ('psutil', 'psutil'),
        ('tqdm', 'tqdm'),
        ('faker', 'faker'),
        ('sklearn', 'sklearn')
    ]
    
    for pkg_name, import_name in packages:
        checks.append(check_package(pkg_name, import_name))
    
    print("\n3. Checking Redis...")
    checks.append(check_redis())
    
    print("\n4. Checking file structure...")
    checks.append(check_file_structure())
    
    print("\n5. Checking system resources...")
    checks.append(check_disk_space())
    checks.append(check_memory())
    
    print("\n" + "="*70)
    
    if all(checks):
        print("ALL CHECKS PASSED")
        print("\nYou can now run:")
        print("  python run_complete_evaluation.py --test")
        print("="*70)
        return 0
    else:
        print("SOME CHECKS FAILED")
        print("\nPlease fix the issues above before running evaluation.")
        print("See README.md for detailed setup instructions.")
        print("="*70)
        return 1

if __name__ == "__main__":
    sys.exit(main())
