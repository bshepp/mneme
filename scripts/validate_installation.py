#!/usr/bin/env python3
"""Validate Mneme installation and environment setup."""

import sys
import importlib
import subprocess
from pathlib import Path


def check_python_version():
    """Check Python version."""
    print("Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"✗ Python {version.major}.{version.minor}.{version.micro} (3.8+ required)")
        return False


def check_package_import(package_name):
    """Check if package can be imported."""
    try:
        module = importlib.import_module(package_name)
        version = getattr(module, "__version__", "unknown")
        print(f"✓ {package_name} {version}")
        return True
    except ImportError as e:
        print(f"✗ {package_name}: {e}")
        return False


def check_mneme_modules():
    """Check Mneme submodules."""
    print("\nChecking Mneme modules...")
    modules = [
        "mneme",
        "mneme.core",
        "mneme.core.field_theory",
        "mneme.core.topology",
        "mneme.core.attractors",
        "mneme.models",
        "mneme.data",
        "mneme.analysis",
        "mneme.utils",
        "mneme.types",
    ]
    
    success = True
    for module in modules:
        try:
            importlib.import_module(module)
            print(f"✓ {module}")
        except ImportError as e:
            print(f"✗ {module}: {e}")
            success = False
    
    return success


def check_dependencies():
    """Check key dependencies."""
    print("\nChecking dependencies...")
    dependencies = [
        "numpy",
        "scipy",
        "pandas",
        "matplotlib",
        "torch",
        "sklearn",
        "yaml",
        "h5py",
        "tqdm",
        "pydantic",
    ]
    
    success = True
    for dep in dependencies:
        if not check_package_import(dep):
            success = False
    
    return success


def check_optional_dependencies():
    """Check optional dependencies."""
    print("\nChecking optional dependencies...")
    optional = [
        ("pysr", "Symbolic regression"),
        ("gudhi", "Topological data analysis"),
        ("pytest", "Testing"),
        ("jupyter", "Notebooks"),
    ]
    
    for dep, description in optional:
        try:
            importlib.import_module(dep)
            print(f"✓ {dep} ({description})")
        except ImportError:
            print(f"⚠ {dep} ({description}) - not installed")


def check_directories():
    """Check project directory structure."""
    print("\nChecking directory structure...")
    directories = [
        "src/mneme",
        "data",
        "experiments",
        "notebooks",
        "tests",
        "docs",
        "config",
    ]
    
    success = True
    for dir_path in directories:
        path = Path(dir_path)
        if path.exists():
            print(f"✓ {dir_path}")
        else:
            print(f"✗ {dir_path} - missing")
            success = False
    
    return success


def check_gpu():
    """Check GPU availability."""
    print("\nChecking GPU support...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  Number of GPUs: {torch.cuda.device_count()}")
        else:
            print("⚠ No CUDA GPU available (CPU mode will be used)")
    except Exception as e:
        print(f"⚠ Could not check GPU: {e}")


def run_simple_test():
    """Run a simple functionality test."""
    print("\nRunning simple functionality test...")
    try:
        import numpy as np
        from mneme.types import Field, validate_field_data, FieldDataSchema
        
        # Create test field
        test_data = np.random.randn(64, 64)
        field = Field(data=test_data, resolution=(64, 64))
        
        # Validate
        schema = FieldDataSchema(shape=(64, 64), dtype="float64")
        is_valid, errors = validate_field_data(test_data, schema)
        
        if is_valid:
            print("✓ Basic functionality test passed")
            return True
        else:
            print(f"✗ Validation failed: {errors}")
            return False
            
    except Exception as e:
        print(f"✗ Functionality test failed: {e}")
        return False


def main():
    """Run all validation checks."""
    print("=" * 50)
    print("Mneme Installation Validation")
    print("=" * 50)
    
    checks = [
        ("Python version", check_python_version),
        ("Directory structure", check_directories),
        ("Mneme modules", check_mneme_modules),
        ("Dependencies", check_dependencies),
        ("Functionality test", run_simple_test),
    ]
    
    results = []
    for name, check_func in checks:
        results.append((name, check_func()))
    
    # Optional checks (don't affect overall success)
    check_optional_dependencies()
    check_gpu()
    
    # Summary
    print("\n" + "=" * 50)
    print("Summary:")
    print("=" * 50)
    
    all_passed = all(result for _, result in results)
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{name}: {status}")
    
    if all_passed:
        print("\n🎉 All validation checks passed!")
        print("Mneme is ready to use.")
        return 0
    else:
        print("\n❌ Some validation checks failed.")
        print("Please run 'pip install -r requirements.txt' and try again.")
        return 1


if __name__ == "__main__":
    sys.exit(main())