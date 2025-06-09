#!/usr/bin/env python3
"""
Test Runner for Free Fermion Library

This script provides a comprehensive test runner for the Free Fermion Library
with options for different test categories, verbosity levels, and reporting.

Usage:
    python run_tests.py [options]

Options:
    --category: Run specific test category (tutorials, examples, lib, combinatorics, graph, utils, integration, performance)
    --verbose: Increase verbosity level
    --coverage: Run with coverage reporting
    --benchmark: Run performance benchmarks
    --quick: Run only quick tests (skip slow tests)
    --parallel: Run tests in parallel
    --report: Generate detailed test report
"""

import sys
import os
import argparse
import time
import subprocess
import json
from pathlib import Path

# Add the source directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import pytest
import numpy as np

# Test categories
TEST_CATEGORIES = {
    'tutorials': 'tests/test_tutorials.py',
    'examples': 'tests/test_examples.py',
    'lib': 'tests/test_ff_lib.py',
    'combinatorics': 'tests/test_ff_combinatorics.py',
    'graph': 'tests/test_ff_graph_theory.py',
    'utils': 'tests/test_ff_utils.py',
    'integration': 'tests/test_integration.py',
    'performance': 'tests/test_performance.py'
}

def run_pytest_command(args, test_files=None):
    """Run pytest with specified arguments"""
    cmd = ['python3', '-m', 'pytest']
    
    if test_files:
        if isinstance(test_files, str):
            cmd.append(test_files)
        else:
            cmd.extend(test_files)
    else:
        cmd.append('tests/')
    
    cmd.extend(args)
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    return result

def run_category_tests(category, verbose=False, quick=False):
    """Run tests for a specific category"""
    if category not in TEST_CATEGORIES:
        print(f"Unknown category: {category}")
        print(f"Available categories: {', '.join(TEST_CATEGORIES.keys())}")
        return False
    
    test_file = TEST_CATEGORIES[category]
    args = []
    
    if verbose:
        args.append('-v')
    
    if quick:
        args.extend(['-m', 'not slow'])
    
    print(f"\n{'='*60}")
    print(f"Running {category} tests: {test_file}")
    print(f"{'='*60}")
    
    result = run_pytest_command(args, test_file)
    
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    return result.returncode == 0

def run_all_tests(verbose=False, quick=False, parallel=False):
    """Run all test categories"""
    args = []
    
    if verbose:
        args.append('-v')
    
    if quick:
        args.extend(['-m', 'not slow'])
    
    if parallel:
        try:
            import xdist
            args.extend(['-n', 'auto'])
        except ImportError:
            print("Warning: pytest-xdist not available, running sequentially")
    
    print(f"\n{'='*60}")
    print("Running all tests")
    print(f"{'='*60}")
    
    result = run_pytest_command(args)
    
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    return result.returncode == 0

def run_with_coverage():
    """Run tests with coverage reporting"""
    try:
        import coverage
    except ImportError:
        print("Coverage package not available. Install with: pip install coverage")
        return False
    
    args = [
        '--cov=ff',
        '--cov-report=html',
        '--cov-report=term-missing',
        '--cov-report=xml'
    ]
    
    print(f"\n{'='*60}")
    print("Running tests with coverage")
    print(f"{'='*60}")
    
    result = run_pytest_command(args)
    
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    if result.returncode == 0:
        print("\nCoverage report generated:")
        print("- HTML: htmlcov/index.html")
        print("- XML: coverage.xml")
    
    return result.returncode == 0

def run_benchmarks():
    """Run performance benchmarks"""
    args = [
        'tests/test_performance.py',
        '-v',
        '--benchmark-only',
        '--benchmark-sort=mean'
    ]
    
    print(f"\n{'='*60}")
    print("Running performance benchmarks")
    print(f"{'='*60}")
    
    try:
        result = run_pytest_command(args)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return result.returncode == 0
    except Exception as e:
        print(f"Benchmark plugin not available: {e}")
        print("Install with: pip install pytest-benchmark")
        return False

def generate_test_report():
    """Generate a detailed test report"""
    args = [
        '--html=test_report.html',
        '--self-contained-html',
        '-v'
    ]
    
    print(f"\n{'='*60}")
    print("Generating detailed test report")
    print(f"{'='*60}")
    
    try:
        result = run_pytest_command(args)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        if result.returncode == 0:
            print("\nDetailed test report generated: test_report.html")
        
        return result.returncode == 0
    except Exception as e:
        print(f"HTML report plugin not available: {e}")
        print("Install with: pip install pytest-html")
        return False

def check_dependencies():
    """Check if required dependencies are available"""
    required_packages = ['numpy', 'networkx', 'pytest']
    optional_packages = ['coverage', 'pytest-xdist', 'pytest-benchmark', 'pytest-html']
    
    print("Checking dependencies...")
    
    missing_required = []
    missing_optional = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} (REQUIRED)")
            missing_required.append(package)
    
    for package in optional_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✓ {package}")
        except ImportError:
            print(f"- {package} (optional)")
            missing_optional.append(package)
    
    if missing_required:
        print(f"\nMissing required packages: {', '.join(missing_required)}")
        print("Install with: pip install " + ' '.join(missing_required))
        return False
    
    if missing_optional:
        print(f"\nOptional packages not available: {', '.join(missing_optional)}")
        print("Install with: pip install " + ' '.join(missing_optional))
    
    return True

def main():
    """Main test runner function"""
    parser = argparse.ArgumentParser(
        description="Test runner for Free Fermion Library",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_tests.py                    # Run all tests
    python run_tests.py --category lib     # Run library tests only
    python run_tests.py --coverage         # Run with coverage
    python run_tests.py --benchmark        # Run benchmarks
    python run_tests.py --quick            # Skip slow tests
    python run_tests.py --verbose          # Verbose output
        """
    )
    
    parser.add_argument(
        '--category',
        choices=list(TEST_CATEGORIES.keys()),
        help='Run specific test category'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Increase verbosity level'
    )
    
    parser.add_argument(
        '--coverage',
        action='store_true',
        help='Run with coverage reporting'
    )
    
    parser.add_argument(
        '--benchmark',
        action='store_true',
        help='Run performance benchmarks'
    )
    
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run only quick tests (skip slow tests)'
    )
    
    parser.add_argument(
        '--parallel',
        action='store_true',
        help='Run tests in parallel'
    )
    
    parser.add_argument(
        '--report',
        action='store_true',
        help='Generate detailed HTML test report'
    )
    
    parser.add_argument(
        '--check-deps',
        action='store_true',
        help='Check dependencies and exit'
    )
    
    args = parser.parse_args()
    
    # Check dependencies
    if args.check_deps:
        check_dependencies()
        return
    
    if not check_dependencies():
        print("\nCannot run tests due to missing required dependencies.")
        sys.exit(1)
    
    start_time = time.time()
    success = True
    
    try:
        if args.benchmark:
            success = run_benchmarks()
        elif args.coverage:
            success = run_with_coverage()
        elif args.report:
            success = generate_test_report()
        elif args.category:
            success = run_category_tests(
                args.category,
                verbose=args.verbose,
                quick=args.quick
            )
        else:
            success = run_all_tests(
                verbose=args.verbose,
                quick=args.quick,
                parallel=args.parallel
            )
    
    except KeyboardInterrupt:
        print("\nTests interrupted by user")
        success = False
    except Exception as e:
        print(f"\nError running tests: {e}")
        success = False
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    print(f"\n{'='*60}")
    print(f"Test run completed in {elapsed:.2f} seconds")
    
    if success:
        print("✓ All tests passed!")
        sys.exit(0)
    else:
        print("✗ Some tests failed!")
        sys.exit(1)

if __name__ == '__main__':
    main()