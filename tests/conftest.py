"""
Pytest configuration and fixtures for Free Fermion Library tests

This module provides common fixtures and configuration for all test modules.
"""

import pytest
import numpy as np
import sys
import os

# Add the src directory to the path so we can import ff
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import ff


@pytest.fixture
def random_seed():
    """Set random seed for reproducible tests"""
    np.random.seed(42)
    yield
    # Reset to random state after test
    np.random.seed(None)


@pytest.fixture
def small_system():
    """Create a small 2-site test system"""
    n_sites = 2
    alphas = ff.jordan_wigner_alphas(n_sites)
    A = np.array([[1.0, 0.5], [0.5, 1.0]])
    H = ff.build_H(n_sites, A)
    return {
        'n_sites': n_sites,
        'alphas': alphas,
        'A': A,
        'H': H
    }


@pytest.fixture
def medium_system():
    """Create a medium 4-site test system"""
    n_sites = 4
    alphas = ff.jordan_wigner_alphas(n_sites)
    # Create a random Hermitian matrix
    np.random.seed(42)
    A = np.random.random((n_sites, n_sites))
    A = A + A.T  # Ensure Hermiticity
    H = ff.build_H(n_sites, A)
    return {
        'n_sites': n_sites,
        'alphas': alphas,
        'A': A,
        'H': H
    }


@pytest.fixture
def test_matrices():
    """Provide standard test matrices for various tests"""
    return {
        # Skew-symmetric matrices for pfaffian tests
        'skew_2x2': np.array([[0, 1], [-1, 0]]),
        'skew_4x4': np.array([[0, 1, 2, 3],
                              [-1, 0, 4, 5],
                              [-2, -4, 0, 6],
                              [-3, -5, -6, 0]]),
        
        # Hermitian matrices for Hamiltonian tests
        'hermitian_2x2': np.array([[1.0, 0.5], [0.5, 1.0]]),
        'hermitian_3x3': np.array([[2.0, 0.5, 0.2],
                                   [0.5, 1.5, 0.3],
                                   [0.2, 0.3, 1.0]]),
        
        # Pairing matrices (antisymmetric)
        'pairing_2x2': np.array([[0, 0.2], [-0.2, 0]]),
        'pairing_3x3': np.array([[0, 0.1, 0.2],
                                 [-0.1, 0, 0.3],
                                 [-0.2, -0.3, 0]]),
    }


@pytest.fixture
def kitaev_chain_params():
    """Parameters for Kitaev chain model tests"""
    return {
        'n_sites': 6,
        'mu': 0.5,      # Chemical potential
        't': 1.0,       # Hopping strength
        'delta': 0.8    # Pairing strength
    }


@pytest.fixture
def tolerance():
    """Standard numerical tolerance for tests"""
    return {
        'rtol': 1e-10,
        'atol': 1e-12
    }


def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "tutorial: marks tests that validate tutorial examples"
    )
    config.addinivalue_line(
        "markers", "example: marks tests that validate documentation examples"
    )


def pytest_collection_modifyitems(config, items):
    """Automatically mark certain tests based on their location/name"""
    for item in items:
        # Mark tutorial tests
        if "tutorial" in item.nodeid:
            item.add_marker(pytest.mark.tutorial)
        
        # Mark example tests
        if "example" in item.nodeid:
            item.add_marker(pytest.mark.example)
        
        # Mark integration tests
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        
        # Mark slow tests (those involving large systems or many iterations)
        if any(keyword in item.name.lower() for keyword in ['performance', 'benchmark', 'large']):
            item.add_marker(pytest.mark.slow)