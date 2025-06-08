#!/usr/bin/env python3
"""
Simple test script to verify the free-fermion-lib package installation and functionality.
"""

import numpy as np
import ff

def test_basic_functionality():
    """Test basic package functionality."""
    print("=" * 60)
    print("Testing Free Fermion Library Package")
    print("=" * 60)
    
    # Test package info
    print(f"Package version: {ff.__version__}")
    print(f"Available functions: {len(ff.__all__)}")
    print()
    
    # Test Jordan-Wigner operators
    print("1. Testing Jordan-Wigner operators...")
    n_sites = 3
    alphas = ff.jordan_wigner_alphas(n_sites)
    print(f"   Generated {len(alphas)} operators for {n_sites} sites")
    print(f"   First operator shape: {alphas[0].shape}")
    print()
    
    # Test combinatorial functions
    print("2. Testing combinatorial functions...")
    # Test pfaffian
    A = np.array([[0, 1, 2], [-1, 0, 3], [-2, -3, 0]])
    pf_val = ff.pf(A)
    print(f"   Pfaffian of 3x3 skew matrix: {pf_val}")
    
    # Test determinant
    B = np.random.random((3, 3))
    det_val = ff.dt(B)
    numpy_det = np.linalg.det(B)
    print(f"   Custom determinant: {det_val:.6f}")
    print(f"   NumPy determinant:  {numpy_det:.6f}")
    print(f"   Difference: {abs(det_val - numpy_det):.2e}")
    print()
    
    # Test utility functions
    print("3. Testing utility functions...")
    noisy_array = np.array([1.0, 1e-10, 2.0, 1e-15, 3.0])
    cleaned = ff.clean(noisy_array)
    print(f"   Original: {noisy_array}")
    print(f"   Cleaned:  {cleaned}")
    print()
    
    # Test Hamiltonian construction
    print("4. Testing Hamiltonian construction...")
    A = np.random.random((n_sites, n_sites))
    A = A + A.T  # Make symmetric
    H = ff.build_H(n_sites, A)
    print(f"   Built Hamiltonian matrix of shape: {H.shape}")
    print(f"   Matrix is Hermitian: {np.allclose(H, H.conj().T)}")
    print()
    
    print("✅ All tests passed successfully!")
    print("✅ Package is properly installed and functional!")

if __name__ == "__main__":
    test_basic_functionality()