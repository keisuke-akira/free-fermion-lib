# Free Fermion Library Test Suite Implementation Plan

## Overview

This document outlines the comprehensive test suite for the Free Fermion Library based on the tutorial and example code from `docs/tutorials.rst` and `docs/examples.rst`.

## Test Structure

### Directory Structure
```
tests/
├── __init__.py                 # Test package initialization
├── conftest.py                 # Pytest configuration and fixtures
├── test_tutorials.py           # Tutorial code validation (START HERE)
├── test_examples.py            # Example code validation
├── test_ff_lib.py              # Core library functionality
├── test_ff_combinatorics.py    # Combinatorial functions
├── test_ff_graph_theory.py     # Graph algorithms
├── test_ff_utils.py            # Utility functions
├── test_integration.py         # End-to-end workflows
├── test_performance.py         # Performance benchmarks
└── test_data/                  # Test matrices and reference data
    ├── __init__.py
    ├── known_matrices.py       # Pre-computed test matrices
    └── reference_results.py    # Expected results for validation
```

## Priority 1: Tutorial Validation Tests (`test_tutorials.py`)

### Test Cases Based on `docs/tutorials.rst`

#### 1. Working with Correlation Matrices (lines 95-123)
```python
def test_correlation_matrices_tutorial():
    """Test the correlation matrices example from tutorials.rst lines 95-123"""
    # Create a system with 4 sites
    n_sites = 4
    alphas = ff.jordan_wigner_alphas(n_sites)
    
    # Build a random Hamiltonian
    A = np.random.random((n_sites, n_sites))
    A = A + A.T  # Ensure Hermiticity
    H = ff.build_H(n_sites, A)
    
    # Generate the ground state
    rho = ff.generate_gaussian_state(n_sites, H, alphas)
    
    # Compute various correlation matrices
    gamma = ff.compute_2corr_matrix(rho, n_sites, alphas)
    cov = ff.compute_cov_matrix(rho, n_sites, alphas)
    
    # Validate results
    assert gamma.shape == (2 * n_sites, 2 * n_sites)
    assert cov.shape == (2 * n_sites, 2 * n_sites)
    assert np.allclose(gamma, gamma.conj().T)  # Should be Hermitian
```

#### 2. Symplectic Eigenvalue Problems (lines 125-139)
```python
def test_symplectic_eigenvalue_tutorial():
    """Test symplectic eigenvalue example from tutorials.rst lines 125-139"""
    n_sites = 4
    A = np.random.random((n_sites, n_sites))
    A = A + A.T  # Ensure Hermiticity
    H = ff.build_H(n_sites, A)
    
    # Diagonalize in symplectic form
    eigenvals, eigenvecs = ff.eigh_sp(H)
    
    # Verify symplectic property
    is_symplectic = ff.is_symp(eigenvecs)
    assert is_symplectic, "Eigenvectors should be symplectic"
    
    # Check canonical form
    is_canonical = ff.check_canonical_form(eigenvals)
    assert is_canonical, "Eigenvalues should be in canonical form"
```

#### 3. Graph Theory Applications (lines 141-160)
```python
def test_graph_theory_tutorial():
    """Test graph theory example from tutorials.rst lines 141-160"""
    # Generate a planar graph
    G = ff.generate_random_planar_graph(8, seed=123)
    
    if G is not None:
        # Apply pfaffian ordering
        pfo_matrix = ff.pfo_algorithm(G, verbose=False)
        
        # Find perfect matchings
        matchings = ff.find_perfect_matchings(G)
        
        # Compute pfaffian (should equal number of matchings)
        pf_value = ff.pf(pfo_matrix)
        
        # Validate results
        assert len(matchings) == int(abs(pf_value)), "Pfaffian should equal number of perfect matchings"
        assert pfo_matrix.shape[0] == len(G.nodes()), "PFO matrix should match graph size"
```

## Priority 2: Example Validation Tests (`test_examples.py`)

### Test Cases Based on `docs/examples.rst`

#### 1. Simple Pfaffian Calculation (lines 9-30)
```python
def test_simple_pfaffian_example():
    """Test simple pfaffian example from examples.rst lines 9-30"""
    # Create a 4x4 skew-symmetric matrix
    A = np.array([[0, 1, 2, 3],
                  [-1, 0, 4, 5],
                  [-2, -4, 0, 6],
                  [-3, -5, -6, 0]])
    
    # Compute pfaffian
    pf_value = ff.pf(A)
    
    # Verify: pf(A)^2 should equal det(A)
    det_value = np.linalg.det(A)
    
    assert np.allclose(pf_value**2, det_value), "pf(A)^2 should equal det(A)"
    assert np.allclose(A, -A.T), "Matrix should be skew-symmetric"
```

#### 2. Two-Site System Analysis (lines 32-59)
```python
def test_two_site_system_example():
    """Test two-site system from examples.rst lines 32-59"""
    # Two-site system
    n_sites = 2
    alphas = ff.jordan_wigner_alphas(n_sites)
    
    # Hopping Hamiltonian: H = -t(a†₀a₁ + a†₁a₀)
    t = 1.0
    A = np.array([[0, -t], [-t, 0]])
    H = ff.build_H(n_sites, A)
    
    # Generate ground state
    rho = ff.generate_gaussian_state(n_sites, H, alphas)
    
    # Compute correlation matrix
    gamma = ff.compute_2corr_matrix(rho, n_sites, alphas)
    
    # Validate results
    assert H.shape == (4, 4), "H should be 4x4 for 2-site system"
    assert gamma.shape == (4, 4), "Gamma should be 4x4"
    assert np.allclose(np.trace(rho), 1.0), "State should be normalized"
```

#### 3. Kitaev Chain Model (lines 64-111)
```python
def test_kitaev_chain_example():
    """Test Kitaev chain model from examples.rst lines 64-111"""
    def kitaev_chain(n_sites, mu, t, delta):
        """Create Kitaev chain Hamiltonian."""
        # Chemical potential term
        A = -mu * np.eye(n_sites)
        
        # Hopping term
        for i in range(n_sites - 1):
            A[i, i+1] = -t
            A[i+1, i] = -t
        
        # Pairing term
        B = np.zeros((n_sites, n_sites))
        for i in range(n_sites - 1):
            B[i, i+1] = delta
            B[i+1, i] = -delta
        
        return ff.build_H(n_sites, A, B)
    
    # Parameters
    n_sites = 6
    mu = 0.5      # Chemical potential
    t = 1.0       # Hopping strength
    delta = 0.8   # Pairing strength
    
    # Build Hamiltonian
    H = kitaev_chain(n_sites, mu, t, delta)
    
    # Diagonalize
    eigenvals, eigenvecs = ff.eigh_sp(H)
    
    # Validate results
    assert H.shape == (2 * n_sites, 2 * n_sites), "H should be 2N x 2N"
    assert ff.is_symp(eigenvecs), "Eigenvectors should be symplectic"
    
    # Check for zero modes (topological phase)
    energies = np.diag(eigenvals)[n_sites:]  # Positive eigenvalues
    zero_modes = np.abs(energies) < 1e-10
    # Note: Number of zero modes depends on parameters and boundary conditions
```

#### 4. Perfect Matching in Graphs (lines 201-246)
```python
def test_perfect_matching_example():
    """Test perfect matching example from examples.rst lines 201-246"""
    def analyze_perfect_matchings(n_vertices):
        """Analyze perfect matchings using pfaffian method."""
        if n_vertices % 2 != 0:
            return None, None, None
        
        # Generate random planar graph
        G = ff.generate_random_planar_graph(n_vertices, seed=42)
        
        if G is None:
            return None, None, None
        
        # Method 1: Brute force enumeration
        matchings_brute = ff.find_perfect_matchings(G)
        
        # Method 2: Pfaffian calculation
        pfo_matrix = ff.pfo_algorithm(G, verbose=False)
        pf_value = ff.pf(pfo_matrix)
        
        return G, matchings_brute, pf_value
    
    # Test with different sizes
    for n in [4, 6]:
        G, matchings_brute, pf_value = analyze_perfect_matchings(n)
        
        if G is not None:
            # Verify they match
            assert abs(len(matchings_brute) - abs(pf_value)) < 1e-10, \
                "Brute force and pfaffian methods should agree"
```

#### 5. Symplectic Transformation Example (lines 248-291)
```python
def test_symplectic_transformation_example():
    """Test symplectic transformation from examples.rst lines 248-291"""
    n_sites = 3
    
    # Create random Hamiltonian
    np.random.seed(42)  # For reproducibility
    A = np.random.randn(n_sites, n_sites)
    A = A + A.T
    H = ff.build_H(n_sites, A)
    
    # Symplectic diagonalization
    L, U = ff.eigh_sp(H)
    
    # Validate symplectic properties
    assert ff.is_symp(U), "U should be symplectic"
    assert ff.check_canonical_form(L), "L should be in canonical form"
    
    # Verify diagonalization: U† H U = L
    H_diag = U.conj().T @ H @ U
    
    assert np.allclose(H_diag, L), "Diagonalization should be correct"
```

## Priority 3: Core Library Tests (`test_ff_lib.py`)

### Jordan-Wigner Operators
```python
def test_jordan_wigner_alphas():
    """Test Jordan-Wigner alpha operators"""
    n_sites = 3
    alphas = ff.jordan_wigner_alphas(n_sites)
    
    assert len(alphas) == 2 * n_sites, "Should have 2N operators"
    
    # Test anticommutation relations
    for i in range(2 * n_sites):
        for j in range(2 * n_sites):
            anticomm = alphas[i] @ alphas[j] + alphas[j] @ alphas[i]
            if i == j:
                # {α_i, α_i} = 2δ_ij for some cases
                pass  # Specific test depends on operator type
            else:
                # Should anticommute for fermions
                pass  # Implement specific anticommutation tests

def test_jordan_wigner_majoranas():
    """Test Majorana operators"""
    n_sites = 2
    majoranas = ff.jordan_wigner_majoranas(n_sites)
    
    assert len(majoranas) == 2 * n_sites, "Should have 2N Majorana operators"
    
    # Test Hermiticity: γ† = γ
    for gamma in majoranas:
        assert np.allclose(gamma, gamma.conj().T), "Majorana operators should be Hermitian"
```

### Matrix Construction
```python
def test_build_H():
    """Test Hamiltonian matrix construction"""
    n_sites = 2
    A = np.array([[1, 0.5], [0.5, 1]])
    B = np.array([[0, 0.2], [-0.2, 0]])
    
    H = ff.build_H(n_sites, A, B)
    
    assert H.shape == (4, 4), "H should be 2N x 2N"
    
    # Check block structure
    assert np.allclose(H[:n_sites, :n_sites], -A.conj()), "Top-left block should be -A*"
    assert np.allclose(H[:n_sites, n_sites:], B), "Top-right block should be B"
    assert np.allclose(H[n_sites:, :n_sites], -B.conj()), "Bottom-left block should be -B*"
    assert np.allclose(H[n_sites:, n_sites:], A), "Bottom-right block should be A"

def test_build_V():
    """Test generator matrix construction"""
    n_sites = 2
    A = np.array([[1, 0.5], [0.5, 1]])
    
    V = ff.build_V(n_sites, A)
    
    assert V.shape == (4, 4), "V should be 2N x 2N"
    # Add specific tests for V matrix structure
```

## Priority 4: Combinatorics Tests (`test_ff_combinatorics.py`)

### Basic Functions
```python
def test_pfaffian_known_matrices():
    """Test pfaffian with known results"""
    # 2x2 case
    A2 = np.array([[0, 1], [-1, 0]])
    assert ff.pf(A2) == 1, "pf([[0,1],[-1,0]]) should be 1"
    
    # 4x4 case from examples
    A4 = np.array([[0, 1, 2, 3],
                   [-1, 0, 4, 5],
                   [-2, -4, 0, 6],
                   [-3, -5, -6, 0]])
    pf_val = ff.pf(A4)
    det_val = np.linalg.det(A4)
    assert np.allclose(pf_val**2, det_val), "pf(A)^2 should equal det(A)"

def test_pfaffian_properties():
    """Test mathematical properties of pfaffian"""
    # Odd dimension should give 0
    A_odd = np.random.randn(3, 3)
    assert ff.pf(A_odd) == 0, "Pfaffian of odd-dimensional matrix should be 0"
    
    # Skew-symmetric property
    n = 4
    A = np.random.randn(n, n)
    A_skew = A - A.T
    pf_val = ff.pf(A_skew)
    det_val = np.linalg.det(A_skew)
    assert np.allclose(pf_val**2, det_val), "pf(A)^2 = det(A) for skew-symmetric A"
```

## Priority 5: Graph Theory Tests (`test_ff_graph_theory.py`)

### Graph Generation and Analysis
```python
def test_generate_random_planar_graph():
    """Test planar graph generation"""
    G = ff.generate_random_planar_graph(6, seed=123)
    
    if G is not None:
        assert len(G.nodes()) == 6, "Graph should have 6 nodes"
        is_planar, _ = nx.check_planarity(G)
        assert is_planar, "Generated graph should be planar"

def test_pfo_algorithm():
    """Test Pfaffian ordering algorithm"""
    # Create a simple planar graph
    G = nx.cycle_graph(4)  # Square graph
    
    pfo_matrix = ff.pfo_algorithm(G, verbose=False)
    
    assert pfo_matrix.shape == (4, 4), "PFO matrix should match graph size"
    assert np.allclose(pfo_matrix, -pfo_matrix.T), "PFO matrix should be skew-symmetric"
```

## Priority 6: Utility Tests (`test_ff_utils.py`)

### Utility Functions
```python
def test_clean_function():
    """Test numerical cleaning function"""
    # Test with small values
    noisy_array = np.array([1.0, 1e-12, 0.5, -1e-15])
    cleaned = ff.clean(noisy_array, threshold=1e-10)
    expected = np.array([1.0, 0.0, 0.5, 0.0])
    assert np.allclose(cleaned, expected), "Small values should be cleaned"

def test_print_function():
    """Test custom printing function"""
    # Test that _print doesn't crash with various inputs
    test_matrix = np.array([[1.23456789, 1e-8 + 2.3456j],
                           [0.000123456, 9.87654321]])
    
    # Should not raise an exception
    ff._print(test_matrix, k=3)
    ff._print(test_matrix, k=6)
```

## Implementation Notes

### Test Configuration (`conftest.py`)
```python
import pytest
import numpy as np
import ff

@pytest.fixture
def random_seed():
    """Set random seed for reproducible tests"""
    np.random.seed(42)

@pytest.fixture
def small_system():
    """Create a small test system"""
    n_sites = 2
    alphas = ff.jordan_wigner_alphas(n_sites)
    A = np.array([[1, 0.5], [0.5, 1]])
    H = ff.build_H(n_sites, A)
    return n_sites, alphas, A, H

@pytest.fixture
def test_matrices():
    """Provide standard test matrices"""
    return {
        'skew_2x2': np.array([[0, 1], [-1, 0]]),
        'skew_4x4': np.array([[0, 1, 2, 3],
                              [-1, 0, 4, 5],
                              [-2, -4, 0, 6],
                              [-3, -5, -6, 0]]),
        'hermitian_2x2': np.array([[1, 0.5], [0.5, 1]]),
    }
```

### Testing Guidelines

1. **Reproducibility**: Use fixed random seeds
2. **Mathematical Validation**: Verify known mathematical relationships
3. **Edge Cases**: Test boundary conditions and error cases
4. **Numerical Stability**: Test with various precisions
5. **Documentation Compliance**: Ensure all documented examples work

### Execution Order

1. Start with `test_tutorials.py` - validates core tutorial examples
2. Implement `test_examples.py` - validates all documentation examples
3. Add `test_ff_lib.py` - comprehensive core library testing
4. Continue with remaining modules as needed

This plan ensures that all documented functionality works correctly and provides a solid foundation for the test suite.