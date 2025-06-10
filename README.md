# Free Fermion Library

A comprehensive Python library for working with free fermion quantum systems, providing tools for combinatorial functions, graph theory algorithms, and quantum physics utilities.

## Features

### Core Modules

- **`ff_lib`**: Core quantum physics and linear algebra functions
  - Jordan-Wigner transformations (Dirac and Majorana fermions)
  - Symplectic free-fermion diagonalization
  - Gaussian state generation and manipulation
  - Fermionic correlation matrix computations
  - Wick's theorem implementation

- **`ff_combinatorics`**: Combinatorial matrix functions
  - Pfaffian computation via combinatorial formula
  - Hafnian computation
  - Permanent and determinant calculations
  - Sign of permutation functions

- **`ff_graph_theory`**: Graph algorithms and visualization
  - Pfaffian ordering algorithm (FKT algorithm) for planar graphs
  - Perfect matching algorithms
  - Planar graph generation and visualization
  - Dual graph construction

- **`ff_utils`**: Common utility functions
  - Matrix cleaning and formatting
  - Random bitstring generation
  - Direct sum operations
  - Pretty printing with numerical precision control

## Installation

### From Source

```bash
git clone https://github.com/your-username/free-fermion-lib.git
cd free-fermion-lib
pip install -e .
```

### Dependencies

- Python >= 3.8
- NumPy >= 1.20.0
- SciPy >= 1.7.0
- NetworkX >= 2.6.0
- Matplotlib >= 3.3.0

## Quick Start

```python
import numpy as np
from ff import *

# Generate Jordan-Wigner operators for 3 sites
n_sites = 3
alphas = jordan_wigner_alphas(n_sites)

# Create a simple Hamiltonian matrix
A = np.random.random((n_sites, n_sites))
A = A + A.T  # Make symmetric
H = build_H(n_sites, A)

# Generate a Gaussian state
rho = ff.random_FF_state(n_sites)

# Compute correlation matrix
gamma = compute_2corr_matrix(rho, n_sites, alphas)

# Compute pfaffian of a skew-symmetric matrix
skew_matrix = np.array([[0, 1, -2], [-1, 0, 3], [2, -3, 0]])
pfaffian_value = pf(skew_matrix)
```

## Development

### Running Tests

```bash
pip install -e ".[dev]"
pytest
```

### Code Formatting

```bash
black src/
flake8 src/
```

## License

MIT License - see LICENSE file for details.

## Citation

If you use this library in your research, please cite:

```bibtex
@software{free_fermion_lib,
  author = {James D. Whitfield},
  title = {Free Fermion Library: A Python package for quantum free fermion systems},
  year = {2025},
  url = {https://github.com/your-username/free-fermion-lib}
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

James D. Whitfield - James.D.Whitfield@dartmouth.edu
