# Black-Box Optimization via FMQA

This repository provides a PyTorch-based implementation for Black-Box Optimization using Factorization Machines with Quantum Annealing (FMQA). The implementation builds upon the original work from [tsudalab/fmqa](https://github.com/tsudalab/fmqa).

## Overview

This package enables data-driven optimization of black-box functions using a trainable Binary Quadratic Model (BQM) based on Factorization Machines. In combination with annealing solvers (simulated or quantum), it can optimize complex objective functions without requiring explicit analytical forms.

The core approach:
1. Sample initial points from the search space
2. Train a Factorization Machine as a surrogate model
3. Use an annealing solver to optimize the surrogate
4. Evaluate promising candidates on the true objective
5. Iteratively refine the model with new data

## Features

- **PyTorch-based Factorization Machine**: Efficient training with automatic differentiation
- **Integration with D-Wave Ocean**: Compatible with `dimod` for various annealing solvers
- **Flexible encoding**: Support for binary and integer variable representations
- **Black-box optimization**: No analytical form required for objective functions

## Installation

### Prerequisites
- Python >= 3.7
- pip

### Install from source

1. Clone this repository:
```bash
git clone https://github.com/SECQUOIA/bbo_via_fmqa.git
cd bbo_via_fmqa
```

2. Install the fmqa package:
```bash
cd fmqa
pip install -e .
cd ..
```

3. Install the bbo_via_fmqa package:
```bash
cd bbo_via_fmqa
pip install -e .
cd ..
```

### Dependencies

The main dependencies include:
- `numpy >= 1.21.0`
- `torch` (PyTorch)
- `dimod >= 0.1.4` (D-Wave Ocean SDK)
- `pandas >= 1.3.0`
- `matplotlib >= 3.4.0`

## Usage

### Basic Example

See `bbo_via_fmqa/fmqa_test.py` for a complete working example. The basic workflow:

```python
import numpy as np
import dimod
from fmqa import FMBQM

# Prepare your dataset (binary vectors and objective values)
x_data = np.array([[0, 1, 0, 1], [1, 0, 1, 0], ...])  # Binary vectors
y_data = np.array([0.5, -0.3, ...])                    # Objective values

# Train a surrogate model
fm_model = FMBQM.from_data(
    x=x_data, 
    y=y_data, 
    num_epoch=1000, 
    learning_rate=1e-2
)

# Use an annealing solver to optimize the surrogate
sampler = dimod.SimulatedAnnealingSampler()
results = sampler.sample(fm_model, num_reads=100)

# Extract the best candidate solution
for sample, energy in results.data(['sample', 'energy']):
    print(f"Candidate: {sample}, Energy: {energy}")
    break
```

### Grid-based Optimization

The repository includes utilities for optimizing over discrete grids:

1. Load grid data from CSV (with columns: `x`, `y`, `Objective`)
2. Encode coordinates as binary strings
3. Run iterative optimization with FMQA

See `bbo_via_fmqa/fmqa_test.py` for implementation details.

## Project Structure

```
bbo_via_fmqa/
├── README.md                  # This file
├── LICENSE                    # MIT License
├── bbo_via_fmqa/              # Main package
│   ├── setup.py              # Package setup
│   ├── fmqa_test.py          # Example optimization script
│   ├── read_grid.py          # Grid data utilities
│   └── tests/                # Test suite
└── fmqa/                     # Original fmqa implementation
    ├── LICENSE               # Original MIT License
    ├── README.md             # Original documentation
    └── fmqa/                 # Core FMQA package
        ├── factorization_machine.py
        └── fm_binary_quadratic_model.py
```

## Testing

Run the test suite:

```bash
pytest bbo_via_fmqa/tests/
```

Or run tests with coverage:

```bash
pytest --cov=bbo_via_fmqa bbo_via_fmqa/tests/
```

## Acknowledgments

This implementation is based on the original [fmqa package](https://github.com/tsudalab/fmqa) developed by Tsuda Laboratory. The original implementation used MXNet, while this version uses PyTorch for the Factorization Machine implementation.

### Citation

If you use this package in your research, please cite the original paper:

```bibtex
@article{PhysRevResearch.2.013319,
  title = {Designing metamaterials with quantum annealing and factorization machines},
  author = {Kitai, Koki and Guo, Jiang and Ju, Shenghong and Tanaka, Shu and Tsuda, Koji and Shiomi, Junichiro and Tamura, Ryo},
  journal = {Phys. Rev. Research},
  volume = {2},
  issue = {1},
  pages = {013319},
  numpages = {10},
  year = {2020},
  month = {Mar},
  publisher = {American Physical Society},
  doi = {10.1103/PhysRevResearch.2.013319},
  url = {https://link.aps.org/doi/10.1103/PhysRevResearch.2.013319}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

The original fmqa package is also licensed under the MIT License. See [fmqa/LICENSE](fmqa/LICENSE) for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For issues and questions, please open an issue on the [GitHub repository](https://github.com/SECQUOIA/bbo_via_fmqa/issues).
