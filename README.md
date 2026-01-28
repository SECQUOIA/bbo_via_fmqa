# bbo_via_fmqa  
**Black-Box Optimization via Factorization Machines and Quantum Annealing (FMQA)**

## Overview


**FMQA (Factorization Machine with Quantum Annealing)** is a **quantum–classical hybrid black-box optimization algorithm** designed to optimize expensive, discrete, or combinatorial objective functions where gradients or analytical forms are unavailable.

The central idea of FMQA is to combine **surrogate modeling** with **annealing-based optimization**:

- A **Factorization Machine (FM)** is trained as a surrogate model to approximate the unknown black-box objective.
- The learned FM is **mapped to a QUBO/Ising Hamiltonian**.
- The Hamiltonian is solved using **quantum annealing or annealing-based solvers** to propose promising candidate solutions.
- Newly evaluated samples are fed back into the surrogate, forming an iterative optimization loop.

This repository provides an **end-to-end implementation of FMQA for black-box optimization**, with support for multiple annealing backends.

Head to the link below to install FMQA packages prior to running the project. 
https://github.com/SECQUOIA/fmqa_pytorch/blob/main/README.md
---

## Algorithm Description

FMQA follows an iterative optimization loop:

1. **Initialization**
   - Sample an initial set of binary decision variables.
   - Evaluate the black-box objective function.

2. **Surrogate Modeling (Factorization Machine)**
    - Train a Factorization Machine of the form:  
$\hat{f}(x) = w_0 + \sum_i w_i x_i + \sum_{i<j} \langle v_i, v_j \rangle x_i x_j$

   - This naturally corresponds to a quadratic objective suitable for QUBO/Ising formulation.

3. **Hamiltonian Encoding**
   - Convert the learned FM parameters into a QUBO or Ising Hamiltonian.

4. **Annealing-Based Optimization**
   - Solve the Hamiltonian using one of the supported solvers to obtain candidate solutions.

5. **Evaluation and Update**
   - Evaluate candidates using the true black-box objective.
   - Augment the dataset and retrain the surrogate.

This approach balances **global exploration** (annealing solvers) with **data-efficient learning** (surrogate modeling).

---

## Solver Backends

Based on the original FMQA framework developed by **Tsuda Lab**, this implementation adapts the algorithm to run on multiple annealing backends:

| Backend | Description |
|--------|------------|
| D-Wave | Quantum annealing hardware |
| QCI (Dirac) | Classical Ising solver |
| Simulated Annealing | Classical baseline for benchmarking |

The FMQA algorithm remains unchanged across backends; only the Hamiltonian solver interface differs.

---

## Repository Structure

```text
bbo_via_fmqa/
├── fmqa/                 # Core FMQA optimization loop
├── solvers/              # D-Wave, QCI, and simulated annealing backends
├── examples/             # Example black-box optimization problems
├── experiments/          # Experiment and benchmarking scripts
└── README.md

