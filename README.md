# Hop-wise Adaptive Spectral Filters for Graph Representation Learning (HASM)

## Overview

**HASM** (Hop-wise Adaptive Spectral Message-passing) addresses a fundamental limitation in spectral GNNs. Existing methods apply a **single shared filter** across all propagation depths, treating 1-hop and 5-hop neighborhoods identically despite carrying structurally distinct information.

HASM assigns an **independent Chebyshev spectral filter** to each hop distance. Each filter is parameterized by its own learnable polynomial coefficients and a per-channel gating mechanism, enabling the model to simultaneously learn different frequency responses at different propagation depths, low-pass smoothing for some hops, high-pass filtering for others.

---

## Key Contributions

- **Per-hop adaptive spectral filters** — independent Chebyshev filter per hop distance, allowing the model to learn distinct frequency responses at each structural scale
- **Concatenate-and-transform aggregation** — preserves individual hop information that is lost under standard additive aggregation
- **Theoretical guarantees** — formal proof that GPR-GNN is a special case of HASM, and that additive aggregation is provably non-injective over hop configurations
- **Performance gains of 0.08% to 6.27%** over state-of-the-art baselines across heterophilic benchmarks

---

## Results

Node classification accuracy (%) and ROC AUC for Minesweeper, Tolokers, Questions:

| Model | Physics | Roman-Empire | Minesweeper | Tolokers | Questions | Chameleon |
|-------|---------|--------------|-------------|----------|-----------|-----------|
| MLP | 97.10 | 66.64 | 50.97 | 74.12 | 71.87 | 41.84 |
| GCN | 96.17 | 53.45 | 72.23 | 77.22 | 76.28 | 43.43 |
| GAT | 96.62 | 51.51 | 81.39 | 77.87 | 74.94 | 40.14 |
| GPR-GNN | 97.74 | 74.08 | 90.10 | 77.25 | 74.36 | 42.28 |
| BernNet | 97.64 | 72.70 | 77.93 | 76.83 | 74.25 | 42.57 |
| ChebNetII | 97.25 | 74.64 | 83.64 | 79.23 | 74.41 | 42.67 |
| H2GCN | 93.75 | 60.11 | 89.71 | 73.35 | 63.59 | 43.42 |
| PolyFormer | 98.08 | 80.27 | 91.90 | 83.48 | 77.26 | 45.35 |
| **HASM** | **97.10** | **86.54** | **93.24** | **83.56** | **77.28** | **46.93** |

Key highlights:
- **+6.27%** over PolyFormer on Roman-Empire
- **+1.34%** over PolyFormer on Minesweeper
- **+1.58%** over PolyFormer on Chameleon
- **+33%** over GCN on Roman-Empire

---

## Repository Structure

```
├── configs.py        # Hyperparameter configurations and argument parsing
├── data.py           # Dataset loading and preprocessing utilities
├── graph_utils.py    # Graph propagation, Chebyshev polynomial computation
├── model.py          # HASM model: per-hop filters, channel gates, node update
├── trainer.py        # Training loop, evaluation, early stopping
├── metrics.py        # Accuracy and ROC AUC computation
├── main.py           # Entry point
└── README.md
```

## Installation

```bash
git clone https://github.com/graphlearning2026/HASM.git
cd HASM
pip install -r requirements.txt
```

### Requirements

```
torch>=1.12.0
torch-geometric>=2.0.0
numpy
scikit-learn
```


This project is licensed under the MIT License.
