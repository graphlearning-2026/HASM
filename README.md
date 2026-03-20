# Hop-wise Adaptive Spectral Filters for Graph Representation Learning (HASM)

> Official implementation of the paper **"Hop-Wise Adaptive Spectral Filters for Graph Representation Learning"** accepted at [GRADES-NDA 2026](https://gradesnda.github.io/), co-located with ACM SIGMOD 2026, Bengaluru, India.

---

<p align="center">
  <img src="https://img.shields.io/badge/Venue-GRADES--NDA%202026-blue"/>
  <img src="https://img.shields.io/badge/Framework-PyTorch-orange"/>
  <img src="https://img.shields.io/badge/License-MIT-green"/>
</p>

---

## Overview

**HASM** (Hop-wise Adaptive Spectral Message-passing) addresses a fundamental limitation in spectral GNNs — existing methods apply a **single shared filter** across all propagation depths, treating 1-hop and 5-hop neighborhoods identically despite carrying structurally distinct information.

HASM assigns an **independent Chebyshev spectral filter** to each hop distance. Each filter is parameterized by its own learnable polynomial coefficients and a per-channel gating mechanism, enabling the model to simultaneously learn different frequency responses at different propagation depths — low-pass smoothing for some hops, high-pass filtering for others.

---

## Key Contributions

- **Per-hop adaptive spectral filters** — independent Chebyshev filter per hop distance, allowing the model to learn distinct frequency responses at each structural scale
- **Concatenate-and-transform aggregation** — preserves individual hop information that is lost under standard additive aggregation
- **Theoretical guarantees** — formal proof that GPR-GNN is a special case of HASM, and that additive aggregation is provably non-injective over hop configurations
- **Performance gains of 0.08% to 6.27%** over state-of-the-art baselines across heterophilic benchmarks

---

## Architecture

```
Input Features X
      |
      v
Feature Propagation P = D^(-1/2) * A_hat * D^(-1/2)
      |
      |-- Z(1) = P*X     --> Chebyshev Filter (hop 1) + Channel Gate --> H(1)
      |
      |-- Z(2) = P*Z(1)  --> Chebyshev Filter (hop 2) + Channel Gate --> H(2)
      |
      |-- ...
      |
      |-- Z(K) = P*Z(K-1)--> Chebyshev Filter (hop K) + Channel Gate --> H(K)
                                                                          |
                                                                          v
                                        Concatenate: [H(0) || H(1) || ... || H(K)]
                                                                          |
                                                                          v
                                        h'v = ReLU(MLP(Z(0)v + W^T [H(0)||...||H(K)]))
```

### Per-hop Filter Equation

```
H(k) = ( sum_j  gamma(k)_j * T_j(L) * Z(k) )  *  sigmoid( Z(k) * W(k)_ch )
        |__________________________________|      |__________________________|
          (i) polynomial spectral shaping              (ii) channel gate
```

Where:
- `Z(k)` = k-hop propagated features
- `gamma(k)_j` = learnable spectral coefficients at hop k
- `T_j` = Chebyshev polynomial of degree j
- `W(k)_ch` = per-channel gate weight at hop k

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

---

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

---

## Usage

### Basic Run

```bash
python main.py --dataset roman-empire --K 4 --J 3 --lr 0.01
```

### Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--dataset` | Dataset name (`physics`, `roman-empire`, `minesweeper`, `tolokers`, `questions`, `chameleon`) | `roman-empire` |
| `--K` | Number of propagation hops from `{2, 4, 6, 8}` | `4` |
| `--J` | Chebyshev polynomial degree per hop filter | `3` |
| `--lr` | Learning rate | `0.01` |
| `--dropout` | Dropout rate | `0.5` |
| `--epochs` | Number of training epochs | `1000` |
| `--runs` | Number of random splits to evaluate on | `10` |
| `--hidden` | Hidden layer dimension | `256` |

### Reproduce Paper Results

```bash
# Roman-Empire (86.54)
python main.py --dataset roman-empire --K 4 --lr 0.01

# Minesweeper (93.24)
python main.py --dataset minesweeper --K 4 --lr 0.01

# Co-author Physics (97.10)
python main.py --dataset physics --K 2 --lr 0.01

# Chameleon (46.93)
python main.py --dataset chameleon --K 6 --lr 0.01
```

---

## How It Works

### 1. Feature Propagation

```python
# Normalized adjacency
P = D^(-1/2) * A_hat * D^(-1/2)

# Propagate features hop by hop
Z = [X]
for k in range(1, K+1):
    Z.append(P @ Z[k-1])
```

### 2. Per-hop Chebyshev Filter + Channel Gate

```python
for k in range(K+1):
    # Chebyshev polynomial spectral shaping
    T = chebyshev_basis(L_tilde, J)  # [T0, T1, ..., TJ]
    spectral = sum(gamma[k][j] * T[j] @ Z[k] for j in range(J+1))

    # Channel gate
    gate = sigmoid(Z[k] @ W_ch[k])

    H[k] = spectral * gate  # element-wise multiply
```

### 3. Concatenate and Transform

```python
# Concatenate all hop outputs
concat = torch.cat([H[0], H[1], ..., H[K]], dim=-1)

# Final node update with skip connection
h_out = ReLU(MLP(X + W_t @ concat))
```

---

## Datasets

All datasets are automatically downloaded via PyTorch Geometric.

| Dataset | Type | Features | Nodes | Edges | Classes |
|---------|------|----------|-------|-------|---------|
| Co-author Physics | Homophilous | 8,415 | 34,493 | 495,924 | 5 |
| Roman-Empire | Heterophilous | 300 | 22,662 | 32,927 | 18 |
| Tolokers | Heterophilous | 10 | 11,758 | 519,000 | 2 |
| Questions | Heterophilous | 301 | 48,921 | 153,540 | 2 |
| Minesweeper | Heterophilous | 7 | 10,000 | 39,402 | 2 |
| Chameleon | Heterophilous | 2,223 | 890 | 17,708 | 5 |

---

## Theoretical Results

**Theorem 1 — GPR-GNN is a special case of HASM:**
GPR-GNN can be recovered from HASM when all hop filters are identical, filter order J=0 with channel gate removed, and hop projection W(k)_t = alpha_k * I. This proves HASM is strictly more expressive than GPR-GNN.

**Theorem 2 — Concatenation vs Additive Aggregation:**
Additive aggregation is provably non-injective over hop configurations — distinct hop outputs can collapse to identical aggregated representations. HASM's concatenation-based aggregation provably distinguishes all distinct hop configurations, enabling the model to learn relative hop importance from data.

---

## Ablation Study

| Configuration | Physics | Roman-Empire | Minesweeper | Chameleon |
|---|---|---|---|---|
| HASM w/ Additive Agg. | 95.82 | 83.21 | 91.53 | 45.32 |
| HASM w/o Channel Gate | 96.74 | 85.43 | 92.85 | 45.66 |
| **HASM (Full)** | **97.10** | **86.54** | **93.24** | **46.93** |

Both the concatenation-based aggregation and the channel gate contribute meaningfully to final performance.

---

## Citation

If you use this work, please cite:

```bibtex
@inproceedings{hasm2026,
  title     = {Hop-Wise Adaptive Spectral Filters for Graph Representation Learning},
  booktitle = {Proceedings of the 9th Joint Workshop on Graph Data Management
               Experiences and Systems (GRADES) and Network Data Analytics (NDA)},
  series    = {GRADES-NDA '26},
  year      = {2026},
  publisher = {ACM},
  address   = {Bengaluru, India},
  doi       = {XXXXXXX.XXXXXXX}
}
```

---

## License

This project is licensed under the MIT License.
