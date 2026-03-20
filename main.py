import argparse
import numpy as np
import torch

from configs import CONFIGS, DEFAULT_CFG, N_SPLITS, DEFAULT_N_SPLITS, VALID_DATASETS
from data import load_dataset
from graph_utils import precompute_hop_cheb
from metrics import AUC_DATASETS
from model import HASM
from trainer import run_split


def run(dataset_name):
    cfg      = CONFIGS.get(dataset_name.lower(), DEFAULT_CFG)
    n_splits = N_SPLITS.get(dataset_name.lower(), DEFAULT_N_SPLITS)
    DEVICE   = 'cuda' if torch.cuda.is_available() else 'cpu'
    metric_name = 'ROC AUC' if dataset_name.lower() in AUC_DATASETS else 'Accuracy'

    print(f"\n{'='*60}")
    print(f"  HASM — Hop-wise Learnable Spectral Filters")
    print(f"  Dataset : {dataset_name}   Metric: {metric_name}   Device: {DEVICE}")
    print(f"{'='*60}")

    data, num_classes = load_dataset(dataset_name, DEVICE)
    x = data.x.float()
    N, F = x.shape

    print(f"  Nodes: {N:,}  |  Features: {F}  |  Classes: {num_classes}")
    print(f"  Splits: {n_splits}   Config: {cfg}\n")

    tmp_model = HASM(
        in_dim=F, d_model=cfg['d'], num_classes=num_classes,
        K=cfg['K'], filter_order=cfg['order'],
        num_layers=cfg['layers'], dropout=cfg['dropout'],
    )

    print(f"  Projecting features {F} -> {cfg['d']} on CPU ...")
    with torch.no_grad():
        x_proj = tmp_model.input_proj(x.cpu()).cpu()
    del tmp_model

    J = cfg['order']
    print(f"  Precomputing Chebyshev basis (K={cfg['K']} hops, J={J} order) on CPU ...")
    hop_cheb = precompute_hop_cheb(x_proj, data.edge_index, cfg['K'], J, N)
    mb_total = sum(t.numel() * 4 for b in hop_cheb for t in b) / 1e6
    print(f"  Done. CPU RAM: {mb_total:.1f} MB\n")

    test_scores = []
    for split_idx in range(n_splits):
        print(f"\n  ── Split {split_idx + 1}/{n_splits} ──────────────────────────")
        score = run_split(dataset_name, data, num_classes, cfg,
                          hop_cheb, x, split_idx, DEVICE)
        test_scores.append(score)

    mean = np.mean(test_scores)
    std  = np.std(test_scores)
    print(f"\n{'─'*60}")
    print(f"  {dataset_name}  |  {metric_name}")
    print(f"  Scores : {[f'{s:.4f}' for s in test_scores]}")
    print(f"  Result : {mean*100:.2f} ± {std*100:.2f}")
    print(f"{'─'*60}")

    print(f"\n  Learned gamma^(k) per hop (last split):")
    header = '  ' + f"{'hop':>4}  " + '  '.join(f"γ_{j}" for j in range(J+1))
    print(header)
    print(f"  {'─'*52}")

    return mean, std


def run_all():
    datasets = ['roman-empire', 'questions', 'amazon-ratings',
                'tolokers', 'cora', 'amazon-photo', 'wikics']
    results  = {}
    for ds in datasets:
        mean, std = run(ds)
        results[ds] = (mean, std)

    metric_label = lambda ds: 'ROC AUC' if ds in AUC_DATASETS else 'Acc'
    print(f"\n\n{'='*60}")
    print(f"  HASM — Final Results")
    print(f"{'='*60}")
    print(f"  {'Dataset':<20}  {'Metric':<10}  {'Result':>18}")
    print(f"  {'─'*52}")
    for ds, (mean, std) in results.items():
        print(f"  {ds:<20}  {metric_label(ds):<10}  "
              f"{mean*100:>6.2f} ± {std*100:>5.2f}")
    print(f"{'='*60}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='computers',
                        choices=VALID_DATASETS)
    args, _ = parser.parse_known_args()
    if args.dataset == 'all':
        run_all()
    else:
        run(args.dataset)
