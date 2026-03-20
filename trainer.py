import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from metrics import AUC_DATASETS, compute_metrics
from data import get_masks
from model import HASM


def train_epoch(model, optimizer, x, edge_index, N, hop_cheb, y, mask):
    model.train()
    optimizer.zero_grad()
    logits = model(x, edge_index, N, hop_cheb)
    loss   = F.cross_entropy(logits[mask], y[mask])
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, x, edge_index, N, hop_cheb, y, masks, dataset_name):
    model.eval()
    logits = model(x, edge_index, N, hop_cheb)
    return compute_metrics(logits, y, masks, dataset_name)


def run_split(dataset_name, data, num_classes, cfg, hop_cheb, x, split_idx, device):
    SEED = 42 + split_idx
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    N = data.num_nodes
    masks = get_masks(data, dataset_name, split_idx, device)

    model = HASM(
        in_dim       = x.shape[1],
        d_model      = cfg['d'],
        num_classes  = num_classes,
        K            = cfg['K'],
        filter_order = cfg['order'],
        num_layers   = cfg['layers'],
        dropout      = cfg['dropout'],
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['wd'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['epochs'])

    metric_name = 'ROC AUC' if dataset_name.lower() in AUC_DATASETS else 'Acc'
    best_val, best_test, best_ep = 0.0, 0.0, 0
    freq = 50 if cfg['epochs'] >= 400 else 25

    for epoch in range(1, cfg['epochs'] + 1):
        loss = train_epoch(model, optimizer, x, data.edge_index, N,
                           hop_cheb, data.y, masks['train'])
        scheduler.step()

        if epoch % freq == 0 or epoch == 1:
            metrics = evaluate(model, x, data.edge_index, N, hop_cheb,
                               data.y, masks, dataset_name)
            marker = ''
            if metrics['val'] > best_val:
                best_val, best_test, best_ep = metrics['val'], metrics['test'], epoch
                marker = '  ◄ best'
            print(f"    Ep {epoch:04d} | loss {loss:.4f} | "
                  f"train {metrics['train']:.4f} | val {metrics['val']:.4f} | "
                  f"test {metrics['test']:.4f}{marker}")

    print(f"    Split {split_idx} | Best {metric_name}: {best_test:.4f}  (ep {best_ep})")
    return best_test
