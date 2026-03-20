import torch
import numpy as np
from torch_geometric.datasets import HeterophilousGraphDataset, Coauthor
from torch_geometric.transforms import NormalizeFeatures


def make_random_splits(num_nodes, train_ratio=0.6, val_ratio=0.2, seed=42):
    rng  = np.random.default_rng(seed)
    idx  = rng.permutation(num_nodes)
    n_tr = int(num_nodes * train_ratio)
    n_va = int(num_nodes * val_ratio)
    def mask(indices):
        m = torch.zeros(num_nodes, dtype=torch.bool)
        m[torch.tensor(indices)] = True
        return m
    return {
        'train': mask(idx[:n_tr]),
        'val':   mask(idx[n_tr:n_tr + n_va]),
        'test':  mask(idx[n_tr + n_va:]),
    }


def load_dataset(name, device):
    key = name.lower()
    if key in ['roman-empire', 'amazon-ratings', 'tolokers', 'minesweeper', 'questions']:
        ds   = HeterophilousGraphDataset(root=f'/tmp/{key}', name=name)
        data = ds[0].to(device)
        return data, ds.num_classes
    elif key in ['physics', 'coauthor-physics']:
        ds   = Coauthor(root='/tmp/coauthor-physics', name='Physics', transform=NormalizeFeatures())
        return ds[0].to(device), ds.num_classes
    else:
        raise ValueError(f"Unknown dataset: {name}")


def get_masks(data, dataset_name, split_idx, device):
    key = dataset_name.lower()
    if key in ['roman-empire', 'amazon-ratings', 'tolokers', 'minesweeper', 'questions']:
        return {
            'train': data.train_mask[:, split_idx],
            'val':   data.val_mask[:, split_idx],
            'test':  data.test_mask[:, split_idx],
        }
    else:
        splits = make_random_splits(data.num_nodes, seed=split_idx)
        return {k: v.to(device) for k, v in splits.items()}
