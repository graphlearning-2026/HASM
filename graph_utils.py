import torch
from torch_geometric.utils import add_self_loops, degree


def get_norm(edge_index, num_nodes, dtype):
    ei, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    row, col = ei
    deg = degree(col, num_nodes=num_nodes, dtype=dtype)
    deg_inv_sqrt = deg.pow(-0.5).clamp(max=1e6)
    norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
    return ei, norm


def sparse_propagate(h, ei, norm):
    row, col = ei
    agg = torch.zeros_like(h)
    agg.scatter_add_(0,
                     col.unsqueeze(-1).expand(-1, h.size(1)),
                     norm.unsqueeze(-1) * h[row])
    return agg


def cheb_basis(z, ei, norm, order):
    Tx = [z]
    if order >= 1:
        Lz = z - sparse_propagate(z, ei, norm)
        Tx.append(Lz)
    for _ in range(2, order + 1):
        L_Tx1 = Tx[-1] - sparse_propagate(Tx[-1], ei, norm)
        Tx.append(2.0 * L_Tx1 - Tx[-2])
    return Tx


def precompute_hop_cheb(x_proj, edge_index, K, order, num_nodes):
    x_cpu  = x_proj.cpu().float()
    ei_cpu = edge_index.cpu()
    ei_sl, norm = get_norm(ei_cpu, num_nodes, x_cpu.dtype)

    hop_cheb = []
    h = x_cpu
    for k in range(K + 1):
        basis = cheb_basis(h, ei_sl, norm, order)
        hop_cheb.append([b.cpu() for b in basis])
        if k < K:
            h = sparse_propagate(h, ei_sl, norm)
    return hop_cheb
