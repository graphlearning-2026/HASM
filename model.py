import torch
import torch.nn as nn

from graph_utils import precompute_hop_cheb


class HopFilter(nn.Module):
    def __init__(self, d, order=4):
        super().__init__()
        self.order = order
        gamma_init    = torch.zeros(order + 1)
        gamma_init[0] = 1.0
        self.gamma        = nn.Parameter(gamma_init + torch.randn(order + 1) * 0.01)
        self.channel_gate = nn.Linear(d, d)

    def forward(self, cheb_basis_k, z_k):
        out = torch.zeros_like(z_k)
        for j, Tj_z in enumerate(cheb_basis_k):
            out = out + self.gamma[j] * Tj_z
        cg = torch.sigmoid(self.channel_gate(z_k))
        return out * cg


class HASMLayer(nn.Module):
    def __init__(self, d, K=6, order=4, dropout=0.3):
        super().__init__()
        self.K       = K
        self.filters = nn.ModuleList([HopFilter(d, order) for _ in range(K + 1)])
        self.hop_proj = nn.Linear((K + 1) * d, d)
        self.mlp = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d, d),
            nn.ReLU(),
        )
        self.norm    = nn.LayerNorm(d)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hop_cheb, edge_index, num_nodes):
        device = next(self.parameters()).device
        H_list = []
        Z_v    = None
        for k in range(self.K + 1):
            basis_k = [b.to(device) for b in hop_cheb[k]]
            z_k     = basis_k[0]
            if k == 0:
                Z_v = z_k
            H_list.append(self.filters[k](basis_k, z_k))

        C = torch.cat(H_list, dim=-1)
        p = self.dropout(self.hop_proj(C))
        r = p + Z_v
        h = self.norm(self.mlp(r))
        return h


class HASM(nn.Module):
    def __init__(self, in_dim, d_model, num_classes,
                 K=6, filter_order=4, num_layers=2, dropout=0.3):
        super().__init__()
        self.K            = K
        self.filter_order = filter_order

        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.LayerNorm(d_model),
        )
        self.layers = nn.ModuleList([
            HASMLayer(d=d_model, K=K, order=filter_order, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(self, x, edge_index, num_nodes, hop_cheb):
        h = None
        for i, layer in enumerate(self.layers):
            cheb = hop_cheb if i == 0 else precompute_hop_cheb(
                h, edge_index, self.K, self.filter_order, num_nodes)
            h = layer(cheb, edge_index, num_nodes)
        return self.classifier(h)
