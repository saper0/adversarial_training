# This file has been mostly taken from the work bei Geisler et al. 
# "Robustness of Graph Neural Networks at Scale" (NeurIPS, 2021) and adapted
# for this work: https://github.com/sigeisler/robustness_of_gnns_at_scale
import os.path as osp
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from torch_geometric.nn import GATConv
from torch_geometric.utils import softmax

from torch_sparse import SparseTensor


class WeightedGATConv(GATConv):
    """Extended GAT to allow for weighted edges (disabling edge features)."""
    def edge_update(self, alpha_j: Tensor, alpha_i: Optional[Tensor],
                    edge_attr: Optional[Tensor], index: Tensor,
                    ptr: Optional[Tensor], size_i: Optional[int]) -> Tensor:
        # Given edge-level attention coefficients for source and target nodes,
        # we simply need to sum them up to "emulate" concatenation:
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
        alpha = F.leaky_relu(alpha, self.negative_slope)

        if edge_attr is not None:
            assert edge_attr.dim() == 1, 'Only scalar edge weights supported'
            edge_attr = edge_attr.view(-1, 1)
            # `alpha` unchanged if edge_attr == 1 and -Inf if edge_attr == 0;
            # We choose log to counteract underflow in subsequent exp/softmax
            alpha = alpha + torch.log2(edge_attr)

        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return alpha


class GAT(torch.nn.Module):
    """GAT that supports weights edges."""
    def __init__(self,  n_features: int, n_classes: int,
                 hidden_dim: int = 16, dropout = 0.5, **kwargs):
        super().__init__()
        self.dropout = dropout
        # We add self-loops and initialize them to 1.
        self.conv1 = WeightedGATConv(n_features, hidden_dim,
                                     add_self_loops=True, fill_value=1.)
        self.conv2 = WeightedGATConv(hidden_dim, n_classes,
                                     add_self_loops=True, fill_value=1.)

    def forward(self, data, adj, **kwargs):
        # handle different adj representations
        if isinstance(adj, SparseTensor):
            row, col, edge_weight = adj.t().coo()
            edge_index = torch.stack([row, col], dim=0)
        elif isinstance(adj, tuple):
            edge_index, edge_weight = adj

        data = self.conv1(data, edge_index, edge_weight).relu()
        data = F.dropout(data, p=self.dropout, training=self.training)
        data = self.conv2(data, edge_index, edge_weight)
        return data