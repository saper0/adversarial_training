# This file has been taken from https://github.com/ivam-he/chebnetii 
# from He et al. 2022 and adapted for this work.

import math

import torch
from torch.nn import Linear, Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_self_loops, get_laplacian
import torch.nn.functional as F
from torch_sparse import coalesce, SparseTensor


def cheby(i, x):
    if i == 0:
        return 1
    elif i == 1:
        return x
    else:
        T0 = 1
        T1 = x
        for ii in range(2, i + 1):
            T2 = 2 * x * T1 - T0
            T0, T1 = T1, T2
        return T2


class ChebnetII_prop(MessagePassing):
    def __init__(self, K, Init=False, bias=True, exact_norm=False, **kwargs):
        super(ChebnetII_prop, self).__init__(aggr='add', **kwargs)

        self.K = K
        self.exact_norm = exact_norm
        self.temp = Parameter(torch.Tensor(self.K + 1))
        self.Init = Init
        self.reset_parameters()

    def reset_parameters(self):
        self.temp.data.fill_(1.0)

        if self.Init:
            for j in range(self.K + 1):
                x_j = math.cos((self.K - j + 0.5) * math.pi / (self.K + 1))
                self.temp.data[j] = x_j**2

    def forward(self, x, edge_index, edge_weight=None):
        coe_tmp = F.relu(self.temp)
        coe = coe_tmp.clone()

        for i in range(self.K + 1):
            coe[i] = coe_tmp[0] * cheby(i, math.cos((self.K + 0.5) * math.pi / (self.K + 1)))
            for j in range(1, self.K + 1):
                x_j = math.cos((self.K - j + 0.5) * math.pi / (self.K + 1))
                coe[i] = coe[i] + coe_tmp[j] * cheby(i, x_j)
            coe[i] = 2 * coe[i] / (self.K + 1)

        # L=I-D^(-0.5)AD^(-0.5)
        edge_index1, norm1 = get_laplacian(edge_index, edge_weight, normalization='sym',
                                           dtype=x.dtype, num_nodes=x.size(self.node_dim))

        if self.exact_norm:
            with torch.no_grad():  # TODO: currently no gradient supported by torch.lobpcg for sparse input
                adj = torch.sparse_coo_tensor(
                    edge_index1, norm1, (x.size(self.node_dim), x.size(self.node_dim)))
                max_eigenvalue = torch.lobpcg(adj, k=1)[0]
            norm1 = 2 / max_eigenvalue * norm1  # Then the eigenvalue is always in the range [0, 2]

        # L_tilde=2/lambda_ax * L - I
        edge_index_tilde, norm_tilde = add_self_loops(
            edge_index1, norm1, fill_value=-1.0, num_nodes=x.size(self.node_dim))
        edge_index_tilde, norm_tilde = coalesce(
            edge_index_tilde, norm_tilde, x.size(self.node_dim), x.size(self.node_dim))

        Tx_0 = x
        Tx_1 = self.propagate(edge_index_tilde, x=x, norm=norm_tilde, size=None)

        out = coe[0] / 2 * Tx_0 + coe[1] * Tx_1

        for i in range(2, self.K + 1):
            Tx_2 = self.propagate(edge_index_tilde, x=Tx_1, norm=norm_tilde, size=None)
            Tx_2 = 2 * Tx_2 - Tx_0
            out = out + coe[i] * Tx_2
            Tx_0, Tx_1 = Tx_1, Tx_2

        return out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K,
                                          self.temp)


class ChebNetII(torch.nn.Module):
    def __init__(self, n_features, n_classes, n_filters=64, K=10, dropout_NN=0.7,
                 dropout_GPR=0.7, exact_norm=False, prop_wd=5e-3, prop_lr=1e-3, **kwargs):
        super(ChebNetII, self).__init__()
        self.lin1 = Linear(n_features, n_filters)
        self.lin2 = Linear(n_filters, n_classes)
        self.prop1 = ChebnetII_prop(K, exact_norm=exact_norm)

        self.dprate = dropout_GPR
        self.dropout = dropout_NN
        self.reset_parameters()

        self.prop_wd = prop_wd
        self.prop_lr = prop_lr

    def reset_parameters(self):
        self.prop1.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data, adj):

        # handle different adj representations
        if isinstance(adj, SparseTensor):
            row, col, edge_weight = adj.t().coo()
            edge_index = torch.stack([row, col], dim=0)
        elif isinstance(adj, tuple):
            edge_index, edge_weight = adj

        x = F.dropout(data, p=self.dropout, training=self.training)
        x = self.lin1(x)
        x = F.relu(x)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        if self.dprate == 0.0:
            x = self.prop1(x, edge_index, edge_weight)
        else:
            x = F.dropout(x, p=self.dprate, training=self.training)
            x = self.prop1(x, edge_index, edge_weight)

        return x
