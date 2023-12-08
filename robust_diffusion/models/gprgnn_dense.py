import copy
from typing import Any, Dict

import numpy as np
import torch
from torch.nn import Linear, Parameter
import torch.nn.functional as F
from torch_sparse import SparseTensor
from torch_geometric.nn.conv.gcn_conv import gcn_norm

from robust_diffusion.aggregation import dense_soft_median


def from_sparse(sparse_model):
    args = {'n_features': 1,
            'n_classes': 1,
            'hidden': 1,
            'dropout_NN': sparse_model.dropout_NN,
            'dropout_GPR': sparse_model.dropout_GPR,
            'drop_GPR': sparse_model.drop_GPR,
            'propagation': 'GPR_prop',
            'K': sparse_model.prop1.K}
    dense_model = DenseGPRGNN(**args)
    dense_model.prop1.temp = copy.deepcopy(sparse_model.prop1.temp)
    dense_model.lin1 = copy.deepcopy(sparse_model.lin1)
    dense_model.lin2 = copy.deepcopy(sparse_model.lin2)
    return dense_model


def dropout_rows(x: torch.tensor, dropout: float, training: bool):
    ''' drops each row of x with probability dropout'''
    if not training:
        return x
    data = x.clone()
    a = dropout * torch.ones(x.shape[0])
    a = torch.bernoulli(a)
    idx = a.nonzero()
    data[idx, :] = 0
    return data / (1 - dropout)


class GPR_prop_dense(torch.nn.Module):
    '''
    propagation class for GPR_GNN
    '''

    def __init__(self, K, APPNP=False, alpha=None, norm=False, mean=None,
                 mean_kwargs: Dict[str, Any] = dict(temperature=2.5), **kwargs):
        super(GPR_prop_dense, self).__init__()
        # only random init supported
        self.K = K
        if APPNP:
            TEMP = alpha * (1 - alpha) ** np.arange(K + 1)
            TEMP[-1] = (1 - alpha) ** K
            self.temp = torch.tensor(TEMP)
        else:
            bound = np.sqrt(3 / (K + 1))
            TEMP = np.random.uniform(-bound, bound, K + 1)
            TEMP = TEMP / np.sum(np.abs(TEMP))
            self.temp = Parameter(torch.tensor(TEMP))
        self.norm = norm
        self.mean = mean  # E.g. choose 'soft_median'
        self.mean_kwargs = mean_kwargs

    def normalize_coefficients(self):
        temp = torch.sign(self.temp) * torch.softmax(torch.abs(self.temp), -1)
        return temp

    def forward(self, x, adj):
        if self.norm:
            # Normalize coefficients to avoid ambiguity
            temp = self.normalize_coefficients()
        else:
            temp = self.temp

        if isinstance(adj, SparseTensor):
            adj = adj.to_dense()

        if isinstance(adj, torch.Tensor):
            adj_norm = self.normalize_adjacency_matrix(adj)
        else:
            assert isinstance(adj, tuple)
            n, _ = x.shape
            adj_norm = gcn_norm(
                *adj, num_nodes=n, add_self_loops=True, dtype=x.dtype)
            adj_norm = torch.sparse_coo_tensor(*adj_norm, 2 * [n]).to_dense()

        if self.mean:
            n, _ = x.shape
            device = temp.device

            propagation_matrix = temp[0] * torch.eye(n, device=device)
            propagation_matrix += temp[1] * adj_norm
            adj_norm_power_k = adj_norm
            for k in range(2, self.K + 1):
                adj_norm_power_k = adj_norm @ adj_norm_power_k
                propagation_matrix += temp[k] * adj_norm_power_k

            hidden = dense_soft_median(
                propagation_matrix, x, self.mean_kwargs['temperature'])
            #hidden = (soft_weights[..., None] * x_signed.transpose(1, 0, 2)).sum(1)
        else:
            hidden = x * temp[0]
            for k in range(1, self.K + 1):
                x = adj_norm @ x
                gamma = temp[k]
                hidden = hidden + gamma * x
        return hidden

    @ staticmethod
    def normalize_adjacency_matrix(adj):
        adj_norm = torch.triu(adj, diagonal=1) + torch.triu(adj, diagonal=1).T
        adj_norm.data[torch.arange(adj.shape[0]), torch.arange(adj.shape[0])] = 1
        deg = torch.diag(torch.pow(adj_norm.sum(axis=1), - 1 / 2))
        adj_norm = deg @ adj_norm @ deg
        return adj_norm

    def __repr__(self):
        return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K,
                                          self.temp)


class DenseGPRGNN(torch.nn.Module):
    '''dense implementation of GPRGNN
    Constructor takes saved GPRGNN to copy parameters'''

    def __init__(self, n_features, hidden, n_classes, propagation, K, dropout_NN=0, dropout_GPR=0, drop_GPR=None, **kwargs):
        super(DenseGPRGNN, self).__init__()

        self.lin1 = Linear(n_features, hidden)
        self.lin2 = Linear(hidden, n_classes)
        self.dropout_NN = dropout_NN
        self.dropout_GPR = dropout_GPR
        self.drop_GPR = drop_GPR

        if propagation == 'PPNP':
            self.prop1 = GPR_prop_dense(K=K, APPNP=True, **kwargs)
        elif propagation == 'GPR_prop':
            self.prop1 = GPR_prop_dense(K=K, **kwargs)

    def forward(self, data, adj):

        if isinstance(adj, SparseTensor):
            adj = adj.to_dense()

        # MLP
        data = F.dropout(data, p=self.dropout_NN, training=self.training)
        data = F.relu(self.lin1(data))
        data = F.dropout(data, p=self.dropout_NN, training=self.training)
        data = self.lin2(data)

        # Propagation
        if self.dropout_GPR > 0:
            if self.drop_GPR == 'nodes':
                data = dropout_rows(data, self.dropout_GPR, training=self.training)
            elif self.drop_GPR == 'edges':
                adj = F.dropout(adj, p=self.dropout_GPR, training=self.training)
            elif self.drop_GPR == 'attr':
                data = F.dropout(data, p=self.dropout_GPR, training=self.training)
            else:
                assert False, f'dropout_GPR type {self.drop_GPR} not implemented'
        data = self.prop1(data, adj)

        return data
