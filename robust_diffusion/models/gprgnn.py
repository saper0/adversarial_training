# The below is an adaption of https://github.com/jianhao2016/GPRGNN from
# Chien et al. 2021
import torch
import copy
import torch.nn.functional as F
import numpy as np
from torch_sparse import SparseTensor

from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch.nn import Parameter
from torch.nn import Linear

from torch_geometric.nn.conv import MessagePassing, APPNP



def from_dense(dense_model):
    args = {'n_features': 1,
            'n_classes': 1,
            'hidden': 1,
            'dropout_NN': dense_model.dropout_NN,
            'dropout_GPR': dense_model.dropout_GPR,
            'drop_GPR': dense_model.drop_GPR,
            'propagation': 'GPR_prop',
            'K': dense_model.prop1.K}
    sparse_model = GPRGNN(**args)
    sparse_model.prop1.temp = copy.deepcopy(dense_model.prop1.temp)
    sparse_model.lin1 = copy.deepcopy(dense_model.lin1)
    sparse_model.lin2 = copy.deepcopy(dense_model.lin2)
    return sparse_model


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


class GPR_prop(MessagePassing):
    '''
    propagation class for GPR_GNN
    '''

    def __init__(self, K, Init='Random', dropout_GPR=0, Gamma=None, alpha=None, norm=False, **kwargs):
        super(GPR_prop, self).__init__(aggr='add')
        self.K = K
        self.dropout = dropout_GPR
        self.norm = norm

        assert Init in ['SGC', 'PPR', 'NPPR', 'Random', 'WS']
        if Init == 'SGC':
            # SGC-like, note that in this case, alpha has to be a integer. It means where the peak at when initializing GPR weights.
            TEMP = 0.0 * np.ones(K + 1)
            TEMP[alpha] = 1.0
        elif Init == 'PPR':
            # PPR-like
            TEMP = alpha * (1 - alpha)**np.arange(K + 1)
            TEMP[-1] = (1 - alpha)**K
        elif Init == 'NPPR':
            # Negative PPR
            TEMP = (alpha)**np.arange(K + 1)
            TEMP = TEMP / np.sum(np.abs(TEMP))
        elif Init == 'Random':
            # Random
            bound = np.sqrt(3 / (K + 1))
            TEMP = np.random.uniform(-bound, bound, K + 1)
            TEMP = TEMP / np.sum(np.abs(TEMP))
        elif Init == 'WS':
            # Specify Gamma
            TEMP = Gamma

        self.temp = Parameter(torch.tensor(TEMP))

    def normalize_coefficients(self):
        temp = torch.sign(self.temp) * torch.softmax(torch.abs(self.temp), -1)
        return temp

    def forward(self, x, edge_index, edge_weight=None):

        if isinstance(edge_index, SparseTensor):
            row, col, edge_attr = edge_index.t().coo()
            edge_index = torch.stack([row, col], dim=0)

        # normalize adj
        edge_index, edge_weight = gcn_norm(
            edge_index, edge_weight, num_nodes=x.size(0), add_self_loops=True, dtype=x.dtype)

        if self.norm:
            # Normalize coefficients to avoid ambiguity
            temp = self.normalize_coefficients()
        else:
            temp = self.temp

        # propagate
        hidden = x * temp[0]
        for k in range(1, self.K + 1):
            x = self.propagate(edge_index, x=x, norm=edge_weight)
            gamma = temp[k]
            hidden = hidden + gamma * x
        return hidden

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K,
                                          self.temp)


class GPRGNN(torch.nn.Module):
    def __init__(self, n_features, n_classes, hidden, K, propagation='GPR_prop', dropout_NN=0, dropout_GPR=0, drop_GPR=None, **kwargs):
        super(GPRGNN, self).__init__()

        # predictive MLP
        self.lin1 = Linear(n_features, hidden)
        self.lin2 = Linear(hidden, n_classes)

        # type of propagation (APPNP or GPR-GNN)
        if propagation == 'PPNP':
            self.prop1 = APPNP(K, alpha=kwargs['alpha'])
        elif propagation == 'GPR_prop':
            self.prop1 = GPR_prop(K, **kwargs)

        # dropout
        self.dropout_NN = dropout_NN
        self.dropout_GPR = dropout_GPR
        self.drop_GPR = drop_GPR

    def forward(self, data, adj):

        # handle different adj representations
        if isinstance(adj, SparseTensor):
            row, col, edge_weight = adj.t().coo()
            edge_index = torch.stack([row, col], dim=0)
        elif isinstance(adj, tuple):
            edge_index, edge_weight = adj

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
                edge_weight = F.dropout(data, p=self.dropout_GPR, training=self.training)
            else:
                data = F.dropout(data, p=self.dropout_GPR, training=self.training)
        data = self.prop1(data, edge_index, edge_weight)

        return data
