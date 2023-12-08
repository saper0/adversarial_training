# This file has been mostly taken from the work bei Geisler et al. 
# "Robustness of Graph Neural Networks at Scale" (NeurIPS, 2021) and adapted
# for this work: https://github.com/sigeisler/robustness_of_gnns_at_scale
import logging

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from robust_diffusion.helper.utils import accuracy
from robust_diffusion.models import GPRGNN, DenseGPRGNN, ChebNetII
from robust_diffusion.models.gprgnn import GPR_prop


def train(model, attr, adj, labels, idx_train, idx_val,
          lr, weight_decay, patience, max_epochs, display_step=50, early_stopping=True):
    """Train a model using either standard training.
    Parameters
    ----------
    model: torch.nn.Module
        Model which we want to train.
    attr: torch.Tensor [n, d]
        Dense attribute matrix.
    adj: torch.Tensor [n, n]
        Dense adjacency matrix.
    labels: torch.Tensor [n]
        Ground-truth labels of all nodes,
    idx_train: array-like [?]
        Indices of the training nodes.
    idx_val: array-like [?]
        Indices of the validation nodes.
    lr: float
        Learning rate.
    weight_decay : float
        Weight decay.
    patience: int
        The number of epochs to wait for the validation loss to improve before stopping early.
    max_epochs: int
        Maximum number of epochs for training.
    display_step : int
        How often to print information.
    Returns
    -------
    train_val, trace_val: list
        A tupole of lists of values of the validation loss during training.
    """
    trace_loss_train = []
    trace_loss_val = []
    trace_acc_train = []
    trace_acc_val = []
    if isinstance(model, ChebNetII):
        optimizer = torch.optim.Adam([
            {'params': model.lin1.parameters(), 'weight_decay': weight_decay, 'lr': lr},
            {'params': model.lin2.parameters(), 'weight_decay': weight_decay, 'lr': lr},
            {'params': model.prop1.parameters(), 'weight_decay': model.prop_wd, 'lr': model.prop_lr}])
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_loss = np.inf

    model.train()
    for it in tqdm(range(max_epochs), desc='Training...'):
        optimizer.zero_grad()

        logits = model(attr, adj)
        loss_train = F.cross_entropy(logits[idx_train], labels[idx_train])
        loss_val = F.cross_entropy(logits[idx_val], labels[idx_val])

        loss_train.backward()
        optimizer.step()

        trace_loss_train.append(loss_train.detach().item())
        trace_loss_val.append(loss_val.detach().item())

        train_acc = accuracy(logits, labels, idx_train)
        val_acc = accuracy(logits, labels, idx_val)

        trace_acc_train.append(train_acc)
        trace_acc_val.append(val_acc)

        if loss_val < best_loss:
            best_loss = loss_val
            best_epoch = it
            best_state = {key: value.cpu() for key, value in model.state_dict().items()}
        else:
            if (it >= best_epoch + patience) and early_stopping:
                break

        if it % display_step == 0:
            logging.info(f'\nEpoch {it:4}: loss_train: {loss_train.item():.5f}, loss_val: {loss_val.item():.5f}, '
                         f'acc_train: {train_acc:.5f}, acc_val: {val_acc:.5f} ')

    # restore the best validation state
    if early_stopping:
        model.load_state_dict(best_state)
    return trace_loss_val, trace_loss_train, trace_acc_val, trace_acc_train


def train_inductive(model, attr_training, attr_validation, adj_training, adj_validation, labels_training, labels_validation, idx_train, idx_val,
          lr, weight_decay, patience, max_epochs, display_step=50):
    """Train a model using inductive training.
    Parameters
    ----------
    model: torch.nn.Module
        Model which we want to train.
    attr_training: torch.Tensor [n, d]
        Dense attribute matrix of training graph.
    attr_validation: torch.Tensor [n, d]
        Dense attribute matrix of training graph.
    adj_training:
        Sparse adjacency matrix of training graph
    adj_validation:
        Sparse adjacency matrix of validation graph
    labels_training: torch.Tensor [n]
        Ground-truth labels of all nodes in the training graph,
    labels_validation: torch.Tensor [n]
        Ground-truth labels of all nodes in the validation graph,
    idx_train: array-like
        Indices of the training nodes in the training graph.
    idx_validation: array-like
        Indices of the validation nodes in the validation graph.
    lr: float
        Learning rate.
    weight_decay : float
        Weight decay.
    patience: int
        The number of epochs to wait for the validation loss to improve before stopping early.
    max_epochs: int
        Maximum number of epochs for training.
    display_step : int
        How often to print information.
    Returns
    -------
    train_val, trace_val: list
        A tupole of lists of values of the validation loss during training.
    """
    trace_loss_train = []
    trace_loss_val = []
    trace_acc_train = []
    trace_acc_val = []
    if isinstance(model, (GPRGNN, DenseGPRGNN)) and isinstance(model.prop1, GPR_prop) and model.prop1.norm == True: # exclude prop coeffs from weight decay
        logging.info('Excluding GPR-GNN coefficients from weight decay as we use normalization')
        grouped_parameters = [
                {
                    "params": [p for n, p in model.named_parameters() if 'prop1.temp' != n],
                    "weight_decay": weight_decay,
                },
                {
                    "params": [p for n, p in model.named_parameters() if 'prop1.temp' == n],
                    "weight_decay": 0.0,
                },
            ]
        optimizer = torch.optim.Adam(grouped_parameters, lr=lr) 
    elif isinstance(model, ChebNetII):
        optimizer = torch.optim.Adam([
            {'params': model.lin1.parameters(), 'weight_decay': weight_decay, 'lr': lr},
            {'params': model.lin2.parameters(), 'weight_decay': weight_decay, 'lr': lr},
            {'params': model.prop1.parameters(), 'weight_decay': model.prop_wd, 'lr': model.prop_lr}])
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_loss = np.inf

    model.train()
    for it in tqdm(range(max_epochs), desc='Training...'):
        #### Train step ####
        optimizer.zero_grad()
        model.train()
        logits = model(attr_training, adj_training) # no attacks here so no harm in taking full attr
        loss_train = F.cross_entropy(logits[idx_train], labels_training[idx_train])
        loss_train.backward()
        optimizer.step()

        #### Validation ####
        with torch.no_grad():
            model.eval()
            logits_val = model(attr_validation, adj_validation) # no attacks here so no harm in taking full attr
            loss_val = F.cross_entropy(logits_val[idx_val], labels_validation[idx_val])

        trace_loss_train.append(loss_train.detach().item())
        trace_loss_val.append(loss_val.detach().item())

        train_acc = accuracy(logits, labels_training, idx_train)
        val_acc = accuracy(logits_val, labels_validation, idx_val)

        trace_acc_train.append(train_acc)
        trace_acc_val.append(val_acc)

        if loss_val < best_loss:
            best_loss = loss_val
            best_epoch = it
            best_state = {key: value.cpu() for key, value in model.state_dict().items()}
        else:
            if it >= best_epoch + patience:
                break

        if it % display_step == 0:
            logging.info(f'\nEpoch {it:4}: loss_train: {loss_train.item():.5f}, loss_val: {loss_val.item():.5f}, '
                         f'acc_train: {train_acc:.5f}, acc_val: {val_acc:.5f} ')

    # restore the best validation state
    model.load_state_dict(best_state)
    return trace_loss_val, trace_loss_train, trace_acc_val, trace_acc_train
