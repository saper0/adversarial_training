import logging
from math import comb
from typing import Any, Dict, Union

import numpy as np
from sacred import Experiment
try:
    import seml
except:  # noqa: E722
    seml = None
import torch
from torch_sparse import SparseTensor
from tqdm import tqdm
import torch.nn.functional as F

from robust_diffusion.data import prep_graph, split, count_edges_for_idx, split_squirrel
from robust_diffusion.attacks import create_attack, PGD
from robust_diffusion.helper.io import Storage
from robust_diffusion.models import create_model, GPRGNN, DenseGPRGNN
from robust_diffusion.models.gprgnn import GPR_prop
from robust_diffusion.models.gprgnn_dense import GPR_prop_dense
from robust_diffusion.train import train
from robust_diffusion.helper.utils import accuracy, calculate_loss

ex = Experiment()
if seml is not None:
    seml.setup_logger(ex)


@ex.post_run_hook
def collect_stats(_run):
    seml.collect_exp_stats(_run)


@ex.config
def config():
    overwrite = None
    db_collection = None
    if db_collection is not None:
        ex.observers.append(seml.create_mongodb_observer(db_collection, overwrite=overwrite))


    # default params
    n_per_class = 20
    data_dir = './data'
    dataset = 'cora_ml'
    make_undirected = True
    binary_attr = False
    data_device = 0

    device = 0
    seed = 0

    artifact_dir = 'cache_debug'
    model_storage_type = 'pretrained'
    model_params = dict(
        label="GPRGNN",
        model="GPRGNN",
        n_filters=64
    )

    train_params = dict(
        lr=1e-2,
        weight_decay=1e-3,
        patience=300,
        max_epochs=3000
    )

    debug_level = "info"
    display_steps = 10
    loss_type = 'tanhMargin'
    pretrain_epochs = 0
    self_training = False
    validate_every = 20
    robust_epsilon = 0.1
    train_attack_params = {'epochs': 5}
    val_attack_params = {'epochs': 100}


@ex.automain
def run(data_dir: str, dataset: str, model_params: Dict[str, Any], train_params: Dict[str, Any],  self_training: bool,
        artifact_dir: str, model_storage_type: str, robust_epsilon: float, pretrain_epochs: int, binary_attr: bool, make_undirected: bool, display_steps: int,
        loss_type: str, val_attack_params: Dict, train_attack_params: Dict, validate_every: int,
        seed: int, device: Union[str, int], data_device: Union[str, int],_run: Dict, n_per_class: int):
        
    logging.info({
        'setting': 'transductive', 'dataset': dataset, 'label': model_params["label"], 'robust_epsilon': robust_epsilon, 'epochs_train': train_attack_params['epochs'], 'self_training': self_training,
        'loss_type': loss_type, 'seed': seed})
    torch.manual_seed(seed)
    np.random.seed(seed)

    ########################### LOAD DATA ###########################
    graph = prep_graph(dataset, data_device, dataset_root=data_dir, make_undirected=make_undirected,
                       binary_attr=binary_attr, return_original_split=dataset.startswith('ogbn'))

    attr, adj, labels = graph[:3]

    ########################## Create Model ###############################
    hyperparams = dict(model_params)
    n_features = attr.shape[1]
    n_classes = int(labels.max() + 1)
    hyperparams.update({
        'n_features': n_features,
        'n_classes': n_classes
    })
    model = create_model(hyperparams).to(device)

    ########################### Split Transductive#########################
    if dataset in ['squirrel', 'chameleon']:
        logging.info('splitting for squirrel 60/20/20')
        idx_train, idx_val, idx_test = split_squirrel(labels.cpu().numpy(), train_size=0.6, val_size=0.2, test_size=0.2, seed=seed)
    elif len(graph) == 3 or graph[3] is None:
        idx_train, idx_val, idx_test = split(labels.cpu().numpy(), n_per_class=n_per_class)
    else:
        idx_train, idx_val, idx_test = graph[3]['train'], graph[3]['valid'], graph[3]['test']

    n_features = attr.shape[1]
    n_classes = int(labels[~labels.isnan()].max() + 1)

    ########################## PRE-TRAINING ###############################
    # train model for x epochs without attack (on train idx only) 
    if pretrain_epochs > 0:
        pretrain_params = train_params.copy()
        pretrain_params['max_epochs'] = pretrain_epochs
        train(model=model, 
                attr=attr, 
                adj=adj.to_dense(), 
                labels=labels, 
                idx_train=idx_train, 
                idx_val=idx_val,
                **pretrain_params)
        logits = model(attr, adj.to_dense())
        test_accuracy_pre = accuracy(logits.cpu(), labels.cpu(), idx_test)
        logging.info(f'Accuracy of model pretrained {pretrain_epochs} epochs: {test_accuracy_pre}')

    ########################### Self-Training ###############################
    # TODO implement split by confidence (top-k?)
    if self_training:
        baseline_model = create_model(hyperparams).to(device)
        pseudo_params = train_params.copy()
        pseudo_params['max_epochs'] = 500
        early_stopping = False
        if dataset == 'squirrel':
            early_stopping = False
        logging.info(f'early stopping is {early_stopping}')
        train(model=baseline_model, attr=attr, adj=adj.to_dense(), labels=labels, idx_train=idx_train, idx_val=idx_val,
                display_step=display_steps, early_stopping=early_stopping, **pseudo_params)
        logits = baseline_model(attr, adj.to_dense())
        pseudolabels = torch.argmax(logits, dim=1)
        pseudolabels[idx_train] = labels[idx_train]
        idx_train = np.concatenate([idx_train, idx_val, idx_test])
        train_labels = pseudolabels
        val_accuracy_pseudo = accuracy(logits.cpu(), labels.cpu(), idx_val)
        logging.info(f'Accuracy of pseudolabels (val): {val_accuracy_pseudo}')
    else:
        train_labels = labels


    ############################### OPTIMIZER ###################################
    optimizer = torch.optim.Adam(model.parameters(), lr=train_params['lr'], weight_decay=train_params['weight_decay'])

    ############################### N PERTURBATIONS ##############################
    if robust_epsilon > 0:

        n_train_edges = count_edges_for_idx(adj, idx_train) # num edges connected to train nodes
        m_train = int(n_train_edges) / 2
        n_perturbations_train = int(round(robust_epsilon * m_train))

        n_val_edges = count_edges_for_idx(adj, idx_val) # num edges connected to validation nodes
        m_val = int(n_val_edges) / 2
        n_perturbations_val = int(round(robust_epsilon * m_val))

    ############################### Train Loop ###################################
    # init attack adjs, default for eps = 0
    adj_attacked_val = adj.detach().to_dense()
    adj_attacked_train = adj.detach().to_dense()
    # init trace variables
    acc_trace_train = []
    acc_trace_val = []
    acc_trace_train_pert = []
    acc_trace_val_pert = []
    loss_trace = []
    loss_trace_val = []
    gamma_trace = []
    best_loss=np.inf

    adversary = PGD(attr=attr, adj=adj, labels=train_labels, model=model, idx_attack=idx_train,
                                  device=device, data_device=data_device, binary_attr=binary_attr,
                                  make_undirected=make_undirected, **train_attack_params)
    adversary.reset_changes()

    for it in tqdm(range(train_params['max_epochs']), desc='Training...'):

        # Generate adversarial adjacency
        if robust_epsilon > 0:
            #adversary = create_attack(attack, attr=attr, adj=adj, labels=train_labels, model=model, idx_attack=idx_train,
            #                      device=device, data_device=data_device, binary_attr=binary_attr,
            #                      make_undirected=make_undirected, attack_loss_type=attack_loss_type, **attack_params)
            model.eval()
            if train_attack_params['continuous']:
                adversary.attack_continuous(n_perturbations_train, 200/np.sqrt(it+1))
            else:
                adversary = PGD(attr=attr, adj=adj, labels=train_labels, model=model, idx_attack=idx_train,
                                  device=device, data_device=data_device, binary_attr=binary_attr,
                                  make_undirected=make_undirected, **train_attack_params)
                adversary.attack(n_perturbations_train)
            adj_pert = adversary.get_modified_adj()
            del adversary
            #adj_attacked_train = (adj_pert[0].detach(), adj_pert[1].detach())
            adj_attacked_train = adj_pert

        # train step
        model.train()
        optimizer.zero_grad()
        logits = model(attr, adj_attacked_train)
        loss = calculate_loss(logits[idx_train], train_labels[idx_train], loss_type)
        loss.backward()
        optimizer.step()        
        train_accuracy = accuracy(logits.cpu(), train_labels.cpu(), idx_train)
        acc_trace_train_pert.append(train_accuracy)


        # log clean accuracy
        logits = model(attr, adj.to_dense())
        train_accuracy_clean = accuracy(logits.cpu(), train_labels.cpu(), idx_train)
        val_accuracy_clean = accuracy(logits.cpu(), labels.cpu(), idx_val)
        acc_trace_train.append(train_accuracy_clean)
        acc_trace_val.append(val_accuracy_clean)


        # val step 
        if it % validate_every == 0:
            if robust_epsilon > 0: 
                #if not self_training:
                    # ALWAYS RUN FULL DISCRETE ATTACK FOR VAL
                    adversary_val = PGD(attr=attr, adj=adj, labels=labels, model=model, idx_attack=idx_val,
                                  device=device, data_device=data_device, binary_attr=binary_attr,
                                  make_undirected=make_undirected, **val_attack_params)
                    model.eval()
                    adversary_val.attack(n_perturbations_val)
                    adj_pert = adversary_val.get_modified_adj()
                    #adj_attacked_val = (adj_pert[0].detach(), adj_pert[1].detach())
                    adj_attacked_val = adj_pert
                    del adversary_val
                #else: # when self training use training attacked adj -> TODO change
                #    adj_attacked_val = adj_attacked_train

            with torch.no_grad():
                model.eval()
                logits_val = model(attr, adj_attacked_val)
                loss_val = calculate_loss(logits_val[idx_val], labels[idx_val], loss_type)
                # save val statistic
                loss_trace_val.append(loss_val.item())
                val_accuracy = accuracy(logits_val.cpu(), labels.cpu(), idx_val)
                acc_trace_val_pert.append(val_accuracy)
                if isinstance(model, (GPRGNN, DenseGPRGNN)) and isinstance(model.prop1, (GPR_prop, GPR_prop_dense)):
                    logging.info(f'model gammas: {model.prop1.temp.detach().cpu()}')
                logging.info(f'train acc (pert/clean): {train_accuracy} / {train_accuracy_clean}')
                logging.info(f'val acc (pert/clean): {val_accuracy} / {val_accuracy_clean}')
       
        # save train statistics
        if isinstance(model, (GPRGNN, DenseGPRGNN)) and isinstance(model.prop1, (GPR_prop, GPR_prop_dense)):
            gamma_trace.append(model.prop1.temp.detach().cpu())
        loss_trace.append(loss.item())

        # save new best model and break if patience is reached
        if loss_val < best_loss:
            best_loss = loss_val
            best_epoch = it
            best_state = {key: value.cpu() for key, value in model.state_dict().items()}
        else:
            if it >= best_epoch + train_params['patience']:
                break

    # restore the best validation state
    model.load_state_dict(best_state)
    model.eval()

    # store model and params
    storage = Storage(artifact_dir, experiment=ex)
    train_attack_params_prefix = {'train_attack_params_' + str(key): val for key, val in train_attack_params.items()}
    val_attack_params_prefix = {'val_attack_params_' + str(key): val for key, val in val_attack_params.items()}
    params = dict(dataset=dataset, binary_attr=binary_attr, make_undirected=make_undirected,seed=seed,
                  robust_epsilon=robust_epsilon, self_training=self_training, pretrain_epochs=pretrain_epochs, split_type='transductive', n_per_class = n_per_class,
                  loss_type=loss_type, attack = 'PGD', **train_attack_params_prefix, **val_attack_params_prefix, **hyperparams)
    model_path = storage.save_model(model_storage_type, params, model)

    return {
        'loss_trace': loss_trace,
        'loss_trace_val': loss_trace_val,
        'gamma_trace': gamma_trace,
        'model_path': model_path,
        'acc_trace_train': acc_trace_train,
        'acc_trace_val': acc_trace_val,
        'acc_trace_train_pert': acc_trace_train_pert,
        'acc_trace_val_pert': acc_trace_val_pert
    }