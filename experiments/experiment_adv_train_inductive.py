import logging
from math import comb
from typing import Any, Dict, Union
import gc
import time

import numpy as np
from sacred import Experiment
try:
    import seml
except:  # noqa: E722
    seml = None
import torch
from tqdm import tqdm

from robust_diffusion.data import prep_graph, split_inductive, filter_data_for_idx, count_edges_for_idx
from robust_diffusion.attacks import create_attack
from robust_diffusion.helper.io import Storage
from robust_diffusion.models import create_model, GPRGNN, DenseGPRGNN, ChebNetII
from robust_diffusion.models.gprgnn import GPR_prop
from robust_diffusion.train import train_inductive
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
    balance_test = True

    device = 0
    seed = 0

    artifact_dir = 'cache_debug'
    model_storage_type = 'pretrained'
    # model_params = dict(
    #     label="GPRGNN",
    #     model="GPRGNN",
    #     n_filters=64
    # )
    model_params = dict(
        label="ChebNetII",
        model="ChebNetII",
        n_filters=64,
        K=10,
        dropout_NN=0.8,
        dropout_GPR=0.5,
        prop_wd=5e-3,
        prop_lr=1e-3,
        exact_norm=True
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
    validate_every = 10    
    balance_test=True
    robust_epsilon = 0.1
    attack = 'LRBCD'
    train_attack_params = {'epochs': 5, 
                    'fine_tune_epochs': 0,
                    'keep_heuristic': 'WeightOnly',
                    'search_space_size': 100_000,
                    'do_synchronize': True,
                    'attack_loss_type': 'tanhMargin'}
    val_attack_params = {'epochs': 200, 
                    'fine_tune_epochs': 0,
                    'keep_heuristic': 'WeightOnly',
                    'search_space_size': 100_000,
                    'do_synchronize': True,
                    'attack_loss_type': 'tanhMargin'}



@ex.automain
def run(data_dir: str, dataset: str, model_params: Dict[str, Any], train_params: Dict[str, Any], self_training: bool,
        artifact_dir: str, model_storage_type: str, robust_epsilon: float, pretrain_epochs: int, binary_attr: bool, make_undirected: bool, validate_every: int,
        loss_type: str, attack: str, train_attack_params: Dict, val_attack_params: Dict,
        seed: int, device: Union[str, int], data_device: Union[str, int],_run: Dict, n_per_class: int, balance_test: True):
        
    logging.info({
        'dataset': dataset, 'label': model_params["label"], 'robust_epsilon': robust_epsilon, 'self_training': self_training, 'split_type': 'inductive',
        'loss_type': loss_type})
    torch.manual_seed(seed)
    np.random.seed(seed)

    ########################### LOAD DATA ###########################
    graph = prep_graph(dataset, data_device, dataset_root=data_dir, make_undirected=make_undirected,
                       binary_attr=binary_attr, return_original_split=dataset.startswith('ogbn'))

    attr_orig, adj_orig, labels = graph[:3]

    ########################## Create Model ###############################
    hyperparams = dict(model_params)
    n_features = attr_orig.shape[1]
    n_classes = int(labels.max() + 1)
    hyperparams.update({
        'n_features': n_features,
        'n_classes': n_classes
    })
    model = create_model(hyperparams).to(device)
    logging.info(model)

    ########################### Split ########################################
    if dataset == 'ogbn-arxiv':
        adj_orig = adj_orig.type_as(torch.ones(1, dtype=torch.float32, device=device)) # cast to float
        idx_train = graph[3]['train']
        idx_val = graph[3]['valid']
        idx_test = graph[3]['test']
        idx_unlabeled = np.array([], dtype=bool)
    else:
        idx_train, idx_unlabeled, idx_val, idx_test = split_inductive(labels.cpu().numpy(), n_per_class=n_per_class, balance_test=balance_test)

    ########################### handle data #####################
    # delete 
    attr_train, adj_train, labels_train, mapping_proj_train = filter_data_for_idx(attr_orig.clone(), adj_orig.clone(), labels,  np.concatenate([idx_train, idx_unlabeled]))
    idx_train_train = mapping_proj_train[idx_train] # idx of training nodes in training graph
    idx_unlabeled_train = mapping_proj_train[idx_unlabeled] # idx of unlabeled nodes in training graph
    attr_val, adj_val, labels_val, mapping_proj_val = filter_data_for_idx(attr_orig.clone(), adj_orig.clone(), labels, np.concatenate([idx_train, idx_val, idx_unlabeled]))
    idx_val_val = mapping_proj_val[idx_val] # idx of val nodes in val graph
   
   
    ########################## PRETRAIN ###############################
    # train model for x epochs without attack
    if pretrain_epochs > 0:
        pretrain_params = train_params.copy()
        pretrain_params['max_epochs'] = pretrain_epochs
        train_inductive(model=model, 
                        attr_training=attr_train, 
                        attr_validation=attr_val, 
                        adj_training= adj_train, 
                        adj_validation=adj_val,
                        labels_training=labels_train,
                        labels_validation=labels_val,
                        idx_train=idx_train_train,
                        idx_val=idx_val_val,
                        **pretrain_params)
    ########################### SELF-TRAINING ####################
    if self_training:
        baseline_model = create_model(hyperparams).to(device)
        train_inductive(model=baseline_model, 
                        attr_training=attr_train, 
                        attr_validation=attr_val, 
                        adj_training= adj_train, 
                        adj_validation=adj_val, 
                        labels_training=labels_train, 
                        labels_validation=labels_val, 
                        idx_train=idx_train_train, 
                        idx_val=idx_val_val,
                        **train_params)
        logits = baseline_model(attr_train, adj_train)
        pseudolabels = torch.argmax(logits, dim=1)
        pseudolabels[idx_train_train] = labels_train[idx_train_train]
        labels_train = pseudolabels 
        idx_train_train = np.concatenate([idx_train_train, idx_unlabeled_train]) # TODO more sophisticated selection possible


    ############################### REST ###################################
    if isinstance(model, (GPRGNN, DenseGPRGNN)) and isinstance(model.prop1, GPR_prop) and model.prop1.norm == True: # exclude prop coeffs from weight decay
        logging.info('Excluding GPR-GNN coefficients from weight decay as we use normalization')
        grouped_parameters = [
                {
                    "params": [p for n, p in model.named_parameters() if 'prop1.temp' != n],
                    "weight_decay": train_params['weight_decay'],
                    'lr':train_params['lr']
                },
                {
                    "params": [p for n, p in model.named_parameters() if 'prop1.temp' == n],
                    "weight_decay": 0.0,
                    'lr':train_params['lr']
                },
            ]
        optimizer = torch.optim.Adam(grouped_parameters)
    elif isinstance(model, ChebNetII):
        optimizer = torch.optim.Adam([
            {'params': model.lin1.parameters(), 'weight_decay': train_params['weight_decay'], 'lr': train_params['lr']},
            {'params': model.lin2.parameters(), 'weight_decay': train_params['weight_decay'], 'lr': train_params['lr']},
            {'params': model.prop1.parameters(), 'weight_decay': model.prop_wd, 'lr': model.prop_lr}])
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=train_params['lr'], weight_decay=train_params['weight_decay'])

    ############################## n_pertubations ##########################
    if robust_epsilon > 0:
        n_train_edges = count_edges_for_idx(adj_train, idx_train_train) # num edges connected to train nodes
        m_train = int(n_train_edges) / 2
        n_perturbations_train = int(round(robust_epsilon * m_train))

        n_val_edges = count_edges_for_idx(adj_val, idx_val_val) # num edges connected to val nodes
        m_val = int(n_val_edges) / 2
        n_perturbations_val = int(round(robust_epsilon * m_val))

    # init attack adjs 
    adj_attacked_val = adj_val.detach()
    adj_attacked_train = adj_train.detach()
    # init trace variables
    acc_trace_train = []
    acc_trace_val = []
    acc_trace_train_pert = []
    acc_trace_val_pert = []
    loss_trace = []
    loss_trace_val = []
    gamma_trace = []
    best_loss=np.inf

    ############################### Train Loop ###################################
    for it in tqdm(range(train_params['max_epochs']), desc='Training...'):

        # Generate adversarial adjacency
        if robust_epsilon > 0:
            torch.cuda.empty_cache()
            adversary = create_attack(attack, attr=attr_train, adj=adj_train, labels=labels_train, model=model, idx_attack=idx_train_train,
                                  device=device, data_device=data_device, binary_attr=binary_attr,
                                  make_undirected=make_undirected, **train_attack_params)
            model.eval()
            adversary.attack(n_perturbations_train)
            adj_pert = adversary.get_modified_adj()
            adj_attacked_train = (adj_pert[0].detach(), adj_pert[1].detach())
            del adversary

        # train step
        model.train()
        optimizer.zero_grad()
        logits = model(attr_train, adj_attacked_train)
        loss = calculate_loss(logits[idx_train_train], labels_train[idx_train_train], loss_type)
        loss.backward()
        optimizer.step()
        train_accuracy = accuracy(logits.cpu(), labels_train.cpu(), idx_train_train)
        acc_trace_train_pert.append(train_accuracy)
        if isinstance(model, GPRGNN) and isinstance(model.prop1, GPR_prop):
            gamma_trace.append(model.prop1.temp.detach().cpu())
        loss_trace.append(loss.item())



        # val step 
        if it % validate_every == 0:
            if robust_epsilon > 0:                 
                torch.cuda.empty_cache()
                adversary_val = create_attack(attack, attr=attr_val, adj=adj_val, labels=labels_val, model=model, idx_attack=idx_val_val,
                                    device=device, data_device=data_device, binary_attr=binary_attr,
                                    make_undirected=make_undirected, **val_attack_params)
                model.eval()
                adversary_val.attack(n_perturbations_val)
                adj_pert = adversary_val.get_modified_adj()
                adj_attacked_val = (adj_pert[0].detach(), adj_pert[1].detach())
                del adversary_val
                
            with torch.no_grad():
                # validation
                model.eval()
                logits_val = model(attr_val, adj_attacked_val) #TODO not entirely correct; attack on labels[idx_val] would be better
                loss_val = calculate_loss(logits_val[idx_val_val], labels_val[idx_val_val], loss_type)
                loss_trace_val.append(loss_val.item())                
                val_accuracy = accuracy(logits_val.cpu(), labels_val.cpu(), idx_val_val)
                acc_trace_val_pert.append(val_accuracy)
                # log clean accuracy on train graph
                logits_clean_train = model(attr_train, adj_train)
                train_accuracy_clean = accuracy(logits_clean_train.cpu(), labels_train.cpu(), idx_train_train)
                acc_trace_train.append(train_accuracy_clean)
                # log clean accuracy on val graph
                logits_clean_val = model(attr_val, adj_val)
                val_accuracy_clean = accuracy(logits_clean_val.cpu(), labels_val.cpu(), idx_val_val)
                acc_trace_val.append(val_accuracy_clean)
                # print output
                if isinstance(model, GPRGNN) and isinstance(model.prop1, GPR_prop):
                    print(f'model gammas: {model.prop1.normalize_coefficients().detach().cpu()}')
                print(f'train acc (pert/clean): {train_accuracy} / {train_accuracy_clean}')
                print(f'val acc (pert/clean): {val_accuracy} / {val_accuracy_clean}')

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
    params = dict(dataset=dataset, binary_attr=binary_attr, make_undirected=make_undirected,seed=seed, validate_every=validate_every,
                  robust_epsilon=robust_epsilon, self_training=self_training, pretrain_epochs=pretrain_epochs, split_type='inductive', n_per_class=n_per_class, balance_test=balance_test,
                  loss_type=loss_type, attack=attack, **train_attack_params_prefix, **val_attack_params_prefix, **hyperparams)
    model_path = storage.save_model(model_storage_type, params, model)

    logits_clean_test = model.to(device)(attr_orig.to(device), adj_orig.to(device))
    test_accuracy_clean = accuracy(logits_clean_test.cpu(), labels.cpu(), idx_test)


    return {
        'loss_trace': loss_trace,
        'loss_trace_val': loss_trace_val,
        'gamma_trace': gamma_trace,
        'model_path': model_path,
        'acc_trace_train': acc_trace_train,
        'acc_trace_val': acc_trace_val,
        'acc_trace_train_pert': acc_trace_train_pert,
        'acc_trace_val_pert': acc_trace_val_pert,
        'test_accuracy_clean': test_accuracy_clean
    }