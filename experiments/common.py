from typing import Any, Dict, Sequence, Union
import logging

import numpy as np
import torch
import torch.nn.functional as F
from sacred import Experiment
from torch_sparse import SparseTensor

from robust_diffusion.data import prep_graph, split,split_squirrel,  split_inductive, count_edges_for_idx
from robust_diffusion.helper.io import Storage
from robust_diffusion.models import BATCHED_PPR_MODELS
from robust_diffusion.helper.utils import accuracy


def prepare_attack_experiment(data_dir: str, dataset: str, split_type: str, balance_test: bool, n_per_class: int, attack: str, attack_params: Dict[str, Any],
                              epsilons: Sequence[float], idx_to_attack: str, binary_attr: bool, make_undirected: bool,
                              seed: int, artifact_dir: str, pert_adj_storage_type: str, pert_attr_storage_type: str,
                              model_label: str, model_params_add: Dict[str, Any], model_storage_type: str, device: Union[str, int],
                              surrogate_model_label: str, data_device: Union[str, int], debug_level: str,
                              ex: Experiment):

    if debug_level is not None and isinstance(debug_level, str):
        logger = logging.getLogger()
        if debug_level.lower() == "info":
            logger.setLevel(logging.INFO)
        if debug_level.lower() == "debug":
            logger.setLevel(logging.DEBUG)
        if debug_level.lower() == "critical":
            logger.setLevel(logging.CRITICAL)
        if debug_level.lower() == "error":
            logger.setLevel(logging.ERROR)

    if not torch.cuda.is_available():
        assert device == "cpu", "CUDA is not availble, set device to 'cpu'"
        assert data_device == "cpu", "CUDA is not availble, set device to 'cpu'"

    logging.info({
        'dataset': dataset, 'attack': attack, 'attack_params': attack_params, 'epsilons': epsilons,
        'make_undirected': make_undirected, 'binary_attr': binary_attr, 'seed': seed,
        'artifact_dir':  artifact_dir, 'pert_adj_storage_type': pert_adj_storage_type,
        'pert_attr_storage_type': pert_attr_storage_type, 'model_label': model_label,
        'model_storage_type': model_storage_type, 'device': device, 'data_device': data_device
    })

    assert sorted(epsilons) == epsilons, 'argument `epsilons` must be a sorted list'
    assert len(np.unique(epsilons)) == len(epsilons),\
        'argument `epsilons` must be unique (strictly increasing)'
    assert all([eps >= 0 for eps in epsilons]), 'all elements in `epsilons` must be greater than 0'

    # To increase consistency between runs
    torch.manual_seed(seed)
    np.random.seed(seed)

    graph = prep_graph(dataset, data_device, dataset_root=data_dir, make_undirected=make_undirected,
                       binary_attr=binary_attr, return_original_split=dataset.startswith('ogbn'))

    attr, adj, labels = graph[:3]

    # different data splits
    if dataset.startswith('ogbn'):
        split_train = graph[3]['train']
        split_val = graph[3]['valid']
        split_test = graph[3]['test']
    elif dataset in ['squirrel', 'chameleon']:
        logging.info(f'splitting for {dataset} 60/20/20')
        split_train, split_val, split_test = split_squirrel(labels.cpu().numpy(), train_size=0.6, val_size=0.2, test_size=0.2, seed=seed)
    elif split_type == 'transductive':
        split_train, split_val, split_test = split(labels.cpu().numpy(), n_per_class=n_per_class)
    elif split_type == 'inductive':
        split_train, split_unlabeled, split_val, split_test = split_inductive(labels.cpu().numpy(), n_per_class=n_per_class, balance_test=balance_test)

    # which idx to attack 
    if idx_to_attack == 'train':
        idx_attack = split_train
    if idx_to_attack == 'test':
        idx_attack = split_test
    if idx_to_attack == 'val':
        idx_attack = split_val
    if idx_to_attack == 'unlabeled':
        assert split_type == 'inductive'
        idx_attack = split_unlabeled

    storage = Storage(artifact_dir, experiment=ex)

    attack_params = dict(attack_params)
    if "ppr_cache_params" in attack_params.keys():
        ppr_cache_params = dict(attack_params["ppr_cache_params"])
        ppr_cache_params['dataset'] = dataset
        attack_params["ppr_cache_params"] = ppr_cache_params

    pert_params = dict(dataset=dataset,
                       binary_attr=binary_attr,
                       make_undirected=make_undirected,
                       seed=seed,
                       attack=attack,
                       model=model_label,
                       surrogate_model=surrogate_model_label,
                       attack_params=attack_params)
    pert_params.update(model_params_add)
    pert_params.update({'split_type':split_type})
    pert_params.update({'n_per_class':n_per_class})
    pert_params.update({'idx_to_attack':idx_to_attack})

    model_params = dict(dataset=dataset,
                        binary_attr=binary_attr,
                        make_undirected=make_undirected,
                        seed=seed)
    model_params.update(model_params_add)
    model_params.update({'split_type':split_type})
    model_params.update({'n_per_class':n_per_class})

    if model_label is not None and model_label:
        model_params["label"] = model_label

    n_attack_edges = count_edges_for_idx(adj, idx_attack) # num edges connected to attacked nodes
    if make_undirected:
        m = int(n_attack_edges / 2)
    else:
        m = int(n_attack_edges)

    return attr, adj, labels, idx_attack, storage, attack_params, pert_params, model_params, m


def run_global_attack(epsilon, m, storage, pert_adj_storage_type, pert_attr_storage_type,
                      pert_params, adversary, model_label, save_artifacts=True):
    n_perturbations = int(round(epsilon * m))

    pert_adj = storage.load_artifact(pert_adj_storage_type, {**pert_params, **{'epsilon': epsilon}})
    if type(pert_attr_storage_type) == torch.Tensor: # if pert_attr_storage_type we skip loading and take it as attr (for structure-only attacks)
        pert_attr = pert_attr_storage_type.clone()
    else:
        pert_attr = storage.load_artifact(pert_attr_storage_type, {**pert_params, **{'epsilon': epsilon}})

    if pert_adj is not None and pert_attr is not None:
        logging.info(
            f"Found cached perturbed adjacency and attribute matrix for model '{model_label}' and eps {epsilon}")
        adversary.set_pertubations(pert_adj, pert_attr)
    else:
        logging.info(f"No cached perturbations found for model '{model_label}' and eps {epsilon}. Execute attack...")
        adversary.attack(n_perturbations)
        pert_adj, pert_attr = adversary.get_pertubations()

        if (n_perturbations > 0) and save_artifacts:
            storage.save_artifact(pert_adj_storage_type, {**pert_params, **{'epsilon': epsilon}}, pert_adj)
            if type(pert_attr_storage_type) != torch.Tensor:  # if pert_attr_storage_type we skip saving (for structure-only attacks)
                storage.save_artifact(pert_attr_storage_type, {**pert_params, **{'epsilon': epsilon}}, pert_attr)


def sample_attack_nodes(logits: torch.Tensor, labels: torch.Tensor, nodes_idx,
                        adj: SparseTensor, topk: int, min_node_degree: int):
    assert logits.shape[0] == labels.shape[0]
    if isinstance(nodes_idx, torch.Tensor):
        nodes_idx = nodes_idx.cpu()
    node_degrees = adj[nodes_idx.tolist()].sum(-1)

    suitable_nodes_mask = (node_degrees >= min_node_degree).cpu()

    labels = labels.cpu()[suitable_nodes_mask]
    confidences = F.softmax(logits.cpu()[suitable_nodes_mask], dim=-1)

    correctly_classifed = confidences.max(-1).indices == labels

    logging.info(
        f"Found {sum(suitable_nodes_mask)} suitable '{min_node_degree}+ degree' nodes out of {len(nodes_idx)} "
        f"candidate nodes to be sampled from for the attack of which {correctly_classifed.sum().item()} have the "
        "correct class label")

    assert sum(suitable_nodes_mask) >= (topk * 4), \
        f"There are not enough suitable nodes to sample {(topk*4)} nodes from"

    _, max_confidence_nodes_idx = torch.topk(confidences[correctly_classifed].max(-1).values, k=topk)
    _, min_confidence_nodes_idx = torch.topk(-confidences[correctly_classifed].max(-1).values, k=topk)

    rand_nodes_idx = np.arange(correctly_classifed.sum().item())
    rand_nodes_idx = np.setdiff1d(rand_nodes_idx, max_confidence_nodes_idx)
    rand_nodes_idx = np.setdiff1d(rand_nodes_idx, min_confidence_nodes_idx)
    rnd_sample_size = min((topk * 2), len(rand_nodes_idx))
    rand_nodes_idx = np.random.choice(rand_nodes_idx, size=rnd_sample_size, replace=False)

    return (np.array(nodes_idx[suitable_nodes_mask][correctly_classifed][max_confidence_nodes_idx])[None].flatten(),
            np.array(nodes_idx[suitable_nodes_mask][correctly_classifed][min_confidence_nodes_idx])[None].flatten(),
            np.array(nodes_idx[suitable_nodes_mask][correctly_classifed][rand_nodes_idx])[None].flatten())


def get_local_attack_nodes(attr, adj, labels, surrogate_model, idx_test, device, topk=10, min_node_degree=2):

    with torch.no_grad():
        surrogate_model = surrogate_model.to(device)
        surrogate_model.eval()
        if type(surrogate_model) in BATCHED_PPR_MODELS.__args__:
            logits = surrogate_model.forward(attr, adj, ppr_idx=np.array(idx_test))
        else:
            logits = surrogate_model(attr.to(device), adj.to(device))[idx_test]

        acc = accuracy(logits.cpu(), labels.cpu()[idx_test], np.arange(logits.shape[0]))

    logging.info(f"Sample Attack Nodes for model with accuracy {acc:.4}")

    max_confidence_nodes_idx, min_confidence_nodes_idx, rand_nodes_idx = sample_attack_nodes(
        logits, labels[idx_test], idx_test, adj, topk,  min_node_degree)
    tmp_nodes = np.concatenate((max_confidence_nodes_idx, min_confidence_nodes_idx, rand_nodes_idx))
    logging.info(
        f"Sample the following attack nodes:\n{max_confidence_nodes_idx}\n{min_confidence_nodes_idx}\n{rand_nodes_idx}")
    return tmp_nodes
