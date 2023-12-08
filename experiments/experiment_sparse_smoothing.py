import logging
import os
from typing import Any, Dict, Optional, Sequence, Union

import numpy as np
from sacred import Experiment
import seml
from sparse_smoothing.models import GCN, GAT, APPNPNet, CNN_MNIST, GIN
from sparse_smoothing.training import train_gnn, train_pytorch
from sparse_smoothing.prediction import predict_smooth_gnn, predict_smooth_pytorch
from sparse_smoothing.cert import binary_certificate
from sparse_smoothing.utils import (load_and_standardize, split, accuracy_majority,
                                    sample_perturbed_mnist, sample_batch_pyg, get_mnist_dataloaders)
import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader as PyGDataLoader
from torch_sparse import SparseTensor


from robust_diffusion.attacks import Attack, create_attack
from experiments.common import prepare_attack_experiment, run_global_attack

from robust_diffusion.attacks import SPARSE_ATTACKS, LOCAL_ATTACKS
from robust_diffusion.models.gprgnn import from_dense
from robust_diffusion.models.gprgnn_dense import from_sparse
from robust_diffusion.models import create_model
from robust_diffusion.helper.io import Storage

ex = Experiment()
seml.setup_logger(ex)


@ex.config
def config():
    overwrite = None

    if seml is not None:
        db_collection = None
        if db_collection is not None:
            ex.observers.append(seml.create_mongodb_observer(db_collection, overwrite=overwrite))

    # default params
    data_dir = './data'
    dataset = 'cora_ml'
    make_undirected = True
    binary_attr = False
    data_device = 0
    balance_test = True
    n_per_class = 20

    device = 0
    seed = 0

    split_type = 'inductive'
    idx_to_attack = 'test'

    artifact_dir = 'cache'
    model_label = "ChebNetII"
    model_storage_type = 'pretrained_ind_adv_final'

    pf_plus_adj = 0.01
    pf_minus_adj = 0.6

    n_samples_pre_eval = 10
    n_samples_eval = 1000
    batch_size_eval = 10

    conf_alpha = 0.05

    save_dir = 'cache/sparse_smoothing'

    debug_level = "info"

    model_params_add = {}

    model_path = None


class SparseSmoothingModel(torch.nn.Module):

    def __init__(self, model, attr):
        super().__init__()
        self.model = model
        self.attr = attr

    def forward(self, attr_idx, edge_idx, n, d):
        batch_size = n // self.attr.shape[0]
        return self.model(self.attr.repeat(batch_size, 1), (edge_idx, None))


def get_heatmap_loup(grid_lower, grid_upper, idx_test=None, mask: np.ndarray = None):
    if mask is None:
        grid_lower = grid_lower[idx_test]
        grid_upper = grid_upper[idx_test]
    else:
        grid_lower = grid_lower[idx_test][mask[idx_test]]
        grid_upper = grid_upper[idx_test][mask[idx_test]]
    heatmap_loup = (grid_lower > grid_upper).mean(0).T
    heatmap_loup[0, 0] = 1
    return heatmap_loup


@ex.automain
def run(_config, data_dir: str, dataset: str, split_type: str, n_per_class: int, idx_to_attack: str,
        binary_attr: bool, make_undirected: bool, seed: int, artifact_dir: str, balance_test: bool, model_label: str, model_params_add: Dict[str, Any], model_storage_type: str, device: Union[str, int],
        data_device: Union[str, int], debug_level: str, pf_plus_adj: float, pf_minus_adj: float, n_samples_pre_eval: int, n_samples_eval: int, batch_size_eval: int, conf_alpha: float, save_dir: Optional[str], model_path: Optional[str]):
    """
    Instantiates a sacred experiment executing a global direct attack run for a given model configuration.
    Caches the perturbed adjacency to storage and evaluates the models perturbed accuracy. 
    Global evasion attacks allow all nodes of the graph to be perturbed under the given budget.
    Direct attacks are used to attack a model without the use of a surrogate model.

    Parameters
    ----------
    data_dir : str
        Path to data folder that contains the dataset
    dataset : str
        Name of the dataset. Either one of: `cora_ml`, `citeseer`, `pubmed` or an ogbn dataset
    device : Union[int, torch.device]
        The device to use for training. Must be `cpu` or GPU id
    data_device : Union[int, torch.device]
        The device to use for storing the dataset. For batched models (like PPRGo) this may differ from the device parameter. 
        In all other cases device takes precedence over data_device
    make_undirected : bool
        Normalizes adjacency matrix with symmetric degree normalization (non-scalable implementation!)
    binary_attr : bool
        If true the attributes are binarized (!=0)
    attack : str
        The name of the attack class to use. Supported attacks are:
            - PRBCD
            - GreedyRBCD
            - DICE
            - FGSM
            - PGD
    attack_params : Dict[str, Any], optional
        The attack hyperparams to be passed as keyword arguments to the constructor of the attack class
    epsilons: List[float]
        The budgets for which the attack on the model should be executed.
    model_label : str, optional
        The name given to the model at train time using the experiment_train.py 
        This name is used as an identifier in combination with the dataset configuration to retrieve 
        the model to be attacked from storage. If None, all models that were fit on the given dataset 
        are attacked.
    artifact_dir: str
        The path to the folder that acts as TinyDB Storage for pretrained models
    model_storage_type: str
        The name of the storage (TinyDB) table name the model to be attacked is retrieved from.
    pert_adj_storage_type: str
        The name of the storage (TinyDB) table name the perturbed adjacency matrix is stored to
    pert_attr_storage_type: str
        The name of the storage (TinyDB) table name the perturbed attribute matrix is stored to

    Returns
    -------
    List[Dict[str, any]]
        List of result dictionaries. One for every combination of model and epsilon.
        Each result dictionary contains the model labels, epsilon value and the perturbed accuracy
    """

    if model_path:
        *coll, id_ = '.'.join(
            model_path.split(os.path.sep)[-1].split('.')[:-1]).split('_')
        coll = '_'.join(coll)
        id_ = int(id_)
        artifact_dir = os.path.join(model_path.split('cache')[0], 'cache')
        storage = Storage(artifact_dir, experiment=None)
        document = storage._get_db(coll).get(doc_id=id_)
        document['artifact'] = torch.load(model_path)
        dataset = document['params']['dataset']
        split_type = document['params']['split_type']
        n_per_class = document['params']['n_per_class']
        binary_attr = document['params']['binary_attr']
        make_undirected = document['params']['make_undirected']
        seed = document['params']['seed']
        model_label = document['params']['label']
        model_params_add = {}

    print(_config)

    pf_plus_att = 0
    pf_minus_att = 0

    batch_size_eval = min(batch_size_eval, n_samples_eval)

    sample_config = {
        'pf_plus_adj': pf_plus_adj,
        'pf_plus_att': pf_plus_att,
        'pf_minus_adj': pf_minus_adj,
        'pf_minus_att': pf_minus_att,
    }

    sample_config_eval = sample_config.copy()
    sample_config_eval['n_samples'] = n_samples_eval

    sample_config_pre_eval = sample_config.copy()
    sample_config_pre_eval['n_samples'] = n_samples_pre_eval

    results = []
    surrogate_model_label = False

    (
        attr, adj, labels, idx_attack, storage, attack_params, pert_params, model_params, m
    ) = prepare_attack_experiment(
        data_dir, dataset, split_type, balance_test, n_per_class, 'PRBCD', {}, [
            0], idx_to_attack, binary_attr, make_undirected,  seed, artifact_dir,
        'pert_adj_storage_type', 'pert_attr_storage_type', model_label, model_params_add, model_storage_type, device, surrogate_model_label,
        data_device, debug_level, ex
    )

    if dataset == 'ogbn-arxiv':
        adj = adj.float()

    if model_label is not None and model_label:
        model_params['label'] = model_label

    if model_path:
        model = create_model(document['params'])
        model.load_state_dict(document['artifact'])
        models_and_hyperparams = [(model, document['params'])]
    else:
        models_and_hyperparams = storage.find_models(model_storage_type, model_params)

    if isinstance(adj, SparseTensor):
        row, col, edge_weight = adj.t().coo()
        edge_idx = torch.stack([row, col], dim=0).to(device)
    elif isinstance(adj, tuple):
        edge_idx = adj[0].to(device)
    attr_idx = torch.tensor([[0, 0]]).cuda()
    labels = labels.cuda()

    n, d = attr.shape
    nc = labels.max() + 1

    for model, hyperparams in models_and_hyperparams:
        model_label = hyperparams["label"]
        logging.info(
            f"Evaluate sparse smoothing for model '{model_label} of type {type(model)}' with hyperparams {hyperparams}.")

        if hasattr(model, 'gdc_params') and isinstance(model.gdc_params, dict):
            model.gdc_params['use_cpu'] = True

        model_ = SparseSmoothingModel(model.to(device), attr.to(device))

        pre_votes = predict_smooth_gnn(attr_idx=attr_idx, edge_idx=edge_idx,
                                       sample_config=sample_config_pre_eval,
                                       model=model_, n=n, d=d, nc=nc,
                                       batch_size=batch_size_eval)

        votes = predict_smooth_gnn(attr_idx=attr_idx, edge_idx=edge_idx,
                                   sample_config=sample_config_eval,
                                   model=model_, n=n, d=d, nc=nc,
                                   batch_size=batch_size_eval)

        acc_majority = accuracy_majority(
            votes=votes, labels=labels.cpu().numpy(), idx=idx_attack)

        votes_max = votes.max(1)[idx_attack]

        agreement = (votes.argmax(1) == pre_votes.argmax(1)).mean()

        grid_base, grid_lower, grid_upper = binary_certificate(
            votes=votes, pre_votes=pre_votes, n_samples=n_samples_eval,
            conf_alpha=conf_alpha, pf_plus=pf_plus_adj, pf_minus=pf_minus_adj)

        mean_max_ra_base = (grid_base > 0.5)[:, :, 0].argmin(1).mean()
        mean_max_rd_base = (grid_base > 0.5)[:, 0, :].argmin(1).mean()
        mean_max_ra_loup = (grid_lower >= grid_upper)[:, :, 0].argmin(1).mean()
        mean_max_rd_loup = (grid_lower >= grid_upper)[:, 0, :].argmin(1).mean()

        run_id = _config['overwrite']
        db_collection = _config['db_collection']

        if save_dir is not None:
            dict_to_save = {'idx_attack': idx_attack,
                            'pre_votes': pre_votes,
                            'votes': votes,
                            'grid_base': grid_base,
                            'grid_lower': grid_lower,
                            'grid_upper': grid_upper}
            torch.save(dict_to_save,
                       f'{save_dir}/{db_collection}_{run_id}')

        results.append({
            'label': model_label,
            'acc_majority': acc_majority,
            'above_99': (votes_max >= 0.99 * n_samples_eval).mean(),
            'above_95': (votes_max >= 0.95 * n_samples_eval).mean(),
            'above_90': (votes_max >= 0.90 * n_samples_eval).mean(),
            'mean_max_ra_base': mean_max_ra_base,
            'mean_max_rd_base': mean_max_rd_base,
            'mean_max_ra_loup': mean_max_ra_loup,
            'mean_max_rd_loup': mean_max_rd_loup,
            'heatmap': get_heatmap_loup(grid_lower, grid_upper),
            'heatmap_correct': get_heatmap_loup(grid_lower, grid_upper, idx_attack, votes.argmax(-1) == labels.cpu().numpy()),
            'agreement': agreement,
        })

    assert results

    return results
