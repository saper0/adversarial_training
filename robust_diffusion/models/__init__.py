# import re
from typing import Any, Dict, Union

from robust_diffusion.models.rgnn import RGNN
from robust_diffusion.models.gcn import GCN, DenseGCN
from robust_diffusion.models.sgc import SGC
from robust_diffusion.models.gprgnn import GPRGNN
from robust_diffusion.models.gprgnn_dense import DenseGPRGNN
from robust_diffusion.models.chebynet2 import ChebNetII
from robust_diffusion.models.gat_weighted import GAT


MODEL_TYPE = Union[SGC, GCN, DenseGCN, GPRGNN, DenseGPRGNN, ChebNetII, GAT, RGNN]


def create_model(hyperparams: Dict[str, Any]) -> MODEL_TYPE:
    """Creates the model instance given the hyperparameters.

    Parameters
    ----------
    hyperparams : Dict[str, Any]
        Containing the hyperparameters.

    Returns
    -------
    model: MODEL_TYPE
        The created instance.
    """
    if 'model' not in hyperparams or hyperparams['model'] == 'GCN':
        return GCN(**hyperparams)
    if hyperparams['model'] == "SGC":
        return SGC(**hyperparams)
    if hyperparams['model'] == 'DenseGCN':
        return DenseGCN(**hyperparams)
    if hyperparams['model'] == "GPRGNN":
        return GPRGNN(**hyperparams)
    if hyperparams['model'] == "DenseGPRGNN":
        return DenseGPRGNN(**hyperparams)
    if hyperparams['model'] == 'ChebNetII':
        return ChebNetII(**hyperparams)
    if hyperparams['model'] == "GAT":
        return GAT(**hyperparams)
    return RGNN(**hyperparams)


__all__ = [GCN,
           DenseGCN,
           GPRGNN,
           GAT,
           DenseGPRGNN,
           ChebNetII,
           RGNN,
           create_model,
           MODEL_TYPE]
