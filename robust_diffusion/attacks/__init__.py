from typing import Union

from .dice import DICE
from .fgsm import FGSM
from .greedy_rbcd import GreedyRBCD
from .pgd import PGD
from .prbcd import PRBCD
from .nettack import Nettack
from .base_attack import Attack
from .sga import SGA
from .prbcd_constrained import LRBCD

ATTACK_TYPE = Union[SGA, DICE, FGSM, GreedyRBCD, PGD, PRBCD, LRBCD, Nettack]
SPARSE_ATTACKS = [GreedyRBCD.__name__, PRBCD.__name__, DICE.__name__, LRBCD.__name__]
LOCAL_ATTACKS = [SGA.__name__, Nettack.__name__]


def create_attack(attack: str, *args, **kwargs) -> Attack:
    """Creates the model instance given the hyperparameters.

    Parameters
    ----------
    attack : str
        Identifier of the attack
    kwargs
        Containing the hyperparameters

    Returns
    -------
    Union[FGSM, GreedyRBCD, PRBCD]
        The created instance
    """
    if not any([attack.lower() == attack_model.__name__.lower() for attack_model in ATTACK_TYPE.__args__]):
        raise ValueError(f'The attack {attack} is not in {ATTACK_TYPE.__args__}')

    return globals()[attack](*args, **kwargs)


__all__ = [FGSM, GreedyRBCD,
           PRBCD, LRBCD, create_attack, ATTACK_TYPE, SPARSE_ATTACKS, Nettack, SGA]
