"""
Walk-forward backtesting framework for time series validation.
"""

from .walk_forward import WalkForwardBacktest
from .validator import Validator
from .trainers import PerPlayerTrainer, PerSlateTrainer

__all__ = [
    'WalkForwardBacktest',
    'Validator',
    'PerPlayerTrainer',
    'PerSlateTrainer',
]