"""
Model training strategies for walk-forward backtesting.
"""

from .per_player import PerPlayerTrainer
from .per_slate import PerSlateTrainer

__all__ = [
    'PerPlayerTrainer',
    'PerSlateTrainer',
]