"""
Backward compatibility module for walk_forward_backtest imports.

This module provides backward compatibility for code that imports
from src.walk_forward_backtest. New code should import from
src.evaluation.backtest.walk_forward instead.

Deprecated: This module will be removed in a future version.
"""

import warnings
from src.evaluation.backtest.walk_forward import WalkForwardBacktest
from src.evaluation.backtest.trainers.per_player import _train_single_player_model

warnings.warn(
    "Importing from src.walk_forward_backtest is deprecated. "
    "Please import from src.evaluation.backtest instead.",
    DeprecationWarning,
    stacklevel=2
)

__all__ = [
    'WalkForwardBacktest',
    '_train_single_player_model',
]