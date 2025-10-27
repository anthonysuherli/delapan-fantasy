"""
Backward compatibility module for walk_forward_backtest.

This module has been refactored into src.evaluation.backtest.
Please update your imports to use the new location.
"""

import warnings

# Issue deprecation warning
warnings.warn(
    "The walk_forward_backtest module has been refactored. "
    "Please import from src.evaluation.backtest instead:\n"
    "  from src.evaluation.backtest import WalkForwardBacktest\n"
    "This compatibility module will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export from new location for backward compatibility
from src.evaluation.backtest import WalkForwardBacktest
from src.evaluation.backtest.trainers.per_player import _train_single_player_model

__all__ = [
    'WalkForwardBacktest',
    '_train_single_player_model',
]