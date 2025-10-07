"""
Run walk-forward backtest for NBA DFS projections.

Refactored to use simplified module structure:
- src.backtest (consolidated backtest logic)
- src.features (feature engineering)
- src.metrics (evaluation)
- src.data (data loading)
"""

import sys
from pathlib import Path
import logging
import argparse
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backtest import BacktestConfig, run_backtest

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='Run walk-forward backtest for NBA DFS projections'
    )

    parser.add_argument(
        '--start',
        type=str,
        required=True,
        help='Start date in YYYYMMDD format'
    )
    parser.add_argument(
        '--end',
        type=str,
        required=True,
        help='End date in YYYYMMDD format'
    )
    parser.add_argument(
        '--lookback',
        type=int,
        default=90,
        help='Training window in days (default: 90)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='xgboost',
        choices=['xgboost', 'random_forest', 'linear'],
        help='Model type (default: xgboost)'
    )
    parser.add_argument(
        '--windows',
        type=int,
        nargs='+',
        default=[4],
        help='Window sizes for rolling features (default: 4). Examples: --windows 4, --windows 3 4 5 10'
    )
    parser.add_argument(
        '--db',
        type=str,
        default='nba_dfs.db',
        help='Path to database file (default: nba_dfs.db)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/backtest_results',
        help='Output directory for results (default: data/backtest_results)'
    )
    parser.add_argument(
        '--min-training-games',
        type=int,
        default=500,
        help='Minimum training games required (default: 500)'
    )
    parser.add_argument(
        '--per-player-models',
        action='store_true',
        help='Train separate model for each player'
    )
    parser.add_argument(
        '--min-player-games',
        type=int,
        default=10,
        help='Minimum games for per-player models (default: 10)'
    )
    parser.add_argument(
        '--no-save-daily',
        action='store_true',
        help='Do not save daily results'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    config = BacktestConfig(
        start_date=args.start,
        end_date=args.end,
        lookback_days=args.lookback,
        model_type=args.model,
        rolling_window_sizes=args.windows,
        per_player_models=args.per_player_models,
        min_player_games=args.min_player_games,
        min_training_games=args.min_training_games,
        save_daily_results=not args.no_save_daily,
        output_dir=args.output_dir
    )

    print("\n" + "="*80)
    print("NBA DFS WALK-FORWARD BACKTEST")
    print("="*80)
    print(f"Period: {config.start_date} to {config.end_date}")
    print(f"Training Window: {config.lookback_days} days")
    print(f"Model: {config.model_type}")
    windows_str = ', '.join(map(str, config.rolling_window_sizes))
    num_features = len(config.rolling_window_sizes) * 6 + 1
    print(f"Rolling Windows: [{windows_str}] ({num_features} features)")
    print(f"Min Training Games: {config.min_training_games}")
    print(f"Per-Player Models: {config.per_player_models}")
    if config.per_player_models:
        print(f"Min Player Games: {config.min_player_games}")
    print(f"Output Directory: {config.output_dir}")
    print("="*80)

    config_path = Path(config.output_dir) / 'backtest_config.json'
    config.save(str(config_path))
    logger.info(f"Config saved to {config_path}")

    results = run_backtest(config, db_path=args.db)

    if 'error' in results:
        print(f"\nERROR: {results['error']}")
        return 1

    print("\n" + "="*80)
    print("BACKTEST RESULTS")
    print("="*80)
    print(f"Slates Processed: {results['num_slates']}")
    print(f"Date Range: {results['date_range']}")
    print(f"\nMean MAPE: {results['mean_mape']:.2f}%")
    print(f"Median MAPE: {results['median_mape']:.2f}%")
    print(f"Std MAPE: {results['std_mape']:.2f}%")
    print(f"\nMean RMSE: {results['mean_rmse']:.2f}")
    print(f"Mean Correlation: {results['mean_correlation']:.3f}")
    print(f"\nTotal Players: {results['total_players_evaluated']}")
    print(f"Avg Players/Slate: {results['avg_players_per_slate']:.1f}")
    print("="*80)

    results_path = Path(config.output_dir) / 'backtest_results.json'
    with open(results_path, 'w') as f:
        results_copy = results.copy()
        if 'daily_results' in results_copy:
            results_copy['daily_results'] = results_copy['daily_results'].to_dict('records')
        json.dump(results_copy, f, indent=2, default=str)

    print(f"\nResults saved to {results_path}")

    print("\n" + "="*80)
    print("BACKTEST COMPLETE")
    print("="*80)

    if results.get('mean_mape', 100) < 30:
        print("SUCCESS: Model meets target accuracy (MAPE < 30%)")
        return 0
    elif results.get('mean_mape', 100) < 35:
        print("GOOD: Model performance is close to target")
        return 0
    else:
        print("WARNING: Model performance needs improvement")
        return 0


if __name__ == '__main__':
    sys.exit(main())
