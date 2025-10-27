#!/usr/bin/env python
"""
Run walk-forward backtest simulation across multiple historical dates.

This script validates the production DFS pipeline (predict → optimize → score)
on historical data by iterating through slates sequentially and tracking
lineup performance vs actual results.

Usage:
    python scripts/run_walk_forward_backtest.py --start-date 20250201 --end-date 20250210
    python scripts/run_walk_forward_backtest.py --start-date 20250201 --end-date 20250210 --num-lineups 20
    python scripts/run_walk_forward_backtest.py --start-date 20250201 --end-date 20250210 --model stacked_xgb_rf
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.evaluation.walk_forward_simulation import WalkForwardSimulation


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run walk-forward backtest simulation across historical dates"
    )

    parser.add_argument(
        "--start-date",
        required=True,
        help="Start date in YYYYMMDD format"
    )

    parser.add_argument(
        "--end-date",
        required=True,
        help="End date in YYYYMMDD format (exclusive)"
    )

    parser.add_argument(
        "--num-lineups",
        type=int,
        default=1,
        help="Number of lineups to generate per slate (default: 1)"
    )

    parser.add_argument(
        "--strategy",
        default="balanced",
        choices=['conservative', 'balanced', 'aggressive'],
        help="Lineup generation strategy (default: balanced)"
    )

    parser.add_argument(
        "--features",
        default="default_features",
        help="Feature configuration name (default: default_features)"
    )

    parser.add_argument(
        "--model",
        default="xgboost_default",
        help="Model configuration name (default: xgboost_default)"
    )

    parser.add_argument(
        "--num-training-seasons",
        type=int,
        default=2,
        help="Number of seasons of training data (default: 2)"
    )

    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory containing historical parquet data (default: data)"
    )

    parser.add_argument(
        "--output-dir",
        default="data/backtest_results",
        help="Directory to save simulation results (default: data/backtest_results)"
    )

    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save prediction CSVs for each date"
    )

    parser.add_argument(
        "--save-lineups",
        action="store_true",
        help="Save lineup CSVs for each date"
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress messages"
    )

    return parser.parse_args()


def validate_dates(start_date: str, end_date: str) -> None:
    """Validate date format and range."""
    if len(start_date) != 8 or not start_date.isdigit():
        raise ValueError(f"Invalid start_date format: {start_date}. Use YYYYMMDD.")

    if len(end_date) != 8 or not end_date.isdigit():
        raise ValueError(f"Invalid end_date format: {end_date}. Use YYYYMMDD.")

    if start_date >= end_date:
        raise ValueError(f"start_date must be before end_date: {start_date} >= {end_date}")


def main():
    args = parse_args()

    # Validate inputs
    try:
        validate_dates(args.start_date, args.end_date)
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    # Initialize simulation
    simulation = WalkForwardSimulation(
        data_dir=args.data_dir,
        feature_config=args.features,
        model_config=args.model,
        output_dir=args.output_dir,
        verbose=not args.quiet
    )

    # Run simulation
    try:
        results = simulation.run(
            start_date=args.start_date,
            end_date=args.end_date,
            num_lineups=args.num_lineups,
            strategy=args.strategy,
            num_training_seasons=args.num_training_seasons,
            save_predictions=args.save_predictions,
            save_lineups=args.save_lineups
        )

        if not results:
            print("ERROR: No results generated. Check date range and data availability.")
            sys.exit(1)

        if not args.quiet:
            print("\n✓ Simulation completed successfully")

    except Exception as e:
        print(f"ERROR: Simulation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
