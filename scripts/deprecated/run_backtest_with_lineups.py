#!/usr/bin/env python
"""
Run walk-forward backtest with lineup generation using pydfs-lineup-optimizer.

This script extends run_backtest.py by generating optimal DraftKings lineups
for each slate based on model predictions, evaluating lineup performance,
and exporting lineups in DraftKings CSV format.

Usage:
    # Cash game strategy with 1 lineup
    python scripts/run_backtest_with_lineups.py \
        --test-start 20250205 --test-end 20250206 \
        --contest-config cash_game.json --num-lineups 1

    # GPP tournament strategy with 20 lineups
    python scripts/run_backtest_with_lineups.py \
        --test-start 20250201 --test-end 20250210 --per-player \
        --contest-config gpp_tournament.json --num-lineups 20

    # Multi-entry with custom config
    python scripts/run_backtest_with_lineups.py \
        --test-start 20250201 --test-end 20250228 --per-player \
        --contest-config config/contests/multi_entry.json --num-lineups 50
"""

import argparse
import sys
import os
import logging
import yaml
from pathlib import Path
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.optimization.deprecated.backtest_lineup_integration import BacktestWithLineups
from src.data.loaders.historical_loader import HistoricalDataLoader
from src.filters import ColumnFilter, InjuryFilter
from src.filters.player_filters import PlayerIDFilter, PlayerNameFilter, PlayerIDFromCSVFilter


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run walk-forward backtest with lineup generation and evaluation"
    )

    # Data arguments
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Base data directory containing parquet files (default: data)"
    )

    # Test period arguments
    parser.add_argument(
        "--test-start",
        required=True,
        help="Test start date (YYYYMMDD format)"
    )

    parser.add_argument(
        "--test-end",
        required=True,
        help="Test end date (YYYYMMDD format)"
    )

    parser.add_argument(
        "--num-seasons",
        type=int,
        default=1,
        help="Number of seasons for training data (default: 1)"
    )

    # Model arguments
    parser.add_argument(
        "--model-type",
        choices=["xgboost", "random_forest", "linear"],
        default="xgboost",
        help="Model type (default: xgboost)"
    )

    parser.add_argument(
        "--feature-config",
        default="default_features",
        help="Feature configuration name or comma-separated list"
    )

    parser.add_argument(
        "--model-config",
        default=None,
        help="Path to YAML model configuration file"
    )

    parser.add_argument(
        "--per-player",
        action="store_true",
        help="Use per-player models instead of slate-level model"
    )

    parser.add_argument(
        "--min-player-games",
        type=int,
        default=10,
        help="Minimum games for per-player models (default: 10)"
    )

    parser.add_argument(
        "--min-benchmark-games",
        type=int,
        default=5,
        help="Minimum games for benchmark (default: 5)"
    )

    parser.add_argument(
        "--recalibrate-days",
        type=int,
        default=7,
        help="Recalibrate model every N days (default: 7)"
    )

    # Lineup generation arguments
    parser.add_argument(
        "--generate-lineups",
        action="store_true",
        default=True,
        help="Generate lineups (default: True, use --no-generate-lineups to disable)"
    )

    parser.add_argument(
        "--no-generate-lineups",
        action="store_false",
        dest="generate_lineups",
        help="Disable lineup generation"
    )

    parser.add_argument(
        "--contest-config",
        default="cash_game.json",
        help="Contest configuration file (default: cash_game.json). "
             "Available: cash_game.json, gpp_tournament.json, single_entry.json, multi_entry.json"
    )

    parser.add_argument(
        "--num-lineups",
        type=int,
        default=10,
        help="Number of lineups to generate per slate (default: 10)"
    )

    parser.add_argument(
        "--track-lineup-performance",
        action="store_true",
        default=True,
        help="Track lineup performance against actuals (default: True)"
    )

    parser.add_argument(
        "--no-track-lineup-performance",
        action="store_false",
        dest="track_lineup_performance",
        help="Disable lineup performance tracking"
    )

    # Output arguments
    parser.add_argument(
        "--output-dir",
        default="data/backtest_results",
        help="Output directory (default: data/backtest_results)"
    )

    parser.add_argument(
        "--no-save-models",
        action="store_true",
        help="Do not save trained models"
    )

    parser.add_argument(
        "--no-save-predictions",
        action="store_true",
        help="Do not save predictions to parquet"
    )

    # XGBoost hyperparameters
    parser.add_argument(
        "--max-depth",
        type=int,
        default=6,
        help="XGBoost max_depth (default: 6)"
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.05,
        help="XGBoost learning_rate (default: 0.05)"
    )

    parser.add_argument(
        "--n-estimators",
        type=int,
        default=200,
        help="XGBoost n_estimators (default: 200)"
    )

    # Other arguments
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Number of parallel jobs for per-player model training (default: -1 for all cores)"
    )

    parser.add_argument(
        "--rewrite-models",
        action="store_true",
        help="Force retraining of models"
    )

    parser.add_argument(
        "--resume-from-run",
        default=None,
        help="Resume from an existing run timestamp (e.g., 20250205_143022)"
    )

    parser.add_argument(
        "--salary-tiers",
        nargs="+",
        type=int,
        default=[0, 4000, 6000, 8000, 15000],
        help="Salary tier bins for analysis (default: 0 4000 6000 8000 15000)"
    )

    parser.add_argument(
        "--minutes-threshold",
        type=int,
        default=12,
        help="Minutes threshold for filtered metrics (default: 12)"
    )

    parser.add_argument(
        "--cmape-cap",
        type=float,
        default=8.0,
        help="Denominator cap for cMAPE (default: 8.0 FPTS)"
    )

    parser.add_argument(
        "--wmape-weight",
        choices=["actual_fpts", "actual_mins", "expected_mins"],
        default="actual_fpts",
        help="Weight type for WMAPE (default: actual_fpts)"
    )

    # Player filtering arguments
    parser.add_argument(
        "--filter-salary-min",
        type=int,
        default=None,
        help="Minimum salary filter"
    )

    parser.add_argument(
        "--filter-salary-max",
        type=int,
        default=None,
        help="Maximum salary filter"
    )

    parser.add_argument(
        "--filter-exclude-out",
        action="store_true",
        help="Exclude players ruled out with injuries"
    )

    parser.add_argument(
        "--filter-exclude-doubtful",
        action="store_true",
        help="Exclude players with doubtful injury status"
    )

    parser.add_argument(
        "--filter-exclude-questionable",
        action="store_true",
        help="Exclude players with questionable injury status"
    )

    parser.add_argument(
        "--filter-player-ids",
        type=str,
        default=None,
        help="Filter by player ID(s). Comma or space-separated list"
    )

    parser.add_argument(
        "--filter-player-names",
        type=str,
        default=None,
        help="Filter by player name(s). Comma or space-separated list"
    )

    parser.add_argument(
        "--filter-players-csv",
        type=str,
        default=None,
        help="Filter by player IDs from CSV file"
    )

    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Interactive mode: pause after each slate"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("="*80)
    print("NBA DFS WALK-FORWARD BACKTEST WITH LINEUP GENERATION")
    print("="*80)

    # Data directories
    print(f"Data Directory: {args.data_dir}")
    print(f"Output Directory: {args.output_dir}")

    # Backtest configuration
    print(f"Test Period: {args.test_start} to {args.test_end}")
    print(f"Model Type: {args.model_type}")
    print(f"Feature Config: {args.feature_config}")
    print(f"Per-Player Models: {args.per_player}")
    print(f"Number of Seasons: {args.num_seasons}")
    print(f"Parallel Jobs: {args.n_jobs} ({'all cores' if args.n_jobs == -1 else 'sequential' if args.n_jobs == 1 else f'{args.n_jobs} workers'})")

    # Lineup generation configuration
    print("\n" + "="*80)
    print("LINEUP GENERATION SETTINGS")
    print("="*80)
    print(f"Generate Lineups: {args.generate_lineups}")
    if args.generate_lineups:
        print(f"Contest Config: {args.contest_config}")
        print(f"Number of Lineups: {args.num_lineups}")
        print(f"Track Performance: {args.track_lineup_performance}")

    print("\n" + "="*80)
    print(f"Rewrite Models: {args.rewrite_models}")
    print(f"Save Models: {not args.no_save_models}")
    print(f"Save Predictions: {not args.no_save_predictions}")
    print(f"Resume From Run: {args.resume_from_run if args.resume_from_run else 'None (fresh start)'}")
    print(f"Salary Tiers: {args.salary_tiers}")
    print(f"Interactive Mode: {args.interactive}")
    print("="*80)
    print()

    # Calculate training period
    test_end_dt = datetime.strptime(args.test_end, '%Y%m%d')
    train_end = (test_end_dt - timedelta(days=1)).strftime('%Y%m%d')

    if args.num_seasons == 1:
        train_start = HistoricalDataLoader.get_season_start_date(args.test_start)
    else:
        train_start = HistoricalDataLoader.get_previous_season_start_date(args.test_start)

    print(f"Calculated Training Period: {train_start} to {train_end}\n")

    # Load model configuration
    if args.model_config:
        print(f"Loading model configuration from: {args.model_config}")
        with open(args.model_config, 'r') as f:
            model_config = yaml.safe_load(f)

        model_params = model_config.get('hyperparameters', {})

        if 'optimization_metadata' in model_config:
            metadata = model_config['optimization_metadata']
            print("Loaded optimized hyperparameters:")
            print(f"  Optimized at: {metadata.get('optimized_at', 'Unknown')}")
            print(f"  Training period: {metadata.get('train_start', 'Unknown')} to {metadata.get('train_end', 'Unknown')}")
            print(f"  Best MAPE: {metadata.get('best_mape', 'Unknown'):.2f}%")
            print(f"  Trials: {metadata.get('optimization_trials', 'Unknown')}")

        print("\nHyperparameters:")
        for key, value in model_params.items():
            print(f"  {key}: {value}")
        print()
    else:
        model_params = {
            'max_depth': args.max_depth,
            'learning_rate': args.learning_rate,
            'n_estimators': args.n_estimators,
            'min_child_weight': 5,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'objective': 'reg:squarederror',
            'random_state': 42
        }
        print("Using default hyperparameters from command-line arguments\n")

    # Build player filters
    player_filters = []

    if args.filter_salary_min is not None:
        salary_filter = ColumnFilter('salary', '>=', args.filter_salary_min)
        player_filters.append(salary_filter)
        print(f"Filter: salary >= {args.filter_salary_min}")

    if args.filter_salary_max is not None:
        salary_filter = ColumnFilter('salary', '<=', args.filter_salary_max)
        player_filters.append(salary_filter)
        print(f"Filter: salary <= {args.filter_salary_max}")

    if args.filter_exclude_out or args.filter_exclude_doubtful or args.filter_exclude_questionable:
        injury_filter = InjuryFilter(
            exclude_out=args.filter_exclude_out,
            exclude_doubtful=args.filter_exclude_doubtful,
            exclude_questionable=args.filter_exclude_questionable
        )
        player_filters.append(injury_filter)
        excluded = []
        if args.filter_exclude_out:
            excluded.append('OUT')
        if args.filter_exclude_doubtful:
            excluded.append('DOUBTFUL')
        if args.filter_exclude_questionable:
            excluded.append('QUESTIONABLE')
        print(f"Filter: exclude injury status {', '.join(excluded)}")

    if args.filter_player_ids:
        player_ids = [pid.strip() for pid in args.filter_player_ids.replace(',', ' ').split() if pid.strip()]
        if player_ids:
            player_id_filter = PlayerIDFilter(player_ids)
            player_filters.append(player_id_filter)
            ids_display = ', '.join(player_ids[:5])
            if len(player_ids) > 5:
                ids_display += f", ... (+{len(player_ids) - 5} more)"
            print(f"Filter: player ID in [{ids_display}]")

    if args.filter_player_names:
        player_names = [name.strip() for name in args.filter_player_names.replace(',', '|').split('|') if name.strip()]
        if player_names:
            player_name_filter = PlayerNameFilter(player_names, case_sensitive=False)
            player_filters.append(player_name_filter)
            names_display = ', '.join(player_names[:3])
            if len(player_names) > 3:
                names_display += f", ... (+{len(player_names) - 3} more)"
            print(f"Filter: player name contains [{names_display}]")

    if args.filter_players_csv:
        try:
            csv_filter = PlayerIDFromCSVFilter(args.filter_players_csv)
            player_filters.append(csv_filter)
            print(f"Filter: player IDs from CSV ({len(csv_filter.player_ids)} players)")
        except FileNotFoundError as e:
            print(f"ERROR: {e}")
            sys.exit(1)
        except ValueError as e:
            print(f"ERROR: {e}")
            sys.exit(1)

    if player_filters:
        print(f"\nTotal filters: {len(player_filters)}\n")

    # Create backtest with lineup generation
    backtest = BacktestWithLineups(
        train_start=train_start,
        train_end=train_end,
        test_start=args.test_start,
        test_end=args.test_end,
        model_type=args.model_type,
        model_params=model_params,
        feature_config=args.feature_config,
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        per_player_models=args.per_player,
        min_player_games=args.min_player_games,
        min_games_for_benchmark=args.min_benchmark_games,
        recalibrate_days=args.recalibrate_days,
        num_seasons=args.num_seasons,
        salary_tiers=args.salary_tiers,
        save_models=not args.no_save_models,
        save_predictions=not args.no_save_predictions,
        n_jobs=args.n_jobs,
        rewrite_models=args.rewrite_models,
        resume_from_run=args.resume_from_run,
        minutes_threshold=args.minutes_threshold,
        cmape_cap=args.cmape_cap,
        wmape_weight=args.wmape_weight,
        player_filters=player_filters if player_filters else None,
        interactive=args.interactive,
        # Lineup generation specific parameters
        generate_lineups=args.generate_lineups,
        contest_config=args.contest_config,
        num_lineups=args.num_lineups,
        track_lineup_performance=args.track_lineup_performance
    )

    # Run backtest with lineup generation
    results = backtest.run_with_lineups()

    if 'error' in results:
        print(f"\nERROR: {results['error']}")
        sys.exit(1)

    print("\n" + "="*80)
    print("BACKTEST COMPLETE")
    print("="*80)
    print(f"Test Slates: {results.get('test_slates', 0)}")
    print(f"Total Players Evaluated: {results.get('total_players', 0)}")

    if 'lineup_summary' in results:
        lineup_summary = results['lineup_summary']
        print("\nLINEUP GENERATION SUMMARY:")
        print(f"  Total Lineups Generated: {lineup_summary.get('total_lineups', 0)}")
        print(f"  Avg Projected vs Actual Correlation: {lineup_summary.get('avg_correlation', 0):.3f}")
        print(f"  Avg Error: {lineup_summary.get('avg_error_pct', 0):.1f}%")

    print("\nResults saved to:")
    print(f"  {results.get('output_dir', args.output_dir)}")

    if args.generate_lineups:
        print(f"\nLineup files:")
        print(f"  CSV (DraftKings upload): {results.get('output_dir')}/lineups/*_lineups.csv")
        print(f"  JSON (full details): {results.get('output_dir')}/lineups/*_lineups.json")
        print(f"  Performance report: {results.get('output_dir')}/lineup_performance_report.csv")

    print("="*80)


if __name__ == "__main__":
    main()