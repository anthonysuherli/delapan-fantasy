#!/usr/bin/env python
"""
Run walk-forward backtest with per-player or slate-level models.

This script implements the same functionality as backtest_season.ipynb but
as a command-line tool for easier automation and CI/CD integration.

Usage:
    python scripts/run_backtest.py --test-start 20250205 --test-end 20250206
    python scripts/run_backtest.py --test-start 20250201 --test-end 20250228 --per-player
    python scripts/run_backtest.py --test-start 20250201 --test-end 20250228 --feature-config base_features
"""

import argparse
import sys
import os
import logging
import yaml
from pathlib import Path
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.walk_forward_backtest import WalkForwardBacktest
from src.data.loaders.historical_loader import HistoricalDataLoader
from src.filters import ColumnFilter, InjuryFilter, CompositeFilter


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run walk-forward backtest with benchmark comparison"
    )

    parser.add_argument(
        "--db-path",
        default="nba_dfs.db",
        help="Path to SQLite database (default: nba_dfs.db)"
    )

    parser.add_argument(
        "--data-dir",
        default=None,
        help="Data directory for separated architecture (optional). If set, db-path and output-dir are relative to this directory."
    )

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

    parser.add_argument(
        "--model-type",
        choices=["xgboost", "random_forest", "linear"],
        default="xgboost",
        help="Model type (default: xgboost)"
    )

    parser.add_argument(
        "--feature-config",
        default="default_features",
        help="Feature configuration name or comma-separated list to combine. "
             "Examples: 'default_features', 'base_features,opponent_features', "
             "'default_features,opponent_features' (default: default_features)"
    )

    parser.add_argument(
        "--model-config",
        default=None,
        help="Path to YAML model configuration file with optimized hyperparameters"
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

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Number of parallel jobs for per-player model training (default: -1 for all cores, use 1 for sequential)"
    )

    parser.add_argument(
        "--rewrite-models",
        action="store_true",
        help="Force retraining of models even if cached versions exist"
    )

    parser.add_argument(
        "--resume-from-run",
        default=None,
        help="Resume from an existing run by providing the timestamp (e.g., 20250205_143022)"
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

    parser.add_argument(
        "--filter-salary-min",
        type=int,
        default=None,
        help="Minimum salary filter (default: None)"
    )

    parser.add_argument(
        "--filter-salary-max",
        type=int,
        default=None,
        help="Maximum salary filter (default: None)"
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

    return parser.parse_args()


def main():
    args = parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("="*80)
    print("NBA DFS WALK-FORWARD BACKTEST")
    print("="*80)
    if args.data_dir:
        print(f"Architecture: Separated")
        print(f"  Data Directory: {args.data_dir}")
        print(f"  Database: {args.db_path} (relative to data-dir)")
        print(f"  Output: {args.output_dir} (relative to data-dir)")
    else:
        print(f"Architecture: Default (data in project directory)")
        print(f"  Database: {args.db_path}")
        print(f"  Output: {args.output_dir}")
    print(f"Test Period: {args.test_start} to {args.test_end}")
    print(f"Model Type: {args.model_type}")
    print(f"Feature Config: {args.feature_config}")
    print(f"Per-Player Models: {args.per_player}")
    print(f"Number of Seasons: {args.num_seasons}")
    print(f"Parallel Jobs: {args.n_jobs} ({'all cores' if args.n_jobs == -1 else 'sequential' if args.n_jobs == 1 else f'{args.n_jobs} workers'})")
    print(f"Rewrite Models: {args.rewrite_models}")
    print(f"Save Models: {not args.no_save_models}")
    print(f"Save Predictions: {not args.no_save_predictions}")
    print(f"Resume From Run: {args.resume_from_run if args.resume_from_run else 'None (fresh start)'}")
    print(f"Salary Tiers: {args.salary_tiers}")
    print("="*80)
    print()

    test_end_dt = datetime.strptime(args.test_end, '%Y%m%d')
    train_end = (test_end_dt - timedelta(days=1)).strftime('%Y%m%d')

    if args.num_seasons == 1:
        train_start = HistoricalDataLoader.get_season_start_date(args.test_start)
    else:
        train_start = HistoricalDataLoader.get_previous_season_start_date(args.test_start)

    print(f"Calculated Training Period: {train_start} to {train_end}\n")

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

    if player_filters:
        print(f"\nTotal filters: {len(player_filters)}\n")

    backtest = WalkForwardBacktest(
        db_path=args.db_path,
        data_dir=args.data_dir,
        train_start=train_start,
        train_end=train_end,
        test_start=args.test_start,
        test_end=args.test_end,
        model_type=args.model_type,
        model_params=model_params,
        feature_config=args.feature_config,
        output_dir=args.output_dir,
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
        player_filters=player_filters if player_filters else None
    )

    results = backtest.run()

    if 'error' in results:
        print(f"\nERROR: {results['error']}")
        sys.exit(1)

    output_path = backtest.run_output_dir

    csv_path = output_path / f"backtest_results_{args.test_start}_to_{args.test_end}.csv"
    results['daily_results'].to_csv(csv_path, index=False)
    print(f"\nDaily results saved to: {csv_path}")

    summary_path = output_path / f"summary_{args.test_start}_to_{args.test_end}.md"
    with open(summary_path, 'w') as f:
        f.write('# Backtest Results Summary\n\n')

        f.write('## Overview\n\n')
        f.write(f"**Date Range:** {results['date_range']}  \n")
        f.write(f"**Number of Slates:** {results['num_slates']}  \n")
        f.write(f"**Total Players Evaluated:** {results['total_players_evaluated']:.0f}  \n")
        f.write(f"**Average Players per Slate:** {results['avg_players_per_slate']:.1f}  \n\n")

        f.write('## Model Performance\n\n')
        f.write('**MAPE (Mean Absolute Percentage Error):** Target <30% for elite players ($8k+), <50% overall. Lower is better.  \n')
        f.write('**Correlation:** Strong correlation >0.7, moderate 0.5-0.7, weak <0.5. Higher is better.  \n')
        f.write('**RMSE/MAE:** Lower values indicate better predictions. Context-dependent on fantasy points scale.  \n\n')
        f.write('| Metric | Value |\n')
        f.write('|--------|-------|\n')
        f.write(f"| Mean MAPE | {results['model_mean_mape']:.2f}% |\n")
        f.write(f"| Median MAPE | {results['model_median_mape']:.2f}% |\n")
        f.write(f"| Std MAPE | {results['model_std_mape']:.2f}% |\n")
        f.write(f"| Mean RMSE | {results['model_mean_rmse']:.2f} |\n")
        f.write(f"| Std RMSE | {results['model_std_rmse']:.2f} |\n")
        f.write(f"| Mean MAE | {results['model_mean_mae']:.2f} |\n")
        f.write(f"| Mean Correlation | {results['model_mean_correlation']:.3f} |\n")
        f.write(f"| Std Correlation | {results['model_std_correlation']:.3f} |\n\n")

        f.write('## Benchmark Performance\n\n')
        f.write('| Metric | Value |\n')
        f.write('|--------|-------|\n')
        f.write(f"| Mean MAPE | {results['benchmark_mean_mape']:.2f}% |\n")
        f.write(f"| Median MAPE | {results['benchmark_median_mape']:.2f}% |\n\n")

        f.write('## Model vs Benchmark\n\n')
        f.write('**Improvement Interpretation:** Positive values indicate model outperforms baseline. ')
        f.write('Target >5% improvement for meaningful practical value. ')
        f.write('Improvement >10% indicates strong model performance.  \n\n')
        improvement = results['mape_improvement']
        status = 'Better' if improvement > 0 else 'Worse'
        f.write(f"**MAPE Improvement:** {improvement:+.2f}% ({status})  \n\n")

        charts_dir = output_path / 'charts'
        if charts_dir.exists() and (charts_dir / 'model_vs_benchmark.png').exists():
            f.write('![Model vs Benchmark](charts/model_vs_benchmark.png)\n\n')

        if 'statistical_test' in results:
            f.write('## Statistical Significance\n\n')
            f.write('**p-value:** <0.05 indicates statistically significant difference. Lower is stronger evidence.  \n')
            f.write('**Cohen\'s d:** Small effect 0.2-0.5, medium 0.5-0.8, large >0.8. Measures practical significance.  \n')
            f.write('**Interpretation:** Both statistical significance (p<0.05) and practical significance (d>0.5) required for confident model superiority.  \n\n')
            test = results['statistical_test']
            f.write('| Test | Value |\n')
            f.write('|------|-------|\n')
            f.write(f"| t-statistic | {test['t_statistic']:.4f} |\n")
            f.write(f"| p-value | {test['p_value']:.6f} |\n")
            f.write(f"| Cohen's d | {test['cohens_d']:.4f} |\n")
            f.write(f"| Effect size | {test['effect_size']} |\n\n")

        if 'tier_comparison' in results:
            f.write('## Performance by Salary Tier\n\n')
            f.write('**Elite Players ($8k+):** Target <30% MAPE. Most critical for DFS optimization.  \n')
            f.write('**Mid-Tier ($5k-$8k):** Target <40% MAPE. Balance of salary efficiency and reliability.  \n')
            f.write('**Budget (<$5k):** Higher MAPE acceptable. Low output creates percentage inflation.  \n\n')
            tier_df = results['tier_comparison']
            f.write('| Tier | Count | Model MAPE | Benchmark MAPE | Improvement | Status |\n')
            f.write('|------|-------|------------|----------------|-------------|--------|\n')
            for _, row in tier_df.iterrows():
                tier = str(row['salary_tier'])
                count = int(row['count'])
                model_mape = row['model_mape']
                bench_mape = row['benchmark_mape']
                imp = row['mape_improvement']
                stat = 'Better' if imp > 0 else 'Worse'
                f.write(f"| {tier} | {count} | {model_mape:.1f}% | {bench_mape:.1f}% | {imp:+.1f}% | {stat} |\n")
            f.write('\n')

            if charts_dir.exists() and (charts_dir / 'salary_tier_performance.png').exists():
                f.write('![Salary Tier Performance](charts/salary_tier_performance.png)\n\n')

        f.write('## Visualizations\n\n')

        if charts_dir.exists():
            chart_files = [
                ('daily_mape.png', 'Daily MAPE Over Time'),
                ('error_distribution.png', 'Error Distribution'),
                ('correlation_scatter.png', 'Prediction vs Actual Correlation'),
                ('metrics_comparison.png', 'Metrics Comparison')
            ]

            for chart_file, title in chart_files:
                chart_path = charts_dir / chart_file
                if chart_path.exists():
                    f.write(f'### {title}\n\n')
                    f.write(f'![{title}](charts/{chart_file})\n\n')

    print(f"Summary saved to: {summary_path}")

    if 'tier_comparison' in results:
        tier_path = output_path / f"tier_comparison_{args.test_start}_to_{args.test_end}.csv"
        results['tier_comparison'].to_csv(tier_path, index=False)
        print(f"Tier comparison saved to: {tier_path}")

    if 'report_path' in results:
        print(f"\nComprehensive report: {results['report_path']}")

    print("\n" + "="*80)
    print("MODEL VS BENCHMARK COMPARISON")
    print("="*80)

    model_mape = results.get('model_mean_mape', 0)
    bench_mape = results.get('benchmark_mean_mape', 0)
    improvement = results.get('mape_improvement', 0)

    print(f"\nModel Performance:")
    print(f"  MAPE:        {model_mape:.2f}%")
    print(f"  RMSE:        {results.get('model_mean_rmse', 0):.2f}")
    print(f"  MAE:         {results.get('model_mean_mae', 0):.2f}")
    print(f"  Correlation: {results.get('model_mean_correlation', 0):.3f}")

    print(f"\nBenchmark Performance:")
    print(f"  MAPE:        {bench_mape:.2f}%")

    status = "✓ MODEL BETTER" if improvement > 0 else "✗ BENCHMARK BETTER"
    print(f"\nImprovement:   {improvement:+.2f}% {status}")

    if 'statistical_test' in results:
        test = results['statistical_test']
        sig = "✓ SIGNIFICANT" if test['p_value'] < 0.05 else "✗ NOT SIGNIFICANT"
        print(f"Significance:  p={test['p_value']:.4f} {sig}")
        print(f"Effect Size:   {test['effect_size']} (d={test['cohens_d']:.4f})")

    if 'tier_comparison' in results:
        print("\n" + "-"*80)
        print("PERFORMANCE BY SALARY TIER")
        print("-"*80)
        tier_df = results['tier_comparison']
        print(f"\n{'Tier':<15} {'Count':>8} {'Model MAPE':>12} {'Bench MAPE':>12} {'Improve':>10} {'Status':>10}")
        print("-"*80)
        for _, row in tier_df.iterrows():
            tier = str(row['salary_tier'])[:14]
            count = int(row['count'])
            model = row['model_mape']
            bench = row['benchmark_mape']
            imp = row['mape_improvement']
            stat = "✓ Better" if imp > 0 else "✗ Worse"
            print(f"{tier:<15} {count:>8} {model:>11.1f}% {bench:>11.1f}% {imp:>9.1f}% {stat:>10}")
        print("-"*80)

    charts_dir = output_path / 'charts'
    if charts_dir.exists():
        chart_count = len(list(charts_dir.glob('*.png')))
        print(f"\nGenerated {chart_count} visualization charts in: {charts_dir}")

    print("\n" + "="*80)
    print("BACKTEST COMPLETE")
    print("="*80)
    print(f"\nAll outputs saved to: {output_path}")


if __name__ == "__main__":
    main()
