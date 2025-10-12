#!/usr/bin/env python
"""
Evaluate model predictions against season average benchmark.

This script provides comprehensive evaluation comparing model predictions
to a simple season average baseline, helping determine if the model
provides value over naive predictions.

Usage:
    python scripts/evaluate_with_benchmark.py --date 20250115 --model-type per-player
    python scripts/evaluate_with_benchmark.py --date 20250115 --model-type slate --train-days 90
    python scripts/evaluate_with_benchmark.py --date 20250115 --export-results
"""

import argparse
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional
import json

# Data
from src.data.loaders.historical_loader import HistoricalDataLoader
from src.data.storage.parquet_storage import ParquetStorage

# Features
from src.features.pipeline import FeaturePipeline
from src.utils.feature_config import load_feature_config

# Models
from src.models.xgboost_model import XGBoostModel

# Evaluation
from src.evaluation.benchmarks.season_average import SeasonAverageBenchmark


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate predictions against season average benchmark"
    )

    parser.add_argument(
        "--date",
        required=True,
        help="Prediction date (YYYYMMDD format)"
    )

    parser.add_argument(
        "--model-type",
        choices=["per-player", "slate"],
        default="per-player",
        help="Model type to evaluate (default: per-player)"
    )

    parser.add_argument(
        "--lookback-days",
        type=int,
        default=365,
        help="Days of historical data for per-player models (default: 365)"
    )

    parser.add_argument(
        "--train-days",
        type=int,
        default=90,
        help="Days of training data for slate model (default: 90)"
    )

    parser.add_argument(
        "--min-games",
        type=int,
        default=10,
        help="Minimum games for per-player model (default: 10)"
    )

    parser.add_argument(
        "--feature-config",
        default="default_features",
        help="Feature configuration name (default: default_features)"
    )

    parser.add_argument(
        "--export-results",
        action="store_true",
        help="Export detailed results to CSV"
    )

    parser.add_argument(
        "--output-dir",
        default="outputs/benchmark_comparison",
        help="Output directory for results (default: outputs/benchmark_comparison)"
    )

    return parser.parse_args()


def train_per_player_models(
    historical_features: pd.DataFrame,
    slate_players: list,
    feature_names: list,
    min_games: int,
    config: dict
) -> Dict:
    """Train individual models for each player."""
    models = {}
    training_info = {}

    for player_id in slate_players:
        player_data = historical_features[
            historical_features['playerID'] == player_id
        ].copy()

        if len(player_data) < min_games:
            continue

        player_data = player_data.sort_values('gameDate')

        X = player_data[feature_names]
        y = player_data['fpts']

        model = XGBoostModel(config)
        model.train(X, y)

        models[player_id] = model
        training_info[player_id] = {
            'n_games': len(player_data),
            'mean_fpts': y.mean(),
            'std_fpts': y.std()
        }

    return models, training_info


def train_slate_model(
    train_features: pd.DataFrame,
    feature_names: list,
    config: dict
) -> XGBoostModel:
    """Train a single slate-level model."""
    X = train_features[feature_names]
    y = train_features['fpts']

    # Remove NaN
    mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[mask]
    y = y[mask]

    model = XGBoostModel(config)
    model.train(X, y)

    return model


def generate_predictions(
    models: Dict,
    historical_features: pd.DataFrame,
    slate_data: pd.DataFrame,
    feature_names: list,
    model_type: str = "per-player",
    slate_model: Optional[XGBoostModel] = None
) -> pd.DataFrame:
    """Generate predictions using trained models."""
    predictions = []

    if model_type == "per-player":
        for _, player in slate_data.iterrows():
            player_id = player['playerID']

            if player_id in models:
                player_history = historical_features[
                    historical_features['playerID'] == player_id
                ]

                if not player_history.empty:
                    latest_features = player_history.sort_values('gameDate').iloc[-1][feature_names]
                    pred = models[player_id].predict(latest_features.values.reshape(1, -1))[0]
                else:
                    pred = 0
            else:
                pred = 0

            predictions.append({
                'playerID': player_id,
                'playerName': player.get('playerName', 'Unknown'),
                'salary': player.get('salary', 0),
                'position': player.get('position', 'Unknown'),
                'model_pred': pred,
                'actual': player.get('fpts', 0)
            })

    else:  # slate model
        # Get latest features for each player
        latest_features = historical_features.sort_values('gameDate').groupby('playerID').last()

        for _, player in slate_data.iterrows():
            player_id = player['playerID']

            if player_id in latest_features.index:
                features = latest_features.loc[player_id, feature_names].values.reshape(1, -1)
                pred = slate_model.predict(features)[0]
            else:
                pred = 0

            predictions.append({
                'playerID': player_id,
                'playerName': player.get('playerName', 'Unknown'),
                'salary': player.get('salary', 0),
                'position': player.get('position', 'Unknown'),
                'model_pred': pred,
                'actual': player.get('fpts', 0)
            })

    return pd.DataFrame(predictions)


def main():
    """Main evaluation pipeline."""
    args = parse_args()

    print("=" * 80)
    print("MODEL EVALUATION WITH BENCHMARK COMPARISON")
    print("=" * 80)
    print(f"Date: {args.date}")
    print(f"Model Type: {args.model_type}")
    print(f"Feature Config: {args.feature_config}")
    print()

    # Initialize storage and loader
    storage = ParquetStorage()
    loader = HistoricalDataLoader(storage)

    # Load slate data
    print("Loading slate data...")
    slate_data = loader.load_slate_data(args.date)
    print(f"  Loaded {len(slate_data)} players")

    # Load historical data
    if args.model_type == "per-player":
        print(f"\nLoading {args.lookback_days} days of historical data...")
        historical_data = loader.load_historical_player_logs(
            args.date,
            lookback_days=args.lookback_days
        )
    else:  # slate model
        end_date = datetime.strptime(args.date, "%Y%m%d") - timedelta(days=1)
        start_date = end_date - timedelta(days=args.train_days)
        print(f"\nLoading training data from {start_date:%Y%m%d} to {end_date:%Y%m%d}...")
        historical_data = loader.load_historical_data(
            start_date.strftime("%Y%m%d"),
            end_date.strftime("%Y%m%d")
        )

    print(f"  Loaded {len(historical_data)} records")

    # Initialize benchmark
    print("\nInitializing season average benchmark...")
    benchmark = SeasonAverageBenchmark(min_games=5)
    benchmark.fit(historical_data)
    print(f"  Benchmark fitted for {len(benchmark.player_averages)} players")

    # Generate benchmark predictions
    slate_data['benchmark_pred'] = slate_data['playerID'].map(
        benchmark.player_averages
    ).fillna(0)

    # Build features
    print("\nBuilding features...")
    feature_config = load_feature_config(args.feature_config)
    pipeline = feature_config.build_pipeline(FeaturePipeline)

    historical_features = pipeline.fit_transform(historical_data)

    feature_names = [
        col for col in historical_features.columns
        if col not in ['playerID', 'gameDate', 'fpts', 'playerName']
    ]
    print(f"  Generated {len(feature_names)} features")

    # Model configuration
    model_config = {
        'max_depth': 6 if args.model_type == "slate" else 3,
        'learning_rate': 0.05,
        'n_estimators': 200 if args.model_type == "slate" else 100,
        'min_child_weight': 5,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'objective': 'reg:squarederror',
        'random_state': 42
    }

    # Train models
    if args.model_type == "per-player":
        print(f"\nTraining per-player models (min {args.min_games} games)...")
        models, training_info = train_per_player_models(
            historical_features,
            slate_data['playerID'].unique(),
            feature_names,
            args.min_games,
            model_config
        )
        print(f"  Trained {len(models)} player models")
        slate_model = None
    else:
        print("\nTraining slate-level model...")
        slate_model = train_slate_model(
            historical_features,
            feature_names,
            model_config
        )
        models = {}
        print("  Slate model trained")

    # Generate predictions
    print("\nGenerating model predictions...")
    results = generate_predictions(
        models,
        historical_features,
        slate_data,
        feature_names,
        model_type=args.model_type,
        slate_model=slate_model
    )

    # Add benchmark predictions
    results['benchmark_pred'] = results['playerID'].map(
        benchmark.player_averages
    ).fillna(0)

    # Filter to valid comparisons
    has_both = (results['model_pred'] > 0) & (results['benchmark_pred'] > 0)
    results_valid = results[has_both].copy()

    print(f"  Valid comparisons: {len(results_valid)}/{len(results)} players")

    # Compare model vs benchmark
    print("\n" + "=" * 80)
    comparison = benchmark.compare_with_model(
        actual=results_valid['actual'],
        model_pred=results_valid['model_pred'],
        benchmark_pred=results_valid['benchmark_pred']
    )
    print(comparison['summary'])

    # Salary tier comparison
    print("\n" + "=" * 80)
    print("PERFORMANCE BY SALARY TIER")
    print("=" * 80)

    salary_tiers = [0, 4000, 6000, 8000, 15000]
    tier_comparison = benchmark.compare_by_salary_tier(results_valid, salary_tiers)

    for _, row in tier_comparison.iterrows():
        improvement = row['mape_improvement']
        status = "BETTER" if improvement > 0 else "WORSE"

        print(f"\n{row['salary_tier']}:")
        print(f"  Players: {row['n_players']}")
        print(f"  Model MAPE: {row['model_mape']:.1f}%")
        print(f"  Benchmark MAPE: {row['benchmark_mape']:.1f}%")
        print(f"  Improvement: {improvement:+.1f}% ({status})")
        print(f"  Model RMSE: {row['model_rmse']:.2f}")
        print(f"  Benchmark RMSE: {row['benchmark_rmse']:.2f}")

    # Export results if requested
    if args.export_results:
        os.makedirs(args.output_dir, exist_ok=True)

        # Detailed results
        output_file = os.path.join(
            args.output_dir,
            f"benchmark_comparison_{args.model_type}_{args.date}.csv"
        )
        results_valid.to_csv(output_file, index=False)
        print(f"\nDetailed results saved to {output_file}")

        # Tier comparison
        tier_file = os.path.join(
            args.output_dir,
            f"tier_comparison_{args.model_type}_{args.date}.csv"
        )
        tier_comparison.to_csv(tier_file, index=False)
        print(f"Tier comparison saved to {tier_file}")

        # Summary metrics
        summary = {
            'date': args.date,
            'model_type': args.model_type,
            'model_mape': comparison['model']['mape'],
            'benchmark_mape': comparison['benchmark']['mape'],
            'mape_improvement': comparison['improvement']['mape_improvement'],
            'model_rmse': comparison['model']['rmse'],
            'benchmark_rmse': comparison['benchmark']['rmse'],
            'rmse_improvement': comparison['improvement']['rmse_improvement'],
            'model_correlation': comparison['model']['correlation'],
            'benchmark_correlation': comparison['benchmark']['correlation'],
            'n_players_compared': len(results_valid),
            'n_models_trained': len(models) if args.model_type == "per-player" else 1
        }

        summary_file = os.path.join(
            args.output_dir,
            f"summary_{args.model_type}_{args.date}.json"
        )
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Summary saved to {summary_file}")

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()