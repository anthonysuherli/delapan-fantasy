#!/usr/bin/env python
"""
Full slate prediction script using per-player models.

Extracts Phase 2 notebook logic into reusable, modular script.
Supports swappable feature sets and model types via registry pattern.

Usage:
    python scripts/predict_slate.py --date 20250205
    python scripts/predict_slate.py --date 20250205 --features default_features --model random_forest
    python scripts/predict_slate.py --date 20250205 --analyze --output predictions.csv
"""

import argparse
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import warnings
from tqdm import tqdm
from datetime import datetime

from src.data.loaders.historical_loader import HistoricalDataLoader
from src.utils.feature_config import load_feature_config
from src.features.pipeline import FeaturePipeline
from src.models.registry import registry as model_registry
from src.utils.config_loader import load_yaml
from src.utils.fantasy_points import calculate_dk_fantasy_points
from src.evaluation.metrics.segmentation import (
    analyze_by_salary,
    analyze_by_position,
    summary_report
)

warnings.filterwarnings('ignore')


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict full slate using per-player models"
    )

    parser.add_argument(
        "--date",
        required=True,
        help="Slate date in YYYYMMDD format (e.g., 20250205)"
    )

    parser.add_argument(
        "--data-dir",
        default="data",
        help="Base data directory (default: data)"
    )

    parser.add_argument(
        "--features",
        default="base_features",
        help="Feature config name from config/features/ (default: base_features)"
    )

    parser.add_argument(
        "--model",
        default="xgboost_default",
        help="Model config name from config/models/ (default: xgboost_default)"
    )

    parser.add_argument(
        "--num-seasons",
        type=int,
        default=2,
        help="Number of seasons of historical data (default: 2)"
    )

    parser.add_argument(
        "--min-games",
        type=int,
        default=10,
        help="Minimum games required in training window (default: 10)"
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Output file path (default: data/outputs/predictions_YYYYMMDD.csv)"
    )

    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Include position/salary analysis in output"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress information"
    )

    parser.add_argument(
        "--use-quantiles",
        action="store_true",
        help="Use quantile regression for variance prediction (floor/median/ceiling)"
    )

    return parser.parse_args()


def load_slate_data(loader, test_date, verbose=False):
    """Load test slate and player logs."""
    if verbose:
        print(f"\nLoading slate data for {test_date}...")

    test_slate = loader.load_slate_data(test_date)
    schedule = test_slate['games']

    if verbose:
        print(f"Games: {len(schedule)}")

    # Get teams playing
    teams_playing = set()
    if 'away' in schedule.columns and 'home' in schedule.columns:
        teams_playing.update(schedule['away'].unique())
        teams_playing.update(schedule['home'].unique())

    # Load player logs
    player_logs_all = loader.load_historical_data(
        start_date=test_date,
        end_date=test_date,
        data_types=['player_logs_extracted']
    )['player_logs_extracted']

    player_logs = player_logs_all[player_logs_all['team'].isin(teams_playing)].copy()

    # Calculate fantasy points
    player_logs = player_logs.drop(columns=['fantasyPoints', 'fantasyPts'], errors='ignore')
    player_logs['fantasyPoints'] = player_logs.apply(calculate_dk_fantasy_points, axis=1)

    # Merge with salaries
    salaries = test_slate['dfs_salaries']
    test_slate_combined = player_logs.merge(
        salaries,
        on='playerID',
        how='inner',
        suffixes=('', '_dfs')
    )

    if verbose:
        print(f"Player logs: {len(player_logs)}")
        print(f"Players with logs and salaries: {len(test_slate_combined)}")

    return test_slate_combined, player_logs


def load_historical_data(loader, test_date, player_ids, num_seasons, verbose=False):
    """Load historical logs for training."""
    if verbose:
        print(f"\nLoading historical data for {len(player_ids)} players...")

    historical_logs = loader.load_historical_player_logs(
        end_date=test_date,
        num_seasons=num_seasons,
        player_ids=player_ids
    )

    # Calculate fantasy points
    historical_logs = historical_logs.drop(columns=['fantasyPoints', 'fantasyPts'], errors='ignore')
    historical_logs['fantasyPoints'] = historical_logs.apply(calculate_dk_fantasy_points, axis=1)

    if verbose:
        print(f"Historical logs shape: {historical_logs.shape}")
        print(f"Date range: {historical_logs['gameDate'].min()} to {historical_logs['gameDate'].max()}")

    # Verify no temporal leakage
    max_training_date = str(historical_logs['gameDate'].max())[:10].replace('-', '')
    assert max_training_date < test_date, f"Temporal leakage: {max_training_date} >= {test_date}"

    if verbose:
        print(f"[OK] No temporal leakage: {max_training_date} < {test_date}")

    return historical_logs


def filter_eligible_players(test_slate_combined, historical_logs, min_games, verbose=False):
    """Filter players with sufficient training data."""
    player_game_counts = historical_logs.groupby('playerID').size()
    eligible_players = player_game_counts[player_game_counts >= min_games].index.tolist()
    test_slate_filtered = test_slate_combined[test_slate_combined['playerID'].isin(eligible_players)].copy()

    if verbose:
        print(f"\nPlayers with {min_games}+ games: {len(eligible_players)}")
        print(f"Players filtered out: {len(test_slate_combined) - len(test_slate_filtered)}")
        print(f"Coverage: {len(test_slate_filtered) / len(test_slate_combined) * 100:.1f}%")

    return test_slate_filtered, eligible_players


def train_and_predict(
    eligible_players,
    test_slate_filtered,
    historical_logs,
    player_logs,
    feature_config,
    model_config,
    test_date,
    use_quantiles=False,
    verbose=False
):
    """Train per-player models and generate predictions."""
    # Build feature pipeline
    pipeline = feature_config.build_pipeline(FeaturePipeline)
    target = 'fantasyPoints'

    if verbose:
        print(f"\nFeature statistics: {feature_config.stats}")
        print(f"Transformers: {len(pipeline.transformers)}")

    # Load model configuration
    model_type = model_config.get('model_type', 'xgboost')
    model_params = model_config.get('hyperparameters', model_config.get('params', {}))

    results = []
    failed_players = []

    iterator = tqdm(eligible_players, desc="Training models") if verbose else eligible_players

    for player_id in iterator:
        try:
            # Get player info
            player_info = test_slate_filtered[test_slate_filtered['playerID'] == player_id].iloc[0]

            # Get historical data
            player_history = historical_logs[historical_logs['playerID'] == player_id].copy()
            player_history = player_history.sort_values('gameDate')

            # Transform training data
            training_features = pipeline.fit_transform(player_history)

            # Get feature columns
            feature_cols = [col for col in training_features.columns if col not in player_history.columns]
            feature_cols_no_target = [col for col in feature_cols if col not in [target, 'target']]

            # Prepare training data
            training_clean = training_features.dropna(subset=feature_cols_no_target).copy()
            training_clean[target] = training_clean.groupby('playerID')[target].shift(-1)
            training_clean = training_clean.dropna()

            if len(training_clean) == 0:
                failed_players.append({
                    'playerID': player_id,
                    'name': player_info['longName'],
                    'reason': 'No valid training samples after dropna'
                })
                continue

            X_train = training_clean[feature_cols_no_target]
            y_train = training_clean[target]

            # Train model using registry
            model = model_registry.create(model_type, model_params)
            model.train(X_train, y_train)

            # Prepare test data
            test_game = player_logs[player_logs['playerID'] == player_id].copy()
            test_game['gameDate'] = pd.to_datetime(test_date, format='%Y%m%d')

            combined = pd.concat([player_history, test_game], ignore_index=True)
            combined = combined.sort_values('gameDate')

            # Transform test data
            test_features = pipeline.transform(combined)
            test_date_dt = pd.to_datetime(test_date, format='%Y%m%d')
            test_row = test_features[test_features['gameDate'] == test_date_dt]

            X_test = test_row[feature_cols_no_target]
            y_test = pd.to_numeric(test_row[target], errors='coerce')

            # Generate prediction
            if use_quantiles:
                # Get quantile predictions with variance
                quantile_preds = model.predict_with_variance(X_test)
                y_pred = quantile_preds['median'].values[0]
                floor_pred = quantile_preds['floor'].values[0]
                ceiling_pred = quantile_preds['ceiling'].values[0]
                variance = quantile_preds['variance'].values[0]
                iqr = quantile_preds['iqr'].values[0]
                cv = quantile_preds['cv'].values[0]
            else:
                # Standard point prediction
                y_pred = model.predict(X_test)[0]
                floor_pred = None
                ceiling_pred = None
                variance = None
                iqr = None
                cv = None

            # Store results - use pos_dfs from salaries
            primary_pos = player_info.get('pos_dfs', 'Unknown')

            result = {
                'playerID': player_id,
                'name': player_info['longName'],
                'team': player_info['team'],
                'salary': player_info['salary'],
                'primary_position': primary_pos,
                'allValidPositions': player_info.get('allValidPositions', []),
                'actual': y_test.values[0],
                'predicted': y_pred,
                'error': y_pred - y_test.values[0],
                'abs_error': abs(y_pred - y_test.values[0]),
                'pct_error': abs(y_pred - y_test.values[0]) / y_test.values[0] * 100 if y_test.values[0] > 0 else 0,
                'training_samples': len(training_clean),
                'minutes': player_info['mins']
            }

            # Add quantile predictions if available
            if use_quantiles:
                result['floor'] = floor_pred
                result['ceiling'] = ceiling_pred
                result['variance'] = variance
                result['iqr'] = iqr
                result['cv'] = cv

            results.append(result)

        except Exception as e:
            failed_players.append({
                'playerID': player_id,
                'name': player_info.get('longName', 'Unknown'),
                'reason': str(e)
            })

    results_df = pd.DataFrame(results)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Successful predictions: {len(results_df)}")
        print(f"Failed: {len(failed_players)}")
        print(f"Success rate: {len(results_df) / len(eligible_players) * 100:.1f}%")

        if len(failed_players) > 0:
            print(f"\nFailed players: {len(failed_players)}")
            for fp in failed_players[:5]:
                print(f"  - {fp['name']}: {fp['reason']}")

    return results_df


def main():
    args = parse_args()

    # Initialize data loader
    loader = HistoricalDataLoader(data_dir=args.data_dir)

    # Load feature configuration
    feature_config = load_feature_config(args.features)

    # Load model configuration
    # If using quantiles, override to quantile_regression config
    if args.use_quantiles:
        model_config_path = project_root / 'config' / 'models' / 'quantile_regression.yaml'
    else:
        model_config_path = project_root / 'config' / 'models' / f'{args.model}.yaml'

    model_config = load_yaml(str(model_config_path))

    if args.verbose:
        print(f"Configuration:")
        print(f"  Date: {args.date}")
        print(f"  Features: {args.features}")
        print(f"  Model: {args.model} ({model_config.get('model_type', 'xgboost')})")
        print(f"  Use quantiles: {args.use_quantiles}")
        print(f"  Num seasons: {args.num_seasons}")
        print(f"  Min games: {args.min_games}")

    # Load slate data
    test_slate_combined, player_logs = load_slate_data(
        loader, args.date, args.verbose
    )

    # Load historical data
    player_ids = test_slate_combined['playerID'].unique().tolist()
    historical_logs = load_historical_data(
        loader, args.date, player_ids, args.num_seasons, args.verbose
    )

    # Filter eligible players
    test_slate_filtered, eligible_players = filter_eligible_players(
        test_slate_combined, historical_logs, args.min_games, args.verbose
    )

    # Train models and predict
    results_df = train_and_predict(
        eligible_players,
        test_slate_filtered,
        historical_logs,
        player_logs,
        feature_config,
        model_config,
        args.date,
        use_quantiles=args.use_quantiles,
        verbose=args.verbose
    )

    # Add salary tiers for analysis
    results_df['salary_tier'] = pd.cut(
        results_df['salary'],
        bins=[0, 5000, 7000, 9000, 15000],
        labels=['$3-5k', '$5-7k', '$7-9k', '$9k+']
    )

    # Overall metrics
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from scipy.stats import pearsonr

    mae = mean_absolute_error(results_df['actual'], results_df['predicted'])
    rmse = np.sqrt(mean_squared_error(results_df['actual'], results_df['predicted']))
    mape = results_df['pct_error'].mean()
    correlation, _ = pearsonr(results_df['actual'], results_df['predicted'])

    print(f"\n{'='*60}")
    print(f"OVERALL PERFORMANCE METRICS")
    print(f"{'='*60}")
    print(f"Players predicted: {len(results_df)}")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"Correlation: {correlation:.3f}")

    # Segmented analysis if requested
    if args.analyze:
        print(summary_report(results_df))

    # Save results
    if args.output is None:
        output_dir = project_root / 'data' / 'outputs'
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f'predictions_{args.date}.csv'
    else:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
