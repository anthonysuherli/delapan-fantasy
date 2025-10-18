"""
Script to optimize XGBoost hyperparameters using Bayesian optimization.

This script:
1. Loads historical training data
2. Builds features using the feature pipeline
3. Runs Bayesian optimization to find optimal hyperparameters
4. Saves results to a configuration file
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from joblib import Parallel, delayed
import argparse

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from src.data.storage.sqlite_storage import SQLiteStorage
from src.data.loaders.historical_loader import HistoricalDataLoader
from src.features.pipeline import FeaturePipeline
from src.utils.fantasy_points import calculate_dk_fantasy_points
from src.utils.bayesian_optimizer import optimize_xgboost_cv
from src.utils.feature_config import load_feature_config


def optimize_single_player(player_id, player_data, feature_cols, param_bounds, cv_folds, n_iterations, n_initial_points, early_stopping_rounds):
    """Optimize hyperparameters for a single player."""
    X_player = player_data[feature_cols].fillna(0)
    y_player = player_data['target']

    try:
        best_params, best_score, history = optimize_xgboost_cv(
            X=X_player,
            y=y_player,
            param_bounds=param_bounds,
            cv_folds=cv_folds,
            n_iterations=n_iterations,
            n_initial_points=n_initial_points,
            scoring='neg_mean_absolute_error',
            random_state=42,
            verbose=False,
            early_stopping_rounds=early_stopping_rounds
        )

        config = {
            'max_depth': int(best_params['max_depth']),
            'learning_rate': float(best_params['learning_rate']),
            'n_estimators': int(best_params['n_estimators']),
            'min_child_weight': int(best_params['min_child_weight']),
            'subsample': float(best_params['subsample']),
            'colsample_bytree': float(best_params['colsample_bytree']),
            'gamma': float(best_params['gamma']),
            'reg_alpha': float(best_params['reg_alpha']),
            'reg_lambda': float(best_params['reg_lambda']),
            'objective': 'reg:squarederror',
            'random_state': 42,
            'enable_categorical': True
        }

        summary = {
            'playerID': player_id,
            'num_games': len(X_player),
            'cv_mae': -best_score,
            'max_depth': config['max_depth'],
            'learning_rate': config['learning_rate'],
            'n_estimators': config['n_estimators'],
            'success': True,
            'error': None
        }

        return player_id, config, summary

    except Exception as e:
        return player_id, None, {
            'playerID': player_id,
            'num_games': len(player_data),
            'cv_mae': None,
            'max_depth': None,
            'learning_rate': None,
            'n_estimators': None,
            'success': False,
            'error': str(e)
        }


def main():
    parser = argparse.ArgumentParser(description='Optimize XGBoost hyperparameters')
    parser.add_argument('--per-player', action='store_true', help='Optimize per player instead of global')
    parser.add_argument('--target-date', type=str, default='20250205', help='Target date (YYYYMMDD)')
    parser.add_argument('--num-seasons', type=int, default=1, help='Number of seasons for training')
    parser.add_argument('--feature-config', type=str, default='default_features', help='Feature config name')
    parser.add_argument('--cv-folds', type=int, default=5, help='Number of CV folds')
    parser.add_argument('--iterations', type=int, default=50, help='Number of optimization iterations')
    parser.add_argument('--min-games', type=int, default=30, help='Minimum games for per-player optimization')
    parser.add_argument('--n-jobs', type=int, default=-1, help='Number of parallel jobs (-1 = all cores)')
    parser.add_argument('--early-stopping-rounds', type=int, default=None, help='Stop if no improvement after N rounds')

    args = parser.parse_args()

    DB_PATH = repo_root / 'nba_dfs.db'
    TARGET_DATE = args.target_date
    NUM_SEASONS = args.num_seasons
    FEATURE_CONFIG = args.feature_config
    CV_FOLDS = args.cv_folds
    N_ITERATIONS = args.iterations
    N_INITIAL_POINTS = min(10, N_ITERATIONS // 5)
    MIN_GAMES = args.min_games
    N_JOBS = args.n_jobs
    EARLY_STOPPING_ROUNDS = args.early_stopping_rounds
    RANDOM_STATE = 42

    feature_config = load_feature_config(FEATURE_CONFIG)

    if NUM_SEASONS == 1:
        TRAIN_START = HistoricalDataLoader.get_season_start_date(TARGET_DATE)
    else:
        TRAIN_START = HistoricalDataLoader.get_previous_season_start_date(TARGET_DATE)

    target_dt = datetime.strptime(TARGET_DATE, '%Y%m%d')
    train_end_dt = target_dt - timedelta(days=1)
    TRAIN_END = train_end_dt.strftime('%Y%m%d')

    print("XGBoost Hyperparameter Optimization")
    print("=" * 50)
    print(f"Mode: {'Per-Player' if args.per_player else 'Global'}")
    print(f"Database: {DB_PATH}")
    print(f"Target date: {TARGET_DATE}")
    print(f"Training period: {TRAIN_START} to {TRAIN_END}")
    print(f"  (Season-based: {NUM_SEASONS} season(s))")
    print(f"\nFeature Configuration: {feature_config.name}")
    print(f"  Description: {feature_config.description}")
    print(f"  Version: {feature_config.version}")
    print(f"  Stats: {len(feature_config.stats)} features")
    print(f"  Rolling windows: {feature_config.rolling_windows}")
    print(f"  EWMA span: {feature_config.ewma_span}")
    print(f"\nCV folds: {CV_FOLDS}")
    print(f"Optimization iterations: {N_ITERATIONS}")
    if EARLY_STOPPING_ROUNDS is not None:
        print(f"Early stopping rounds: {EARLY_STOPPING_ROUNDS}")
    if args.per_player:
        print(f"Min games for optimization: {MIN_GAMES}")
        print(f"Parallel jobs: {N_JOBS}")
    print("=" * 50)

    print("\nLoading training data...")
    storage = SQLiteStorage(str(DB_PATH))
    loader = HistoricalDataLoader(storage)
    training_data = loader.load_historical_player_logs(start_date=TRAIN_START, end_date=TRAIN_END)

    if 'plusMinus' in training_data.columns:
        training_data['plusMinus'] = training_data['plusMinus'].apply(lambda x: int(x) if pd.notna(x) else 0)

    print(f"Loaded {len(training_data)} training samples")
    print(f"Unique players: {training_data['playerID'].nunique()}")

    print("\nBuilding features...")
    df = training_data.copy()
    df['gameDate'] = pd.to_datetime(df['gameDate'], format='%Y%m%d', errors='coerce')
    df = df.sort_values(['playerID', 'gameDate'])

    if 'plusMinus' in df.columns:
        df['plusMinus'] = df['plusMinus'].apply(lambda x: int(x) if pd.notna(x) else 0)

    if 'fpts' not in df.columns:
        df['fpts'] = df.apply(calculate_dk_fantasy_points, axis=1)

    df['target'] = df.groupby('playerID')['fpts'].shift(-1)

    print(f"Building features from config: {feature_config.name}")
    pipeline = feature_config.build_pipeline(FeaturePipeline)

    df = pipeline.fit_transform(df)
    df = df.drop(columns=feature_config.stats, errors='ignore')
    df = df.dropna(subset=['target'])

    metadata_cols = [
        'playerID', 'playerName', 'team', 'pos', 'gameDate',
        'fpts', 'target', 'pts', 'reb', 'ast', 'stl', 'blk', 'TOV', 'mins'
    ]
    feature_cols = [
        col for col in df.columns
        if col not in metadata_cols and df[col].dtype in ['int64', 'float64', 'bool']
    ]

    X_train = df[feature_cols].fillna(0)
    y_train = df['target']

    print(f"Training data shape: {X_train.shape}")
    print(f"Features: {len(feature_cols)}")

    param_bounds = {
        'max_depth': (3, 10),
        'learning_rate': (0.01, 0.3),
        'n_estimators': (50, 500 if not args.per_player else 300),
        'min_child_weight': (1, 10),
        'subsample': (0.5, 1.0),
        'colsample_bytree': (0.5, 1.0),
        'gamma': (0.0, 5.0),
        'reg_alpha': (0.0, 1.0),
        'reg_lambda': (0.0, 1.0)
    }

    if args.per_player:
        print("\nRunning per-player Bayesian optimization with parallel processing...")

        player_game_counts = df.groupby('playerID').size()
        qualified_players = player_game_counts[player_game_counts >= MIN_GAMES].index

        print(f"Players with >= {MIN_GAMES} games: {len(qualified_players)}")

        players_to_optimize = []
        player_data_dict = {}

        for player_id in qualified_players:
            player_data = df[df['playerID'] == player_id]
            if len(player_data) >= MIN_GAMES:
                players_to_optimize.append(player_id)
                player_data_dict[player_id] = player_data

        print(f"Optimizing for {len(players_to_optimize)} players")
        print("Starting parallel optimization...")
        print()

        results = Parallel(n_jobs=N_JOBS, verbose=10)(
            delayed(optimize_single_player)(
                player_id,
                player_data_dict[player_id],
                feature_cols,
                param_bounds,
                CV_FOLDS,
                N_ITERATIONS,
                N_INITIAL_POINTS,
                EARLY_STOPPING_ROUNDS
            )
            for player_id in players_to_optimize
        )

        per_player_configs = {}
        optimization_summary = []

        for player_id, config, summary in results:
            if config is not None:
                per_player_configs[player_id] = config
            optimization_summary.append(summary)

        print("\n" + "=" * 50)
        print("Per-Player Optimization Complete")
        print("=" * 50)
        print(f"Successfully optimized: {len(per_player_configs)} players")

        opt_summary_df = pd.DataFrame(optimization_summary)
        successful = opt_summary_df[opt_summary_df['success'] == True]
        failed = opt_summary_df[opt_summary_df['success'] == False]

        if len(failed) > 0:
            print(f"Failed: {len(failed)} players")

        if len(successful) > 0:
            print("\nOptimization Summary:")
            print(f"  Mean CV MAE: {successful['cv_mae'].mean():.2f}")
            print(f"  Median CV MAE: {successful['cv_mae'].median():.2f}")
            print(f"  Std CV MAE: {successful['cv_mae'].std():.2f}")
            print(f"\nHyperparameter Distributions:")
            print(f"  max_depth: mean={successful['max_depth'].mean():.1f}, median={successful['max_depth'].median():.0f}")
            print(f"  learning_rate: mean={successful['learning_rate'].mean():.3f}, median={successful['learning_rate'].median():.3f}")
            print(f"  n_estimators: mean={successful['n_estimators'].mean():.0f}, median={successful['n_estimators'].median():.0f}")

        best_params = None
        best_score = None
        history = opt_summary_df

    else:
        print("\nRunning global Bayesian optimization...")
        print("This may take a while...")

        best_params, best_score, history = optimize_xgboost_cv(
            X=X_train,
            y=y_train,
            param_bounds=param_bounds,
            cv_folds=CV_FOLDS,
            n_iterations=N_ITERATIONS,
            n_initial_points=N_INITIAL_POINTS,
            scoring='neg_mean_absolute_error',
            random_state=RANDOM_STATE,
            verbose=True,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS
        )

        print("\n" + "=" * 50)
        print("Optimization Complete")
        print("=" * 50)
        print(f"\nBest CV Score (MAE): {-best_score:.4f}")
        print("\nBest Parameters:")
        for param, value in sorted(best_params.items()):
            print(f"  {param}: {value}")

        per_player_configs = None

    output_dir = repo_root / 'config' / 'models'
    output_dir.mkdir(parents=True, exist_ok=True)

    import yaml

    if args.per_player:
        config_path = output_dir / 'xgboost_per_player_configs.yaml'
        history_path = output_dir / 'per_player_optimization_history.csv'

        config_data = {
            'model_type': 'xgboost_per_player',
            'per_player_configs': {
                str(player_id): config
                for player_id, config in per_player_configs.items()
            },
            'optimization_metadata': {
                'mode': 'per_player',
                'num_players_optimized': len(per_player_configs),
                'cv_folds': CV_FOLDS,
                'n_iterations': N_ITERATIONS,
                'min_games': MIN_GAMES,
                'training_period': f"{TRAIN_START} to {TRAIN_END}",
                'optimized_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        }

        with open(config_path, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)

        print(f"\nSaved per-player configs to: {config_path}")

        history.to_csv(history_path, index=False)
        print(f"Saved optimization history to: {history_path}")

    else:
        config_path = output_dir / 'xgboost_optimized.yaml'
        history_path = output_dir / 'optimization_history.csv'

        config_data = {
            'model_type': 'xgboost',
            'hyperparameters': {
                'max_depth': int(best_params['max_depth']),
                'learning_rate': float(best_params['learning_rate']),
                'n_estimators': int(best_params['n_estimators']),
                'min_child_weight': int(best_params['min_child_weight']),
                'subsample': float(best_params['subsample']),
                'colsample_bytree': float(best_params['colsample_bytree']),
                'gamma': float(best_params['gamma']),
                'reg_alpha': float(best_params['reg_alpha']),
                'reg_lambda': float(best_params['reg_lambda']),
                'objective': 'reg:squarederror',
                'random_state': RANDOM_STATE,
                'enable_categorical': True
            },
            'optimization_metadata': {
                'mode': 'global',
                'cv_score_mae': float(-best_score),
                'cv_folds': CV_FOLDS,
                'n_iterations': N_ITERATIONS,
                'training_samples': len(X_train),
                'training_period': f"{TRAIN_START} to {TRAIN_END}",
                'optimized_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        }

        with open(config_path, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)

        print(f"\nSaved optimized config to: {config_path}")

        history.to_csv(history_path, index=False)
        print(f"Saved optimization history to: {history_path}")

        print("\nTop 10 iterations by score:")
        print(history.nlargest(10, 'score')[['score', 'max_depth', 'learning_rate', 'n_estimators']])


if __name__ == '__main__':
    main()
