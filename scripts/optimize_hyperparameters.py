#!/usr/bin/env python
"""
Bayesian hyperparameter optimization for XGBoost models using Optuna.

This script runs expensive hyperparameter search once and saves optimal
parameters to a YAML configuration file for use in backtesting.

Usage:
    python scripts/optimize_hyperparameters.py --output config/models/optimized_xgboost.yaml
    python scripts/optimize_hyperparameters.py --trials 100 --timeout 3600 --sample-size 10000
    python scripts/optimize_hyperparameters.py --db-path nba_dfs.db --data-dir /path/to/data
"""

import argparse
import sys
import os
import logging
import yaml
from pathlib import Path
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_percentage_error
import xgboost as xgb

from src.data.storage.sqlite_storage import SQLiteStorage
from src.data.loaders.historical_loader import HistoricalDataLoader
from src.utils.feature_config import load_feature_config
from src.features.pipeline import FeaturePipeline
from src.utils.fantasy_points import calculate_dk_fantasy_points


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Bayesian hyperparameter optimization for XGBoost models"
    )

    parser.add_argument(
        "--db-path",
        default="nba_dfs.db",
        help="Path to SQLite database (default: nba_dfs.db)"
    )

    parser.add_argument(
        "--data-dir",
        default=None,
        help="Data directory for separated architecture (optional)"
    )

    parser.add_argument(
        "--train-start",
        default="20241001",
        help="Training start date (YYYYMMDD format, default: 20241001)"
    )

    parser.add_argument(
        "--train-end",
        default="20250204",
        help="Training end date (YYYYMMDD format, default: 20250204)"
    )

    parser.add_argument(
        "--feature-config",
        default="default_features",
        help="Feature configuration name (default: default_features)"
    )

    parser.add_argument(
        "--trials",
        type=int,
        default=50,
        help="Number of optimization trials (default: 50)"
    )

    parser.add_argument(
        "--cv-folds",
        type=int,
        default=3,
        help="Cross-validation folds (default: 3)"
    )

    parser.add_argument(
        "--sample-size",
        type=int,
        default=5000,
        help="Training sample size for optimization (default: 5000)"
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=1800,
        help="Maximum optimization time in seconds (default: 1800)"
    )

    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=10,
        help="Early stopping patience in trials (default: 10)"
    )

    parser.add_argument(
        "--output",
        default="config/models/optimized_xgboost.yaml",
        help="Output YAML file path (default: config/models/optimized_xgboost.yaml)"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    return parser.parse_args()


class EarlyStoppingCallback:
    def __init__(self, patience=10, min_delta=0.01):
        self.patience = patience
        self.min_delta = min_delta
        self.best_value = float('inf')
        self.trials_without_improvement = 0

    def __call__(self, study, trial):
        current_value = trial.value

        if current_value < self.best_value - self.min_delta:
            self.best_value = current_value
            self.trials_without_improvement = 0
        else:
            self.trials_without_improvement += 1

        if self.trials_without_improvement >= self.patience:
            print(f"\nEarly stopping triggered after {trial.number + 1} trials")
            print(f"No improvement for {self.patience} consecutive trials")
            print(f"Best MAPE: {self.best_value:.2f}%")
            study.stop()


def objective(trial, X_sample, y_sample, cv_folds):
    """
    Optuna objective function for XGBoost hyperparameter optimization.

    Args:
        trial: Optuna trial object
        X_sample: Feature matrix
        y_sample: Target vector
        cv_folds: Number of CV folds

    Returns:
        float: Mean MAPE across CV folds
    """
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0.0, 5.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
        'objective': 'reg:squarederror',
        'random_state': 42,
        'n_jobs': -1
    }

    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    mape_scores = []

    for train_idx, val_idx in kf.split(X_sample):
        X_train_fold = X_sample.iloc[train_idx]
        y_train_fold = y_sample.iloc[train_idx]
        X_val_fold = X_sample.iloc[val_idx]
        y_val_fold = y_sample.iloc[val_idx]

        model = xgb.XGBRegressor(**params)
        model.fit(X_train_fold, y_train_fold, verbose=False)

        preds = model.predict(X_val_fold)
        preds = np.maximum(preds, 0)

        mask = y_val_fold > 0
        if mask.sum() > 0:
            mape = mean_absolute_percentage_error(y_val_fold[mask], preds[mask]) * 100
            mape_scores.append(mape)

    return np.mean(mape_scores) if mape_scores else float('inf')


def main():
    args = parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    logging.getLogger('optuna').setLevel(logging.WARNING)

    print("="*80)
    print("BAYESIAN HYPERPARAMETER OPTIMIZATION")
    print("="*80)
    print(f"Training period: {args.train_start} to {args.train_end}")
    print(f"Feature config: {args.feature_config}")
    print(f"Optimization trials: {args.trials}")
    print(f"CV folds: {args.cv_folds}")
    print(f"Sample size: {args.sample_size}")
    print(f"Timeout: {args.timeout}s ({args.timeout/60:.1f} min)")
    print(f"Early stopping patience: {args.early_stopping_patience}")
    print(f"Output: {args.output}")
    print("="*80)
    print()

    if args.data_dir:
        data_path = Path(args.data_dir)
        db_path_obj = Path(args.db_path)
        if not db_path_obj.is_absolute():
            db_path = str(data_path / args.db_path)
        else:
            db_path = args.db_path
    else:
        db_path = args.db_path

    storage = SQLiteStorage(db_path)
    loader = HistoricalDataLoader(storage)

    print("Loading training data...")
    training_data = loader.load_historical_player_logs(
        start_date=args.train_start,
        end_date=args.train_end,
        num_seasons=1
    )
    print(f"  Loaded {len(training_data)} records")

    print("Building features...")
    feature_config = load_feature_config(args.feature_config)
    pipeline = feature_config.build_pipeline(FeaturePipeline)

    df = training_data.copy()
    df['gameDate'] = pd.to_datetime(df['gameDate'], format='%Y%m%d', errors='coerce')
    df = df.sort_values(['playerID', 'gameDate'])

    if 'fpts' not in df.columns:
        df['fpts'] = df.apply(calculate_dk_fantasy_points, axis=1)

    df['target'] = df.groupby('playerID')['fpts'].shift(-1)
    train_features = pipeline.fit_transform(df)
    train_features = train_features.dropna(subset=['target'])

    metadata_cols = [
        'playerID', 'playerName', 'longName', 'team', 'teamAbv', 'teamID',
        'pos', 'gameDate', 'gameID', 'fpts', 'fantasyPoints', 'fantasyPts',
        'target', 'pts', 'reb', 'ast', 'stl', 'blk', 'TOV', 'mins',
        'tech', 'created_at', 'updated_at'
    ]
    feature_cols = [col for col in train_features.columns if col not in metadata_cols]

    X_full = train_features[feature_cols].fillna(0)
    y_full = train_features['target']

    if len(X_full) > args.sample_size:
        print(f"Sampling {args.sample_size} records from {len(X_full)} for optimization...")
        sample_idx = np.random.choice(len(X_full), args.sample_size, replace=False)
        X_sample = X_full.iloc[sample_idx]
        y_sample = y_full.iloc[sample_idx]
    else:
        X_sample = X_full
        y_sample = y_full

    print(f"Optimization dataset: {len(X_sample)} samples, {len(feature_cols)} features")
    print()

    early_stopping = EarlyStoppingCallback(patience=args.early_stopping_patience)

    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=42)
    )

    print(f"Running optimization with TPE sampler...")
    print("="*80)

    study.optimize(
        lambda trial: objective(trial, X_sample, y_sample, args.cv_folds),
        n_trials=args.trials,
        timeout=args.timeout,
        callbacks=[early_stopping],
        show_progress_bar=True
    )

    print()
    print("="*80)
    print("OPTIMIZATION COMPLETE")
    print("="*80)
    print(f"Best trial: #{study.best_trial.number}")
    print(f"Best MAPE: {study.best_value:.2f}%")
    print(f"Total trials completed: {len(study.trials)}")
    print()
    print("Best hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print()

    optimized_config = {
        'model_type': 'xgboost',
        'optimization_metadata': {
            'optimized_at': datetime.now().isoformat(),
            'train_start': args.train_start,
            'train_end': args.train_end,
            'feature_config': args.feature_config,
            'optimization_trials': len(study.trials),
            'best_trial': study.best_trial.number,
            'best_mape': float(study.best_value),
            'cv_folds': args.cv_folds,
            'sample_size': len(X_sample)
        },
        'hyperparameters': {
            **study.best_params,
            'objective': 'reg:squarederror',
            'random_state': 42
        }
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        yaml.dump(optimized_config, f, default_flow_style=False, sort_keys=False)

    print(f"Saved optimized configuration to: {output_path}")
    print()

    importance = optuna.importance.get_param_importances(study)
    print("Parameter importance:")
    for param, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True):
        print(f"  {param}: {imp:.4f}")

    print()
    print("="*80)
    print("OPTIMIZATION COMPLETE")
    print("="*80)
    print(f"\nUse optimized parameters in backtest:")
    print(f"  python scripts/run_backtest.py --model-config {args.output} ...")


if __name__ == "__main__":
    main()
