"""
Per-slate model training strategy for walk-forward backtesting.

This module implements slate-level model training where a single
model is trained on all historical player data for the slate.
"""

import pandas as pd
import numpy as np
import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from tqdm import tqdm

from src.models.xgboost_model import XGBoostModel
from src.models.random_forest_model import RandomForestModel
from src.utils.fantasy_points import calculate_dk_fantasy_points
from src.config.paths import (
    PER_SLATE_MODEL_DIR,
    PER_SLATE_TRAINING_INPUTS_DIR
)
from ..trainer_base import BacktestTrainer

logger = logging.getLogger(__name__)


class PerSlateTrainer(BacktestTrainer):
    """
    Per-slate model training strategy.

    A single model is trained on all available historical data
    for all players in the slate.
    """

    def __init__(self, backtest):
        """
        Initialize per-slate trainer.

        Args:
            backtest: Parent WalkForwardBacktest instance
        """
        super().__init__(backtest)
        self.current_model = None
        self.last_training_date = None

    def train_models(
        self,
        training_data: pd.DataFrame,
        slate_data: Dict[str, Any],
        test_date: str
    ) -> Dict[str, Any]:
        """
        Train slate-level model.

        Args:
            training_data: Historical training data
            slate_data: Current slate data
            test_date: Test date

        Returns:
            Dictionary with single slate model
        """
        should_recalibrate = self.backtest._should_recalibrate(test_date)

        if not should_recalibrate and self.current_model is not None:
            logger.info(f"Reusing cached slate model from {self.last_training_date}")
            return {'slate_model': self.current_model}

        logger.info("Training slate-level model")
        model_start_time = time.perf_counter()

        # Prepare model directory
        models_dir = Path(PER_SLATE_MODEL_DIR)
        models_dir.mkdir(parents=True, exist_ok=True)

        # Build training features
        injuries_data = slate_data.get('injuries', pd.DataFrame())

        if self.backtest.enable_feature_caching:
            X_train, y_train = self.backtest._build_training_features_cached(training_data, injuries_data)
        else:
            X_train, y_train = self.backtest._build_training_features(training_data, injuries_data)

        if X_train.empty or y_train.empty:
            logger.warning("No training features generated")
            return {}

        logger.info(f"Training {self.backtest.model_type} on {len(X_train)} samples with {X_train.shape[1]} features")

        # Train model
        model = self._train_slate_model(X_train, y_train, test_date, models_dir)

        self.current_model = model
        self.last_training_date = test_date

        model_elapsed = time.perf_counter() - model_start_time
        logger.info(f"Slate model trained in {self.backtest._format_time(model_elapsed)}")

        # Save model if configured
        if self.backtest.save_models and hasattr(model, 'save'):
            model_file = models_dir / f"slate_model_{test_date}.pkl"
            self._save_slate_model(model, model_file, test_date, len(X_train))
            logger.info(f"Saved slate model to {model_file}")

        return {'slate_model': model}

    def _train_slate_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        test_date: str,
        models_dir: Path
    ) -> Any:
        """
        Train a slate-level model.

        Args:
            X_train: Training features
            y_train: Training targets
            test_date: Test date
            models_dir: Directory to save model inputs

        Returns:
            Trained model
        """
        logger.debug(f"Training {self.backtest.model_type} model on {len(X_train)} samples")

        # Save training inputs if configured
        input_save_path = None
        if self.backtest.save_models:
            input_file = self.backtest.run_inputs_dir / f"slate_{test_date}_inputs.parquet"
            input_save_path = str(input_file)

        if self.backtest.model_type == 'xgboost':
            try:
                model = XGBoostModel(self.backtest.model_params)
                model.train(X_train, y_train, save_inputs=True, input_save_path=input_save_path)
                return model
            except ImportError:
                logger.warning("XGBoost not available, falling back to Random Forest")
                self.backtest.model_type = 'random_forest'

        if self.backtest.model_type == 'random_forest':
            # Filter out XGBoost-specific parameters
            rf_params = {
                k: v for k, v in self.backtest.model_params.items()
                if k not in [
                    'learning_rate', 'objective', 'colsample_bytree',
                    'subsample', 'min_child_weight', 'device', 'tree_method'
                ]
            }
            model = RandomForestModel(rf_params)
            model.train(X_train, y_train, save_inputs=True, input_save_path=input_save_path)
            return model

        if self.backtest.model_type == 'linear':
            from sklearn.linear_model import Ridge
            model = Ridge(alpha=1.0)
            model.fit(X_train, y_train)
            return model

        raise ValueError(f"Unsupported model type: {self.backtest.model_type}")

    def generate_projections(
        self,
        slate_data: Dict[str, Any],
        training_data: pd.DataFrame,
        test_date: str
    ) -> pd.DataFrame:
        """
        Generate slate-level projections.

        Args:
            slate_data: Current slate data
            training_data: Historical training data
            test_date: Test date

        Returns:
            DataFrame with projections
        """
        logger.info("Generating slate-level projections")

        if self.current_model is None:
            logger.warning("No slate model available")
            return pd.DataFrame()

        # Build slate features
        slate_features = self.backtest._build_slate_features(slate_data, training_data)

        if slate_features.empty:
            logger.warning("No slate features generated")
            return pd.DataFrame()

        # Apply player filters if configured
        if self.backtest.player_filters:
            logger.info(f"Applying {len(self.backtest.player_filters)} player filters...")
            initial_count = len(slate_features)

            for pf in self.backtest.player_filters:
                slate_features = pf.apply(slate_features)
                logger.info(f"  Applied {pf}: {len(slate_features)} players remaining")

            logger.info(f"Filter results: {len(slate_features)}/{initial_count} players passed all filters")
            slate_features = slate_features.reset_index(drop=True)

        # Generate predictions
        projections_df = self._generate_slate_projections(self.current_model, slate_features)

        logger.info(f"Generated projections for {len(projections_df)} players with columns: {list(projections_df.columns)}")
        return projections_df

    def _generate_slate_projections(
        self,
        model: Any,
        slate_features: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Generate projections using slate model.

        Args:
            model: Trained model
            slate_features: Slate features

        Returns:
            DataFrame with projections
        """
        logger.debug(f"Generating projections for {len(slate_features)} players")

        # Base metadata columns (only those that exist)
        potential_metadata = ['playerID', 'playerName', 'longName', 'team', 'pos', 'salary']
        metadata_cols = [col for col in potential_metadata if col in slate_features.columns]

        # Add injury-related columns if they exist
        injury_cols = [
            'status', 'is_out', 'is_doubtful', 'is_questionable', 'is_injured',
            'injury_status', 'injury_designation', 'injury_description', 'gameInfo'
        ]
        for col in injury_cols:
            if col in slate_features.columns and col not in metadata_cols:
                metadata_cols.append(col)

        # Select only numeric columns for model input (avoid dtype issues with XGBoost)
        numeric_cols = slate_features.select_dtypes(include=['int64', 'float64']).columns.tolist()

        # Exclude metadata and target columns
        exclude_cols = metadata_cols + ['target']
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]

        # Prepare features for prediction
        X = slate_features[feature_cols].fillna(0).infer_objects(copy=False)

        # Generate predictions
        if hasattr(model, 'predict'):
            if hasattr(model, 'is_trained') and not model.is_trained:
                raise ValueError("Model must be trained before prediction")
            predictions = model.predict(X)
        else:
            # Sklearn-style model
            predictions = model.predict(X)

        # Build projections dataframe
        projections = slate_features[metadata_cols].copy()
        projections['projected_fpts'] = predictions

        # Add value metric if salary is available
        if 'salary' in projections.columns:
            projections['salary'] = pd.to_numeric(projections['salary'], errors='coerce')
            projections['value'] = projections['projected_fpts'] / (projections['salary'] / 1000)
        else:
            projections['value'] = 0.0

        logger.debug(
            f"Projections: min={predictions.min():.2f}, "
            f"max={predictions.max():.2f}, "
            f"mean={predictions.mean():.2f}"
        )

        return projections

    def _save_slate_model(
        self,
        model: Any,
        model_file: Path,
        test_date: str,
        num_samples: int
    ):
        """
        Save slate model with metadata.

        Args:
            model: Trained model
            model_file: Path to save model
            test_date: Test date
            num_samples: Number of training samples
        """
        try:
            # Save with metadata
            metadata = {
                'test_date': test_date,
                'model_type': self.backtest.model_type,
                'num_training_samples': num_samples,
                'feature_config': self.backtest.feature_config_name,
                'model_params': self.backtest.model_params,
                'trained_at': pd.Timestamp.now().isoformat()
            }

            if hasattr(model, 'save'):
                model.save(str(model_file))
            else:
                # Sklearn-style model
                import pickle
                with open(model_file, 'wb') as f:
                    pickle.dump({'model': model, 'metadata': metadata}, f)

            # Save metadata separately
            metadata_file = model_file.with_suffix('.meta.json')
            import json
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)

            logger.debug(f"Saved slate model to {model_file}")

        except Exception as e:
            logger.error(f"Failed to save slate model: {e}")