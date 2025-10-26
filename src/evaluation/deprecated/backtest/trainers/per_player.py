"""
Per-player model training strategy for walk-forward backtesting.

This module implements per-player model training where each player
gets their own dedicated model trained on their historical performance.
"""

import pandas as pd
import numpy as np
import logging
import pickle
import time
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from tqdm import tqdm
from joblib import Parallel, delayed
from datetime import datetime

from src.models.xgboost_model import XGBoostModel
from src.models.random_forest_model import RandomForestModel
from src.utils.fantasy_points import calculate_dk_fantasy_points
from src.features.pipeline import FeaturePipeline
from src.utils.feature_config import load_feature_config
from src.config.paths import (
    PER_PLAYER_MODEL_DIR,
    PER_PLAYER_TRAINING_INPUTS_DIR
)
from ..trainer_base import BacktestTrainer

logger = logging.getLogger(__name__)


def _train_single_player_model(
    player_row: pd.Series,
    training_data: pd.DataFrame,
    feature_config_name: str,
    model_type: str,
    model_params: Dict,
    min_player_games: int,
    save_models: bool,
    models_dir: Path,
    inputs_dir: Path,
    injuries: Optional[pd.DataFrame] = None
) -> Optional[Dict[str, Any]]:
    """
    Worker function for parallel per-player model training.

    Returns dict with projection data or None if player should be skipped.

    Note: Creates a fresh feature pipeline in each worker thread to avoid
    thread safety issues with shared pipeline state.
    """
    player_id = player_row.get('playerID')
    player_name = player_row.get('longName')

    player_training_data = training_data[training_data['playerID'] == player_id].copy()

    if len(player_training_data) < min_player_games:
        return None

    try:
        # Create fresh pipeline in this worker thread to avoid thread safety issues
        config = load_feature_config(feature_config_name)
        feature_pipeline = config.build_pipeline(FeaturePipeline)

        df = player_training_data.copy()

        if 'gameDate' not in df.columns:
            logger.debug(f"gameDate column missing for player {player_id}")
            return None

        df['gameDate'] = pd.to_datetime(df['gameDate'], format='%Y%m%d', errors='coerce')
        df = df.sort_values(['playerID', 'gameDate'])

        if 'fpts' not in df.columns:
            df['fpts'] = df.apply(calculate_dk_fantasy_points, axis=1)

        # Always pass injuries in context to ensure consistent feature columns
        injuries_to_use = injuries if injuries is not None else pd.DataFrame()
        context = {'injuries': injuries_to_use}

        df = feature_pipeline.fit_transform(df, context=context)
        df = df.dropna(subset=['target'])

        # Select only numeric columns (int64, float64) to avoid dtype issues with XGBoost
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

        # Remove target from features (it's separate)
        feature_cols = [col for col in numeric_cols if col != 'target']

        X_train = df[feature_cols].copy()
        X_train = X_train.fillna(0).infer_objects(copy=False)

        y_train = df['target'].fillna(0)

        if X_train.empty or y_train.empty or len(X_train) < 3:
            return None

        if model_type == 'xgboost':
            model = XGBoostModel(model_params)
        elif model_type == 'random_forest':
            model = RandomForestModel(model_params)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        model.train(X_train, y_train)

        # Return projection data
        projection_data = {
            'playerID': player_id,
            'playerName': player_name,
            'longName': player_name,
            'pos': player_row.get('pos'),
            'teamAbv': player_row.get('teamAbv'),
            'salary': player_row.get('salary'),
            'projection': 0,  # Will be filled by caller
            'actual': 0,      # Will be filled by caller
            'model': model
        }

        return projection_data

    except Exception as e:
        logger.debug(f"Error training model for {player_name}: {str(e)}")
        return None


class PerPlayerTrainer(BacktestTrainer):
    """
    Per-player model training strategy.

    Each player gets their own dedicated model trained on their
    historical performance data.
    """

    def __init__(self, backtest):
        """
        Initialize per-player trainer.

        Args:
            backtest: Parent WalkForwardBacktest instance
        """
        super().__init__(backtest)
        self.player_models = {}
        self.last_training_date = None

    def train_models(
        self,
        training_data: pd.DataFrame,
        slate_data: Dict[str, Any],
        test_date: str
    ) -> Dict[str, Any]:
        """
        Train per-player models.

        Args:
            training_data: Historical training data
            slate_data: Current slate data
            test_date: Test date

        Returns:
            Dictionary of trained models
        """
        should_recalibrate = self.backtest._should_recalibrate(test_date)

        if not should_recalibrate and self.player_models:
            logger.info(f"Reusing {len(self.player_models)} cached player models from {self.last_training_date}")
            return self.player_models

        logger.info("Training per-player models")
        salaries_df = slate_data.get('dfs_salaries', pd.DataFrame())

        # Apply injury features
        injuries_data = slate_data.get('injuries', pd.DataFrame())
        from src.features.transformers.injury import InjuryTransformer
        injury_transformer = InjuryTransformer()
        injury_transformer.fit(salaries_df)
        salaries_df = injury_transformer.transform(salaries_df, injuries_data)

        # Apply player filters
        if self.backtest.player_filters:
            logger.info(f"Applying {len(self.backtest.player_filters)} player filters...")
            initial_count = len(salaries_df)
            for pf in self.backtest.player_filters:
                salaries_df = pf.apply(salaries_df)
                logger.info(f"  Applied {pf}: {len(salaries_df)} players remaining")
            logger.info(f"Filter results: {len(salaries_df)}/{initial_count} players passed all filters")

        # Prepare model directory
        models_dir = Path(PER_PLAYER_MODEL_DIR)
        models_dir.mkdir(parents=True, exist_ok=True)

        # Train models in parallel or sequentially
        if self.backtest.n_jobs != 1:
            models = self._train_parallel(salaries_df, training_data, injuries_data, models_dir)
        else:
            models = self._train_sequential(salaries_df, training_data, injuries_data, models_dir)

        self.player_models = models
        self.last_training_date = test_date

        logger.info(f"Trained {len(models)} per-player models")
        return models

    def _train_parallel(
        self,
        salaries_df: pd.DataFrame,
        training_data: pd.DataFrame,
        injuries_data: pd.DataFrame,
        models_dir: Path
    ) -> Dict[str, Any]:
        """
        Train models in parallel.

        Args:
            salaries_df: Slate salaries data
            training_data: Historical training data
            injuries_data: Injury data
            models_dir: Directory to save models

        Returns:
            Dictionary of trained models
        """
        logger.info(f"Training models in parallel with {self.backtest.n_jobs} workers")

        player_rows = [row for _, row in salaries_df.iterrows()]

        # Use threading backend for better interrupt handling on Windows
        results = Parallel(
            n_jobs=self.backtest.n_jobs,
            verbose=10,
            backend='threading',
            timeout=600  # 10 minute timeout per worker
        )(
            delayed(_train_single_player_model)(
                player_row,
                training_data,
                self.backtest.feature_config_name,
                self.backtest.model_type,
                self.backtest.model_params,
                self.backtest.min_player_games,
                self.backtest.save_models,
                models_dir,
                self.backtest.run_inputs_dir,
                injuries_data
            )
            for player_row in player_rows
        )

        models = {}
        for result in results:
            if result is not None:
                player_id = result['playerID']
                model = result.pop('model')
                if model is not None:
                    models[player_id] = model

        return models

    def _train_sequential(
        self,
        salaries_df: pd.DataFrame,
        training_data: pd.DataFrame,
        injuries_data: pd.DataFrame,
        models_dir: Path
    ) -> Dict[str, Any]:
        """
        Train models sequentially.

        Args:
            salaries_df: Slate salaries data
            training_data: Historical training data
            injuries_data: Injury data
            models_dir: Directory to save models

        Returns:
            Dictionary of trained models
        """
        models = {}
        model_train_times = []
        total_players = len(salaries_df)
        log_interval = max(1, total_players // 10)

        for idx, player_row in tqdm(
            salaries_df.iterrows(),
            total=len(salaries_df),
            desc="Per-player models",
            leave=False
        ):
            player_id = player_row.get('playerID')
            player_name = player_row.get('longName')

            player_training_data = training_data[training_data['playerID'] == player_id].copy()

            if len(player_training_data) < self.backtest.min_player_games:
                logger.debug(f"Skipping {player_name}: only {len(player_training_data)} games")
                continue

            try:
                model_start_time = time.perf_counter()

                # Build features
                X_train, y_train = self.backtest._build_training_features(
                    player_training_data,
                    injuries_data
                )

                if X_train.empty or y_train.empty or len(X_train) < 3:
                    logger.debug(f"Insufficient features for {player_name}")
                    continue

                # Train model
                if self.backtest.model_type == 'xgboost':
                    model = XGBoostModel(self.backtest.model_params)
                elif self.backtest.model_type == 'random_forest':
                    model = RandomForestModel(self.backtest.model_params)
                else:
                    raise ValueError(f"Unknown model type: {self.backtest.model_type}")

                model.train(X_train, y_train)
                models[player_id] = model

                # Save model if configured
                if self.backtest.save_models:
                    safe_player_name = "".join(
                        c if c.isalnum() or c in (' ', '_', '-') else '_'
                        for c in player_name
                    ).replace(' ', '_')
                    model_file = models_dir / f"{safe_player_name}_{player_id}.pkl"
                    self.backtest._save_model(model, model_file, player_name, player_id, len(X_train))

                model_elapsed = time.perf_counter() - model_start_time
                model_train_times.append(model_elapsed)

                if len(models) % log_interval == 0:
                    avg_model_time = sum(model_train_times) / len(model_train_times)
                    models_per_sec = 1 / avg_model_time if avg_model_time > 0 else 0
                    remaining_models = total_players - (idx + 1)
                    eta_models = remaining_models * avg_model_time
                    logger.info(
                        f"  Progress: {len(models)} models trained "
                        f"({len(models)/total_players*100:.1f}%) - "
                        f"{models_per_sec:.2f} models/sec - "
                        f"ETA: {self.backtest._format_time(eta_models)}"
                    )

            except Exception as e:
                logger.warning(f"Error training model for {player_name}: {str(e)}")
                continue

        if model_train_times:
            avg_train_time = sum(model_train_times) / len(model_train_times)
            logger.info(
                f"Average model training time: {self.backtest._format_time(avg_train_time)} "
                f"({1/avg_train_time:.2f} models/sec)"
            )

        return models

    def generate_projections(
        self,
        slate_data: Dict[str, Any],
        training_data: pd.DataFrame,
        test_date: str
    ) -> pd.DataFrame:
        """
        Generate per-player projections.

        Args:
            slate_data: Current slate data
            training_data: Historical training data
            test_date: Test date

        Returns:
            DataFrame with projections
        """
        # Check for GPU batch processing
        if self.backtest.gpu_batch_trainer and self.backtest.gpu_batch_size > 1:
            return self._generate_projections_gpu_batch(slate_data, training_data, test_date)

        return self._generate_projections_sequential(slate_data, training_data, test_date)

    def _generate_projections_sequential(
        self,
        slate_data: Dict[str, Any],
        training_data: pd.DataFrame,
        test_date: str
    ) -> pd.DataFrame:
        """
        Generate projections sequentially.

        Args:
            slate_data: Current slate data
            training_data: Historical training data
            test_date: Test date

        Returns:
            DataFrame with projections
        """
        logger.info("Generating per-player model projections")

        salaries_df = slate_data['dfs_salaries'].copy()

        # Add injury features
        injuries_data = slate_data.get('injuries', pd.DataFrame())
        from src.features.transformers.injury import InjuryTransformer
        injury_transformer = InjuryTransformer()
        injury_transformer.fit(salaries_df)
        salaries_df = injury_transformer.transform(salaries_df, injuries_data)

        # Apply filters
        if self.backtest.player_filters:
            initial_count = len(salaries_df)
            for pf in self.backtest.player_filters:
                salaries_df = pf.apply(salaries_df)
            logger.info(f"Filter results: {len(salaries_df)}/{initial_count} players passed all filters")
            salaries_df = salaries_df.reset_index(drop=True)

        all_projections = []

        for idx, player_row in tqdm(
            salaries_df.iterrows(),
            total=len(salaries_df),
            desc="Generating projections",
            leave=False
        ):
            player_id = player_row.get('playerID')
            player_name = player_row.get('longName')

            if player_id not in self.player_models:
                logger.debug(f"No model for {player_name}")
                continue

            try:
                model = self.player_models[player_id]

                # Build features for this player
                player_training_data = training_data[training_data['playerID'] == player_id].copy()
                slate_data_single = {
                    'dfs_salaries': salaries_df.iloc[[idx]],
                    'date': slate_data['date'],
                    'schedule': slate_data.get('schedule', pd.DataFrame()),
                    'betting_odds': slate_data.get('betting_odds', pd.DataFrame()),
                    'injuries': slate_data.get('injuries', pd.DataFrame())
                }

                slate_features = self.backtest._build_slate_features(slate_data_single, player_training_data)

                if slate_features.empty:
                    logger.debug(f"No features generated for {player_name}")
                    continue

                projection = self.backtest._generate_projections(model, slate_features)
                all_projections.append(projection)

            except Exception as e:
                logger.warning(f"Error generating projection for {player_name}: {str(e)}")
                continue

        if not all_projections:
            logger.warning("No player projections generated")
            return pd.DataFrame()

        projections_df = pd.concat(all_projections, ignore_index=True)
        logger.info(f"Generated projections for {len(projections_df)} players")

        return projections_df

    def _generate_projections_gpu_batch(
        self,
        slate_data: Dict[str, Any],
        training_data: pd.DataFrame,
        test_date: str
    ) -> pd.DataFrame:
        """
        Generate projections using GPU batch processing.

        Args:
            slate_data: Current slate data
            training_data: Historical training data
            test_date: Test date

        Returns:
            DataFrame with projections
        """
        logger.info(f"Generating per-player projections with GPU batch processing (batch_size={self.backtest.gpu_batch_size})")

        # Placeholder for GPU batch implementation
        # This would utilize the gpu_batch_trainer for efficient batch predictions
        # For now, fall back to sequential
        return self._generate_projections_sequential(slate_data, training_data, test_date)