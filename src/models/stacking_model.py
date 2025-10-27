"""
Stacking Ensemble Model for combining multiple base models.

Stacking (stacked generalization) trains multiple base models and uses
their predictions as features for a meta-model that makes the final prediction.
"""

import pandas as pd
import numpy as np
import pickle
from typing import Dict, Any, List
from pathlib import Path

from .base import BaseModel


class StackingModel(BaseModel):
    """
    Stacking ensemble that combines predictions from multiple base models.

    The stacking model:
    1. Trains multiple base models on the training data
    2. Generates out-of-fold predictions from base models
    3. Trains a meta-model on base model predictions
    4. Final prediction is meta-model output on base model predictions

    Config format:
    {
        "model_type": "stacking",
        "base_models": [
            {"type": "xgboost", "params": {...}},
            {"type": "random_forest", "params": {...}}
        ],
        "meta_model": {
            "type": "xgboost",
            "params": {...}
        },
        "cv_folds": 5  # For out-of-fold predictions
    }
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize stacking model.

        Parameters
        ----------
        config : Dict[str, Any]
            Model configuration with base_models and meta_model specs
        """
        super().__init__(config)

        # Import registry here to avoid circular imports
        from .registry import registry

        self.base_models = []
        self.meta_model = None
        self.cv_folds = config.get('cv_folds', 5)

        # Create base models from config
        for base_config in config.get('base_models', []):
            model_type = base_config['type']
            model_params = base_config.get('params', {})
            base_model = registry.create(model_type, model_params)
            self.base_models.append(base_model)

        # Create meta model from config
        meta_config = config.get('meta_model', {'type': 'xgboost', 'params': {}})
        self.meta_model = registry.create(
            meta_config['type'],
            meta_config.get('params', {})
        )

    def _generate_oof_predictions(self, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        """
        Generate out-of-fold predictions from base models.

        Parameters
        ----------
        X : pd.DataFrame
            Training features
        y : pd.Series
            Training targets

        Returns
        -------
        np.ndarray
            Out-of-fold predictions with shape (n_samples, n_base_models)
        """
        from sklearn.model_selection import KFold

        n_samples = len(X)
        n_models = len(self.base_models)
        oof_predictions = np.zeros((n_samples, n_models))

        # Generate out-of-fold predictions for each base model
        kfold = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)

        for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X)):
            X_train_fold = X.iloc[train_idx]
            y_train_fold = y.iloc[train_idx]
            X_val_fold = X.iloc[val_idx]

            # Train each base model on this fold
            for model_idx, base_model in enumerate(self.base_models):
                # Create a fresh model instance for this fold
                from .registry import registry
                model_config = self.config['base_models'][model_idx]
                fold_model = registry.create(model_config['type'], model_config.get('params', {}))

                # Train on fold
                fold_model.train(X_train_fold, y_train_fold)

                # Predict on validation set
                fold_predictions = fold_model.predict(X_val_fold)

                # Store out-of-fold predictions
                oof_predictions[val_idx, model_idx] = fold_predictions

        return oof_predictions

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        save_inputs: bool = False,
        input_save_path: str = None
    ) -> 'StackingModel':
        """
        Train stacking ensemble.

        Parameters
        ----------
        X : pd.DataFrame
            Training features
        y : pd.Series
            Training targets
        save_inputs : bool
            Whether to save training inputs
        input_save_path : str
            Path to save training inputs

        Returns
        -------
        StackingModel
            Self for method chaining
        """
        if len(X) != len(y):
            raise ValueError(f"X and y have mismatched lengths: {len(X)} vs {len(y)}")

        # Step 1: Train base models on full data
        for base_model in self.base_models:
            base_model.train(X, y)

        # Step 2: Generate out-of-fold predictions for meta-model training
        oof_predictions = self._generate_oof_predictions(X, y)

        # Step 3: Train meta-model on base model predictions
        meta_features = pd.DataFrame(
            oof_predictions,
            columns=[f'base_model_{i}' for i in range(len(self.base_models))]
        )
        self.meta_model.train(meta_features, y)

        self._is_trained = True

        # Save inputs if requested
        if save_inputs and input_save_path:
            Path(input_save_path).parent.mkdir(parents=True, exist_ok=True)
            with open(input_save_path, 'wb') as f:
                pickle.dump({'X': X, 'y': y}, f)

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions using stacking ensemble.

        Parameters
        ----------
        X : pd.DataFrame
            Features for prediction

        Returns
        -------
        np.ndarray
            Predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        # Step 1: Get predictions from each base model
        base_predictions = np.zeros((len(X), len(self.base_models)))

        for i, base_model in enumerate(self.base_models):
            base_predictions[:, i] = base_model.predict(X)

        # Step 2: Use base predictions as features for meta-model
        meta_features = pd.DataFrame(
            base_predictions,
            columns=[f'base_model_{i}' for i in range(len(self.base_models))]
        )

        # Step 3: Get final predictions from meta-model
        final_predictions = self.meta_model.predict(meta_features)

        return final_predictions

    def save(self, path: str) -> None:
        """
        Save stacking model to disk.

        Parameters
        ----------
        path : str
            File path to save model
        """
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        model_state = {
            'config': self.config,
            'base_models': self.base_models,
            'meta_model': self.meta_model,
            'is_trained': self._is_trained
        }

        with open(save_path, 'wb') as f:
            pickle.dump(model_state, f)

    def load(self, path: str) -> 'StackingModel':
        """
        Load stacking model from disk.

        Parameters
        ----------
        path : str
            File path to load model from

        Returns
        -------
        StackingModel
            Self for method chaining
        """
        with open(path, 'rb') as f:
            model_state = pickle.load(f)

        self.config = model_state['config']
        self.base_models = model_state['base_models']
        self.meta_model = model_state['meta_model']
        self._is_trained = model_state['is_trained']

        return self
