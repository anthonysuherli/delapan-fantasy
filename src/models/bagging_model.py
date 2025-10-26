"""
Bagging Ensemble Model for variance reduction through bootstrap aggregating.

Bagging trains multiple instances of the same model on bootstrap samples
of the training data and averages their predictions for improved stability.
"""

import pandas as pd
import numpy as np
import pickle
from typing import Dict, Any, List
from pathlib import Path

from .base import BaseModel


class BaggingModel(BaseModel):
    """
    Bagging (Bootstrap Aggregating) ensemble model.

    The bagging model:
    1. Creates bootstrap samples from training data
    2. Trains a base model on each bootstrap sample
    3. Averages predictions from all models

    Config format:
    {
        "model_type": "bagging",
        "n_estimators": 10,
        "bootstrap_samples": 0.8,
        "bootstrap_features": 1.0,
        "base_model": {
            "type": "xgboost",
            "params": {...}
        }
    }
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize bagging model.

        Parameters
        ----------
        config : Dict[str, Any]
            Model configuration with base_model spec and bagging parameters
        """
        super().__init__(config)

        self.n_estimators = config.get('n_estimators', 10)
        self.bootstrap_samples = config.get('bootstrap_samples', 0.8)
        self.bootstrap_features = config.get('bootstrap_features', 1.0)
        self.base_model_config = config.get('base_model', {'type': 'xgboost', 'params': {}})
        self.estimators = []

    def _create_bootstrap_sample(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_ratio: float,
        feature_ratio: float,
        random_state: int
    ) -> tuple:
        """
        Create bootstrap sample of data.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target variable
        sample_ratio : float
            Fraction of samples to include
        feature_ratio : float
            Fraction of features to include
        random_state : int
            Random seed

        Returns
        -------
        tuple
            (X_bootstrap, y_bootstrap, selected_features)
        """
        np.random.seed(random_state)

        n_samples = int(len(X) * sample_ratio)
        n_features = int(X.shape[1] * feature_ratio)

        # Sample rows with replacement
        sample_indices = np.random.choice(len(X), size=n_samples, replace=True)

        # Sample columns without replacement
        feature_indices = np.random.choice(X.shape[1], size=n_features, replace=False)
        selected_features = X.columns[feature_indices].tolist()

        X_bootstrap = X.iloc[sample_indices][selected_features]
        y_bootstrap = y.iloc[sample_indices]

        return X_bootstrap, y_bootstrap, selected_features

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        save_inputs: bool = False,
        input_save_path: str = None
    ) -> 'BaggingModel':
        """
        Train bagging ensemble.

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
        BaggingModel
            Self for method chaining
        """
        if len(X) != len(y):
            raise ValueError(f"X and y have mismatched lengths: {len(X)} vs {len(y)}")

        # Import registry here to avoid circular imports
        from .registry import registry

        self.estimators = []

        # Train each estimator on a bootstrap sample
        for i in range(self.n_estimators):
            # Create bootstrap sample
            X_boot, y_boot, selected_features = self._create_bootstrap_sample(
                X, y,
                self.bootstrap_samples,
                self.bootstrap_features,
                random_state=42 + i  # Different seed for each estimator
            )

            # Create and train model
            model = registry.create(
                self.base_model_config['type'],
                self.base_model_config.get('params', {})
            )
            model.train(X_boot, y_boot)

            # Store model with its selected features
            self.estimators.append({
                'model': model,
                'features': selected_features
            })

        self._is_trained = True

        # Save inputs if requested
        if save_inputs and input_save_path:
            Path(input_save_path).parent.mkdir(parents=True, exist_ok=True)
            with open(input_save_path, 'wb') as f:
                pickle.dump({'X': X, 'y': y}, f)

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions by averaging predictions from all estimators.

        Parameters
        ----------
        X : pd.DataFrame
            Features for prediction

        Returns
        -------
        np.ndarray
            Average predictions from all estimators
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        # Get predictions from each estimator
        predictions = np.zeros((len(X), self.n_estimators))

        for i, estimator_dict in enumerate(self.estimators):
            model = estimator_dict['model']
            features = estimator_dict['features']

            # Use only the features this model was trained on
            X_subset = X[features]

            predictions[:, i] = model.predict(X_subset)

        # Average predictions
        avg_predictions = predictions.mean(axis=1)

        return avg_predictions

    def predict_with_variance(self, X: pd.DataFrame) -> tuple:
        """
        Generate predictions with variance estimates.

        Parameters
        ----------
        X : pd.DataFrame
            Features for prediction

        Returns
        -------
        tuple
            (mean_predictions, std_predictions)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        # Get predictions from each estimator
        predictions = np.zeros((len(X), self.n_estimators))

        for i, estimator_dict in enumerate(self.estimators):
            model = estimator_dict['model']
            features = estimator_dict['features']
            X_subset = X[features]
            predictions[:, i] = model.predict(X_subset)

        # Calculate mean and std
        mean_predictions = predictions.mean(axis=1)
        std_predictions = predictions.std(axis=1)

        return mean_predictions, std_predictions

    def save(self, path: str) -> None:
        """
        Save bagging model to disk.

        Parameters
        ----------
        path : str
            File path to save model
        """
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        model_state = {
            'config': self.config,
            'estimators': self.estimators,
            'is_trained': self._is_trained
        }

        with open(save_path, 'wb') as f:
            pickle.dump(model_state, f)

    def load(self, path: str) -> 'BaggingModel':
        """
        Load bagging model from disk.

        Parameters
        ----------
        path : str
            File path to load model from

        Returns
        -------
        BaggingModel
            Self for method chaining
        """
        with open(path, 'rb') as f:
            model_state = pickle.load(f)

        self.config = model_state['config']
        self.estimators = model_state['estimators']
        self._is_trained = model_state['is_trained']

        # Restore config parameters
        self.n_estimators = self.config.get('n_estimators', len(self.estimators))
        self.bootstrap_samples = self.config.get('bootstrap_samples', 0.8)
        self.bootstrap_features = self.config.get('bootstrap_features', 1.0)
        self.base_model_config = self.config.get('base_model', {})

        return self
