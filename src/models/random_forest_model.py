import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, Any
from .base import BaseModel


class RandomForestModel(BaseModel):
    """Random Forest regression model for player projections"""

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize Random Forest model.

        Args:
            config: Model configuration. Defaults to sensible hyperparameters if None.
        """
        default_config = {
            'n_estimators': 200,
            'max_depth': 10,
            'min_samples_split': 10,
            'min_samples_leaf': 5,
            'max_features': 'sqrt',
            'random_state': 42,
            'n_jobs': -1
        }
        config = {**default_config, **(config or {})}
        super().__init__(config)
        self.model = None

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        save_inputs: bool = False,
        input_save_path: str = None
    ) -> 'RandomForestModel':
        """
        Train model on data.

        Args:
            X: Feature matrix with shape (n_samples, n_features)
            y: Target variable with shape (n_samples,)
            save_inputs: Whether to save training inputs to disk
            input_save_path: Path to save training inputs

        Returns:
            Self for method chaining

        Raises:
            ValueError: If X and y have mismatched lengths
        """
        from sklearn.ensemble import RandomForestRegressor

        if len(X) != len(y):
            raise ValueError(f"X and y have mismatched lengths: {len(X)} vs {len(y)}")

        if save_inputs and input_save_path:
            self._save_training_inputs(X, y, input_save_path)

        self.model = RandomForestRegressor(**self.config)
        self.model.fit(X, y)
        self._is_trained = True
        return self

    def _save_training_inputs(self, X: pd.DataFrame, y: pd.Series, path: str) -> None:
        """
        Save training inputs to disk.

        Args:
            X: Feature matrix
            y: Target variable
            path: Path to save inputs
        """
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        training_data = X.copy()
        training_data['target'] = y
        training_data.to_parquet(path_obj, index=False)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions.

        Args:
            X: Feature matrix with shape (n_samples, n_features)

        Returns:
            Predictions with shape (n_samples,)

        Raises:
            ValueError: If model has not been trained
        """
        if not self._is_trained:
            raise ValueError("Model must be trained before prediction")
        return self.model.predict(X)

    def save(self, path: str) -> None:
        """
        Serialize model to disk.

        Args:
            path: File path to save model
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)

    def load(self, path: str) -> 'RandomForestModel':
        """
        Load model from disk.

        Args:
            path: File path to load model from

        Returns:
            Self for method chaining

        Raises:
            FileNotFoundError: If model file does not exist
        """
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
        self._is_trained = True
        return self

    def get_feature_importance(self) -> pd.Series:
        """
        Get feature importance scores.

        Returns:
            Series with feature names and importance scores

        Raises:
            ValueError: If model has not been trained
        """
        if not self._is_trained:
            raise ValueError("Model must be trained to get feature importance")

        return pd.Series(
            self.model.feature_importances_,
            index=self.model.feature_names_in_
        ).sort_values(ascending=False)
