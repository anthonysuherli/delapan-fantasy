import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, Any
from .base import BaseModel


class XGBoostModel(BaseModel):
    """XGBoost regression model for player projections"""

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize XGBoost model.

        Args:
            config: Model configuration. Defaults to sensible hyperparameters if None.
        """
        default_config = {
            'max_depth': 6,
            'learning_rate': 0.05,
            'n_estimators': 200,
            'min_child_weight': 5,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'objective': 'reg:squarederror',
            'random_state': 42,
            'enable_categorical': True
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
    ) -> 'XGBoostModel':
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
        import xgboost as xgb

        if len(X) != len(y):
            raise ValueError(f"X and y have mismatched lengths: {len(X)} vs {len(y)}")

        if save_inputs and input_save_path:
            self._save_training_inputs(X, y, input_save_path)

        self.model = xgb.XGBRegressor(**self.config)
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

    def load(self, path: str) -> 'XGBoostModel':
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

        importance = self.model.feature_importances_
        feature_names = self.model.get_booster().feature_names
        return pd.Series(importance, index=feature_names).sort_values(ascending=False)
