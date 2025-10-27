"""
Minutes Projection Model.

Separate model to predict player minutes. Used to:
1. Filter out low-minute players before fantasy point prediction
2. As a feature for fantasy point models
3. Improve overall prediction accuracy by reducing MAPE inflation from DNPs/low minutes
"""

import numpy as np
import pandas as pd
from typing import Optional
from xgboost import XGBRegressor
from .base import BaseModel


class MinutesProjectionModel(BaseModel):
    """
    Model specifically for projecting player minutes.

    Uses similar features as fantasy point models but targets minutes played.
    Helps identify players likely to get significant playing time.
    """

    def __init__(self, config: dict):
        """
        Initialize minutes projection model.

        Parameters
        ----------
        config : dict
            Model configuration with hyperparameters
        """
        super().__init__(config)

        # XGBoost regressor for minutes
        self.model = XGBRegressor(
            max_depth=config.get('max_depth', 4),
            learning_rate=config.get('learning_rate', 0.05),
            n_estimators=config.get('n_estimators', 100),
            min_child_weight=config.get('min_child_weight', 5),
            subsample=config.get('subsample', 0.8),
            colsample_bytree=config.get('colsample_bytree', 0.8),
            objective='reg:squarederror',
            random_state=config.get('random_state', 42),
            tree_method=config.get('tree_method', 'hist')
        )

        # Minutes thresholds
        self.min_minutes_threshold = config.get('min_minutes_threshold', 15)

    def train(self, X: pd.DataFrame, y: pd.Series) -> 'MinutesProjectionModel':
        """
        Train the minutes projection model.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target minutes values

        Returns
        -------
        self
        """
        # Fit model
        self.model.fit(X, y, verbose=False)
        self._is_trained = True

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict minutes for players.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix

        Returns
        -------
        np.ndarray
            Predicted minutes
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        predictions = self.model.predict(X)

        # Clip predictions to reasonable range [0, 48]
        predictions = np.clip(predictions, 0, 48)

        return predictions

    def predict_with_threshold(self, X: pd.DataFrame) -> tuple:
        """
        Predict minutes and filter based on threshold.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix

        Returns
        -------
        tuple
            (predictions, mask) where mask indicates players above threshold
        """
        predictions = self.predict(X)
        mask = predictions >= self.min_minutes_threshold

        return predictions, mask

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance scores.

        Returns
        -------
        pd.DataFrame
            Feature importance scores
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting feature importance")

        importance = self.model.feature_importances_
        feature_names = self.model.get_booster().feature_names

        return pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)

    def save(self, path: str) -> None:
        """
        Save model to file.

        Parameters
        ----------
        path : str
            File path to save model
        """
        import joblib
        joblib.dump({
            'model': self.model,
            'config': self.config,
            'min_minutes_threshold': self.min_minutes_threshold
        }, path)

    def load(self, path: str) -> 'MinutesProjectionModel':
        """
        Load model from file.

        Parameters
        ----------
        path : str
            File path to load model from

        Returns
        -------
        self
        """
        import joblib
        data = joblib.load(path)

        self.model = data['model']
        self.config = data.get('config', {})
        self.min_minutes_threshold = data.get('min_minutes_threshold', 15)
        self._is_trained = True

        return self
