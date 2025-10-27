"""
Quantile Regression Model.

Predicts multiple quantiles (floor/median/ceiling) for fantasy points using XGBoost.
Enables variance estimation and confidence intervals for risk-adjusted optimization.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
from xgboost import XGBRegressor
from .base import BaseModel


class QuantileRegressionModel(BaseModel):
    """
    Quantile regression model for variance prediction.

    Trains three separate XGBoost models to predict:
    - 10th percentile (floor): Conservative estimate
    - 50th percentile (median): Expected value
    - 90th percentile (ceiling): Optimistic estimate

    Used for GPP tournament optimization where high-ceiling players are prioritized.
    """

    def __init__(self, config: dict):
        """
        Initialize quantile regression model.

        Parameters
        ----------
        config : dict
            Model configuration with hyperparameters and quantiles
        """
        super().__init__(config)

        # Quantiles to predict
        self.quantiles = config.get('quantiles', [0.1, 0.5, 0.9])

        # Create model for each quantile
        self.models: Dict[float, XGBRegressor] = {}

        for q in self.quantiles:
            self.models[q] = XGBRegressor(
                max_depth=config.get('max_depth', 6),
                learning_rate=config.get('learning_rate', 0.05),
                n_estimators=config.get('n_estimators', 200),
                min_child_weight=config.get('min_child_weight', 5),
                subsample=config.get('subsample', 0.8),
                colsample_bytree=config.get('colsample_bytree', 0.8),
                objective=f'reg:quantileerror',
                random_state=config.get('random_state', 42),
                tree_method=config.get('tree_method', 'hist'),
                quantile_alpha=q  # XGBoost 2.0+ parameter for quantile loss
            )

    def train(self, X: pd.DataFrame, y: pd.Series) -> 'QuantileRegressionModel':
        """
        Train quantile regression models.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target values

        Returns
        -------
        self
        """
        # Train each quantile model
        for q, model in self.models.items():
            model.fit(X, y, verbose=False)

        self._is_trained = True
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Predict quantiles for each observation.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: floor (q10), median (q50), ceiling (q90)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        predictions = {}

        for q, model in self.models.items():
            pred = model.predict(X)

            # Ensure non-negative predictions
            pred = np.maximum(pred, 0)

            # Map quantile to column name
            if q == 0.1:
                predictions['floor'] = pred
            elif q == 0.5:
                predictions['median'] = pred
            elif q == 0.9:
                predictions['ceiling'] = pred
            else:
                predictions[f'q{int(q*100)}'] = pred

        return pd.DataFrame(predictions, index=X.index)

    def predict_with_variance(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Predict with variance estimates.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix

        Returns
        -------
        pd.DataFrame
            Predictions with floor, median, ceiling, variance, iqr
        """
        predictions = self.predict(X)

        # Calculate variance metrics
        predictions['variance'] = (predictions['ceiling'] - predictions['floor']) / 2
        predictions['iqr'] = predictions['ceiling'] - predictions['floor']
        predictions['cv'] = predictions['variance'] / (predictions['median'] + 1e-6)  # Coefficient of variation

        return predictions

    def get_feature_importance(self, quantile: float = 0.5) -> pd.DataFrame:
        """
        Get feature importance for a specific quantile model.

        Parameters
        ----------
        quantile : float
            Quantile to get importance for (default 0.5 for median)

        Returns
        -------
        pd.DataFrame
            Feature importance scores
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting feature importance")

        if quantile not in self.models:
            raise ValueError(f"Quantile {quantile} not in trained models: {list(self.models.keys())}")

        model = self.models[quantile]
        importance = model.feature_importances_
        feature_names = model.get_booster().feature_names

        return pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)

    def save(self, path: str) -> None:
        """
        Save quantile models to file.

        Parameters
        ----------
        path : str
            File path to save models (will create multiple files with _q{quantile} suffix)
        """
        import joblib
        import os

        base_path = path.rsplit('.', 1)[0] if '.' in path else path

        # Save each quantile model
        for q, model in self.models.items():
            q_path = f"{base_path}_q{int(q*100)}.pkl"
            joblib.dump({
                'model': model,
                'quantile': q,
                'config': self.config
            }, q_path)

    def load(self, path: str) -> 'QuantileRegressionModel':
        """
        Load quantile models from file.

        Parameters
        ----------
        path : str
            Base file path to load models from

        Returns
        -------
        self
        """
        import joblib
        import os
        import glob

        base_path = path.rsplit('.', 1)[0] if '.' in path else path

        # Find all quantile model files
        q_files = glob.glob(f"{base_path}_q*.pkl")

        if not q_files:
            raise FileNotFoundError(f"No quantile model files found matching {base_path}_q*.pkl")

        # Load each quantile model
        self.models = {}
        for q_path in q_files:
            data = joblib.load(q_path)
            q = data['quantile']
            self.models[q] = data['model']
            self.config = data.get('config', {})

        self.quantiles = sorted(self.models.keys())
        self._is_trained = True

        return self
