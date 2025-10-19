import pandas as pd
import numpy as np
import pickle
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from .base import BaseModel

logger = logging.getLogger(__name__)


def _get_gpu_preprocessor():
    """GPU preprocessor removed - functionality integrated into model."""
    return None


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
            'enable_categorical': True,
            'tree_method': 'hist',
            'n_jobs': 1  # Disable XGBoost parallelization to avoid conflicts with joblib
        }
        config = {**default_config, **(config or {})}
        super().__init__(config)
        self.model = None

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        save_inputs: bool = False,
        input_save_path: str = None,
        use_gpu_preprocessing: bool = True
    ) -> 'XGBoostModel':
        """
        Train model on data with optional GPU preprocessing.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix with shape (n_samples, n_features)
        y : pd.Series
            Target variable with shape (n_samples,)
        save_inputs : bool, optional
            Whether to save training inputs to disk. Default: False
        input_save_path : str, optional
            Path to save training inputs
        use_gpu_preprocessing : bool, optional
            Preprocess data for GPU optimization (float32, memory-optimized).
            Default: True

        Returns
        -------
        XGBoostModel
            Self for method chaining

        Raises
        ------
        ValueError
            If X and y have mismatched lengths

        Notes
        -----
        GPU preprocessing converts data to float32 and ensures C-contiguous
        memory layout for faster GPU transfer during training. This is beneficial
        even when not using GPU training, as it reduces memory footprint.
        """
        import xgboost as xgb

        if len(X) != len(y):
            raise ValueError(f"X and y have mismatched lengths: {len(X)} vs {len(y)}")

        # Preprocess data for GPU optimization
        if use_gpu_preprocessing:
            X, y = self._preprocess_for_gpu(X, y)

        if save_inputs and input_save_path:
            self._save_training_inputs(X, y, input_save_path)

        self.model = xgb.XGBRegressor(**self.config)
        self.model.fit(X, y)
        self._is_trained = True
        return self

    def _preprocess_for_gpu(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> tuple:
        """
        Preprocess data for GPU optimization.

        Converts to float32 and ensures memory-efficient layout.
        This improves performance even on CPU by reducing memory footprint.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target variable

        Returns
        -------
        tuple
            (X_preprocessed, y_preprocessed) as optimized arrays/series
        """
        # Convert to float32 (saves 50% memory vs float64)
        X_preprocessed = X.astype(np.float32, copy=False)
        y_preprocessed = y.astype(np.float32, copy=False)

        return X_preprocessed, y_preprocessed

    def _save_training_inputs(self, X, y, path: str) -> None:
        """
        Save training inputs to disk.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Feature matrix
        y : pd.Series or np.ndarray
            Target variable
        path : str
            Path to save inputs
        """
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        # Convert to DataFrame if numpy array
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        if isinstance(y, np.ndarray):
            y = pd.Series(y)

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

        Notes
        -----
        If model was trained on GPU but input data is on CPU, XGBoost will
        automatically fall back to CPU inference. This is expected behavior and
        device mismatch warnings are suppressed as the fallback is intentional.
        For optimal performance with GPU models, ensure training and inference
        use the same device or keep device='cpu' for CPU-only inference.
        """
        if not self._is_trained:
            raise ValueError("Model must be trained before prediction")

        # Suppress XGBoost device mismatch warnings
        # When GPU model runs on CPU data, XGBoost correctly falls back to CPU
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
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
