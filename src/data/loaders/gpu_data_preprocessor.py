"""GPU-optimized data preprocessor for efficient XGBoost training.

This module prepares data in CPU memory for fast GPU training by:
1. Converting dtypes to optimal sizes (float32 instead of float64)
2. Pre-allocating contiguous memory layouts
3. Normalizing data ranges
4. Caching preprocessed data for repeated access
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class GPUDataPreprocessor:
    """Preprocess data for optimal GPU training performance."""

    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize GPU data preprocessor.

        Parameters
        ----------
        cache_dir : Optional[Path]
            Directory to cache preprocessed data. If None, no caching.
        """
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def preprocess_features(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        optimize_dtypes: bool = True,
        normalize: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Preprocess feature matrix and target for GPU training.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix (n_samples, n_features)
        y : Optional[pd.Series]
            Target variable (n_samples,)
        optimize_dtypes : bool, optional
            Convert float64 to float32 for GPU memory efficiency. Default: True
        normalize : bool, optional
            Normalize features to [0, 1] range. Default: False

        Returns
        -------
        Tuple[np.ndarray, Optional[np.ndarray]]
            (X_preprocessed, y_preprocessed) as numpy arrays optimized for GPU

        Notes
        -----
        - Converts to float32 by default (saves 50% GPU memory vs float64)
        - Ensures C-contiguous memory layout for efficient GPU transfer
        - Handles NaN values (fills with 0)
        - All operations maintain numerical stability
        """
        # Convert to numpy for GPU-friendly format
        X_np = X.values.copy() if isinstance(X, pd.DataFrame) else X.copy()

        # Fill NaN values before dtype conversion
        X_np = np.nan_to_num(X_np, nan=0.0, posinf=0.0, neginf=0.0)

        # Optimize dtype for GPU memory efficiency
        if optimize_dtypes:
            X_np = X_np.astype(np.float32, copy=False)
        else:
            X_np = X_np.astype(np.float64, copy=False)

        # Ensure C-contiguous memory layout for GPU transfer
        X_np = np.ascontiguousarray(X_np)

        # Normalize features if requested (improves GPU numerical stability)
        if normalize:
            X_np = self._normalize_features(X_np)

        # Process target variable
        y_np = None
        if y is not None:
            y_np = y.values.copy() if isinstance(y, pd.Series) else y.copy()
            y_np = np.nan_to_num(y_np, nan=0.0, posinf=0.0, neginf=0.0)
            y_np = y_np.astype(np.float32, copy=False)
            y_np = np.ascontiguousarray(y_np)

        return X_np, y_np

    def batch_preprocess(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        batch_size: int = 10000,
        optimize_dtypes: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Preprocess large datasets in batches to manage memory.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix
        y : Optional[pd.Series]
            Target variable
        batch_size : int, optional
            Number of samples per batch. Default: 10000
        optimize_dtypes : bool, optional
            Convert to float32. Default: True

        Returns
        -------
        Tuple[np.ndarray, Optional[np.ndarray]]
            Concatenated preprocessed arrays
        """
        n_samples = len(X)
        X_preprocessed = []
        y_preprocessed = []

        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            X_batch = X.iloc[i:end_idx]
            y_batch = y.iloc[i:end_idx] if y is not None else None

            X_batch_proc, y_batch_proc = self.preprocess_features(
                X_batch, y_batch, optimize_dtypes=optimize_dtypes
            )
            X_preprocessed.append(X_batch_proc)
            if y_batch_proc is not None:
                y_preprocessed.append(y_batch_proc)

        X_result = np.vstack(X_preprocessed)
        y_result = np.concatenate(y_preprocessed) if y_preprocessed else None

        return X_result, y_result

    def cache_data(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray],
        cache_key: str
    ) -> Path:
        """
        Cache preprocessed data for repeated access.

        Parameters
        ----------
        X : np.ndarray
            Preprocessed feature matrix
        y : Optional[np.ndarray]
            Preprocessed target variable
        cache_key : str
            Unique identifier for cache file

        Returns
        -------
        Path
            Path to cached file
        """
        if not self.cache_dir:
            raise ValueError("cache_dir not configured")

        cache_path = self.cache_dir / f"{cache_key}.npz"

        if y is not None:
            np.savez_compressed(cache_path, X=X, y=y)
        else:
            np.savez_compressed(cache_path, X=X)

        logger.info(f"Data cached: {cache_path}")
        return cache_path

    def load_cached_data(self, cache_key: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Load cached preprocessed data.

        Parameters
        ----------
        cache_key : str
            Unique identifier for cache file

        Returns
        -------
        Tuple[np.ndarray, Optional[np.ndarray]]
            (X, y) loaded from cache

        Raises
        ------
        FileNotFoundError
            If cache file not found
        """
        if not self.cache_dir:
            raise ValueError("cache_dir not configured")

        cache_path = self.cache_dir / f"{cache_key}.npz"
        if not cache_path.exists():
            raise FileNotFoundError(f"Cache not found: {cache_path}")

        data = np.load(cache_path)
        X = data['X']
        y = data.get('y', None)

        logger.info(f"Data loaded from cache: {cache_path}")
        return X, y

    def is_cached(self, cache_key: str) -> bool:
        """
        Check if data is cached.

        Parameters
        ----------
        cache_key : str
            Unique identifier for cache file

        Returns
        -------
        bool
            True if cache exists and is valid
        """
        if not self.cache_dir:
            return False
        cache_path = self.cache_dir / f"{cache_key}.npz"
        return cache_path.exists()

    @staticmethod
    def _normalize_features(X: np.ndarray) -> np.ndarray:
        """
        Normalize features to [0, 1] range per column.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix

        Returns
        -------
        np.ndarray
            Normalized features

        Notes
        -----
        Handles edge cases: constant columns, extreme values
        """
        X_norm = X.copy()

        for j in range(X.shape[1]):
            col = X_norm[:, j]
            col_min = np.nanmin(col)
            col_max = np.nanmax(col)
            col_range = col_max - col_min

            # Skip normalization for constant columns
            if col_range > 1e-10:
                X_norm[:, j] = (col - col_min) / col_range
            else:
                # Constant column, set to 0.5
                X_norm[:, j] = 0.5

        return X_norm

    def get_memory_stats(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Get memory usage statistics for preprocessed data.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        y : Optional[np.ndarray]
            Target variable

        Returns
        -------
        Dict[str, float]
            Memory statistics in MB
        """
        X_bytes = X.nbytes
        y_bytes = y.nbytes if y is not None else 0
        total_bytes = X_bytes + y_bytes

        return {
            'X_mb': X_bytes / (1024 * 1024),
            'y_mb': y_bytes / (1024 * 1024),
            'total_mb': total_bytes / (1024 * 1024),
            'X_dtype': str(X.dtype),
            'y_dtype': str(y.dtype) if y is not None else None
        }


class GPUDataPipeline:
    """End-to-end GPU data pipeline with preprocessing and caching."""

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        batch_size: int = 10000
    ):
        """
        Initialize GPU data pipeline.

        Parameters
        ----------
        cache_dir : Optional[Path]
            Directory for caching preprocessed data
        batch_size : int, optional
            Batch size for processing large datasets. Default: 10000
        """
        self.preprocessor = GPUDataPreprocessor(cache_dir=cache_dir)
        self.batch_size = batch_size

    def prepare_training_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cache_key: Optional[str] = None,
        use_cache: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data with optional caching.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target variable
        cache_key : Optional[str]
            Cache identifier. If None, caching disabled.
        use_cache : bool, optional
            Whether to use cached data if available. Default: True

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (X_preprocessed, y_preprocessed) ready for GPU training

        Notes
        -----
        If cache_key provided and data cached:
        - Loads from cache (skip preprocessing)
        If cache_key provided and data not cached:
        - Preprocesses data and saves to cache
        """
        # Try loading from cache
        if cache_key and use_cache and self.preprocessor.is_cached(cache_key):
            logger.info(f"Loading cached training data: {cache_key}")
            return self.preprocessor.load_cached_data(cache_key)

        # Preprocess data
        logger.info(f"Preprocessing {len(X)} samples for GPU training")
        X_np, y_np = self.preprocessor.batch_preprocess(
            X, y, batch_size=self.batch_size, optimize_dtypes=True
        )

        # Cache if key provided
        if cache_key:
            self.preprocessor.cache_data(X_np, y_np, cache_key)

        # Log memory usage
        stats = self.preprocessor.get_memory_stats(X_np, y_np)
        logger.info(f"Preprocessed data: {stats['total_mb']:.2f} MB "
                   f"(X: {stats['X_mb']:.2f} MB, y: {stats['y_mb']:.2f} MB)")

        return X_np, y_np

    def prepare_inference_data(
        self,
        X: pd.DataFrame,
        cache_key: Optional[str] = None,
        use_cache: bool = True
    ) -> np.ndarray:
        """
        Prepare inference data with optional caching.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix
        cache_key : Optional[str]
            Cache identifier. If None, caching disabled.
        use_cache : bool, optional
            Whether to use cached data if available. Default: True

        Returns
        -------
        np.ndarray
            Preprocessed features ready for GPU inference
        """
        # Try loading from cache
        if cache_key and use_cache and self.preprocessor.is_cached(cache_key):
            logger.info(f"Loading cached inference data: {cache_key}")
            X_np, _ = self.preprocessor.load_cached_data(cache_key)
            return X_np

        # Preprocess data
        logger.info(f"Preprocessing {len(X)} samples for GPU inference")
        X_np, _ = self.preprocessor.preprocess_features(
            X, y=None, optimize_dtypes=True
        )

        # Cache if key provided
        if cache_key:
            self.preprocessor.cache_data(X_np, None, cache_key)

        stats = self.preprocessor.get_memory_stats(X_np)
        logger.info(f"Preprocessed inference data: {stats['X_mb']:.2f} MB")

        return X_np
