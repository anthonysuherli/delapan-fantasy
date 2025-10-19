from abc import ABC, abstractmethod
import numpy as np
from typing import Any, Tuple, Optional
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest


class BaseMetric(ABC):
    """Abstract base for evaluation metrics"""

    def __init__(self, name: str, remove_outliers: bool = False, outlier_method: str = 'iqr', outlier_threshold: float = 1.5):
        """
        Initialize metric.

        Args:
            name: Unique name for this metric
            remove_outliers: Whether to remove outliers before calculation
            outlier_method: Method for outlier detection ('iqr', 'zscore', 'percentile')
            outlier_threshold: Threshold for outlier detection (IQR multiplier, z-score, or percentile)
        """
        self.name = name
        self.remove_outliers = remove_outliers
        self.outlier_method = outlier_method
        self.outlier_threshold = outlier_threshold

    def _remove_outliers(self, y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Remove outliers from prediction errors using scikit-learn methods.

        Args:
            y_true: Ground truth values
            y_pred: Predicted values

        Returns:
            Tuple of filtered (y_true, y_pred)
        """
        errors = np.abs(y_true - y_pred).reshape(-1, 1)

        if self.outlier_method == 'iqr':
            q1 = np.percentile(errors, 25)
            q3 = np.percentile(errors, 75)
            iqr = q3 - q1
            lower_bound = q1 - self.outlier_threshold * iqr
            upper_bound = q3 + self.outlier_threshold * iqr
            mask = (errors.ravel() >= lower_bound) & (errors.ravel() <= upper_bound)

        elif self.outlier_method == 'zscore':
            mean = np.mean(errors)
            std = np.std(errors)
            z_scores = np.abs((errors - mean) / (std + 1e-8))
            mask = z_scores.ravel() <= self.outlier_threshold

        elif self.outlier_method == 'percentile':
            threshold_value = np.percentile(errors, self.outlier_threshold)
            mask = errors.ravel() <= threshold_value

        elif self.outlier_method == 'isolation_forest':
            # Use IsolationForest for more robust outlier detection
            contamination = 0.1  # Assume ~10% outliers
            clf = IsolationForest(contamination=contamination, random_state=42)
            predictions = clf.fit_predict(errors)
            mask = predictions == 1  # 1 = inlier, -1 = outlier

        elif self.outlier_method == 'elliptic_envelope':
            # Use Elliptic Envelope for robust Mahalanobis distance-based detection
            contamination = 0.1  # Assume ~10% outliers
            try:
                clf = EllipticEnvelope(contamination=contamination, random_state=42)
                predictions = clf.fit_predict(errors)
                mask = predictions == 1  # 1 = inlier, -1 = outlier
            except Exception:
                # Fallback to IQR if covariance estimation fails
                q1 = np.percentile(errors, 25)
                q3 = np.percentile(errors, 75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                mask = (errors.ravel() >= lower_bound) & (errors.ravel() <= upper_bound)

        else:
            raise ValueError(f"Unknown outlier method: {self.outlier_method}. "
                           f"Supported methods: 'iqr', 'zscore', 'percentile', 'isolation_forest', 'elliptic_envelope'")

        return y_true[mask], y_pred[mask]

    @abstractmethod
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate metric value.

        Args:
            y_true: Ground truth values
            y_pred: Predicted values

        Returns:
            Metric value

        Raises:
            ValueError: If arrays have mismatched shapes
        """
        pass

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Allow metric to be called as a function"""
        if self.remove_outliers:
            y_true, y_pred = self._remove_outliers(y_true, y_pred)
        return self.calculate(y_true, y_pred)
