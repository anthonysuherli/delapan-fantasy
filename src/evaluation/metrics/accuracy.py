import numpy as np
from .base import BaseMetric


class MAPEMetric(BaseMetric):
    """Mean Absolute Percentage Error"""

    def __init__(self, remove_outliers: bool = False, outlier_method: str = 'iqr', outlier_threshold: float = 1.5):
        """
        Initialize MAPE metric.

        Args:
            remove_outliers: Whether to remove outliers before calculation
            outlier_method: Method for outlier detection ('iqr', 'zscore', 'percentile')
            outlier_threshold: Threshold for outlier detection
        """
        super().__init__('mape', remove_outliers, outlier_method, outlier_threshold)

    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate MAPE.

        Args:
            y_true: Ground truth values
            y_pred: Predicted values

        Returns:
            MAPE value (as percentage, e.g., 15.2 for 15.2%)

        Raises:
            ValueError: If arrays have mismatched shapes
        """
        if len(y_true) != len(y_pred):
            raise ValueError(f"Array length mismatch: {len(y_true)} vs {len(y_pred)}")

        mask = y_true != 0
        if not mask.any():
            return 0.0

        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


class RMSEMetric(BaseMetric):
    """Root Mean Squared Error"""

    def __init__(self, remove_outliers: bool = False, outlier_method: str = 'iqr', outlier_threshold: float = 1.5):
        """
        Initialize RMSE metric.

        Args:
            remove_outliers: Whether to remove outliers before calculation
            outlier_method: Method for outlier detection ('iqr', 'zscore', 'percentile')
            outlier_threshold: Threshold for outlier detection
        """
        super().__init__('rmse', remove_outliers, outlier_method, outlier_threshold)

    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate RMSE.

        Args:
            y_true: Ground truth values
            y_pred: Predicted values

        Returns:
            RMSE value

        Raises:
            ValueError: If arrays have mismatched shapes
        """
        if len(y_true) != len(y_pred):
            raise ValueError(f"Array length mismatch: {len(y_true)} vs {len(y_pred)}")

        return np.sqrt(np.mean((y_true - y_pred) ** 2))


class MAEMetric(BaseMetric):
    """Mean Absolute Error"""

    def __init__(self, remove_outliers: bool = False, outlier_method: str = 'iqr', outlier_threshold: float = 1.5):
        """
        Initialize MAE metric.

        Args:
            remove_outliers: Whether to remove outliers before calculation
            outlier_method: Method for outlier detection ('iqr', 'zscore', 'percentile')
            outlier_threshold: Threshold for outlier detection
        """
        super().__init__('mae', remove_outliers, outlier_method, outlier_threshold)

    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate MAE.

        Args:
            y_true: Ground truth values
            y_pred: Predicted values

        Returns:
            MAE value

        Raises:
            ValueError: If arrays have mismatched shapes
        """
        if len(y_true) != len(y_pred):
            raise ValueError(f"Array length mismatch: {len(y_true)} vs {len(y_pred)}")

        return np.mean(np.abs(y_true - y_pred))


class CorrelationMetric(BaseMetric):
    """Pearson Correlation Coefficient"""

    def __init__(self, remove_outliers: bool = False, outlier_method: str = 'iqr', outlier_threshold: float = 1.5):
        """
        Initialize Correlation metric.

        Args:
            remove_outliers: Whether to remove outliers before calculation
            outlier_method: Method for outlier detection ('iqr', 'zscore', 'percentile')
            outlier_threshold: Threshold for outlier detection
        """
        super().__init__('correlation', remove_outliers, outlier_method, outlier_threshold)

    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Pearson correlation.

        Args:
            y_true: Ground truth values
            y_pred: Predicted values

        Returns:
            Correlation coefficient between -1 and 1

        Raises:
            ValueError: If arrays have mismatched shapes
        """
        if len(y_true) != len(y_pred):
            raise ValueError(f"Array length mismatch: {len(y_true)} vs {len(y_pred)}")

        return np.corrcoef(y_true, y_pred)[0, 1]


class CappedMAPEMetric(BaseMetric):
    """Capped Mean Absolute Percentage Error (cMAPE)

    Uses denominator = max(actual, cap) to avoid inflation when actuals are near zero.
    Returns percentage.
    """

    def __init__(self, cap: float = 8.0, remove_outliers: bool = False, outlier_method: str = 'iqr', outlier_threshold: float = 1.5):
        super().__init__('cmape', remove_outliers, outlier_method, outlier_threshold)
        self.cap = float(cap)

    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        if len(y_true) != len(y_pred):
            raise ValueError(f"Array length mismatch: {len(y_true)} vs {len(y_pred)}")

        if len(y_true) == 0:
            return np.nan

        denom = np.maximum(np.asarray(y_true, dtype=float), self.cap)
        abs_err = np.abs(np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float))
        return float(np.mean(abs_err / denom) * 100.0)


class SMAPEMetric(BaseMetric):
    """Symmetric Mean Absolute Percentage Error (SMAPE)

    2*|pred-true|/(|pred|+|true|+eps). Returns percentage.
    """

    def __init__(self, eps: float = 1.0, remove_outliers: bool = False, outlier_method: str = 'iqr', outlier_threshold: float = 1.5):
        super().__init__('smape', remove_outliers, outlier_method, outlier_threshold)
        self.eps = float(eps)

    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        if len(y_true) != len(y_pred):
            raise ValueError(f"Array length mismatch: {len(y_true)} vs {len(y_pred)}")

        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        num = np.abs(y_pred - y_true)
        denom = np.maximum(np.abs(y_pred) + np.abs(y_true), self.eps)
        return float(np.mean(2.0 * num / denom) * 100.0)


class WMAPEMetric(BaseMetric):
    """Weighted Mean Absolute Percentage Error (WMAPE)

    sum(weights * |pred-true|) / sum(weights). Returns percentage.
    Default weights are max(actual, 1.0) (i.e., weight by actual fpts).
    """

    def __init__(self, remove_outliers: bool = False, outlier_method: str = 'iqr', outlier_threshold: float = 1.5):
        super().__init__('wmape', remove_outliers, outlier_method, outlier_threshold)

    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray = None) -> float:
        if len(y_true) != len(y_pred):
            raise ValueError(f"Array length mismatch: {len(y_true)} vs {len(y_pred)}")

        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        abs_err = np.abs(y_pred - y_true)

        if weights is None:
            weights = np.maximum(y_true, 1.0)
        else:
            weights = np.asarray(weights, dtype=float)

        total_weight = np.sum(weights)
        if total_weight <= 0:
            return np.nan

        return float(np.sum(weights * abs_err) / total_weight * 100.0)