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
