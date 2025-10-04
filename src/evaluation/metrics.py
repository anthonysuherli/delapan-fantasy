import numpy as np
import pandas as pd
from typing import Dict, Optional
from src.evaluation.base import MetricCalculator


class MAPECalculator(MetricCalculator):

    def calculate(self, actual: pd.Series, predicted: pd.Series) -> float:
        actual = actual.replace(0, np.nan)
        mask = ~(actual.isna() | predicted.isna())
        if mask.sum() == 0:
            return np.nan
        return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100


class RMSECalculator(MetricCalculator):

    def calculate(self, actual: pd.Series, predicted: pd.Series) -> float:
        mask = ~(actual.isna() | predicted.isna())
        if mask.sum() == 0:
            return np.nan
        return np.sqrt(np.mean((actual[mask] - predicted[mask]) ** 2))


class CorrelationCalculator(MetricCalculator):

    def calculate(self, actual: pd.Series, predicted: pd.Series) -> float:
        mask = ~(actual.isna() | predicted.isna())
        if mask.sum() < 2:
            return np.nan
        return actual[mask].corr(predicted[mask])


def calculate_mape(actual: pd.Series, predicted: pd.Series) -> float:
    calculator = MAPECalculator()
    return calculator.calculate(actual, predicted)


def calculate_rmse(actual: pd.Series, predicted: pd.Series) -> float:
    calculator = RMSECalculator()
    return calculator.calculate(actual, predicted)


def calculate_correlation(actual: pd.Series, predicted: pd.Series) -> float:
    calculator = CorrelationCalculator()
    return calculator.calculate(actual, predicted)


def evaluate_predictions(
    actual: pd.Series,
    predicted: pd.Series,
    metrics: Optional[list[str]] = None
) -> Dict[str, float]:
    if metrics is None:
        metrics = ['mape', 'rmse', 'correlation']

    results = {}

    if 'mape' in metrics:
        results['mape'] = calculate_mape(actual, predicted)

    if 'rmse' in metrics:
        results['rmse'] = calculate_rmse(actual, predicted)

    if 'correlation' in metrics:
        results['correlation'] = calculate_correlation(actual, predicted)

    if 'mae' in metrics:
        mask = ~(actual.isna() | predicted.isna())
        if mask.sum() > 0:
            results['mae'] = np.mean(np.abs(actual[mask] - predicted[mask]))
        else:
            results['mae'] = np.nan

    if 'r2' in metrics:
        mask = ~(actual.isna() | predicted.isna())
        if mask.sum() > 1:
            ss_res = np.sum((actual[mask] - predicted[mask]) ** 2)
            ss_tot = np.sum((actual[mask] - actual[mask].mean()) ** 2)
            results['r2'] = 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan
        else:
            results['r2'] = np.nan

    return results
