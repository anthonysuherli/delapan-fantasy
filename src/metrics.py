import numpy as np
import pandas as pd
from typing import Dict, Optional, List


def calculate_mape(actual: pd.Series, predicted: pd.Series) -> float:
    actual = actual.replace(0, np.nan)
    mask = ~(actual.isna() | predicted.isna())
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100


def calculate_rmse(actual: pd.Series, predicted: pd.Series) -> float:
    mask = ~(actual.isna() | predicted.isna())
    if mask.sum() == 0:
        return np.nan
    return np.sqrt(np.mean((actual[mask] - predicted[mask]) ** 2))


def calculate_mae(actual: pd.Series, predicted: pd.Series) -> float:
    mask = ~(actual.isna() | predicted.isna())
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs(actual[mask] - predicted[mask]))


def calculate_correlation(actual: pd.Series, predicted: pd.Series) -> float:
    mask = ~(actual.isna() | predicted.isna())
    if mask.sum() < 2:
        return np.nan
    return actual[mask].corr(predicted[mask])


def calculate_r2(actual: pd.Series, predicted: pd.Series) -> float:
    mask = ~(actual.isna() | predicted.isna())
    if mask.sum() < 2:
        return np.nan
    ss_res = np.sum((actual[mask] - predicted[mask]) ** 2)
    ss_tot = np.sum((actual[mask] - actual[mask].mean()) ** 2)
    return 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan


def evaluate_predictions(
    actual: pd.Series,
    predicted: pd.Series,
    metrics: Optional[List[str]] = None
) -> Dict[str, float]:
    if metrics is None:
        metrics = ['mape', 'rmse', 'mae', 'correlation', 'r2']

    results = {}

    if 'mape' in metrics:
        results['mape'] = calculate_mape(actual, predicted)
    if 'rmse' in metrics:
        results['rmse'] = calculate_rmse(actual, predicted)
    if 'mae' in metrics:
        results['mae'] = calculate_mae(actual, predicted)
    if 'correlation' in metrics:
        results['correlation'] = calculate_correlation(actual, predicted)
    if 'r2' in metrics:
        results['r2'] = calculate_r2(actual, predicted)

    return results
