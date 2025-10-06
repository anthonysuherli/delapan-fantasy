import numpy as np
import pandas as pd
from typing import Dict, Optional, List
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


def simulate_cash_contest(
    lineup_scores: List[float],
    field_mean: float = 280.0,
    field_std: float = 25.0,
    payout_threshold: float = 0.45,
    entry_fee: float = 10.0,
    payout: float = 18.0,
    num_simulations: int = 10000
) -> Dict[str, float]:
    if not lineup_scores:
        return {
            'win_rate': 0.0,
            'roi': 0.0,
            'profit': 0.0,
            'avg_percentile': 0.0
        }

    lineup_scores = np.array(lineup_scores)

    simulated_fields = np.random.normal(field_mean, field_std, (num_simulations, 1))

    lineup_scores_expanded = lineup_scores.reshape(1, -1)

    percentiles = np.sum(lineup_scores_expanded > simulated_fields, axis=0) / num_simulations

    cashes = (percentiles > payout_threshold).astype(int)

    win_rate = cashes.mean()

    total_entry_fees = len(lineup_scores) * entry_fee
    total_winnings = cashes.sum() * payout
    profit = total_winnings - total_entry_fees
    roi = (profit / total_entry_fees) * 100 if total_entry_fees > 0 else 0.0

    return {
        'win_rate': round(win_rate, 4),
        'roi': round(roi, 2),
        'profit': round(profit, 2),
        'avg_percentile': round(percentiles.mean(), 4),
        'total_winnings': round(total_winnings, 2),
        'total_entry_fees': round(total_entry_fees, 2)
    }


def simulate_gpp_contest(
    lineup_scores: List[float],
    field_size: int = 100000,
    entry_fee: float = 3.0
) -> Dict[str, float]:
    if not lineup_scores:
        return {
            'best_finish': 0,
            'total_winnings': 0.0,
            'roi': 0.0,
            'avg_finish': 0
        }

    payout_structure = {
        1: 20000,
        2: 10000,
        3: 5000,
        4: 3000,
        5: 2000,
        6: 1500,
        7: 1200,
        8: 1000,
        9: 900,
        10: 800
    }

    for rank in range(11, 21):
        payout_structure[rank] = 500

    for rank in range(21, 51):
        payout_structure[rank] = 200

    for rank in range(51, 101):
        payout_structure[rank] = 100

    for rank in range(101, 201):
        payout_structure[rank] = 50

    for rank in range(201, 501):
        payout_structure[rank] = 25

    for rank in range(501, 1001):
        payout_structure[rank] = 10

    for rank in range(1001, 2001):
        payout_structure[rank] = 5

    lineup_scores = np.array(lineup_scores)

    percentiles = np.random.uniform(0.5, 0.99, len(lineup_scores))

    finishes = ((1 - percentiles) * field_size).astype(int) + 1

    finishes = np.maximum(1, finishes)
    finishes = np.minimum(field_size, finishes)

    total_winnings = sum(payout_structure.get(finish, 0) for finish in finishes)

    best_finish = int(finishes.min())

    total_entry_fees = len(lineup_scores) * entry_fee
    profit = total_winnings - total_entry_fees
    roi = (profit / total_entry_fees) * 100 if total_entry_fees > 0 else 0.0

    return {
        'best_finish': best_finish,
        'avg_finish': int(finishes.mean()),
        'total_winnings': round(total_winnings, 2),
        'total_entry_fees': round(total_entry_fees, 2),
        'profit': round(profit, 2),
        'roi': round(roi, 2)
    }
