import pytest
import numpy as np
import pandas as pd
from src.evaluation.metrics import (
    calculate_mape,
    calculate_rmse,
    calculate_correlation,
    evaluate_predictions,
    MAPECalculator,
    RMSECalculator,
    CorrelationCalculator
)


class TestMAPE:

    def test_perfect_prediction(self):
        actual = pd.Series([10, 20, 30, 40])
        predicted = pd.Series([10, 20, 30, 40])
        mape = calculate_mape(actual, predicted)
        assert mape == 0.0

    def test_basic_error(self):
        actual = pd.Series([10, 20, 30, 40])
        predicted = pd.Series([11, 22, 33, 44])
        mape = calculate_mape(actual, predicted)
        assert 9 < mape < 11

    def test_handles_zero_actual(self):
        actual = pd.Series([0, 20, 30, 40])
        predicted = pd.Series([10, 20, 30, 40])
        mape = calculate_mape(actual, predicted)
        assert not np.isnan(mape)

    def test_all_nan(self):
        actual = pd.Series([np.nan, np.nan])
        predicted = pd.Series([10, 20])
        mape = calculate_mape(actual, predicted)
        assert np.isnan(mape)

    def test_calculator_interface(self):
        calculator = MAPECalculator()
        actual = pd.Series([10, 20, 30])
        predicted = pd.Series([11, 22, 33])
        result = calculator.calculate(actual, predicted)
        assert result > 0


class TestRMSE:

    def test_perfect_prediction(self):
        actual = pd.Series([10, 20, 30, 40])
        predicted = pd.Series([10, 20, 30, 40])
        rmse = calculate_rmse(actual, predicted)
        assert rmse == 0.0

    def test_basic_error(self):
        actual = pd.Series([10, 20, 30, 40])
        predicted = pd.Series([12, 22, 32, 42])
        rmse = calculate_rmse(actual, predicted)
        assert rmse == 2.0

    def test_handles_nan(self):
        actual = pd.Series([10, np.nan, 30, 40])
        predicted = pd.Series([12, 22, 32, 42])
        rmse = calculate_rmse(actual, predicted)
        assert not np.isnan(rmse)

    def test_calculator_interface(self):
        calculator = RMSECalculator()
        actual = pd.Series([10, 20, 30])
        predicted = pd.Series([12, 22, 32])
        result = calculator.calculate(actual, predicted)
        assert result == 2.0


class TestCorrelation:

    def test_perfect_correlation(self):
        actual = pd.Series([10, 20, 30, 40])
        predicted = pd.Series([10, 20, 30, 40])
        corr = calculate_correlation(actual, predicted)
        assert corr == 1.0

    def test_negative_correlation(self):
        actual = pd.Series([10, 20, 30, 40])
        predicted = pd.Series([40, 30, 20, 10])
        corr = calculate_correlation(actual, predicted)
        assert corr == -1.0

    def test_handles_nan(self):
        actual = pd.Series([10, 20, np.nan, 40])
        predicted = pd.Series([10, 20, 30, 40])
        corr = calculate_correlation(actual, predicted)
        assert not np.isnan(corr)

    def test_insufficient_data(self):
        actual = pd.Series([10])
        predicted = pd.Series([10])
        corr = calculate_correlation(actual, predicted)
        assert np.isnan(corr)

    def test_calculator_interface(self):
        calculator = CorrelationCalculator()
        actual = pd.Series([10, 20, 30])
        predicted = pd.Series([10, 20, 30])
        result = calculator.calculate(actual, predicted)
        assert result == 1.0


class TestEvaluatePredictions:

    def test_default_metrics(self):
        actual = pd.Series([10, 20, 30, 40])
        predicted = pd.Series([11, 22, 33, 44])
        results = evaluate_predictions(actual, predicted)

        assert 'mape' in results
        assert 'rmse' in results
        assert 'correlation' in results
        assert len(results) == 3

    def test_custom_metrics(self):
        actual = pd.Series([10, 20, 30, 40])
        predicted = pd.Series([11, 22, 33, 44])
        results = evaluate_predictions(
            actual,
            predicted,
            metrics=['mape', 'mae', 'r2']
        )

        assert 'mape' in results
        assert 'mae' in results
        assert 'r2' in results
        assert 'rmse' not in results

    def test_mae_calculation(self):
        actual = pd.Series([10, 20, 30, 40])
        predicted = pd.Series([12, 22, 32, 42])
        results = evaluate_predictions(actual, predicted, metrics=['mae'])
        assert results['mae'] == 2.0

    def test_r2_calculation(self):
        actual = pd.Series([10, 20, 30, 40])
        predicted = pd.Series([10, 20, 30, 40])
        results = evaluate_predictions(actual, predicted, metrics=['r2'])
        assert results['r2'] == 1.0

    def test_handles_empty_series(self):
        actual = pd.Series([])
        predicted = pd.Series([])
        results = evaluate_predictions(actual, predicted)

        assert np.isnan(results['mape'])
        assert np.isnan(results['rmse'])
        assert np.isnan(results['correlation'])
