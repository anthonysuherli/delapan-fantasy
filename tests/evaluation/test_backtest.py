import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.evaluation.backtest import (
    WalkForwardValidator,
    BacktestRunner,
    SlateBacktester
)


class MockModel:

    def fit(self, X, y):
        self.coef = 1.0
        return self

    def predict(self, X):
        return X.sum(axis=1).values


class TestWalkForwardValidator:

    def test_basic_split(self):
        dates = pd.date_range('2024-01-01', periods=10, freq='D')
        data = pd.DataFrame({
            'date': dates,
            'value': range(10)
        })

        validator = WalkForwardValidator(
            train_window=5,
            test_window=1,
            step_size=1
        )
        splits = validator.split(data, 'date')

        assert len(splits) > 0
        train, test = splits[0]
        assert len(train) == 5
        assert len(test) == 1

    def test_step_size(self):
        dates = pd.date_range('2024-01-01', periods=20, freq='D')
        data = pd.DataFrame({
            'date': dates,
            'value': range(20)
        })

        validator = WalkForwardValidator(
            train_window=5,
            test_window=1,
            step_size=2
        )
        splits = validator.split(data, 'date')

        assert len(splits) > 0

    def test_no_overlap(self):
        dates = pd.date_range('2024-01-01', periods=15, freq='D')
        data = pd.DataFrame({
            'date': dates,
            'value': range(15)
        })

        validator = WalkForwardValidator(
            train_window=5,
            test_window=2,
            step_size=1
        )
        splits = validator.split(data, 'date')

        for train, test in splits:
            train_dates = set(train['date'])
            test_dates = set(test['date'])
            assert len(train_dates.intersection(test_dates)) == 0

    def test_chronological_order(self):
        dates = pd.date_range('2024-01-01', periods=15, freq='D')
        data = pd.DataFrame({
            'date': dates,
            'value': range(15)
        })

        validator = WalkForwardValidator(
            train_window=5,
            test_window=1,
            step_size=1
        )
        splits = validator.split(data, 'date')

        for train, test in splits:
            assert train['date'].max() < test['date'].min()

    def test_insufficient_data(self):
        dates = pd.date_range('2024-01-01', periods=3, freq='D')
        data = pd.DataFrame({
            'date': dates,
            'value': range(3)
        })

        validator = WalkForwardValidator(
            train_window=5,
            test_window=1,
            step_size=1
        )
        splits = validator.split(data, 'date')

        assert len(splits) == 0


class TestBacktestRunner:

    @pytest.fixture
    def mock_storage(self):
        class MockStorage:
            def load_data(self, data_type, start_date, end_date):
                return pd.DataFrame()
        return MockStorage()

    def test_initialization(self, mock_storage):
        runner = BacktestRunner(
            model_fn=lambda: MockModel(),
            storage=mock_storage
        )
        assert runner.model_fn is not None
        assert runner.storage is not None
        assert runner.validator is not None
        assert isinstance(runner.results, list)

    def test_get_results_empty(self, mock_storage):
        runner = BacktestRunner(
            model_fn=lambda: MockModel(),
            storage=mock_storage
        )
        results = runner.get_results()
        assert isinstance(results, pd.DataFrame)
        assert len(results) == 0

    def test_run_with_mock_data(self, mock_storage):
        dates = pd.date_range('2024-01-01', periods=15, freq='D')
        data = pd.DataFrame({
            'date': dates,
            'feature1': np.random.randn(15),
            'feature2': np.random.randn(15),
            'fantasyPoints': np.random.uniform(10, 50, 15)
        })

        class CustomBacktestRunner(BacktestRunner):
            def _load_data(self, start_date, end_date, **kwargs):
                return data

        runner = CustomBacktestRunner(
            model_fn=lambda: MockModel(),
            storage=mock_storage,
            validator=WalkForwardValidator(
                train_window=5,
                test_window=1,
                step_size=1
            )
        )

        results = runner.run(
            start_date='20240101',
            end_date='20240115',
            feature_columns=['feature1', 'feature2'],
            target_column='fantasyPoints'
        )

        assert 'avg_mape' in results
        assert 'avg_rmse' in results
        assert 'num_folds' in results

    def test_aggregate_results(self, mock_storage):
        runner = BacktestRunner(
            model_fn=lambda: MockModel(),
            storage=mock_storage
        )

        runner.results = [
            {'mape': 25.0, 'rmse': 5.0, 'correlation': 0.8},
            {'mape': 30.0, 'rmse': 6.0, 'correlation': 0.75}
        ]

        agg = runner._aggregate_results()

        assert agg['num_folds'] == 2
        assert agg['avg_mape'] == 27.5
        assert agg['avg_rmse'] == 5.5
        assert agg['avg_correlation'] == 0.775


class TestSlateBacktester:

    @pytest.fixture
    def mock_storage(self):
        class MockStorage:
            pass
        return MockStorage()

    def test_initialization(self, mock_storage):
        backtester = SlateBacktester(mock_storage)
        assert backtester.storage is not None
        assert isinstance(backtester.results, list)

    def test_backtest_slate(self, mock_storage):
        backtester = SlateBacktester(mock_storage)

        projections = pd.DataFrame({
            'playerID': ['A', 'B', 'C'],
            'fantasyPoints_proj': [30.0, 25.0, 20.0]
        })

        actuals = pd.DataFrame({
            'playerID': ['A', 'B', 'C'],
            'fantasyPoints_actual': [32.0, 24.0, 22.0]
        })

        result = backtester.backtest_slate(
            date='20240101',
            projections=projections,
            actuals=actuals
        )

        assert result['date'] == '20240101'
        assert result['num_players'] == 3
        assert 'projection_mape' in result
        assert 'projection_rmse' in result
        assert 'projection_correlation' in result

    def test_get_summary(self, mock_storage):
        backtester = SlateBacktester(mock_storage)

        backtester.results = [
            {
                'date': '20240101',
                'projection_mape': 25.0,
                'projection_rmse': 5.0,
                'projection_correlation': 0.8
            },
            {
                'date': '20240102',
                'projection_mape': 30.0,
                'projection_rmse': 6.0,
                'projection_correlation': 0.75
            }
        ]

        summary = backtester.get_summary()

        assert summary['num_slates'] == 2
        assert summary['avg_projection_mape'] == 27.5
        assert summary['avg_projection_rmse'] == 5.5
        assert summary['avg_projection_correlation'] == 0.775

    def test_empty_summary(self, mock_storage):
        backtester = SlateBacktester(mock_storage)
        summary = backtester.get_summary()
        assert summary == {}
