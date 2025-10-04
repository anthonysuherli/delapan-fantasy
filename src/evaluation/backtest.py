import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from src.evaluation.base import BacktestStrategy, ValidationStrategy
from src.evaluation.metrics import evaluate_predictions


class WalkForwardValidator(ValidationStrategy):

    def __init__(
        self,
        train_window: int = 30,
        test_window: int = 1,
        step_size: int = 1
    ):
        self.train_window = train_window
        self.test_window = test_window
        self.step_size = step_size

    def split(
        self,
        data: pd.DataFrame,
        date_column: str = 'date'
    ) -> List[tuple[pd.DataFrame, pd.DataFrame]]:
        data = data.sort_values(date_column).reset_index(drop=True)
        dates = pd.to_datetime(data[date_column])
        unique_dates = sorted(dates.unique())

        splits = []
        start_idx = 0

        while start_idx + self.train_window + self.test_window <= len(unique_dates):
            train_end_idx = start_idx + self.train_window
            test_end_idx = train_end_idx + self.test_window

            train_dates = unique_dates[start_idx:train_end_idx]
            test_dates = unique_dates[train_end_idx:test_end_idx]

            train_mask = dates.isin(train_dates)
            test_mask = dates.isin(test_dates)

            train_data = data[train_mask].copy()
            test_data = data[test_mask].copy()

            splits.append((train_data, test_data))

            start_idx += self.step_size

        return splits


class BacktestRunner(BacktestStrategy):

    def __init__(
        self,
        model_fn: Callable,
        storage,
        validator: Optional[ValidationStrategy] = None
    ):
        self.model_fn = model_fn
        self.storage = storage
        self.validator = validator or WalkForwardValidator()
        self.results = []

    def run(
        self,
        start_date: str,
        end_date: str,
        feature_columns: List[str],
        target_column: str = 'fantasyPoints',
        date_column: str = 'date',
        **kwargs
    ) -> Dict[str, Any]:
        data = self._load_data(start_date, end_date, **kwargs)

        if data.empty:
            return {'error': 'No data available for date range'}

        splits = self.validator.split(data, date_column)

        for idx, (train_data, test_data) in enumerate(splits):
            if train_data.empty or test_data.empty:
                continue

            X_train = train_data[feature_columns]
            y_train = train_data[target_column]
            X_test = test_data[feature_columns]
            y_test = test_data[target_column]

            model = self.model_fn()
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

            metrics = evaluate_predictions(y_test, pd.Series(predictions))

            self.results.append({
                'fold': idx,
                'train_start': train_data[date_column].min(),
                'train_end': train_data[date_column].max(),
                'test_start': test_data[date_column].min(),
                'test_end': test_data[date_column].max(),
                'train_size': len(train_data),
                'test_size': len(test_data),
                **metrics
            })

        return self._aggregate_results()

    def get_results(self) -> pd.DataFrame:
        return pd.DataFrame(self.results)

    def _load_data(
        self,
        start_date: str,
        end_date: str,
        **kwargs
    ) -> pd.DataFrame:
        return pd.DataFrame()

    def _aggregate_results(self) -> Dict[str, Any]:
        if not self.results:
            return {}

        df = pd.DataFrame(self.results)

        aggregated = {
            'num_folds': len(self.results),
            'avg_mape': df['mape'].mean(),
            'std_mape': df['mape'].std(),
            'avg_rmse': df['rmse'].mean(),
            'std_rmse': df['rmse'].std(),
            'avg_correlation': df['correlation'].mean(),
            'std_correlation': df['correlation'].std()
        }

        if 'mae' in df.columns:
            aggregated['avg_mae'] = df['mae'].mean()
            aggregated['std_mae'] = df['mae'].std()

        if 'r2' in df.columns:
            aggregated['avg_r2'] = df['r2'].mean()
            aggregated['std_r2'] = df['r2'].std()

        return aggregated


class SlateBacktester:

    def __init__(self, storage):
        self.storage = storage
        self.results = []

    def backtest_slate(
        self,
        date: str,
        projections: pd.DataFrame,
        actuals: pd.DataFrame,
        lineup_optimizer: Optional[Any] = None
    ) -> Dict[str, Any]:
        merged = projections.merge(
            actuals,
            on='playerID',
            suffixes=('_proj', '_actual')
        )

        projection_metrics = evaluate_predictions(
            merged['fantasyPoints_actual'],
            merged['fantasyPoints_proj']
        )

        result = {
            'date': date,
            'num_players': len(merged),
            'projection_mape': projection_metrics['mape'],
            'projection_rmse': projection_metrics['rmse'],
            'projection_correlation': projection_metrics['correlation']
        }

        if lineup_optimizer is not None:
            lineup = lineup_optimizer.optimize(projections)
            lineup_actual = merged[merged['playerID'].isin(lineup['playerID'])]

            result['lineup_projected_score'] = lineup['fantasyPoints_proj'].sum()
            result['lineup_actual_score'] = lineup_actual['fantasyPoints_actual'].sum()
            result['lineup_error'] = (
                result['lineup_actual_score'] - result['lineup_projected_score']
            )

        self.results.append(result)
        return result

    def run_multi_slate(
        self,
        start_date: str,
        end_date: str,
        projection_fn: Callable,
        lineup_optimizer: Optional[Any] = None
    ) -> pd.DataFrame:
        dates = pd.date_range(start_date, end_date, freq='D')

        for date in dates:
            date_str = date.strftime('%Y%m%d')

            try:
                projections = projection_fn(date_str)
                actuals = self._load_actuals(date_str)

                if projections.empty or actuals.empty:
                    continue

                self.backtest_slate(
                    date_str,
                    projections,
                    actuals,
                    lineup_optimizer
                )

            except Exception as e:
                print(f"Error processing {date_str}: {str(e)}")
                continue

        return pd.DataFrame(self.results)

    def _load_actuals(self, date: str) -> pd.DataFrame:
        return pd.DataFrame()

    def get_summary(self) -> Dict[str, Any]:
        if not self.results:
            return {}

        df = pd.DataFrame(self.results)

        summary = {
            'num_slates': len(df),
            'avg_projection_mape': df['projection_mape'].mean(),
            'avg_projection_rmse': df['projection_rmse'].mean(),
            'avg_projection_correlation': df['projection_correlation'].mean()
        }

        if 'lineup_actual_score' in df.columns:
            summary['avg_lineup_score'] = df['lineup_actual_score'].mean()
            summary['avg_lineup_error'] = df['lineup_error'].mean()
            summary['std_lineup_error'] = df['lineup_error'].std()

        return summary
