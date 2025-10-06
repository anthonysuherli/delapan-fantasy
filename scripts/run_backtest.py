import sys
from pathlib import Path
import logging
import argparse
from typing import List, Dict, Any, Callable, Optional
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backtest.backtest import BacktestRunner, WalkForwardValidator
from src.data.storage_utils import get_all_files_in_date_range

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FeatureDataLoader:
    """
    Load and prepare feature data for backtesting.
    Implements data loading interface required by BacktestRunner.
    """

    def __init__(self, player_logs_dir: Path):
        self.player_logs_dir = Path(player_logs_dir)
        logger.info(f"Initialized FeatureDataLoader with directory: {self.player_logs_dir}")

    def load_data(
        self,
        start_date: str,
        end_date: str,
        feature_columns: Optional[List[str]] = None,
        exclude_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Load player feature data from parquet files.

        Args:
            start_date: Start date in YYYYMMDD format
            end_date: End date in YYYYMMDD format
            feature_columns: Optional list of feature columns to include (None = all)
            exclude_columns: Optional list of columns to exclude

        Returns:
            DataFrame with player features and target
        """
        logger.info(f"Loading data from {start_date} to {end_date}")

        parquet_files = get_all_files_in_date_range(
            str(self.player_logs_dir),
            start_date=start_date,
            end_date=end_date,
            pattern='*.parquet'
        )
        logger.info(f"Found {len(parquet_files)} player log files in date range")

        if not parquet_files:
            logger.warning("No player log files found")
            return pd.DataFrame()

        all_data = []
        start_dt = pd.to_datetime(start_date, format='%Y%m%d')
        end_dt = pd.to_datetime(end_date, format='%Y%m%d')

        for file_path in parquet_files:
            try:
                df = pd.read_parquet(file_path, engine='pyarrow')

                if 'gameDate' in df.columns:
                    df['date'] = pd.to_datetime(df['gameDate'], format='%Y%m%d', errors='coerce')
                    df = df[(df['date'] >= start_dt) & (df['date'] <= end_dt)]

                    if not df.empty:
                        all_data.append(df)

            except Exception as e:
                logger.debug(f"Failed to load {file_path.name}: {str(e)}")
                try:
                    df = pd.read_parquet(file_path, engine='fastparquet')

                    if 'gameDate' in df.columns:
                        df['date'] = pd.to_datetime(df['gameDate'], format='%Y%m%d', errors='coerce')
                        df = df[(df['date'] >= start_dt) & (df['date'] <= end_dt)]

                        if not df.empty:
                            all_data.append(df)
                            logger.debug(f"Successfully loaded {file_path.name} with fastparquet")

                except Exception as e2:
                    logger.warning(f"Failed to load {file_path.name} with both engines: {str(e2)}")
                    continue

        if not all_data:
            logger.warning("No data loaded from parquet files")
            return pd.DataFrame()

        combined = pd.concat(all_data, ignore_index=True)
        logger.info(f"Loaded {len(combined)} rows from {len(all_data)} player files")
        logger.info(f"Date range in loaded data: {combined['date'].min()} to {combined['date'].max()}")
        logger.info(f"Total columns before filtering: {len(combined.columns)}")

        if feature_columns is not None:
            logger.info(f"Filtering to {len(feature_columns)} specified feature columns")
            available_cols = [col for col in feature_columns if col in combined.columns]
            missing_cols = [col for col in feature_columns if col not in combined.columns]

            if missing_cols:
                logger.warning(f"Missing columns: {missing_cols}")

            if 'date' not in available_cols:
                available_cols.append('date')

            combined = combined[available_cols]
            logger.info(f"Columns after filtering: {len(combined.columns)}")
        else:
            logger.info(f"Using all {len(combined.columns)} available columns")

            if exclude_columns:
                cols_to_drop = [col for col in exclude_columns if col in combined.columns]
                if cols_to_drop:
                    logger.info(f"Excluding {len(cols_to_drop)} columns: {cols_to_drop}")
                    combined = combined.drop(columns=cols_to_drop)
                    logger.info(f"Columns after exclusion: {len(combined.columns)}")

        logger.info(f"Final dataset shape: {combined.shape}")
        return combined


class CustomBacktestRunner(BacktestRunner):
    """
    Extended BacktestRunner with custom data loading.
    """

    def __init__(
        self,
        model_fn: Callable,
        data_loader: FeatureDataLoader,
        validator: Optional[WalkForwardValidator] = None
    ):
        super().__init__(model_fn=model_fn, storage=None, validator=validator)
        self.data_loader = data_loader

    def _load_data(
        self,
        start_date: str,
        end_date: str,
        feature_columns: Optional[List[str]] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Load data using FeatureDataLoader.
        """
        return self.data_loader.load_data(start_date, end_date, feature_columns)


def create_dummy_model():
    """
    Create a simple baseline model for testing.
    """
    from sklearn.ensemble import RandomForestRegressor

    return RandomForestRegressor(
        n_estimators=50,
        max_depth=10,
        min_samples_split=10,
        random_state=42,
        n_jobs=-1
    )


def run_walk_forward_backtest(
    player_logs_dir: str,
    start_date: str,
    end_date: str,
    feature_columns: Optional[List[str]] = None,
    exclude_columns: Optional[List[str]] = None,
    target_column: str = 'fpts',
    train_window: int = 30,
    test_window: int = 7,
    step_size: int = 7,
    model_fn: Optional[Callable] = None
) -> Dict[str, Any]:
    """
    Run walk-forward backtesting framework.

    Args:
        player_logs_dir: Directory containing player log parquet files
        start_date: Start date in YYYYMMDD format
        end_date: End date in YYYYMMDD format
        feature_columns: List of feature column names (None = all columns)
        exclude_columns: List of columns to exclude when using all features
        target_column: Target column name (default: fpts)
        train_window: Training window size in days
        test_window: Test window size in days
        step_size: Step size for walk-forward
        model_fn: Function that returns model instance (default: RandomForest)

    Returns:
        Dict with aggregated backtest results
    """
    logger.info("="*80)
    logger.info("STARTING WALK-FORWARD BACKTEST FRAMEWORK")
    logger.info("="*80)
    logger.info(f"Date range: {start_date} to {end_date}")
    logger.info(f"Train window: {train_window} days, Test window: {test_window} days, Step size: {step_size} days")
    logger.info(f"Target column: {target_column}")

    logger.info("\n[1/5] Initializing data loader...")
    data_loader = FeatureDataLoader(player_logs_dir=Path(player_logs_dir))

    logger.info("\n[2/5] Initializing walk-forward validator...")
    validator = WalkForwardValidator(
        train_window=train_window,
        test_window=test_window,
        step_size=step_size
    )

    logger.info("\n[3/5] Setting up model...")
    model_fn = model_fn or create_dummy_model
    logger.info(f"Using model: {model_fn.__name__ if hasattr(model_fn, '__name__') else 'RandomForestRegressor'}")

    logger.info("\n[4/5] Loading and preparing data...")
    runner = CustomBacktestRunner(
        model_fn=model_fn,
        data_loader=data_loader,
        validator=validator
    )

    data = data_loader.load_data(start_date, end_date, feature_columns, exclude_columns)

    if data.empty:
        logger.error("CRITICAL: No data loaded from parquet files")
        return {'error': 'No data loaded'}

    logger.info(f"Successfully loaded data with {len(data)} rows")

    default_exclude = ['date', 'gameDate', 'playerID', 'playerName', 'team', 'pos', target_column]
    if feature_columns is None:
        feature_columns = [col for col in data.columns if col not in default_exclude]
        logger.info(f"Auto-detected {len(feature_columns)} feature columns")
        logger.info(f"Feature columns: {', '.join(feature_columns[:10])}{'...' if len(feature_columns) > 10 else ''}")

    if target_column not in data.columns:
        logger.error(f"CRITICAL: Target column '{target_column}' not found in data")
        logger.info(f"Available columns: {', '.join(data.columns.tolist())}")
        return {'error': f"Target column '{target_column}' not found"}

    logger.info(f"Target column '{target_column}' found with {data[target_column].notna().sum()} non-null values")

    logger.info("\n[5/5] Running walk-forward cross-validation...")
    logger.info("This may take several minutes depending on data size and number of folds...")

    results = runner.run(
        start_date=start_date,
        end_date=end_date,
        feature_columns=feature_columns,
        target_column=target_column,
        date_column='date'
    )

    logger.info("\n" + "="*80)
    logger.info("WALK-FORWARD BACKTEST COMPLETE")
    logger.info("="*80)

    return results


def print_results(results: Dict[str, Any]):
    """
    Print backtest results.
    """
    if 'error' in results:
        print("\n" + "="*80)
        print("ERROR")
        print("="*80)
        print(f"\n{results['error']}")
        print("\n" + "="*80)
        return

    print("\n" + "="*80)
    print("BACKTEST RESULTS SUMMARY")
    print("="*80)

    print(f"\nNumber of Folds Completed: {results.get('num_folds', 0)}")

    print("\n" + "-"*80)
    print("PREDICTION ACCURACY METRICS")
    print("-"*80)

    print("\nMean Absolute Percentage Error (MAPE):")
    print(f"  Average across folds: {results.get('avg_mape', 0):.2f}%")
    print(f"  Standard deviation:   {results.get('std_mape', 0):.2f}%")

    print("\nRoot Mean Squared Error (RMSE):")
    print(f"  Average across folds: {results.get('avg_rmse', 0):.2f}")
    print(f"  Standard deviation:   {results.get('std_rmse', 0):.2f}")

    print("\nPrediction Correlation:")
    print(f"  Average across folds: {results.get('avg_correlation', 0):.3f}")
    print(f"  Standard deviation:   {results.get('std_correlation', 0):.3f}")

    if 'avg_mae' in results:
        print("\nMean Absolute Error (MAE):")
        print(f"  Average across folds: {results.get('avg_mae', 0):.2f}")
        print(f"  Standard deviation:   {results.get('std_mae', 0):.2f}")

    if 'avg_r2' in results:
        print("\nR-Squared (R2):")
        print(f"  Average across folds: {results.get('avg_r2', 0):.3f}")
        print(f"  Standard deviation:   {results.get('std_r2', 0):.3f}")

    print("\n" + "="*80)

    if results.get('avg_mape', 100) < 30:
        print("STATUS: Good prediction accuracy (MAPE < 30%)")
    elif results.get('avg_mape', 100) < 50:
        print("STATUS: Moderate prediction accuracy (MAPE 30-50%)")
    else:
        print("STATUS: Poor prediction accuracy (MAPE > 50%)")

    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description='Modular walk-forward backtest framework for NBA DFS'
    )

    parser.add_argument(
        '--player-logs-dir',
        type=str,
        default='data/player_logs',
        help='Directory containing player log parquet files'
    )
    parser.add_argument(
        '--start-date',
        type=str,
        required=True,
        help='Start date in YYYYMMDD format'
    )
    parser.add_argument(
        '--end-date',
        type=str,
        required=True,
        help='End date in YYYYMMDD format'
    )
    parser.add_argument(
        '--features',
        type=str,
        nargs='+',
        default=None,
        help='Feature column names (omit to use all columns)'
    )
    parser.add_argument(
        '--exclude',
        type=str,
        nargs='+',
        default=None,
        help='Columns to exclude when using all features'
    )
    parser.add_argument(
        '--target',
        type=str,
        default='fpts',
        help='Target column name (default: fpts)'
    )
    parser.add_argument(
        '--train-window',
        type=int,
        default=30,
        help='Training window size in days (default: 30)'
    )
    parser.add_argument(
        '--test-window',
        type=int,
        default=7,
        help='Test window size in days (default: 7)'
    )
    parser.add_argument(
        '--step-size',
        type=int,
        default=7,
        help='Step size for walk-forward (default: 7)'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    results = run_walk_forward_backtest(
        player_logs_dir=args.player_logs_dir,
        start_date=args.start_date,
        end_date=args.end_date,
        feature_columns=args.features,
        exclude_columns=args.exclude,
        target_column=args.target,
        train_window=args.train_window,
        test_window=args.test_window,
        step_size=args.step_size
    )

    print_results(results)


if __name__ == '__main__':
    main()
