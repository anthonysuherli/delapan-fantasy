import argparse
import logging
import os
import time
from pathlib import Path
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv

from sagemaker.sagemaker_trainer import SageMakerPerPlayerTrainer
from src.data.storage.s3_storage import S3Storage
from src.data.loaders.historical_loader import HistoricalDataLoader
from src.evaluation.metrics.accuracy import MAPEMetric, RMSEMetric, MAEMetric, CorrelationMetric
from src.utils.fantasy_points import calculate_dk_fantasy_points

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run walk-forward backtest with SageMaker')

    parser.add_argument('--train-start', type=str, required=True, help='Training start date (YYYYMMDD)')
    parser.add_argument('--train-end', type=str, required=True, help='Training end date (YYYYMMDD)')
    parser.add_argument('--test-start', type=str, required=True, help='Test start date (YYYYMMDD)')
    parser.add_argument('--test-end', type=str, required=True, help='Test end date (YYYYMMDD)')

    parser.add_argument('--bucket', type=str, required=True, help='S3 bucket name')
    parser.add_argument('--role', type=str, required=True, help='SageMaker execution role ARN')
    parser.add_argument('--region', type=str, default='us-east-1', help='AWS region')

    parser.add_argument('--instance-type', type=str, default='ml.m5.xlarge', help='Instance type')
    parser.add_argument('--use-spot', action='store_true', help='Use spot instances')
    parser.add_argument('--max-concurrent-jobs', type=int, default=50, help='Max concurrent training jobs')

    parser.add_argument('--model-type', type=str, default='xgboost', help='Model type')
    parser.add_argument('--feature-config', type=str, default='default_features', help='Feature config')
    parser.add_argument('--min-player-games', type=int, default=10, help='Min games for player')
    parser.add_argument('--recalibrate-days', type=int, default=7, help='Recalibration frequency')

    parser.add_argument('--max-depth', type=int, default=6)
    parser.add_argument('--learning-rate', type=float, default=0.05)
    parser.add_argument('--n-estimators', type=int, default=200)
    parser.add_argument('--min-child-weight', type=int, default=5)
    parser.add_argument('--subsample', type=float, default=0.8)
    parser.add_argument('--colsample-bytree', type=float, default=0.8)

    parser.add_argument('--upload-data', action='store_true', help='Upload local data to S3')
    parser.add_argument('--local-data-dir', type=str, default='data/inputs', help='Local data directory')

    return parser.parse_args()


def upload_data_to_s3(local_dir: str, s3_storage: S3Storage):
    """
    Upload local data to S3.

    Args:
        local_dir: Local data directory
        s3_storage: S3 storage backend
    """
    logger.info(f"Uploading data from {local_dir} to S3")
    s3_storage.upload_directory(local_dir, '')
    logger.info("Data upload complete")


def prepare_training_data(
    train_start: str,
    train_end: str,
    s3_storage: S3Storage,
    bucket: str
) -> str:
    """
    Prepare training data and return S3 path.

    Args:
        train_start: Training start date
        train_end: Training end date
        s3_storage: S3 storage backend
        bucket: S3 bucket name

    Returns:
        S3 path to training data
    """
    logger.info(f"Preparing training data from {train_start} to {train_end}")

    loader = HistoricalDataLoader(s3_storage)
    training_data = loader.load_historical_player_logs(
        start_date=train_start,
        end_date=train_end,
        num_seasons=1
    )

    logger.info(f"Loaded {len(training_data)} training records")

    train_s3_path = f"s3://{bucket}/data/training/{train_start}_{train_end}/"
    logger.info(f"Training data path: {train_s3_path}")

    return train_s3_path


def run_backtest_slate(
    test_date: str,
    train_start: str,
    train_end: str,
    s3_storage: S3Storage,
    trainer: SageMakerPerPlayerTrainer,
    model_params: dict,
    feature_config: str,
    min_player_games: int,
    max_concurrent_jobs: int
) -> pd.DataFrame:
    """
    Run backtest for single slate.

    Args:
        test_date: Test date
        train_start: Training start date
        train_end: Training end date
        s3_storage: S3 storage backend
        trainer: SageMaker trainer
        model_params: Model hyperparameters
        feature_config: Feature configuration
        min_player_games: Minimum player games
        max_concurrent_jobs: Max concurrent jobs

    Returns:
        DataFrame with predictions
    """
    logger.info(f"Processing slate {test_date}")

    loader = HistoricalDataLoader(s3_storage)
    slate_data = loader.load_slate_data(test_date)

    salaries_df = slate_data.get('dfs_salaries', pd.DataFrame())
    if salaries_df.empty:
        logger.warning(f"No salary data for {test_date}")
        return pd.DataFrame()

    training_data = loader.load_historical_player_logs(
        start_date=train_start,
        end_date=train_end,
        num_seasons=1
    )

    eligible_players = []
    for _, player_row in salaries_df.iterrows():
        player_id = player_row['playerID']
        player_data = training_data[training_data['playerID'] == player_id]
        if len(player_data) >= min_player_games:
            eligible_players.append(player_row)

    if not eligible_players:
        logger.warning(f"No eligible players for {test_date}")
        return pd.DataFrame()

    eligible_df = pd.DataFrame(eligible_players)
    logger.info(f"Found {len(eligible_df)} eligible players")

    train_s3_path = f"s3://{trainer.bucket}/data/training/{train_start}_{train_end}/"

    job_names = trainer.train_multiple_players(
        players=eligible_df,
        train_data_s3=train_s3_path,
        model_params=model_params,
        feature_config=feature_config,
        max_concurrent_jobs=max_concurrent_jobs
    )

    logger.info(f"Waiting for {len(job_names)} training jobs to complete")
    status_summary = trainer.wait_for_jobs(poll_interval=30)

    completed_count = sum(1 for s in status_summary.values() if s == 'Completed')
    logger.info(f"Training complete: {completed_count}/{len(status_summary)} jobs succeeded")

    projections = []
    for player_id, status in status_summary.items():
        if status == 'Completed':
            player_row = eligible_df[eligible_df['playerID'] == player_id].iloc[0]
            projections.append({
                'playerID': player_id,
                'playerName': player_row.get('longName', player_row.get('playerName', '')),
                'team': player_row.get('team', ''),
                'pos': player_row.get('pos', ''),
                'salary': player_row.get('salary', 0),
                'date': test_date
            })

    if not projections:
        logger.warning(f"No successful projections for {test_date}")
        return pd.DataFrame()

    return pd.DataFrame(projections)


def evaluate_slate(
    test_date: str,
    projections: pd.DataFrame,
    s3_storage: S3Storage
) -> dict:
    """
    Evaluate predictions against actuals.

    Args:
        test_date: Test date
        projections: Projections DataFrame
        s3_storage: S3 storage backend

    Returns:
        Evaluation metrics
    """
    logger.info(f"Evaluating slate {test_date}")

    loader = HistoricalDataLoader(s3_storage)
    filters = {'start_date': test_date, 'end_date': test_date}
    actuals_df = s3_storage.load('box_scores', filters)

    if actuals_df.empty:
        logger.warning(f"No actuals for {test_date}")
        return {}

    actuals_df['actual_fpts'] = actuals_df.apply(calculate_dk_fantasy_points, axis=1)

    merged = projections.merge(
        actuals_df[['playerID', 'actual_fpts']],
        on='playerID',
        how='inner'
    )

    if merged.empty:
        logger.warning(f"No matching players for {test_date}")
        return {}

    mape_metric = MAPEMetric()
    rmse_metric = RMSEMetric()
    mae_metric = MAEMetric()
    corr_metric = CorrelationMetric()

    y_true = merged['actual_fpts'].values
    y_pred = merged['projected_fpts'].values

    results = {
        'date': test_date,
        'num_players': len(merged),
        'mape': mape_metric.calculate(y_true, y_pred),
        'rmse': rmse_metric.calculate(y_true, y_pred),
        'mae': mae_metric.calculate(y_true, y_pred),
        'corr': corr_metric.calculate(y_true, y_pred),
        'mean_actual': merged['actual_fpts'].mean(),
        'mean_projected': merged['projected_fpts'].mean()
    }

    logger.info(f"Evaluation results for {test_date}:")
    logger.info(f"  Players: {results['num_players']}")
    logger.info(f"  MAPE: {results['mape']:.2f}%")
    logger.info(f"  RMSE: {results['rmse']:.2f}")
    logger.info(f"  MAE: {results['mae']:.2f}")
    logger.info(f"  Correlation: {results['corr']:.3f}")

    return results


def main():
    """Main execution function."""
    args = parse_args()

    logger.info("="*80)
    logger.info("SAGEMAKER WALK-FORWARD BACKTEST")
    logger.info("="*80)
    logger.info(f"Training period: {args.train_start} to {args.train_end}")
    logger.info(f"Testing period: {args.test_start} to {args.test_end}")
    logger.info(f"S3 bucket: {args.bucket}")
    logger.info(f"Instance type: {args.instance_type}")
    logger.info(f"Use spot instances: {args.use_spot}")
    logger.info(f"Max concurrent jobs: {args.max_concurrent_jobs}")
    logger.info("="*80)

    s3_storage = S3Storage(bucket=args.bucket, prefix='data/inputs', region=args.region)

    if args.upload_data:
        upload_data_to_s3(args.local_data_dir, s3_storage)

    trainer = SageMakerPerPlayerTrainer(
        role=args.role,
        bucket=args.bucket,
        instance_type=args.instance_type,
        instance_count=1,
        use_spot_instances=args.use_spot,
        region=args.region
    )

    model_params = {
        'model-type': args.model_type,
        'max-depth': args.max_depth,
        'learning-rate': args.learning_rate,
        'n-estimators': args.n_estimators,
        'min-child-weight': args.min_child_weight,
        'subsample': args.subsample,
        'colsample-bytree': args.colsample_bytree,
        'random-state': 42
    }

    loader = HistoricalDataLoader(s3_storage)
    slate_dates = loader.load_slate_dates(args.test_start, args.test_end)

    if not slate_dates:
        logger.error("No slate dates found")
        return

    logger.info(f"Processing {len(slate_dates)} slates")

    results = []
    for i, test_date in enumerate(slate_dates):
        logger.info(f"\nProcessing slate {i+1}/{len(slate_dates)}: {test_date}")

        projections = run_backtest_slate(
            test_date=test_date,
            train_start=args.train_start,
            train_end=args.train_end,
            s3_storage=s3_storage,
            trainer=trainer,
            model_params=model_params,
            feature_config=args.feature_config,
            min_player_games=args.min_player_games,
            max_concurrent_jobs=args.max_concurrent_jobs
        )

        if not projections.empty:
            evaluation = evaluate_slate(test_date, projections, s3_storage)
            if evaluation:
                results.append(evaluation)

    if results:
        results_df = pd.DataFrame(results)
        logger.info("\n" + "="*80)
        logger.info("BACKTEST SUMMARY")
        logger.info("="*80)
        logger.info(f"Slates processed: {len(results_df)}")
        logger.info(f"Mean MAPE: {results_df['mape'].mean():.2f}%")
        logger.info(f"Mean RMSE: {results_df['rmse'].mean():.2f}")
        logger.info(f"Mean Correlation: {results_df['corr'].mean():.3f}")
        logger.info("="*80)

        output_path = f"sagemaker_backtest_results_{int(time.time())}.csv"
        results_df.to_csv(output_path, index=False)
        logger.info(f"Results saved to {output_path}")

    logger.info("Backtest complete")


if __name__ == '__main__':
    main()
