import logging
import time
from typing import List, Dict, Any, Optional
from pathlib import Path
import pandas as pd
import boto3
from sagemaker.estimator import Estimator
from sagemaker import Session
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


class SageMakerPerPlayerTrainer:
    """
    SageMaker-based trainer for parallel per-player model training.
    """

    def __init__(
        self,
        role: str,
        bucket: str,
        instance_type: str = 'ml.m5.xlarge',
        instance_count: int = 1,
        use_spot_instances: bool = True,
        max_wait: int = 86400,
        max_run: int = 3600,
        region: str = 'us-east-1',
        image_uri: Optional[str] = None
    ):
        """
        Initialize SageMaker trainer.

        Args:
            role: SageMaker execution role ARN
            bucket: S3 bucket for data and models
            instance_type: EC2 instance type
            instance_count: Number of instances per training job
            use_spot_instances: Use spot instances for cost savings
            max_wait: Maximum time to wait for spot instances (seconds)
            max_run: Maximum training time (seconds)
            region: AWS region
            image_uri: Custom container image URI (optional)
        """
        self.role = role
        self.bucket = bucket
        self.instance_type = instance_type
        self.instance_count = instance_count
        self.use_spot_instances = use_spot_instances
        self.max_wait = max_wait
        self.max_run = max_run
        self.region = region
        self.image_uri = image_uri

        self.session = Session(boto3.session.Session(region_name=region))
        self.training_jobs = {}

    def create_estimator(
        self,
        training_job_name: str,
        hyperparameters: Dict[str, Any],
        output_path: str
    ) -> Estimator:
        """
        Create SageMaker estimator for single training job.

        Args:
            training_job_name: Unique training job name
            hyperparameters: Model hyperparameters
            output_path: S3 output path for models

        Returns:
            Configured estimator
        """
        if self.image_uri:
            estimator = Estimator(
                image_uri=self.image_uri,
                role=self.role,
                instance_count=self.instance_count,
                instance_type=self.instance_type,
                output_path=output_path,
                sagemaker_session=self.session,
                hyperparameters=hyperparameters,
                use_spot_instances=self.use_spot_instances,
                max_wait=self.max_wait if self.use_spot_instances else None,
                max_run=self.max_run,
                base_job_name=training_job_name
            )
        else:
            from sagemaker.xgboost import XGBoost

            estimator = XGBoost(
                entry_point='train.py',
                source_dir='sagemaker',
                role=self.role,
                instance_count=self.instance_count,
                instance_type=self.instance_type,
                output_path=output_path,
                sagemaker_session=self.session,
                hyperparameters=hyperparameters,
                framework_version='1.7-1',
                py_version='py3',
                use_spot_instances=self.use_spot_instances,
                max_wait=self.max_wait if self.use_spot_instances else None,
                max_run=self.max_run,
                base_job_name=training_job_name
            )

        return estimator

    def train_player_model(
        self,
        player_id: str,
        player_name: str,
        train_data_s3: str,
        model_params: Dict[str, Any],
        feature_config: str = 'default_features',
        wait: bool = False
    ) -> str:
        """
        Submit training job for single player.

        Args:
            player_id: Player ID
            player_name: Player name
            train_data_s3: S3 path to training data
            model_params: Model hyperparameters
            feature_config: Feature configuration name
            wait: Wait for job to complete

        Returns:
            Training job name
        """
        safe_player_name = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in player_name)
        safe_player_name = safe_player_name.replace(' ', '_')

        training_job_name = f"nba-dfs-{safe_player_name}-{int(time.time())}"

        hyperparameters = {
            'player-id': player_id,
            'player-name': player_name,
            'feature-config': feature_config,
            **model_params
        }

        output_path = f"s3://{self.bucket}/models/{player_id}"

        estimator = self.create_estimator(training_job_name, hyperparameters, output_path)

        logger.info(f"Submitting training job for {player_name} ({player_id})")
        logger.info(f"Job name: {training_job_name}")

        estimator.fit(
            inputs={'train': train_data_s3},
            wait=wait,
            logs=False
        )

        self.training_jobs[player_id] = {
            'job_name': training_job_name,
            'estimator': estimator,
            'player_name': player_name,
            'status': 'InProgress'
        }

        return training_job_name

    def train_multiple_players(
        self,
        players: pd.DataFrame,
        train_data_s3: str,
        model_params: Dict[str, Any],
        feature_config: str = 'default_features',
        max_concurrent_jobs: int = 50
    ) -> Dict[str, str]:
        """
        Submit training jobs for multiple players in parallel.

        Args:
            players: DataFrame with playerID, playerName columns
            train_data_s3: S3 path to training data
            model_params: Model hyperparameters
            feature_config: Feature configuration name
            max_concurrent_jobs: Maximum concurrent training jobs

        Returns:
            Dict mapping player_id to job_name
        """
        logger.info(f"Submitting {len(players)} training jobs with max_concurrent={max_concurrent_jobs}")

        job_names = {}

        with ThreadPoolExecutor(max_workers=max_concurrent_jobs) as executor:
            futures = {}

            for _, player_row in players.iterrows():
                player_id = player_row['playerID']
                player_name = player_row.get('longName', player_row.get('playerName', ''))

                future = executor.submit(
                    self.train_player_model,
                    player_id,
                    player_name,
                    train_data_s3,
                    model_params,
                    feature_config,
                    wait=False
                )
                futures[future] = player_id

            for future in as_completed(futures):
                player_id = futures[future]
                try:
                    job_name = future.result()
                    job_names[player_id] = job_name
                    logger.info(f"Job submitted for player {player_id}: {job_name}")
                except Exception as e:
                    logger.error(f"Failed to submit job for player {player_id}: {str(e)}")

        logger.info(f"Submitted {len(job_names)} training jobs")
        return job_names

    def wait_for_jobs(self, poll_interval: int = 30) -> Dict[str, str]:
        """
        Wait for all training jobs to complete.

        Args:
            poll_interval: Polling interval in seconds

        Returns:
            Dict mapping player_id to job status
        """
        logger.info(f"Waiting for {len(self.training_jobs)} training jobs to complete")

        sm_client = boto3.client('sagemaker', region_name=self.region)

        completed = 0
        total = len(self.training_jobs)

        while True:
            in_progress = 0

            for player_id, job_info in self.training_jobs.items():
                if job_info['status'] in ['Completed', 'Failed', 'Stopped']:
                    continue

                job_name = job_info['job_name']

                try:
                    response = sm_client.describe_training_job(TrainingJobName=job_name)
                    status = response['TrainingJobStatus']

                    if status != job_info['status']:
                        logger.info(f"Job {job_name} status: {status}")
                        job_info['status'] = status

                        if status in ['Completed', 'Failed', 'Stopped']:
                            completed += 1
                            logger.info(f"Progress: {completed}/{total} jobs complete")

                    if status == 'InProgress':
                        in_progress += 1

                except Exception as e:
                    logger.error(f"Failed to check status for {job_name}: {str(e)}")

            if in_progress == 0:
                break

            logger.info(f"Waiting for {in_progress} jobs to complete...")
            time.sleep(poll_interval)

        status_summary = {pid: info['status'] for pid, info in self.training_jobs.items()}

        completed_count = sum(1 for s in status_summary.values() if s == 'Completed')
        failed_count = sum(1 for s in status_summary.values() if s == 'Failed')

        logger.info(f"All jobs finished: {completed_count} completed, {failed_count} failed")

        return status_summary

    def download_models(self, output_dir: str) -> List[str]:
        """
        Download trained models from S3.

        Args:
            output_dir: Local output directory

        Returns:
            List of downloaded model paths
        """
        logger.info(f"Downloading models to {output_dir}")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        s3_client = boto3.client('s3', region_name=self.region)
        downloaded_models = []

        for player_id, job_info in self.training_jobs.items():
            if job_info['status'] != 'Completed':
                continue

            model_s3_prefix = f"models/{player_id}/"

            try:
                response = s3_client.list_objects_v2(
                    Bucket=self.bucket,
                    Prefix=model_s3_prefix
                )

                if 'Contents' not in response:
                    logger.warning(f"No model found for player {player_id}")
                    continue

                for obj in response['Contents']:
                    key = obj['Key']
                    file_name = Path(key).name
                    local_path = output_path / file_name

                    s3_client.download_file(self.bucket, key, str(local_path))
                    downloaded_models.append(str(local_path))
                    logger.info(f"Downloaded {key} to {local_path}")

            except Exception as e:
                logger.error(f"Failed to download model for player {player_id}: {str(e)}")

        logger.info(f"Downloaded {len(downloaded_models)} model files")
        return downloaded_models

    def get_job_metrics(self) -> pd.DataFrame:
        """
        Get training metrics for all jobs.

        Returns:
            DataFrame with job metrics
        """
        sm_client = boto3.client('sagemaker', region_name=self.region)

        metrics = []

        for player_id, job_info in self.training_jobs.items():
            job_name = job_info['job_name']

            try:
                response = sm_client.describe_training_job(TrainingJobName=job_name)

                metrics.append({
                    'player_id': player_id,
                    'player_name': job_info['player_name'],
                    'job_name': job_name,
                    'status': response['TrainingJobStatus'],
                    'training_time': response.get('TrainingTimeInSeconds', 0),
                    'billable_time': response.get('BillableTimeInSeconds', 0),
                    'instance_type': self.instance_type,
                    'failure_reason': response.get('FailureReason', '')
                })

            except Exception as e:
                logger.error(f"Failed to get metrics for {job_name}: {str(e)}")

        return pd.DataFrame(metrics)
