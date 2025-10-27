# SageMaker Deployment Guide

Guide for deploying NBA DFS per-player model training to AWS SageMaker.

## Architecture Overview

```
Local Machine (Orchestration)
    |
    | Submit training jobs
    v
AWS SageMaker Training Jobs (500+ parallel)
    |
    | Read training data
    v
S3 Bucket (Data + Models)
```

## Prerequisites

### AWS Account Setup

1. AWS account with appropriate permissions
2. AWS CLI installed and configured
3. Python 3.8+ with required dependencies

### IAM Role Setup

Create SageMaker execution role with policies:

```bash
aws iam create-role \
  --role-name SageMakerNBADFSRole \
  --assume-role-policy-document '{
    "Version": "2012-10-17",
    "Statement": [{
      "Effect": "Allow",
      "Principal": {"Service": "sagemaker.amazonaws.com"},
      "Action": "sts:AssumeRole"
    }]
  }'

aws iam attach-role-policy \
  --role-name SageMakerNBADFSRole \
  --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess

aws iam attach-role-policy \
  --role-name SageMakerNBADFSRole \
  --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess
```

Get role ARN:
```bash
aws iam get-role --role-name SageMakerNBADFSRole --query 'Role.Arn' --output text
```

### S3 Bucket Setup

Create bucket for data and models:

```bash
aws s3 mb s3://nba-dfs-training-YOUR-ACCOUNT-ID --region us-east-1
```

## Installation

Install dependencies:

```bash
pip install -r requirements.txt
pip install sagemaker boto3
```

## Data Preparation

### Option 1: Upload Local Data to S3

Upload existing Parquet data:

```bash
python scripts/run_sagemaker_backtest.py \
  --upload-data \
  --local-data-dir data/inputs \
  --bucket nba-dfs-training-YOUR-ACCOUNT-ID \
  --role arn:aws:iam::YOUR-ACCOUNT-ID:role/SageMakerNBADFSRole \
  --train-start 20241001 \
  --train-end 20241130 \
  --test-start 20241201 \
  --test-end 20241215
```

### Option 2: Direct S3 Collection

Modify data collection scripts to write directly to S3 using S3Storage:

```python
from src.data.storage.s3_storage import S3Storage

storage = S3Storage(
    bucket='nba-dfs-training-YOUR-ACCOUNT-ID',
    prefix='data/inputs',
    region='us-east-1'
)
```

## Training Configuration

### Model Hyperparameters

Default XGBoost parameters:

```python
model_params = {
    'max_depth': 6,
    'learning_rate': 0.05,
    'n_estimators': 200,
    'min_child_weight': 5,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42
}
```

### Instance Types

Recommended instance types:

- `ml.m5.xlarge`: 4 vCPU, 16 GB RAM, $0.23/hr on-demand, $0.07/hr spot
- `ml.m5.2xlarge`: 8 vCPU, 32 GB RAM, $0.46/hr on-demand, $0.14/hr spot
- `ml.c5.xlarge`: 4 vCPU, 8 GB RAM, $0.204/hr on-demand, $0.06/hr spot (CPU-optimized)

Use spot instances for 70% cost savings.

### Concurrent Job Limits

SageMaker default limits:
- Training jobs: 100 concurrent
- Request limit increase via AWS Support if needed

## Running Backtest

### Basic Usage

```bash
python scripts/run_sagemaker_backtest.py \
  --train-start 20241001 \
  --train-end 20241130 \
  --test-start 20241201 \
  --test-end 20241215 \
  --bucket nba-dfs-training-YOUR-ACCOUNT-ID \
  --role arn:aws:iam::YOUR-ACCOUNT-ID:role/SageMakerNBADFSRole \
  --region us-east-1 \
  --instance-type ml.m5.xlarge \
  --use-spot \
  --max-concurrent-jobs 50
```

### With Spot Instances

```bash
python scripts/run_sagemaker_backtest.py \
  --train-start 20241001 \
  --train-end 20250131 \
  --test-start 20250201 \
  --test-end 20250228 \
  --bucket nba-dfs-training-YOUR-ACCOUNT-ID \
  --role arn:aws:iam::YOUR-ACCOUNT-ID:role/SageMakerNBADFSRole \
  --use-spot \
  --instance-type ml.m5.xlarge \
  --max-concurrent-jobs 50
```

### Custom Hyperparameters

```bash
python scripts/run_sagemaker_backtest.py \
  --train-start 20241001 \
  --train-end 20241130 \
  --test-start 20241201 \
  --test-end 20241215 \
  --bucket nba-dfs-training-YOUR-ACCOUNT-ID \
  --role arn:aws:iam::YOUR-ACCOUNT-ID:role/SageMakerNBADFSRole \
  --max-depth 8 \
  --learning-rate 0.03 \
  --n-estimators 300 \
  --feature-config default_features
```

## Cost Estimation

### Per-Player Model Training

Assumptions:
- 500 players per slate
- 5 minutes training time per player
- ml.m5.xlarge spot instances ($0.07/hr)

Cost per slate:
```
500 players × 5 min × $0.07/hr ÷ 60 min/hr = $2.92/slate
```

### Full Season Backtest

Assumptions:
- 100 slates in season
- Recalibration every 7 days (15 training cycles)

Total cost:
```
15 training cycles × $2.92 = $43.80 for full season
```

With on-demand instances:
```
15 training cycles × $9.58 = $143.70 for full season
```

### Optimization Tips

1. Use spot instances (70% savings)
2. Batch training jobs (train once every 7 days vs daily)
3. Filter players by salary threshold (train only $5k+ players)
4. Use smaller instance types for players with limited data
5. Cache models between slates (reuse if no recalibration)

## Monitoring

### Training Job Status

Monitor via AWS Console:
- Navigate to SageMaker > Training jobs
- Filter by job name prefix: `nba-dfs-`
- View logs in CloudWatch

Monitor via CLI:
```bash
aws sagemaker list-training-jobs \
  --name-contains nba-dfs \
  --sort-by CreationTime \
  --sort-order Descending \
  --max-results 50
```

### CloudWatch Logs

View training logs:
```bash
aws logs tail /aws/sagemaker/TrainingJobs --follow
```

### Job Metrics

Get job metrics via script:
```python
from sagemaker.sagemaker_trainer import SageMakerPerPlayerTrainer

trainer = SageMakerPerPlayerTrainer(...)
metrics_df = trainer.get_job_metrics()
print(metrics_df)
```

## Troubleshooting

### Issue: Training Job Fails

Check CloudWatch logs:
```bash
aws logs get-log-events \
  --log-group-name /aws/sagemaker/TrainingJobs \
  --log-stream-name <job-name>/algo-1-<timestamp>
```

Common causes:
- Insufficient training data for player
- Feature generation errors
- Out of memory (use larger instance type)

### Issue: Spot Instance Interruption

SageMaker automatically retries interrupted spot jobs. Set `max_wait` appropriately:

```python
trainer = SageMakerPerPlayerTrainer(
    use_spot_instances=True,
    max_wait=86400,  # 24 hours
    max_run=3600     # 1 hour per job
)
```

### Issue: Too Many Concurrent Jobs

Reduce `max_concurrent_jobs`:
```bash
python scripts/run_sagemaker_backtest.py \
  --max-concurrent-jobs 25 \
  ...
```

Or request limit increase via AWS Support.

### Issue: S3 Access Denied

Verify IAM role has S3 permissions:
```bash
aws iam list-attached-role-policies --role-name SageMakerNBADFSRole
```

## Advanced Configuration

### Custom Container Image

Build custom Docker image with dependencies:

```dockerfile
FROM public.ecr.aws/sagemaker/xgboost:1.7-1

COPY requirements.txt /opt/ml/code/requirements.txt
RUN pip install -r /opt/ml/code/requirements.txt

COPY sagemaker/ /opt/ml/code/
COPY src/ /opt/ml/code/src/
COPY config/ /opt/ml/code/config/

ENV PYTHONPATH="/opt/ml/code:$PYTHONPATH"
```

Build and push:
```bash
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com

docker build -t nba-dfs-training:latest -f sagemaker/Dockerfile .
docker tag nba-dfs-training:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/nba-dfs-training:latest
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/nba-dfs-training:latest
```

Use in training:
```python
trainer = SageMakerPerPlayerTrainer(
    image_uri='<account-id>.dkr.ecr.us-east-1.amazonaws.com/nba-dfs-training:latest',
    ...
)
```

### Hyperparameter Tuning

Use SageMaker Automatic Model Tuning:

```python
from sagemaker.tuner import HyperparameterTuner, IntegerParameter, ContinuousParameter

hyperparameter_ranges = {
    'max_depth': IntegerParameter(3, 10),
    'learning_rate': ContinuousParameter(0.01, 0.2),
    'n_estimators': IntegerParameter(100, 500)
}

tuner = HyperparameterTuner(
    estimator,
    objective_metric_name='validation:rmse',
    hyperparameter_ranges=hyperparameter_ranges,
    max_jobs=20,
    max_parallel_jobs=5
)

tuner.fit({'train': train_s3_path})
```

### Distributed Training

For large datasets, use distributed XGBoost:

```python
estimator = Estimator(
    instance_count=2,  # Multiple instances
    instance_type='ml.m5.xlarge',
    ...
)
```

## Production Deployment

### Model Inference

Deploy trained models for real-time inference:

```python
from src.data.storage.s3_storage import S3Storage

s3_storage = S3Storage(bucket='nba-dfs-training-YOUR-ACCOUNT-ID')
s3_storage.load_model('models/player_123/model.pkl', 'local_model.pkl')

from src.models.xgboost_model import XGBoostModel
model = XGBoostModel()
model.load('local_model.pkl')

predictions = model.predict(X_features)
```

### Automated Pipeline

Schedule backtests via AWS Lambda or EventBridge:

1. Create Lambda function invoking `run_sagemaker_backtest.py`
2. Configure EventBridge schedule (e.g., weekly)
3. Store results in S3 or RDS

### CI/CD Integration

GitHub Actions workflow:

```yaml
name: SageMaker Training

on:
  schedule:
    - cron: '0 0 * * 0'  # Weekly

jobs:
  train:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v1
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: us-east-1
      - name: Run backtest
        run: |
          python scripts/run_sagemaker_backtest.py \
            --bucket nba-dfs-training \
            --role ${{ secrets.SAGEMAKER_ROLE }} \
            --train-start 20240901 \
            --train-end 20241130 \
            --test-start 20241201 \
            --test-end 20241215
```

## Support

For issues:
- Check CloudWatch logs
- Review SageMaker documentation: https://docs.aws.amazon.com/sagemaker/
- AWS Support: https://console.aws.amazon.com/support/

## References

- SageMaker Python SDK: https://sagemaker.readthedocs.io/
- XGBoost on SageMaker: https://docs.aws.amazon.com/sagemaker/latest/dg/xgboost.html
- Spot Instance Training: https://docs.aws.amazon.com/sagemaker/latest/dg/model-managed-spot-training.html
