import argparse
import os
import json
import pandas as pd
import logging
from pathlib import Path

from src.models.xgboost_model import XGBoostModel
from src.models.random_forest_model import RandomForestModel
from src.features.pipeline import FeaturePipeline
from src.utils.feature_config import load_feature_config
from src.utils.fantasy_points import calculate_dk_fantasy_points

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    """
    Parse training arguments from SageMaker.

    SageMaker passes hyperparameters as command line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--player-id', type=str, required=True)
    parser.add_argument('--player-name', type=str, required=True)
    parser.add_argument('--model-type', type=str, default='xgboost')
    parser.add_argument('--feature-config', type=str, default='default_features')
    parser.add_argument('--min-player-games', type=int, default=10)

    parser.add_argument('--max-depth', type=int, default=6)
    parser.add_argument('--learning-rate', type=float, default=0.05)
    parser.add_argument('--n-estimators', type=int, default=200)
    parser.add_argument('--min-child-weight', type=int, default=5)
    parser.add_argument('--subsample', type=float, default=0.8)
    parser.add_argument('--colsample-bytree', type=float, default=0.8)
    parser.add_argument('--random-state', type=int, default=42)

    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', '/opt/ml/model'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train'))
    parser.add_argument('--output-data-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', '/opt/ml/output'))

    return parser.parse_args()


def load_training_data(train_dir: str, player_id: str) -> pd.DataFrame:
    """
    Load training data for specific player from input channel.

    Args:
        train_dir: Training data directory
        player_id: Player ID to filter

    Returns:
        DataFrame with player training data
    """
    logger.info(f"Loading training data from {train_dir}")

    train_path = Path(train_dir)
    parquet_files = list(train_path.glob('*.parquet'))

    if not parquet_files:
        raise ValueError(f"No parquet files found in {train_dir}")

    logger.info(f"Found {len(parquet_files)} parquet files")

    dfs = []
    for file_path in parquet_files:
        df = pd.read_parquet(file_path)
        if 'playerID' in df.columns:
            df = df[df['playerID'] == player_id]
            if not df.empty:
                dfs.append(df)

    if not dfs:
        raise ValueError(f"No training data found for player {player_id}")

    training_data = pd.concat(dfs, ignore_index=True)
    logger.info(f"Loaded {len(training_data)} training records for player {player_id}")

    return training_data


def build_features(
    training_data: pd.DataFrame,
    feature_config: str
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build training features using FeaturePipeline.

    Args:
        training_data: Raw training data
        feature_config: Feature configuration name

    Returns:
        Tuple of (X_train, y_train)
    """
    logger.info("Building features")

    config = load_feature_config(feature_config)
    feature_pipeline = config.build_pipeline(FeaturePipeline)

    df = training_data.copy()
    df['gameDate'] = pd.to_datetime(df['gameDate'], format='%Y%m%d', errors='coerce')
    df = df.sort_values(['playerID', 'gameDate'])

    if 'fpts' not in df.columns:
        df['fpts'] = df.apply(calculate_dk_fantasy_points, axis=1)

    df['target'] = df.groupby('playerID')['fpts'].shift(-1)
    df = feature_pipeline.fit_transform(df)
    df = df.dropna(subset=['target'])

    metadata_cols = [
        'playerID', 'playerName', 'longName', 'team', 'teamAbv', 'teamID',
        'pos', 'gameDate', 'gameID', 'fpts', 'fantasyPoints', 'fantasyPts',
        'target', 'pts', 'reb', 'ast', 'stl', 'blk', 'TOV', 'mins',
        'tech', 'created_at', 'updated_at'
    ]

    feature_cols = [col for col in df.columns if col not in metadata_cols]

    X_train = df[feature_cols].copy()
    X_train = X_train.fillna(0)

    for col in X_train.columns:
        if X_train[col].dtype == 'object':
            X_train[col] = pd.to_numeric(X_train[col], errors='coerce').fillna(0)

    y_train = df['target']

    logger.info(f"Features built: {len(X_train)} samples, {len(feature_cols)} features")

    return X_train, y_train


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: str,
    model_params: dict
):
    """
    Train model.

    Args:
        X_train: Feature matrix
        y_train: Target variable
        model_type: Model type (xgboost or random_forest)
        model_params: Model hyperparameters

    Returns:
        Trained model
    """
    logger.info(f"Training {model_type} model on {len(X_train)} samples")

    if model_type == 'xgboost':
        model = XGBoostModel(model_params)
    elif model_type == 'random_forest':
        rf_params = {k: v for k, v in model_params.items()
                     if k not in ['learning_rate', 'objective', 'colsample_bytree', 'subsample', 'min_child_weight']}
        model = RandomForestModel(rf_params)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    model.train(X_train, y_train)
    logger.info("Model training complete")

    return model


def save_model(model, model_dir: str, player_id: str, player_name: str, metadata: dict):
    """
    Save trained model and metadata.

    Args:
        model: Trained model
        model_dir: Model output directory
        player_id: Player ID
        player_name: Player name
        metadata: Additional metadata
    """
    logger.info(f"Saving model to {model_dir}")

    model_dir_path = Path(model_dir)
    model_dir_path.mkdir(parents=True, exist_ok=True)

    safe_player_name = "".join(c if c.isalnum() or c in (' ', '_', '-') else '_' for c in player_name)
    safe_player_name = safe_player_name.replace(' ', '_')

    model_file = model_dir_path / f"{safe_player_name}_{player_id}.pkl"
    model.save(str(model_file))
    logger.info(f"Model saved to {model_file}")

    metadata_file = model_dir_path / f"{safe_player_name}_{player_id}.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Metadata saved to {metadata_file}")


def main():
    """
    Main training function for SageMaker.
    """
    args = parse_args()

    logger.info("="*80)
    logger.info("SAGEMAKER TRAINING JOB")
    logger.info("="*80)
    logger.info(f"Player: {args.player_name} ({args.player_id})")
    logger.info(f"Model type: {args.model_type}")
    logger.info(f"Feature config: {args.feature_config}")
    logger.info(f"Min player games: {args.min_player_games}")
    logger.info(f"Train dir: {args.train}")
    logger.info(f"Model dir: {args.model_dir}")
    logger.info("="*80)

    training_data = load_training_data(args.train, args.player_id)

    if len(training_data) < args.min_player_games:
        logger.warning(f"Insufficient training data: {len(training_data)} < {args.min_player_games}")
        raise ValueError(f"Insufficient training data for player {args.player_id}")

    X_train, y_train = build_features(training_data, args.feature_config)

    if X_train.empty or y_train.empty or len(X_train) < 3:
        logger.warning("Insufficient features after processing")
        raise ValueError("Insufficient features for training")

    model_params = {
        'max_depth': args.max_depth,
        'learning_rate': args.learning_rate,
        'n_estimators': args.n_estimators,
        'min_child_weight': args.min_child_weight,
        'subsample': args.subsample,
        'colsample_bytree': args.colsample_bytree,
        'random_state': args.random_state,
        'objective': 'reg:squarederror'
    }

    model = train_model(X_train, y_train, args.model_type, model_params)

    metadata = {
        'player_id': args.player_id,
        'player_name': args.player_name,
        'model_type': args.model_type,
        'feature_config': args.feature_config,
        'num_training_samples': len(X_train),
        'num_features': len(X_train.columns),
        'model_params': model_params
    }

    save_model(model, args.model_dir, args.player_id, args.player_name, metadata)

    logger.info("="*80)
    logger.info("TRAINING COMPLETE")
    logger.info("="*80)


if __name__ == '__main__':
    main()
