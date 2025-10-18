"""
Example: Using the Feature Pipeline with Notebook-Derived Transformers

This demonstrates how to use the modularized feature engineering
from the player_feat.ipynb notebook.
"""

import sys
from pathlib import Path
import pandas as pd

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from src.features.pipeline import FeaturePipeline
from src.features.transformers import (
    RollingStatsTransformer,
    RollingMinMaxTransformer,
    EWMATransformer,
    TargetTransformer
)
from src.data.storage.sqlite_storage import SQLiteStorage
from src.data.loaders.historical_loader import HistoricalDataLoader


def create_feature_pipeline():
    """
    Create feature pipeline matching notebook implementation.

    Returns:
        FeaturePipeline configured with all transformers
    """
    pipeline = FeaturePipeline()

    pipeline.add(RollingStatsTransformer(
        windows=[3, 5, 10],
        stats=['pts', 'reb', 'ast', 'stl', 'blk', 'mins'],
        include_std=True
    ))

    pipeline.add(RollingMinMaxTransformer(
        windows=[3, 5, 10],
        stats=['pts', 'reb', 'ast', 'stl', 'blk', 'mins']
    ))

    pipeline.add(EWMATransformer(
        span=3,
        stats=['pts', 'reb', 'ast', 'stl', 'blk', 'mins']
    ))

    pipeline.add(TargetTransformer(
        target_col='fpts',
        shift_periods=-1
    ))

    return pipeline


def main():
    """Example usage"""

    TARGET_DATE = '20250210'
    NUM_SEASONS = 2
    DB_PATH = repo_root / 'nba_dfs.db'

    print('=' * 60)
    print('Feature Pipeline Example')
    print('=' * 60)

    storage = SQLiteStorage(str(DB_PATH))
    loader = HistoricalDataLoader(storage)

    print(f'\n1. Loading historical data (target date: {TARGET_DATE})')
    historical_data = loader.load_historical_player_logs(
        end_date=TARGET_DATE,
        num_seasons=NUM_SEASONS
    )
    print(f'   Loaded {len(historical_data)} games')

    print('\n2. Creating feature pipeline')
    pipeline = create_feature_pipeline()

    print('\n3. Fitting and transforming data')
    features = pipeline.fit_transform(historical_data)

    print(f'\n4. Results:')
    print(f'   Original columns: {len(historical_data.columns)}')
    print(f'   Feature columns: {len(features.columns)}')
    print(f'   New features added: {len(features.columns) - len(historical_data.columns)}')

    feature_cols = [col for col in features.columns if any([
        '_ma' in col,
        '_std' in col,
        '_min' in col,
        '_max' in col,
        '_ewma' in col,
        col == 'target'
    ])]

    print(f'\n5. Feature breakdown:')
    print(f'   Rolling averages: {len([c for c in feature_cols if "_ma" in c])}')
    print(f'   Rolling std devs: {len([c for c in feature_cols if "_std" in c])}')
    print(f'   Rolling min/max: {len([c for c in feature_cols if "_min" in c or "_max" in c])}')
    print(f'   EWMA features: {len([c for c in feature_cols if "_ewma" in c])}')
    print(f'   Target column: {"target" in features.columns}')

    print('\n6. Sample features (first player, last game):')
    sample_player = features['playerID'].iloc[0]
    sample_data = features[features['playerID'] == sample_player].iloc[-1]

    print(f'\n   Player ID: {sample_player}')
    print(f'   Points features:')
    for col in sorted([c for c in feature_cols if c.startswith('pts')]):
        if col in sample_data.index:
            print(f'      {col:20s}: {sample_data[col]:.2f}')

    print('\n' + '=' * 60)
    print('Pipeline execution complete')
    print('=' * 60)


if __name__ == '__main__':
    main()
