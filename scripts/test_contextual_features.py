#!/usr/bin/env python
"""
Test script for contextual features transformer.

Validates that contextual features (home/away, rest days, back-to-back) are calculated correctly.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from src.data.loaders.historical_loader import HistoricalDataLoader
from src.utils.feature_config import load_feature_config
from src.features.pipeline import FeaturePipeline

# Test configuration
TEST_DATE = '20250205'
TEST_PLAYER_ID = '28046691632'  # Stephen Curry
NUM_SEASONS = 1
DATA_DIR = 'data'

print("="*60)
print("CONTEXTUAL FEATURES VALIDATION TEST")
print("="*60)

# Load data
print(f"\n1. Loading data for {TEST_DATE}...")
loader = HistoricalDataLoader(data_dir=DATA_DIR)

# Load historical data
historical_logs = loader.load_historical_player_logs(
    end_date=TEST_DATE,
    num_seasons=NUM_SEASONS,
    player_ids=[TEST_PLAYER_ID]
)

print(f"   Loaded {len(historical_logs)} historical games")
print(f"   Columns: {historical_logs.columns.tolist()}")

# Load feature config
print(f"\n2. Loading contextual feature configuration...")
feature_config = load_feature_config('contextual_features')
pipeline = feature_config.build_pipeline(FeaturePipeline)

print(f"   Transformers: {len(pipeline.transformers)}")
for t in pipeline.transformers:
    print(f"     - {t.__class__.__name__}")

# Sort and transform data
print(f"\n3. Transforming data...")
training_data = historical_logs.sort_values(['playerID', 'gameDate']).copy()

# Check required columns before transform
required_cols = ['playerID', 'gameDate', 'teamAbv', 'gameID']
missing_cols = [col for col in required_cols if col not in training_data.columns]

if missing_cols:
    print(f"   WARNING: Missing required columns: {missing_cols}")
    print(f"   Available columns: {training_data.columns.tolist()}")
else:
    print(f"   [OK] All required columns present")

try:
    transformed = pipeline.fit_transform(training_data)
    print(f"   [OK] Transformation successful")
    print(f"   Shape: {transformed.shape}")

    # Check for new contextual features
    new_features = [col for col in transformed.columns if col not in training_data.columns]
    contextual_features = [f for f in new_features if f in ['home_game', 'rest_days', 'back_to_back', 'days_off_3plus']]

    if contextual_features:
        print(f"\n4. Contextual features added:")
        for feat in contextual_features:
            print(f"   - {feat}")

        # Show sample values
        print(f"\n5. Sample contextual feature values (last 10 games):")
        display_cols = ['gameDate', 'teamAbv', 'gameID'] + contextual_features
        display_cols = [col for col in display_cols if col in transformed.columns]

        sample = transformed[display_cols].tail(10)
        print(sample.to_string(index=False))

        # Statistics
        print(f"\n6. Feature statistics:")
        for feat in contextual_features:
            if feat in transformed.columns:
                print(f"   {feat}:")
                print(f"     Mean: {transformed[feat].mean():.2f}")
                print(f"     Min: {transformed[feat].min():.0f}")
                print(f"     Max: {transformed[feat].max():.0f}")
                print(f"     Null count: {transformed[feat].isna().sum()}")
    else:
        print(f"\n   [ERROR] No contextual features found!")
        print(f"   New features: {new_features[:10]}")

    print(f"\n[OK] Test completed successfully")

except Exception as e:
    print(f"\n[ERROR] Error during transformation: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("="*60)
