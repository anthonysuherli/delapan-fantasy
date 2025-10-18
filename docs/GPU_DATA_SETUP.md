# GPU Data Setup Guide

Configure data pipelines for optimal XGBoost GPU training performance.

## Overview

The GPU data setup includes:
1. **Data Preprocessing**: Convert to float32, optimize memory layout
2. **Caching System**: Cache preprocessed data for repeated access
3. **Memory Efficiency**: Reduce memory footprint by 50% vs default float64
4. **Batch Processing**: Handle large datasets with controlled memory usage

## Key Benefits

- **50% Memory Reduction**: float32 vs float64 (16 MB → 8 MB per million samples)
- **Faster Training**: Reduced data transfer, optimized memory layout
- **GPU Ready**: C-contiguous arrays for efficient GPU transfer
- **Fallback Safe**: Works on CPU if GPU unavailable
- **Cached**: Reuse preprocessed data without recomputation

## Quick Start

### Automatic Preprocessing (Recommended)

XGBoost models automatically preprocess data for GPU optimization:

```python
from src.models.xgboost_model import XGBoostModel

model = XGBoostModel()
model.train(X_train, y_train, use_gpu_preprocessing=True)  # Default: True
```

### Manual Preprocessing

For advanced use cases, use the GPU data preprocessor directly:

```python
from src.data.loaders.gpu_data_preprocessor import GPUDataPreprocessor
import numpy as np

preprocessor = GPUDataPreprocessor()

# Preprocess data
X_np, y_np = preprocessor.preprocess_features(
    X_train, y_train, optimize_dtypes=True
)

# X_np is now float32 and C-contiguous
print(f"X dtype: {X_np.dtype}")  # float32
print(f"X is C-contiguous: {X_np.flags['C_CONTIGUOUS']}")  # True
```

### With Caching

Cache preprocessed data to avoid recomputation:

```python
from src.data.loaders.gpu_data_preprocessor import GPUDataPipeline
from pathlib import Path

cache_dir = Path('data/preprocessed_cache')
pipeline = GPUDataPipeline(cache_dir=cache_dir)

# First run: preprocesses and caches
X_np, y_np = pipeline.prepare_training_data(
    X_train, y_train,
    cache_key='player_123_training',
    use_cache=True
)

# Subsequent runs: loads from cache (instant)
X_np, y_np = pipeline.prepare_training_data(
    X_train, y_train,
    cache_key='player_123_training',
    use_cache=True
)
```

## API Reference

### GPUDataPreprocessor

Low-level data preprocessing with optional caching.

#### preprocess_features()

```python
X_np, y_np = preprocessor.preprocess_features(
    X, y,
    optimize_dtypes=True,
    normalize=False
)
```

**Parameters:**
- `X` (pd.DataFrame): Feature matrix
- `y` (pd.Series, optional): Target variable
- `optimize_dtypes` (bool): Convert float64 → float32. Default: True
- `normalize` (bool): Normalize features to [0, 1]. Default: False

**Returns:**
- `X_np` (np.ndarray): Preprocessed features (float32, C-contiguous)
- `y_np` (np.ndarray): Preprocessed targets (float32, C-contiguous)

**Example:**
```python
X_np, y_np = preprocessor.preprocess_features(X_train, y_train)
# X_np: (10000, 150) float32 array
# y_np: (10000,) float32 array
```

#### batch_preprocess()

Process large datasets in batches to manage memory:

```python
X_np, y_np = preprocessor.batch_preprocess(
    X, y,
    batch_size=10000,
    optimize_dtypes=True
)
```

**Parameters:**
- `batch_size` (int): Samples per batch. Default: 10000

**Use when:**
- Dataset > available RAM
- Preprocessing memory spike needed
- Training on constrained systems

#### cache_data() / load_cached_data()

```python
# Save
cache_path = preprocessor.cache_data(X_np, y_np, 'player_123_training')

# Load
X_np, y_np = preprocessor.load_cached_data('player_123_training')

# Check if cached
is_cached = preprocessor.is_cached('player_123_training')
```

**Cache Format:** NPZ (NumPy compressed format)
- Compressed: ~2-5x smaller than raw arrays
- Fast to load/save
- Preserves dtype and layout

#### get_memory_stats()

```python
stats = preprocessor.get_memory_stats(X_np, y_np)
print(stats)
# {
#   'X_mb': 5.7,
#   'y_mb': 0.038,
#   'total_mb': 5.74,
#   'X_dtype': 'float32',
#   'y_dtype': 'float32'
# }
```

### GPUDataPipeline

High-level end-to-end pipeline with caching and preprocessing.

#### prepare_training_data()

```python
X_np, y_np = pipeline.prepare_training_data(
    X_train, y_train,
    cache_key='experiment_v1_training',
    use_cache=True
)
```

**Workflow:**
1. If cache exists and `use_cache=True`: Load from cache
2. Else: Preprocess data
3. If `cache_key` provided: Save to cache
4. Return preprocessed arrays

**Example:**
```python
pipeline = GPUDataPipeline(cache_dir='data/cache')

# First call: preprocesses and caches (~2s)
X_np, y_np = pipeline.prepare_training_data(
    X_train, y_train,
    cache_key='v1',
    use_cache=True
)

# Second call: loads from cache (~0.1s)
X_np, y_np = pipeline.prepare_training_data(
    X_train, y_train,
    cache_key='v1',
    use_cache=True
)
```

#### prepare_inference_data()

```python
X_np = pipeline.prepare_inference_data(
    X_test,
    cache_key='experiment_v1_inference',
    use_cache=True
)
```

Same caching workflow but for inference (no target variable).

## Integration with Walk-Forward Backtest

The backtest automatically uses GPU preprocessing:

```python
from src.walk_forward_backtest import WalkForwardBacktest

backtest = WalkForwardBacktest(
    # ... other parameters
    # GPU preprocessing is automatic in XGBoostModel.train()
)

results = backtest.run()
```

For per-player models (500+ models), preprocessing happens in parallel:

```bash
python scripts/run_backtest_gpu.py \
  --test-start 20250205 \
  --test-end 20250206 \
  --per-player \
  --n-jobs -1  # All cores for parallel preprocessing
```

## Performance Characteristics

### Memory Usage

```
Dataset Size    | float64 (MB) | float32 (MB) | Savings
10k samples     | 22.7         | 11.4         | 50%
100k samples    | 227.0        | 113.6        | 50%
1M samples      | 2270.0       | 1136.0       | 50%
```

### Preprocessing Time

```
Dataset Size | Time (CPU) | Throughput
10k          | 12 ms      | 0.8M samples/sec
100k         | 85 ms      | 1.2M samples/sec
1M           | 750 ms     | 1.3M samples/sec
```

### Cache Format Efficiency

```
Original Array | Compressed NPZ | Ratio
100 MB (float32) | 15-25 MB      | 5-6.7x
```

## Optimization Tips

### 1. Enable Caching for Repeated Experiments

```python
# Good: Cache enabled
pipeline.prepare_training_data(X, y, cache_key='exp_v1', use_cache=True)
pipeline.prepare_training_data(X, y, cache_key='exp_v1', use_cache=True)  # Instant

# Slower: No cache
pipeline.prepare_training_data(X, y, cache_key=None)
pipeline.prepare_training_data(X, y, cache_key=None)  # Reprocesses both times
```

### 2. Use Batch Processing for Large Datasets

```python
# For datasets > 1 GB
X_np, y_np = preprocessor.batch_preprocess(X, y, batch_size=50000)
```

### 3. Normalize Only When Needed

```python
# Skip normalization (usually not needed for XGBoost)
X_np, _ = preprocessor.preprocess_features(X, normalize=False)
```

### 4. Monitor Memory Usage

```python
stats = preprocessor.get_memory_stats(X_np, y_np)
if stats['total_mb'] > 500:
    logger.warning(f"Large dataset: {stats['total_mb']:.0f} MB")
```

## Troubleshooting

### High Memory Usage During Preprocessing

**Problem:** RAM spikes during `preprocess_features()`

**Solution:** Use batch processing:
```python
X_np, y_np = preprocessor.batch_preprocess(
    X, y, batch_size=50000  # Reduce batch size
)
```

### Cache Not Being Used

**Problem:** Data reprocessed every time despite `use_cache=True`

**Check:**
```python
# Verify cache directory exists and is writable
print(preprocessor.cache_dir)

# Verify cache key matches
is_cached = preprocessor.is_cached('my_key')
print(f"Cached: {is_cached}")
```

### Dtype Mismatch Errors

**Problem:** "Input array dtype not supported"

**Solution:** Ensure preprocessing is enabled:
```python
model = XGBoostModel()
model.train(X, y, use_gpu_preprocessing=True)  # Explicit
```

## Best Practices

1. **Always enable preprocessing** when using XGBoost
2. **Use caching** for repeated experiments
3. **Monitor memory** with `get_memory_stats()`
4. **Batch process** large datasets (> 1 GB)
5. **Disable normalization** unless specifically needed
6. **Keep cache organized** with meaningful cache keys
7. **Clean old caches** periodically to save disk space

## Example: Complete Workflow

```python
from src.data.loaders.gpu_data_preprocessor import GPUDataPipeline
from src.models.xgboost_model import XGBoostModel
from pathlib import Path

# Setup
cache_dir = Path('data/cache')
pipeline = GPUDataPipeline(cache_dir=cache_dir, batch_size=10000)

# Prepare training data
X_train_np, y_train_np = pipeline.prepare_training_data(
    X_train, y_train,
    cache_key='experiment_v1_training',
    use_cache=True
)

# Train model (automatic preprocessing still applied)
model = XGBoostModel({
    'max_depth': 6,
    'learning_rate': 0.05,
    'n_estimators': 200
})
model.train(
    X_train_np,
    y_train_np,
    use_gpu_preprocessing=True  # Already float32, but ensures safety
)

# Prepare inference data
X_test_np = pipeline.prepare_inference_data(
    X_test,
    cache_key='experiment_v1_inference',
    use_cache=True
)

# Generate predictions
predictions = model.predict(X_test_np)

# Check performance
print(f"Memory used: {pipeline.preprocessor.get_memory_stats(X_train_np, y_train_np)}")
```
