# GPU Training Guide

Guide for running GPU-accelerated training on cloud machines.

## Requirements

### Hardware
- NVIDIA GPU (CUDA-compatible)
- Recommended: 8GB+ VRAM for per-player models
- 32+ CPU cores for parallel per-player training
- 128GB+ RAM for large datasets

### Software
- CUDA Toolkit 11.2+ (12.x recommended)
- cuDNN 8.1+
- Python 3.8+

### Python Packages
```bash
pip install xgboost>=2.0.0
pip install cudf-cu12 cupy-cuda12x
pip install tensorflow-gpu
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Note: cuDF and cuPy required for GPU-accelerated predictions (XGBoost model.predict with GPU DataFrames)

## Verify GPU Setup

### Check CUDA availability
```bash
nvidia-smi
```

### Verify XGBoost GPU support
```python
import xgboost as xgb
print(xgb.config.get_config())
```

### Verify TensorFlow GPU
```python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
```

### Verify PyTorch GPU
```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

## Configuration

### XGBoost GPU Parameters

Updated config/models/xgboost_a100.yaml (XGBoost 2.0+):
```yaml
model:
  type: "xgboost"
  params:
    max_depth: 10
    learning_rate: 0.05
    n_estimators: 500
    min_child_weight: 3
    subsample: 0.85
    colsample_bytree: 0.85
    objective: "reg:squarederror"
    random_state: 42
    tree_method: "hist"
    device: "cuda:0"
    max_bin: 512
```

Key GPU parameters (XGBoost 2.0+):
- `tree_method: "hist"` - Histogram-based tree construction (works on CPU/GPU)
- `device: "cuda:0"` - GPU device specification (cuda:0, cuda:1, etc.)

Deprecated parameters (do not use):
- `tree_method: "gpu_hist"` - Deprecated since XGBoost 2.0.0
- `predictor: "gpu_predictor"` - Deprecated, automatically inferred from device
- `gpu_id: 0` - Deprecated, use device parameter instead

### Multi-GPU Setup

For multiple GPUs, specify device per process:
```python
model_params_gpu0 = {**base_params, 'device': 'cuda:0'}
model_params_gpu1 = {**base_params, 'device': 'cuda:1'}
```

## Usage

### GPU-Accelerated Backtest Script

New script: scripts/run_backtest_gpu.py

Basic usage:
```bash
python scripts/run_backtest_gpu.py \
  --db-path data/nba_dfs.db \
  --train-start 20241001 \
  --train-end 20241231 \
  --test-start 20250101 \
  --test-end 20250131 \
  --per-player \
  --n-jobs -1 \
  --num-workers 16 \
  --gpu-id 0
```

Parameters:
- `--gpu-id`: GPU device ID (default: 0)
- `--n-jobs`: Parallel CPU jobs for per-player models (-1 = all cores)
- `--num-workers`: Data loader workers (default: 16)
- `--model-config`: Path to model config YAML
- `--feature-config`: Feature configuration name

### Optimized Data Loading

The backtest automatically uses optimized data loaders when available.

Priority: TensorFlow > PyTorch > Fallback

To force specific backend:
```python
from src.data.loaders.optimized_historical_loader import OptimizedHistoricalDataLoader

loader = OptimizedHistoricalDataLoader(
    storage,
    loader_type='tensorflow',
    num_workers=16
)
```

## Performance Optimization

### Per-Player Models with GPU

Hybrid CPU-GPU strategy:
- CPU: Parallel per-player model training (joblib)
- GPU: XGBoost tree construction per model
- Data: TensorFlow parallel loading

Expected speedup: 5-10x vs CPU-only

Configuration:
```bash
python scripts/run_backtest_gpu.py \
  --per-player \
  --n-jobs -1 \
  --num-workers 16 \
  --model-config config/models/xgboost_default.yaml
```

### Batch Processing

For slate-level models, increase batch size:
```python
from src.data.loaders.tensorflow_loader import TensorFlowDataLoader

loader = TensorFlowDataLoader()
dataset = loader.create_cached_dataset(
    data=training_data,
    feature_columns=feature_cols,
    target_column='target',
    batch_size=2048,
    shuffle=True
)
```

### Memory Management

Monitor GPU memory:
```bash
watch -n 1 nvidia-smi
```

If OOM errors occur:
1. Reduce n_estimators in model config
2. Reduce max_depth
3. Train fewer players per batch
4. Reduce num_workers for data loading

## Benchmarking

Compare CPU vs GPU performance:
```bash
python scripts/benchmark_data_loaders.py
```

Expected results on large machine:
- Data loading: 2-5x speedup with TensorFlow/PyTorch loaders
- XGBoost GPU training: 3-10x speedup per model
- Per-player models (500 players): 30 min vs 3-5 hours (CPU)

## Troubleshooting

### CUDA Out of Memory
Reduce batch size or n_estimators:
```yaml
params:
  n_estimators: 100
  max_depth: 5
```

### GPU Not Detected
Check CUDA installation:
```bash
nvcc --version
echo $CUDA_HOME
echo $LD_LIBRARY_PATH
```

### Slow Data Loading
Increase num_workers:
```bash
--num-workers 32
```

Or use RAM disk for data directory.

### Device Mismatch Warnings
If you see warnings about CPU/GPU data mismatch:
```
WARNING: Falling back to prediction using DMatrix due to mismatched devices
```

Solution: Install cuDF and cuPy for GPU-accelerated predictions:
```bash
pip install cudf-cu12 cupy-cuda12x
```

The XGBoostModel class automatically converts pandas DataFrames to cuDF when `device='cuda'`.

### Model Training Errors
Verify XGBoost version:
```bash
pip install xgboost>=2.0.0 --upgrade
```

Verify cuDF/cuPy installation:
```bash
pip list | grep cu
```

Expected output:
```
cudf-cu12        24.x.x
cupy-cuda12x     13.x.x
```

## Cloud-Specific Configurations

### AWS SageMaker
See scripts/run_sagemaker_backtest.py for GPU instance configuration.

Recommended instance: ml.p3.2xlarge (V100 GPU, 8 vCPUs, 61 GB RAM)

### Google Cloud Platform
Recommended: n1-highmem-16 + NVIDIA Tesla T4

Attach GPU:
```bash
gcloud compute instances attach-gpu INSTANCE_NAME \
  --gpu-type nvidia-tesla-t4 \
  --zone us-central1-a
```

### Azure
Recommended: NC6s_v3 (V100 GPU, 6 vCPUs, 112 GB RAM)

## Example Workflows

### Full Season Backtest with GPU
```bash
python scripts/run_backtest_gpu.py \
  --db-path /mnt/data/nba_dfs.db \
  --train-start 20241001 \
  --train-end 20250131 \
  --test-start 20250201 \
  --test-end 20250430 \
  --per-player \
  --n-jobs -1 \
  --num-workers 32 \
  --gpu-id 0 \
  --recalibrate-days 7 \
  --save-models \
  --output-dir /mnt/data/backtest_results
```

### Resume Failed Run
```bash
python scripts/run_backtest_gpu.py \
  --db-path /mnt/data/nba_dfs.db \
  --train-start 20241001 \
  --train-end 20250131 \
  --test-start 20250201 \
  --test-end 20250430 \
  --per-player \
  --n-jobs -1 \
  --resume-from 20250115_143022
```

### Hyperparameter Tuning on GPU
```bash
python scripts/optimize_hyperparameters.py \
  --use-gpu \
  --gpu-id 0 \
  --n-trials 100
```

## Performance Metrics

Target performance on 32-core + V100 GPU machine:

| Task | CPU-only | GPU-accelerated | Speedup |
|------|----------|-----------------|---------|
| Data loading (1 season) | 120s | 30s | 4x |
| Per-player model (10 games) | 0.5s | 0.05s | 10x |
| Full backtest (30 days, 500 players) | 4h | 30min | 8x |
| Hyperparameter optimization (100 trials) | 8h | 1.5h | 5.3x |

## Best Practices

1. Profile before optimizing - identify bottlenecks
2. Use GPU for XGBoost tree construction
3. Use CPU parallelism for per-player model batching
4. Use optimized data loaders (TensorFlow/PyTorch)
5. Monitor GPU utilization with nvidia-smi
6. Cache preprocessed features when possible
7. Use checkpointing for long-running backtests
8. Balance num_workers with available CPU cores

## References

- XGBoost GPU documentation: https://xgboost.readthedocs.io/en/latest/gpu/
- TensorFlow GPU guide: https://www.tensorflow.org/guide/gpu
- PyTorch CUDA semantics: https://pytorch.org/docs/stable/notes/cuda.html
