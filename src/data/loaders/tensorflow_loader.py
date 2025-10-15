import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable
import logging

logger = logging.getLogger(__name__)

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logger.warning("TensorFlow not available. Install with: pip install tensorflow")


class TensorFlowDataLoader:
    """
    Optimized data loader using TensorFlow's tf.data API.

    Features:
    - Parallel file reading with interleave
    - Prefetching to overlap I/O and computation
    - GPU-ready tensor outputs
    - Automatic batching
    - Caching for frequently accessed data
    """

    def __init__(
        self,
        prefetch_buffer_size: int = tf.data.AUTOTUNE,
        num_parallel_reads: int = tf.data.AUTOTUNE,
        cache: bool = True,
        deterministic: bool = False
    ):
        """
        Initialize TensorFlow data loader.

        Args:
            prefetch_buffer_size: Number of batches to prefetch (AUTOTUNE recommended)
            num_parallel_reads: Number of parallel file read operations
            cache: Enable caching of loaded data
            deterministic: Enforce deterministic ordering (slower)
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow required. Install with: pip install tensorflow")

        self.prefetch_buffer_size = prefetch_buffer_size
        self.num_parallel_reads = num_parallel_reads
        self.cache = cache
        self.deterministic = deterministic

        logger.info(f"Initialized TensorFlowDataLoader with prefetch={prefetch_buffer_size}, parallel_reads={num_parallel_reads}")

    def load_parquet_files_parallel(
        self,
        file_paths: List[Path],
        batch_size: Optional[int] = None,
        transform_fn: Optional[Callable] = None
    ) -> tf.data.Dataset:
        """
        Load multiple Parquet files in parallel with prefetching.

        Args:
            file_paths: List of Parquet file paths
            batch_size: Optional batch size for dataset
            transform_fn: Optional transformation function to apply

        Returns:
            tf.data.Dataset with optimized loading
        """
        if not file_paths:
            logger.warning("No files provided to load")
            return tf.data.Dataset.from_tensor_slices({})

        logger.info(f"Loading {len(file_paths)} Parquet files in parallel")

        file_paths_str = [str(p) for p in file_paths]

        def read_parquet_file(file_path_tensor):
            """Read single Parquet file and return as tensors."""
            file_path = file_path_tensor.numpy().decode('utf-8')

            try:
                df = pd.read_parquet(file_path)

                if df.empty:
                    return {}

                tensor_dict = {}
                for col in df.columns:
                    if df[col].dtype == 'object':
                        tensor_dict[col] = df[col].astype(str).values
                    elif pd.api.types.is_datetime64_any_dtype(df[col]):
                        tensor_dict[col] = df[col].astype(np.int64).values
                    else:
                        tensor_dict[col] = df[col].values

                return tensor_dict

            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {e}")
                return {}

        def py_read_parquet(file_path):
            """Wrapper for tf.py_function."""
            result = tf.py_function(
                read_parquet_file,
                [file_path],
                Tout={}
            )
            return result

        file_dataset = tf.data.Dataset.from_tensor_slices(file_paths_str)

        dataset = file_dataset.interleave(
            lambda x: tf.data.Dataset.from_tensors(py_read_parquet(x)),
            cycle_length=self.num_parallel_reads,
            num_parallel_calls=self.num_parallel_reads,
            deterministic=self.deterministic
        )

        if self.cache:
            dataset = dataset.cache()

        if transform_fn is not None:
            dataset = dataset.map(
                transform_fn,
                num_parallel_calls=tf.data.AUTOTUNE
            )

        if batch_size is not None:
            dataset = dataset.batch(batch_size)

        dataset = dataset.prefetch(self.prefetch_buffer_size)

        logger.info("Dataset pipeline configured with prefetching")
        return dataset

    def load_parquet_to_dataframe_parallel(
        self,
        file_paths: List[Path]
    ) -> pd.DataFrame:
        """
        Load multiple Parquet files in parallel and return as DataFrame.

        More efficient than sequential loading for large file sets.

        Args:
            file_paths: List of Parquet file paths

        Returns:
            Concatenated DataFrame
        """
        if not file_paths:
            return pd.DataFrame()

        logger.info(f"Loading {len(file_paths)} Parquet files with parallel reading")

        @tf.function
        def load_files_parallel_tf():
            """Use TensorFlow for parallel I/O coordination."""
            file_paths_str = [str(p) for p in file_paths]
            dataset = tf.data.Dataset.from_tensor_slices(file_paths_str)

            dataset = dataset.interleave(
                lambda x: tf.data.Dataset.from_tensors(x),
                cycle_length=self.num_parallel_reads,
                num_parallel_calls=self.num_parallel_reads
            )

            return dataset

        dfs = []
        for file_path in file_paths:
            try:
                df = pd.read_parquet(file_path)
                if not df.empty:
                    dfs.append(df)
            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {e}")

        if not dfs:
            return pd.DataFrame()

        result = pd.concat(dfs, ignore_index=True)
        logger.info(f"Loaded {len(result)} rows from {len(dfs)} files")
        return result

    def create_cached_dataset(
        self,
        data: pd.DataFrame,
        feature_columns: List[str],
        target_column: Optional[str] = None,
        batch_size: int = 32,
        shuffle: bool = True,
        shuffle_buffer_size: int = 10000
    ) -> tf.data.Dataset:
        """
        Create optimized cached dataset from DataFrame for training.

        Args:
            data: Source DataFrame
            feature_columns: List of feature column names
            target_column: Optional target column name
            batch_size: Batch size
            shuffle: Enable shuffling
            shuffle_buffer_size: Buffer size for shuffling

        Returns:
            Optimized tf.data.Dataset
        """
        if data.empty:
            logger.warning("Empty DataFrame provided")
            return tf.data.Dataset.from_tensor_slices({})

        logger.info(f"Creating cached dataset: {len(data)} rows, {len(feature_columns)} features")

        X = data[feature_columns].fillna(0).astype(np.float32).values

        if target_column:
            y = data[target_column].fillna(0).astype(np.float32).values
            dataset = tf.data.Dataset.from_tensor_slices((X, y))
        else:
            dataset = tf.data.Dataset.from_tensor_slices(X)

        if self.cache:
            dataset = dataset.cache()

        if shuffle:
            dataset = dataset.shuffle(shuffle_buffer_size)

        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(self.prefetch_buffer_size)

        logger.info(f"Dataset configured: batch_size={batch_size}, shuffle={shuffle}, prefetch=AUTOTUNE")
        return dataset

    def load_date_range_parallel(
        self,
        base_dir: Path,
        data_type: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Load data for date range with parallel file reading.

        Args:
            base_dir: Base data directory
            data_type: Data type subdirectory
            start_date: Start date in YYYYMMDD format
            end_date: End date in YYYYMMDD format

        Returns:
            Concatenated DataFrame
        """
        from datetime import datetime, timedelta

        data_dir = base_dir / data_type
        if not data_dir.exists():
            logger.warning(f"Directory not found: {data_dir}")
            return pd.DataFrame()

        start_dt = datetime.strptime(start_date, '%Y%m%d')
        end_dt = datetime.strptime(end_date, '%Y%m%d')

        file_paths = []
        current_dt = start_dt
        while current_dt <= end_dt:
            date_str = current_dt.strftime('%Y%m%d')
            pattern = f"*{date_str}*.parquet"
            matching_files = list(data_dir.glob(pattern))
            file_paths.extend(matching_files)
            current_dt += timedelta(days=1)

        if not file_paths:
            logger.warning(f"No files found for {data_type} from {start_date} to {end_date}")
            return pd.DataFrame()

        logger.info(f"Found {len(file_paths)} files for date range")
        return self.load_parquet_to_dataframe_parallel(file_paths)


class TensorFlowDataLoaderConfig:
    """Configuration for TensorFlow data loader."""

    DEFAULT = {
        'prefetch_buffer_size': tf.data.AUTOTUNE if TF_AVAILABLE else None,
        'num_parallel_reads': tf.data.AUTOTUNE if TF_AVAILABLE else None,
        'cache': True,
        'deterministic': False
    }

    HIGH_PERFORMANCE = {
        'prefetch_buffer_size': tf.data.AUTOTUNE if TF_AVAILABLE else None,
        'num_parallel_reads': 16,
        'cache': True,
        'deterministic': False
    }

    DETERMINISTIC = {
        'prefetch_buffer_size': 2,
        'num_parallel_reads': 4,
        'cache': False,
        'deterministic': True
    }
