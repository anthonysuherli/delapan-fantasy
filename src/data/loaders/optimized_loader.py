import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Callable
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


class OptimizedParquetLoader:
    """
    Optimized data loader with parallel I/O, prefetching, and caching.

    Features:
    - Parallel file reading with ThreadPoolExecutor
    - Memory-efficient streaming for large datasets
    - LRU cache for frequently accessed data
    - Batch processing support
    - GPU-ready data format
    """

    def __init__(
        self,
        base_dir: str = 'data/inputs',
        max_workers: int = 4,
        cache_size: int = 100,
        prefetch_size: int = 2
    ):
        """
        Initialize optimized loader.

        Args:
            base_dir: Base directory for data files
            max_workers: Number of parallel workers for file I/O
            cache_size: Maximum cache entries (in MB)
            prefetch_size: Number of batches to prefetch
        """
        self.base_dir = Path(base_dir)
        self.max_workers = max_workers
        self.cache_size = cache_size
        self.prefetch_size = prefetch_size
        self.cache = {}
        self.cache_keys = []

        logger.info(f"OptimizedParquetLoader initialized: workers={max_workers}, cache={cache_size}MB")

    def _read_parquet_file(self, file_path: Path) -> pd.DataFrame:
        """Read single parquet file with error handling."""
        try:
            return pd.read_parquet(file_path)
        except Exception as e:
            logger.warning(f"Failed to read {file_path}: {e}")
            return pd.DataFrame()

    def _cache_key(self, data_type: str, start_date: Optional[str], end_date: Optional[str]) -> str:
        """Generate cache key."""
        return f"{data_type}_{start_date}_{end_date}"

    def _update_cache(self, key: str, data: pd.DataFrame):
        """Update LRU cache."""
        if key in self.cache:
            self.cache_keys.remove(key)
        elif len(self.cache_keys) >= self.cache_size:
            oldest_key = self.cache_keys.pop(0)
            del self.cache[oldest_key]

        self.cache[key] = data
        self.cache_keys.append(key)

    def load_parallel(
        self,
        data_type: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Load data with parallel file reading.

        Args:
            data_type: Type of data (e.g., 'box_scores', 'dfs_salaries')
            start_date: Start date in YYYYMMDD format
            end_date: End date in YYYYMMDD format
            use_cache: Whether to use caching

        Returns:
            DataFrame with loaded data
        """
        cache_key = self._cache_key(data_type, start_date, end_date)

        if use_cache and cache_key in self.cache:
            logger.debug(f"Cache hit: {cache_key}")
            return self.cache[cache_key].copy()

        data_dir = self.base_dir / data_type
        if not data_dir.exists():
            logger.warning(f"Directory not found: {data_dir}")
            return pd.DataFrame()

        files = list(data_dir.glob("*.parquet"))

        if start_date or end_date:
            files = self._filter_files_by_date(files, start_date, end_date)

        if not files:
            return pd.DataFrame()

        logger.debug(f"Loading {len(files)} files in parallel with {self.max_workers} workers")

        dfs = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_file = {executor.submit(self._read_parquet_file, f): f for f in files}

            for future in as_completed(future_to_file):
                df = future.result()
                if not df.empty:
                    dfs.append(df)

        if not dfs:
            return pd.DataFrame()

        result = pd.concat(dfs, ignore_index=True)

        if use_cache:
            self._update_cache(cache_key, result)

        logger.debug(f"Loaded {len(result)} rows from {len(dfs)} files")
        return result

    def _filter_files_by_date(
        self,
        files: List[Path],
        start_date: Optional[str],
        end_date: Optional[str]
    ) -> List[Path]:
        """Filter files by date range based on filename pattern."""
        filtered = []
        for f in files:
            try:
                parts = f.stem.split('_')
                date_str = parts[-1]

                if len(date_str) == 8 and date_str.isdigit():
                    if start_date and date_str < start_date:
                        continue
                    if end_date and date_str > end_date:
                        continue
                    filtered.append(f)
                else:
                    filtered.append(f)
            except (IndexError, ValueError):
                filtered.append(f)

        return filtered

    def load_streaming(
        self,
        data_type: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        batch_size: int = 10000
    ):
        """
        Generator for streaming large datasets in batches.

        Args:
            data_type: Type of data
            start_date: Start date
            end_date: End date
            batch_size: Number of rows per batch

        Yields:
            DataFrame batches
        """
        data_dir = self.base_dir / data_type
        if not data_dir.exists():
            return

        files = list(data_dir.glob("*.parquet"))

        if start_date or end_date:
            files = self._filter_files_by_date(files, start_date, end_date)

        for file in files:
            try:
                parquet_file = pq.ParquetFile(file)

                for batch in parquet_file.iter_batches(batch_size=batch_size):
                    df = batch.to_pandas()
                    yield df

            except Exception as e:
                logger.warning(f"Error streaming {file}: {e}")
                continue

    def preload_data(
        self,
        data_types: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Preload multiple data types in parallel.

        Args:
            data_types: List of data types to load
            start_date: Start date
            end_date: End date

        Returns:
            Dictionary mapping data_type to DataFrame
        """
        logger.info(f"Preloading {len(data_types)} data types in parallel")

        results = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_type = {
                executor.submit(self.load_parallel, dt, start_date, end_date): dt
                for dt in data_types
            }

            for future in as_completed(future_to_type):
                data_type = future_to_type[future]
                try:
                    results[data_type] = future.result()
                    logger.debug(f"Preloaded {data_type}: {len(results[data_type])} rows")
                except Exception as e:
                    logger.error(f"Failed to preload {data_type}: {e}")
                    results[data_type] = pd.DataFrame()

        return results

    def clear_cache(self):
        """Clear the data cache."""
        self.cache.clear()
        self.cache_keys.clear()
        logger.info("Cache cleared")

    def get_cache_stats(self) -> Dict:
        """Get cache statistics."""
        total_size_mb = sum(
            df.memory_usage(deep=True).sum() / (1024 * 1024)
            for df in self.cache.values()
        )

        return {
            'entries': len(self.cache),
            'size_mb': total_size_mb,
            'keys': self.cache_keys.copy()
        }


class TensorFlowDataLoader:
    """
    TensorFlow-based data loader with prefetching and GPU acceleration.

    Requires: tensorflow
    """

    def __init__(
        self,
        base_dir: str = 'data/inputs',
        prefetch_buffer: int = 2,
        num_parallel_reads: int = 4
    ):
        """
        Initialize TensorFlow data loader.

        Args:
            base_dir: Base directory for data files
            prefetch_buffer: Number of batches to prefetch
            num_parallel_reads: Number of parallel file readers
        """
        try:
            import tensorflow as tf
            self.tf = tf
            self.available = True
        except ImportError:
            logger.warning("TensorFlow not available, falling back to OptimizedParquetLoader")
            self.available = False
            return

        self.base_dir = Path(base_dir)
        self.prefetch_buffer = prefetch_buffer
        self.num_parallel_reads = num_parallel_reads

        logger.info(f"TensorFlowDataLoader initialized with GPU support: {len(tf.config.list_physical_devices('GPU'))} GPUs")

    def create_dataset(
        self,
        data_type: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        batch_size: int = 256,
        shuffle: bool = False,
        shuffle_buffer: int = 10000
    ):
        """
        Create tf.data.Dataset with optimized pipeline.

        Args:
            data_type: Type of data
            start_date: Start date
            end_date: End date
            batch_size: Batch size
            shuffle: Whether to shuffle data
            shuffle_buffer: Shuffle buffer size

        Returns:
            tf.data.Dataset configured for optimal performance
        """
        if not self.available:
            raise ImportError("TensorFlow not available")

        data_dir = self.base_dir / data_type
        if not data_dir.exists():
            raise FileNotFoundError(f"Directory not found: {data_dir}")

        files = [str(f) for f in data_dir.glob("*.parquet")]

        if start_date or end_date:
            files = self._filter_files_by_date_str(files, start_date, end_date)

        if not files:
            raise ValueError(f"No files found for {data_type}")

        def parse_parquet_file(filename):
            """Parse parquet file using pandas and convert to tensors."""
            df = pd.read_parquet(filename.numpy().decode('utf-8'))

            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

            features = {}
            for col in numeric_cols:
                features[col] = df[col].values.astype(np.float32)

            return features

        dataset = self.tf.data.Dataset.from_tensor_slices(files)

        dataset = dataset.interleave(
            lambda x: self.tf.data.Dataset.from_tensors(x).map(
                lambda y: self.tf.py_function(parse_parquet_file, [y], self.tf.float32),
                num_parallel_calls=self.tf.data.AUTOTUNE
            ),
            cycle_length=self.num_parallel_reads,
            num_parallel_calls=self.tf.data.AUTOTUNE
        )

        if shuffle:
            dataset = dataset.shuffle(shuffle_buffer)

        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(self.tf.data.AUTOTUNE)

        return dataset

    def _filter_files_by_date_str(
        self,
        files: List[str],
        start_date: Optional[str],
        end_date: Optional[str]
    ) -> List[str]:
        """Filter file paths by date range."""
        filtered = []
        for f in files:
            try:
                fname = Path(f).stem
                parts = fname.split('_')
                date_str = parts[-1]

                if len(date_str) == 8 and date_str.isdigit():
                    if start_date and date_str < start_date:
                        continue
                    if end_date and date_str > end_date:
                        continue
                    filtered.append(f)
                else:
                    filtered.append(f)
            except (IndexError, ValueError):
                filtered.append(f)

        return filtered


class PyTorchDataLoader:
    """
    PyTorch-based data loader with multi-worker support.

    Requires: torch
    """

    def __init__(
        self,
        base_dir: str = 'data/inputs',
        num_workers: int = 4,
        pin_memory: bool = True
    ):
        """
        Initialize PyTorch data loader.

        Args:
            base_dir: Base directory for data files
            num_workers: Number of worker processes
            pin_memory: Pin memory for faster GPU transfer
        """
        try:
            import torch
            from torch.utils.data import Dataset, DataLoader
            self.torch = torch
            self.Dataset = Dataset
            self.DataLoader = DataLoader
            self.available = True
        except ImportError:
            logger.warning("PyTorch not available")
            self.available = False
            return

        self.base_dir = Path(base_dir)
        self.num_workers = num_workers
        self.pin_memory = pin_memory and torch.cuda.is_available()

        logger.info(f"PyTorchDataLoader initialized: workers={num_workers}, GPU={torch.cuda.is_available()}")

    def create_dataset(
        self,
        data_type: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ):
        """
        Create PyTorch Dataset.

        Args:
            data_type: Type of data
            start_date: Start date
            end_date: End date

        Returns:
            ParquetDataset instance
        """
        if not self.available:
            raise ImportError("PyTorch not available")

        return ParquetDataset(
            self.base_dir,
            data_type,
            start_date,
            end_date
        )

    def create_dataloader(
        self,
        dataset,
        batch_size: int = 256,
        shuffle: bool = False
    ):
        """
        Create PyTorch DataLoader with optimized settings.

        Args:
            dataset: PyTorch Dataset
            batch_size: Batch size
            shuffle: Whether to shuffle

        Returns:
            DataLoader configured for optimal performance
        """
        if not self.available:
            raise ImportError("PyTorch not available")

        return self.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False,
            prefetch_factor=2 if self.num_workers > 0 else None
        )


class ParquetDataset:
    """PyTorch Dataset for Parquet files."""

    def __init__(
        self,
        base_dir: Path,
        data_type: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ):
        """
        Initialize dataset.

        Args:
            base_dir: Base directory
            data_type: Type of data
            start_date: Start date
            end_date: End date
        """
        try:
            from torch.utils.data import Dataset
        except ImportError:
            raise ImportError("PyTorch not available")

        self.base_dir = base_dir
        self.data_type = data_type

        data_dir = base_dir / data_type
        files = list(data_dir.glob("*.parquet"))

        if start_date or end_date:
            files = self._filter_files(files, start_date, end_date)

        self.data = pd.concat(
            [pd.read_parquet(f) for f in files],
            ignore_index=True
        )

        logger.info(f"ParquetDataset initialized: {len(self.data)} rows")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        import torch
        row = self.data.iloc[idx]

        numeric_data = row.select_dtypes(include=[np.number]).values.astype(np.float32)
        return torch.from_numpy(numeric_data)

    def _filter_files(self, files, start_date, end_date):
        """Filter files by date."""
        filtered = []
        for f in files:
            try:
                parts = f.stem.split('_')
                date_str = parts[-1]

                if len(date_str) == 8 and date_str.isdigit():
                    if start_date and date_str < start_date:
                        continue
                    if end_date and date_str > end_date:
                        continue
                    filtered.append(f)
                else:
                    filtered.append(f)
            except (IndexError, ValueError):
                filtered.append(f)

        return filtered
