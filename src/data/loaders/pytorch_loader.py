import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

try:
    import torch
    from torch.utils.data import Dataset, DataLoader, IterableDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. Install with: pip install torch")


class ParquetDataset(Dataset):
    """
    PyTorch Dataset for loading Parquet files with caching.

    Supports lazy loading and in-memory caching for frequently accessed data.
    """

    def __init__(
        self,
        file_paths: List[Path],
        feature_columns: List[str],
        target_column: Optional[str] = None,
        transform: Optional[Callable] = None,
        cache_in_memory: bool = False
    ):
        """
        Initialize Parquet dataset.

        Args:
            file_paths: List of Parquet file paths
            feature_columns: Feature column names
            target_column: Optional target column name
            transform: Optional transformation function
            cache_in_memory: Cache all data in memory on first access
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required. Install with: pip install torch")

        self.file_paths = file_paths
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.transform = transform
        self.cache_in_memory = cache_in_memory

        self.data_cache = None
        self.file_to_indices = {}
        self.total_samples = 0

        self._build_index()

        logger.info(f"Initialized ParquetDataset: {len(file_paths)} files, {self.total_samples} samples")

    def _build_index(self):
        """Build index mapping sample indices to files."""
        current_idx = 0
        for file_path in self.file_paths:
            try:
                df = pd.read_parquet(file_path)
                num_samples = len(df)
                self.file_to_indices[current_idx] = {
                    'file_path': file_path,
                    'start_idx': current_idx,
                    'end_idx': current_idx + num_samples,
                    'num_samples': num_samples
                }
                current_idx += num_samples
            except Exception as e:
                logger.warning(f"Failed to index {file_path}: {e}")

        self.total_samples = current_idx
        logger.debug(f"Built index: {len(self.file_to_indices)} files, {self.total_samples} samples")

    def _load_all_data(self):
        """Load all data into memory cache."""
        if self.data_cache is not None:
            return

        logger.info("Loading all data into memory cache")
        dfs = []
        for file_path in self.file_paths:
            try:
                df = pd.read_parquet(file_path)
                if not df.empty:
                    dfs.append(df)
            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {e}")

        if not dfs:
            self.data_cache = pd.DataFrame()
            return

        self.data_cache = pd.concat(dfs, ignore_index=True)
        logger.info(f"Cached {len(self.data_cache)} samples in memory")

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        if self.cache_in_memory:
            if self.data_cache is None:
                self._load_all_data()
            row = self.data_cache.iloc[idx]
        else:
            for file_info in self.file_to_indices.values():
                if file_info['start_idx'] <= idx < file_info['end_idx']:
                    df = pd.read_parquet(file_info['file_path'])
                    local_idx = idx - file_info['start_idx']
                    row = df.iloc[local_idx]
                    break
            else:
                raise IndexError(f"Index {idx} out of range")

        X = row[self.feature_columns].fillna(0).astype(np.float32).values

        if self.target_column:
            y = row[self.target_column]
            if pd.isna(y):
                y = 0.0
            y = np.float32(y)

            sample = (torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
        else:
            sample = torch.tensor(X, dtype=torch.float32)

        if self.transform:
            sample = self.transform(sample)

        return sample


class StreamingParquetDataset(IterableDataset):
    """
    Streaming dataset for large Parquet files that don't fit in memory.

    Loads data in chunks with parallel processing.
    """

    def __init__(
        self,
        file_paths: List[Path],
        feature_columns: List[str],
        target_column: Optional[str] = None,
        chunk_size: int = 10000,
        transform: Optional[Callable] = None
    ):
        """
        Initialize streaming Parquet dataset.

        Args:
            file_paths: List of Parquet file paths
            feature_columns: Feature column names
            target_column: Optional target column name
            chunk_size: Number of rows to read per chunk
            transform: Optional transformation function
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required. Install with: pip install torch")

        self.file_paths = file_paths
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.chunk_size = chunk_size
        self.transform = transform

        logger.info(f"Initialized StreamingParquetDataset: {len(file_paths)} files, chunk_size={chunk_size}")

    def __iter__(self):
        for file_path in self.file_paths:
            try:
                df = pd.read_parquet(file_path)

                for start_idx in range(0, len(df), self.chunk_size):
                    end_idx = min(start_idx + self.chunk_size, len(df))
                    chunk = df.iloc[start_idx:end_idx]

                    for _, row in chunk.iterrows():
                        X = row[self.feature_columns].fillna(0).astype(np.float32).values

                        if self.target_column:
                            y = row[self.target_column]
                            if pd.isna(y):
                                y = 0.0
                            y = np.float32(y)
                            sample = (torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
                        else:
                            sample = torch.tensor(X, dtype=torch.float32)

                        if self.transform:
                            sample = self.transform(sample)

                        yield sample

            except Exception as e:
                logger.warning(f"Failed to read {file_path}: {e}")
                continue


class PyTorchDataLoader:
    """
    Optimized data loader using PyTorch's DataLoader with multi-worker support.

    Features:
    - Multi-worker parallel data loading
    - Pin memory for faster GPU transfer
    - Automatic batching with custom collate functions
    - Support for both in-memory and streaming datasets
    """

    def __init__(
        self,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        prefetch_factor: int = 2
    ):
        """
        Initialize PyTorch data loader.

        Args:
            num_workers: Number of worker processes for data loading
            pin_memory: Pin memory for faster GPU transfer
            persistent_workers: Keep workers alive between epochs
            prefetch_factor: Number of batches to prefetch per worker
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required. Install with: pip install torch")

        self.num_workers = num_workers
        self.pin_memory = pin_memory and torch.cuda.is_available()
        self.persistent_workers = persistent_workers and num_workers > 0
        self.prefetch_factor = prefetch_factor

        logger.info(f"Initialized PyTorchDataLoader: workers={num_workers}, pin_memory={self.pin_memory}")

    def create_dataloader(
        self,
        dataset: Dataset,
        batch_size: int = 32,
        shuffle: bool = True,
        drop_last: bool = False,
        collate_fn: Optional[Callable] = None
    ) -> DataLoader:
        """
        Create optimized DataLoader from dataset.

        Args:
            dataset: PyTorch Dataset
            batch_size: Batch size
            shuffle: Enable shuffling
            drop_last: Drop last incomplete batch
            collate_fn: Custom collate function

        Returns:
            Optimized DataLoader
        """
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            drop_last=drop_last,
            collate_fn=collate_fn
        )

        logger.info(f"Created DataLoader: batch_size={batch_size}, shuffle={shuffle}, workers={self.num_workers}")
        return dataloader

    def load_parquet_files_parallel(
        self,
        file_paths: List[Path],
        max_workers: int = 8
    ) -> pd.DataFrame:
        """
        Load multiple Parquet files in parallel using ThreadPoolExecutor.

        Args:
            file_paths: List of Parquet file paths
            max_workers: Maximum number of parallel workers

        Returns:
            Concatenated DataFrame
        """
        if not file_paths:
            return pd.DataFrame()

        logger.info(f"Loading {len(file_paths)} Parquet files with {max_workers} workers")

        def load_file(file_path):
            try:
                return pd.read_parquet(file_path)
            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {e}")
                return pd.DataFrame()

        dfs = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(load_file, fp): fp for fp in file_paths}

            for future in as_completed(futures):
                df = future.result()
                if not df.empty:
                    dfs.append(df)

        if not dfs:
            return pd.DataFrame()

        result = pd.concat(dfs, ignore_index=True)
        logger.info(f"Loaded {len(result)} rows from {len(dfs)} files")
        return result

    def create_dataset_from_dataframe(
        self,
        data: pd.DataFrame,
        feature_columns: List[str],
        target_column: Optional[str] = None,
        cache_in_memory: bool = True
    ) -> ParquetDataset:
        """
        Create dataset from DataFrame with temporary Parquet file.

        Args:
            data: Source DataFrame
            feature_columns: Feature column names
            target_column: Optional target column name
            cache_in_memory: Cache data in memory

        Returns:
            ParquetDataset
        """
        import tempfile

        temp_file = Path(tempfile.mktemp(suffix='.parquet'))
        data.to_parquet(temp_file, index=False)

        dataset = ParquetDataset(
            file_paths=[temp_file],
            feature_columns=feature_columns,
            target_column=target_column,
            cache_in_memory=cache_in_memory
        )

        return dataset


class PyTorchDataLoaderConfig:
    """Configuration for PyTorch data loader."""

    DEFAULT = {
        'num_workers': 4,
        'pin_memory': True,
        'persistent_workers': True,
        'prefetch_factor': 2
    }

    HIGH_PERFORMANCE = {
        'num_workers': 8,
        'pin_memory': True,
        'persistent_workers': True,
        'prefetch_factor': 4
    }

    LOW_MEMORY = {
        'num_workers': 2,
        'pin_memory': False,
        'persistent_workers': False,
        'prefetch_factor': 2
    }

    SINGLE_WORKER = {
        'num_workers': 0,
        'pin_memory': True,
        'persistent_workers': False,
        'prefetch_factor': None
    }
