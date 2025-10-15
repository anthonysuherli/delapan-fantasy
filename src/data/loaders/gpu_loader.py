import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class GPUAcceleratedLoader:
    """
    GPU-accelerated data loader using cuDF for NVIDIA GPUs.

    This loader leverages RAPIDS cuDF for GPU-accelerated data processing.
    Falls back to pandas if cuDF is not available.
    """

    def __init__(self, base_dir: str = 'data/inputs', use_gpu: bool = True):
        """
        Initialize GPU-accelerated loader.

        Args:
            base_dir: Base directory for data files
            use_gpu: Whether to attempt GPU acceleration
        """
        self.base_dir = Path(base_dir)
        self.use_gpu = use_gpu
        self.gpu_available = False

        if use_gpu:
            try:
                import cudf
                import cupy as cp
                self.cudf = cudf
                self.cp = cp
                self.gpu_available = True
                logger.info("GPU acceleration enabled with cuDF")
            except ImportError:
                logger.warning("cuDF not available, falling back to CPU processing")
                self.cudf = None
                self.cp = None

    def load_to_gpu(
        self,
        data_type: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ):
        """
        Load data directly to GPU memory.

        Args:
            data_type: Type of data
            start_date: Start date in YYYYMMDD format
            end_date: End date in YYYYMMDD format

        Returns:
            cuDF DataFrame if GPU available, else pandas DataFrame
        """
        data_dir = self.base_dir / data_type
        if not data_dir.exists():
            logger.warning(f"Directory not found: {data_dir}")
            return self._empty_frame()

        files = list(data_dir.glob("*.parquet"))

        if start_date or end_date:
            files = self._filter_files_by_date(files, start_date, end_date)

        if not files:
            return self._empty_frame()

        logger.debug(f"Loading {len(files)} files to GPU")

        if self.gpu_available:
            dfs = [self.cudf.read_parquet(f) for f in files]
            result = self.cudf.concat(dfs, ignore_index=True)
            logger.info(f"Loaded {len(result)} rows to GPU memory")
        else:
            dfs = [pd.read_parquet(f) for f in files]
            result = pd.concat(dfs, ignore_index=True)
            logger.info(f"Loaded {len(result)} rows to CPU memory")

        return result

    def _empty_frame(self):
        """Return empty frame appropriate for GPU/CPU mode."""
        if self.gpu_available:
            return self.cudf.DataFrame()
        else:
            return pd.DataFrame()

    def _filter_files_by_date(
        self,
        files: List[Path],
        start_date: Optional[str],
        end_date: Optional[str]
    ) -> List[Path]:
        """Filter files by date range."""
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

    def to_pandas(self, df) -> pd.DataFrame:
        """
        Convert GPU DataFrame to pandas.

        Args:
            df: cuDF or pandas DataFrame

        Returns:
            pandas DataFrame
        """
        if self.gpu_available and hasattr(df, 'to_pandas'):
            return df.to_pandas()
        return df

    def to_gpu(self, df: pd.DataFrame):
        """
        Convert pandas DataFrame to GPU.

        Args:
            df: pandas DataFrame

        Returns:
            cuDF DataFrame if GPU available, else original DataFrame
        """
        if self.gpu_available:
            return self.cudf.from_pandas(df)
        return df

    def compute_features_gpu(
        self,
        df,
        feature_cols: List[str]
    ):
        """
        Compute features on GPU.

        Args:
            df: cuDF or pandas DataFrame
            feature_cols: List of feature column names

        Returns:
            DataFrame with computed features
        """
        if not self.gpu_available:
            return df[feature_cols]

        result = df[feature_cols].copy()

        for col in feature_cols:
            if result[col].dtype == object:
                result[col] = result[col].astype('float32')

        result = result.fillna(0)

        return result

    def get_memory_usage(self) -> Dict:
        """
        Get GPU memory usage statistics.

        Returns:
            Dictionary with memory stats
        """
        if not self.gpu_available:
            return {
                'gpu_available': False,
                'total_mb': 0,
                'used_mb': 0,
                'free_mb': 0
            }

        mempool = self.cp.get_default_memory_pool()

        return {
            'gpu_available': True,
            'used_bytes': mempool.used_bytes(),
            'used_mb': mempool.used_bytes() / (1024 * 1024),
            'total_bytes': mempool.total_bytes(),
            'total_mb': mempool.total_bytes() / (1024 * 1024)
        }

    def clear_gpu_memory(self):
        """Clear GPU memory pool."""
        if self.gpu_available:
            mempool = self.cp.get_default_memory_pool()
            mempool.free_all_blocks()
            logger.info("GPU memory cleared")
