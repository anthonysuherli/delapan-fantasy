"""
Benchmark script comparing different data loader implementations.

Compares:
- Original HistoricalDataLoader (baseline)
- OptimizedParquetLoader (parallel I/O + caching)
- TensorFlowDataLoader (tf.data pipeline)
- PyTorchDataLoader (multi-worker DataLoader)
- GPUAcceleratedLoader (cuDF/RAPIDS)
"""

import time
import logging
from pathlib import Path
from typing import Dict, List
import pandas as pd
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.data.storage.parquet_storage import ParquetStorage
from src.data.loaders.historical_loader import HistoricalDataLoader
from src.data.loaders.optimized_loader import (
    OptimizedParquetLoader,
    TensorFlowDataLoader,
    PyTorchDataLoader
)
from src.data.loaders.gpu_loader import GPUAcceleratedLoader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LoaderBenchmark:
    """Benchmark different data loader implementations."""

    def __init__(
        self,
        data_dir: str = 'data/inputs',
        start_date: str = '20241201',
        end_date: str = '20241231',
        num_runs: int = 3
    ):
        """
        Initialize benchmark.

        Args:
            data_dir: Base data directory
            start_date: Start date for data loading
            end_date: End date for data loading
            num_runs: Number of runs per loader
        """
        self.data_dir = data_dir
        self.start_date = start_date
        self.end_date = end_date
        self.num_runs = num_runs
        self.results = {}

    def benchmark_original_loader(self) -> Dict:
        """Benchmark original HistoricalDataLoader."""
        logger.info("="*80)
        logger.info("Benchmarking OriginalLoader (baseline)")
        logger.info("="*80)

        storage = ParquetStorage(self.data_dir)
        loader = HistoricalDataLoader(storage)

        times = []
        for run in range(self.num_runs):
            logger.info(f"Run {run + 1}/{self.num_runs}")

            start_time = time.perf_counter()

            data = loader.load_historical_player_logs(
                start_date=self.start_date,
                end_date=self.end_date
            )

            elapsed = time.perf_counter() - start_time
            times.append(elapsed)

            logger.info(f"  Loaded {len(data)} rows in {elapsed:.2f}s")

        avg_time = sum(times) / len(times)
        logger.info(f"Average time: {avg_time:.2f}s")

        return {
            'loader': 'OriginalLoader',
            'times': times,
            'avg_time': avg_time,
            'rows_loaded': len(data)
        }

    def benchmark_optimized_loader(self) -> Dict:
        """Benchmark OptimizedParquetLoader."""
        logger.info("="*80)
        logger.info("Benchmarking OptimizedParquetLoader (parallel + caching)")
        logger.info("="*80)

        loader = OptimizedParquetLoader(
            base_dir=self.data_dir,
            max_workers=4,
            cache_size=100
        )

        times = []
        for run in range(self.num_runs):
            logger.info(f"Run {run + 1}/{self.num_runs}")

            if run == 0:
                loader.clear_cache()

            start_time = time.perf_counter()

            data = loader.load_parallel(
                'box_scores',
                start_date=self.start_date,
                end_date=self.end_date,
                use_cache=True
            )

            elapsed = time.perf_counter() - start_time
            times.append(elapsed)

            cache_stats = loader.get_cache_stats()
            logger.info(f"  Loaded {len(data)} rows in {elapsed:.2f}s (cache: {cache_stats['entries']} entries)")

        avg_time = sum(times) / len(times)
        logger.info(f"Average time: {avg_time:.2f}s")

        return {
            'loader': 'OptimizedParquetLoader',
            'times': times,
            'avg_time': avg_time,
            'rows_loaded': len(data),
            'cache_stats': loader.get_cache_stats()
        }

    def benchmark_tensorflow_loader(self) -> Dict:
        """Benchmark TensorFlowDataLoader."""
        logger.info("="*80)
        logger.info("Benchmarking TensorFlowDataLoader (tf.data pipeline)")
        logger.info("="*80)

        try:
            loader = TensorFlowDataLoader(
                base_dir=self.data_dir,
                prefetch_buffer=2,
                num_parallel_reads=4
            )

            if not loader.available:
                logger.warning("TensorFlow not available, skipping")
                return {
                    'loader': 'TensorFlowDataLoader',
                    'error': 'TensorFlow not available'
                }

            times = []
            for run in range(self.num_runs):
                logger.info(f"Run {run + 1}/{self.num_runs}")

                start_time = time.perf_counter()

                dataset = loader.create_dataset(
                    'box_scores',
                    start_date=self.start_date,
                    end_date=self.end_date,
                    batch_size=256
                )

                total_batches = 0
                for batch in dataset:
                    total_batches += 1

                elapsed = time.perf_counter() - start_time
                times.append(elapsed)

                logger.info(f"  Processed {total_batches} batches in {elapsed:.2f}s")

            avg_time = sum(times) / len(times)
            logger.info(f"Average time: {avg_time:.2f}s")

            return {
                'loader': 'TensorFlowDataLoader',
                'times': times,
                'avg_time': avg_time,
                'batches_processed': total_batches
            }

        except Exception as e:
            logger.error(f"TensorFlow benchmark failed: {e}")
            return {
                'loader': 'TensorFlowDataLoader',
                'error': str(e)
            }

    def benchmark_pytorch_loader(self) -> Dict:
        """Benchmark PyTorchDataLoader."""
        logger.info("="*80)
        logger.info("Benchmarking PyTorchDataLoader (multi-worker)")
        logger.info("="*80)

        try:
            loader = PyTorchDataLoader(
                base_dir=self.data_dir,
                num_workers=4,
                pin_memory=True
            )

            if not loader.available:
                logger.warning("PyTorch not available, skipping")
                return {
                    'loader': 'PyTorchDataLoader',
                    'error': 'PyTorch not available'
                }

            times = []
            for run in range(self.num_runs):
                logger.info(f"Run {run + 1}/{self.num_runs}")

                start_time = time.perf_counter()

                dataset = loader.create_dataset(
                    'box_scores',
                    start_date=self.start_date,
                    end_date=self.end_date
                )

                dataloader = loader.create_dataloader(
                    dataset,
                    batch_size=256,
                    shuffle=False
                )

                total_batches = 0
                for batch in dataloader:
                    total_batches += 1

                elapsed = time.perf_counter() - start_time
                times.append(elapsed)

                logger.info(f"  Processed {total_batches} batches in {elapsed:.2f}s")

            avg_time = sum(times) / len(times)
            logger.info(f"Average time: {avg_time:.2f}s")

            return {
                'loader': 'PyTorchDataLoader',
                'times': times,
                'avg_time': avg_time,
                'batches_processed': total_batches
            }

        except Exception as e:
            logger.error(f"PyTorch benchmark failed: {e}")
            return {
                'loader': 'PyTorchDataLoader',
                'error': str(e)
            }

    def benchmark_gpu_loader(self) -> Dict:
        """Benchmark GPUAcceleratedLoader."""
        logger.info("="*80)
        logger.info("Benchmarking GPUAcceleratedLoader (cuDF/RAPIDS)")
        logger.info("="*80)

        loader = GPUAcceleratedLoader(
            base_dir=self.data_dir,
            use_gpu=True
        )

        times = []
        for run in range(self.num_runs):
            logger.info(f"Run {run + 1}/{self.num_runs}")

            if run > 0:
                loader.clear_gpu_memory()

            start_time = time.perf_counter()

            data = loader.load_to_gpu(
                'box_scores',
                start_date=self.start_date,
                end_date=self.end_date
            )

            elapsed = time.perf_counter() - start_time
            times.append(elapsed)

            mem_stats = loader.get_memory_usage()
            logger.info(f"  Loaded {len(data)} rows in {elapsed:.2f}s")
            if mem_stats['gpu_available']:
                logger.info(f"  GPU memory: {mem_stats['used_mb']:.2f} MB")

        avg_time = sum(times) / len(times)
        logger.info(f"Average time: {avg_time:.2f}s")

        return {
            'loader': 'GPUAcceleratedLoader',
            'times': times,
            'avg_time': avg_time,
            'rows_loaded': len(data),
            'gpu_available': loader.gpu_available,
            'memory_stats': loader.get_memory_usage()
        }

    def run_all_benchmarks(self) -> pd.DataFrame:
        """
        Run all benchmarks and return results.

        Returns:
            DataFrame with benchmark results
        """
        logger.info("="*80)
        logger.info("STARTING LOADER BENCHMARKS")
        logger.info("="*80)
        logger.info(f"Data directory: {self.data_dir}")
        logger.info(f"Date range: {self.start_date} to {self.end_date}")
        logger.info(f"Number of runs: {self.num_runs}")
        logger.info("")

        benchmarks = [
            ('original', self.benchmark_original_loader),
            ('optimized', self.benchmark_optimized_loader),
            ('tensorflow', self.benchmark_tensorflow_loader),
            ('pytorch', self.benchmark_pytorch_loader),
            ('gpu', self.benchmark_gpu_loader)
        ]

        results = []
        for name, benchmark_func in benchmarks:
            try:
                result = benchmark_func()
                self.results[name] = result
                results.append(result)
                logger.info("")
            except Exception as e:
                logger.error(f"Benchmark {name} failed: {e}")
                results.append({
                    'loader': name,
                    'error': str(e)
                })
                logger.info("")

        self._print_summary(results)

        return pd.DataFrame(results)

    def _print_summary(self, results: List[Dict]):
        """Print benchmark summary."""
        logger.info("="*80)
        logger.info("BENCHMARK SUMMARY")
        logger.info("="*80)

        baseline_time = None
        for result in results:
            if 'error' in result:
                logger.info(f"{result['loader']:30} ERROR: {result['error']}")
                continue

            avg_time = result.get('avg_time', 0)
            loader_name = result['loader']

            if loader_name == 'OriginalLoader':
                baseline_time = avg_time
                logger.info(f"{loader_name:30} {avg_time:8.2f}s (baseline)")
            elif baseline_time:
                speedup = baseline_time / avg_time if avg_time > 0 else 0
                pct_faster = ((baseline_time - avg_time) / baseline_time * 100) if baseline_time > 0 else 0
                logger.info(f"{loader_name:30} {avg_time:8.2f}s ({speedup:.2f}x speedup, {pct_faster:+.1f}%)")
            else:
                logger.info(f"{loader_name:30} {avg_time:8.2f}s")

        logger.info("="*80)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Benchmark data loaders')
    parser.add_argument('--data-dir', default='data/inputs', help='Data directory')
    parser.add_argument('--start-date', default='20241201', help='Start date (YYYYMMDD)')
    parser.add_argument('--end-date', default='20241231', help='End date (YYYYMMDD)')
    parser.add_argument('--runs', type=int, default=3, help='Number of runs per loader')
    parser.add_argument('--output', default='data/outputs/loader_benchmark.csv', help='Output CSV path')

    args = parser.parse_args()

    benchmark = LoaderBenchmark(
        data_dir=args.data_dir,
        start_date=args.start_date,
        end_date=args.end_date,
        num_runs=args.runs
    )

    results_df = benchmark.run_all_benchmarks()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)

    logger.info("")
    logger.info(f"Results saved to: {output_path}")


if __name__ == '__main__':
    main()
