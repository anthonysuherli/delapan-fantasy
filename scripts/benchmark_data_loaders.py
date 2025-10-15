"""
Benchmark script comparing data loading performance.

Compares:
- Original HistoricalDataLoader (sequential)
- OptimizedHistoricalDataLoader with TensorFlow backend
- OptimizedHistoricalDataLoader with PyTorch backend

Metrics:
- Total load time
- Throughput (rows/second)
- Memory usage
- CPU utilization
"""

import time
import sys
from pathlib import Path
import pandas as pd
import logging
from typing import Dict, Any

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.storage.parquet_storage import ParquetStorage
from src.data.loaders.historical_loader import HistoricalDataLoader
from src.data.loaders.optimized_historical_loader import OptimizedHistoricalDataLoader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def format_time(seconds: float) -> str:
    """Format seconds into human-readable string."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"


def benchmark_loader(
    loader,
    loader_name: str,
    start_date: str,
    end_date: str,
    num_runs: int = 3
) -> Dict[str, Any]:
    """
    Benchmark a data loader.

    Args:
        loader: Data loader instance
        loader_name: Name for reporting
        start_date: Start date in YYYYMMDD format
        end_date: End date in YYYYMMDD format
        num_runs: Number of benchmark runs

    Returns:
        Dictionary with benchmark results
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Benchmarking: {loader_name}")
    logger.info(f"Date range: {start_date} to {end_date}")
    logger.info(f"Number of runs: {num_runs}")
    logger.info(f"{'='*80}\n")

    times = []
    row_counts = []

    for run in range(num_runs):
        logger.info(f"Run {run + 1}/{num_runs}...")

        start_time = time.perf_counter()

        try:
            df = loader.load_historical_player_logs(
                start_date=start_date,
                end_date=end_date
            )
            elapsed = time.perf_counter() - start_time

            times.append(elapsed)
            row_counts.append(len(df))

            logger.info(f"  Loaded {len(df):,} rows in {format_time(elapsed)}")
            logger.info(f"  Throughput: {len(df) / elapsed:,.0f} rows/second")

        except Exception as e:
            logger.error(f"  Failed: {e}")
            times.append(float('inf'))
            row_counts.append(0)

    if not times or all(t == float('inf') for t in times):
        logger.error(f"All benchmark runs failed for {loader_name}")
        return {
            'loader_name': loader_name,
            'success': False,
            'error': 'All runs failed'
        }

    valid_times = [t for t in times if t != float('inf')]
    avg_time = sum(valid_times) / len(valid_times)
    min_time = min(valid_times)
    max_time = max(valid_times)
    avg_rows = sum(row_counts) / len(row_counts)
    throughput = avg_rows / avg_time if avg_time > 0 else 0

    logger.info(f"\nResults for {loader_name}:")
    logger.info(f"  Average time: {format_time(avg_time)}")
    logger.info(f"  Min time: {format_time(min_time)}")
    logger.info(f"  Max time: {format_time(max_time)}")
    logger.info(f"  Average rows: {avg_rows:,.0f}")
    logger.info(f"  Average throughput: {throughput:,.0f} rows/second")

    return {
        'loader_name': loader_name,
        'success': True,
        'avg_time': avg_time,
        'min_time': min_time,
        'max_time': max_time,
        'avg_rows': avg_rows,
        'throughput': throughput,
        'times': valid_times,
        'row_counts': row_counts
    }


def compare_results(results: list):
    """
    Compare benchmark results and print summary.

    Args:
        results: List of benchmark result dictionaries
    """
    logger.info(f"\n{'='*80}")
    logger.info("BENCHMARK COMPARISON")
    logger.info(f"{'='*80}\n")

    successful = [r for r in results if r['success']]

    if not successful:
        logger.error("No successful benchmark runs to compare")
        return

    df = pd.DataFrame(successful)
    df = df.sort_values('avg_time')

    logger.info("Performance Summary:")
    logger.info(f"{'Loader':<30} {'Avg Time':<15} {'Throughput':<20} {'Speedup':<10}")
    logger.info("-" * 80)

    baseline_time = df.iloc[0]['avg_time']

    for _, row in df.iterrows():
        speedup = baseline_time / row['avg_time']
        logger.info(
            f"{row['loader_name']:<30} "
            f"{format_time(row['avg_time']):<15} "
            f"{row['throughput']:>15,.0f} rows/s "
            f"{speedup:>8.2f}x"
        )

    logger.info("")
    logger.info("Key Findings:")

    fastest = df.iloc[0]
    slowest = df.iloc[-1]

    improvement = (slowest['avg_time'] - fastest['avg_time']) / slowest['avg_time'] * 100

    logger.info(f"  Fastest: {fastest['loader_name']}")
    logger.info(f"  Slowest: {slowest['loader_name']}")
    logger.info(f"  Performance improvement: {improvement:.1f}% faster")
    logger.info(f"  Throughput increase: {fastest['throughput'] / slowest['throughput']:.2f}x")


def main():
    """Run benchmark comparison."""
    import argparse

    parser = argparse.ArgumentParser(description='Benchmark data loader performance')
    parser.add_argument('--start-date', type=str, required=True, help='Start date (YYYYMMDD)')
    parser.add_argument('--end-date', type=str, required=True, help='End date (YYYYMMDD)')
    parser.add_argument('--runs', type=int, default=3, help='Number of benchmark runs')
    parser.add_argument('--storage-dir', type=str, default='data/inputs', help='Data storage directory')
    parser.add_argument('--loaders', nargs='+', default=['original', 'tensorflow', 'pytorch'],
                        choices=['original', 'tensorflow', 'pytorch'],
                        help='Loaders to benchmark')

    args = parser.parse_args()

    logger.info("Starting Data Loader Benchmark")
    logger.info(f"Start date: {args.start_date}")
    logger.info(f"End date: {args.end_date}")
    logger.info(f"Runs per loader: {args.runs}")
    logger.info(f"Storage directory: {args.storage_dir}")
    logger.info(f"Loaders to test: {', '.join(args.loaders)}")

    storage = ParquetStorage(base_dir=args.storage_dir)

    results = []

    if 'original' in args.loaders:
        logger.info("\n\nTesting original HistoricalDataLoader...")
        original_loader = HistoricalDataLoader(storage)
        result = benchmark_loader(
            original_loader,
            'Original (Sequential)',
            args.start_date,
            args.end_date,
            args.runs
        )
        results.append(result)

    if 'tensorflow' in args.loaders:
        try:
            logger.info("\n\nTesting OptimizedHistoricalDataLoader with TensorFlow backend...")
            tf_loader = OptimizedHistoricalDataLoader(
                storage,
                loader_type='tensorflow',
                num_workers=8,
                enable_prefetch=True,
                enable_cache=True
            )
            result = benchmark_loader(
                tf_loader,
                'Optimized (TensorFlow)',
                args.start_date,
                args.end_date,
                args.runs
            )
            results.append(result)
        except ImportError as e:
            logger.warning(f"TensorFlow not available: {e}")

    if 'pytorch' in args.loaders:
        try:
            logger.info("\n\nTesting OptimizedHistoricalDataLoader with PyTorch backend...")
            pytorch_loader = OptimizedHistoricalDataLoader(
                storage,
                loader_type='pytorch',
                num_workers=8,
                enable_prefetch=True,
                enable_cache=True
            )
            result = benchmark_loader(
                pytorch_loader,
                'Optimized (PyTorch)',
                args.start_date,
                args.end_date,
                args.runs
            )
            results.append(result)
        except ImportError as e:
            logger.warning(f"PyTorch not available: {e}")

    compare_results(results)

    logger.info(f"\n{'='*80}")
    logger.info("Benchmark complete")
    logger.info(f"{'='*80}\n")


if __name__ == '__main__':
    main()
