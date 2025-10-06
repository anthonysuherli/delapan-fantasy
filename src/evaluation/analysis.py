import pandas as pd
import numpy as np
from typing import Dict, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def analyze_backtest_results(results: Dict[str, Any]) -> None:
    if 'error' in results:
        print("\n" + "="*80)
        print("ERROR")
        print("="*80)
        print(f"\n{results['error']}")
        print("\n" + "="*80)
        return

    print("\n" + "="*80)
    print("BACKTEST RESULTS SUMMARY")
    print("="*80)

    print(f"\nPeriod: {results['date_range']}")
    print(f"Total Slates Processed: {results['num_slates']}")
    print(f"Total Players Evaluated: {results['total_players_evaluated']}")
    print(f"Average Players per Slate: {results['avg_players_per_slate']:.1f}")

    print("\n" + "-"*80)
    print("PROJECTION ACCURACY METRICS")
    print("-"*80)

    mean_mape = results['mean_mape']
    print(f"\nMean Absolute Percentage Error (MAPE):")
    print(f"  Mean:   {mean_mape:.2f}%")
    print(f"  Median: {results['median_mape']:.2f}%")
    print(f"  Std:    {results['std_mape']:.2f}%")

    if mean_mape < 30:
        print(f"  Status: EXCELLENT (target achieved)")
    elif mean_mape < 35:
        print(f"  Status: GOOD (close to target)")
    elif mean_mape < 40:
        print(f"  Status: FAIR (needs improvement)")
    else:
        print(f"  Status: POOR (significant improvement needed)")

    print(f"\nRoot Mean Squared Error (RMSE):")
    print(f"  Mean: {results['mean_rmse']:.2f} points")
    print(f"  Std:  {results['std_rmse']:.2f} points")

    print(f"\nPrediction Correlation:")
    print(f"  Mean: {results['mean_correlation']:.3f}")
    print(f"  Std:  {results['std_correlation']:.3f}")

    print("\n" + "="*80)


def plot_backtest_results(daily_df: pd.DataFrame, output_path: str) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        from datetime import datetime

        daily_df = daily_df.copy()
        daily_df['date_dt'] = pd.to_datetime(daily_df['date'], format='%Y%m%d')
        daily_df = daily_df.sort_values('date_dt')

        fig, axes = plt.subplots(3, 1, figsize=(14, 10))

        axes[0].plot(daily_df['date_dt'], daily_df['mape'], 'o-', linewidth=2, markersize=4, color='#2E86AB')
        axes[0].axhline(y=30, color='#A23B72', linestyle='--', linewidth=2, label='Target (30%)')
        axes[0].fill_between(daily_df['date_dt'], 0, 30, alpha=0.2, color='#06A77D')
        axes[0].set_ylabel('MAPE (%)', fontsize=12, fontweight='bold')
        axes[0].set_title('Projection Accuracy Over Time', fontsize=14, fontweight='bold')
        axes[0].legend(loc='upper right')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim(bottom=0)

        axes[1].plot(daily_df['date_dt'], daily_df['rmse'], 's-', linewidth=2, markersize=4, color='#F18F01')
        axes[1].set_ylabel('RMSE (points)', fontsize=12, fontweight='bold')
        axes[1].set_title('Root Mean Squared Error Over Time', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim(bottom=0)

        axes[2].plot(daily_df['date_dt'], daily_df['correlation'], '^-', linewidth=2, markersize=4, color='#C73E1D')
        axes[2].axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        axes[2].set_ylabel('Correlation', fontsize=12, fontweight='bold')
        axes[2].set_xlabel('Date', fontsize=12, fontweight='bold')
        axes[2].set_title('Prediction Correlation Over Time', fontsize=14, fontweight='bold')
        axes[2].grid(True, alpha=0.3)
        axes[2].set_ylim(-0.1, 1.0)

        for ax in axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Plots saved to {output_path}")
        print(f"\nPlots saved to {output_path}")

    except ImportError:
        logger.warning("matplotlib not available, skipping plots")
        print("\nWarning: matplotlib not installed, skipping visualization")
    except Exception as e:
        logger.error(f"Failed to create plots: {str(e)}")
        print(f"\nError creating plots: {str(e)}")


def analyze_errors(daily_df: pd.DataFrame) -> None:
    print("\n" + "="*80)
    print("ERROR ANALYSIS")
    print("="*80)

    worst_days = daily_df.nlargest(5, 'mape')[['date', 'mape', 'rmse', 'num_players']]
    print("\nWorst Prediction Days (Highest MAPE):")
    print(worst_days.to_string(index=False))

    best_days = daily_df.nsmallest(5, 'mape')[['date', 'mape', 'rmse', 'num_players']]
    print("\nBest Prediction Days (Lowest MAPE):")
    print(best_days.to_string(index=False))

    if 'correlation' in daily_df.columns:
        corr = daily_df['mape'].corr(daily_df['num_players'])
        print(f"\nCorrelation between MAPE and Number of Players: {corr:.3f}")

    print("\n" + "="*80)


def generate_backtest_report(
    results: Dict[str, Any],
    output_dir: str
) -> None:
    logger.info(f"Generating backtest report in {output_dir}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    analyze_backtest_results(results)

    if 'daily_results' in results and not results['daily_results'].empty:
        daily_df = results['daily_results']

        analyze_errors(daily_df)

        plot_path = output_path / 'backtest_plots.png'
        plot_backtest_results(daily_df, str(plot_path))

        csv_path = output_path / 'daily_results.csv'
        daily_df.to_csv(csv_path, index=False)
        print(f"\nDaily results saved to {csv_path}")

        summary_path = output_path / 'summary.txt'
        with open(summary_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("BACKTEST SUMMARY\n")
            f.write("="*80 + "\n\n")
            f.write(f"Period: {results['date_range']}\n")
            f.write(f"Total Slates: {results['num_slates']}\n")
            f.write(f"Total Players: {results['total_players_evaluated']}\n\n")
            f.write("-"*80 + "\n")
            f.write("METRICS\n")
            f.write("-"*80 + "\n")
            f.write(f"Mean MAPE: {results['mean_mape']:.2f}%\n")
            f.write(f"Median MAPE: {results['median_mape']:.2f}%\n")
            f.write(f"Mean RMSE: {results['mean_rmse']:.2f}\n")
            f.write(f"Mean Correlation: {results['mean_correlation']:.3f}\n")

        print(f"Summary saved to {summary_path}")

    logger.info("Backtest report generation complete")