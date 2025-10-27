"""
Generate comprehensive reports and visualizations from walk-forward backtest results.

Creates performance summaries, charts, and analysis of lineup optimization
across historical slates.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns


class BacktestReport:
    """
    Generate reports and visualizations from walk-forward backtest results.
    """

    def __init__(self, results_path: str):
        """
        Initialize report generator.

        Parameters
        ----------
        results_path : str
            Path to simulation results JSON file
        """
        self.results_path = Path(results_path)

        # Load results
        with open(self.results_path, 'r') as f:
            self.results = json.load(f)

        # Convert to DataFrames
        self.daily_df = pd.DataFrame(self.results['daily_breakdown'])
        self.lineup_df = pd.DataFrame(self.results['lineup_results'])

    def generate_summary(self) -> str:
        """
        Generate text summary of backtest results.

        Returns
        -------
        str
            Formatted summary report
        """
        metrics = self.results['aggregate_metrics']

        summary = []
        summary.append("="*70)
        summary.append("WALK-FORWARD BACKTEST SUMMARY")
        summary.append("="*70)
        summary.append("")

        # Overview
        summary.append("Overview:")
        summary.append(f"  Slates simulated: {self.results['num_slates']}")
        summary.append(f"  Total lineups: {self.results['total_lineups']}")
        summary.append(f"  Date range: {self.results['date_range']['start']} to {self.results['date_range']['end']}")
        summary.append("")

        # Aggregate metrics
        summary.append("Aggregate Performance:")
        summary.append(f"  Avg actual points: {metrics['avg_actual_points']:.2f}")
        summary.append(f"  Avg projected points: {metrics['avg_projected_points']:.2f}")
        summary.append(f"  Avg error: {metrics['avg_error']:.2f}")
        summary.append(f"  MAE: {metrics['mae']:.2f}")
        summary.append(f"  RMSE: {metrics['rmse']:.2f}")
        summary.append("")

        # Distribution
        summary.append("Score Distribution:")
        summary.append(f"  Min actual points: {metrics['min_actual_points']:.2f}")
        summary.append(f"  Max actual points: {metrics['max_actual_points']:.2f}")
        summary.append(f"  Std actual points: {self.lineup_df['actual_points'].std():.2f}")
        summary.append("")

        # Best/worst days
        summary.append("Best Performing Slates:")
        top_slates = self.daily_df.nlargest(5, 'avg_actual_points')
        for _, row in top_slates.iterrows():
            summary.append(f"  {row['date']}: {row['avg_actual_points']:.2f} pts (projected: {row['avg_projected_points']:.2f})")
        summary.append("")

        summary.append("Worst Performing Slates:")
        bottom_slates = self.daily_df.nsmallest(5, 'avg_actual_points')
        for _, row in bottom_slates.iterrows():
            summary.append(f"  {row['date']}: {row['avg_actual_points']:.2f} pts (projected: {row['avg_projected_points']:.2f})")
        summary.append("")

        # Projection accuracy
        summary.append("Projection Accuracy:")
        over_projections = (self.lineup_df['error'] > 0).sum()
        under_projections = (self.lineup_df['error'] < 0).sum()
        pct_over = (over_projections / len(self.lineup_df)) * 100
        summary.append(f"  Over-projected: {over_projections} lineups ({pct_over:.1f}%)")
        summary.append(f"  Under-projected: {under_projections} lineups ({100-pct_over:.1f}%)")
        summary.append("")

        summary.append("="*70)

        return "\n".join(summary)

    def plot_daily_performance(self, save_path: Optional[str] = None) -> None:
        """
        Plot daily lineup performance over time.

        Parameters
        ----------
        save_path : Optional[str]
            Path to save plot (if None, displays plot)
        """
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        # Convert date strings to datetime
        self.daily_df['date_dt'] = pd.to_datetime(self.daily_df['date'], format='%Y%m%d')

        # Plot 1: Actual vs Projected Points
        axes[0].plot(self.daily_df['date_dt'], self.daily_df['avg_actual_points'],
                    marker='o', label='Actual Points', color='blue')
        axes[0].plot(self.daily_df['date_dt'], self.daily_df['avg_projected_points'],
                    marker='x', label='Projected Points', color='orange', linestyle='--')
        axes[0].set_xlabel('Date')
        axes[0].set_ylabel('Average Points')
        axes[0].set_title('Daily Lineup Performance: Actual vs Projected')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].tick_params(axis='x', rotation=45)

        # Plot 2: Prediction Error
        axes[1].plot(self.daily_df['date_dt'], self.daily_df['avg_error'],
                    marker='o', color='red')
        axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1].set_xlabel('Date')
        axes[1].set_ylabel('Average Error (Actual - Projected)')
        axes[1].set_title('Daily Prediction Error')
        axes[1].grid(True, alpha=0.3)
        axes[1].tick_params(axis='x', rotation=45)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_score_distribution(self, save_path: Optional[str] = None) -> None:
        """
        Plot distribution of lineup scores.

        Parameters
        ----------
        save_path : Optional[str]
            Path to save plot (if None, displays plot)
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: Histogram of actual points
        axes[0].hist(self.lineup_df['actual_points'], bins=30, edgecolor='black', alpha=0.7)
        axes[0].axvline(self.lineup_df['actual_points'].mean(), color='red',
                       linestyle='--', label=f"Mean: {self.lineup_df['actual_points'].mean():.2f}")
        axes[0].set_xlabel('Actual Points')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Distribution of Lineup Scores')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Plot 2: Error distribution
        axes[1].hist(self.lineup_df['error'], bins=30, edgecolor='black', alpha=0.7, color='orange')
        axes[1].axvline(0, color='black', linestyle='--', alpha=0.5)
        axes[1].axvline(self.lineup_df['error'].mean(), color='red',
                       linestyle='--', label=f"Mean: {self.lineup_df['error'].mean():.2f}")
        axes[1].set_xlabel('Error (Actual - Projected)')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Distribution of Prediction Errors')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_error_vs_projected(self, save_path: Optional[str] = None) -> None:
        """
        Plot prediction error vs projected points to identify bias patterns.

        Parameters
        ----------
        save_path : Optional[str]
            Path to save plot (if None, displays plot)
        """
        plt.figure(figsize=(10, 6))

        plt.scatter(self.lineup_df['projected_points'], self.lineup_df['error'],
                   alpha=0.5)
        plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)

        # Add trend line
        z = np.polyfit(self.lineup_df['projected_points'], self.lineup_df['error'], 1)
        p = np.poly1d(z)
        plt.plot(self.lineup_df['projected_points'],
                p(self.lineup_df['projected_points']),
                "r--", alpha=0.8, label=f'Trend: y={z[0]:.2f}x+{z[1]:.2f}')

        plt.xlabel('Projected Points')
        plt.ylabel('Error (Actual - Projected)')
        plt.title('Prediction Error vs Projected Points')
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        else:
            plt.show()

        plt.close()

    def generate_full_report(self, output_dir: str) -> None:
        """
        Generate complete report with all visualizations.

        Parameters
        ----------
        output_dir : str
            Directory to save report files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print("Generating backtest report...")

        # Save text summary
        summary = self.generate_summary()
        summary_path = output_path / 'summary.txt'
        with open(summary_path, 'w') as f:
            f.write(summary)
        print(f"✓ Summary saved to: {summary_path}")

        # Generate plots
        self.plot_daily_performance(str(output_path / 'daily_performance.png'))
        print(f"✓ Daily performance plot saved")

        self.plot_score_distribution(str(output_path / 'score_distribution.png'))
        print(f"✓ Score distribution plot saved")

        self.plot_error_vs_projected(str(output_path / 'error_vs_projected.png'))
        print(f"✓ Error analysis plot saved")

        # Save detailed results CSV
        self.lineup_df.to_csv(output_path / 'lineup_results.csv', index=False)
        print(f"✓ Lineup results CSV saved")

        self.daily_df.to_csv(output_path / 'daily_results.csv', index=False)
        print(f"✓ Daily results CSV saved")

        print(f"\n✓ Full report generated in: {output_path}")


def main():
    """CLI entry point for generating reports from existing results."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate report from walk-forward backtest results"
    )
    parser.add_argument(
        "--results",
        required=True,
        help="Path to simulation results JSON file"
    )
    parser.add_argument(
        "--output-dir",
        default="data/backtest_results/report",
        help="Directory to save report files (default: data/backtest_results/report)"
    )

    args = parser.parse_args()

    # Generate report
    report = BacktestReport(args.results)
    report.generate_full_report(args.output_dir)


if __name__ == '__main__':
    main()
