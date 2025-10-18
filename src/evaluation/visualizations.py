import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class BacktestVisualizer:
    """
    Generate visualizations for backtest results.
    """

    def __init__(self, output_dir: Path):
        """
        Initialize visualizer.

        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = Path(output_dir)
        self.charts_dir = self.output_dir / 'charts'
        self.charts_dir.mkdir(parents=True, exist_ok=True)

    def generate_all_charts(self, results: Dict[str, Any]) -> Dict[str, Path]:
        """
        Generate all visualization charts.

        Args:
            results: Backtest results dictionary

        Returns:
            Dictionary mapping chart names to file paths
        """
        chart_paths = {}

        try:
            chart_paths['daily_mape'] = self._plot_daily_mape(results)
            chart_paths['model_vs_benchmark'] = self._plot_model_vs_benchmark(results)
            chart_paths['error_distribution'] = self._plot_error_distribution(results)
            chart_paths['salary_tier'] = self._plot_salary_tier_performance(results)
            chart_paths['correlation_scatter'] = self._plot_correlation_scatter(results)
            chart_paths['metrics_comparison'] = self._plot_metrics_comparison(results)

            logger.info(f"Generated {len(chart_paths)} charts in {self.charts_dir}")

        except Exception as e:
            logger.error(f"Error generating charts: {str(e)}")

        return chart_paths

    def _plot_daily_mape(self, results: Dict[str, Any]) -> Optional[Path]:
        """Plot daily MAPE over time."""
        if 'daily_results' not in results:
            return None

        daily_df = results['daily_results']

        fig, ax = plt.subplots(figsize=(14, 6))

        ax.plot(range(len(daily_df)), daily_df['model_mape'],
                marker='o', linewidth=2, markersize=6,
                label='Model MAPE', color='#2E86AB')

        if 'benchmark_mape' in daily_df.columns:
            ax.plot(range(len(daily_df)), daily_df['benchmark_mape'],
                    marker='s', linewidth=2, markersize=6,
                    label='Benchmark MAPE', color='#A23B72', linestyle='--')

        ax.axhline(daily_df['model_mape'].mean(), color='#2E86AB',
                   linestyle=':', alpha=0.7, label=f"Model Mean: {daily_df['model_mape'].mean():.2f}%")

        ax.set_xlabel('Slate Number', fontsize=12, fontweight='bold')
        ax.set_ylabel('MAPE (%)', fontsize=12, fontweight='bold')
        ax.set_title('Daily MAPE Performance Over Time', fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        output_path = self.charts_dir / 'daily_mape.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        return output_path

    def _plot_model_vs_benchmark(self, results: Dict[str, Any]) -> Optional[Path]:
        """Plot model vs benchmark comparison."""
        if 'daily_results' not in results:
            return None

        daily_df = results['daily_results']

        if 'benchmark_mape' not in daily_df.columns:
            return None

        valid_data = daily_df[daily_df['benchmark_mape'].notna()].copy()

        if valid_data.empty:
            return None

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        axes[0, 0].scatter(valid_data['benchmark_mape'], valid_data['model_mape'],
                           alpha=0.6, s=100, color='#2E86AB')
        max_val = max(valid_data['benchmark_mape'].max(), valid_data['model_mape'].max())
        axes[0, 0].plot([0, max_val], [0, max_val], 'r--', alpha=0.5, linewidth=2)
        axes[0, 0].set_xlabel('Benchmark MAPE (%)', fontsize=11, fontweight='bold')
        axes[0, 0].set_ylabel('Model MAPE (%)', fontsize=11, fontweight='bold')
        axes[0, 0].set_title('MAPE: Model vs Benchmark', fontsize=12, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)

        improvement = valid_data['benchmark_mape'] - valid_data['model_mape']
        colors = ['#06A77D' if x > 0 else '#D62828' for x in improvement]
        axes[0, 1].bar(range(len(improvement)), improvement, color=colors, alpha=0.7)
        axes[0, 1].axhline(0, color='black', linewidth=1)
        axes[0, 1].set_xlabel('Slate Number', fontsize=11, fontweight='bold')
        axes[0, 1].set_ylabel('MAPE Improvement (%)', fontsize=11, fontweight='bold')
        axes[0, 1].set_title('Daily MAPE Improvement (Positive = Model Better)', fontsize=12, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3, axis='y')

        axes[1, 0].hist(valid_data['model_mape'], bins=20, alpha=0.6,
                        label='Model', color='#2E86AB', edgecolor='black')
        axes[1, 0].hist(valid_data['benchmark_mape'], bins=20, alpha=0.6,
                        label='Benchmark', color='#A23B72', edgecolor='black')
        axes[1, 0].axvline(valid_data['model_mape'].mean(), color='#2E86AB',
                           linestyle='--', linewidth=2, label=f"Model Mean: {valid_data['model_mape'].mean():.2f}%")
        axes[1, 0].axvline(valid_data['benchmark_mape'].mean(), color='#A23B72',
                           linestyle='--', linewidth=2, label=f"Benchmark Mean: {valid_data['benchmark_mape'].mean():.2f}%")
        axes[1, 0].set_xlabel('MAPE (%)', fontsize=11, fontweight='bold')
        axes[1, 0].set_ylabel('Frequency', fontsize=11, fontweight='bold')
        axes[1, 0].set_title('MAPE Distribution', fontsize=12, fontweight='bold')
        axes[1, 0].legend(fontsize=9)
        axes[1, 0].grid(True, alpha=0.3, axis='y')

        improvement_data = pd.DataFrame({
            'Improvement (%)': improvement,
            'Slate': range(len(improvement))
        })
        axes[1, 1].boxplot([valid_data['model_mape'], valid_data['benchmark_mape']],
                           labels=['Model', 'Benchmark'],
                           patch_artist=True,
                           boxprops=dict(facecolor='#2E86AB', alpha=0.6),
                           medianprops=dict(color='red', linewidth=2))
        axes[1, 1].set_ylabel('MAPE (%)', fontsize=11, fontweight='bold')
        axes[1, 1].set_title('MAPE Distribution Comparison', fontsize=12, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        output_path = self.charts_dir / 'model_vs_benchmark.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        return output_path

    def _plot_error_distribution(self, results: Dict[str, Any]) -> Optional[Path]:
        """Plot error distribution analysis."""
        if 'all_predictions' not in results or results['all_predictions'].empty:
            return None

        all_preds = results['all_predictions']

        if 'actual_fpts' not in all_preds.columns or 'projected_fpts' not in all_preds.columns:
            return None

        all_preds = all_preds.copy()
        all_preds['error'] = all_preds['projected_fpts'] - all_preds['actual_fpts']
        all_preds['abs_error'] = np.abs(all_preds['error'])

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        axes[0, 0].hist(all_preds['error'], bins=50, alpha=0.7,
                        color='#2E86AB', edgecolor='black')
        axes[0, 0].axvline(0, color='red', linestyle='--', linewidth=2)
        axes[0, 0].axvline(all_preds['error'].mean(), color='orange',
                           linestyle='--', linewidth=2, label=f"Mean: {all_preds['error'].mean():.2f}")
        axes[0, 0].set_xlabel('Prediction Error (fpts)', fontsize=11, fontweight='bold')
        axes[0, 0].set_ylabel('Frequency', fontsize=11, fontweight='bold')
        axes[0, 0].set_title('Error Distribution', fontsize=12, fontweight='bold')
        axes[0, 0].legend(fontsize=9)
        axes[0, 0].grid(True, alpha=0.3, axis='y')

        axes[0, 1].hist(all_preds['abs_error'], bins=50, alpha=0.7,
                        color='#A23B72', edgecolor='black')
        axes[0, 1].axvline(all_preds['abs_error'].mean(), color='orange',
                           linestyle='--', linewidth=2,
                           label=f"MAE: {all_preds['abs_error'].mean():.2f}")
        axes[0, 1].axvline(all_preds['abs_error'].median(), color='green',
                           linestyle='--', linewidth=2,
                           label=f"Median: {all_preds['abs_error'].median():.2f}")
        axes[0, 1].set_xlabel('Absolute Error (fpts)', fontsize=11, fontweight='bold')
        axes[0, 1].set_ylabel('Frequency', fontsize=11, fontweight='bold')
        axes[0, 1].set_title('Absolute Error Distribution', fontsize=12, fontweight='bold')
        axes[0, 1].legend(fontsize=9)
        axes[0, 1].grid(True, alpha=0.3, axis='y')

        percentiles = [10, 25, 50, 75, 90]
        percentile_values = [np.percentile(all_preds['abs_error'], p) for p in percentiles]
        axes[1, 0].bar([str(p) for p in percentiles], percentile_values,
                       color='#06A77D', alpha=0.7, edgecolor='black')
        axes[1, 0].set_xlabel('Percentile', fontsize=11, fontweight='bold')
        axes[1, 0].set_ylabel('Absolute Error (fpts)', fontsize=11, fontweight='bold')
        axes[1, 0].set_title('Error Percentiles', fontsize=12, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3, axis='y')

        for i, v in enumerate(percentile_values):
            axes[1, 0].text(i, v + 0.5, f'{v:.1f}', ha='center', va='bottom',
                            fontweight='bold', fontsize=9)

        overest = (all_preds['error'] > 0).sum()
        underest = (all_preds['error'] < 0).sum()
        axes[1, 1].pie([overest, underest],
                       labels=['Overestimated', 'Underestimated'],
                       autopct='%1.1f%%',
                       colors=['#D62828', '#2E86AB'],
                       startangle=90,
                       textprops={'fontsize': 11, 'fontweight': 'bold'})
        axes[1, 1].set_title('Prediction Bias', fontsize=12, fontweight='bold')

        plt.tight_layout()

        output_path = self.charts_dir / 'error_distribution.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        return output_path

    def _plot_salary_tier_performance(self, results: Dict[str, Any]) -> Optional[Path]:
        """Plot performance by salary tier."""
        if 'tier_comparison' not in results:
            return None

        tier_df = results['tier_comparison']

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        x = range(len(tier_df))
        width = 0.35

        axes[0].bar([i - width/2 for i in x], tier_df['model_mape'],
                    width, label='Model', color='#2E86AB', alpha=0.8, edgecolor='black')
        axes[0].bar([i + width/2 for i in x], tier_df['benchmark_mape'],
                    width, label='Benchmark', color='#A23B72', alpha=0.8, edgecolor='black')

        axes[0].set_xlabel('Salary Tier', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('MAPE (%)', fontsize=12, fontweight='bold')
        axes[0].set_title('MAPE by Salary Tier', fontsize=13, fontweight='bold')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(tier_df['salary_tier'], rotation=0)
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3, axis='y')

        colors = ['#06A77D' if imp > 0 else '#D62828' for imp in tier_df['mape_improvement']]
        axes[1].bar(x, tier_df['mape_improvement'], color=colors, alpha=0.8, edgecolor='black')
        axes[1].axhline(0, color='black', linewidth=1)
        axes[1].set_xlabel('Salary Tier', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('MAPE Improvement (%)', fontsize=12, fontweight='bold')
        axes[1].set_title('Model Improvement by Salary Tier', fontsize=13, fontweight='bold')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(tier_df['salary_tier'], rotation=0)
        axes[1].grid(True, alpha=0.3, axis='y')

        for i, v in enumerate(tier_df['mape_improvement']):
            axes[1].text(i, v + (1 if v > 0 else -1), f'{v:+.1f}%',
                        ha='center', va='bottom' if v > 0 else 'top',
                        fontweight='bold', fontsize=9)

        plt.tight_layout()

        output_path = self.charts_dir / 'salary_tier_performance.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        return output_path

    def _plot_correlation_scatter(self, results: Dict[str, Any]) -> Optional[Path]:
        """Plot actual vs predicted scatter plot."""
        if 'all_predictions' not in results or results['all_predictions'].empty:
            return None

        all_preds = results['all_predictions']

        if 'actual_fpts' not in all_preds.columns or 'projected_fpts' not in all_preds.columns:
            return None

        fig, ax = plt.subplots(figsize=(10, 10))

        ax.scatter(all_preds['actual_fpts'], all_preds['projected_fpts'],
                   alpha=0.3, s=20, color='#2E86AB')

        max_val = max(all_preds['actual_fpts'].max(), all_preds['projected_fpts'].max())
        ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect Prediction')

        from scipy import stats as scipy_stats
        slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(
            all_preds['actual_fpts'], all_preds['projected_fpts']
        )
        line = slope * all_preds['actual_fpts'] + intercept
        ax.plot(all_preds['actual_fpts'], line, 'g-', linewidth=2,
                label=f'Fit: y={slope:.2f}x+{intercept:.2f} (r={r_value:.3f})')

        ax.set_xlabel('Actual Fantasy Points', fontsize=12, fontweight='bold')
        ax.set_ylabel('Predicted Fantasy Points', fontsize=12, fontweight='bold')
        ax.set_title('Actual vs Predicted Fantasy Points', fontsize=14, fontweight='bold', pad=20)
        ax.legend(fontsize=10, loc='upper left')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        output_path = self.charts_dir / 'correlation_scatter.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        return output_path

    def _plot_metrics_comparison(self, results: Dict[str, Any]) -> Optional[Path]:
        """Plot comprehensive metrics comparison."""
        if 'daily_results' not in results:
            return None

        daily_df = results['daily_results']

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        axes[0, 0].plot(daily_df['model_mape'], marker='o', linewidth=2,
                        color='#2E86AB', label='MAPE')
        axes[0, 0].fill_between(range(len(daily_df)),
                                daily_df['model_mape'],
                                alpha=0.3, color='#2E86AB')
        axes[0, 0].set_ylabel('MAPE (%)', fontsize=11, fontweight='bold')
        axes[0, 0].set_title('Mean Absolute Percentage Error', fontsize=12, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend(fontsize=9)

        axes[0, 1].plot(daily_df['model_rmse'], marker='s', linewidth=2,
                        color='#A23B72', label='RMSE')
        axes[0, 1].fill_between(range(len(daily_df)),
                                daily_df['model_rmse'],
                                alpha=0.3, color='#A23B72')
        axes[0, 1].set_ylabel('RMSE', fontsize=11, fontweight='bold')
        axes[0, 1].set_title('Root Mean Squared Error', fontsize=12, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend(fontsize=9)

        axes[1, 0].plot(daily_df['model_mae'], marker='^', linewidth=2,
                        color='#06A77D', label='MAE')
        axes[1, 0].fill_between(range(len(daily_df)),
                                daily_df['model_mae'],
                                alpha=0.3, color='#06A77D')
        axes[1, 0].set_xlabel('Slate Number', fontsize=11, fontweight='bold')
        axes[1, 0].set_ylabel('MAE', fontsize=11, fontweight='bold')
        axes[1, 0].set_title('Mean Absolute Error', fontsize=12, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend(fontsize=9)

        axes[1, 1].plot(daily_df['model_corr'], marker='D', linewidth=2,
                        color='#F77F00', label='Correlation')
        axes[1, 1].fill_between(range(len(daily_df)),
                                daily_df['model_corr'],
                                alpha=0.3, color='#F77F00')
        axes[1, 1].axhline(0.7, color='green', linestyle='--', alpha=0.5, label='Target (0.7)')
        axes[1, 1].set_xlabel('Slate Number', fontsize=11, fontweight='bold')
        axes[1, 1].set_ylabel('Correlation', fontsize=11, fontweight='bold')
        axes[1, 1].set_title('Pearson Correlation', fontsize=12, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend(fontsize=9)

        plt.tight_layout()

        output_path = self.charts_dir / 'metrics_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        return output_path
