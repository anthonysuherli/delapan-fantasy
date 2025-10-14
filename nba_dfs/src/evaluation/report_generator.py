import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import logging

from src.evaluation.visualizations import BacktestVisualizer

logger = logging.getLogger(__name__)


class BacktestReportGenerator:
    """
    Generate comprehensive markdown reports for backtest results.
    """

    def __init__(self, output_dir: Path):
        """
        Initialize report generator.

        Args:
            output_dir: Directory to save the report
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_report(
        self,
        results: Dict[str, Any],
        config: Dict[str, Any],
        run_timestamp: str,
        generate_charts: bool = True
    ) -> Path:
        """
        Generate comprehensive backtest report.

        Args:
            results: Backtest results dictionary
            config: Backtest configuration parameters
            run_timestamp: Timestamp of the run
            generate_charts: Whether to generate visualization charts

        Returns:
            Path to generated report file
        """
        chart_paths = {}
        if generate_charts:
            try:
                logger.info("Generating visualization charts...")
                visualizer = BacktestVisualizer(self.output_dir)
                chart_paths = visualizer.generate_all_charts(results)
                logger.info(f"Generated {len(chart_paths)} charts")
            except Exception as e:
                logger.error(f"Failed to generate charts: {str(e)}")

        report_path = self.output_dir / f"backtest_report_{run_timestamp}.md"

        with open(report_path, 'w') as f:
            self._write_header(f, run_timestamp)
            self._write_configuration(f, config)
            self._write_executive_summary(f, results)
            self._write_performance_metrics(f, results)

            if chart_paths.get('daily_mape'):
                self._write_section_with_chart(f, "Daily MAPE Trend", chart_paths['daily_mape'])

            self._write_benchmark_comparison(f, results)

            if chart_paths.get('model_vs_benchmark'):
                self._write_section_with_chart(f, "Model vs Benchmark Visualizations",
                                               chart_paths['model_vs_benchmark'])

            self._write_salary_tier_analysis(f, results)

            if chart_paths.get('salary_tier'):
                self._write_section_with_chart(f, "Salary Tier Performance Chart",
                                               chart_paths['salary_tier'])

            self._write_statistical_tests(f, results)
            self._write_daily_performance(f, results)
            self._write_error_analysis(f, results)

            if chart_paths.get('error_distribution'):
                self._write_section_with_chart(f, "Error Distribution Visualizations",
                                               chart_paths['error_distribution'])

            if chart_paths.get('correlation_scatter'):
                self._write_section_with_chart(f, "Correlation Analysis",
                                               chart_paths['correlation_scatter'])

            if chart_paths.get('metrics_comparison'):
                self._write_section_with_chart(f, "All Metrics Comparison",
                                               chart_paths['metrics_comparison'])

            self._write_footer(f, run_timestamp)

        logger.info(f"Generated comprehensive report: {report_path}")
        return report_path

    def _write_section_with_chart(self, f, title: str, chart_path: Path):
        """Write a section with embedded chart image."""
        f.write(f"## {title}\n\n")
        relative_path = Path('charts') / chart_path.name
        f.write(f"![{title}]({relative_path})\n\n")
        f.write("---\n\n")

    def _write_header(self, f, run_timestamp: str):
        """Write report header."""
        f.write("# NBA DFS Backtest Report\n\n")
        f.write(f"**Run Timestamp:** {run_timestamp}\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")

    def _write_configuration(self, f, config: Dict[str, Any]):
        """Write backtest configuration section."""
        f.write("## Configuration\n\n")

        f.write("### Date Ranges\n")
        f.write(f"- **Training Period:** {config.get('train_start', 'N/A')} to {config.get('train_end', 'N/A')}\n")
        f.write(f"- **Testing Period:** {config.get('test_start', 'N/A')} to {config.get('test_end', 'N/A')}\n")
        f.write(f"- **Number of Seasons:** {config.get('num_seasons', 'N/A')}\n\n")

        f.write("### Model Configuration\n")
        f.write(f"- **Model Type:** {config.get('model_type', 'N/A')}\n")
        f.write(f"- **Feature Config:** {config.get('feature_config', 'N/A')}\n")
        f.write(f"- **Per-Player Models:** {config.get('per_player_models', False)}\n")
        f.write(f"- **Recalibrate Days:** {config.get('recalibrate_days', 'N/A')}\n")
        f.write(f"- **Parallel Jobs:** {config.get('n_jobs', 1)}\n")
        f.write(f"- **Rewrite Models:** {config.get('rewrite_models', False)}\n\n")

        if config.get('model_params'):
            f.write("### Model Hyperparameters\n")
            for key, value in config['model_params'].items():
                f.write(f"- **{key}:** {value}\n")
            f.write("\n")

        f.write("---\n\n")

    def _write_executive_summary(self, f, results: Dict[str, Any]):
        """Write executive summary section."""
        f.write("## Executive Summary\n\n")

        if 'error' in results:
            f.write(f"**ERROR:** {results['error']}\n\n")
            return

        f.write(f"**Date Range:** {results.get('date_range', 'N/A')}\n\n")
        f.write(f"**Total Slates Processed:** {results.get('num_slates', 0)}\n\n")
        f.write(f"**Total Players Evaluated:** {results.get('total_players_evaluated', 0):.0f}\n\n")
        f.write(f"**Average Players per Slate:** {results.get('avg_players_per_slate', 0):.1f}\n\n")

        model_mape = results.get('model_mean_mape', 0)
        benchmark_mape = results.get('benchmark_mean_mape', 0)
        improvement = results.get('mape_improvement', 0)

        f.write("### Key Performance Indicators\n\n")
        f.write(f"- **Model MAPE:** {model_mape:.2f}%\n")
        f.write(f"- **Benchmark MAPE:** {benchmark_mape:.2f}%\n")
        f.write(f"- **MAPE Improvement:** {improvement:+.2f}% ")
        f.write(f"({'✓ Model Better' if improvement > 0 else '✗ Benchmark Better'})\n")
        f.write(f"- **Model Correlation:** {results.get('model_mean_correlation', 0):.3f}\n\n")

        if 'statistical_test' in results:
            p_value = results['statistical_test']['p_value']
            is_significant = p_value < 0.05
            f.write(f"- **Statistical Significance:** ")
            f.write(f"{'✓ YES' if is_significant else '✗ NO'} (p={p_value:.4f})\n\n")

        f.write("---\n\n")

    def _write_performance_metrics(self, f, results: Dict[str, Any]):
        """Write detailed performance metrics."""
        f.write("## Performance Metrics\n\n")

        f.write("### Model Performance\n\n")
        f.write("| Metric | Mean | Median | Std Dev |\n")
        f.write("|--------|------|--------|----------|\n")
        f.write(f"| MAPE (%) | {results.get('model_mean_mape', 0):.2f} | ")
        f.write(f"{results.get('model_median_mape', 0):.2f} | ")
        f.write(f"{results.get('model_std_mape', 0):.2f} |\n")
        f.write(f"| RMSE | {results.get('model_mean_rmse', 0):.2f} | ")
        f.write(f"N/A | {results.get('model_std_rmse', 0):.2f} |\n")
        f.write(f"| MAE | {results.get('model_mean_mae', 0):.2f} | N/A | N/A |\n")
        f.write(f"| Correlation | {results.get('model_mean_correlation', 0):.3f} | ")
        f.write(f"N/A | {results.get('model_std_correlation', 0):.3f} |\n\n")

        f.write("### Benchmark Performance\n\n")
        f.write("| Metric | Mean | Median |\n")
        f.write("|--------|------|--------|\n")
        f.write(f"| MAPE (%) | {results.get('benchmark_mean_mape', 0):.2f} | ")
        f.write(f"{results.get('benchmark_median_mape', 0):.2f} |\n\n")

        f.write("---\n\n")

    def _write_benchmark_comparison(self, f, results: Dict[str, Any]):
        """Write benchmark comparison analysis."""
        if 'benchmark_comparison' not in results:
            return

        f.write("## Model vs Benchmark Comparison\n\n")

        comparison = results['benchmark_comparison']
        if 'summary' in comparison:
            f.write(comparison['summary'])
            f.write("\n\n")

        f.write("---\n\n")

    def _write_salary_tier_analysis(self, f, results: Dict[str, Any]):
        """Write salary tier performance breakdown."""
        if 'tier_comparison' not in results:
            return

        f.write("## Performance by Salary Tier\n\n")

        tier_df = results['tier_comparison']

        f.write("| Salary Tier | Count | Model MAPE | Benchmark MAPE | Improvement | Status |\n")
        f.write("|-------------|-------|------------|----------------|-------------|--------|\n")

        for _, row in tier_df.iterrows():
            tier = row['salary_tier']
            count = row['count']
            model_mape = row['model_mape']
            bench_mape = row['benchmark_mape']
            improvement = row['mape_improvement']
            status = '✓ Better' if improvement > 0 else '✗ Worse'

            f.write(f"| {tier} | {count} | {model_mape:.1f}% | {bench_mape:.1f}% | ")
            f.write(f"{improvement:+.1f}% | {status} |\n")

        f.write("\n")
        f.write("---\n\n")

    def _write_statistical_tests(self, f, results: Dict[str, Any]):
        """Write statistical significance tests."""
        if 'statistical_test' not in results:
            return

        f.write("## Statistical Significance Testing\n\n")

        test = results['statistical_test']

        f.write("### Paired t-Test Results\n\n")
        f.write(f"- **t-statistic:** {test['t_statistic']:.4f}\n")
        f.write(f"- **p-value:** {test['p_value']:.6f}\n")
        f.write(f"- **Significance Level:** 0.05\n\n")

        if test['p_value'] < 0.05:
            if test['t_statistic'] < 0:
                result = "✓ Model is SIGNIFICANTLY BETTER than benchmark"
            else:
                result = "✗ Model is SIGNIFICANTLY WORSE than benchmark"
        else:
            result = "~ No significant difference between model and benchmark"

        f.write(f"**Result:** {result}\n\n")

        f.write("### Effect Size (Cohen's d)\n\n")
        f.write(f"- **Cohen's d:** {test['cohens_d']:.4f}\n")
        f.write(f"- **Effect Size:** {test['effect_size']}\n\n")

        effect_interp = {
            'negligible': 'd < 0.2 (negligible effect)',
            'small': '0.2 ≤ d < 0.5 (small effect)',
            'medium': '0.5 ≤ d < 0.8 (medium effect)',
            'large': 'd ≥ 0.8 (large effect)'
        }
        f.write(f"**Interpretation:** {effect_interp.get(test['effect_size'], 'Unknown')}\n\n")

        f.write("---\n\n")

    def _write_daily_performance(self, f, results: Dict[str, Any]):
        """Write daily performance breakdown."""
        if 'daily_results' not in results:
            return

        f.write("## Daily Performance\n\n")

        daily_df = results['daily_results']

        f.write("### Top 5 Best Days (Lowest MAPE)\n\n")
        top_5 = daily_df.nsmallest(5, 'model_mape')

        f.write("| Date | MAPE (%) | RMSE | Players | Benchmark MAPE |\n")
        f.write("|------|----------|------|---------|----------------|\n")

        for _, row in top_5.iterrows():
            f.write(f"| {row['date']} | {row['model_mape']:.2f} | ")
            f.write(f"{row['model_rmse']:.2f} | {row['num_players']} | ")
            f.write(f"{row.get('benchmark_mape', 0):.2f} |\n")

        f.write("\n")

        f.write("### Bottom 5 Worst Days (Highest MAPE)\n\n")
        bottom_5 = daily_df.nlargest(5, 'model_mape')

        f.write("| Date | MAPE (%) | RMSE | Players | Benchmark MAPE |\n")
        f.write("|------|----------|------|---------|----------------|\n")

        for _, row in bottom_5.iterrows():
            f.write(f"| {row['date']} | {row['model_mape']:.2f} | ")
            f.write(f"{row['model_rmse']:.2f} | {row['num_players']} | ")
            f.write(f"{row.get('benchmark_mape', 0):.2f} |\n")

        f.write("\n")

        f.write("### Performance Distribution\n\n")
        f.write(f"- **25th Percentile MAPE:** {daily_df['model_mape'].quantile(0.25):.2f}%\n")
        f.write(f"- **50th Percentile MAPE:** {daily_df['model_mape'].quantile(0.50):.2f}%\n")
        f.write(f"- **75th Percentile MAPE:** {daily_df['model_mape'].quantile(0.75):.2f}%\n\n")

        f.write("---\n\n")

    def _write_error_analysis(self, f, results: Dict[str, Any]):
        """Write error analysis section."""
        if 'all_predictions' not in results or results['all_predictions'].empty:
            return

        f.write("## Error Analysis\n\n")

        all_preds = results['all_predictions']

        if 'actual_fpts' in all_preds.columns and 'projected_fpts' in all_preds.columns:
            all_preds = all_preds.copy()
            all_preds['error'] = all_preds['projected_fpts'] - all_preds['actual_fpts']
            all_preds['abs_error'] = np.abs(all_preds['error'])
            all_preds['pct_error'] = np.abs(all_preds['error'] / all_preds['actual_fpts']) * 100

            valid_preds = all_preds[all_preds['actual_fpts'] > 0].copy()

            f.write("### Error Distribution\n\n")
            f.write(f"- **Mean Error:** {all_preds['error'].mean():+.2f} fpts\n")
            f.write(f"- **Mean Absolute Error:** {all_preds['abs_error'].mean():.2f} fpts\n")
            f.write(f"- **Median Absolute Error:** {all_preds['abs_error'].median():.2f} fpts\n")
            f.write(f"- **Std Dev of Error:** {all_preds['error'].std():.2f} fpts\n\n")

            f.write("### Error Percentiles\n\n")
            f.write(f"- **10th Percentile:** {all_preds['abs_error'].quantile(0.10):.2f} fpts\n")
            f.write(f"- **25th Percentile:** {all_preds['abs_error'].quantile(0.25):.2f} fpts\n")
            f.write(f"- **50th Percentile:** {all_preds['abs_error'].quantile(0.50):.2f} fpts\n")
            f.write(f"- **75th Percentile:** {all_preds['abs_error'].quantile(0.75):.2f} fpts\n")
            f.write(f"- **90th Percentile:** {all_preds['abs_error'].quantile(0.90):.2f} fpts\n\n")

            overestimated = (all_preds['error'] > 0).sum()
            underestimated = (all_preds['error'] < 0).sum()
            total = len(all_preds)

            f.write("### Prediction Bias\n\n")
            f.write(f"- **Overestimated:** {overestimated} ({overestimated/total*100:.1f}%)\n")
            f.write(f"- **Underestimated:** {underestimated} ({underestimated/total*100:.1f}%)\n\n")

        f.write("---\n\n")

    def _write_footer(self, f, run_timestamp: str):
        """Write report footer."""
        f.write("## Report Metadata\n\n")
        f.write(f"- **Report Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- **Run Timestamp:** {run_timestamp}\n")
        f.write(f"- **Generator Version:** 1.0.0\n\n")
        f.write("---\n\n")
        f.write("*Generated by NBA DFS Backtest Report Generator*\n")
