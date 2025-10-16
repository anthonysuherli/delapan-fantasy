import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import logging

from src.evaluation.visualizations import BacktestVisualizer
from src.evaluation.plotly_visualizations import PlotlyBacktestVisualizer

logger = logging.getLogger(__name__)


class BacktestReportGenerator:
    """
    Generate comprehensive HTML reports for backtest results.
    """

    def __init__(self, output_dir: Path, use_plotly: bool = True):
        """
        Initialize report generator.

        Args:
            output_dir: Directory to save the report
            use_plotly: Whether to use Plotly for interactive charts (default: True)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_plotly = use_plotly

    def generate_report(
        self,
        results: Dict[str, Any],
        config: Dict[str, Any],
        run_timestamp: str,
        generate_charts: bool = True
    ) -> Path:
        """
        Generate comprehensive backtest report in HTML format.

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
                if self.use_plotly:
                    logger.info("Using Plotly for interactive visualizations...")
                    visualizer = PlotlyBacktestVisualizer(self.output_dir)
                else:
                    logger.info("Using matplotlib for static visualizations...")
                    visualizer = BacktestVisualizer(self.output_dir)
                chart_paths = visualizer.generate_all_charts(results)
                logger.info(f"Generated {len(chart_paths)} charts")
            except Exception as e:
                logger.error(f"Failed to generate charts: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())

        report_path = self.output_dir / f"backtest_report_{run_timestamp}.html"

        with open(report_path, 'w', encoding='utf-8') as f:
            self._write_html_header(f)
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

            if chart_paths.get('error_analysis'):
                self._write_section_with_chart(f, "Error Analysis Dashboard",
                                               chart_paths['error_analysis'])

            if chart_paths.get('correlation_scatter'):
                self._write_section_with_chart(f, "Correlation Analysis",
                                               chart_paths['correlation_scatter'])

            if chart_paths.get('correlation_analysis'):
                self._write_section_with_chart(f, "Interactive Correlation Analysis",
                                               chart_paths['correlation_analysis'])

            if chart_paths.get('metrics_comparison'):
                self._write_section_with_chart(f, "All Metrics Comparison",
                                               chart_paths['metrics_comparison'])

            if chart_paths.get('position_analysis'):
                self._write_section_with_chart(f, "Performance by Position",
                                               chart_paths['position_analysis'])

            if chart_paths.get('minutes_analysis'):
                self._write_section_with_chart(f, "Performance by Minutes Played",
                                               chart_paths['minutes_analysis'])

            if chart_paths.get('team_analysis'):
                self._write_section_with_chart(f, "Performance by Team",
                                               chart_paths['team_analysis'])

            if chart_paths.get('residual_analysis'):
                self._write_section_with_chart(f, "Residual Analysis",
                                               chart_paths['residual_analysis'])

            if chart_paths.get('statistical_tests'):
                self._write_section_with_chart(f, "Statistical Test Results",
                                               chart_paths['statistical_tests'])

            if chart_paths.get('calibration_curve'):
                self._write_section_with_chart(f, "Model Calibration",
                                               chart_paths['calibration_curve'])

            if chart_paths.get('error_heatmap'):
                self._write_section_with_chart(f, "Error Heatmap (Tier x Position)",
                                               chart_paths['error_heatmap'])

            if chart_paths.get('comprehensive_dashboard'):
                self._write_section_with_chart(f, "Comprehensive Dashboard",
                                               chart_paths['comprehensive_dashboard'])

            self._write_footer(f, run_timestamp)
            self._write_html_footer(f)

        logger.info(f"Generated comprehensive report: {report_path}")
        return report_path

    def _write_html_header(self, f):
        """Write HTML document header with CSS styling."""
        f.write("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NBA DFS Backtest Report</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #0d1117;
            color: #c9d1d9;
        }
        .container {
            background-color: #161b22;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.4);
            padding: 40px;
            margin-bottom: 20px;
            border: 1px solid #30363d;
        }
        h1 {
            color: #f0f6fc;
            border-bottom: 3px solid #1f6feb;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }
        h2 {
            color: #e6edf3;
            border-bottom: 2px solid #30363d;
            padding-bottom: 8px;
            margin-top: 30px;
            margin-bottom: 15px;
        }
        h3 {
            color: #c9d1d9;
            margin-top: 20px;
            margin-bottom: 10px;
        }
        table {
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
            box-shadow: 0 1px 3px rgba(0,0,0,0.3);
            background-color: #0d1117;
        }
        th, td {
            text-align: left;
            padding: 12px;
            border-bottom: 1px solid #30363d;
        }
        th {
            background-color: #1f6feb;
            color: #ffffff;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.85em;
            letter-spacing: 0.5px;
        }
        tr:hover {
            background-color: #1c2128;
        }
        .metric-value {
            font-weight: 600;
            color: #58a6ff;
        }
        .positive {
            color: #3fb950;
        }
        .negative {
            color: #f85149;
        }
        .chart-container {
            margin: 30px 0;
            padding: 20px;
            background-color: #0d1117;
            border-radius: 8px;
            border: 1px solid #30363d;
        }
        iframe {
            border: none;
            border-radius: 4px;
            background-color: #ffffff;
        }
        .metadata {
            background-color: #0d1117;
            padding: 15px;
            border-left: 4px solid #1f6feb;
            margin: 20px 0;
            border-radius: 4px;
            border: 1px solid #30363d;
        }
        .summary-box {
            background-color: #1c2d41;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
            border-left: 5px solid #1f6feb;
            border: 1px solid #30363d;
        }
        .warning-box {
            background-color: #3d2c00;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
            border-left: 5px solid #d29922;
            border: 1px solid #30363d;
        }
        .error-box {
            background-color: #3d1f1f;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
            border-left: 5px solid #f85149;
            border: 1px solid #30363d;
        }
        .success-box {
            background-color: #1f3d26;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
            border-left: 5px solid #3fb950;
            border: 1px solid #30363d;
        }
        ul {
            padding-left: 20px;
        }
        li {
            margin: 8px 0;
        }
        hr {
            border: none;
            border-top: 1px solid #30363d;
            margin: 30px 0;
        }
        .footer {
            text-align: center;
            color: #8b949e;
            font-size: 0.9em;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #30363d;
        }
        code {
            background-color: #0d1117;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
            color: #79c0ff;
            border: 1px solid #30363d;
        }
    </style>
</head>
<body>
<div class="container">
""")

    def _write_html_footer(self, f):
        """Write HTML document footer."""
        f.write("""
</div>
</body>
</html>
""")

    def _write_section_with_chart(self, f, title: str, chart_path: Path):
        """Write a section with embedded chart image or interactive chart."""
        f.write(f"<h2>{title}</h2>\n")
        relative_path = Path('charts') / chart_path.name

        f.write('<div class="chart-container">\n')
        if chart_path.suffix == '.html':
            f.write(f'<iframe src="{relative_path}" width="100%" height="600px"></iframe>\n')
        else:
            f.write(f'<img src="{relative_path}" alt="{title}" style="max-width: 100%; height: auto;">\n')
        f.write('</div>\n\n')

    def _write_header(self, f, run_timestamp: str):
        """Write report header."""
        f.write("<h1>NBA DFS Backtest Report</h1>\n")
        f.write('<div class="metadata">\n')
        f.write(f"<strong>Run Timestamp:</strong> {run_timestamp}<br>\n")
        f.write(f"<strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("</div>\n\n")

    def _write_configuration(self, f, config: Dict[str, Any]):
        """Write backtest configuration section."""
        f.write("<h2>Configuration</h2>\n")

        f.write("<h3>Date Ranges</h3>\n<ul>\n")
        f.write(f"<li><strong>Training Period:</strong> {config.get('train_start', 'N/A')} to {config.get('train_end', 'N/A')}</li>\n")
        f.write(f"<li><strong>Testing Period:</strong> {config.get('test_start', 'N/A')} to {config.get('test_end', 'N/A')}</li>\n")
        f.write(f"<li><strong>Number of Seasons:</strong> {config.get('num_seasons', 'N/A')}</li>\n")
        f.write("</ul>\n\n")

        f.write("<h3>Model Configuration</h3>\n<ul>\n")
        f.write(f"<li><strong>Model Type:</strong> {config.get('model_type', 'N/A')}</li>\n")
        f.write(f"<li><strong>Feature Config:</strong> {config.get('feature_config', 'N/A')}</li>\n")
        f.write(f"<li><strong>Per-Player Models:</strong> {config.get('per_player_models', False)}</li>\n")
        f.write(f"<li><strong>Recalibrate Days:</strong> {config.get('recalibrate_days', 'N/A')}</li>\n")
        f.write(f"<li><strong>Parallel Jobs:</strong> {config.get('n_jobs', 1)}</li>\n")
        f.write(f"<li><strong>Rewrite Models:</strong> {config.get('rewrite_models', False)}</li>\n")
        f.write("</ul>\n\n")

        if config.get('model_params'):
            f.write("<h3>Model Hyperparameters</h3>\n<ul>\n")
            for key, value in config['model_params'].items():
                f.write(f"<li><strong>{key}:</strong> {value}</li>\n")
            f.write("</ul>\n\n")

        f.write("<hr>\n\n")

    def _write_executive_summary(self, f, results: Dict[str, Any]):
        """Write executive summary section."""
        f.write("<h2>Executive Summary</h2>\n")

        if 'error' in results:
            f.write(f'<div class="error-box"><strong>ERROR:</strong> {results["error"]}</div>\n\n')
            return

        f.write('<div class="summary-box">\n')
        f.write(f"<strong>Date Range:</strong> {results.get('date_range', 'N/A')}<br>\n")
        f.write(f"<strong>Total Slates Processed:</strong> {results.get('num_slates', 0)}<br>\n")
        f.write(f"<strong>Total Players Evaluated:</strong> {results.get('total_players_evaluated', 0):.0f}<br>\n")
        f.write(f"<strong>Average Players per Slate:</strong> {results.get('avg_players_per_slate', 0):.1f}\n")
        f.write("</div>\n\n")

        model_mape = results.get('model_mean_mape', 0)
        benchmark_mape = results.get('benchmark_mean_mape', 0)
        improvement = results.get('mape_improvement', 0)

        f.write("<h3>Key Performance Indicators</h3>\n<ul>\n")
        f.write(f'<li><strong>Model MAPE:</strong> <span class="metric-value">{model_mape:.2f}%</span></li>\n')
        f.write(f'<li><strong>Benchmark MAPE:</strong> <span class="metric-value">{benchmark_mape:.2f}%</span></li>\n')
        improvement_class = 'positive' if improvement > 0 else 'negative'
        improvement_icon = '✓ Model Better' if improvement > 0 else '✗ Benchmark Better'
        f.write(f'<li><strong>MAPE Improvement:</strong> <span class="{improvement_class}">{improvement:+.2f}% ({improvement_icon})</span></li>\n')
        f.write(f'<li><strong>Model Correlation:</strong> <span class="metric-value">{results.get("model_mean_correlation", 0):.3f}</span></li>\n')

        if 'statistical_test' in results:
            p_value = results['statistical_test']['p_value']
            is_significant = p_value < 0.05
            sig_class = 'positive' if is_significant else 'negative'
            sig_text = '✓ YES' if is_significant else '✗ NO'
            f.write(f'<li><strong>Statistical Significance:</strong> <span class="{sig_class}">{sig_text} (p={p_value:.4f})</span></li>\n')

        f.write("</ul>\n\n<hr>\n\n")

    def _write_performance_metrics(self, f, results: Dict[str, Any]):
        """Write detailed performance metrics."""
        f.write("<h2>Performance Metrics</h2>\n")

        f.write("<h3>Model Performance</h3>\n")
        f.write("<table>\n<thead><tr><th>Metric</th><th>Mean</th><th>Median</th><th>Std Dev</th></tr></thead>\n<tbody>\n")
        f.write(f"<tr><td>MAPE (%)</td><td>{results.get('model_mean_mape', 0):.2f}</td>")
        f.write(f"<td>{results.get('model_median_mape', 0):.2f}</td>")
        f.write(f"<td>{results.get('model_std_mape', 0):.2f}</td></tr>\n")
        f.write(f"<tr><td>cMAPE (%)</td><td>{results.get('model_mean_cmape', 0):.2f}</td>")
        f.write(f"<td>N/A</td><td>N/A</td></tr>\n")
        f.write(f"<tr><td>SMAPE (%)</td><td>{results.get('model_mean_smape', 0):.2f}</td>")
        f.write(f"<td>N/A</td><td>N/A</td></tr>\n")
        f.write(f"<tr><td>WMAPE (%)</td><td>{results.get('model_mean_wmape', 0):.2f}</td>")
        f.write(f"<td>N/A</td><td>N/A</td></tr>\n")
        f.write(f"<tr><td>RMSE</td><td>{results.get('model_mean_rmse', 0):.2f}</td>")
        f.write(f"<td>N/A</td><td>{results.get('model_std_rmse', 0):.2f}</td></tr>\n")
        f.write(f"<tr><td>MAE</td><td>{results.get('model_mean_mae', 0):.2f}</td><td>N/A</td><td>N/A</td></tr>\n")
        f.write(f"<tr><td>Correlation</td><td>{results.get('model_mean_correlation', 0):.3f}</td>")
        f.write(f"<td>N/A</td><td>{results.get('model_std_correlation', 0):.3f}</td></tr>\n")
        f.write("</tbody>\n</table>\n\n")

        f.write("<h3>Benchmark Performance</h3>\n")
        f.write("<table>\n<thead><tr><th>Metric</th><th>Mean</th><th>Median</th></tr></thead>\n<tbody>\n")
        f.write(f"<tr><td>MAPE (%)</td><td>{results.get('benchmark_mean_mape', 0):.2f}</td>")
        f.write(f"<td>{results.get('benchmark_median_mape', 0):.2f}</td></tr>\n")
        f.write(f"<tr><td>cMAPE (%)</td><td>{results.get('benchmark_mean_cmape', 0):.2f}</td><td>N/A</td></tr>\n")
        f.write(f"<tr><td>WMAPE (%)</td><td>{results.get('benchmark_mean_wmape', 0):.2f}</td><td>N/A</td></tr>\n")
        f.write("</tbody>\n</table>\n\n")

        if 'low_minutes_metrics' in results:
            f.write("<h3>Low-Minutes Cohort (< threshold)</h3>\n")
            lm = results['low_minutes_metrics']
            f.write("<table>\n<thead><tr><th>Metric</th><th>Value</th></tr></thead>\n<tbody>\n")
            f.write(f"<tr><td>Count</td><td>{lm.get('count', 0)}</td></tr>\n")
            f.write(f"<tr><td>cMAPE (%)</td><td>{lm.get('cmape', float('nan')):.2f}</td></tr>\n")
            f.write(f"<tr><td>SMAPE (%)</td><td>{lm.get('smape', float('nan')):.2f}</td></tr>\n")
            f.write(f"<tr><td>WMAPE (%)</td><td>{lm.get('wmape', float('nan')):.2f}</td></tr>\n")
            f.write("</tbody>\n</table>\n\n")

        f.write("<hr>\n\n")

    def _write_benchmark_comparison(self, f, results: Dict[str, Any]):
        """Write benchmark comparison analysis."""
        if 'benchmark_comparison' not in results:
            return

        f.write("<h2>Model vs Benchmark Comparison</h2>\n")

        comparison = results['benchmark_comparison']
        if 'summary' in comparison:
            f.write('<div class="summary-box">\n')
            f.write(comparison['summary'])
            f.write("\n</div>\n\n")

        f.write("<hr>\n\n")

    def _write_salary_tier_analysis(self, f, results: Dict[str, Any]):
        """Write salary tier performance breakdown."""
        if 'tier_comparison' not in results:
            return

        f.write("<h2>Performance by Salary Tier</h2>\n")

        tier_df = results['tier_comparison']

        f.write("<table>\n<thead><tr><th>Salary Tier</th><th>Count</th><th>Model MAPE</th>")
        f.write("<th>Benchmark MAPE</th><th>Improvement</th><th>Status</th></tr></thead>\n<tbody>\n")

        for _, row in tier_df.iterrows():
            tier = row['salary_tier']
            count = row['count']
            model_mape = row['model_mape']
            bench_mape = row['benchmark_mape']
            improvement = row['mape_improvement']
            status_class = 'positive' if improvement > 0 else 'negative'
            status_text = '✓ Better' if improvement > 0 else '✗ Worse'

            f.write(f"<tr><td>{tier}</td><td>{count}</td><td>{model_mape:.1f}%</td>")
            f.write(f"<td>{bench_mape:.1f}%</td>")
            f.write(f'<td class="{status_class}">{improvement:+.1f}%</td>')
            f.write(f'<td class="{status_class}">{status_text}</td></tr>\n')

        f.write("</tbody>\n</table>\n\n<hr>\n\n")

    def _write_statistical_tests(self, f, results: Dict[str, Any]):
        """Write statistical significance tests."""
        if 'statistical_test' not in results:
            return

        f.write("<h2>Statistical Significance Testing</h2>\n")

        test = results['statistical_test']

        f.write("<h3>Paired t-Test Results</h3>\n<ul>\n")
        f.write(f"<li><strong>t-statistic:</strong> {test['t_statistic']:.4f}</li>\n")
        f.write(f"<li><strong>p-value:</strong> {test['p_value']:.6f}</li>\n")
        f.write(f"<li><strong>Significance Level:</strong> 0.05</li>\n")
        f.write("</ul>\n\n")

        if test['p_value'] < 0.05:
            if test['t_statistic'] < 0:
                result = "✓ Model is SIGNIFICANTLY BETTER than benchmark"
                box_class = "success-box"
            else:
                result = "✗ Model is SIGNIFICANTLY WORSE than benchmark"
                box_class = "error-box"
        else:
            result = "~ No significant difference between model and benchmark"
            box_class = "warning-box"

        f.write(f'<div class="{box_class}"><strong>Result:</strong> {result}</div>\n\n')

        f.write("<h3>Effect Size (Cohen's d)</h3>\n<ul>\n")
        f.write(f"<li><strong>Cohen's d:</strong> {test['cohens_d']:.4f}</li>\n")
        f.write(f"<li><strong>Effect Size:</strong> {test['effect_size']}</li>\n")

        effect_interp = {
            'negligible': 'd < 0.2 (negligible effect)',
            'small': '0.2 ≤ d < 0.5 (small effect)',
            'medium': '0.5 ≤ d < 0.8 (medium effect)',
            'large': 'd ≥ 0.8 (large effect)'
        }
        f.write(f"<li><strong>Interpretation:</strong> {effect_interp.get(test['effect_size'], 'Unknown')}</li>\n")
        f.write("</ul>\n\n<hr>\n\n")

    def _write_daily_performance(self, f, results: Dict[str, Any]):
        """Write daily performance breakdown."""
        if 'daily_results' not in results:
            return

        f.write("<h2>Daily Performance</h2>\n")

        daily_df = results['daily_results']

        f.write("<h3>Top 5 Best Days (Lowest MAPE)</h3>\n")
        top_5 = daily_df.nsmallest(5, 'model_mape')

        f.write("<table>\n<thead><tr><th>Date</th><th>MAPE (%)</th><th>RMSE</th>")
        f.write("<th>Players</th><th>Benchmark MAPE</th></tr></thead>\n<tbody>\n")

        for _, row in top_5.iterrows():
            f.write(f"<tr><td>{row['date']}</td><td>{row['model_mape']:.2f}</td>")
            f.write(f"<td>{row['model_rmse']:.2f}</td><td>{row['num_players']}</td>")
            f.write(f"<td>{row.get('benchmark_mape', 0):.2f}</td></tr>\n")

        f.write("</tbody>\n</table>\n\n")

        f.write("<h3>Bottom 5 Worst Days (Highest MAPE)</h3>\n")
        bottom_5 = daily_df.nlargest(5, 'model_mape')

        f.write("<table>\n<thead><tr><th>Date</th><th>MAPE (%)</th><th>RMSE</th>")
        f.write("<th>Players</th><th>Benchmark MAPE</th></tr></thead>\n<tbody>\n")

        for _, row in bottom_5.iterrows():
            f.write(f"<tr><td>{row['date']}</td><td>{row['model_mape']:.2f}</td>")
            f.write(f"<td>{row['model_rmse']:.2f}</td><td>{row['num_players']}</td>")
            f.write(f"<td>{row.get('benchmark_mape', 0):.2f}</td></tr>\n")

        f.write("</tbody>\n</table>\n\n")

        f.write("<h3>Performance Distribution</h3>\n<ul>\n")
        f.write(f"<li><strong>25th Percentile MAPE:</strong> {daily_df['model_mape'].quantile(0.25):.2f}%</li>\n")
        f.write(f"<li><strong>50th Percentile MAPE:</strong> {daily_df['model_mape'].quantile(0.50):.2f}%</li>\n")
        f.write(f"<li><strong>75th Percentile MAPE:</strong> {daily_df['model_mape'].quantile(0.75):.2f}%</li>\n")
        f.write("</ul>\n\n<hr>\n\n")

    def _write_error_analysis(self, f, results: Dict[str, Any]):
        """Write error analysis section."""
        if 'all_predictions' not in results or results['all_predictions'].empty:
            return

        f.write("<h2>Error Analysis</h2>\n")

        all_preds = results['all_predictions']

        if 'actual_fpts' in all_preds.columns and 'projected_fpts' in all_preds.columns:
            all_preds = all_preds.copy()
            all_preds['error'] = all_preds['projected_fpts'] - all_preds['actual_fpts']
            all_preds['abs_error'] = np.abs(all_preds['error'])
            all_preds['pct_error'] = np.abs(all_preds['error'] / all_preds['actual_fpts']) * 100

            valid_preds = all_preds[all_preds['actual_fpts'] > 0].copy()

            f.write("<h3>Error Distribution</h3>\n<ul>\n")
            f.write(f"<li><strong>Mean Error:</strong> {all_preds['error'].mean():+.2f} fpts</li>\n")
            f.write(f"<li><strong>Mean Absolute Error:</strong> {all_preds['abs_error'].mean():.2f} fpts</li>\n")
            f.write(f"<li><strong>Median Absolute Error:</strong> {all_preds['abs_error'].median():.2f} fpts</li>\n")
            f.write(f"<li><strong>Std Dev of Error:</strong> {all_preds['error'].std():.2f} fpts</li>\n")
            f.write("</ul>\n\n")

            f.write("<h3>Error Percentiles</h3>\n<ul>\n")
            f.write(f"<li><strong>10th Percentile:</strong> {all_preds['abs_error'].quantile(0.10):.2f} fpts</li>\n")
            f.write(f"<li><strong>25th Percentile:</strong> {all_preds['abs_error'].quantile(0.25):.2f} fpts</li>\n")
            f.write(f"<li><strong>50th Percentile:</strong> {all_preds['abs_error'].quantile(0.50):.2f} fpts</li>\n")
            f.write(f"<li><strong>75th Percentile:</strong> {all_preds['abs_error'].quantile(0.75):.2f} fpts</li>\n")
            f.write(f"<li><strong>90th Percentile:</strong> {all_preds['abs_error'].quantile(0.90):.2f} fpts</li>\n")
            f.write("</ul>\n\n")

            overestimated = (all_preds['error'] > 0).sum()
            underestimated = (all_preds['error'] < 0).sum()
            total = len(all_preds)

            f.write("<h3>Prediction Bias</h3>\n<ul>\n")
            f.write(f"<li><strong>Overestimated:</strong> {overestimated} ({overestimated/total*100:.1f}%)</li>\n")
            f.write(f"<li><strong>Underestimated:</strong> {underestimated} ({underestimated/total*100:.1f}%)</li>\n")
            f.write("</ul>\n\n")

        f.write("<hr>\n\n")

    def _write_footer(self, f, run_timestamp: str):
        """Write report footer."""
        f.write("<h2>Report Metadata</h2>\n")
        f.write('<div class="metadata">\n<ul>\n')
        f.write(f"<li><strong>Report Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</li>\n")
        f.write(f"<li><strong>Run Timestamp:</strong> {run_timestamp}</li>\n")
        f.write(f"<li><strong>Generator Version:</strong> 1.0.0</li>\n")
        f.write("</ul>\n</div>\n\n")
        f.write('<div class="footer">Generated by NBA DFS Backtest Report Generator</div>\n')
