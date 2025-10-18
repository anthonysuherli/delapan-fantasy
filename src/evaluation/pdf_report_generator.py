"""
PDF-styled backtest report generator with dark theme charts and analysis.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any
from datetime import datetime
import logging

from src.evaluation.chart_analyzer import ChartAnalyzer

logger = logging.getLogger(__name__)


class PDFStyleBacktestReportGenerator:
    """
    Generate PDF-styled HTML reports with dark theme charts and analysis.
    """

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.analyzer = None

    def generate_report(
        self,
        results: Dict[str, Any],
        config: Dict[str, Any],
        run_timestamp: str,
        chart_paths: Dict[str, Path]
    ) -> Path:
        """
        Generate comprehensive PDF-styled backtest report.

        Args:
            results: Backtest results dictionary
            config: Backtest configuration
            run_timestamp: Run timestamp
            chart_paths: Dictionary of chart paths

        Returns:
            Path to generated report
        """
        self.analyzer = ChartAnalyzer(results)

        report_path = self.output_dir / f"backtest_report_{run_timestamp}.html"

        with open(report_path, 'w', encoding='utf-8') as f:
            self._write_pdf_html_header(f)
            self._write_cover_page(f, run_timestamp, results)
            self._write_table_of_contents(f)
            self._write_executive_summary(f, results)
            self._write_configuration(f, config)
            self._write_performance_metrics_section(f, results)

            # Individual charts with analysis
            self._write_chart_with_analysis(
                f, "Model vs Benchmark Performance",
                chart_paths.get('model_vs_benchmark'),
                self.analyzer.analyze_model_vs_benchmark()
            )

            self._write_chart_with_analysis(
                f, "Salary Tier Analysis",
                chart_paths.get('salary_tier'),
                self.analyzer.analyze_salary_tier_performance()
            )

            self._write_chart_with_analysis(
                f, "Correlation Analysis",
                chart_paths.get('correlation_analysis'),
                self.analyzer.analyze_correlation()
            )

            self._write_chart_with_analysis(
                f, "Error Distribution",
                chart_paths.get('error_analysis'),
                self.analyzer.analyze_error_distribution()
            )

            self._write_chart_with_analysis(
                f, "Model Calibration",
                chart_paths.get('calibration_curve'),
                self.analyzer.analyze_calibration()
            )

            self._write_chart_with_analysis(
                f, "Residual Analysis",
                chart_paths.get('residual_analysis'),
                self.analyzer.analyze_residuals()
            )

            self._write_chart_with_analysis(
                f, "Position Performance",
                chart_paths.get('position_analysis'),
                self.analyzer.analyze_position_performance()
            )

            self._write_chart_with_analysis(
                f, "Minutes Impact Analysis",
                chart_paths.get('minutes_analysis'),
                self.analyzer.analyze_minutes_impact()
            )

            self._write_detailed_metrics(f, results)
            self._write_statistical_tests(f, results)
            self._write_recommendations(f, results)
            self._write_report_footer(f, run_timestamp)
            self._write_pdf_html_footer(f)

        logger.info(f"Generated PDF-styled report: {report_path}")
        return report_path

    def _write_pdf_html_header(self, f):
        """Write PDF-styled HTML header with print-optimized CSS."""
        f.write("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NBA DFS Backtest Report</title>
    <style>
        @media print {
            .page-break { page-break-before: always; }
            .no-print { display: none; }
            body { background-color: white !important; }
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            line-height: 1.6;
            background-color: #0d1117;
            color: #c9d1d9;
            font-size: 11pt;
        }

        .page {
            max-width: 210mm;
            min-height: 297mm;
            padding: 20mm;
            margin: 0 auto 10mm;
            background-color: #161b22;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
            border: 1px solid #30363d;
        }

        .cover-page {
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            text-align: center;
            min-height: 297mm;
            background: linear-gradient(135deg, #161b22 0%, #0d1117 100%);
            border: 2px solid #1f6feb;
        }

        .cover-title {
            font-size: 42pt;
            font-weight: 700;
            color: #f0f6fc;
            margin-bottom: 20px;
            letter-spacing: -1px;
        }

        .cover-subtitle {
            font-size: 18pt;
            color: #8b949e;
            margin-bottom: 40px;
        }

        .cover-meta {
            font-size: 14pt;
            color: #58a6ff;
            margin: 10px 0;
        }

        .cover-stats {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 30px;
            margin-top: 60px;
            width: 100%;
            max-width: 600px;
        }

        .cover-stat {
            background-color: #0d1117;
            padding: 20px;
            border-radius: 8px;
            border: 1px solid #30363d;
        }

        .cover-stat-value {
            font-size: 32pt;
            font-weight: 700;
            color: #58a6ff;
        }

        .cover-stat-label {
            font-size: 12pt;
            color: #8b949e;
            margin-top: 5px;
        }

        h1 {
            color: #f0f6fc;
            font-size: 28pt;
            margin: 30px 0 20px 0;
            padding-bottom: 10px;
            border-bottom: 3px solid #1f6feb;
        }

        h2 {
            color: #e6edf3;
            font-size: 20pt;
            margin: 25px 0 15px 0;
            padding-bottom: 8px;
            border-bottom: 2px solid #30363d;
        }

        h3 {
            color: #c9d1d9;
            font-size: 16pt;
            margin: 20px 0 10px 0;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background-color: #0d1117;
            font-size: 10pt;
        }

        th {
            background-color: #1f6feb;
            color: #ffffff;
            padding: 12px;
            text-align: left;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 9pt;
            letter-spacing: 0.5px;
            border: 1px solid #1f6feb;
        }

        td {
            padding: 10px 12px;
            border: 1px solid #30363d;
        }

        tr:nth-child(even) {
            background-color: #161b22;
        }

        .metric-card {
            background-color: #0d1117;
            padding: 20px;
            margin: 15px 0;
            border-radius: 8px;
            border-left: 4px solid #1f6feb;
            border: 1px solid #30363d;
        }

        .metric-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 15px;
            margin: 20px 0;
        }

        .metric-value {
            font-size: 24pt;
            font-weight: 700;
            color: #58a6ff;
        }

        .metric-label {
            font-size: 10pt;
            color: #8b949e;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .positive { color: #3fb950; }
        .negative { color: #f85149; }
        .neutral { color: #d29922; }

        .summary-box {
            background-color: #1c2d41;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
            border-left: 5px solid #1f6feb;
            border: 1px solid #30363d;
        }

        .analysis-box {
            background-color: #0d1117;
            padding: 20px;
            margin: 15px 0;
            border-radius: 8px;
            border-left: 4px solid #58a6ff;
            border: 1px solid #30363d;
            line-height: 1.8;
        }

        .chart-container {
            margin: 30px 0;
            padding: 20px;
            background-color: #0d1117;
            border-radius: 8px;
            border: 1px solid #30363d;
            page-break-inside: avoid;
        }

        .chart-title {
            font-size: 16pt;
            color: #f0f6fc;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid #30363d;
        }

        iframe {
            width: 100%;
            height: 600px;
            border: none;
            border-radius: 4px;
            background-color: #0d1117;
        }

        .toc {
            background-color: #0d1117;
            padding: 30px;
            border-radius: 8px;
            border: 1px solid #30363d;
            margin: 20px 0;
        }

        .toc-item {
            padding: 8px 0;
            border-bottom: 1px solid #30363d;
        }

        .toc-item:last-child {
            border-bottom: none;
        }

        .toc-link {
            color: #58a6ff;
            text-decoration: none;
        }

        .toc-link:hover {
            text-decoration: underline;
        }

        .footer {
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #30363d;
            text-align: center;
            color: #8b949e;
            font-size: 9pt;
        }

        code {
            background-color: #0d1117;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
            font-size: 9pt;
            color: #79c0ff;
            border: 1px solid #30363d;
        }

        .recommendation-box {
            background-color: #1f3d26;
            padding: 20px;
            margin: 15px 0;
            border-radius: 8px;
            border-left: 5px solid #3fb950;
            border: 1px solid #30363d;
        }

        .warning-box {
            background-color: #3d2c00;
            padding: 20px;
            margin: 15px 0;
            border-radius: 8px;
            border-left: 5px solid #d29922;
            border: 1px solid #30363d;
        }

        ul {
            padding-left: 20px;
            margin: 10px 0;
        }

        li {
            margin: 5px 0;
        }

        @page {
            size: A4;
            margin: 0;
        }
    </style>
</head>
<body>
""")

    def _write_cover_page(self, f, run_timestamp: str, results: Dict[str, Any]):
        """Write report cover page."""
        model_mape = results.get('model_mean_mape', 0)
        improvement = results.get('mape_improvement', 0)
        num_slates = results.get('num_slates', 0)
        total_players = results.get('total_players_evaluated', 0)

        f.write(f"""
<div class="page cover-page">
    <div class="cover-title">NBA DFS Backtest Report</div>
    <div class="cover-subtitle">Predictive Model Performance Analysis</div>
    <div class="cover-meta">Run ID: {run_timestamp}</div>
    <div class="cover-meta">Generated: {datetime.now().strftime('%B %d, %Y at %H:%M:%S')}</div>

    <div class="cover-stats">
        <div class="cover-stat">
            <div class="cover-stat-value">{model_mape:.1f}%</div>
            <div class="cover-stat-label">Model MAPE</div>
        </div>
        <div class="cover-stat">
            <div class="cover-stat-value {'positive' if improvement > 0 else 'negative'}">{improvement:+.1f}%</div>
            <div class="cover-stat-label">vs Benchmark</div>
        </div>
        <div class="cover-stat">
            <div class="cover-stat-value">{num_slates}</div>
            <div class="cover-stat-label">Slates Analyzed</div>
        </div>
        <div class="cover-stat">
            <div class="cover-stat-value">{int(total_players)}</div>
            <div class="cover-stat-label">Players Evaluated</div>
        </div>
    </div>
</div>

<div class="page-break"></div>
""")

    def _write_table_of_contents(self, f):
        """Write table of contents."""
        f.write("""
<div class="page">
    <h1>Table of Contents</h1>
    <div class="toc">
        <div class="toc-item"><a href="#executive-summary" class="toc-link">1. Executive Summary</a></div>
        <div class="toc-item"><a href="#configuration" class="toc-link">2. Configuration</a></div>
        <div class="toc-item"><a href="#performance-metrics" class="toc-link">3. Performance Metrics</a></div>
        <div class="toc-item"><a href="#model-vs-benchmark" class="toc-link">4. Model vs Benchmark Analysis</a></div>
        <div class="toc-item"><a href="#salary-tier" class="toc-link">5. Salary Tier Analysis</a></div>
        <div class="toc-item"><a href="#correlation" class="toc-link">6. Correlation Analysis</a></div>
        <div class="toc-item"><a href="#error-distribution" class="toc-link">7. Error Distribution</a></div>
        <div class="toc-item"><a href="#calibration" class="toc-link">8. Model Calibration</a></div>
        <div class="toc-item"><a href="#residuals" class="toc-link">9. Residual Analysis</a></div>
        <div class="toc-item"><a href="#position" class="toc-link">10. Position Performance</a></div>
        <div class="toc-item"><a href="#minutes" class="toc-link">11. Minutes Impact Analysis</a></div>
        <div class="toc-item"><a href="#detailed-metrics" class="toc-link">12. Detailed Metrics</a></div>
        <div class="toc-item"><a href="#statistical-tests" class="toc-link">13. Statistical Significance</a></div>
        <div class="toc-item"><a href="#recommendations" class="toc-link">14. Recommendations</a></div>
    </div>
</div>

<div class="page-break"></div>
""")

    def _write_executive_summary(self, f, results: Dict[str, Any]):
        """Write executive summary section."""
        model_mape = results.get('model_mean_mape', 0)
        benchmark_mape = results.get('benchmark_mean_mape', 0)
        improvement = results.get('mape_improvement', 0)
        correlation = results.get('model_mean_correlation', 0)
        num_slates = results.get('num_slates', 0)
        total_players = results.get('total_players_evaluated', 0)

        f.write(f"""
<div class="page">
    <h1 id="executive-summary">Executive Summary</h1>

    <div class="summary-box">
        <h3>Overview</h3>
        <p>This report analyzes the performance of a machine learning model for NBA DFS fantasy point predictions across {num_slates} game slates, evaluating {int(total_players)} player-game instances.</p>
    </div>

    <div class="metric-grid">
        <div class="metric-card">
            <div class="metric-value">{model_mape:.1f}%</div>
            <div class="metric-label">Model MAPE</div>
        </div>
        <div class="metric-card">
            <div class="metric-value {'positive' if improvement > 0 else 'negative'}">{improvement:+.1f}%</div>
            <div class="metric-label">vs Benchmark</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{correlation:.3f}</div>
            <div class="metric-label">Correlation</div>
        </div>
    </div>

    <h3>Key Findings</h3>
    <ul>
        <li>Model achieved {model_mape:.1f}% MAPE compared to benchmark's {benchmark_mape:.1f}% ({'outperformance' if improvement > 0 else 'underperformance'} of {abs(improvement):.1f}%)</li>
        <li>Correlation of {correlation:.3f} indicates {'strong' if correlation > 0.7 else 'moderate' if correlation > 0.6 else 'weak'} predictive relationship</li>
        <li>Model evaluated across {num_slates} distinct game slates with {int(total_players/num_slates):.0f} players per slate on average</li>
    </ul>
</div>

<div class="page-break"></div>
""")

    def _write_configuration(self, f, config: Dict[str, Any]):
        """Write configuration section."""
        f.write(f"""
<div class="page">
    <h1 id="configuration">Configuration</h1>

    <h3>Training Period</h3>
    <p><strong>Start:</strong> {config.get('train_start', 'N/A')}<br>
    <strong>End:</strong> {config.get('train_end', 'N/A')}<br>
    <strong>Seasons:</strong> {config.get('num_seasons', 'N/A')}</p>

    <h3>Testing Period</h3>
    <p><strong>Start:</strong> {config.get('test_start', 'N/A')}<br>
    <strong>End:</strong> {config.get('test_end', 'N/A')}</p>

    <h3>Model Configuration</h3>
    <table>
        <tr><th>Parameter</th><th>Value</th></tr>
        <tr><td>Model Type</td><td>{config.get('model_type', 'N/A')}</td></tr>
        <tr><td>Feature Config</td><td>{config.get('feature_config', 'N/A')}</td></tr>
        <tr><td>Per-Player Models</td><td>{'Yes' if config.get('per_player_models', False) else 'No'}</td></tr>
        <tr><td>Recalibration Frequency</td><td>{config.get('recalibrate_days', 'N/A')} days</td></tr>
        <tr><td>Parallel Jobs</td><td>{config.get('n_jobs', 1)}</td></tr>
    </table>
</div>

<div class="page-break"></div>
""")

    def _write_performance_metrics_section(self, f, results: Dict[str, Any]):
        """Write performance metrics section."""
        f.write(f"""
<div class="page">
    <h1 id="performance-metrics">Performance Metrics</h1>

    <h3>Model Performance</h3>
    <table>
        <tr><th>Metric</th><th>Mean</th><th>Median</th><th>Std Dev</th></tr>
        <tr><td>MAPE (%)</td><td>{results.get('model_mean_mape', 0):.2f}</td><td>{results.get('model_median_mape', 0):.2f}</td><td>{results.get('model_std_mape', 0):.2f}</td></tr>
        <tr><td>RMSE</td><td>{results.get('model_mean_rmse', 0):.2f}</td><td>-</td><td>{results.get('model_std_rmse', 0):.2f}</td></tr>
        <tr><td>MAE</td><td>{results.get('model_mean_mae', 0):.2f}</td><td>-</td><td>-</td></tr>
        <tr><td>Correlation</td><td>{results.get('model_mean_correlation', 0):.3f}</td><td>-</td><td>{results.get('model_std_correlation', 0):.3f}</td></tr>
    </table>

    <h3>Benchmark Performance</h3>
    <table>
        <tr><th>Metric</th><th>Mean</th><th>Median</th></tr>
        <tr><td>MAPE (%)</td><td>{results.get('benchmark_mean_mape', 0):.2f}</td><td>{results.get('benchmark_median_mape', 0):.2f}</td></tr>
    </table>
</div>

<div class="page-break"></div>
""")

    def _write_chart_with_analysis(self, f, title: str, chart_path: Path, analysis: str):
        """Write chart with analysis section."""
        if not chart_path or not chart_path.exists():
            return

        section_id = title.lower().replace(' ', '-')
        relative_path = Path('charts') / chart_path.name

        f.write(f"""
<div class="page">
    <h1 id="{section_id}">{title}</h1>

    <div class="chart-container">
        <iframe src="{relative_path}"></iframe>
    </div>

    <div class="analysis-box">
        <h3>Analysis & Conclusions</h3>
        <p>{analysis}</p>
    </div>
</div>

<div class="page-break"></div>
""")

    def _write_detailed_metrics(self, f, results: Dict[str, Any]):
        """Write detailed metrics section."""
        if 'daily_results' not in results:
            return

        daily_df = results['daily_results']

        f.write("""
<div class="page">
    <h1 id="detailed-metrics">Detailed Daily Metrics</h1>

    <table>
        <tr>
            <th>Date</th>
            <th>MAPE (%)</th>
            <th>RMSE</th>
            <th>MAE</th>
            <th>Correlation</th>
            <th>Players</th>
        </tr>
""")

        for _, row in daily_df.iterrows():
            f.write(f"""
        <tr>
            <td>{row['date']}</td>
            <td>{row['model_mape']:.2f}</td>
            <td>{row['model_rmse']:.2f}</td>
            <td>{row['model_mae']:.2f}</td>
            <td>{row['model_corr']:.3f}</td>
            <td>{int(row['num_players'])}</td>
        </tr>
""")

        f.write("""
    </table>
</div>

<div class="page-break"></div>
""")

    def _write_statistical_tests(self, f, results: Dict[str, Any]):
        """Write statistical significance section."""
        if 'statistical_test' not in results:
            return

        test = results['statistical_test']
        is_significant = test['p_value'] < 0.05

        f.write(f"""
<div class="page">
    <h1 id="statistical-tests">Statistical Significance Testing</h1>

    <h3>Paired t-Test Results</h3>
    <table>
        <tr><th>Statistic</th><th>Value</th></tr>
        <tr><td>t-statistic</td><td>{test['t_statistic']:.4f}</td></tr>
        <tr><td>p-value</td><td>{test['p_value']:.6f}</td></tr>
        <tr><td>Cohen's d</td><td>{test['cohens_d']:.4f}</td></tr>
        <tr><td>Effect Size</td><td>{test['effect_size']}</td></tr>
    </table>

    <div class="{'summary-box' if is_significant else 'warning-box'}">
        <h3>Conclusion</h3>
        <p>{'✓ The difference is statistically significant (p < 0.05). Results are reliable and not due to random chance.' if is_significant else '⚠ The difference is not statistically significant (p ≥ 0.05). Results may be due to random variation.'}</p>
    </div>
</div>

<div class="page-break"></div>
""")

    def _write_recommendations(self, f, results: Dict[str, Any]):
        """Write recommendations section."""
        model_mape = results.get('model_mean_mape', 0)
        correlation = results.get('model_mean_correlation', 0)
        improvement = results.get('mape_improvement', 0)

        f.write("""
<div class="page">
    <h1 id="recommendations">Recommendations</h1>

    <h3>Priority Actions</h3>
""")

        # Generate recommendations based on results
        if model_mape > 50:
            f.write("""
    <div class="recommendation-box">
        <h4>High Priority: Improve Base Model Performance</h4>
        <ul>
            <li>Add contextual features: opponent defense ratings, pace factors, rest days</li>
            <li>Include injury impact indicators and player usage trends</li>
            <li>Consider position-specific models given variance across positions</li>
            <li>Implement feature selection to reduce noise</li>
        </ul>
    </div>
""")

        if correlation < 0.7:
            f.write("""
    <div class="recommendation-box">
        <h4>High Priority: Feature Engineering</h4>
        <ul>
            <li>Correlation below target (0.7) indicates missing predictive features</li>
            <li>Add recent form indicators (last 3-5 game performance)</li>
            <li>Include teammate injury impacts and rotation changes</li>
            <li>Consider interaction features between usage and minutes</li>
        </ul>
    </div>
""")

        if improvement < 0:
            f.write("""
    <div class="warning-box">
        <h4>Critical: Model Underperforms Baseline</h4>
        <ul>
            <li>Current model worse than simple season averages</li>
            <li>Review model architecture and hyperparameters</li>
            <li>Check for data leakage or feature contamination</li>
            <li>Consider ensemble methods or different model families</li>
        </ul>
    </div>
""")

        f.write("""
    <h3>Optimization Strategies</h3>
    <div class="recommendation-box">
        <h4>Data Quality</h4>
        <ul>
            <li>Filter low-minute players (< 12 mins) to reduce noise</li>
            <li>Implement outlier detection for extreme performances</li>
            <li>Validate data freshness and API reliability</li>
        </ul>
    </div>

    <div class="recommendation-box">
        <h4>Model Refinement</h4>
        <ul>
            <li>Hyperparameter tuning with Bayesian optimization</li>
            <li>Cross-validation across multiple seasons</li>
            <li>Ensemble predictions from multiple model types</li>
            <li>Calibration techniques for systematic bias correction</li>
        </ul>
    </div>

    <div class="recommendation-box">
        <h4>Operational Excellence</h4>
        <ul>
            <li>Automated daily model retraining pipeline</li>
            <li>Real-time data updates 30 minutes before lineup lock</li>
            <li>A/B testing framework for model variants</li>
            <li>Monitoring dashboards for prediction drift</li>
        </ul>
    </div>
</div>
""")

    def _write_report_footer(self, f, run_timestamp: str):
        """Write report footer."""
        f.write(f"""
<div class="page">
    <div class="footer">
        <h3>Report Information</h3>
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>Run ID:</strong> {run_timestamp}</p>
        <p><strong>Generator:</strong> NBA DFS Backtest Analysis System v2.0</p>
        <p><strong>Model:</strong> XGBoost Per-Player Fantasy Point Predictor</p>
        <br>
        <p style="font-size: 8pt; color: #6e7681;">
            This report contains proprietary predictive analytics. All metrics, charts, and recommendations
            are generated through automated analysis of historical NBA performance data.
        </p>
    </div>
</div>
""")

    def _write_pdf_html_footer(self, f):
        """Write HTML footer."""
        f.write("""
</body>
</html>
""")