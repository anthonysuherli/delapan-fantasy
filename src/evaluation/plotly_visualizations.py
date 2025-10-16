import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)


class PlotlyBacktestVisualizer:
    """
    Generate interactive Plotly visualizations for backtest results.
    """

    def __init__(self, output_dir: Path):
        """
        Initialize Plotly visualizer.

        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = Path(output_dir)
        self.charts_dir = self.output_dir / 'charts'
        self.charts_dir.mkdir(parents=True, exist_ok=True)

        self.color_palette = {
            'model': '#2E86AB',
            'benchmark': '#A23B72',
            'positive': '#06A77D',
            'negative': '#D62828',
            'neutral': '#F77F00'
        }

    def generate_all_charts(self, results: Dict[str, Any]) -> Dict[str, Path]:
        """
        Generate all interactive visualization charts.

        Args:
            results: Backtest results dictionary

        Returns:
            Dictionary mapping chart names to file paths
        """
        chart_paths = {}

        try:
            chart_paths['daily_performance'] = self._plot_daily_performance(results)
            chart_paths['model_vs_benchmark'] = self._plot_model_vs_benchmark_dashboard(results)
            chart_paths['error_analysis'] = self._plot_error_analysis_dashboard(results)
            chart_paths['salary_tier'] = self._plot_salary_tier_analysis(results)
            chart_paths['correlation_analysis'] = self._plot_correlation_analysis(results)
            chart_paths['position_analysis'] = self._plot_position_performance(results)
            chart_paths['minutes_analysis'] = self._plot_minutes_analysis(results)
            chart_paths['team_analysis'] = self._plot_team_performance(results)
            chart_paths['residual_analysis'] = self._plot_residual_analysis(results)
            chart_paths['statistical_tests'] = self._plot_statistical_tests(results)
            chart_paths['calibration_curve'] = self._plot_calibration_curve(results)
            chart_paths['error_heatmap'] = self._plot_error_heatmap(results)
            chart_paths['comprehensive_dashboard'] = self._plot_comprehensive_dashboard(results)

            logger.info(f"Generated {len(chart_paths)} interactive charts in {self.charts_dir}")

        except Exception as e:
            logger.error(f"Error generating Plotly charts: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())

        return chart_paths

    def _plot_daily_performance(self, results: Dict[str, Any]) -> Optional[Path]:
        """Plot daily performance metrics over time."""
        if 'daily_results' not in results:
            return None

        daily_df = results['daily_results'].copy()

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('MAPE Over Time', 'RMSE Over Time',
                          'MAE Over Time', 'Correlation Over Time'),
            specs=[[{'secondary_y': False}, {'secondary_y': False}],
                   [{'secondary_y': False}, {'secondary_y': False}]]
        )

        fig.add_trace(
            go.Scatter(x=daily_df['date'], y=daily_df['model_mape'],
                      mode='lines+markers', name='Model MAPE',
                      line=dict(color=self.color_palette['model'], width=2),
                      marker=dict(size=6)),
            row=1, col=1
        )

        if 'benchmark_mape' in daily_df.columns:
            fig.add_trace(
                go.Scatter(x=daily_df['date'], y=daily_df['benchmark_mape'],
                          mode='lines+markers', name='Benchmark MAPE',
                          line=dict(color=self.color_palette['benchmark'], width=2, dash='dash'),
                          marker=dict(size=6)),
                row=1, col=1
            )

        fig.add_trace(
            go.Scatter(x=daily_df['date'], y=daily_df['model_rmse'],
                      mode='lines+markers', name='RMSE',
                      line=dict(color=self.color_palette['neutral'], width=2),
                      marker=dict(size=6)),
            row=1, col=2
        )

        fig.add_trace(
            go.Scatter(x=daily_df['date'], y=daily_df['model_mae'],
                      mode='lines+markers', name='MAE',
                      line=dict(color=self.color_palette['positive'], width=2),
                      marker=dict(size=6)),
            row=2, col=1
        )

        fig.add_trace(
            go.Scatter(x=daily_df['date'], y=daily_df['model_corr'],
                      mode='lines+markers', name='Correlation',
                      line=dict(color=self.color_palette['model'], width=2),
                      marker=dict(size=6)),
            row=2, col=2
        )

        fig.add_hline(y=0.7, line_dash="dash", line_color="green",
                     annotation_text="Target (0.7)", row=2, col=2)

        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_xaxes(title_text="Date", row=2, col=2)
        fig.update_yaxes(title_text="MAPE (%)", row=1, col=1)
        fig.update_yaxes(title_text="RMSE", row=1, col=2)
        fig.update_yaxes(title_text="MAE", row=2, col=1)
        fig.update_yaxes(title_text="Correlation", row=2, col=2)

        fig.update_layout(
            height=800,
            title_text="Daily Performance Metrics Timeline",
            showlegend=True,
            hovermode='x unified'
        )

        output_path = self.charts_dir / 'daily_performance.html'
        fig.write_html(output_path)

        return output_path

    def _plot_model_vs_benchmark_dashboard(self, results: Dict[str, Any]) -> Optional[Path]:
        """Comprehensive model vs benchmark comparison dashboard."""
        if 'daily_results' not in results:
            return None

        daily_df = results['daily_results']

        if 'benchmark_mape' not in daily_df.columns:
            return None

        valid_data = daily_df[daily_df['benchmark_mape'].notna()].copy()

        if valid_data.empty:
            return None

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('MAPE Scatter: Model vs Benchmark',
                          'Daily MAPE Improvement',
                          'MAPE Distribution Comparison',
                          'Cumulative Improvement'),
            specs=[[{'type': 'scatter'}, {'type': 'bar'}],
                   [{'type': 'box'}, {'type': 'scatter'}]]
        )

        fig.add_trace(
            go.Scatter(
                x=valid_data['benchmark_mape'],
                y=valid_data['model_mape'],
                mode='markers',
                name='Slates',
                marker=dict(size=10, color=self.color_palette['model'], opacity=0.6),
                text=valid_data['date'],
                hovertemplate='<b>Date: %{text}</b><br>Benchmark: %{x:.2f}%<br>Model: %{y:.2f}%<extra></extra>'
            ),
            row=1, col=1
        )

        max_mape = max(valid_data['benchmark_mape'].max(), valid_data['model_mape'].max())
        fig.add_trace(
            go.Scatter(
                x=[0, max_mape],
                y=[0, max_mape],
                mode='lines',
                name='Perfect Match',
                line=dict(color='red', dash='dash', width=2)
            ),
            row=1, col=1
        )

        improvement = valid_data['benchmark_mape'] - valid_data['model_mape']
        colors = [self.color_palette['positive'] if x > 0 else self.color_palette['negative']
                 for x in improvement]

        fig.add_trace(
            go.Bar(
                x=valid_data['date'],
                y=improvement,
                name='Improvement',
                marker=dict(color=colors),
                hovertemplate='<b>Date: %{x}</b><br>Improvement: %{y:+.2f}%<extra></extra>'
            ),
            row=1, col=2
        )

        fig.add_trace(
            go.Box(y=valid_data['model_mape'], name='Model',
                  marker=dict(color=self.color_palette['model']),
                  boxmean='sd'),
            row=2, col=1
        )

        fig.add_trace(
            go.Box(y=valid_data['benchmark_mape'], name='Benchmark',
                  marker=dict(color=self.color_palette['benchmark']),
                  boxmean='sd'),
            row=2, col=1
        )

        cumulative_improvement = improvement.cumsum()
        fig.add_trace(
            go.Scatter(
                x=valid_data['date'],
                y=cumulative_improvement,
                mode='lines+markers',
                name='Cumulative Improvement',
                line=dict(color=self.color_palette['neutral'], width=3),
                fill='tozeroy',
                fillcolor='rgba(247, 127, 0, 0.2)'
            ),
            row=2, col=2
        )

        fig.update_xaxes(title_text="Benchmark MAPE (%)", row=1, col=1)
        fig.update_xaxes(title_text="Date", row=1, col=2)
        fig.update_xaxes(title_text="Date", row=2, col=2)
        fig.update_yaxes(title_text="Model MAPE (%)", row=1, col=1)
        fig.update_yaxes(title_text="Improvement (%)", row=1, col=2)
        fig.update_yaxes(title_text="MAPE (%)", row=2, col=1)
        fig.update_yaxes(title_text="Cumulative (%)", row=2, col=2)

        fig.update_layout(
            height=900,
            title_text="Model vs Benchmark Comprehensive Analysis",
            showlegend=True,
            hovermode='closest'
        )

        output_path = self.charts_dir / 'model_vs_benchmark_dashboard.html'
        fig.write_html(output_path)

        return output_path

    def _plot_error_analysis_dashboard(self, results: Dict[str, Any]) -> Optional[Path]:
        """Comprehensive error analysis dashboard."""
        if 'all_predictions' not in results or results['all_predictions'].empty:
            return None

        all_preds = results['all_predictions'].copy()

        if 'actual_fpts' not in all_preds.columns or 'projected_fpts' not in all_preds.columns:
            return None

        all_preds['error'] = all_preds['projected_fpts'] - all_preds['actual_fpts']
        all_preds['abs_error'] = np.abs(all_preds['error'])
        all_preds['pct_error'] = (all_preds['error'] / all_preds['actual_fpts'].replace(0, np.nan)) * 100

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Error Distribution', 'Absolute Error Distribution',
                          'Error by Actual FPTS', 'Prediction Bias Analysis'),
            specs=[[{'type': 'histogram'}, {'type': 'histogram'}],
                   [{'type': 'scatter'}, {'type': 'pie'}]]
        )

        fig.add_trace(
            go.Histogram(
                x=all_preds['error'],
                nbinsx=50,
                name='Error',
                marker=dict(color=self.color_palette['model'], opacity=0.7),
                hovertemplate='Error: %{x:.2f}<br>Count: %{y}<extra></extra>'
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Histogram(
                x=all_preds['abs_error'],
                nbinsx=50,
                name='Absolute Error',
                marker=dict(color=self.color_palette['benchmark'], opacity=0.7),
                hovertemplate='Abs Error: %{x:.2f}<br>Count: %{y}<extra></extra>'
            ),
            row=1, col=2
        )

        fig.add_trace(
            go.Scatter(
                x=all_preds['actual_fpts'],
                y=all_preds['abs_error'],
                mode='markers',
                name='Abs Error vs Actual',
                marker=dict(size=4, color=self.color_palette['neutral'], opacity=0.4),
                hovertemplate='Actual: %{x:.2f}<br>Abs Error: %{y:.2f}<extra></extra>'
            ),
            row=2, col=1
        )

        overest = (all_preds['error'] > 0).sum()
        underest = (all_preds['error'] < 0).sum()

        fig.add_trace(
            go.Pie(
                labels=['Overestimated', 'Underestimated'],
                values=[overest, underest],
                marker=dict(colors=[self.color_palette['negative'], self.color_palette['model']]),
                hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
            ),
            row=2, col=2
        )

        fig.update_xaxes(title_text="Prediction Error (fpts)", row=1, col=1)
        fig.update_xaxes(title_text="Absolute Error (fpts)", row=1, col=2)
        fig.update_xaxes(title_text="Actual FPTS", row=2, col=1)
        fig.update_yaxes(title_text="Frequency", row=1, col=1)
        fig.update_yaxes(title_text="Frequency", row=1, col=2)
        fig.update_yaxes(title_text="Absolute Error (fpts)", row=2, col=1)

        fig.update_layout(
            height=900,
            title_text="Error Analysis Dashboard",
            showlegend=True,
            hovermode='closest'
        )

        output_path = self.charts_dir / 'error_analysis_dashboard.html'
        fig.write_html(output_path)

        return output_path

    def _plot_salary_tier_analysis(self, results: Dict[str, Any]) -> Optional[Path]:
        """Interactive salary tier performance analysis."""
        if 'tier_comparison' not in results:
            return None

        tier_df = results['tier_comparison']

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('MAPE by Salary Tier', 'Improvement by Tier'),
            specs=[[{'type': 'bar'}, {'type': 'bar'}]]
        )

        fig.add_trace(
            go.Bar(
                x=tier_df['salary_tier'],
                y=tier_df['model_mape'],
                name='Model MAPE',
                marker=dict(color=self.color_palette['model']),
                hovertemplate='<b>%{x}</b><br>Model MAPE: %{y:.2f}%<extra></extra>'
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Bar(
                x=tier_df['salary_tier'],
                y=tier_df['benchmark_mape'],
                name='Benchmark MAPE',
                marker=dict(color=self.color_palette['benchmark']),
                hovertemplate='<b>%{x}</b><br>Benchmark MAPE: %{y:.2f}%<extra></extra>'
            ),
            row=1, col=1
        )

        colors = [self.color_palette['positive'] if imp > 0 else self.color_palette['negative']
                 for imp in tier_df['mape_improvement']]

        fig.add_trace(
            go.Bar(
                x=tier_df['salary_tier'],
                y=tier_df['mape_improvement'],
                name='Improvement',
                marker=dict(color=colors),
                text=[f"{imp:+.1f}%" for imp in tier_df['mape_improvement']],
                textposition='outside',
                hovertemplate='<b>%{x}</b><br>Improvement: %{y:+.2f}%<extra></extra>'
            ),
            row=1, col=2
        )

        fig.update_xaxes(title_text="Salary Tier", row=1, col=1)
        fig.update_xaxes(title_text="Salary Tier", row=1, col=2)
        fig.update_yaxes(title_text="MAPE (%)", row=1, col=1)
        fig.update_yaxes(title_text="Improvement (%)", row=1, col=2)

        fig.update_layout(
            height=500,
            title_text="Performance by Salary Tier",
            showlegend=True,
            hovermode='x unified',
            barmode='group'
        )

        output_path = self.charts_dir / 'salary_tier_analysis.html'
        fig.write_html(output_path)

        return output_path

    def _plot_correlation_analysis(self, results: Dict[str, Any]) -> Optional[Path]:
        """Interactive correlation scatter plot with regression."""
        if 'all_predictions' not in results or results['all_predictions'].empty:
            return None

        all_preds = results['all_predictions']

        if 'actual_fpts' not in all_preds.columns or 'projected_fpts' not in all_preds.columns:
            return None

        slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(
            all_preds['actual_fpts'], all_preds['projected_fpts']
        )

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=all_preds['actual_fpts'],
                y=all_preds['projected_fpts'],
                mode='markers',
                name='Predictions',
                marker=dict(size=5, color=self.color_palette['model'], opacity=0.3),
                hovertemplate='<b>Actual: %{x:.2f}</b><br>Predicted: %{y:.2f}<extra></extra>'
            )
        )

        max_val = max(all_preds['actual_fpts'].max(), all_preds['projected_fpts'].max())

        fig.add_trace(
            go.Scatter(
                x=[0, max_val],
                y=[0, max_val],
                mode='lines',
                name='Perfect Prediction',
                line=dict(color='red', dash='dash', width=2)
            )
        )

        x_line = np.array([all_preds['actual_fpts'].min(), all_preds['actual_fpts'].max()])
        y_line = slope * x_line + intercept

        fig.add_trace(
            go.Scatter(
                x=x_line,
                y=y_line,
                mode='lines',
                name=f'Regression (r={r_value:.3f})',
                line=dict(color='green', width=2)
            )
        )

        fig.update_layout(
            title=f'Actual vs Predicted Fantasy Points<br><sub>r={r_value:.3f}, p={p_value:.2e}</sub>',
            xaxis_title='Actual Fantasy Points',
            yaxis_title='Predicted Fantasy Points',
            height=700,
            hovermode='closest'
        )

        output_path = self.charts_dir / 'correlation_analysis.html'
        fig.write_html(output_path)

        return output_path

    def _plot_position_performance(self, results: Dict[str, Any]) -> Optional[Path]:
        """Performance breakdown by player position."""
        if 'all_predictions' not in results or results['all_predictions'].empty:
            return None

        all_preds = results['all_predictions']

        if 'position' not in all_preds.columns:
            return None

        all_preds_copy = all_preds.copy()
        all_preds_copy['abs_error'] = np.abs(all_preds_copy['projected_fpts'] - all_preds_copy['actual_fpts'])
        all_preds_copy['pct_error'] = (np.abs(all_preds_copy['projected_fpts'] - all_preds_copy['actual_fpts']) /
                                       all_preds_copy['actual_fpts'].replace(0, np.nan)) * 100

        position_stats = all_preds_copy.groupby('position').agg({
            'abs_error': ['mean', 'median', 'std'],
            'pct_error': 'mean',
            'playerID': 'count'
        }).reset_index()

        position_stats.columns = ['position', 'mae', 'median_ae', 'std_ae', 'mape', 'count']

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('MAE by Position', 'MAPE by Position')
        )

        fig.add_trace(
            go.Bar(
                x=position_stats['position'],
                y=position_stats['mae'],
                name='MAE',
                marker=dict(color=self.color_palette['model']),
                text=[f"{mae:.2f}" for mae in position_stats['mae']],
                textposition='outside',
                error_y=dict(type='data', array=position_stats['std_ae']),
                hovertemplate='<b>%{x}</b><br>MAE: %{y:.2f}<br>Count: ' +
                             position_stats['count'].astype(str) + '<extra></extra>'
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Bar(
                x=position_stats['position'],
                y=position_stats['mape'],
                name='MAPE',
                marker=dict(color=self.color_palette['neutral']),
                text=[f"{mape:.1f}%" for mape in position_stats['mape']],
                textposition='outside',
                hovertemplate='<b>%{x}</b><br>MAPE: %{y:.2f}%<br>Count: ' +
                             position_stats['count'].astype(str) + '<extra></extra>'
            ),
            row=1, col=2
        )

        fig.update_xaxes(title_text="Position", row=1, col=1)
        fig.update_xaxes(title_text="Position", row=1, col=2)
        fig.update_yaxes(title_text="MAE (fpts)", row=1, col=1)
        fig.update_yaxes(title_text="MAPE (%)", row=1, col=2)

        fig.update_layout(
            height=500,
            title_text="Model Performance by Position",
            showlegend=True,
            hovermode='x unified'
        )

        output_path = self.charts_dir / 'position_analysis.html'
        fig.write_html(output_path)

        return output_path

    def _plot_minutes_analysis(self, results: Dict[str, Any]) -> Optional[Path]:
        """Performance analysis by minutes played."""
        if 'all_predictions' not in results or results['all_predictions'].empty:
            return None

        all_preds = results['all_predictions']

        if 'actual_mins' not in all_preds.columns:
            return None

        all_preds_copy = all_preds.copy()
        all_preds_copy['abs_error'] = np.abs(all_preds_copy['projected_fpts'] - all_preds_copy['actual_fpts'])

        all_preds_copy['minutes_bin'] = pd.cut(
            all_preds_copy['actual_mins'],
            bins=[0, 10, 20, 30, 40, 50],
            labels=['0-10', '10-20', '20-30', '30-40', '40+']
        )

        minutes_stats = all_preds_copy.groupby('minutes_bin', observed=True).agg({
            'abs_error': ['mean', 'median'],
            'playerID': 'count'
        }).reset_index()

        minutes_stats.columns = ['minutes_bin', 'mae', 'median_ae', 'count']

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=minutes_stats['minutes_bin'].astype(str),
                y=minutes_stats['mae'],
                name='MAE',
                marker=dict(color=self.color_palette['positive']),
                text=[f"{mae:.2f}" for mae in minutes_stats['mae']],
                textposition='outside',
                hovertemplate='<b>%{x} mins</b><br>MAE: %{y:.2f}<br>Count: ' +
                             minutes_stats['count'].astype(str) + '<extra></extra>'
            )
        )

        fig.update_layout(
            title='Model Performance by Minutes Played',
            xaxis_title='Minutes Bin',
            yaxis_title='MAE (fpts)',
            height=500,
            hovermode='x unified'
        )

        output_path = self.charts_dir / 'minutes_analysis.html'
        fig.write_html(output_path)

        return output_path

    def _plot_team_performance(self, results: Dict[str, Any]) -> Optional[Path]:
        """Performance breakdown by team."""
        if 'all_predictions' not in results or results['all_predictions'].empty:
            return None

        all_preds = results['all_predictions']

        if 'team' not in all_preds.columns:
            return None

        all_preds_copy = all_preds.copy()
        all_preds_copy['abs_error'] = np.abs(all_preds_copy['projected_fpts'] - all_preds_copy['actual_fpts'])
        all_preds_copy['pct_error'] = (np.abs(all_preds_copy['projected_fpts'] - all_preds_copy['actual_fpts']) /
                                       all_preds_copy['actual_fpts'].replace(0, np.nan)) * 100

        team_stats = all_preds_copy.groupby('team').agg({
            'abs_error': 'mean',
            'pct_error': 'mean',
            'playerID': 'count'
        }).reset_index()

        team_stats.columns = ['team', 'mae', 'mape', 'count']
        team_stats = team_stats.sort_values('mape')

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=team_stats['team'],
                y=team_stats['mape'],
                name='MAPE',
                marker=dict(color=self.color_palette['model']),
                text=[f"{mape:.1f}%" for mape in team_stats['mape']],
                textposition='outside',
                hovertemplate='<b>%{x}</b><br>MAPE: %{y:.2f}%<br>Count: ' +
                             team_stats['count'].astype(str) + '<extra></extra>'
            )
        )

        fig.update_layout(
            title='Model Performance by Team',
            xaxis_title='Team',
            yaxis_title='MAPE (%)',
            height=600,
            hovermode='x unified'
        )

        output_path = self.charts_dir / 'team_analysis.html'
        fig.write_html(output_path)

        return output_path

    def _plot_residual_analysis(self, results: Dict[str, Any]) -> Optional[Path]:
        """Residual analysis for model diagnostics."""
        if 'all_predictions' not in results or results['all_predictions'].empty:
            return None

        all_preds = results['all_predictions']

        if 'actual_fpts' not in all_preds.columns or 'projected_fpts' not in all_preds.columns:
            return None

        all_preds_copy = all_preds.copy()
        all_preds_copy['residual'] = all_preds_copy['actual_fpts'] - all_preds_copy['projected_fpts']
        all_preds_copy['fitted'] = all_preds_copy['projected_fpts']

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Residuals vs Fitted', 'Residual Distribution')
        )

        fig.add_trace(
            go.Scatter(
                x=all_preds_copy['fitted'],
                y=all_preds_copy['residual'],
                mode='markers',
                name='Residuals',
                marker=dict(size=4, color=self.color_palette['model'], opacity=0.3),
                hovertemplate='Fitted: %{x:.2f}<br>Residual: %{y:.2f}<extra></extra>'
            ),
            row=1, col=1
        )

        fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)

        fig.add_trace(
            go.Histogram(
                x=all_preds_copy['residual'],
                nbinsx=50,
                name='Residuals',
                marker=dict(color=self.color_palette['neutral'], opacity=0.7),
                hovertemplate='Residual: %{x:.2f}<br>Count: %{y}<extra></extra>'
            ),
            row=1, col=2
        )

        fig.update_xaxes(title_text="Fitted Values", row=1, col=1)
        fig.update_xaxes(title_text="Residuals", row=1, col=2)
        fig.update_yaxes(title_text="Residuals", row=1, col=1)
        fig.update_yaxes(title_text="Frequency", row=1, col=2)

        fig.update_layout(
            height=500,
            title_text="Residual Analysis",
            showlegend=True,
            hovermode='closest'
        )

        output_path = self.charts_dir / 'residual_analysis.html'
        fig.write_html(output_path)

        return output_path

    def _plot_statistical_tests(self, results: Dict[str, Any]) -> Optional[Path]:
        """Visualize statistical test results."""
        if 'statistical_test' not in results:
            return None

        test = results['statistical_test']

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Statistical Significance', 'Effect Size'),
            specs=[[{'type': 'indicator'}, {'type': 'indicator'}]]
        )

        p_value = test['p_value']
        is_significant = p_value < 0.05

        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=p_value,
                title={'text': f"p-value ({'Significant' if is_significant else 'Not Significant'})"},
                delta={'reference': 0.05},
                gauge={
                    'axis': {'range': [0, 0.1]},
                    'bar': {'color': self.color_palette['positive'] if is_significant else self.color_palette['negative']},
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 0.05
                    }
                }
            ),
            row=1, col=1
        )

        cohens_d = abs(test['cohens_d'])

        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=cohens_d,
                title={'text': f"Cohen's d ({test['effect_size']})"},
                gauge={
                    'axis': {'range': [0, 1.0]},
                    'bar': {'color': self.color_palette['model']},
                    'steps': [
                        {'range': [0, 0.2], 'color': "lightgray"},
                        {'range': [0.2, 0.5], 'color': "gray"},
                        {'range': [0.5, 0.8], 'color': "darkgray"},
                        {'range': [0.8, 1.0], 'color': "black"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 0.5
                    }
                }
            ),
            row=1, col=2
        )

        fig.update_layout(
            height=400,
            title_text=f"Statistical Test Results (t={test['t_statistic']:.4f})"
        )

        output_path = self.charts_dir / 'statistical_tests.html'
        fig.write_html(output_path)

        return output_path

    def _plot_calibration_curve(self, results: Dict[str, Any]) -> Optional[Path]:
        """Plot calibration curve for predictions."""
        if 'all_predictions' not in results or results['all_predictions'].empty:
            return None

        all_preds = results['all_predictions']

        if 'actual_fpts' not in all_preds.columns or 'projected_fpts' not in all_preds.columns:
            return None

        all_preds_sorted = all_preds.sort_values('projected_fpts')
        n_bins = 20
        bin_size = len(all_preds_sorted) // n_bins

        predicted_means = []
        actual_means = []

        for i in range(n_bins):
            start_idx = i * bin_size
            end_idx = start_idx + bin_size if i < n_bins - 1 else len(all_preds_sorted)

            bin_data = all_preds_sorted.iloc[start_idx:end_idx]
            predicted_means.append(bin_data['projected_fpts'].mean())
            actual_means.append(bin_data['actual_fpts'].mean())

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=predicted_means,
                y=actual_means,
                mode='markers+lines',
                name='Calibration Curve',
                marker=dict(size=10, color=self.color_palette['model']),
                line=dict(width=2),
                hovertemplate='Predicted: %{x:.2f}<br>Actual: %{y:.2f}<extra></extra>'
            )
        )

        max_val = max(max(predicted_means), max(actual_means))
        fig.add_trace(
            go.Scatter(
                x=[0, max_val],
                y=[0, max_val],
                mode='lines',
                name='Perfect Calibration',
                line=dict(color='red', dash='dash', width=2)
            )
        )

        fig.update_layout(
            title='Calibration Curve: Predicted vs Actual FPTS',
            xaxis_title='Mean Predicted FPTS (binned)',
            yaxis_title='Mean Actual FPTS',
            height=600,
            hovermode='closest'
        )

        output_path = self.charts_dir / 'calibration_curve.html'
        fig.write_html(output_path)

        return output_path

    def _plot_error_heatmap(self, results: Dict[str, Any]) -> Optional[Path]:
        """Error heatmap by salary tier and position."""
        if 'all_predictions' not in results or results['all_predictions'].empty:
            return None

        all_preds = results['all_predictions']

        if 'position' not in all_preds.columns or 'salary' not in all_preds.columns:
            return None

        all_preds_copy = all_preds.copy()
        all_preds_copy['abs_error'] = np.abs(all_preds_copy['projected_fpts'] - all_preds_copy['actual_fpts'])

        all_preds_copy['salary_tier'] = pd.cut(
            all_preds_copy['salary'],
            bins=[0, 4000, 6000, 8000, 12000],
            labels=['Low', 'Mid', 'High', 'Elite']
        )

        heatmap_data = all_preds_copy.groupby(['salary_tier', 'position'], observed=True)['abs_error'].mean().unstack(fill_value=np.nan)

        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=heatmap_data.columns,
            y=heatmap_data.index.astype(str),
            colorscale='RdYlGn_r',
            hovertemplate='Position: %{x}<br>Tier: %{y}<br>MAE: %{z:.2f}<extra></extra>',
            colorbar=dict(title="MAE (fpts)")
        ))

        fig.update_layout(
            title='Mean Absolute Error Heatmap (Salary Tier x Position)',
            xaxis_title='Position',
            yaxis_title='Salary Tier',
            height=500
        )

        output_path = self.charts_dir / 'error_heatmap.html'
        fig.write_html(output_path)

        return output_path

    def _plot_comprehensive_dashboard(self, results: Dict[str, Any]) -> Optional[Path]:
        """Create comprehensive single-page dashboard."""
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=(
                'Overall MAPE', 'Model Correlation', 'Daily MAPE Trend',
                'Salary Tier Performance', 'Position Performance', 'Error Distribution',
                'Prediction Bias', 'Statistical Significance', 'Calibration Quality'
            ),
            specs=[
                [{'type': 'indicator'}, {'type': 'indicator'}, {'type': 'scatter'}],
                [{'type': 'bar'}, {'type': 'bar'}, {'type': 'histogram'}],
                [{'type': 'pie'}, {'type': 'indicator'}, {'type': 'scatter'}]
            ]
        )

        model_mape = results.get('model_mean_mape', 0)
        benchmark_mape = results.get('benchmark_mean_mape', 0)

        fig.add_trace(
            go.Indicator(
                mode="number+delta",
                value=model_mape,
                title={'text': "Model MAPE (%)"},
                delta={'reference': benchmark_mape, 'relative': False, 'suffix': '%'},
                number={'suffix': '%'}
            ),
            row=1, col=1
        )

        model_corr = results.get('model_mean_correlation', 0)

        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=model_corr,
                title={'text': "Correlation"},
                gauge={
                    'axis': {'range': [0, 1]},
                    'bar': {'color': self.color_palette['positive']},
                    'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 0.7}
                }
            ),
            row=1, col=2
        )

        if 'daily_results' in results:
            daily_df = results['daily_results']
            fig.add_trace(
                go.Scatter(
                    x=daily_df['date'],
                    y=daily_df['model_mape'],
                    mode='lines+markers',
                    name='MAPE',
                    line=dict(color=self.color_palette['model'], width=2),
                    marker=dict(size=4)
                ),
                row=1, col=3
            )

        if 'tier_comparison' in results:
            tier_df = results['tier_comparison']
            fig.add_trace(
                go.Bar(
                    x=tier_df['salary_tier'],
                    y=tier_df['model_mape'],
                    name='Tier MAPE',
                    marker=dict(color=self.color_palette['model'])
                ),
                row=2, col=1
            )

        if 'all_predictions' in results and not results['all_predictions'].empty:
            all_preds = results['all_predictions']

            if 'position' in all_preds.columns:
                position_stats = all_preds.groupby('position').agg({
                    'projected_fpts': 'count'
                }).reset_index()
                position_stats.columns = ['position', 'count']

                fig.add_trace(
                    go.Bar(
                        x=position_stats['position'],
                        y=position_stats['count'],
                        name='Position Count',
                        marker=dict(color=self.color_palette['neutral'])
                    ),
                    row=2, col=2
                )

            if 'actual_fpts' in all_preds.columns and 'projected_fpts' in all_preds.columns:
                errors = all_preds['projected_fpts'] - all_preds['actual_fpts']

                fig.add_trace(
                    go.Histogram(
                        x=errors,
                        nbinsx=30,
                        name='Errors',
                        marker=dict(color=self.color_palette['model'], opacity=0.7)
                    ),
                    row=2, col=3
                )

                overest = (errors > 0).sum()
                underest = (errors < 0).sum()

                fig.add_trace(
                    go.Pie(
                        labels=['Overestimated', 'Underestimated'],
                        values=[overest, underest],
                        marker=dict(colors=[self.color_palette['negative'], self.color_palette['model']])
                    ),
                    row=3, col=1
                )

        if 'statistical_test' in results:
            p_value = results['statistical_test']['p_value']

            fig.add_trace(
                go.Indicator(
                    mode="number",
                    value=p_value,
                    title={'text': "p-value"},
                    number={'valueformat': '.4f'}
                ),
                row=3, col=2
            )

        fig.update_layout(
            height=1200,
            title_text="Comprehensive Backtest Dashboard",
            showlegend=False
        )

        output_path = self.charts_dir / 'comprehensive_dashboard.html'
        fig.write_html(output_path)

        return output_path
