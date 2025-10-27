"""
Altair-based visualization module for interactive charting.

This module provides a comprehensive set of interactive charts using Vega-Altair,
offering modern web-based visualizations with built-in interactivity features.
"""

import pandas as pd
import numpy as np
import altair as alt
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple
import logging

logger = logging.getLogger(__name__)

# Configure Altair
alt.themes.enable('dark')
alt.data_transformers.enable('json')


class AltairVisualizer:
    """
    Interactive visualization generator using Vega-Altair
    
    Provides modern, interactive charts with built-in features like:
    - Tooltips and hover effects
    - Zooming and panning
    - Brushing and linking
    - Selection and filtering
    """

    def __init__(self, output_dir: Path, theme: str = 'dark'):
        """
        Initialize Altair visualizer.

        Args:
            output_dir: Directory to save visualizations
            theme: Chart theme ('dark', 'default', 'fivethirtyeight', etc.)
        """
        self.output_dir = Path(output_dir)
        self.charts_dir = self.output_dir / 'altair_charts'
        self.charts_dir.mkdir(parents=True, exist_ok=True)
        
        self.theme = theme
        if theme != 'dark':
            alt.themes.enable(theme)
            
        # Color schemes
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72',
            'success': '#06A77D',
            'danger': '#D62828',
            'warning': '#F77F00',
            'info': '#277DA1'
        }

    def create_scatter_plot(
        self,
        data: pd.DataFrame,
        x: str,
        y: str,
        color: Optional[str] = None,
        size: Optional[str] = None,
        title: str = "Scatter Plot",
        width: int = 600,
        height: int = 400
    ) -> alt.Chart:
        """
        Create interactive scatter plot.

        Args:
            data: DataFrame containing the data
            x: X-axis column name
            y: Y-axis column name
            color: Optional color encoding column
            size: Optional size encoding column
            title: Chart title
            width: Chart width
            height: Chart height

        Returns:
            Altair Chart object
        """
        # Base chart
        base = alt.Chart(data).add_selection(
            alt.selection_interval(bind='scales')  # Enable zoom/pan
        ).properties(
            width=width,
            height=height,
            title=alt.TitleParams(text=title, fontSize=16, fontWeight='bold')
        )

        # Build encoding
        encoding = {
            'x': alt.X(x, scale=alt.Scale(nice=True)),
            'y': alt.Y(y, scale=alt.Scale(nice=True)),
            'tooltip': [x, y]
        }

        if color:
            encoding['color'] = alt.Color(
                color,
                scale=alt.Scale(scheme='category10')
            )
            encoding['tooltip'].append(color)

        if size:
            encoding['size'] = alt.Size(
                size,
                scale=alt.Scale(range=[50, 400])
            )
            encoding['tooltip'].append(size)

        # Create scatter plot
        scatter = base.mark_circle(
            opacity=0.7,
            stroke='white',
            strokeWidth=1
        ).encode(**encoding)

        # Add trend line if no color grouping
        if not color:
            line = base.mark_line(
                color='red',
                size=2,
                opacity=0.8
            ).transform_regression(x, y).encode(
                x=x,
                y=y
            )
            return scatter + line

        return scatter

    def create_line_chart(
        self,
        data: pd.DataFrame,
        x: str,
        y: str,
        color: Optional[str] = None,
        title: str = "Line Chart",
        width: int = 700,
        height: int = 400,
        show_points: bool = True
    ) -> alt.Chart:
        """
        Create interactive line chart.

        Args:
            data: DataFrame containing the data
            x: X-axis column name
            y: Y-axis column name
            color: Optional color grouping column
            title: Chart title
            width: Chart width
            height: Chart height
            show_points: Whether to show data points

        Returns:
            Altair Chart object
        """
        # Selection for highlighting
        highlight = alt.selection(type='single', on='mouseover')
        
        base = alt.Chart(data).add_selection(
            highlight
        ).properties(
            width=width,
            height=height,
            title=alt.TitleParams(text=title, fontSize=16, fontWeight='bold')
        )

        # Build encoding
        encoding = {
            'x': alt.X(x, scale=alt.Scale(nice=True)),
            'y': alt.Y(y, scale=alt.Scale(nice=True)),
            'tooltip': [x, y]
        }

        if color:
            encoding['color'] = alt.Color(
                color,
                scale=alt.Scale(scheme='category10')
            )
            encoding['tooltip'].append(color)
            encoding['opacity'] = alt.condition(highlight, alt.value(1.0), alt.value(0.7))

        # Create line chart
        line = base.mark_line(
            size=3,
            point=show_points
        ).encode(**encoding)

        return line

    def create_bar_chart(
        self,
        data: pd.DataFrame,
        x: str,
        y: str,
        color: Optional[str] = None,
        title: str = "Bar Chart",
        width: int = 600,
        height: int = 400,
        horizontal: bool = False
    ) -> alt.Chart:
        """
        Create interactive bar chart.

        Args:
            data: DataFrame containing the data
            x: X-axis column name
            y: Y-axis column name
            color: Optional color encoding column
            title: Chart title
            width: Chart width
            height: Chart height
            horizontal: Whether to create horizontal bars

        Returns:
            Altair Chart object
        """
        # Selection for highlighting
        click = alt.selection_multi()
        
        base = alt.Chart(data).add_selection(
            click
        ).properties(
            width=width,
            height=height,
            title=alt.TitleParams(text=title, fontSize=16, fontWeight='bold')
        )

        # Build encoding
        if horizontal:
            encoding = {
                'y': alt.Y(x, sort='-x'),
                'x': alt.X(y),
                'tooltip': [x, y]
            }
        else:
            encoding = {
                'x': alt.X(x),
                'y': alt.Y(y),
                'tooltip': [x, y]
            }

        if color:
            encoding['color'] = alt.Color(
                color,
                scale=alt.Scale(scheme='viridis')
            )
            encoding['tooltip'].append(color)

        # Add selection opacity
        encoding['opacity'] = alt.condition(click, alt.value(1.0), alt.value(0.8))

        # Create bar chart
        bars = base.mark_bar(
            stroke='white',
            strokeWidth=1
        ).encode(**encoding)

        return bars

    def create_histogram(
        self,
        data: pd.DataFrame,
        column: str,
        bins: int = 30,
        title: str = "Histogram",
        width: int = 600,
        height: int = 400,
        color: str = None
    ) -> alt.Chart:
        """
        Create interactive histogram.

        Args:
            data: DataFrame containing the data
            column: Column to create histogram for
            bins: Number of bins
            title: Chart title
            width: Chart width
            height: Chart height
            color: Bar color

        Returns:
            Altair Chart object
        """
        base = alt.Chart(data).properties(
            width=width,
            height=height,
            title=alt.TitleParams(text=title, fontSize=16, fontWeight='bold')
        )

        # Create histogram
        hist = base.mark_bar(
            color=color or self.colors['primary'],
            opacity=0.8,
            stroke='white',
            strokeWidth=1
        ).encode(
            x=alt.X(
                f'{column}:Q',
                bin=alt.Bin(maxbins=bins),
                title=column
            ),
            y=alt.Y('count():Q', title='Frequency'),
            tooltip=['count():Q']
        )

        # Add mean line
        mean_line = base.mark_rule(
            color='red',
            size=3,
            opacity=0.8
        ).encode(
            x=f'mean({column}):Q'
        )

        return hist + mean_line

    def create_heatmap(
        self,
        data: pd.DataFrame,
        x: str,
        y: str,
        value: str,
        title: str = "Heatmap",
        width: int = 500,
        height: int = 400
    ) -> alt.Chart:
        """
        Create interactive heatmap.

        Args:
            data: DataFrame containing the data
            x: X-axis column name
            y: Y-axis column name
            value: Value column for color encoding
            title: Chart title
            width: Chart width
            height: Chart height

        Returns:
            Altair Chart object
        """
        base = alt.Chart(data).properties(
            width=width,
            height=height,
            title=alt.TitleParams(text=title, fontSize=16, fontWeight='bold')
        )

        heatmap = base.mark_rect().encode(
            x=alt.X(f'{x}:O'),
            y=alt.Y(f'{y}:O'),
            color=alt.Color(
                f'{value}:Q',
                scale=alt.Scale(scheme='viridis'),
                title=value
            ),
            tooltip=[x, y, value]
        )

        return heatmap

    def create_box_plot(
        self,
        data: pd.DataFrame,
        x: str,
        y: str,
        title: str = "Box Plot",
        width: int = 500,
        height: int = 400
    ) -> alt.Chart:
        """
        Create interactive box plot.

        Args:
            data: DataFrame containing the data
            x: Categorical column
            y: Continuous column
            title: Chart title
            width: Chart width
            height: Chart height

        Returns:
            Altair Chart object
        """
        base = alt.Chart(data).properties(
            width=width,
            height=height,
            title=alt.TitleParams(text=title, fontSize=16, fontWeight='bold')
        )

        # Box plot
        box = base.mark_boxplot(
            color=self.colors['primary'],
            opacity=0.7
        ).encode(
            x=alt.X(f'{x}:O'),
            y=alt.Y(f'{y}:Q'),
            tooltip=[x, y]
        )

        return box

    def create_correlation_matrix(
        self,
        data: pd.DataFrame,
        columns: Optional[List[str]] = None,
        title: str = "Correlation Matrix"
    ) -> alt.Chart:
        """
        Create correlation matrix heatmap.

        Args:
            data: DataFrame containing the data
            columns: Columns to include (default: all numeric)
            title: Chart title

        Returns:
            Altair Chart object
        """
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()

        # Calculate correlation matrix
        corr_matrix = data[columns].corr().reset_index()
        corr_melted = corr_matrix.melt(id_vars='index', var_name='variable', value_name='correlation')
        corr_melted.rename(columns={'index': 'variable1'}, inplace=True)

        # Create heatmap
        heatmap = alt.Chart(corr_melted).mark_rect().encode(
            x=alt.X('variable1:O', title=''),
            y=alt.Y('variable:O', title=''),
            color=alt.Color(
                'correlation:Q',
                scale=alt.Scale(scheme='redblue', domain=[-1, 1]),
                title='Correlation'
            ),
            tooltip=['variable1', 'variable', 'correlation']
        ).properties(
            width=400,
            height=400,
            title=alt.TitleParams(text=title, fontSize=16, fontWeight='bold')
        )

        # Add correlation values as text
        text = alt.Chart(corr_melted).mark_text(
            baseline='middle',
            fontSize=8,
            fontWeight='bold'
        ).encode(
            x='variable1:O',
            y='variable:O',
            text=alt.Text('correlation:Q', format='.2f'),
            color=alt.condition(
                alt.datum.correlation > 0.5,
                alt.value('white'),
                alt.value('black')
            )
        )

        return heatmap + text

    def create_multi_line_chart(
        self,
        data: pd.DataFrame,
        x: str,
        y_columns: List[str],
        title: str = "Multi-line Chart",
        width: int = 700,
        height: int = 400
    ) -> alt.Chart:
        """
        Create multi-line chart with legend and interactivity.

        Args:
            data: DataFrame containing the data
            x: X-axis column name
            y_columns: List of Y-axis column names
            title: Chart title
            width: Chart width
            height: Chart height

        Returns:
            Altair Chart object
        """
        # Melt data for multiple lines
        melted_data = data.melt(
            id_vars=[x],
            value_vars=y_columns,
            var_name='metric',
            value_name='value'
        )

        # Selection for highlighting
        highlight = alt.selection(type='single', on='mouseover')
        
        base = alt.Chart(melted_data).add_selection(
            highlight
        ).properties(
            width=width,
            height=height,
            title=alt.TitleParams(text=title, fontSize=16, fontWeight='bold')
        )

        # Create lines
        lines = base.mark_line(
            size=3,
            point=True
        ).encode(
            x=alt.X(f'{x}:O'),
            y=alt.Y('value:Q'),
            color=alt.Color(
                'metric:N',
                scale=alt.Scale(scheme='category10')
            ),
            opacity=alt.condition(highlight, alt.value(1.0), alt.value(0.7)),
            tooltip=[x, 'metric', 'value']
        )

        return lines

    def create_dashboard(self, charts: List[alt.Chart], columns: int = 2) -> alt.Chart:
        """
        Combine multiple charts into a dashboard layout.

        Args:
            charts: List of Altair charts
            columns: Number of columns in the layout

        Returns:
            Combined dashboard chart
        """
        if not charts:
            return alt.Chart()

        # Arrange charts in rows and columns
        rows = []
        for i in range(0, len(charts), columns):
            row_charts = charts[i:i+columns]
            if len(row_charts) == 1:
                rows.append(row_charts[0])
            else:
                rows.append(alt.hconcat(*row_charts))

        if len(rows) == 1:
            return rows[0]
        else:
            return alt.vconcat(*rows)

    def save_chart(self, chart: alt.Chart, filename: str, format: str = 'html') -> Path:
        """
        Save chart to file.

        Args:
            chart: Altair chart object
            filename: Output filename
            format: Output format ('html', 'json', 'png', 'svg')

        Returns:
            Path to saved file
        """
        output_path = self.charts_dir / f"{filename}.{format}"
        
        if format == 'html':
            chart.save(str(output_path))
        elif format == 'json':
            chart.save(str(output_path), format='json')
        elif format in ['png', 'svg']:
            # Requires altair_saver and dependencies
            chart.save(str(output_path), format=format)
        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info(f"Chart saved to {output_path}")
        return output_path

    def create_backtest_dashboard(self, results: Dict[str, Any]) -> alt.Chart:
        """
        Create comprehensive backtest dashboard using Altair.

        Args:
            results: Backtest results dictionary

        Returns:
            Dashboard with multiple interactive charts
        """
        charts = []

        # Daily MAPE chart
        if 'daily_results' in results:
            daily_df = results['daily_results'].reset_index()
            daily_df['slate_number'] = range(len(daily_df))
            
            mape_chart = self.create_line_chart(
                daily_df,
                x='slate_number',
                y='model_mape',
                title='Daily MAPE Performance',
                width=600,
                height=300
            )
            charts.append(mape_chart)

        # Correlation scatter plot
        if 'all_predictions' in results and not results['all_predictions'].empty:
            all_preds = results['all_predictions']
            if 'actual_fpts' in all_preds.columns and 'projected_fpts' in all_preds.columns:
                scatter_chart = self.create_scatter_plot(
                    all_preds,
                    x='actual_fpts',
                    y='projected_fpts',
                    title='Actual vs Predicted Fantasy Points',
                    width=500,
                    height=400
                )
                charts.append(scatter_chart)

        # Error distribution histogram
        if 'all_predictions' in results and not results['all_predictions'].empty:
            all_preds = results['all_predictions'].copy()
            if 'actual_fpts' in all_preds.columns and 'projected_fpts' in all_preds.columns:
                all_preds['error'] = all_preds['projected_fpts'] - all_preds['actual_fpts']
                error_hist = self.create_histogram(
                    all_preds,
                    column='error',
                    title='Prediction Error Distribution',
                    width=500,
                    height=300
                )
                charts.append(error_hist)

        # Salary tier performance
        if 'tier_comparison' in results:
            tier_df = results['tier_comparison']
            tier_chart = self.create_bar_chart(
                tier_df,
                x='salary_tier',
                y='model_mape',
                title='MAPE by Salary Tier',
                width=500,
                height=300
            )
            charts.append(tier_chart)

        # Create dashboard
        if charts:
            return self.create_dashboard(charts, columns=2)
        else:
            return alt.Chart().mark_text(text="No data available for visualization")

    def apply_theme(self, theme_name: str = 'dark'):
        """
        Apply a specific theme to charts.

        Args:
            theme_name: Theme name ('dark', 'default', 'fivethirtyeight', etc.)
        """
        alt.themes.enable(theme_name)
        self.theme = theme_name
        logger.info(f"Applied theme: {theme_name}")


class AltairBacktestVisualizer(AltairVisualizer):
    """
    Specialized Altair visualizer for backtest analysis.
    
    Provides comprehensive interactive visualizations for model backtest results,
    including performance metrics, error analysis, and statistical comparisons.
    """

    def __init__(self, output_dir: Path, theme: str = 'dark'):
        """
        Initialize backtest visualizer.

        Args:
            output_dir: Directory to save visualizations
            theme: Chart theme
        """
        super().__init__(output_dir, theme)
        
        # Specialized color palette for backtest charts
        self.backtest_colors = {
            'model': self.colors['primary'],
            'benchmark': self.colors['secondary'],
            'positive': self.colors['success'],
            'negative': self.colors['danger'],
            'neutral': self.colors['warning']
        }

    def generate_all_charts(self, results: Dict[str, Any]) -> Dict[str, Path]:
        """
        Generate all backtest visualization charts.

        Args:
            results: Backtest results dictionary

        Returns:
            Dictionary mapping chart names to file paths
        """
        chart_paths = {}

        try:
            # Generate individual chart types
            chart_methods = [
                ('daily_performance', self._create_daily_performance_chart),
                ('model_vs_benchmark', self._create_model_vs_benchmark_chart),
                ('error_analysis', self._create_error_analysis_chart),
                ('salary_tier_analysis', self._create_salary_tier_chart),
                ('correlation_analysis', self._create_correlation_chart),
                ('position_analysis', self._create_position_analysis_chart),
                ('minutes_analysis', self._create_minutes_analysis_chart),
                ('team_analysis', self._create_team_analysis_chart),
                ('residual_analysis', self._create_residual_analysis_chart),
                ('statistical_tests', self._create_statistical_tests_chart),
                ('calibration_curve', self._create_calibration_chart),
                ('error_heatmap', self._create_error_heatmap_chart),
                ('comprehensive_dashboard', self._create_comprehensive_dashboard)
            ]
            
            for chart_name, chart_method in chart_methods:
                try:
                    chart = chart_method(results)
                    if chart is not None:
                        file_path = self.save_chart(chart, chart_name, 'html')
                        chart_paths[chart_name] = file_path
                        logger.info(f"Generated chart: {chart_name}")
                except Exception as e:
                    logger.warning(f"Failed to generate {chart_name}: {str(e)}")
                    continue

            logger.info(f"Generated {len(chart_paths)} backtest charts")

        except Exception as e:
            logger.error(f"Error generating Altair charts: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())

        return chart_paths

    def _create_daily_performance_chart(self, results: Dict[str, Any]) -> Optional[alt.Chart]:
        """
        Create daily performance metrics timeline chart.
        """
        if 'daily_results' not in results:
            return None

        daily_df = results['daily_results'].copy().reset_index()
        if 'date' not in daily_df.columns:
            daily_df['date'] = daily_df.index
        
        # Create multi-line chart for different metrics
        metrics = ['model_mape', 'model_rmse', 'model_mae', 'model_corr']
        if 'benchmark_mape' in daily_df.columns:
            metrics.append('benchmark_mape')
        
        available_metrics = [m for m in metrics if m in daily_df.columns]
        
        if not available_metrics:
            return None

        # Melt data for multi-line visualization
        melted_data = daily_df.melt(
            id_vars=['date'],
            value_vars=available_metrics,
            var_name='metric',
            value_name='value'
        )

        # Create faceted chart with different scales for each metric
        chart = alt.Chart(melted_data).mark_line(
            point=True,
            size=3,
            opacity=0.8
        ).encode(
            x=alt.X('date:T', title='Date'),
            y=alt.Y('value:Q', title='Value', scale=alt.Scale(nice=True)),
            color=alt.Color(
                'metric:N',
                scale=alt.Scale(scheme='category10'),
                legend=alt.Legend(title="Metric")
            ),
            tooltip=['date:T', 'metric:N', 'value:Q']
        ).facet(
            facet=alt.Facet('metric:N', columns=2),
            title="Daily Performance Metrics Timeline"
        ).resolve_scale(
            y='independent'
        ).properties(
            width=300,
            height=200
        )

        return chart

    def _create_model_vs_benchmark_chart(self, results: Dict[str, Any]) -> Optional[alt.Chart]:
        """
        Create comprehensive model vs benchmark comparison.
        """
        if 'daily_results' not in results:
            return None

        daily_df = results['daily_results'].copy().reset_index()
        if 'benchmark_mape' not in daily_df.columns:
            return None

        valid_data = daily_df[daily_df['benchmark_mape'].notna()]
        if valid_data.empty:
            return None

        # Calculate improvement
        valid_data = valid_data.copy()
        valid_data['improvement'] = valid_data['benchmark_mape'] - valid_data['model_mape']
        valid_data['improvement_color'] = valid_data['improvement'].apply(
            lambda x: 'positive' if x > 0 else 'negative'
        )

        # Scatter plot: Model vs Benchmark MAPE
        scatter = self.create_scatter_plot(
            valid_data,
            x='benchmark_mape',
            y='model_mape',
            title='Model vs Benchmark MAPE',
            width=400,
            height=400
        )

        # Daily improvement bar chart
        improvement_chart = alt.Chart(valid_data).mark_bar().encode(
            x=alt.X('date:T', title='Date'),
            y=alt.Y('improvement:Q', title='MAPE Improvement (%)'),
            color=alt.Color(
                'improvement_color:N',
                scale=alt.Scale(
                    domain=['positive', 'negative'],
                    range=[self.backtest_colors['positive'], self.backtest_colors['negative']]
                ),
                legend=None
            ),
            tooltip=['date:T', 'improvement:Q', 'model_mape:Q', 'benchmark_mape:Q']
        ).properties(
            width=600,
            height=300,
            title='Daily MAPE Improvement'
        )

        # Combine charts
        combined = alt.vconcat(
            alt.hconcat(scatter, improvement_chart),
            title="Model vs Benchmark Analysis"
        )

        return combined

    def _create_error_analysis_chart(self, results: Dict[str, Any]) -> Optional[alt.Chart]:
        """
        Create comprehensive error analysis dashboard.
        """
        if 'all_predictions' not in results or results['all_predictions'].empty:
            return None

        all_preds = results['all_predictions'].copy()
        if 'actual_fpts' not in all_preds.columns or 'projected_fpts' not in all_preds.columns:
            return None

        all_preds['error'] = all_preds['projected_fpts'] - all_preds['actual_fpts']
        all_preds['abs_error'] = np.abs(all_preds['error'])

        # Error distribution histogram
        error_hist = self.create_histogram(
            all_preds,
            column='error',
            title='Error Distribution',
            width=400,
            height=300
        )

        # Absolute error distribution
        abs_error_hist = self.create_histogram(
            all_preds,
            column='abs_error',
            title='Absolute Error Distribution',
            width=400,
            height=300,
            color=self.backtest_colors['benchmark']
        )

        # Error vs actual scatter
        error_scatter = self.create_scatter_plot(
            all_preds,
            x='actual_fpts',
            y='abs_error',
            title='Absolute Error vs Actual FPTS',
            width=400,
            height=300
        )

        # Combine charts
        combined = alt.vconcat(
            alt.hconcat(error_hist, abs_error_hist),
            error_scatter,
            title="Error Analysis Dashboard"
        )

        return combined

    def _create_salary_tier_chart(self, results: Dict[str, Any]) -> Optional[alt.Chart]:
        """
        Create salary tier performance analysis.
        """
        if 'tier_comparison' not in results:
            return None

        tier_df = results['tier_comparison'].copy()
        
        # MAPE comparison by tier
        tier_df['improvement_color'] = tier_df['mape_improvement'].apply(
            lambda x: 'positive' if x > 0 else 'negative'
        )

        # Melt data for grouped bar chart
        mape_data = tier_df.melt(
            id_vars=['salary_tier'],
            value_vars=['model_mape', 'benchmark_mape'],
            var_name='type',
            value_name='mape'
        )

        mape_chart = alt.Chart(mape_data).mark_bar().encode(
            x=alt.X('salary_tier:O', title='Salary Tier'),
            y=alt.Y('mape:Q', title='MAPE (%)'),
            color=alt.Color(
                'type:N',
                scale=alt.Scale(
                    domain=['model_mape', 'benchmark_mape'],
                    range=[self.backtest_colors['model'], self.backtest_colors['benchmark']]
                ),
                legend=alt.Legend(title="Type")
            ),
            tooltip=['salary_tier:O', 'type:N', 'mape:Q']
        ).properties(
            width=400,
            height=300,
            title='MAPE by Salary Tier'
        )

        # Improvement chart
        improvement_chart = alt.Chart(tier_df).mark_bar().encode(
            x=alt.X('salary_tier:O', title='Salary Tier'),
            y=alt.Y('mape_improvement:Q', title='MAPE Improvement (%)'),
            color=alt.Color(
                'improvement_color:N',
                scale=alt.Scale(
                    domain=['positive', 'negative'],
                    range=[self.backtest_colors['positive'], self.backtest_colors['negative']]
                ),
                legend=None
            ),
            tooltip=['salary_tier:O', 'mape_improvement:Q']
        ).properties(
            width=400,
            height=300,
            title='MAPE Improvement by Tier'
        )

        return alt.hconcat(mape_chart, improvement_chart, title="Salary Tier Analysis")

    def _create_correlation_chart(self, results: Dict[str, Any]) -> Optional[alt.Chart]:
        """
        Create correlation scatter plot with regression line.
        """
        if 'all_predictions' not in results or results['all_predictions'].empty:
            return None

        all_preds = results['all_predictions']
        if 'actual_fpts' not in all_preds.columns or 'projected_fpts' not in all_preds.columns:
            return None

        # Calculate correlation
        correlation = all_preds[['actual_fpts', 'projected_fpts']].corr().iloc[0, 1]

        chart = self.create_scatter_plot(
            all_preds,
            x='actual_fpts',
            y='projected_fpts',
            title=f'Actual vs Predicted Fantasy Points (r={correlation:.3f})',
            width=600,
            height=500
        )

        return chart

    def _create_position_analysis_chart(self, results: Dict[str, Any]) -> Optional[alt.Chart]:
        """
        Create performance analysis by player position.
        """
        if 'all_predictions' not in results or results['all_predictions'].empty:
            return None

        all_preds = results['all_predictions']
        if 'position' not in all_preds.columns:
            return None

        # Calculate position statistics
        all_preds_copy = all_preds.copy()
        all_preds_copy['abs_error'] = np.abs(all_preds_copy['projected_fpts'] - all_preds_copy['actual_fpts'])
        all_preds_copy['pct_error'] = (all_preds_copy['abs_error'] / all_preds_copy['actual_fpts'].replace(0, np.nan)) * 100

        position_stats = all_preds_copy.groupby('position').agg({
            'abs_error': 'mean',
            'pct_error': 'mean',
            'playerID': 'count'
        }).reset_index()
        position_stats.columns = ['position', 'mae', 'mape', 'count']

        # MAE by position
        mae_chart = self.create_bar_chart(
            position_stats,
            x='position',
            y='mae',
            title='MAE by Position',
            width=400,
            height=300
        )

        # MAPE by position
        mape_chart = self.create_bar_chart(
            position_stats,
            x='position',
            y='mape',
            title='MAPE by Position',
            width=400,
            height=300
        )

        return alt.hconcat(mae_chart, mape_chart, title="Position Analysis")

    def _create_minutes_analysis_chart(self, results: Dict[str, Any]) -> Optional[alt.Chart]:
        """
        Create performance analysis by minutes played.
        """
        if 'all_predictions' not in results or results['all_predictions'].empty:
            return None

        all_preds = results['all_predictions']
        if 'actual_mins' not in all_preds.columns:
            return None

        all_preds_copy = all_preds.copy()
        all_preds_copy['abs_error'] = np.abs(all_preds_copy['projected_fpts'] - all_preds_copy['actual_fpts'])
        
        # Create minutes bins
        all_preds_copy['minutes_bin'] = pd.cut(
            all_preds_copy['actual_mins'],
            bins=[0, 10, 20, 30, 40, 50],
            labels=['0-10', '10-20', '20-30', '30-40', '40+']
        )

        minutes_stats = all_preds_copy.groupby('minutes_bin', observed=True).agg({
            'abs_error': 'mean',
            'playerID': 'count'
        }).reset_index()
        minutes_stats.columns = ['minutes_bin', 'mae', 'count']
        minutes_stats['minutes_bin'] = minutes_stats['minutes_bin'].astype(str)

        chart = self.create_bar_chart(
            minutes_stats,
            x='minutes_bin',
            y='mae',
            title='MAE by Minutes Played',
            width=600,
            height=400
        )

        return chart

    def _create_team_analysis_chart(self, results: Dict[str, Any]) -> Optional[alt.Chart]:
        """
        Create performance analysis by team.
        """
        if 'all_predictions' not in results or results['all_predictions'].empty:
            return None

        all_preds = results['all_predictions']
        if 'team' not in all_preds.columns:
            return None

        all_preds_copy = all_preds.copy()
        all_preds_copy['pct_error'] = (np.abs(all_preds_copy['projected_fpts'] - all_preds_copy['actual_fpts']) /
                                       all_preds_copy['actual_fpts'].replace(0, np.nan)) * 100

        team_stats = all_preds_copy.groupby('team').agg({
            'pct_error': 'mean',
            'playerID': 'count'
        }).reset_index()
        team_stats.columns = ['team', 'mape', 'count']
        team_stats = team_stats.sort_values('mape')

        chart = self.create_bar_chart(
            team_stats,
            x='team',
            y='mape',
            title='MAPE by Team',
            width=800,
            height=400
        )

        return chart

    def _create_residual_analysis_chart(self, results: Dict[str, Any]) -> Optional[alt.Chart]:
        """
        Create residual analysis for model diagnostics.
        """
        if 'all_predictions' not in results or results['all_predictions'].empty:
            return None

        all_preds = results['all_predictions']
        if 'actual_fpts' not in all_preds.columns or 'projected_fpts' not in all_preds.columns:
            return None

        all_preds_copy = all_preds.copy()
        all_preds_copy['residual'] = all_preds_copy['actual_fpts'] - all_preds_copy['projected_fpts']
        all_preds_copy['fitted'] = all_preds_copy['projected_fpts']

        # Residuals vs fitted scatter
        residual_scatter = self.create_scatter_plot(
            all_preds_copy,
            x='fitted',
            y='residual',
            title='Residuals vs Fitted Values',
            width=400,
            height=300
        )

        # Residual distribution
        residual_hist = self.create_histogram(
            all_preds_copy,
            column='residual',
            title='Residual Distribution',
            width=400,
            height=300
        )

        return alt.hconcat(residual_scatter, residual_hist, title="Residual Analysis")

    def _create_statistical_tests_chart(self, results: Dict[str, Any]) -> Optional[alt.Chart]:
        """
        Create statistical test visualization.
        """
        if 'statistical_test' not in results:
            return None

        test = results['statistical_test']
        
        # Create summary data
        test_data = pd.DataFrame([
            {'metric': 'p-value', 'value': test['p_value'], 'threshold': 0.05},
            {'metric': "Cohen's d", 'value': abs(test['cohens_d']), 'threshold': 0.5}
        ])
        
        test_data['significant'] = test_data.apply(
            lambda row: 'significant' if (
                (row['metric'] == 'p-value' and row['value'] < row['threshold']) or
                (row['metric'] == "Cohen's d" and row['value'] > row['threshold'])
            ) else 'not_significant', axis=1
        )

        chart = alt.Chart(test_data).mark_bar().encode(
            x=alt.X('metric:O', title='Statistical Test'),
            y=alt.Y('value:Q', title='Value'),
            color=alt.Color(
                'significant:N',
                scale=alt.Scale(
                    domain=['significant', 'not_significant'],
                    range=[self.backtest_colors['positive'], self.backtest_colors['negative']]
                ),
                legend=alt.Legend(title="Significance")
            ),
            tooltip=['metric:O', 'value:Q', 'threshold:Q']
        ).properties(
            width=400,
            height=300,
            title=f"Statistical Test Results (t={test['t_statistic']:.4f})"
        )

        return chart

    def _create_calibration_chart(self, results: Dict[str, Any]) -> Optional[alt.Chart]:
        """
        Create calibration curve for predictions.
        """
        if 'all_predictions' not in results or results['all_predictions'].empty:
            return None

        all_preds = results['all_predictions']
        if 'actual_fpts' not in all_preds.columns or 'projected_fpts' not in all_preds.columns:
            return None

        # Create calibration data
        all_preds_sorted = all_preds.sort_values('projected_fpts')
        n_bins = 20
        bin_size = len(all_preds_sorted) // n_bins

        calibration_data = []
        for i in range(n_bins):
            start_idx = i * bin_size
            end_idx = start_idx + bin_size if i < n_bins - 1 else len(all_preds_sorted)
            
            bin_data = all_preds_sorted.iloc[start_idx:end_idx]
            calibration_data.append({
                'predicted_mean': bin_data['projected_fpts'].mean(),
                'actual_mean': bin_data['actual_fpts'].mean(),
                'bin': i
            })

        calibration_df = pd.DataFrame(calibration_data)

        chart = self.create_scatter_plot(
            calibration_df,
            x='predicted_mean',
            y='actual_mean',
            title='Calibration Curve: Predicted vs Actual FPTS',
            width=500,
            height=400
        )

        return chart

    def _create_error_heatmap_chart(self, results: Dict[str, Any]) -> Optional[alt.Chart]:
        """
        Create error heatmap by salary tier and position.
        """
        if 'all_predictions' not in results or results['all_predictions'].empty:
            return None

        all_preds = results['all_predictions']
        if 'position' not in all_preds.columns or 'salary' not in all_preds.columns:
            return None

        all_preds_copy = all_preds.copy()
        all_preds_copy['abs_error'] = np.abs(all_preds_copy['projected_fpts'] - all_preds_copy['actual_fpts'])
        
        # Create salary tiers
        all_preds_copy['salary_tier'] = pd.cut(
            all_preds_copy['salary'],
            bins=[0, 4000, 6000, 8000, 12000],
            labels=['Low', 'Mid', 'High', 'Elite']
        )

        # Calculate mean absolute error by position and salary tier
        heatmap_data = all_preds_copy.groupby(['salary_tier', 'position'], observed=True)['abs_error'].mean().reset_index()
        heatmap_data['salary_tier'] = heatmap_data['salary_tier'].astype(str)

        chart = self.create_heatmap(
            heatmap_data,
            x='position',
            y='salary_tier',
            value='abs_error',
            title='Mean Absolute Error Heatmap (Salary Tier × Position)',
            width=500,
            height=300
        )

        return chart

    def _create_comprehensive_dashboard(self, results: Dict[str, Any]) -> Optional[alt.Chart]:
        """
        Create comprehensive single-page dashboard.
        """
        charts = []

        # Key metrics indicators
        model_mape = results.get('model_mean_mape', 0)
        model_corr = results.get('model_mean_correlation', 0)
        
        # Create summary metrics chart
        summary_data = pd.DataFrame([
            {'metric': 'MAPE (%)', 'value': model_mape},
            {'metric': 'Correlation', 'value': model_corr},
            {'metric': 'RMSE', 'value': results.get('model_mean_rmse', 0)}
        ])
        
        summary_chart = alt.Chart(summary_data).mark_bar(
            color=self.backtest_colors['model']
        ).encode(
            x=alt.X('metric:O', title='Metric'),
            y=alt.Y('value:Q', title='Value'),
            tooltip=['metric:O', 'value:Q']
        ).properties(
            width=300,
            height=200,
            title='Key Performance Metrics'
        )
        
        charts.append(summary_chart)

        # Daily MAPE trend
        if 'daily_results' in results:
            daily_df = results['daily_results'].reset_index()
            if 'model_mape' in daily_df.columns:
                daily_chart = self.create_line_chart(
                    daily_df,
                    x='date' if 'date' in daily_df.columns else daily_df.index.name or 'index',
                    y='model_mape',
                    title='Daily MAPE Trend',
                    width=400,
                    height=200
                )
                charts.append(daily_chart)

        # Error distribution
        if 'all_predictions' in results and not results['all_predictions'].empty:
            all_preds = results['all_predictions']
            if 'actual_fpts' in all_preds.columns and 'projected_fpts' in all_preds.columns:
                all_preds_copy = all_preds.copy()
                all_preds_copy['error'] = all_preds_copy['projected_fpts'] - all_preds_copy['actual_fpts']
                
                error_chart = self.create_histogram(
                    all_preds_copy,
                    column='error',
                    title='Error Distribution',
                    width=300,
                    height=200
                )
                charts.append(error_chart)

        # Salary tier performance
        if 'tier_comparison' in results:
            tier_df = results['tier_comparison']
            tier_chart = self.create_bar_chart(
                tier_df,
                x='salary_tier',
                y='model_mape',
                title='MAPE by Salary Tier',
                width=400,
                height=200
            )
            charts.append(tier_chart)

        # Create dashboard layout
        if charts:
            return self.create_dashboard(charts, columns=2)
        else:
            return alt.Chart().mark_text(
                text="No data available for dashboard",
                fontSize=16
            ).properties(
                width=400,
                height=200
            )
