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
    Interactive visualization generator using Vega-Altair.
    
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