"""
Example usage of the Altair charting module.

This script demonstrates how to use the AltairVisualizer class
to create interactive visualizations for fantasy sports data analysis.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from evaluation.altair_visualizations import AltairVisualizer


def create_sample_data():
    """Create sample data for demonstrations."""
    np.random.seed(42)
    
    # Sample backtest results
    n_slates = 20
    daily_results = pd.DataFrame({
        'slate_number': range(n_slates),
        'model_mape': np.random.normal(15, 3, n_slates),
        'model_rmse': np.random.normal(8, 1.5, n_slates),
        'model_mae': np.random.normal(6, 1, n_slates),
        'model_corr': np.random.beta(4, 1, n_slates) * 0.4 + 0.6,
        'benchmark_mape': np.random.normal(18, 4, n_slates)
    })
    
    # Sample predictions data
    n_predictions = 500
    actual_fpts = np.random.gamma(3, 5, n_predictions)
    noise = np.random.normal(0, 3, n_predictions)
    projected_fpts = actual_fpts * 0.85 + noise + 2
    
    predictions_data = pd.DataFrame({
        'actual_fpts': actual_fpts,
        'projected_fpts': projected_fpts,
        'salary': np.random.uniform(3000, 12000, n_predictions),
        'position': np.random.choice(['PG', 'SG', 'SF', 'PF', 'C'], n_predictions),
        'team': np.random.choice(['LAL', 'GSW', 'BOS', 'MIA', 'CHI'], n_predictions)
    })
    
    # Sample tier comparison
    tier_comparison = pd.DataFrame({
        'salary_tier': ['Low', 'Mid-Low', 'Mid', 'Mid-High', 'High'],
        'model_mape': [18.5, 15.2, 12.8, 14.1, 16.3],
        'benchmark_mape': [22.1, 18.7, 15.4, 16.8, 19.2],
        'mape_improvement': [3.6, 3.5, 2.6, 2.7, 2.9]
    })
    
    return {
        'daily_results': daily_results,
        'all_predictions': predictions_data,
        'tier_comparison': tier_comparison
    }


def demonstrate_basic_charts():
    """Demonstrate basic chart types."""
    print("Creating basic chart examples...")
    
    # Create sample data
    sample_results = create_sample_data()
    
    # Initialize visualizer
    output_dir = Path(__file__).parent / 'output'
    viz = AltairVisualizer(output_dir, theme='dark')
    
    # 1. Scatter Plot Example
    print("Creating scatter plot...")
    scatter_chart = viz.create_scatter_plot(
        data=sample_results['all_predictions'],
        x='actual_fpts',
        y='projected_fpts',
        color='position',
        size='salary',
        title='Fantasy Points: Actual vs Predicted by Position',
        width=600,
        height=500
    )
    viz.save_chart(scatter_chart, 'scatter_example')
    
    # 2. Line Chart Example
    print("Creating line chart...")
    line_chart = viz.create_line_chart(
        data=sample_results['daily_results'],
        x='slate_number',
        y='model_mape',
        title='Model MAPE Over Time',
        width=700,
        height=400
    )
    viz.save_chart(line_chart, 'line_example')
    
    # 3. Bar Chart Example
    print("Creating bar chart...")
    bar_chart = viz.create_bar_chart(
        data=sample_results['tier_comparison'],
        x='salary_tier',
        y='model_mape',
        color='mape_improvement',
        title='MAPE by Salary Tier',
        width=600,
        height=400
    )
    viz.save_chart(bar_chart, 'bar_example')
    
    # 4. Histogram Example
    print("Creating histogram...")
    hist_chart = viz.create_histogram(
        data=sample_results['all_predictions'],
        column='actual_fpts',
        bins=25,
        title='Distribution of Actual Fantasy Points',
        width=600,
        height=400,
        color='#2E86AB'
    )
    viz.save_chart(hist_chart, 'histogram_example')
    
    # 5. Box Plot Example
    print("Creating box plot...")
    box_chart = viz.create_box_plot(
        data=sample_results['all_predictions'],
        x='position',
        y='actual_fpts',
        title='Fantasy Points Distribution by Position',
        width=600,
        height=400
    )
    viz.save_chart(box_chart, 'boxplot_example')
    
    print("Basic charts saved to", viz.charts_dir)


def demonstrate_advanced_charts():
    """Demonstrate advanced chart types and features."""
    print("Creating advanced chart examples...")
    
    # Create sample data
    sample_results = create_sample_data()
    
    # Initialize visualizer
    output_dir = Path(__file__).parent / 'output'
    viz = AltairVisualizer(output_dir, theme='fivethirtyeight')
    
    # 1. Correlation Matrix
    print("Creating correlation matrix...")
    numeric_cols = ['actual_fpts', 'projected_fpts', 'salary']
    corr_chart = viz.create_correlation_matrix(
        data=sample_results['all_predictions'],
        columns=numeric_cols,
        title='Feature Correlation Matrix'
    )
    viz.save_chart(corr_chart, 'correlation_matrix_example')
    
    # 2. Multi-line Chart
    print("Creating multi-line chart...")
    multi_line_data = sample_results['daily_results'][['slate_number', 'model_mape', 'benchmark_mape']]
    multi_line_chart = viz.create_multi_line_chart(
        data=multi_line_data,
        x='slate_number',
        y_columns=['model_mape', 'benchmark_mape'],
        title='Model vs Benchmark Performance',
        width=700,
        height=400
    )
    viz.save_chart(multi_line_chart, 'multiline_example')
    
    # 3. Heatmap Example (create sample heatmap data)
    print("Creating heatmap...")
    
    # Create position vs team performance heatmap data
    positions = ['PG', 'SG', 'SF', 'PF', 'C']
    teams = ['LAL', 'GSW', 'BOS', 'MIA', 'CHI']
    heatmap_data = []
    
    for pos in positions:
        for team in teams:
            avg_fpts = np.random.uniform(12, 25)
            heatmap_data.append({'position': pos, 'team': team, 'avg_fpts': avg_fpts})
    
    heatmap_df = pd.DataFrame(heatmap_data)
    heatmap_chart = viz.create_heatmap(
        data=heatmap_df,
        x='team',
        y='position',
        value='avg_fpts',
        title='Average Fantasy Points by Position and Team',
        width=400,
        height=300
    )
    viz.save_chart(heatmap_chart, 'heatmap_example')
    
    print("Advanced charts saved to", viz.charts_dir)


def demonstrate_dashboard():
    """Demonstrate dashboard creation."""
    print("Creating comprehensive dashboard...")
    
    # Create sample data
    sample_results = create_sample_data()
    
    # Initialize visualizer
    output_dir = Path(__file__).parent / 'output'
    viz = AltairVisualizer(output_dir, theme='dark')
    
    # Create dashboard using built-in method
    dashboard = viz.create_backtest_dashboard(sample_results)
    viz.save_chart(dashboard, 'backtest_dashboard')
    
    # Create custom dashboard
    charts = []
    
    # Performance over time
    performance_chart = viz.create_multi_line_chart(
        data=sample_results['daily_results'],
        x='slate_number',
        y_columns=['model_mape', 'model_rmse', 'model_mae'],
        title='Model Performance Metrics Over Time',
        width=500,
        height=300
    )
    charts.append(performance_chart)
    
    # Error distribution
    sample_results['all_predictions']['error'] = (
        sample_results['all_predictions']['projected_fpts'] - 
        sample_results['all_predictions']['actual_fpts']
    )
    error_hist = viz.create_histogram(
        data=sample_results['all_predictions'],
        column='error',
        title='Prediction Error Distribution',
        width=500,
        height=300
    )
    charts.append(error_hist)
    
    # Position performance
    position_box = viz.create_box_plot(
        data=sample_results['all_predictions'],
        x='position',
        y='actual_fpts',
        title='Fantasy Points by Position',
        width=500,
        height=300
    )
    charts.append(position_box)
    
    # Salary vs performance
    salary_scatter = viz.create_scatter_plot(
        data=sample_results['all_predictions'],
        x='salary',
        y='actual_fpts',
        color='position',
        title='Salary vs Fantasy Points',
        width=500,
        height=300
    )
    charts.append(salary_scatter)
    
    # Combine into custom dashboard
    custom_dashboard = viz.create_dashboard(charts, columns=2)
    viz.save_chart(custom_dashboard, 'custom_dashboard')
    
    print("Dashboards saved to", viz.charts_dir)


def demonstrate_themes():
    """Demonstrate different themes."""
    print("Creating charts with different themes...")
    
    sample_results = create_sample_data()
    output_dir = Path(__file__).parent / 'output'
    
    themes = ['dark', 'default', 'fivethirtyeight']
    
    for theme in themes:
        print(f"Creating chart with {theme} theme...")
        viz = AltairVisualizer(output_dir, theme=theme)
        
        chart = viz.create_scatter_plot(
            data=sample_results['all_predictions'],
            x='actual_fpts',
            y='projected_fpts',
            color='position',
            title=f'Scatter Plot - {theme.title()} Theme',
            width=600,
            height=400
        )
        
        viz.save_chart(chart, f'scatter_{theme}_theme')
    
    print("Theme examples saved to", output_dir / 'altair_charts')


def main():
    """Run all examples."""
    print("Altair Charting Module Examples")
    print("=" * 40)
    
    try:
        demonstrate_basic_charts()
        print()
        
        demonstrate_advanced_charts()
        print()
        
        demonstrate_dashboard()
        print()
        
        demonstrate_themes()
        print()
        
        print("All examples completed successfully!")
        print("Check the 'output/altair_charts' directory for generated charts.")
        
    except Exception as e:
        print(f"Error running examples: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()