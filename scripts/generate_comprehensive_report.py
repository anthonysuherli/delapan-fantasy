import sys
import argparse
import logging
import pandas as pd
from pathlib import Path
from scipy import stats as scipy_stats
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from src.evaluation.report_generator import BacktestReportGenerator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_existing_backtest_results(output_dir: Path):
    """
    Load existing backtest results from output directory.

    Args:
        output_dir: Path to backtest output directory

    Returns:
        Dictionary with results data
    """
    logger.info(f"Loading backtest results from {output_dir}")

    daily_results_path = None
    tier_comparison_path = None

    for csv_file in output_dir.glob('backtest_results_*.csv'):
        daily_results_path = csv_file
        break

    for tier_file in output_dir.glob('tier_comparison_*.csv'):
        tier_comparison_path = tier_file
        break

    if not daily_results_path:
        raise FileNotFoundError(f"No backtest_results_*.csv found in {output_dir}")

    daily_results = pd.read_csv(daily_results_path)
    logger.info(f"Loaded daily results: {len(daily_results)} slates")

    predictions_dir = output_dir / 'predictions'
    all_predictions = []

    if predictions_dir.exists():
        logger.info("Loading prediction files...")
        for pred_file in sorted(predictions_dir.glob('*_with_actuals.parquet')):
            try:
                df = pd.read_parquet(pred_file)
                all_predictions.append(df)
            except Exception as e:
                logger.warning(f"Failed to load {pred_file}: {e}")

        if all_predictions:
            all_predictions = pd.concat(all_predictions, ignore_index=True)
            logger.info(f"Loaded {len(all_predictions)} total predictions")
        else:
            all_predictions = pd.DataFrame()
    else:
        logger.warning(f"Predictions directory not found: {predictions_dir}")
        all_predictions = pd.DataFrame()

    results = {
        'daily_results': daily_results,
        'all_predictions': all_predictions,
        'num_slates': len(daily_results),
        'total_players_evaluated': daily_results['num_players'].sum(),
        'avg_players_per_slate': daily_results['num_players'].mean(),
        'date_range': f"{daily_results['date'].min()} to {daily_results['date'].max()}"
    }

    results['model_mean_mape'] = daily_results['model_mape'].mean()
    results['model_median_mape'] = daily_results['model_mape'].median()
    results['model_std_mape'] = daily_results['model_mape'].std()
    results['model_mean_rmse'] = daily_results['model_rmse'].mean()
    results['model_std_rmse'] = daily_results['model_rmse'].std()
    results['model_mean_mae'] = daily_results['model_mae'].mean()
    results['model_mean_correlation'] = daily_results['model_corr'].mean()
    results['model_std_correlation'] = daily_results['model_corr'].std()

    if 'model_cmape' in daily_results.columns:
        results['model_mean_cmape'] = daily_results['model_cmape'].mean()
    if 'model_smape' in daily_results.columns:
        results['model_mean_smape'] = daily_results['model_smape'].mean()
    if 'model_wmape' in daily_results.columns:
        results['model_mean_wmape'] = daily_results['model_wmape'].mean()

    if 'benchmark_mape' in daily_results.columns:
        results['benchmark_mean_mape'] = daily_results['benchmark_mape'].mean()
        results['benchmark_median_mape'] = daily_results['benchmark_mape'].median()
        results['mape_improvement'] = results['benchmark_mean_mape'] - results['model_mean_mape']

        if 'benchmark_cmape' in daily_results.columns:
            results['benchmark_mean_cmape'] = daily_results['benchmark_cmape'].mean()
        if 'benchmark_wmape' in daily_results.columns:
            results['benchmark_mean_wmape'] = daily_results['benchmark_wmape'].mean()

        valid_pairs = daily_results[['model_mape', 'benchmark_mape']].dropna()
        if len(valid_pairs) > 1:
            t_stat, p_val = scipy_stats.ttest_rel(valid_pairs['model_mape'], valid_pairs['benchmark_mape'])

            pooled_std = np.sqrt(
                (valid_pairs['model_mape'].std()**2 + valid_pairs['benchmark_mape'].std()**2) / 2
            )
            cohens_d = (valid_pairs['model_mape'].mean() - valid_pairs['benchmark_mape'].mean()) / pooled_std

            if abs(cohens_d) < 0.2:
                effect_size = 'negligible'
            elif abs(cohens_d) < 0.5:
                effect_size = 'small'
            elif abs(cohens_d) < 0.8:
                effect_size = 'medium'
            else:
                effect_size = 'large'

            results['statistical_test'] = {
                't_statistic': t_stat,
                'p_value': p_val,
                'cohens_d': cohens_d,
                'effect_size': effect_size
            }

    if tier_comparison_path:
        tier_comparison = pd.read_csv(tier_comparison_path)
        results['tier_comparison'] = tier_comparison
        logger.info(f"Loaded tier comparison: {len(tier_comparison)} tiers")

    return results


def main():
    parser = argparse.ArgumentParser(description='Generate comprehensive report from existing backtest results')

    parser.add_argument('--output-dir', type=str, required=True,
                        help='Path to backtest output directory (e.g., data/outputs/20251015_204608)')
    parser.add_argument('--use-plotly', action='store_true', default=True,
                        help='Use Plotly for interactive charts (default: True)')
    parser.add_argument('--no-plotly', dest='use_plotly', action='store_false',
                        help='Use matplotlib for static charts instead of Plotly')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    if not output_dir.exists():
        logger.error(f"Output directory does not exist: {output_dir}")
        sys.exit(1)

    logger.info("="*80)
    logger.info("COMPREHENSIVE REPORT GENERATION")
    logger.info("="*80)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Visualization: {'Plotly (interactive)' if args.use_plotly else 'matplotlib (static)'}")
    logger.info("="*80)
    logger.info("")

    try:
        results = load_existing_backtest_results(output_dir)

        config = {
            'train_start': 'N/A',
            'train_end': 'N/A',
            'test_start': results['date_range'].split(' to ')[0],
            'test_end': results['date_range'].split(' to ')[1],
            'num_seasons': 'N/A',
            'model_type': 'xgboost',
            'feature_config': 'default_features',
            'per_player_models': True,
            'recalibrate_days': 7,
            'n_jobs': -1,
            'rewrite_models': False,
            'model_params': {}
        }

        report_gen = BacktestReportGenerator(output_dir, use_plotly=args.use_plotly)
        run_timestamp = output_dir.name

        logger.info("Generating comprehensive report with visualizations...")
        comprehensive_report_path = report_gen.generate_report(
            results=results,
            config=config,
            run_timestamp=run_timestamp,
            generate_charts=True
        )

        logger.info("")
        logger.info("="*80)
        logger.info("REPORT GENERATION COMPLETE")
        logger.info("="*80)
        logger.info(f"Comprehensive report: {comprehensive_report_path}")

        charts_dir = output_dir / 'charts'
        if charts_dir.exists():
            if args.use_plotly:
                num_charts = len(list(charts_dir.glob('*.html')))
                logger.info(f"Generated {num_charts} interactive Plotly charts in {charts_dir}")
            else:
                num_charts = len(list(charts_dir.glob('*.png')))
                logger.info(f"Generated {num_charts} static matplotlib charts in {charts_dir}")

        logger.info("="*80)

    except Exception as e:
        logger.error(f"Failed to generate report: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == '__main__':
    main()
