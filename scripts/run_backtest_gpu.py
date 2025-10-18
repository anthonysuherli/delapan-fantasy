import sys, argparse, logging, yaml
from pathlib import Path
from datetime import datetime, timedelta
import traceback

sys.path.append(str(Path(__file__).parent.parent))
from warnings import filterwarnings
filterwarnings('ignore')

from src.walk_forward_backtest import WalkForwardBacktest
from src.data.loaders.historical_loader import HistoricalDataLoader
from src.data.loaders.gpu_data_preprocessor import GPUDataPipeline
from src.filters import ColumnFilter, InjuryFilter, CompositeFilter


def setup_logging(log_file: Path):
    """
    Setup comprehensive logging to both console and file.

    Captures:
    - All application logs at DEBUG level
    - Console output at INFO level
    - All warnings and exceptions with tracebacks
    - Stderr output
    """
    # Create logs directory
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Detailed formatter for file (includes everything)
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(funcName)s() | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console formatter (more concise)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )

    # File handler (DEBUG level - captures everything)
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    # Console handler (INFO level - user-friendly output)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # Error handler (also logs to file with traceback)
    error_handler = logging.FileHandler(log_file.with_stem(log_file.stem + '_errors'), mode='w', encoding='utf-8')
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(file_formatter)
    root_logger.addHandler(error_handler)

    # Capture warnings
    logging.captureWarnings(True)

    return root_logger


class StreamToLogger:
    """
    Redirect stderr/stdout to logging.
    Captures terminal output that bypasses standard logging.
    Parses log levels from messages to avoid logging INFO as ERROR.
    """
    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            # Skip tqdm progress bars (contain percentage or [time<time])
            if not line.strip() or '%|' in line or 'it/s]' in line:
                continue

            # Parse log level from message format: "name - LEVEL - message"
            level = self.log_level
            if ' - INFO - ' in line:
                level = logging.INFO
            elif ' - DEBUG - ' in line:
                level = logging.DEBUG
            elif ' - WARNING - ' in line:
                level = logging.WARNING
            elif ' - ERROR - ' in line:
                level = logging.ERROR
            elif ' - CRITICAL - ' in line:
                level = logging.CRITICAL

            self.logger.log(level, line.rstrip())

    def flush(self):
        pass

    def isatty(self):
        return False


logger = logging.getLogger(__name__)


def load_config(config_path: str):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='Run walk-forward backtest with GPU acceleration')

    parser.add_argument('--db-path', type=str, default='nba_dfs.db',
                        help='Path to SQLite database')
    parser.add_argument('--test-start', type=str, required=True,
                        help='Testing start date (YYYYMMDD)')
    parser.add_argument('--test-end', type=str, required=True,
                        help='Testing end date (YYYYMMDD)')
    parser.add_argument('--log-dir', type=str, default='data/logs',
                        help='Directory for log files (default: data/logs)')
    parser.add_argument('--num-seasons', type=int, default=1,
                        help='Number of seasons for training data (default: 1)')

    parser.add_argument('--model-type', type=str, default='xgboost',
                        choices=['xgboost', 'random_forest'],
                        help='Model type to use')
    parser.add_argument('--model-config', type=str, default='config/models/xgboost_default.yaml',
                        help='Path to model configuration YAML')
    parser.add_argument('--feature-config', type=str, default='default_features',
                        help='Feature configuration name')

    parser.add_argument('--per-player', action='store_true',
                        help='Use per-player models instead of slate-level')
    parser.add_argument('--min-player-games', type=int, default=10,
                        help='Minimum games required for per-player models')
    parser.add_argument('--recalibrate-days', type=int, default=7,
                        help='Days between model recalibration')

    parser.add_argument('--n-jobs', type=int, default=-1,
                        help='Number of parallel jobs for per-player training (-1 = all cores)')
    parser.add_argument('--num-workers', type=int, default=16,
                        help='Number of workers for data loading')

    parser.add_argument('--output-dir', type=str, default='data/backtest_results',
                        help='Output directory for results')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Data directory (for separated architecture)')
 
    parser.add_argument('--save-models', action='store_true', default=True,
                        help='Save trained models to disk')
    parser.add_argument('--save-predictions', action='store_true', default=True,
                        help='Save predictions to disk')
    parser.add_argument('--rewrite-models', action='store_true',
                        help='Force rewrite existing models')

    parser.add_argument('--resume-from', type=str, default=None,
                        help='Resume from existing run timestamp')

    parser.add_argument('--gpu-id', type=int, default=0,
                        help='GPU device ID to use')

    parser.add_argument('--filter-salary-min', type=int, default=None,
                        help='Minimum salary filter (default: None)')
    parser.add_argument('--filter-salary-max', type=int, default=None,
                        help='Maximum salary filter (default: None)')
    parser.add_argument('--filter-exclude-out', action='store_true',
                        help='Exclude players ruled out with injuries')
    parser.add_argument('--filter-exclude-doubtful', action='store_true',
                        help='Exclude players with doubtful injury status')
    parser.add_argument('--filter-exclude-questionable', action='store_true',
                        help='Exclude players with questionable injury status')

    parser.add_argument('--enable-gpu-caching', action='store_true',
                        help='Enable GPU data preprocessing caching for faster reruns')
    parser.add_argument('--cache-dir', type=str, default='data/preprocessed_cache',
                        help='Directory for cached preprocessed data (default: data/preprocessed_cache)')
    
    parser.add_argument('--enable-feature-caching', action='store_true', default=True,
                        help='Enable feature preprocessing caching for faster reruns (default: True)')
    parser.add_argument('--disable-feature-caching', action='store_true',
                        help='Disable feature preprocessing caching')
    parser.add_argument('--gpu-batch-size', type=int, default=1,
                        help='GPU batch size for per-player model training (default: 8)')
    parser.add_argument('--debug-gpu', action='store_true',
                        help='Enable debug logging for GPU batch training')

    args = parser.parse_args()

    # Setup comprehensive logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = Path(args.log_dir)
    log_file = log_dir / f'backtest_{args.test_start}_to_{args.test_end}_{timestamp}.log'

    try:
        root_logger = setup_logging(log_file)
        
        # Enable debug logging if requested
        if args.debug_gpu:
            logger.info("Debug GPU logging enabled")
            gpu_logger = logging.getLogger('src.models.gpu_batch_trainer')
            gpu_logger.setLevel(logging.DEBUG)
            backtest_logger = logging.getLogger('src.walk_forward_backtest')
            backtest_logger.setLevel(logging.DEBUG)
        
        # Redirect stderr to logger
        sys.stderr = StreamToLogger(root_logger, logging.ERROR)
    except Exception as e:
        print(f"Failed to setup logging: {e}")
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    logger.info(f"Logging initialized: {log_file}")
    logger.info(f"Detailed logs at DEBUG level: {log_file}")
    logger.info(f"Errors and tracebacks: {log_file.with_stem(log_file.stem + '_errors')}")

    try:
        test_end_dt = datetime.strptime(args.test_end, '%Y%m%d')
        train_end = (test_end_dt - timedelta(days=1)).strftime('%Y%m%d')

        if args.num_seasons == 1:
            train_start = HistoricalDataLoader.get_season_start_date(args.test_start)
        else:
            train_start = HistoricalDataLoader.get_previous_season_start_date(args.test_start)

        logger.info("="*80)
        logger.info("GPU-ACCELERATED BACKTEST CONFIGURATION")
        logger.info("="*80)
        logger.info(f"Database: {args.db_path}")
        logger.info(f"Training period: {train_start} to {train_end}")
        logger.info(f"Testing period: {args.test_start} to {args.test_end}")
        logger.info(f"Model type: {args.model_type}")
        logger.info(f"Model config: {args.model_config}")
        logger.info(f"Feature config: {args.feature_config}")
        logger.info(f"Per-player models: {args.per_player}")
        logger.info(f"GPU device: {args.gpu_id}")
        logger.info(f"Parallel jobs: {args.n_jobs} ({'all cores' if args.n_jobs == -1 else f'{args.n_jobs} workers'})")
        logger.info(f"Data loader workers: {args.num_workers}")
        logger.info(f"GPU data caching: {'Enabled' if args.enable_gpu_caching else 'Disabled'}")
        if args.enable_gpu_caching:
            logger.info(f"Cache directory: {args.cache_dir}")
        
        # Process optimization settings
        enable_feature_caching = args.enable_feature_caching and not args.disable_feature_caching
        logger.info(f"Feature caching: {'Enabled' if enable_feature_caching else 'Disabled'}")
        logger.info(f"GPU batch size: {args.gpu_batch_size} (for per-player models)")
        logger.info("="*80)

        model_params = None
        if args.model_config:
            try:
                config = load_config(args.model_config)
                model_params = config.get('model', {}).get('params', {})

                if args.model_type == 'xgboost':
                    if 'device' not in model_params:
                        model_params['device'] = f'cuda:{args.gpu_id}'
                    if 'tree_method' not in model_params:
                        model_params['tree_method'] = 'hist'

                logger.info(f"Loaded model parameters from {args.model_config}")
                logger.info(f"GPU configuration: tree_method={model_params.get('tree_method')}, gpu_id={args.gpu_id} device={model_params.get('device')}")
            except Exception as e:
                logger.warning(f"Failed to load model config: {e}. Using defaults.", exc_info=True)

        logger.info("")

        player_filters = []
        if args.filter_salary_min is not None:
            salary_filter = ColumnFilter('salary', '>=', args.filter_salary_min)
            player_filters.append(salary_filter)
            logger.info(f"Filter: salary >= {args.filter_salary_min}")

        if args.filter_salary_max is not None:
            salary_filter = ColumnFilter('salary', '<=', args.filter_salary_max)
            player_filters.append(salary_filter)
            logger.info(f"Filter: salary <= {args.filter_salary_max}")

        if args.filter_exclude_out or args.filter_exclude_doubtful or args.filter_exclude_questionable:
            injury_filter = InjuryFilter(
                exclude_out=args.filter_exclude_out,
                exclude_doubtful=args.filter_exclude_doubtful,
                exclude_questionable=args.filter_exclude_questionable
            )
            player_filters.append(injury_filter)
            excluded = []
            if args.filter_exclude_out:
                excluded.append('OUT')
            if args.filter_exclude_doubtful:
                excluded.append('DOUBTFUL')
            if args.filter_exclude_questionable:
                excluded.append('QUESTIONABLE')
            logger.info(f"Filter: exclude injury status {', '.join(excluded)}")

        if player_filters:
            logger.info(f"Total filters: {len(player_filters)}")
            logger.info("")

        logger.info("Initializing backtest with GPU-optimized configuration...")

        # Initialize GPU data pipeline if caching enabled
        gpu_pipeline = None
        if args.enable_gpu_caching:
            cache_dir = Path(args.cache_dir)
            gpu_pipeline = GPUDataPipeline(cache_dir=cache_dir)
            logger.info(f"GPU data pipeline initialized with cache dir: {cache_dir}")

        backtest = WalkForwardBacktest(
            db_path=args.db_path,
            train_start=train_start,
            train_end=train_end,
            test_start=args.test_start,
            test_end=args.test_end,
            model_type=args.model_type,
            model_params=model_params,
            feature_config=args.feature_config,
            output_dir=args.output_dir,
            data_dir=args.data_dir,
            per_player_models=args.per_player,
            min_player_games=args.min_player_games,
            recalibrate_days=args.recalibrate_days,
            num_seasons=args.num_seasons,
            save_models=args.save_models,
            save_predictions=args.save_predictions,
            n_jobs=args.n_jobs,
            rewrite_models=args.rewrite_models,
            resume_from_run=args.resume_from,
            player_filters=player_filters if player_filters else None,
            gpu_pipeline=gpu_pipeline,
            enable_feature_caching=enable_feature_caching,
            gpu_batch_size=args.gpu_batch_size
        )

        logger.info("Starting GPU-accelerated backtest...")
        logger.info("")

        results = backtest.run()

        if 'error' in results:
            logger.error(f"ERROR: {results['error']}")
            sys.exit(1)

        output_path = backtest.run_output_dir

        csv_path = output_path / f"backtest_results_{args.test_start}_to_{args.test_end}.csv"
        results['daily_results'].to_csv(csv_path, index=False)
        logger.info(f"Daily results saved to: {csv_path}")

        summary_path = output_path / f"summary_{args.test_start}_to_{args.test_end}.md"
        with open(summary_path, 'w') as f:
            f.write('# GPU-Accelerated Backtest Results\n\n')
            f.write('## Overview\n\n')
            f.write(f"**Date Range:** {results['date_range']}  \n")
            f.write(f"**Number of Slates:** {results['num_slates']}  \n")
            f.write(f"**Total Players Evaluated:** {results['total_players_evaluated']:.0f}  \n")
            f.write(f"**Average Players per Slate:** {results['avg_players_per_slate']:.1f}  \n\n")

            f.write('## Model Performance\n\n')
            f.write('| Metric | Value |\n')
            f.write('|--------|-------|\n')
            f.write(f"| Mean MAPE | {results['model_mean_mape']:.2f}% |\n")
            f.write(f"| Median MAPE | {results['model_median_mape']:.2f}% |\n")
            f.write(f"| Mean RMSE | {results['model_mean_rmse']:.2f} |\n")
            f.write(f"| Mean MAE | {results['model_mean_mae']:.2f} |\n")
            f.write(f"| Mean Correlation | {results['model_mean_correlation']:.3f} |\n\n")

            f.write('## Benchmark Performance\n\n')
            f.write('| Metric | Value |\n')
            f.write('|--------|-------|\n')
            f.write(f"| Mean MAPE | {results['benchmark_mean_mape']:.2f}% |\n\n")

            improvement = results['mape_improvement']
            status = 'Better' if improvement > 0 else 'Worse'
            f.write(f"**MAPE Improvement:** {improvement:+.2f}% ({status})  \n\n")

            if 'statistical_test' in results:
                f.write('## Statistical Significance\n\n')
                test = results['statistical_test']
                f.write('| Test | Value |\n')
                f.write('|------|-------|\n')
                f.write(f"| t-statistic | {test['t_statistic']:.4f} |\n")
                f.write(f"| p-value | {test['p_value']:.6f} |\n")
                f.write(f"| Cohen's d | {test['cohens_d']:.4f} |\n")
                f.write(f"| Effect size | {test['effect_size']} |\n\n")

            if 'tier_comparison' in results:
                f.write('## Performance by Salary Tier\n\n')
                tier_df = results['tier_comparison']
                f.write('| Tier | Count | Model MAPE | Benchmark MAPE | Improvement |\n')
                f.write('|------|-------|------------|----------------|--------------|\n')
                for _, row in tier_df.iterrows():
                    tier = str(row['salary_tier'])
                    count = int(row['count'])
                    model_mape = row['model_mape']
                    bench_mape = row['benchmark_mape']
                    imp = row['mape_improvement']
                    f.write(f"| {tier} | {count} | {model_mape:.1f}% | {bench_mape:.1f}% | {imp:+.1f}% |\n")

        logger.info(f"Summary saved to: {summary_path}")

        if 'tier_comparison' in results:
            tier_path = output_path / f"tier_comparison_{args.test_start}_to_{args.test_end}.csv"
            results['tier_comparison'].to_csv(tier_path, index=False)
            logger.info(f"Tier comparison saved to: {tier_path}")

        logger.info("")
        logger.info("="*80)
        logger.info("BACKTEST COMPLETE")
        logger.info("="*80)
        logger.info(f"Slates processed: {results.get('num_slates', 0)}")
        logger.info(f"Model MAPE: {results.get('model_mean_mape', 0):.2f}%")
        logger.info(f"Benchmark MAPE: {results.get('benchmark_mean_mape', 0):.2f}%")
        logger.info(f"Improvement: {results.get('mape_improvement', 0):+.2f}%")
        logger.info("")

        if 'report_url' in results:
            logger.info("REPORT READY - Click to open:")
            logger.info(results['report_url'])
        elif 'report_path' in results:
            from pathlib import Path
            report_file_url = Path(results['report_path']).absolute().as_uri()
            logger.info("REPORT READY - Click to open:")
            logger.info(report_file_url)

        logger.info("")
        logger.info(f"All outputs saved to: {output_path}")
        logger.info("="*80)

    except Exception as e:
        logger.error(f"Backtest failed with error: {str(e)}", exc_info=True)
        logger.error(f"Full traceback:\n{traceback.format_exc()}")
        sys.exit(1)

    finally:
        logger.info(f"Log files saved to: {log_dir}")
        logger.info(f"Main log: {log_file}")
        logger.info(f"Error log: {log_file.with_stem(log_file.stem + '_errors')}")


if __name__ == '__main__':
    main()
