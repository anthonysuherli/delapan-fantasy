"""
Base class for backtest training strategies.

This module provides the abstract base class for different
training strategies used in walk-forward backtesting.
"""

import pandas as pd
import numpy as np
import logging
import pickle
import time
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from abc import ABC, abstractmethod

from src.evaluation.benchmarks.season_average import SeasonAverageBenchmark
from src.utils.fantasy_points import calculate_dk_fantasy_points
from src.evaluation.report_generator import BacktestReportGenerator
from src.evaluation.pdf_report_generator import PDFStyleBacktestReportGenerator

logger = logging.getLogger(__name__)


class BacktestTrainer(ABC):
    """
    Abstract base class for backtest training strategies.

    This class provides the common infrastructure for different
    training approaches (per-player, per-slate, etc.) and defines
    the interface that concrete trainers must implement.
    """

    def __init__(self, backtest):
        """
        Initialize trainer.

        Args:
            backtest: Parent WalkForwardBacktest instance
        """
        self.backtest = backtest
        self.results = []
        self.all_predictions = []
        self.benchmark = None

    @abstractmethod
    def train_models(
        self,
        training_data: pd.DataFrame,
        slate_data: Dict[str, Any],
        test_date: str
    ) -> Dict[str, Any]:
        """
        Train models for the given slate.

        Args:
            training_data: Historical training data
            slate_data: Current slate data
            test_date: Test date

        Returns:
            Dictionary of trained models or model metadata
        """
        pass

    @abstractmethod
    def generate_projections(
        self,
        slate_data: Dict[str, Any],
        training_data: pd.DataFrame,
        test_date: str
    ) -> pd.DataFrame:
        """
        Generate projections for the slate.

        Args:
            slate_data: Current slate data
            training_data: Historical training data
            test_date: Test date

        Returns:
            DataFrame with projections
        """
        pass

    def run(self) -> Dict[str, Any]:
        """
        Execute walk-forward backtest.

        Returns:
            Dictionary containing aggregated results and metrics
        """
        from datetime import datetime as dt

        # Simple timing instead of profiler
        backtest_start_time = time.perf_counter()

        if self.backtest.resume_from_run:
            self.backtest.run_timestamp = self.backtest.resume_from_run
            logger.info(f"RESUMING existing run: {self.backtest.run_timestamp}")
        else:
            self.backtest.run_timestamp = dt.now().strftime('%Y%m%d_%H%M%S')
            logger.info(f"Starting NEW run: {self.backtest.run_timestamp}")

        # Setup output directories
        self._setup_directories()

        # Log backtest start
        logger.info("="*80)
        logger.info("STARTING WALK-FORWARD BACKTEST")
        logger.info("="*80)
        logger.info(f"Run timestamp: {self.backtest.run_timestamp}")
        logger.info(f"Output directory: {self.backtest.run_output_dir}")
        logger.info(f"Training period: {self.backtest.train_start} to {self.backtest.train_end}")
        logger.info(f"Testing period: {self.backtest.test_start} to {self.backtest.test_end}")
        logger.info(f"Model: {self.backtest.model_type}")
        logger.info("="*80)

        # Load checkpoint if resuming
        completed_slates = self._load_checkpoint()
        if completed_slates:
            logger.info(f"RESUMING from checkpoint: {len(completed_slates)} slates already completed")
            logger.info(f"Completed dates: {sorted(completed_slates)}")
            logger.info("="*80)

        # Get slate dates
        slate_dates = self.backtest.loader.load_slate_dates(
            self.backtest.test_start,
            self.backtest.test_end
        )

        if not slate_dates:
            logger.error("No slate dates found")
            return {'error': 'No slate dates found'}

        print(f"\nBacktesting {len(slate_dates)} slates from {self.backtest.test_start} to {self.backtest.test_end}\n")

        # Pre-scan for filtered players if needed
        self._pre_scan_slates(slate_dates)

        # Initialize benchmark
        self._initialize_benchmark()

        # Main backtest loop
        for test_date in tqdm(slate_dates, desc="Backtesting slates", leave=False):
            # Skip if already completed
            if test_date in completed_slates:
                logger.info(f"Skipping {test_date} (already completed)")

                # Load checkpoint data
                checkpoint_data = self._load_slate_checkpoint(test_date)
                if checkpoint_data:
                    self.results.append(checkpoint_data['results'])
                    if 'predictions' in checkpoint_data:
                        self.all_predictions.append(checkpoint_data['predictions'])
                continue

            logger.info(f"\n{'='*80}")
            logger.info(f"Processing slate: {test_date}")
            logger.info(f"{'='*80}")

            try:
                # Process single slate
                daily_results, predictions_df = self._process_slate(test_date)

                # Store results
                self.results.append(daily_results)
                if self.backtest.save_predictions and not predictions_df.empty:
                    self.all_predictions.append(predictions_df)

                # Save checkpoint
                self._save_slate_checkpoint(test_date, daily_results, predictions_df)

                # Interactive mode
                if self.backtest.interactive:
                    action = self._interactive_pause(test_date, daily_results)
                    if action == 'stop':
                        logger.info("Stopping backtest at user request")
                        break

            except Exception as e:
                logger.error(f"Error processing slate {test_date}: {str(e)}")
                continue

        # Log cache stats if enabled
        if self.backtest.enable_feature_caching:
            self._log_cache_stats()

        # Aggregate results
        aggregated = self._aggregate_results()

        # Generate reports
        self._generate_reports(aggregated)

        # Log completion
        backtest_elapsed = time.perf_counter() - backtest_start_time
        logger.info("\n" + "="*80)
        logger.info("BACKTEST COMPLETE")
        logger.info("="*80)
        logger.info(f"Total runtime: {self.backtest._format_time(backtest_elapsed)}")
        logger.info(f"Output directory: {self.backtest.run_output_dir}")
        logger.info("="*80)

        return aggregated

    def _setup_directories(self):
        """Setup output directories for the run."""
        if self.backtest.data_dir:
            data_path = Path(self.backtest.data_dir)
            output_path_obj = Path(self.backtest.output_dir)
            if not output_path_obj.is_absolute():
                base_output = data_path / self.backtest.output_dir
            else:
                base_output = Path(self.backtest.output_dir)
            self.backtest.run_output_dir = base_output / self.backtest.run_timestamp
        else:
            self.backtest.run_output_dir = Path('data') / 'outputs' / self.backtest.run_timestamp

        self.backtest.run_inputs_dir = self.backtest.run_output_dir / 'inputs'
        self.backtest.run_features_dir = self.backtest.run_output_dir / 'features'
        self.backtest.run_predictions_dir = self.backtest.run_output_dir / 'predictions'
        self.backtest.run_checkpoint_dir = self.backtest.run_output_dir / 'checkpoints'

        self.backtest.run_inputs_dir.mkdir(parents=True, exist_ok=True)
        self.backtest.run_features_dir.mkdir(parents=True, exist_ok=True)
        self.backtest.run_predictions_dir.mkdir(parents=True, exist_ok=True)
        self.backtest.run_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _expand_nested_salaries(self, salaries_df: pd.DataFrame) -> pd.DataFrame:
        """
        Expand nested JSON salaries data to flat DataFrame.

        If salaries_df still has nested structure (draftkings/fanduel columns),
        extract and flatten to player records with playerID.
        """
        # Check if data needs expansion
        if 'draftkings' not in salaries_df.columns:
            # Already expanded or invalid structure
            return salaries_df

        try:
            dk_data = salaries_df['draftkings'].iloc[0]
            if not isinstance(dk_data, (list, tuple)):
                return salaries_df

            expanded_data = []
            for player in dk_data:
                if isinstance(player, dict):
                    player_data = player.copy()

                    # Normalize playerID column
                    if 'playerID' not in player_data:
                        for alias in ('playerId', 'player_id', 'playerid', 'id'):
                            if alias in player_data:
                                player_data['playerID'] = str(player_data.get(alias))
                                break
                    else:
                        player_data['playerID'] = str(player_data.get('playerID'))

                    # Normalize player name
                    if 'playerName' not in player_data:
                        for name_alias in ('longName', 'fullName', 'name'):
                            if name_alias in player_data:
                                player_data['playerName'] = player_data.get(name_alias)
                                break

                    expanded_data.append(player_data)

            if expanded_data:
                return pd.DataFrame(expanded_data)

        except Exception as e:
            logger.warning(f"Failed to expand nested salaries data: {e}")

        return salaries_df

    def _pre_scan_slates(self, slate_dates: List[str]):
        """
        Pre-scan slates to identify filtered players.

        Args:
            slate_dates: List of slate dates to scan
        """
        logger.info("="*80)
        logger.info("PRE-SCANNING SLATES FOR PLAYER FILTERING")
        logger.info("="*80)

        filtered_player_ids = set()

        if self.backtest.player_filters:
            logger.info(f"Pre-scanning {len(slate_dates)} slates to identify filtered players...")

            for test_date in tqdm(slate_dates, desc="Scanning slates", leave=False):
                slate_data = self.backtest.loader.load_slate_data(test_date)
                salaries_df = slate_data.get('dfs_salaries', pd.DataFrame())

                if salaries_df.empty:
                    continue

                # Ensure salaries data is expanded from nested structure
                salaries_df = self._expand_nested_salaries(salaries_df)

                if salaries_df.empty or 'playerID' not in salaries_df.columns:
                    logger.debug(f"No valid salaries data for {test_date}, skipping")
                    continue

                # (No direct analog to the training period calculation and backtest instantiation
                # in this pre-scan method. The original logic is to apply injury features and filters.)
                injuries_data = slate_data.get('injuries', pd.DataFrame())
                from src.features.transformers.injury import InjuryTransformer
                injury_transformer = InjuryTransformer()
                injury_transformer.fit(salaries_df)
                salaries_df = injury_transformer.transform(salaries_df, injuries_data)

                logger.info(f'filters : {self.backtest.player_filters}')
                # Apply filters (each may further reduce the player pool)
                for pf in self.backtest.player_filters:
                    filtered = pf.apply(salaries_df)
                    if not filtered.empty:
                        salaries_df = filtered

                # Collect player IDs
                if 'playerID' in salaries_df.columns:
                    filtered_player_ids.update(salaries_df['playerID'].unique())

            logger.info(f"Found {len(filtered_player_ids)} unique players across all slates after filtering")
            filtered_player_ids = list(filtered_player_ids)
        else:
            logger.info("No player filters configured - will load all players")
            filtered_player_ids = None

        # Store for use in cached loading
        self.backtest.filtered_player_ids = filtered_player_ids

    def _initialize_benchmark(self):
        """Initialize benchmark model."""
        # Load initial training data
        logger.info("="*80)
        logger.info("LOADING TRAINING DATA FOR BENCHMARK")
        logger.info("="*80)

        if self.backtest.enable_feature_caching:
            logger.info(f"Loading from cache with filtered_player_ids={self.backtest.filtered_player_ids}")
            training_data = self.backtest._load_training_data_cached(self.backtest.filtered_player_ids)
        else:
            if self.backtest.filtered_player_ids:
                logger.info(f"Loading {len(self.backtest.filtered_player_ids)} filtered players from {self.backtest.train_end} back {self.backtest.num_seasons} seasons")
                training_data = self.backtest.loader.load_historical_player_logs(
                    end_date=self.backtest.train_end,
                    num_seasons=self.backtest.num_seasons,
                    player_ids=self.backtest.filtered_player_ids
                )
            else:
                logger.info(f"Loading all players from {self.backtest.train_end} back {self.backtest.num_seasons} seasons")
                training_data = self.backtest.loader.load_historical_player_logs(
                    end_date=self.backtest.train_end,
                    num_seasons=self.backtest.num_seasons
                )

        logger.info(f"Training data loaded: shape={training_data.shape}, columns={list(training_data.columns) if not training_data.empty else 'EMPTY'}")

        if training_data.empty:
            logger.warning("No training data loaded - benchmark will be skipped")
            self.benchmark = None
            return

        # Validate required columns
        if 'playerID' not in training_data.columns:
            logger.error(f"ERROR: Training data missing 'playerID' column!")
            logger.error(f"Available columns: {list(training_data.columns)}")
            logger.error(f"Training data:\n{training_data.head()}")
            self.benchmark = None
            return

        # Initialize benchmark
        self.benchmark = SeasonAverageBenchmark(
            min_games=self.backtest.min_games_for_benchmark
        )

        benchmark_data = training_data.copy()
        if 'fpts' not in benchmark_data.columns:
            benchmark_data['fpts'] = benchmark_data.apply(calculate_dk_fantasy_points, axis=1)

        self.benchmark.fit(benchmark_data)
        logger.info(f"Benchmark fitted with {len(benchmark_data)} player-games from {len(benchmark_data['playerID'].unique())} unique players")

    def _process_slate(self, test_date: str) -> Tuple[Dict[str, Any], pd.DataFrame]:
        """
        Process a single slate.

        Args:
            test_date: Date of the slate

        Returns:
            Tuple of (results dictionary, predictions dataframe)
        """
        slate_start_time = time.perf_counter()

        # Load slate data
        slate_data = self.backtest.loader.load_slate_data(test_date)

        if slate_data.get('dfs_salaries', pd.DataFrame()).empty:
            logger.warning(f"No DFS salaries found for {test_date}")
            return {}, pd.DataFrame()

        slate_data['date'] = test_date

        # Load training data
        if self.backtest.enable_feature_caching:
            training_data = self.backtest._load_training_data_cached(self.backtest.filtered_player_ids)
        else:
            if self.backtest.filtered_player_ids:
                training_data = self.backtest.loader.load_historical_player_logs(
                    end_date=test_date,
                    num_seasons=self.backtest.num_seasons,
                    player_ids=self.backtest.filtered_player_ids
                )
            else:
                training_data = self.backtest.loader.load_historical_player_logs(
                    end_date=test_date,
                    num_seasons=self.backtest.num_seasons
                )

        if training_data.empty:
            logger.warning(f"No training data available for {test_date}")
            return {}, pd.DataFrame()

        # Train models if needed
        models = self.train_models(training_data, slate_data, test_date)

        # Generate projections
        projections_df = self.generate_projections(slate_data, training_data, test_date)

        if projections_df.empty:
            logger.warning(f"No projections generated for {test_date}")
            return {}, pd.DataFrame()

        # Evaluate slate
        results, merged_df = self.backtest._evaluate_slate(
            projections_df,
            test_date,
            'Model'
        )

        # Add benchmark comparison if available
        if self.benchmark:
            benchmark_projections = self.benchmark.predict(slate_data['dfs_salaries'])
            if not benchmark_projections.empty:
                benchmark_results, _ = self.backtest._evaluate_slate(
                    benchmark_projections,
                    test_date,
                    'Benchmark'
                )
                results['benchmark'] = benchmark_results

        slate_elapsed = time.perf_counter() - slate_start_time
        results['runtime'] = slate_elapsed
        logger.info(f"Slate {test_date} completed in {self.backtest._format_time(slate_elapsed)}")

        return results, merged_df

    def _load_checkpoint(self) -> set:
        """Load completed slates from checkpoint."""
        checkpoint_file = self.backtest.run_checkpoint_dir / 'completed_slates.pkl'
        if checkpoint_file.exists():
            try:
                with open(checkpoint_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Error loading checkpoint: {e}")
        return set()

    def _save_slate_checkpoint(self, test_date: str, daily_results: Dict[str, Any], merged_df: pd.DataFrame):
        """Save slate checkpoint."""
        # Save completed slates list
        completed_slates = self._load_checkpoint()
        completed_slates.add(test_date)

        checkpoint_file = self.backtest.run_checkpoint_dir / 'completed_slates.pkl'
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(completed_slates, f)

        # Save slate-specific data
        slate_checkpoint = {
            'date': test_date,
            'results': daily_results,
            'predictions': merged_df if self.backtest.save_predictions else pd.DataFrame()
        }

        slate_file = self.backtest.run_checkpoint_dir / f'slate_{test_date}.pkl'
        with open(slate_file, 'wb') as f:
            pickle.dump(slate_checkpoint, f)

    def _load_slate_checkpoint(self, test_date: str) -> Optional[Dict[str, Any]]:
        """Load slate checkpoint."""
        slate_file = self.backtest.run_checkpoint_dir / f'slate_{test_date}.pkl'
        if slate_file.exists():
            try:
                with open(slate_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Error loading slate checkpoint for {test_date}: {e}")
        return None

    def _log_cache_stats(self):
        """Log cache statistics."""
        total_accesses = (
            self.backtest.cache_stats['feature_cache_hits'] +
            self.backtest.cache_stats['feature_cache_misses']
        )

        if total_accesses > 0:
            hit_rate = self.backtest.cache_stats['feature_cache_hits'] / total_accesses * 100
            logger.info("="*80)
            logger.info("CACHE STATISTICS")
            logger.info("="*80)
            logger.info(f"Feature cache hits: {self.backtest.cache_stats['feature_cache_hits']}")
            logger.info(f"Feature cache misses: {self.backtest.cache_stats['feature_cache_misses']}")
            logger.info(f"Feature cache hit rate: {hit_rate:.1f}%")
            logger.info(f"Training data loads: {self.backtest.cache_stats['training_data_loads']}")

    def _aggregate_results(self) -> Dict[str, Any]:
        """Aggregate daily results into final metrics."""
        if not self.results:
            return {
                'error': 'No results to aggregate',
                'config': self.backtest.config
            }

        # Convert results to DataFrame for easier analysis
        # Handle nested 'benchmark' dict by excluding it initially
        results_for_df = []
        for result in self.results:
            # Extract top-level metrics, excluding nested dicts
            row = {k: v for k, v in result.items() if not isinstance(v, dict)}
            results_for_df.append(row)
        
        results_df = pd.DataFrame(results_for_df)

        # Filter out invalid results (handle missing num_players column)
        if 'num_players' not in results_df.columns:
            logger.warning("No 'num_players' column found in results")
            return {
                'error': 'No num_players column in results',
                'config': self.backtest.config
            }
        
        valid_results = results_df[results_df['num_players'] > 0].copy()

        if valid_results.empty:
            return {
                'error': 'No valid results',
                'config': self.backtest.config
            }

        # Calculate aggregate metrics
        num_slates = len(valid_results)
        total_players = int(valid_results['num_players'].sum())
        
        aggregated = {
            'num_slates': num_slates,
            'total_players_evaluated': total_players,
            # Backward compatibility aliases
            'test_slates': num_slates,
            'total_players': total_players,
            # Other metrics
            'avg_players_per_slate': valid_results['num_players'].mean(),
            'model_mean_mape': valid_results['model_mape'].mean(),
            'model_median_mape': valid_results['model_mape'].median(),
            'model_std_mape': valid_results['model_mape'].std(),
            'model_mean_rmse': valid_results['model_rmse'].mean(),
            'model_std_rmse': valid_results['model_rmse'].std(),
            'model_mean_mae': valid_results['model_mae'].mean(),
            'model_mean_correlation': valid_results['model_corr'].mean(),
            'model_std_correlation': valid_results['model_corr'].std(),
            'daily_results': valid_results.reset_index(drop=True),  # Convert to clean DataFrame
            'config': self.backtest.config,
            'date_range': f"{self.backtest.test_start} to {self.backtest.test_end}",
            'output_dir': str(self.backtest.run_output_dir)
        }

        # Add benchmark metrics if available (from original results, not DataFrame)
        benchmark_results = [r.get('benchmark') for r in self.results if 'benchmark' in r and isinstance(r['benchmark'], dict)]
        if benchmark_results:
            bench_mapes = [b.get('model_mape', np.nan) for b in benchmark_results]
            if bench_mapes and not all(np.isnan(bench_mapes)):
                aggregated['benchmark_mean_mape'] = np.nanmean(bench_mapes)
                aggregated['benchmark_median_mape'] = np.nanmedian(bench_mapes)
                aggregated['mape_improvement'] = aggregated['model_mean_mape'] - aggregated['benchmark_mean_mape']

        return aggregated

    def _generate_reports(self, aggregated: Dict[str, Any]):
        """Generate backtest reports."""
        # Save configuration
        config_file = self.backtest.run_output_dir / 'config.yaml'
        import yaml
        with open(config_file, 'w') as f:
            yaml.dump(self.backtest.config, f, default_flow_style=False)

        # Generate reports if we have results
        if self.results:
            try:
                # HTML report
                report_gen = BacktestReportGenerator(self.backtest.run_output_dir)
                report_path = report_gen.generate_report(
                    self.results,
                    self.backtest.config,
                    self.backtest.run_timestamp
                )
                logger.info(f"HTML report saved to {report_path}")

                # PDF-style report
                try:
                    pdf_gen = PDFStyleBacktestReportGenerator(self.backtest.run_output_dir)
                    pdf_path = pdf_gen.generate_report(
                        self.results,
                        self.backtest.config,
                        self.backtest.run_timestamp,
                        {}  # Empty chart_paths dict
                    )
                    logger.info(f"PDF-style report saved to {pdf_path}")
                except Exception as e:
                    logger.warning(f"Could not generate PDF report: {e}")

            except Exception as e:
                logger.error(f"Error generating reports: {e}")

        # Save predictions if configured
        if self.backtest.save_predictions and self.all_predictions:
            all_predictions_df = pd.concat(self.all_predictions, ignore_index=True)
            predictions_file = self.backtest.run_predictions_dir / 'all_predictions.parquet'
            all_predictions_df.to_parquet(predictions_file)
            logger.info(f"Saved {len(all_predictions_df)} predictions to {predictions_file}")

    def _interactive_pause(self, test_date: str, daily_results: Dict[str, Any]) -> str:
        """Handle interactive mode pause."""
        return self.backtest._interactive_pause(test_date, daily_results)