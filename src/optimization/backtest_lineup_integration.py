"""
Integration module for adding lineup generation to walk-forward backtesting
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime

from .lineup_generator import LineupGenerator
from .contest_manager import ContestManager
from ..evaluation.backtest import WalkForwardBacktest

logger = logging.getLogger(__name__)


class BacktestWithLineups(WalkForwardBacktest):
    """
    Extended WalkForwardBacktest with lineup generation and evaluation
    """

    def __init__(self,
                 *args,
                 generate_lineups: bool = True,
                 contest_config: Optional[str] = None,
                 num_lineups: int = 10,
                 track_lineup_performance: bool = True,
                 **kwargs):
        """
        Initialize backtest with lineup generation.

        Parameters
        ----------
        generate_lineups : bool
            Whether to generate lineups after predictions
        contest_config : Optional[str]
            Path to contest configuration
        num_lineups : int
            Number of lineups to generate per slate
        track_lineup_performance : bool
            Whether to track lineup performance
        """
        super().__init__(*args, **kwargs)

        self.generate_lineups = generate_lineups
        self.contest_config = contest_config
        self.num_lineups = num_lineups
        self.track_lineup_performance = track_lineup_performance

        if self.generate_lineups:
            self.lineup_generator = LineupGenerator(contest_config)
            self.contest_manager = ContestManager()
            self.lineup_results = []

    def _process_slate_with_lineups(self,
                                     test_date: str,
                                     projections: pd.DataFrame,
                                     actuals: pd.DataFrame) -> Dict[str, Any]:
        """
        Process slate with lineup generation and evaluation.

        Parameters
        ----------
        test_date : str
            Test date
        projections : pd.DataFrame
            Player projections
        actuals : pd.DataFrame
            Actual results

        Returns
        -------
        Dict[str, Any]
            Slate results with lineup performance
        """
        # Get base slate evaluation
        slate_results, merged_df = self._evaluate_slate(test_date, projections, actuals)

        if not self.generate_lineups:
            return slate_results, merged_df

        # Generate lineups
        logger.info(f"Generating {self.num_lineups} lineups for {test_date}")

        try:
            # Prepare predictions for lineup generator
            # Start with required columns
            required_cols = ['playerID', 'playerName', 'team', 'pos', 'projected_fpts', 'salary']

            # Add optional columns if they exist
            optional_cols = ['status', 'is_out', 'is_doubtful', 'is_questionable', 'is_injured',
                           'injury_status', 'injury_designation', 'gameInfo']

            cols_to_copy = [col for col in required_cols if col in projections.columns]
            cols_to_copy.extend([col for col in optional_cols if col in projections.columns])

            predictions_df = projections[cols_to_copy].copy()
            predictions_df = predictions_df.rename(columns={
                'projected_fpts': 'predicted_fpts',
                'pos': 'position'
            })

            # Add default injury columns if they don't exist (to satisfy InjuryFilter)
            if 'is_out' not in predictions_df.columns:
                predictions_df['is_out'] = 0
            if 'is_doubtful' not in predictions_df.columns:
                predictions_df['is_doubtful'] = 0
            if 'is_questionable' not in predictions_df.columns:
                predictions_df['is_questionable'] = 0
            if 'is_injured' not in predictions_df.columns:
                predictions_df['is_injured'] = 0
            if 'injury_status' not in predictions_df.columns:
                predictions_df['injury_status'] = 'Healthy'
            if 'status' not in predictions_df.columns:
                predictions_df['status'] = None

            # Generate lineups
            lineups = self.lineup_generator.generate_lineups(
                predictions_df=predictions_df,
                dfs_salaries_df=None  # Already included in predictions
            )

            # Evaluate lineup performance if actuals available
            if self.track_lineup_performance and not actuals.empty:
                lineup_performance = self._evaluate_lineups(lineups, actuals, test_date)
                slate_results['lineup_performance'] = lineup_performance
                self.lineup_results.append(lineup_performance)

            # Save lineups
            self._save_lineups(test_date, lineups)

            slate_results['lineups_generated'] = len(lineups)

        except Exception as e:
            logger.error(f"Failed to generate lineups for {test_date}: {e}")
            slate_results['lineups_generated'] = 0

        return slate_results, merged_df

    def _evaluate_lineups(self,
                          lineups: List[Dict],
                          actuals: pd.DataFrame,
                          test_date: str) -> Dict[str, Any]:
        """
        Evaluate lineup performance against actual results.

        Parameters
        ----------
        lineups : List[Dict]
            Generated lineups
        actuals : pd.DataFrame
            Actual player performance
        test_date : str
            Test date

        Returns
        -------
        Dict[str, Any]
            Lineup performance metrics
        """
        lineup_scores = []
        player_actuals = {row['playerID']: row['actual_fpts']
                          for _, row in actuals.iterrows()}

        for lineup in lineups:
            actual_score = 0
            projected_score = lineup['projected_points']

            for player in lineup['players']:
                player_id = player['playerID']
                if player_id in player_actuals:
                    actual_score += player_actuals[player_id]

            lineup_scores.append({
                'lineup_num': lineup['lineup_num'],
                'projected_score': projected_score,
                'actual_score': actual_score,
                'difference': actual_score - projected_score,
                'error_pct': abs(actual_score - projected_score) / projected_score * 100 if projected_score > 0 else 0
            })

        # Calculate aggregate metrics
        scores_df = pd.DataFrame(lineup_scores)

        performance = {
            'date': test_date,
            'num_lineups': len(lineups),
            'avg_projected_score': scores_df['projected_score'].mean(),
            'avg_actual_score': scores_df['actual_score'].mean(),
            'best_projected': scores_df['projected_score'].max(),
            'best_actual': scores_df['actual_score'].max(),
            'worst_actual': scores_df['actual_score'].min(),
            'avg_error_pct': scores_df['error_pct'].mean(),
            'correlation': scores_df['projected_score'].corr(scores_df['actual_score']),
            'lineup_scores': lineup_scores
        }

        # Check if optimal lineup would have cashed
        if len(scores_df) > 0:
            best_lineup_idx = scores_df['projected_score'].idxmax()
            best_lineup_actual = scores_df.loc[best_lineup_idx, 'actual_score']
            performance['best_lineup_rank'] = (scores_df['actual_score'] >= best_lineup_actual).sum()

        logger.info(f"Lineup Performance - Avg Projected: {performance['avg_projected_score']:.1f}, "
                    f"Avg Actual: {performance['avg_actual_score']:.1f}, "
                    f"Correlation: {performance['correlation']:.3f}")

        return performance

    def _save_lineups(self, test_date: str, lineups: List[Dict]):
        """
        Save generated lineups to disk.

        Parameters
        ----------
        test_date : str
            Test date
        lineups : List[Dict]
            Generated lineups
        """
        lineup_dir = self.results_dir / 'lineups'
        lineup_dir.mkdir(exist_ok=True)

        # Save as CSV for DraftKings upload
        csv_path = lineup_dir / f"{test_date}_lineups.csv"
        self.lineup_generator.export_lineups(lineups, str(csv_path), format='csv')

        # Save as JSON with full details
        json_path = lineup_dir / f"{test_date}_lineups.json"
        self.lineup_generator.export_lineups(lineups, str(json_path), format='json')

        logger.info(f"Saved {len(lineups)} lineups to {lineup_dir}")

    def generate_lineup_report(self) -> pd.DataFrame:
        """
        Generate summary report of lineup performance across all slates.

        Returns
        -------
        pd.DataFrame
            Lineup performance report
        """
        if not self.lineup_results:
            logger.warning("No lineup results to report")
            return pd.DataFrame()

        report_data = []

        for slate_performance in self.lineup_results:
            report_data.append({
                'date': slate_performance['date'],
                'lineups': slate_performance['num_lineups'],
                'avg_projected': slate_performance['avg_projected_score'],
                'avg_actual': slate_performance['avg_actual_score'],
                'best_actual': slate_performance['best_actual'],
                'worst_actual': slate_performance['worst_actual'],
                'correlation': slate_performance['correlation'],
                'avg_error_pct': slate_performance['avg_error_pct']
            })

        report_df = pd.DataFrame(report_data)

        # Add summary statistics
        if len(report_df) > 0:
            logger.info("\n" + "="*60)
            logger.info("LINEUP GENERATION SUMMARY")
            logger.info("="*60)
            logger.info(f"Total Slates: {len(report_df)}")
            logger.info(f"Total Lineups: {report_df['lineups'].sum()}")
            logger.info(f"Avg Correlation: {report_df['correlation'].mean():.3f}")
            logger.info(f"Avg Error: {report_df['avg_error_pct'].mean():.1f}%")
            logger.info(f"Best Score: {report_df['best_actual'].max():.1f}")

        return report_df

    def run_with_lineups(self) -> Dict[str, Any]:
        """
        Run backtest with lineup generation.

        Returns
        -------
        Dict[str, Any]
            Backtest results with lineup performance
        """
        # Run base backtest
        results = self.run()

        # Add lineup performance if generated
        if self.generate_lineups and self.lineup_results:
            lineup_report = self.generate_lineup_report()
            results['lineup_performance'] = lineup_report
            results['lineup_summary'] = {
                'total_lineups': lineup_report['lineups'].sum() if len(lineup_report) > 0 else 0,
                'avg_correlation': lineup_report['correlation'].mean() if len(lineup_report) > 0 else 0,
                'avg_error_pct': lineup_report['avg_error_pct'].mean() if len(lineup_report) > 0 else 0
            }

            # Save lineup report
            report_path = self.results_dir / 'lineup_performance_report.csv'
            lineup_report.to_csv(report_path, index=False)
            logger.info(f"Saved lineup performance report to {report_path}")

        return results


def create_lineup_optimizer(optimizer_type: str = 'pydfs', **kwargs) -> Any:
    """
    Factory function to create lineup optimizers.

    Parameters
    ----------
    optimizer_type : str
        Type of optimizer ('pydfs', 'custom')
    **kwargs
        Additional arguments for optimizer

    Returns
    -------
    Any
        Lineup optimizer instance
    """
    if optimizer_type == 'pydfs':
        from .lineup_generator import LineupGenerator
        return LineupGenerator(**kwargs)
    elif optimizer_type == 'custom':
        from .optimizers.linear_program import LinearProgramOptimizer
        from .constraints.draftkings import DraftKingsConstraints
        constraints = [DraftKingsConstraints()]
        return LinearProgramOptimizer(constraints, **kwargs)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")