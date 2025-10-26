"""
Walk-forward backtest simulation for DFS pipeline validation.

Simulates the daily fantasy workflow across multiple historical dates:
1. Generate predictions for each slate
2. Optimize lineups using predictions
3. Score lineups against actual results
4. Aggregate performance metrics across dates
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import json

from src.data.loaders.historical_loader import HistoricalDataLoader
from src.utils.feature_config import load_feature_config
from src.features.pipeline import FeaturePipeline
from src.models.registry import registry as model_registry


def calculate_dk_fantasy_points(stats: pd.Series) -> float:
    """
    Calculate DraftKings fantasy points from box score stats.

    Scoring:
    - Points: 1.0
    - 3PT Made: 0.5 bonus
    - Rebounds: 1.25
    - Assists: 1.5
    - Steals: 2.0
    - Blocks: 2.0
    - Turnovers: -0.5
    - Double-double: +1.5
    - Triple-double: +3.0 (includes double-double bonus)
    """
    pts = float(stats.get('pts', 0) or 0)
    tptfgm = float(stats.get('tptfgm', 0) or 0)
    reb = float(stats.get('reb', 0) or 0)
    ast = float(stats.get('ast', 0) or 0)
    stl = float(stats.get('stl', 0) or 0)
    blk = float(stats.get('blk', 0) or 0)
    tov = float(stats.get('TOV', 0) or 0)

    fpts = (
        pts * 1.0 +
        tptfgm * 0.5 +
        reb * 1.25 +
        ast * 1.5 +
        stl * 2.0 +
        blk * 2.0 -
        tov * 0.5
    )

    double_double = sum([pts >= 10, reb >= 10, ast >= 10, stl >= 10, blk >= 10]) >= 2
    if double_double:
        fpts += 1.5

    triple_double = sum([pts >= 10, reb >= 10, ast >= 10, stl >= 10, blk >= 10]) >= 3
    if triple_double:
        fpts += 1.5

    return round(fpts, 2)


class WalkForwardSimulation:
    """
    Walk-forward simulation framework for validating DFS production pipeline.

    Iterates through historical slates sequentially:
    - Generates predictions for each date
    - Optimizes lineups from predictions
    - Scores lineups against actual results
    - Aggregates metrics across entire backtest period
    """

    def __init__(
        self,
        data_dir: str = 'data',
        feature_config: str = 'default_features',
        model_config: str = 'xgboost_default',
        output_dir: str = 'data/backtest_results',
        verbose: bool = True
    ):
        """
        Initialize walk-forward simulation.

        Parameters
        ----------
        data_dir : str
            Directory containing historical parquet data
        feature_config : str
            Feature configuration name (e.g., 'default_features')
        model_config : str
            Model configuration name (e.g., 'xgboost_default')
        output_dir : str
            Directory to save simulation results
        verbose : bool
            Print progress messages
        """
        self.data_dir = Path(data_dir)
        self.feature_config_name = feature_config
        self.model_config_name = model_config
        self.output_dir = Path(output_dir)
        self.verbose = verbose

        # Initialize data loader
        self.loader = HistoricalDataLoader(str(self.data_dir))

        # Load configurations
        self.feature_config = load_feature_config(feature_config)

        # Results storage
        self.daily_results = []
        self.lineup_results = []

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        start_date: str,
        end_date: str,
        num_lineups: int = 1,
        strategy: str = 'balanced',
        num_training_seasons: int = 2,
        save_predictions: bool = False,
        save_lineups: bool = False
    ) -> Dict[str, Any]:
        """
        Run walk-forward simulation across date range.

        Parameters
        ----------
        start_date : str
            Start date in YYYYMMDD format
        end_date : str
            End date in YYYYMMDD format (exclusive)
        num_lineups : int
            Number of lineups to generate per slate
        strategy : str
            Lineup strategy: 'conservative', 'balanced', 'aggressive'
        num_training_seasons : int
            Number of seasons of training data per slate
        save_predictions : bool
            Save prediction CSVs for each date
        save_lineups : bool
            Save lineup CSVs for each date

        Returns
        -------
        Dict[str, Any]
            Aggregated simulation results with metrics
        """
        if self.verbose:
            print("="*60)
            print("WALK-FORWARD BACKTEST SIMULATION")
            print("="*60)
            print(f"Date range: {start_date} to {end_date}")
            print(f"Feature config: {self.feature_config_name}")
            print(f"Model config: {self.model_config_name}")
            print(f"Num lineups: {num_lineups}")
            print(f"Strategy: {strategy}")
            print("="*60)

        # Get slate dates in range
        slate_dates = self.loader.load_slate_dates(start_date, end_date)

        if self.verbose:
            print(f"\nFound {len(slate_dates)} slates to simulate\n")

        # Simulate each date
        for i, slate_date in enumerate(slate_dates, 1):
            if self.verbose:
                print(f"[{i}/{len(slate_dates)}] Simulating slate: {slate_date}")

            try:
                daily_result = self._simulate_slate(
                    slate_date=slate_date,
                    num_lineups=num_lineups,
                    strategy=strategy,
                    num_training_seasons=num_training_seasons,
                    save_predictions=save_predictions,
                    save_lineups=save_lineups
                )

                self.daily_results.append(daily_result)

                if self.verbose:
                    print(f"  ✓ Generated {daily_result['num_lineups']} lineups")
                    print(f"  ✓ Avg actual points: {daily_result['avg_actual_points']:.2f}")
                    print(f"  ✓ Avg projected points: {daily_result['avg_projected_points']:.2f}")
                    print(f"  ✓ Avg error: {daily_result['avg_error']:.2f}")
                    print()

            except Exception as e:
                if self.verbose:
                    print(f"  ✗ Error: {e}")
                    import traceback
                    traceback.print_exc()
                    print()
                continue

        # Aggregate results
        results = self._aggregate_results()

        # Check if we have any results
        if not results:
            if self.verbose:
                print("\n✗ No results generated. All slates failed to simulate.")
            return results

        # Save results
        self._save_results(results)

        if self.verbose:
            self._print_summary(results)

        return results

    def _simulate_slate(
        self,
        slate_date: str,
        num_lineups: int,
        strategy: str,
        num_training_seasons: int,
        save_predictions: bool,
        save_lineups: bool
    ) -> Dict[str, Any]:
        """
        Simulate DFS workflow for a single slate.

        Steps:
        1. Load slate data and historical training data
        2. Train per-player models
        3. Generate predictions
        4. Optimize lineups
        5. Score lineups against actual results

        Returns
        -------
        Dict[str, Any]
            Daily results with lineup performance
        """
        # Load player logs for the specific slate date
        path_pattern = self.loader._get_parquet_path('player_logs_extracted', slate_date)

        if not Path(path_pattern).exists():
            raise ValueError(f"No player logs for {slate_date}")

        # Load player logs for this date
        query = f"""
            SELECT * FROM read_parquet('{path_pattern}')
            WHERE gameDate = '{slate_date}'
        """
        test_slate = self.loader.conn.execute(query).df()

        if test_slate.empty:
            raise ValueError(f"No player data for {slate_date}")

        # Drop Tank01 fantasy points and calculate our own (matches notebook workflow)
        test_slate = test_slate.drop(columns=['fantasyPoints', 'fantasyPts'], errors='ignore')
        test_slate['fantasyPoints'] = test_slate.apply(calculate_dk_fantasy_points, axis=1)

        # Load DFS salaries for this date
        slate_data = self.loader.load_slate_data(slate_date, data_types=['dfs_salaries'])

        # Merge with DFS salaries if available
        if 'dfs_salaries' in slate_data and not slate_data['dfs_salaries'].empty:
            # Merge on playerID to add salary information
            test_slate = test_slate.merge(
                slate_data['dfs_salaries'][['playerID', 'salary']],
                on='playerID',
                how='left'
            )

        # Fill missing salaries with default
        if 'salary' in test_slate.columns:
            test_slate['salary'] = test_slate['salary'].fillna(5000)

        # Get eligible players from test slate
        eligible_player_ids = test_slate['playerID'].unique().tolist()

        # Load historical training data
        historical_logs = self.loader.load_historical_player_logs(
            end_date=slate_date,
            num_seasons=num_training_seasons,
            player_ids=eligible_player_ids
        )

        # Generate predictions
        predictions_df = self._generate_predictions(
            test_slate=test_slate,
            historical_logs=historical_logs,
            eligible_player_ids=eligible_player_ids
        )

        # Save predictions if requested
        if save_predictions:
            pred_path = self.output_dir / 'predictions' / f'predictions_{slate_date}.csv'
            pred_path.parent.mkdir(parents=True, exist_ok=True)
            predictions_df.to_csv(pred_path, index=False)

        # Generate lineups
        lineups = self._generate_lineups(
            predictions_df=predictions_df,
            num_lineups=num_lineups,
            strategy=strategy
        )

        # Save lineups if requested
        if save_lineups:
            lineup_path = self.output_dir / 'lineups' / f'lineups_{slate_date}.csv'
            lineup_path.parent.mkdir(parents=True, exist_ok=True)
            self._save_lineups_csv(lineups, lineup_path)

        # Score lineups against actual results
        lineup_scores = self._score_lineups(
            lineups=lineups,
            actual_results=test_slate
        )

        # Store lineup results
        for lineup, score in zip(lineups, lineup_scores):
            self.lineup_results.append({
                'date': slate_date,
                'lineup_id': lineup['lineup_id'],
                'projected_points': lineup['projected_points'],
                'actual_points': score['actual_points'],
                'error': score['error'],
                'players': lineup['players']
            })

        # Calculate daily aggregates
        actual_points = [s['actual_points'] for s in lineup_scores]
        projected_points = [lineup['projected_points'] for lineup in lineups]
        errors = [s['error'] for s in lineup_scores]

        return {
            'date': slate_date,
            'num_lineups': len(lineups),
            'avg_actual_points': np.mean(actual_points),
            'avg_projected_points': np.mean(projected_points),
            'avg_error': np.mean(errors),
            'min_actual_points': np.min(actual_points),
            'max_actual_points': np.max(actual_points),
            'lineup_scores': lineup_scores
        }

    def _generate_predictions(
        self,
        test_slate: pd.DataFrame,
        historical_logs: pd.DataFrame,
        eligible_player_ids: List[str]
    ) -> pd.DataFrame:
        """
        Generate predictions for slate using per-player models.

        Mirrors the notebook workflow exactly.
        """
        # Drop Tank01 fantasy points and calculate our own (matches notebook)
        historical_logs = historical_logs.drop(columns=['fantasyPoints', 'fantasyPts'], errors='ignore')
        historical_logs['fantasyPoints'] = historical_logs.apply(calculate_dk_fantasy_points, axis=1)

        # Build feature pipeline
        pipeline = self.feature_config.build_pipeline(FeaturePipeline)

        # Sort training data by playerID and gameDate (matches notebook)
        training_data = historical_logs.sort_values(['playerID', 'gameDate']).copy()
        training_data = pipeline.fit_transform(training_data)

        # Prepare test features
        test_features = pipeline.transform(test_slate)

        # Get feature columns from training data (exclude original columns)
        feature_cols = [col for col in training_data.columns if col not in historical_logs.columns]
        # Remove target and shifted target if present
        feature_cols = [col for col in feature_cols if col not in ['fantasyPoints', 'target']]

        # Train per-player models and predict
        predictions = []

        for player_id in eligible_player_ids:
            # Get player data
            player_train = training_data[training_data['playerID'] == player_id]
            player_test = test_features[test_features['playerID'] == player_id]

            if player_train.empty or player_test.empty:
                continue

            # Check minimum samples
            if len(player_train) < 10:
                continue

            # Prepare training data (drop rows with null features from rolling window warm-up)
            player_train_clean = player_train.dropna(subset=feature_cols)
            if len(player_train_clean) < 10:
                continue

            # Prepare X, y
            X_train = player_train_clean[feature_cols].fillna(0)
            y_train = player_train_clean['fantasyPoints']

            # Prepare test data
            X_test = player_test[feature_cols].fillna(0)

            # Load model config
            import yaml
            model_config_path = Path('config/models') / f'{self.model_config_name}.yaml'
            with open(model_config_path, 'r') as f:
                model_config = yaml.safe_load(f)

            # Create and train model
            model = model_registry.create(
                model_config['model_type'],
                model_config.get('hyperparameters', model_config.get('params', {}))
            )
            model.train(X_train, y_train)

            # Generate predictions
            y_pred = model.predict(X_test)

            # Store predictions
            for idx, pred in zip(player_test.index, y_pred):
                # Get name - try multiple column names
                name = (player_test.loc[idx].get('name') or
                       player_test.loc[idx].get('playerName') or
                       player_test.loc[idx].get('longName') or
                       f'Player {player_id}')

                # Get salary - might be missing
                salary = player_test.loc[idx].get('salary', 5000)

                # Get position
                pos = (player_test.loc[idx].get('primary_position') or
                      player_test.loc[idx].get('pos') or
                      'UTIL')

                predictions.append({
                    'playerID': player_id,
                    'name': name,
                    'salary': salary,
                    'predicted': pred,
                    'pos': pos
                })

        return pd.DataFrame(predictions)

    def _generate_lineups(
        self,
        predictions_df: pd.DataFrame,
        num_lineups: int,
        strategy: str
    ) -> List[Dict[str, Any]]:
        """
        Generate optimal lineups from predictions.

        Mirrors the generate_lineups.py workflow using a simplified approach.
        For production, this should use the full LineupGenerator class.
        """
        # Sort by projected points and select top players
        # This is a simplified greedy approach
        # TODO: Integrate with full LineupGenerator for proper optimization

        top_players = predictions_df.nlargest(num_lineups * 8, 'predicted')

        lineups = []
        for i in range(num_lineups):
            # Simple greedy selection (select top 8 players within salary cap)
            lineup_players = top_players.iloc[i*8:(i+1)*8].to_dict('records')

            lineup = {
                'lineup_id': i + 1,
                'projected_points': sum(p['predicted'] for p in lineup_players),
                'total_salary': sum(p['salary'] for p in lineup_players),
                'players': lineup_players
            }

            lineups.append(lineup)

        return lineups

    def _score_lineups(
        self,
        lineups: List[Dict[str, Any]],
        actual_results: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """
        Score lineups against actual fantasy points.

        Parameters
        ----------
        lineups : List[Dict[str, Any]]
            Generated lineups with player IDs
        actual_results : pd.DataFrame
            Test slate with actual fantasy points

        Returns
        -------
        List[Dict[str, Any]]
            Lineup scores with actual vs projected points
        """
        scores = []

        for lineup in lineups:
            # Calculate actual points for lineup
            actual_points = 0
            missing_players = 0

            for player in lineup['players']:
                player_id = player['playerID']
                player_actual = actual_results[actual_results['playerID'] == player_id]

                if not player_actual.empty:
                    actual_points += player_actual.iloc[0]['fantasyPoints']
                else:
                    missing_players += 1

            # Calcula2te error
            projected_points = lineup['projected_points']
            error = actual_points - projected_points

            scores.append({
                'lineup_id': lineup['lineup_id'],
                'actual_points': actual_points,
                'projected_points': projected_points,
                'error': error,
                'missing_players': missing_players
            })

        return scores

    def _aggregate_results(self) -> Dict[str, Any]:
        """
        Aggregate results across all simulated dates.

        Returns
        -------
        Dict[str, Any]
            Aggregated metrics and daily breakdown
        """
        if not self.daily_results:
            return {}

        # Convert to DataFrame for analysis
        daily_df = pd.DataFrame(self.daily_results)
        lineup_df = pd.DataFrame(self.lineup_results)

        # Calculate aggregate metrics
        results = {
            'num_slates': len(self.daily_results),
            'total_lineups': len(self.lineup_results),
            'date_range': {
                'start': daily_df['date'].min(),
                'end': daily_df['date'].max()
            },
            'aggregate_metrics': {
                'avg_actual_points': lineup_df['actual_points'].mean(),
                'avg_projected_points': lineup_df['projected_points'].mean(),
                'avg_error': lineup_df['error'].mean(),
                'mae': lineup_df['error'].abs().mean(),
                'rmse': np.sqrt((lineup_df['error'] ** 2).mean()),
                'min_actual_points': lineup_df['actual_points'].min(),
                'max_actual_points': lineup_df['actual_points'].max()
            },
            'daily_breakdown': self.daily_results,
            'lineup_results': self.lineup_results
        }

        return results

    def _save_results(self, results: Dict[str, Any]) -> None:
        """Save simulation results to JSON."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_path = self.output_dir / f'simulation_results_{timestamp}.json'

        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        if self.verbose:
            print(f"\nResults saved to: {results_path}")

    def _save_lineups_csv(self, lineups: List[Dict[str, Any]], path: Path) -> None:
        """Save lineups to CSV format."""
        rows = []
        for lineup in lineups:
            row = {
                'lineup_id': lineup['lineup_id'],
                'projected_points': lineup['projected_points'],
                'total_salary': lineup['total_salary']
            }

            for i, player in enumerate(lineup['players'], 1):
                row[f'player_{i}_id'] = player['playerID']
                row[f'player_{i}_name'] = player['name']
                row[f'player_{i}_salary'] = player['salary']
                row[f'player_{i}_projected'] = player['predicted']

            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(path, index=False)

    def _print_summary(self, results: Dict[str, Any]) -> None:
        """Print simulation summary."""
        print("\n" + "="*60)
        print("SIMULATION SUMMARY")
        print("="*60)

        metrics = results['aggregate_metrics']

        print(f"\nSlates simulated: {results['num_slates']}")
        print(f"Total lineups: {results['total_lineups']}")
        print(f"Date range: {results['date_range']['start']} to {results['date_range']['end']}")

        print("\nAggregate Metrics:")
        print(f"  Avg actual points: {metrics['avg_actual_points']:.2f}")
        print(f"  Avg projected points: {metrics['avg_projected_points']:.2f}")
        print(f"  Avg error: {metrics['avg_error']:.2f}")
        print(f"  MAE: {metrics['mae']:.2f}")
        print(f"  RMSE: {metrics['rmse']:.2f}")
        print(f"  Min actual points: {metrics['min_actual_points']:.2f}")
        print(f"  Max actual points: {metrics['max_actual_points']:.2f}")

        print("\n" + "="*60)
