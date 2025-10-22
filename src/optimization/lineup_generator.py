"""
Lineup Generator using pydfs-lineup-optimizer
Integrates with existing prediction pipeline to generate contest-specific lineups
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import pandas as pd
import numpy as np
from datetime import datetime

from pydfs_lineup_optimizer import (
    get_optimizer,
    Site,
    Sport,
    Player,
    Lineup,
    PlayerFilter,
    LineupOptimizer
)
from pydfs_lineup_optimizer.fantasy_points_strategy import BaseFantasyPointsStrategy
from pydfs_lineup_optimizer.stacks import PlayersGroup, TeamStack, PositionsStack

logger = logging.getLogger(__name__)


class CustomFantasyPointsStrategy(BaseFantasyPointsStrategy):
    """
    Custom strategy for calculating fantasy points based on our predictions
    """

    def __init__(self, risk_settings: Dict[str, float]):
        """
        Initialize custom fantasy points strategy.

        Parameters
        ----------
        risk_settings : Dict[str, float]
            Weights for variance, ceiling, and floor
        """
        super().__init__()
        self.variance_weight = risk_settings.get('variance_weight', 0.0)
        self.ceiling_weight = risk_settings.get('ceiling_weight', 1.0)
        self.floor_weight = risk_settings.get('floor_weight', 0.0)

    def get_player_fantasy_points(self, player: Player) -> float:
        """
        Calculate fantasy points based on risk settings.

        Parameters
        ----------
        player : Player
            Player object with projections

        Returns
        -------
        float
            Adjusted fantasy points
        """
        base_projection = player.fppg

        # If player has variance/ceiling/floor data, use it
        if hasattr(player, 'std_dev'):
            variance_adj = self.variance_weight * player.std_dev
            ceiling_adj = self.ceiling_weight * (player.ceiling if hasattr(player, 'ceiling') else base_projection * 1.2)
            floor_adj = self.floor_weight * (player.floor if hasattr(player, 'floor') else base_projection * 0.8)

            # Weighted average
            total_weight = self.variance_weight + self.ceiling_weight + self.floor_weight
            if total_weight > 0:
                return (variance_adj + ceiling_adj + floor_adj) / total_weight

        return base_projection


class LineupGenerator:
    """
    Generate optimal lineups for DraftKings contests using predictions from ML models
    """

    def __init__(self, contest_config_path: Optional[str] = None):
        """
        Initialize LineupGenerator.

        Parameters
        ----------
        contest_config_path : Optional[str]
            Path to contest configuration JSON file
        """
        self.config_dir = Path(__file__).parent.parent.parent / "config" / "contests"
        self.contest_config = self._load_contest_config(contest_config_path)
        self.optimizer = None
        self.slate_df = None

    def _load_contest_config(self, config_path: Optional[str] = None) -> Dict:
        """
        Load contest configuration from JSON file.

        Parameters
        ----------
        config_path : Optional[str]
            Path to configuration file

        Returns
        -------
        Dict
            Contest configuration
        """
        if config_path is None:
            # Default to cash game configuration
            config_path = self.config_dir / "cash_game.json"
        else:
            config_path = Path(config_path)
            if not config_path.exists():
                # Try in config directory
                config_path = self.config_dir / config_path.name

        if not config_path.exists():
            logger.warning(f"Config file not found: {config_path}, using default settings")
            return self._get_default_config()

        with open(config_path, 'r') as f:
            config = json.load(f)

        logger.info(f"Loaded contest config: {config['name']}")
        return config

    def _get_default_config(self) -> Dict:
        """
        Get default configuration for cash games.

        Returns
        -------
        Dict
            Default configuration
        """
        return {
            "contest_type": "cash_game",
            "name": "Default Cash Game",
            "optimization_settings": {
                "num_lineups": 1,
                "max_exposure": 1.0,
                "randomness": False,
                "min_salary_cap": 49500,
                "strategy": "floor"
            },
            "player_settings": {
                "min_projected_points": 20.0,
                "exclude_injured": True,
                "exclude_questionable": True
            },
            "risk_settings": {
                "variance_weight": 0.0,
                "ceiling_weight": 0.1,
                "floor_weight": 0.9
            }
        }

    def prepare_slate_data(self,
                           predictions_df: pd.DataFrame,
                           dfs_salaries_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Prepare slate data with predictions for optimization.

        Parameters
        ----------
        predictions_df : pd.DataFrame
            DataFrame with columns: playerID, playerName, predicted_fpts, position, team
        dfs_salaries_df : Optional[pd.DataFrame]
            DataFrame with DraftKings salary data

        Returns
        -------
        pd.DataFrame
            Prepared slate data
        """
        slate_df = predictions_df.copy()

        # Merge with salary data if provided
        if dfs_salaries_df is not None:
            slate_df = slate_df.merge(
                dfs_salaries_df[['playerID', 'salary', 'gameInfo', 'status']],
                on='playerID',
                how='inner'
            )
        else:
            # Use default salary if not provided (for testing)
            if 'salary' not in slate_df.columns:
                slate_df['salary'] = 5000  # Default salary
            if 'status' not in slate_df.columns:
                slate_df['status'] = None

        # Standardize column names
        slate_df = slate_df.rename(columns={
            'predicted_fpts': 'fppg',
            'position': 'pos'
        })

        # Handle multi-position players
        if 'pos' in slate_df.columns:
            slate_df['positions'] = slate_df['pos'].apply(
                lambda x: x.split('/') if isinstance(x, str) else []
            )

        # Add variance estimates if available
        if 'prediction_std' in slate_df.columns:
            slate_df['std_dev'] = slate_df['prediction_std']
            slate_df['ceiling'] = slate_df['fppg'] + (1.5 * slate_df['std_dev'])
            slate_df['floor'] = slate_df['fppg'] - (1.5 * slate_df['std_dev'])

        # Filter based on configuration
        min_points = self.contest_config['player_settings'].get('min_projected_points', 0)
        slate_df = slate_df[slate_df['fppg'] >= min_points]

        # Handle injury status
        if self.contest_config['player_settings'].get('exclude_injured', True):
            slate_df = slate_df[~slate_df['status'].isin(['O', 'IR', 'OUT'])]

        if self.contest_config['player_settings'].get('exclude_questionable', False):
            slate_df = slate_df[~slate_df['status'].isin(['Q', 'GTD', 'QUESTIONABLE'])]

        self.slate_df = slate_df
        logger.info(f"Prepared slate with {len(slate_df)} eligible players")

        return slate_df

    def _create_optimizer(self) -> LineupOptimizer:
        """
        Create and configure pydfs LineupOptimizer.

        Returns
        -------
        LineupOptimizer
            Configured optimizer
        """
        # Create DraftKings NBA optimizer
        optimizer = get_optimizer(Site.DRAFTKINGS, Sport.BASKETBALL)

        # Apply risk strategy
        risk_settings = self.contest_config.get('risk_settings', {})
        if any(risk_settings.values()):
            strategy = CustomFantasyPointsStrategy(risk_settings)
            optimizer.set_fantasy_points_strategy(strategy)

        # Set optimization constraints
        opt_settings = self.contest_config.get('optimization_settings', {})

        if 'min_salary_cap' in opt_settings:
            optimizer.set_min_salary_cap(opt_settings['min_salary_cap'])

        if 'max_repeating_players' in opt_settings:
            optimizer.set_max_repeating_players(opt_settings['max_repeating_players'])

        return optimizer

    def _convert_to_players(self, slate_df: pd.DataFrame) -> List[Player]:
        """
        Convert DataFrame to pydfs Player objects.

        Parameters
        ----------
        slate_df : pd.DataFrame
            Slate data

        Returns
        -------
        List[Player]
            List of Player objects
        """
        players = []

        for idx, row in slate_df.iterrows():
            # Extract name components
            full_name = row.get('playerName', '')
            name_parts = full_name.split(' ', 1) if full_name else ['', '']
            first_name = name_parts[0] if len(name_parts) > 0 else ''
            last_name = name_parts[1] if len(name_parts) > 1 else ''

            # Get positions
            positions = row.get('positions', [])
            if not positions and 'pos' in row:
                positions = row['pos'].split('/') if isinstance(row['pos'], str) else []

            player = Player(
                player_id=str(row.get('playerID', idx)),
                first_name=first_name,
                last_name=last_name,
                positions=positions,
                team=row.get('team', ''),
                salary=int(row.get('salary', 5000)),
                fppg=float(row.get('fppg', 0)),
                projected_ownership=row.get('ownership', None),
                min_exposure=row.get('min_exposure', None),
                max_exposure=row.get('max_exposure', None)
            )

            # Add custom attributes for advanced strategies
            if 'std_dev' in row:
                player.std_dev = row['std_dev']
            if 'ceiling' in row:
                player.ceiling = row['ceiling']
            if 'floor' in row:
                player.floor = row['floor']

            players.append(player)

        return players

    def generate_lineups(self,
                         predictions_df: pd.DataFrame,
                         dfs_salaries_df: Optional[pd.DataFrame] = None,
                         contest_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Generate optimal lineups for specified contests.

        Parameters
        ----------
        predictions_df : pd.DataFrame
            Predictions from ML models
        dfs_salaries_df : Optional[pd.DataFrame]
            DraftKings salary data
        contest_ids : Optional[List[str]]
            List of contest IDs to generate lineups for

        Returns
        -------
        List[Dict[str, Any]]
            List of lineup dictionaries with metadata
        """
        # Prepare slate data
        slate_df = self.prepare_slate_data(predictions_df, dfs_salaries_df)

        # Create optimizer
        self.optimizer = self._create_optimizer()

        # Convert to Player objects and load
        players = self._convert_to_players(slate_df)
        self.optimizer.player_pool.load_players(players)

        logger.info(f"Loaded {len(players)} players into optimizer")

        # Apply ownership settings if configured
        ownership_settings = self.contest_config['player_settings'].get('ownership_settings', {})
        if ownership_settings.get('use_projected_ownership', False):
            # Set ownership constraints
            fade_threshold = ownership_settings.get('fade_threshold', 0.35)
            chalk_threshold = ownership_settings.get('chalk_threshold', 0.25)

            for player in self.optimizer.player_pool.all_players:
                if player.projected_ownership:
                    if player.projected_ownership > fade_threshold:
                        player.max_exposure = 0.2  # Limit high ownership players
                    elif player.projected_ownership < chalk_threshold:
                        player.min_exposure = 0.1  # Ensure some low ownership

        # Apply stacking rules if configured
        self._apply_stacking_rules()

        # Generate lineups
        opt_settings = self.contest_config.get('optimization_settings', {})
        num_lineups = opt_settings.get('num_lineups', 1)
        max_exposure = opt_settings.get('max_exposure', None)
        randomness = opt_settings.get('randomness', False)

        try:
            lineups = list(self.optimizer.optimize(
                n=num_lineups,
                max_exposure=max_exposure,
                randomness=randomness
            ))
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            return []

        # Convert lineups to dictionaries with metadata
        lineup_dicts = []
        for i, lineup in enumerate(lineups):
            lineup_dict = self._lineup_to_dict(lineup, i + 1)

            # Add contest associations
            if contest_ids:
                lineup_dict['contest_ids'] = contest_ids

            lineup_dict['contest_type'] = self.contest_config.get('contest_type', 'unknown')
            lineup_dict['generated_at'] = datetime.now().isoformat()

            lineup_dicts.append(lineup_dict)

        logger.info(f"Generated {len(lineup_dicts)} lineups")

        return lineup_dicts

    def _apply_stacking_rules(self):
        """Apply stacking rules from configuration."""
        correlation_rules = self.contest_config['lineup_rules'].get('correlation_rules', {})
        stack_rules = correlation_rules.get('stack_rules', {})

        # Team stacking
        team_stack = stack_rules.get('team_stack', {})
        if team_stack.get('enabled', False):
            # Group players from same team
            teams = self.slate_df['team'].unique() if self.slate_df is not None else []
            for team in teams:
                team_players = [
                    p for p in self.optimizer.player_pool.all_players
                    if p.team == team
                ]
                if len(team_players) >= team_stack.get('min_players', 2):
                    # Create team stack
                    stack = TeamStack(team_stack.get('min_players', 2))
                    self.optimizer.add_stack(stack)

    def _lineup_to_dict(self, lineup: Lineup, lineup_num: int) -> Dict[str, Any]:
        """
        Convert pydfs Lineup object to dictionary.

        Parameters
        ----------
        lineup : Lineup
            pydfs Lineup object
        lineup_num : int
            Lineup number

        Returns
        -------
        Dict[str, Any]
            Lineup dictionary
        """
        players_list = []

        for player in lineup.players:
            player_dict = {
                'playerID': player.id,
                'playerName': player.full_name,
                'position': player.lineup_position,
                'team': player.team,
                'salary': player.salary,
                'projected_fpts': player.fppg
            }

            # Add additional attributes if available
            if hasattr(player, 'std_dev'):
                player_dict['std_dev'] = player.std_dev
            if hasattr(player, 'ceiling'):
                player_dict['ceiling'] = player.ceiling
            if hasattr(player, 'floor'):
                player_dict['floor'] = player.floor

            players_list.append(player_dict)

        return {
            'lineup_num': lineup_num,
            'players': players_list,
            'total_salary': lineup.salary_costs,
            'projected_points': lineup.fantasy_points_projection,
            'salary_remaining': 50000 - lineup.salary_costs
        }

    def export_lineups(self, lineups: List[Dict], output_path: str, format: str = 'csv'):
        """
        Export lineups to file.

        Parameters
        ----------
        lineups : List[Dict]
            List of lineup dictionaries
        output_path : str
            Output file path
        format : str
            Export format ('csv', 'json')
        """
        output_path = Path(output_path)

        if format == 'csv':
            # Convert to DraftKings CSV format
            rows = []
            for lineup in lineups:
                row = {}
                for i, player in enumerate(lineup['players'], 1):
                    row[f'P{i}'] = f"{player['playerName']} ({player['playerID']})"
                rows.append(row)

            df = pd.DataFrame(rows)
            df.to_csv(output_path, index=False)

        elif format == 'json':
            with open(output_path, 'w') as f:
                json.dump(lineups, f, indent=2)

        logger.info(f"Exported {len(lineups)} lineups to {output_path}")