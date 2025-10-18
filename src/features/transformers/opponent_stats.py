import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from ..base import FeatureTransformer
import logging

logger = logging.getLogger(__name__)


class OpponentStatsTransformer(FeatureTransformer):
    """
    Add opponent team statistics to player logs for matchup context.
    
    Calculates team-level statistics and adds opponent features like:
    - opp_pace: Opponent's pace of play
    - opp_def_rating_last_10: Recent defensive performance 
    - opp_pg_fpts_allowed: Fantasy points allowed to point guards
    - is_home_team: Home/away indicator
    - opp_3pt_defense_rank: 3-point defense ranking
    - And more...
    """
    
    def __init__(
        self,
        features: List[str] = None,
        lookback_days: int = 365,
        recent_games_window: int = 10
    ):
        """
        Initialize opponent stats transformer.
        
        Args:
            features: List of opponent features to calculate. If None, use default set.
            lookback_days: Days to look back for calculating team statistics
            recent_games_window: Number of recent games for recent form metrics
        """
        super().__init__('opponent_stats')
        
        self.default_features = [
            'opp_pace',
            'opp_def_rating_last_10', 
            'opp_pg_fpts_allowed',
            'opp_sg_fpts_allowed', 
            'opp_sf_fpts_allowed',
            'opp_pf_fpts_allowed',
            'opp_c_fpts_allowed',
            'is_home_team',
            'opp_3pt_defense_rank',
            'opp_rest_days',
            'opp_foul_rate',
            'opp_turnover_rate'
        ]
        
        self.features = features or self.default_features
        self.lookback_days = lookback_days
        self.recent_games_window = recent_games_window
        self._team_stats_cache = {}
        
    def fit(self, data: pd.DataFrame) -> 'OpponentStatsTransformer':
        """
        Fit transformer by calculating team statistics from training data.
        
        Args:
            data: Training data with player game logs
            
        Returns:
            Self for method chaining
        """
        logger.info("Fitting OpponentStatsTransformer - calculating team statistics")
        
        # Calculate team statistics from training data
        self._calculate_team_statistics(data)
        self._fitted = True
        
        logger.info(f"Calculated statistics for {len(self._team_stats_cache)} teams")
        return self
        
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data by adding opponent features.
        
        Args:
            data: Player logs to transform
            
        Returns:
            DataFrame with opponent features added
        """
        if not self._fitted:
            raise ValueError(f"Transformer '{self.name}' has not been fitted")
            
        df = data.copy()
        
        # Ensure required columns exist
        required_cols = ['gameDate', 'playerID', 'teamAbv', 'gameID']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        # Convert gameDate to datetime if needed
        if df['gameDate'].dtype == 'object':
            df['gameDate'] = pd.to_datetime(df['gameDate'], format='%Y%m%d', errors='coerce')
        
        # Get opponent team for each game
        df = self._add_opponent_team(df)
        
        # Add opponent features
        for feature in self.features:
            df = self._add_feature(df, feature)
            
        logger.info(f"Added {len(self.features)} opponent features to {len(df)} player logs")
        return df
    
    def _calculate_team_statistics(self, data: pd.DataFrame) -> None:
        """Calculate and cache team-level statistics."""
        df = data.copy()
        
        # Convert gameDate if needed
        if df['gameDate'].dtype == 'object':
            df['gameDate'] = pd.to_datetime(df['gameDate'], format='%Y%m%d', errors='coerce')
            
        # Convert numeric columns
        numeric_cols = ['pts', 'reb', 'ast', 'stl', 'blk', 'mins', 'fga', 'fgm', 'fta', 
                       'ftm', 'tptfga', 'tptfgm', 'TOV', 'PF', 'fantasyPoints']
        for col in numeric_cols:
            if col in df.columns:
                # Handle string concatenation issues by taking first value
                if df[col].dtype == 'object':
                    df[col] = df[col].astype(str).str.split('.').str[0]
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Calculate team statistics by team and date
        team_stats = {}
        
        for team in df['teamAbv'].unique():
            if pd.isna(team):
                continue
                
            team_data = df[df['teamAbv'] == team].copy()
            team_data = team_data.sort_values('gameDate')
            
            team_stats[team] = self._calculate_single_team_stats(team_data)
            
        self._team_stats_cache = team_stats
    
    def _calculate_single_team_stats(self, team_data: pd.DataFrame) -> Dict:
        """Calculate statistics for a single team."""
        stats = {}
        
        # Group by game to get team-level stats
        game_stats = team_data.groupby(['gameDate', 'gameID']).agg({
            'pts': 'sum',
            'reb': 'sum', 
            'ast': 'sum',
            'stl': 'sum',
            'blk': 'sum',
            'mins': 'sum',
            'fga': 'sum',
            'fgm': 'sum',
            'fta': 'sum',
            'ftm': 'sum',
            'tptfga': 'sum',
            'tptfgm': 'sum',
            'TOV': 'sum',
            'PF': 'sum',
            'fantasyPoints': 'sum'
        }).reset_index()
        
        # Calculate pace (possessions per 48 minutes)
        game_stats['possessions'] = (
            game_stats['fga'] + 
            0.44 * game_stats['fta'] + 
            game_stats['TOV'] - 
            (game_stats['reb'] * 0.3)  # Approximate offensive rebounds
        )
        game_stats['pace'] = game_stats['possessions'] * 48 / game_stats['mins']
        
        # Season averages
        stats['pace'] = game_stats['pace'].mean()
        stats['ppg'] = game_stats['pts'].mean()
        stats['foul_rate'] = game_stats['PF'].mean()
        stats['turnover_rate'] = game_stats['TOV'].mean() / game_stats['possessions'].mean()
        stats['3pt_defense'] = game_stats['tptfgm'].mean() / game_stats['tptfga'].mean() if game_stats['tptfga'].mean() > 0 else 0
        
        # Recent form (last N games)
        recent_games = game_stats.tail(self.recent_games_window)
        stats['pace_last_10'] = recent_games['pace'].mean()
        stats['def_rating_last_10'] = recent_games['pts'].mean()  # Simplified - points allowed
        
        # Position-specific fantasy points (requires salary data with positions)
        # For now, use overall fantasy points allowed
        stats['pg_fpts_allowed'] = game_stats['fantasyPoints'].mean()
        stats['sg_fpts_allowed'] = game_stats['fantasyPoints'].mean()
        stats['sf_fpts_allowed'] = game_stats['fantasyPoints'].mean()
        stats['pf_fpts_allowed'] = game_stats['fantasyPoints'].mean()
        stats['c_fpts_allowed'] = game_stats['fantasyPoints'].mean()
        
        # Rest days calculation
        stats['avg_rest_days'] = game_stats['gameDate'].diff().dt.days.mean()
        
        return stats
    
    def _add_opponent_team(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add opponent team column by parsing gameID."""
        # GameID format: YYYYMMDD_AWAY@HOME
        df = df.copy()
        
        def get_opponent(row):
            try:
                game_id = row['gameID']
                if '@' not in game_id:
                    return None
                    
                _, matchup = game_id.split('_')
                away, home = matchup.split('@')
                
                if row['teamAbv'] == home:
                    return away  # Home team faces away team
                elif row['teamAbv'] == away:
                    return home  # Away team faces home team
                else:
                    return None
            except:
                return None
        
        df['opponent_team'] = df.apply(get_opponent, axis=1)
        
        # Add home/away indicator
        def is_home(row):
            try:
                game_id = row['gameID']
                if '@' not in game_id:
                    return None
                    
                _, matchup = game_id.split('_')
                away, home = matchup.split('@')
                
                return 1 if row['teamAbv'] == home else 0
            except:
                return None
                
        df['is_home_team'] = df.apply(is_home, axis=1)
        
        return df
    
    def _add_feature(self, df: pd.DataFrame, feature: str) -> pd.DataFrame:
        """Add a specific opponent feature to the dataframe."""
        if feature == 'is_home_team':
            # Already added in _add_opponent_team
            return df
            
        # Map feature names to team stats
        feature_mapping = {
            'opp_pace': 'pace',
            'opp_def_rating_last_10': 'def_rating_last_10',
            'opp_pg_fpts_allowed': 'pg_fpts_allowed',
            'opp_sg_fpts_allowed': 'sg_fpts_allowed', 
            'opp_sf_fpts_allowed': 'sf_fpts_allowed',
            'opp_pf_fpts_allowed': 'pf_fpts_allowed',
            'opp_c_fpts_allowed': 'c_fpts_allowed',
            'opp_rest_days': 'avg_rest_days',
            'opp_foul_rate': 'foul_rate',
            'opp_turnover_rate': 'turnover_rate'
        }
        
        if feature in feature_mapping:
            stat_name = feature_mapping[feature]
            
            def get_opponent_stat(opponent_team):
                if pd.isna(opponent_team) or opponent_team not in self._team_stats_cache:
                    return np.nan
                return self._team_stats_cache[opponent_team].get(stat_name, np.nan)
            
            df[feature] = df['opponent_team'].apply(get_opponent_stat)
            
        elif feature == 'opp_3pt_defense_rank':
            # Calculate 3pt defense ranking (1 = worst defense)
            defense_values = []
            for team, stats in self._team_stats_cache.items():
                defense_values.append((team, stats.get('3pt_defense', 0)))
            
            # Sort by 3pt_defense descending (higher = worse defense)
            defense_values.sort(key=lambda x: x[1], reverse=True)
            
            # Create ranking dict
            rank_dict = {team: rank + 1 for rank, (team, _) in enumerate(defense_values)}
            
            def get_defense_rank(opponent_team):
                if pd.isna(opponent_team):
                    return np.nan
                return rank_dict.get(opponent_team, np.nan)
            
            df[feature] = df['opponent_team'].apply(get_defense_rank)
            
        else:
            logger.warning(f"Unknown opponent feature: {feature}")
            df[feature] = np.nan
            
        return df