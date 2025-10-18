import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.features.transformers.opponent_stats import OpponentStatsTransformer


class TestOpponentStatsTransformer:
    """Test cases for OpponentStatsTransformer"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample player logs data for testing"""
        dates = ['20250101', '20250102', '20250103', '20250104', '20250105']
        
        data = []
        for i, date in enumerate(dates):
            # Lakers vs Warriors game
            game_id = f"{date}_LAL@GSW"
            
            # Lakers players
            data.append({
                'playerID': 'player1',
                'playerName': 'LeBron James',
                'teamAbv': 'LAL', 
                'gameID': game_id,
                'gameDate': date,
                'pos': 'SF',
                'pts': 25 + i,
                'reb': 8,
                'ast': 7,
                'stl': 2,
                'blk': 1,
                'mins': 35,
                'fga': 18,
                'fgm': 10,
                'fta': 6,
                'ftm': 5,
                'tptfga': 6,
                'tptfgm': 3,
                'TOV': 4,
                'PF': 2,
                'fantasyPoints': 45.5 + i,
                'OffReb': 2,
                'DefReb': 6,
                'usage': 28.5,
                'plusMinus': 8
            })
            
            data.append({
                'playerID': 'player2',
                'playerName': 'Anthony Davis',
                'teamAbv': 'LAL',
                'gameID': game_id, 
                'gameDate': date,
                'pos': 'C',
                'pts': 22,
                'reb': 12,
                'ast': 3,
                'stl': 1,
                'blk': 3,
                'mins': 33,
                'fga': 15,
                'fgm': 9,
                'fta': 4,
                'ftm': 4,
                'tptfga': 2,
                'tptfgm': 1,
                'TOV': 2,
                'PF': 3,
                'fantasyPoints': 42.0,
                'OffReb': 4,
                'DefReb': 8,
                'usage': 25.0,
                'plusMinus': 5
            })
            
            # Warriors players
            data.append({
                'playerID': 'player3',
                'playerName': 'Stephen Curry',
                'teamAbv': 'GSW',
                'gameID': game_id,
                'gameDate': date, 
                'pos': 'PG',
                'pts': 28 + i,
                'reb': 5,
                'ast': 8,
                'stl': 2,
                'blk': 0,
                'mins': 36,
                'fga': 20,
                'fgm': 11,
                'fta': 3,
                'ftm': 3,
                'tptfga': 12,
                'tptfgm': 6,
                'TOV': 3,
                'PF': 1,
                'fantasyPoints': 48.0 + i,
                'OffReb': 1,
                'DefReb': 4,
                'usage': 30.0,
                'plusMinus': -5
            })
            
        return pd.DataFrame(data)
    
    def test_transformer_initialization(self):
        """Test transformer initializes correctly"""
        transformer = OpponentStatsTransformer()
        
        assert transformer.name == 'opponent_stats'
        assert not transformer.is_fitted
        assert len(transformer.default_features) > 0
        assert 'opp_pace' in transformer.features
        assert 'is_home_team' in transformer.features
        
    def test_fit_calculates_team_stats(self, sample_data):
        """Test that fit method calculates team statistics"""
        transformer = OpponentStatsTransformer()
        transformer.fit(sample_data)
        
        assert transformer.is_fitted
        assert len(transformer._team_stats_cache) == 2  # LAL and GSW
        assert 'LAL' in transformer._team_stats_cache
        assert 'GSW' in transformer._team_stats_cache
        
        # Check team stats structure
        lal_stats = transformer._team_stats_cache['LAL']
        assert 'pace' in lal_stats
        assert 'ppg' in lal_stats
        assert 'foul_rate' in lal_stats
        assert isinstance(lal_stats['pace'], (int, float))
        
    def test_opponent_team_identification(self, sample_data):
        """Test opponent team identification from gameID"""
        transformer = OpponentStatsTransformer()
        transformer.fit(sample_data)
        
        result = transformer.transform(sample_data)
        
        # Check opponent team is correctly identified
        lal_data = result[result['teamAbv'] == 'LAL']
        gsw_data = result[result['teamAbv'] == 'GSW']
        
        assert all(lal_data['opponent_team'] == 'GSW')
        assert all(gsw_data['opponent_team'] == 'LAL')
        
    def test_home_away_indicator(self, sample_data):
        """Test home/away indicator is correct"""
        transformer = OpponentStatsTransformer()
        transformer.fit(sample_data)
        
        result = transformer.transform(sample_data)
        
        # In gameID format YYYYMMDD_AWAY@HOME, GSW is home, LAL is away
        lal_data = result[result['teamAbv'] == 'LAL']
        gsw_data = result[result['teamAbv'] == 'GSW']
        
        assert all(lal_data['is_home_team'] == 0)  # LAL is away
        assert all(gsw_data['is_home_team'] == 1)  # GSW is home
        
    def test_opponent_features_added(self, sample_data):
        """Test that opponent features are added to dataframe"""
        features = ['opp_pace', 'is_home_team', 'opp_foul_rate']
        transformer = OpponentStatsTransformer(features=features)
        transformer.fit(sample_data)
        
        result = transformer.transform(sample_data)
        
        # Check all features were added
        for feature in features:
            assert feature in result.columns
            
        # Check features have valid values
        assert result['opp_pace'].notna().any()
        assert result['is_home_team'].isin([0, 1]).all()
        assert result['opp_foul_rate'].notna().any()
        
    def test_missing_required_columns(self, sample_data):
        """Test error handling for missing required columns"""
        transformer = OpponentStatsTransformer()
        transformer.fit(sample_data)
        
        # Remove required column
        bad_data = sample_data.drop(columns=['gameID'])
        
        with pytest.raises(ValueError, match="Missing required columns"):
            transformer.transform(bad_data)
            
    def test_unfitted_transformer_error(self, sample_data):
        """Test error when trying to transform before fitting"""
        transformer = OpponentStatsTransformer()
        
        with pytest.raises(ValueError, match="has not been fitted"):
            transformer.transform(sample_data)
            
    def test_3pt_defense_ranking(self, sample_data):
        """Test 3-point defense ranking calculation"""
        transformer = OpponentStatsTransformer(features=['opp_3pt_defense_rank'])
        transformer.fit(sample_data)
        
        result = transformer.transform(sample_data)
        
        # Should have ranking values
        ranks = result['opp_3pt_defense_rank'].dropna().unique()
        assert len(ranks) > 0
        assert all(rank >= 1 for rank in ranks)  # Rankings start at 1
        
    def test_custom_features_list(self, sample_data):
        """Test using custom features list"""
        custom_features = ['opp_pace', 'is_home_team'] 
        transformer = OpponentStatsTransformer(features=custom_features)
        transformer.fit(sample_data)
        
        result = transformer.transform(sample_data)
        
        # Should only have custom features
        assert 'opp_pace' in result.columns
        assert 'is_home_team' in result.columns
        assert 'opp_foul_rate' not in result.columns  # Not in custom list
        
    def test_team_stats_calculations(self, sample_data):
        """Test team statistics calculations are reasonable"""
        transformer = OpponentStatsTransformer()
        transformer.fit(sample_data)
        
        # Check LAL stats
        lal_stats = transformer._team_stats_cache['LAL']
        
        # Pace should be positive
        assert lal_stats['pace'] > 0
        
        # PPG should match data
        expected_ppg = sample_data[sample_data['teamAbv'] == 'LAL'].groupby(['gameDate', 'gameID'])['pts'].sum().mean()
        assert abs(lal_stats['ppg'] - expected_ppg) < 0.1
        
        # Turnover rate should be between 0 and 1
        assert 0 <= lal_stats['turnover_rate'] <= 1


if __name__ == "__main__":
    # Run basic test manually
    print("Running basic opponent features test...")
    
    # Create sample data
    data = []
    for i in range(5):
        date = f"2025010{i+1}"
        game_id = f"{date}_LAL@GSW"
        
        data.extend([
            {
                'playerID': f'player{i}_lal',
                'teamAbv': 'LAL',
                'gameID': game_id,
                'gameDate': date,
                'pts': 20 + i,
                'reb': 8,
                'ast': 5,
                'stl': 1,
                'blk': 1,
                'mins': 32,
                'fga': 15,
                'fgm': 8,
                'fta': 4,
                'ftm': 3,
                'tptfga': 5,
                'tptfgm': 2,
                'TOV': 3,
                'PF': 2,
                'fantasyPoints': 35.0 + i,
                'OffReb': 2,
                'DefReb': 6,
                'usage': 25.0,
                'plusMinus': 5
            },
            {
                'playerID': f'player{i}_gsw', 
                'teamAbv': 'GSW',
                'gameID': game_id,
                'gameDate': date,
                'pts': 22 + i,
                'reb': 6,
                'ast': 7,
                'stl': 2,
                'blk': 0,
                'mins': 35,
                'fga': 18,
                'fgm': 9,
                'fta': 2,
                'ftm': 2,
                'tptfga': 8,
                'tptfgm': 4,
                'TOV': 2,
                'PF': 1,
                'fantasyPoints': 38.0 + i,
                'OffReb': 1,
                'DefReb': 5,
                'usage': 28.0,
                'plusMinus': -2
            }
        ])
    
    df = pd.DataFrame(data)
    
    # Test transformer
    transformer = OpponentStatsTransformer()
    print(f"Created transformer with {len(transformer.features)} features")
    
    # Fit
    transformer.fit(df)
    print(f"Fitted transformer, calculated stats for {len(transformer._team_stats_cache)} teams")
    
    # Transform
    result = transformer.transform(df)
    print(f"Transformed data: {result.shape[0]} rows, {result.shape[1]} columns")
    
    # Check key features
    print("\nFeature coverage:")
    for feature in ['opponent_team', 'is_home_team', 'opp_pace', 'opp_foul_rate']:
        if feature in result.columns:
            coverage = result[feature].notna().mean()
            print(f"  {feature}: {coverage:.1%} coverage")
    
    print("\nSample results:")
    cols_to_show = ['playerID', 'teamAbv', 'opponent_team', 'is_home_team', 'opp_pace']
    available_cols = [col for col in cols_to_show if col in result.columns]
    print(result[available_cols].head())
    
    print("\nBasic test completed successfully!")