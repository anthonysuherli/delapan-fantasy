#!/usr/bin/env python
"""
Example: Using Opponent Features in DFS Backtesting

This script demonstrates how to use the new opponent features
in your backtesting pipeline for improved DFS predictions.

Usage:
    python examples/opponent_features_usage.py
"""

import sys
import os
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.features import registry
from src.features.transformers.opponent_stats import OpponentStatsTransformer
from src.data.storage.sqlite_storage import SQLiteStorage
from src.data.loaders.historical_loader import HistoricalDataLoader
import pandas as pd


def demo_opponent_features():
    """Demonstrate opponent features functionality"""
    
    print("="*80)
    print("OPPONENT FEATURES DEMO")
    print("="*80)
    
    # 1. Initialize data loader
    storage = SQLiteStorage(db_path="nba_dfs.db")
    loader = HistoricalDataLoader(storage)
    
    print("Loading sample data...")
    
    # Load recent historical data
    end_date = "20250206"  # Adjust to available data
    start_date = "20250101"
    
    try:
        # Load player logs
        player_logs = loader.load_historical_player_logs(
            start_date=start_date,
            end_date=end_date
        )
        
        if player_logs.empty:
            print("No historical data found. Using mock data for demo...")
            player_logs = create_mock_data()
        else:
            print(f"Loaded {len(player_logs)} player logs")
            
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Using mock data for demo...")
        player_logs = create_mock_data()
    
    # 2. Create and configure opponent features transformer
    print("\n" + "-"*60)
    print("CONFIGURING OPPONENT FEATURES")
    print("-"*60)
    
    # Select top features for demo
    high_impact_features = [
        'opp_pace',
        'opp_pg_fpts_allowed',
        'opp_sg_fpts_allowed', 
        'opp_sf_fpts_allowed',
        'opp_pf_fpts_allowed',
        'opp_c_fpts_allowed',
        'is_home_team',
        'opp_3pt_defense_rank',
        'opp_rest_days'
    ]
    
    transformer = OpponentStatsTransformer(
        features=high_impact_features,
        recent_games_window=10
    )
    
    print(f"Created transformer with {len(high_impact_features)} features:")
    for feature in high_impact_features:
        print(f"  - {feature}")
    
    # 3. Fit transformer on training data
    print("\n" + "-"*60) 
    print("FITTING TRANSFORMER")
    print("-"*60)
    
    # Split data for train/test
    split_date = pd.to_datetime("20250205", format="%Y%m%d")
    player_logs['gameDate_dt'] = pd.to_datetime(player_logs['gameDate'], format="%Y%m%d")
    
    train_data = player_logs[player_logs['gameDate_dt'] < split_date].copy()
    test_data = player_logs[player_logs['gameDate_dt'] >= split_date].copy()
    
    print(f"Training data: {len(train_data)} rows")
    print(f"Test data: {len(test_data)} rows")
    
    if len(train_data) > 0:
        # Fit transformer
        transformer.fit(train_data)
        print(f"Calculated team statistics for {len(transformer._team_stats_cache)} teams")
        
        # Show team statistics
        print("\nTeam Statistics Sample:")
        for team, stats in list(transformer._team_stats_cache.items())[:3]:
            print(f"  {team}: pace={stats['pace']:.1f}, foul_rate={stats['foul_rate']:.1f}")
    
    # 4. Transform test data with opponent features
    print("\n" + "-"*60)
    print("ADDING OPPONENT FEATURES")  
    print("-"*60)
    
    if len(test_data) > 0:
        # Transform data
        enhanced_data = transformer.transform(test_data)
        
        print(f"Enhanced data shape: {enhanced_data.shape}")
        print(f"Added {enhanced_data.shape[1] - test_data.shape[1]} new columns")
        
        # Show feature coverage
        print("\nFeature Coverage:")
        for feature in high_impact_features:
            if feature in enhanced_data.columns:
                coverage = enhanced_data[feature].notna().mean()
                print(f"  {feature}: {coverage:.1%}")
        
        # Show sample results
        print("\nSample Enhanced Data:")
        cols_to_show = ['playerID', 'teamAbv', 'opponent_team', 'is_home_team', 'opp_pace']
        available_cols = [col for col in cols_to_show if col in enhanced_data.columns]
        
        if available_cols:
            print(enhanced_data[available_cols].head(10))
        
        # 5. Feature impact analysis
        print("\n" + "-"*60)
        print("FEATURE ANALYSIS")
        print("-"*60)
        
        analyze_features(enhanced_data)
    
    print("\n" + "="*80)
    print("DEMO COMPLETE")
    print("="*80)
    
    print("\nTo use opponent features in backtesting:")
    print("1. Use --feature-config opponent_features")
    print("2. Run: python scripts/run_backtest.py --test-start 20250205 --test-end 20250206 --feature-config opponent_features")
    print("3. Compare results with baseline features")


def create_mock_data():
    """Create mock data for demo when real data is not available"""
    print("Creating mock data...")
    
    teams = ['LAL', 'GSW', 'BOS', 'MIA', 'DEN']
    dates = ['20250201', '20250202', '20250203', '20250204', '20250205']
    
    data = []
    player_counter = 1
    
    for date in dates:
        for i in range(0, len(teams), 2):
            if i + 1 < len(teams):
                home_team = teams[i] 
                away_team = teams[i + 1]
                game_id = f"{date}_{away_team}@{home_team}"
                
                # Add players for both teams
                for team in [home_team, away_team]:
                    for player_num in range(3):  # 3 players per team
                        data.append({
                            'playerID': f'player_{player_counter}',
                            'teamAbv': team,
                            'gameID': game_id,
                            'gameDate': date,
                            'pts': 15 + player_num * 5,
                            'reb': 5 + player_num * 2,
                            'ast': 3 + player_num,
                            'stl': 1,
                            'blk': 0 + player_num,
                            'mins': 25 + player_num * 5,
                            'fga': 10 + player_num * 3,
                            'fgm': 5 + player_num,
                            'fta': 3,
                            'ftm': 2,
                            'tptfga': 4,
                            'tptfgm': 1 + player_num,
                            'TOV': 2,
                            'PF': 2,
                            'fantasyPoints': 25.0 + player_num * 8,
                            'OffReb': 1,
                            'DefReb': 4 + player_num,
                            'usage': 20.0 + player_num * 3,
                            'plusMinus': 3 - player_num
                        })
                        player_counter += 1
    
    return pd.DataFrame(data)


def analyze_features(data):
    """Analyze opponent features for insights"""
    
    # Home vs Away performance
    if 'is_home_team' in data.columns and 'fantasyPoints' in data.columns:
        home_fpts = data[data['is_home_team'] == 1]['fantasyPoints'].mean()
        away_fpts = data[data['is_home_team'] == 0]['fantasyPoints'].mean()
        
        print(f"Home vs Away Fantasy Points:")
        print(f"  Home: {home_fpts:.1f} fpts")
        print(f"  Away: {away_fpts:.1f} fpts") 
        print(f"  Home advantage: {home_fpts - away_fpts:+.1f} fpts")
    
    # Pace impact
    if 'opp_pace' in data.columns and 'fantasyPoints' in data.columns:
        # Split into pace quartiles
        data = data.dropna(subset=['opp_pace', 'fantasyPoints'])
        if len(data) > 0:
            data['pace_quartile'] = pd.qcut(data['opp_pace'], q=4, labels=['Low', 'Med-Low', 'Med-High', 'High'])
            pace_impact = data.groupby('pace_quartile')['fantasyPoints'].mean()
            
            print(f"\nPace Impact on Fantasy Points:")
            for quartile, fpts in pace_impact.items():
                print(f"  {quartile} pace: {fpts:.1f} fpts")
    
    # 3pt defense ranking impact
    if 'opp_3pt_defense_rank' in data.columns:
        data = data.dropna(subset=['opp_3pt_defense_rank'])
        if len(data) > 0:
            best_defense = data[data['opp_3pt_defense_rank'] <= 10]['fantasyPoints'].mean()
            worst_defense = data[data['opp_3pt_defense_rank'] >= 21]['fantasyPoints'].mean()
            
            print(f"\n3pt Defense Impact:")
            print(f"  vs Top 10 defense: {best_defense:.1f} fpts")
            print(f"  vs Bottom 10 defense: {worst_defense:.1f} fpts")
            print(f"  Matchup advantage: {worst_defense - best_defense:+.1f} fpts")


if __name__ == "__main__":
    demo_opponent_features()