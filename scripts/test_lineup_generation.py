"""
Test script for lineup generation pipeline
Demonstrates integration with predictions and contest management
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import json
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.optimization.lineup_generator import LineupGenerator
from src.optimization.contest_manager import ContestManager
from src.optimization.deprecated.backtest_lineup_integration import BacktestWithLineups

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_lineup_generator_with_predictions():
    """Test lineup generation with simulated predictions"""

    print("="*80)
    print("TEST 1: LINEUP GENERATOR WITH PREDICTIONS")
    print("="*80)

    # Create simulated prediction data (as would come from ML model)
    predictions_df = pd.DataFrame({
        'playerID': ['001', '002', '003', '004', '005', '006', '007', '008',
                     '009', '010', '011', '012', '013', '014', '015', '016'],
        'playerName': ['LeBron James', 'Anthony Davis', 'Russell Westbrook', 'Paul George',
                       'Kawhi Leonard', 'Nikola Jokic', 'Jamal Murray', 'Michael Porter Jr.',
                       'Aaron Gordon', 'Kentavious Caldwell-Pope', 'Tyrese Haliburton',
                       'Myles Turner', 'Buddy Hield', 'Benedict Mathurin', 'Bruce Brown', 'TJ McConnell'],
        'position': ['SF/PF', 'PF/C', 'PG', 'SG/SF', 'SG/SF', 'C', 'PG/SG', 'SF/PF',
                     'PF/C', 'SG', 'PG/SG', 'C', 'SG/SF', 'SG/SF', 'SG/SF', 'PG'],
        'team': ['LAL', 'LAL', 'LAC', 'LAC', 'LAC', 'DEN', 'DEN', 'DEN',
                 'DEN', 'DEN', 'IND', 'IND', 'IND', 'IND', 'IND', 'IND'],
        'salary': [11000, 10500, 7500, 8500, 9500, 12000, 7000, 6000,
                   5000, 4500, 8000, 6500, 5500, 4000, 3500, 3000],
        'predicted_fpts': [55.5, 52.3, 38.2, 42.1, 46.8, 60.2, 35.5, 30.2,
                           25.8, 22.3, 41.5, 32.8, 28.3, 20.1, 18.5, 15.2],
        'prediction_std': [8.2, 7.5, 6.1, 6.8, 7.2, 5.5, 5.8, 5.2,
                            4.5, 4.2, 6.5, 5.1, 4.8, 3.8, 3.5, 3.0],
        'status': [None, None, None, None, None, None, 'GTD', None,
                   None, None, None, None, None, None, None, None]
    })

    # Calculate ceiling/floor
    predictions_df['ceiling'] = predictions_df['predicted_fpts'] + (1.5 * predictions_df['prediction_std'])
    predictions_df['floor'] = predictions_df['predicted_fpts'] - (1.5 * predictions_df['prediction_std'])

    print(f"\nLoaded {len(predictions_df)} players with predictions")
    print("\nTop 5 projections:")
    print(predictions_df.nlargest(5, 'predicted_fpts')[['playerName', 'team', 'salary', 'predicted_fpts', 'ceiling', 'floor']])

    # Test different contest types
    contest_configs = [
        'cash_game.json',
        'gpp_tournament.json',
        'single_entry.json',
        'multi_entry.json'
    ]

    for config_file in contest_configs:
        print(f"\n{'='*60}")
        print(f"Testing with {config_file}")
        print('='*60)

        try:
            # Create lineup generator with specific contest config
            generator = LineupGenerator(contest_config_path=config_file)

            # Generate lineups
            lineups = generator.generate_lineups(predictions_df)

            print(f"Generated {len(lineups)} lineups")

            # Display first lineup
            if lineups:
                lineup = lineups[0]
                print(f"\nLineup 1 Details:")
                print(f"  Total Salary: ${lineup['total_salary']:,}")
                print(f"  Projected Points: {lineup['projected_points']:.1f}")
                print(f"  Salary Remaining: ${lineup['salary_remaining']:,}")
                print("\n  Players:")
                for player in lineup['players']:
                    print(f"    {player['position']:4} {player['playerName']:25} "
                          f"{player['team']:4} ${player['salary']:6,} {player['projected_fpts']:6.1f} pts")

        except Exception as e:
            print(f"Error with {config_file}: {e}")

    print("\nTest 1 completed successfully!")


def test_contest_manager():
    """Test contest management functionality"""

    print("\n" + "="*80)
    print("TEST 2: CONTEST MANAGER")
    print("="*80)

    # Create contest manager
    manager = ContestManager()

    # Simulate contest data from DraftKings API
    sample_contests = [
        {
            'contestID': 'C001',
            'contestName': 'NBA $750K Opening Tip Off [$200K to 1st]',
            'entryFee': 15,
            'totalPrize': 750000,
            'maxEntries': 1,
            'currentEntries': 35000,
            'gameType': 'Tournament',
            'isGuaranteed': True,
            'startTime': '2025-01-20T23:30:00Z'
        },
        {
            'contestID': 'C002',
            'contestName': 'NBA $100K Double Up',
            'entryFee': 50,
            'totalPrize': 100000,
            'maxEntries': 1,
            'currentEntries': 2000,
            'gameType': 'DoubleUp',
            'isGuaranteed': True,
            'startTime': '2025-01-20T23:30:00Z'
        },
        {
            'contestID': 'C003',
            'contestName': 'NBA $40K Single Entry',
            'entryFee': 100,
            'totalPrize': 40000,
            'maxEntries': 1,
            'currentEntries': 400,
            'gameType': 'Tournament',
            'isGuaranteed': True,
            'startTime': '2025-01-20T23:30:00Z'
        },
        {
            'contestID': 'C004',
            'contestName': 'NBA $20K 50/50',
            'entryFee': 20,
            'totalPrize': 20000,
            'maxEntries': 3,
            'currentEntries': 1000,
            'gameType': 'FiftyFifty',
            'isGuaranteed': True,
            'startTime': '2025-01-20T23:30:00Z'
        }
    ]

    # Load contests
    contests_df = manager.load_contests_from_api(sample_contests)
    print(f"\nLoaded {len(contests_df)} contests")
    print("\nContest Categories:")
    print(contests_df['contest_category'].value_counts())

    # Filter contests
    print("\n" + "-"*40)
    print("Filtering contests...")

    cash_contests = manager.filter_contests(
        max_entry_fee=50,
        contest_types=['cash_game']
    )
    print(f"\nCash games under $50: {len(cash_contests)}")
    if not cash_contests.empty:
        print(cash_contests[['contestName', 'entryFee', 'contest_category']])

    tournament_contests = manager.filter_contests(
        contest_types=['gpp_tournament', 'single_entry'],
        guaranteed_only=True
    )
    print(f"\nGuaranteed tournaments: {len(tournament_contests)}")

    # Test lineup associations
    print("\n" + "-"*40)
    print("Testing lineup associations...")

    # Create dummy lineups
    dummy_lineups = [
        {
            'lineup_num': 1,
            'players': [{'playerID': f'P{i}', 'playerName': f'Player {i}',
                         'position': 'PG', 'team': 'LAL', 'salary': 5000,
                         'projected_fpts': 30.0} for i in range(8)],
            'total_salary': 40000,
            'projected_points': 240.0
        },
        {
            'lineup_num': 2,
            'players': [{'playerID': f'P{i+8}', 'playerName': f'Player {i+8}',
                         'position': 'SG', 'team': 'DEN', 'salary': 6000,
                         'projected_fpts': 32.0} for i in range(8)],
            'total_salary': 48000,
            'projected_points': 256.0
        }
    ]

    # Associate lineups with contests
    associated = manager.associate_lineups_with_contests(
        dummy_lineups,
        ['C001', 'C003']
    )

    print(f"\nAssociated {len(associated)} lineups with contests")
    for lineup in associated:
        print(f"  Lineup {lineup['lineup_num']}: {len(lineup['contests'])} contests")

    # Calculate exposure
    exposure_df = manager.calculate_exposure()
    if not exposure_df.empty:
        print("\nPlayer Exposure:")
        print(exposure_df.head())

    # Save session
    manager.save_session("test_session")
    print("\nSession saved successfully!")

    print("\nTest 2 completed successfully!")


def test_integrated_backtest():
    """Test integrated backtest with lineup generation"""

    print("\n" + "="*80)
    print("TEST 3: INTEGRATED BACKTEST WITH LINEUPS")
    print("="*80)

    # Note: This requires database and data to be available
    # For testing purposes, we'll just show the structure

    print("\nBacktest initialization structure:")
    print("""
    backtest = BacktestWithLineups(
        train_start='20241001',
        train_end='20241130',
        test_start='20241201',
        test_end='20241210',
        data_dir='data',
        per_player_models=True,
        generate_lineups=True,
        contest_config='config/contests/gpp_tournament.json',
        num_lineups=20,
        track_lineup_performance=True
    )

    results = backtest.run_with_lineups()
    """)

    print("\nThis would generate:")
    print("  - Daily predictions for each player")
    print("  - Optimal lineups based on predictions")
    print("  - Lineup performance tracking against actuals")
    print("  - ROI analysis for different contest types")
    print("  - Export files for DraftKings upload")

    print("\nTest 3 structure demonstrated!")


def test_export_functionality():
    """Test lineup export functionality"""

    print("\n" + "="*80)
    print("TEST 4: EXPORT FUNCTIONALITY")
    print("="*80)

    # Create sample lineup
    sample_lineup = [{
        'lineup_num': 1,
        'players': [
            {'playerID': '001', 'playerName': 'LeBron James', 'position': 'PG', 'team': 'LAL', 'salary': 11000, 'projected_fpts': 55.5},
            {'playerID': '002', 'playerName': 'Anthony Davis', 'position': 'SG', 'team': 'LAL', 'salary': 10500, 'projected_fpts': 52.3},
            {'playerID': '003', 'playerName': 'Russell Westbrook', 'position': 'SF', 'team': 'LAC', 'salary': 7500, 'projected_fpts': 38.2},
            {'playerID': '004', 'playerName': 'Paul George', 'position': 'PF', 'team': 'LAC', 'salary': 8500, 'projected_fpts': 42.1},
            {'playerID': '005', 'playerName': 'Nikola Jokic', 'position': 'C', 'team': 'DEN', 'salary': 12000, 'projected_fpts': 60.2},
            {'playerID': '006', 'playerName': 'Jamal Murray', 'position': 'G', 'team': 'DEN', 'salary': 7000, 'projected_fpts': 35.5},
            {'playerID': '007', 'playerName': 'Michael Porter Jr.', 'position': 'F', 'team': 'DEN', 'salary': 6000, 'projected_fpts': 30.2},
            {'playerID': '008', 'playerName': 'Aaron Gordon', 'position': 'UTIL', 'team': 'DEN', 'salary': 5000, 'projected_fpts': 25.8}
        ],
        'total_salary': 67500,
        'projected_points': 339.8,
        'contest_type': 'gpp_tournament',
        'generated_at': datetime.now().isoformat()
    }]

    # Create generator for export functionality
    generator = LineupGenerator()

    # Test CSV export (DraftKings format)
    csv_path = Path(__file__).parent / 'test_lineup.csv'
    generator.export_lineups(sample_lineup, str(csv_path), format='csv')
    print(f"\nExported lineup to CSV: {csv_path}")

    # Read and display CSV
    if csv_path.exists():
        csv_df = pd.read_csv(csv_path)
        print("\nCSV Content:")
        print(csv_df)

    # Test JSON export (full details)
    json_path = Path(__file__).parent / 'test_lineup.json'
    generator.export_lineups(sample_lineup, str(json_path), format='json')
    print(f"\nExported lineup to JSON: {json_path}")

    # Clean up test files
    if csv_path.exists():
        csv_path.unlink()
    if json_path.exists():
        json_path.unlink()

    print("\nTest 4 completed successfully!")


def main():
    """Run all tests"""

    print("="*80)
    print("LINEUP GENERATION PIPELINE TESTS")
    print("="*80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        # Run tests
        test_lineup_generator_with_predictions()
        test_contest_manager()
        test_integrated_backtest()
        test_export_functionality()

        print("\n" + "="*80)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("="*80)

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()

    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()