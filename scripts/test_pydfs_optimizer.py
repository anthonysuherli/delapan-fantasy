"""
Test script for pydfs-lineup-optimizer
Demonstrates basic usage with DraftKings NBA
"""

from pydfs_lineup_optimizer import (
    get_optimizer,
    Site,
    Sport,
    Player,
    CSVLineupExporter,
    PlayerFilter
)
import pandas as pd
from datetime import datetime

def test_basic_optimizer():
    """Test basic optimizer functionality with dummy data"""

    # Create DraftKings NBA optimizer
    optimizer = get_optimizer(Site.DRAFTKINGS, Sport.BASKETBALL)

    # Create sample players (normally loaded from CSV)
    sample_players = [
        Player(
            player_id="001",
            first_name="LeBron",
            last_name="James",
            positions=["SF", "PF"],
            team="LAL",
            salary=11000,
            fppg=55.5,  # Fantasy points per game (our prediction)
            projected_ownership=0.35
        ),
        Player(
            player_id="002",
            first_name="Anthony",
            last_name="Davis",
            positions=["PF", "C"],
            team="LAL",
            salary=10500,
            fppg=52.3,
            projected_ownership=0.30
        ),
        Player(
            player_id="003",
            first_name="Russell",
            last_name="Westbrook",
            positions=["PG"],
            team="LAC",
            salary=7500,
            fppg=38.2,
            projected_ownership=0.22
        ),
        Player(
            player_id="004",
            first_name="Paul",
            last_name="George",
            positions=["SG", "SF"],
            team="LAC",
            salary=8500,
            fppg=42.1,
            projected_ownership=0.25
        ),
        Player(
            player_id="005",
            first_name="Kawhi",
            last_name="Leonard",
            positions=["SG", "SF"],
            team="LAC",
            salary=9500,
            fppg=46.8,
            projected_ownership=0.28
        ),
        Player(
            player_id="006",
            first_name="Tyrese",
            last_name="Haliburton",
            positions=["PG", "SG"],
            team="IND",
            salary=8000,
            fppg=41.5,
            projected_ownership=0.20
        ),
        Player(
            player_id="007",
            first_name="Myles",
            last_name="Turner",
            positions=["C"],
            team="IND",
            salary=6500,
            fppg=32.8,
            projected_ownership=0.15
        ),
        Player(
            player_id="008",
            first_name="Buddy",
            last_name="Hield",
            positions=["SG", "SF"],
            team="IND",
            salary=5500,
            fppg=28.3,
            projected_ownership=0.12
        ),
        Player(
            player_id="009",
            first_name="Nikola",
            last_name="Jokic",
            positions=["C"],
            team="DEN",
            salary=12000,
            fppg=60.2,
            projected_ownership=0.40
        ),
        Player(
            player_id="010",
            first_name="Jamal",
            last_name="Murray",
            positions=["PG", "SG"],
            team="DEN",
            salary=7000,
            fppg=35.5,
            projected_ownership=0.18
        ),
        Player(
            player_id="011",
            first_name="Michael",
            last_name="Porter Jr.",
            positions=["SF", "PF"],
            team="DEN",
            salary=6000,
            fppg=30.2,
            projected_ownership=0.14
        ),
        Player(
            player_id="012",
            first_name="Aaron",
            last_name="Gordon",
            positions=["PF", "C"],
            team="DEN",
            salary=5000,
            fppg=25.8,
            projected_ownership=0.10
        ),
        Player(
            player_id="013",
            first_name="Kentavious",
            last_name="Caldwell-Pope",
            positions=["SG"],
            team="DEN",
            salary=4500,
            fppg=22.3,
            projected_ownership=0.08
        ),
        Player(
            player_id="014",
            first_name="Reggie",
            last_name="Jackson",
            positions=["PG"],
            team="DEN",
            salary=4000,
            fppg=20.1,
            projected_ownership=0.06
        ),
        Player(
            player_id="015",
            first_name="DeAndre",
            last_name="Jordan",
            positions=["C"],
            team="DEN",
            salary=3500,
            fppg=18.5,
            projected_ownership=0.05
        )
    ]

    # Load players into optimizer
    optimizer.player_pool.load_players(sample_players)

    print("=" * 60)
    print("PYDFS LINEUP OPTIMIZER TEST")
    print("=" * 60)
    print(f"Site: DraftKings")
    print(f"Sport: NBA")
    print(f"Total Players: {len(sample_players)}")
    print(f"Salary Cap: ${optimizer.budget}")
    print(f"Positions: {[pos.name for pos in optimizer.settings.positions]}")
    print()

    # Generate optimal lineup
    print("Generating optimal lineup...")
    lineup = next(optimizer.optimize(1))

    print("\nOptimal Lineup:")
    print("-" * 60)
    for player in lineup.players:
        print(f"{player.lineup_position:4} {player.full_name:25} "
              f"{player.team:4} ${player.salary:6,} "
              f"{player.fppg:6.1f} pts")

    print("-" * 60)
    print(f"Total Salary: ${lineup.salary_costs:,}")
    print(f"Projected Points: {lineup.fantasy_points_projection:.1f}")
    print(f"Salary Remaining: ${optimizer.budget - lineup.salary_costs:,}")

    # Generate multiple lineups with diversity
    print("\n" + "=" * 60)
    print("Generating 5 diverse lineups...")
    print("=" * 60)

    lineups = list(optimizer.optimize(5))
    for i, lineup in enumerate(lineups, 1):
        print(f"\nLineup {i}:")
        print(f"  Players: {', '.join([p.last_name for p in lineup.players])}")
        print(f"  Salary: ${lineup.salary_costs:,}")
        print(f"  Projected: {lineup.fantasy_points_projection:.1f} pts")

    # Test player locking
    print("\n" + "=" * 60)
    print("Testing player locking...")
    print("=" * 60)

    # Reset optimizer
    optimizer.reset_lineup()

    # Lock specific player
    jokic = optimizer.player_pool.get_player_by_name("Nikola Jokic")
    optimizer.add_player_to_lineup(jokic)

    print(f"Locked player: {jokic.full_name}")

    lineup_with_lock = next(optimizer.optimize(1))
    print("\nLineup with locked player:")
    for player in lineup_with_lock.players:
        locked_indicator = " [LOCKED]" if player.id == jokic.id else ""
        print(f"  {player.lineup_position:4} {player.full_name:20}{locked_indicator}")

    print(f"\nTotal Salary: ${lineup_with_lock.salary_costs:,}")
    print(f"Projected Points: {lineup_with_lock.fantasy_points_projection:.1f}")

    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)


def test_integration_with_predictions():
    """Test integration with our prediction pipeline"""

    print("\n" + "=" * 60)
    print("INTEGRATION TEST WITH PREDICTIONS")
    print("=" * 60)

    # Simulate loading predictions from our model
    predictions_df = pd.DataFrame({
        'playerID': ['001', '002', '003', '004', '005'],
        'playerName': ['LeBron James', 'Anthony Davis', 'Russell Westbrook',
                       'Paul George', 'Kawhi Leonard'],
        'position': ['SF/PF', 'PF/C', 'PG', 'SG/SF', 'SG/SF'],
        'team': ['LAL', 'LAL', 'LAC', 'LAC', 'LAC'],
        'salary': [11000, 10500, 7500, 8500, 9500],
        'predicted_fpts': [55.5, 52.3, 38.2, 42.1, 46.8],  # Our model predictions
        'salary_tier': ['Elite', 'Elite', 'Mid', 'Mid', 'High']
    })

    print(f"Loaded {len(predictions_df)} players with predictions")
    print("\nSample predictions:")
    print(predictions_df.to_string())

    print("\nThis demonstrates how to integrate our model predictions")
    print("with pydfs-lineup-optimizer for lineup generation.")


if __name__ == "__main__":
    test_basic_optimizer()
    test_integration_with_predictions()