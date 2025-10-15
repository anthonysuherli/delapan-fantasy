import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from src.data.collectors.tank01_client import Tank01Client
import json

def test_teams_endpoint():
    client = Tank01Client()

    print("Fetching teams from Tank01 API...")
    print("=" * 80)

    teams = client.get_teams()

    print(f"\nTotal teams: {len(teams)}")
    print("=" * 80)

    print("\nFirst 5 teams:")
    print("-" * 80)
    for team in teams[:5]:
        print(json.dumps(team, indent=2))
        print("-" * 80)

    print("\nAll team abbreviations:")
    print(", ".join(sorted([team.get('teamAbv', 'N/A') for team in teams])))

    print(f"\nRemaining API requests: {client.get_remaining_requests()}")

    return teams

if __name__ == "__main__":
    teams = test_teams_endpoint()
