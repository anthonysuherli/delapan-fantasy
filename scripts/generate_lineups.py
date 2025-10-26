#!/usr/bin/env python
"""
Generate optimal lineups from predictions using pydfs-lineup-optimizer.

Simple wrapper around LineupGenerator that loads predictions and
generates lineups for DraftKings contests.

Usage:
    python scripts/generate_lineups.py --predictions predictions.csv --num-lineups 20
    python scripts/generate_lineups.py --predictions predictions.csv --strategy cash_game
    python scripts/generate_lineups.py --predictions predictions.csv --output lineups.csv
"""

import argparse
import sys
import os
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from datetime import datetime

from src.optimization.lineup_generator import LineupGenerator


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate optimal lineups from predictions"
    )

    parser.add_argument(
        "--predictions",
        required=True,
        help="Path to predictions CSV file from predict_slate.py"
    )

    parser.add_argument(
        "--num-lineups",
        type=int,
        default=1,
        help="Number of lineups to generate (default: 1)"
    )

    parser.add_argument(
        "--strategy",
        default="balanced",
        choices=['conservative', 'balanced', 'aggressive'],
        help="Lineup generation strategy (default: balanced)"
    )

    parser.add_argument(
        "--min-salary",
        type=int,
        default=49000,
        help="Minimum salary cap to use (default: 49000)"
    )

    parser.add_argument(
        "--max-exposure",
        type=float,
        default=None,
        help="Maximum player exposure across lineups (0.0-1.0)"
    )

    parser.add_argument(
        "--contest-config",
        default=None,
        help="Path to custom contest config JSON (optional)"
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Output file path (default: data/lineups/lineups_TIMESTAMP.csv)"
    )

    parser.add_argument(
        "--output-format",
        default="csv",
        choices=['csv', 'json'],
        help="Output format (default: csv)"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed information"
    )

    return parser.parse_args()


def load_predictions(predictions_path, verbose=False):
    """Load predictions CSV and prepare for lineup generation."""
    predictions_df = pd.read_csv(predictions_path)

    if verbose:
        print(f"Loaded {len(predictions_df)} predictions from {predictions_path}")
        print(f"Columns: {predictions_df.columns.tolist()}")

    # Ensure required columns exist
    required_cols = ['playerID', 'name', 'salary', 'predicted']
    missing_cols = [col for col in required_cols if col not in predictions_df.columns]

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Rename 'predicted' to 'fppg' for lineup generator
    if 'fppg' not in predictions_df.columns:
        predictions_df['fppg'] = predictions_df['predicted']

    # Rename 'name' to 'playerName' if needed
    if 'playerName' not in predictions_df.columns and 'name' in predictions_df.columns:
        predictions_df['playerName'] = predictions_df['name']

    # Handle position column (use pos_dfs or primary_position)
    if 'pos' not in predictions_df.columns:
        if 'primary_position' in predictions_df.columns:
            predictions_df['pos'] = predictions_df['primary_position']
        else:
            print("Warning: No position column found. Lineup generation may fail.")

    # Add status column if missing (assume healthy)
    if 'status' not in predictions_df.columns:
        predictions_df['status'] = 'ACTIVE'

    if verbose:
        print(f"\nPrepared predictions:")
        print(f"  Players: {len(predictions_df)}")
        print(f"  Avg projected points: {predictions_df['fppg'].mean():.2f}")
        print(f"  Avg salary: ${predictions_df['salary'].mean():.0f}")

    return predictions_df


def create_contest_config(args):
    """Create contest configuration from command-line args."""
    # Risk strategy mapping
    risk_settings = {
        'conservative': {'variance_weight': 0.0, 'ceiling_weight': 0.3, 'floor_weight': 0.7},
        'balanced': {'variance_weight': 0.0, 'ceiling_weight': 0.5, 'floor_weight': 0.5},
        'aggressive': {'variance_weight': 0.0, 'ceiling_weight': 0.8, 'floor_weight': 0.2}
    }

    config = {
        "contest_type": "custom",
        "name": f"{args.strategy.capitalize()} Strategy",
        "optimization_settings": {
            "num_lineups": args.num_lineups,
            "max_exposure": args.max_exposure,
            "randomness": args.num_lineups > 1,  # Enable randomness for multiple lineups
            "min_salary_cap": args.min_salary
        },
        "risk_settings": risk_settings[args.strategy],
        "player_settings": {
            "min_projected_points": 0,  # No minimum (predictions already filtered)
            "exclude_injured": False,   # Assume predictions already filtered
            "exclude_questionable": False,
            "ownership_settings": {
                "use_projected_ownership": False
            }
        },
        "lineup_rules": {
            "correlation_rules": {
                "stack_rules": {}
            }
        }
    }

    return config


def save_lineups(lineups, output_path, output_format, verbose=False):
    """Save lineups to file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_format == 'json':
        with open(output_path, 'w') as f:
            json.dump(lineups, f, indent=2)
        if verbose:
            print(f"\nSaved {len(lineups)} lineups to {output_path} (JSON)")

    else:  # CSV
        # Flatten lineup dictionaries for CSV export
        rows = []
        for lineup in lineups:
            row = {
                'lineup_id': lineup['lineup_id'],
                'total_salary': lineup['total_salary'],
                'projected_points': lineup['projected_points']
            }

            # Add player info
            for i, player in enumerate(lineup['players'], 1):
                row[f'player_{i}_name'] = player['name']
                row[f'player_{i}_pos'] = player['position']
                row[f'player_{i}_salary'] = player['salary']
                row[f'player_{i}_fppg'] = player['fppg']

            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)

        if verbose:
            print(f"\nSaved {len(lineups)} lineups to {output_path} (CSV)")


def print_lineup_summary(lineups):
    """Print summary of generated lineups."""
    print("\n" + "="*60)
    print("LINEUP GENERATION SUMMARY")
    print("="*60)

    for lineup in lineups:
        print(f"\nLineup #{lineup['lineup_id']}:")
        print(f"  Projected Points: {lineup['projected_points']:.2f}")
        print(f"  Total Salary: ${lineup['total_salary']:,}")
        print(f"  Salary Remaining: ${50000 - lineup['total_salary']:,}")

        print("\n  Players:")
        for player in lineup['players']:
            print(f"    {player['position']:4s} {player['name']:20s} "
                  f"${player['salary']:5,} ({player['fppg']:.2f} fppg)")

    print("\n" + "="*60)


def main():
    args = parse_args()

    # Load predictions
    predictions_df = load_predictions(args.predictions, args.verbose)

    # Create lineup generator
    if args.contest_config:
        generator = LineupGenerator(contest_config_path=args.contest_config)
    else:
        # Create temporary config from args
        import tempfile
        config = create_contest_config(args)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config, f)
            temp_config_path = f.name

        generator = LineupGenerator(contest_config_path=temp_config_path)

        if args.verbose:
            print(f"Using {args.strategy} strategy:")
            print(f"  Num lineups: {args.num_lineups}")
            print(f"  Min salary: ${args.min_salary:,}")
            if args.max_exposure:
                print(f"  Max exposure: {args.max_exposure:.1%}")

    # Generate lineups
    if args.verbose:
        print(f"\nGenerating {args.num_lineups} lineup(s)...")

    lineups = generator.generate_lineups(predictions_df)

    if not lineups:
        print("ERROR: No lineups generated. Check predictions and constraints.")
        sys.exit(1)

    # Print summary
    if args.verbose:
        print_lineup_summary(lineups)
    else:
        print(f"\nGenerated {len(lineups)} lineup(s)")
        for lineup in lineups:
            print(f"  Lineup #{lineup['lineup_id']}: "
                  f"{lineup['projected_points']:.2f} pts, "
                  f"${lineup['total_salary']:,} salary")

    # Save lineups
    if args.output is None:
        output_dir = project_root / 'data' / 'lineups'
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        ext = 'json' if args.output_format == 'json' else 'csv'
        output_path = output_dir / f'lineups_{timestamp}.{ext}'
    else:
        output_path = Path(args.output)

    save_lineups(lineups, output_path, args.output_format, args.verbose)

    print(f"\nLineups saved to: {output_path}")


if __name__ == '__main__':
    main()
