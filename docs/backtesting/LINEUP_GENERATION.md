# Lineup Generation Documentation

## Overview

The lineup generation module integrates `pydfs-lineup-optimizer` with our existing ML prediction pipeline to create optimal DraftKings lineups based on player projections. The system supports various contest types (GPP tournaments, cash games, single/multi-entry) with configurable optimization strategies.

## Architecture

### Core Components

1. **LineupGenerator** (`src/optimization/lineup_generator.py`)
   - Main interface for lineup generation
   - Integrates pydfs-lineup-optimizer with our predictions
   - Supports contest-specific configurations

2. **ContestManager** (`src/optimization/contest_manager.py`)
   - Manages DraftKings contest data
   - Associates lineups with specific contests
   - Tracks player exposure across lineups

3. **BacktestWithLineups** (`src/optimization/backtest_lineup_integration.py`)
   - Extended walk-forward backtest with lineup generation
   - Evaluates lineup performance against actual results
   - Calculates ROI metrics for different contest types

## Installation

```bash
pip install pydfs-lineup-optimizer
```

## Usage

### Basic Lineup Generation

```python
from src.optimization.lineup_generator import LineupGenerator

# Initialize generator with contest config
generator = LineupGenerator(contest_config_path='config/contests/gpp_tournament.json')

# Prepare predictions from your ML model
predictions_df = pd.DataFrame({
    'playerID': [...],
    'playerName': [...],
    'position': [...],  # e.g., 'PG', 'SG/SF'
    'team': [...],
    'salary': [...],
    'predicted_fpts': [...],  # Your model predictions
    'prediction_std': [...]   # Optional: for ceiling/floor calculations
})

# Generate lineups
lineups = generator.generate_lineups(predictions_df)

# Export for DraftKings upload
generator.export_lineups(lineups, 'lineups.csv', format='csv')
```

### Contest Configurations

Contest configurations are stored in `config/contests/`:

- **cash_game.json**: Conservative strategy for 50/50s and double-ups
- **gpp_tournament.json**: Aggressive strategy for large-field tournaments
- **single_entry.json**: Balanced approach for single-entry contests
- **multi_entry.json**: Diverse lineups for multi-entry contests

#### Configuration Structure

```json
{
  "contest_type": "gpp_tournament",
  "optimization_settings": {
    "num_lineups": 20,
    "max_exposure": 0.4,
    "randomness": true,
    "min_salary_cap": 48500,
    "strategy": "ceiling"  // "ceiling", "floor", or "balanced"
  },
  "lineup_rules": {
    "correlation_rules": {
      "stack_rules": {
        "team_stack": {
          "enabled": true,
          "min_players": 2,
          "max_players": 4
        }
      }
    }
  },
  "player_settings": {
    "ownership_settings": {
      "use_projected_ownership": true,
      "fade_threshold": 0.35,
      "chalk_threshold": 0.25
    },
    "min_projected_points": 15.0,
    "exclude_injured": true
  },
  "risk_settings": {
    "variance_weight": 0.3,
    "ceiling_weight": 0.7,
    "floor_weight": 0.0
  }
}
```

### Integration with Walk-Forward Backtest

```python
from src.optimization.backtest_lineup_integration import BacktestWithLineups

# Create backtest with lineup generation
backtest = BacktestWithLineups(
    db_path='nba_dfs.db',
    train_start='20241001',
    train_end='20241130',
    test_start='20241201',
    test_end='20241215',
    per_player_models=True,

    # Lineup generation settings
    generate_lineups=True,
    contest_config='config/contests/gpp_tournament.json',
    num_lineups=20,
    track_lineup_performance=True
)

# Run backtest with lineups
results = backtest.run_with_lineups()

# Results include lineup performance metrics
print(f"Total lineups generated: {results['lineup_summary']['total_lineups']}")
print(f"Lineup correlation: {results['lineup_summary']['avg_correlation']:.3f}")
```

### Command Line Usage

```bash
# Run backtest with lineup generation
python scripts/run_backtest_with_lineups.py \
  --train-start 20241001 \
  --train-end 20241130 \
  --test-start 20241201 \
  --test-end 20241215 \
  --per-player \
  --generate-lineups \
  --contest-config gpp_tournament \
  --num-lineups 20 \
  --track-performance
```

## Contest Management

### Loading Contest Data

```python
from src.optimization.contest_manager import ContestManager

manager = ContestManager()

# Load contests from DraftKings API response
contests_df = manager.load_contests_from_api(contests_data)

# Filter contests
cash_games = manager.filter_contests(
    max_entry_fee=50,
    contest_types=['cash_game']
)

# Associate lineups with contests
lineups_with_contests = manager.associate_lineups_with_contests(
    lineups,
    contest_ids=['C001', 'C002']
)

# Calculate player exposure
exposure_df = manager.calculate_exposure()
```

## Optimization Strategies

### Cash Game Strategy
- **Focus**: Maximize floor (consistency)
- **Settings**:
  - No randomness
  - High minimum salary usage (49500+)
  - Exclude questionable players
  - Single optimal lineup

### GPP Tournament Strategy
- **Focus**: Maximize ceiling (upside)
- **Settings**:
  - Enable randomness for diversity
  - Lower minimum salary (48500)
  - Include game/team stacks
  - Generate 20-150 unique lineups
  - Limit player exposure (40% max)

### Single Entry Strategy
- **Focus**: Balanced approach
- **Settings**:
  - Mix of floor and ceiling
  - Moderate stacking
  - Single lineup
  - Consider ownership

## Advanced Features

### Custom Fantasy Points Strategy

```python
from src.optimization.lineup_generator import CustomFantasyPointsStrategy

# Create custom strategy with risk weights
strategy = CustomFantasyPointsStrategy({
    'variance_weight': 0.2,
    'ceiling_weight': 0.7,
    'floor_weight': 0.1
})

optimizer.set_fantasy_points_strategy(strategy)
```

### Player Locking and Exclusion

```python
# Lock specific players
optimizer.add_player_to_lineup(player)

# Set exposure limits
player.max_exposure = 0.4  # Max 40% of lineups
player.min_exposure = 0.1  # Min 10% of lineups

# Remove injured players
optimizer.remove_player(injured_player)
```

### Stacking Rules

```python
# Team stack (2-4 players from same team)
team_stack = TeamStack(min_players=2, max_players=4)
optimizer.add_stack(team_stack)

# Game stack (players from both teams in a game)
game_stack = GameStack(min_players_per_team=1)
optimizer.add_stack(game_stack)
```

## Performance Metrics

The system tracks various performance metrics:

- **Projection Accuracy**: MAPE, RMSE, correlation between projected and actual scores
- **Lineup Performance**: Average score, best/worst lineup, cash line analysis
- **ROI Analysis**: Performance by contest type and entry fee
- **Player Exposure**: Usage rates across lineups

## Export Formats

### DraftKings CSV Format
```csv
P1,P2,P3,P4,P5,P6,P7,P8
LeBron James (001),Anthony Davis (002),...
```

### JSON Format (Full Details)
```json
{
  "lineup_num": 1,
  "players": [
    {
      "playerID": "001",
      "playerName": "LeBron James",
      "position": "PG",
      "team": "LAL",
      "salary": 11000,
      "projected_fpts": 55.5
    },
    ...
  ],
  "total_salary": 49500,
  "projected_points": 285.3,
  "contest_type": "gpp_tournament"
}
```

## Troubleshooting

### Common Issues

1. **"Can't generate lineups" error**
   - Not enough eligible players for all positions
   - Salary constraints too restrictive
   - Increase player pool or adjust minimum projections

2. **Low lineup diversity**
   - Increase randomness setting
   - Adjust max_exposure limits
   - Use different optimization strategies

3. **Poor lineup performance**
   - Review prediction accuracy
   - Adjust risk settings for contest type
   - Consider ownership in GPPs

## Best Practices

1. **Data Quality**
   - Ensure accurate injury status
   - Update salaries before lineup generation
   - Validate predictions are reasonable

2. **Contest Selection**
   - Use appropriate configuration for contest type
   - Consider field size and payout structure
   - Adjust exposure based on contest size

3. **Risk Management**
   - Diversify lineups in multi-entry
   - Track bankroll and ROI
   - Adjust strategies based on results

4. **Performance Monitoring**
   - Track lineup performance over time
   - Analyze which strategies work best
   - Continuously refine configurations

## Future Enhancements

- Late swap optimization
- Ownership projection integration
- Multi-sport support
- Real-time lineup adjustments
- Advanced correlation modeling
- Automated contest selection