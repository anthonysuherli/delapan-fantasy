# Player Filtering Guide

Filter players by name and ID using flexible, composable filtering components.

## Quick Start

### Filter by Player Name

```python
from src.filters import PlayerNameFilter
import pandas as pd

# Load player data
players = pd.read_parquet('data/inputs/box_scores/2025-02-05.parquet')

# Filter by exact name
filter_obj = PlayerNameFilter('LeBron James')
result = filter_obj.apply(players)

# Filter by partial name (case-insensitive)
filter_obj = PlayerNameFilter('LeBron', case_sensitive=False)
result = filter_obj.apply(players)

# Filter by multiple names
filter_obj = PlayerNameFilter(['LeBron James', 'Kevin Durant'])
result = filter_obj.apply(players)
```

### Filter by Player ID

```python
from src.filters import PlayerIDFilter

# Single ID
filter_obj = PlayerIDFilter(101)
result = filter_obj.apply(players)

# Multiple IDs
filter_obj = PlayerIDFilter([101, 102, 103])
result = filter_obj.apply(players)

# String IDs (flexible)
filter_obj = PlayerIDFilter(['101', '102'])
result = filter_obj.apply(players)
```

### Filter from CSV File

Create a CSV with player IDs:

```csv
playerID
101
103
105
```

```python
from src.filters import PlayerIDFromCSVFilter

filter_obj = PlayerIDFromCSVFilter('players.csv')
result = filter_obj.apply(players)

# Custom column name
filter_obj = PlayerIDFromCSVFilter('players.csv', id_column='player_id')
result = filter_obj.apply(players)
```

## Advanced Usage: Filter Registry

Compose multiple filters with AND or OR logic.

### Sequential Filters (AND Logic)

```python
from src.filters import PlayerFilterRegistry

registry = PlayerFilterRegistry()
registry.add_name_filter('James')      # Filter: name contains "James"
registry.add_id_filter([101, 102])     # Filter: ID is 101 or 102

result = registry.apply(players)
# Result: Only LeBron James (ID 101) matches both conditions
```

### Union Filters (OR Logic)

```python
registry = PlayerFilterRegistry()
registry.add_name_filter('LeBron')     # Match names containing "LeBron"
registry.add_id_filter([103])          # OR match ID 103 (Giannis)

result = registry.apply_any(players)
# Result: LeBron James OR Giannis (union of both filters)
```

### Method Chaining

```python
result = (PlayerFilterRegistry()
          .add_name_filter('James')
          .add_id_filter([101])
          .add_csv_filter('players.csv')
          .apply(players))
```

### Clear Filters

```python
registry = PlayerFilterRegistry()
registry.add_name_filter('LeBron')
registry.clear()

result = registry.apply(players)  # Returns all players (no filters applied)
```

## API Reference

### PlayerNameFilter

Filter players by exact or partial name matching.

```python
PlayerNameFilter(
    names: Union[List[str], str],
    case_sensitive: bool = False
)
```

**Parameters:**
- `names`: Single name or list of names to match
- `case_sensitive`: If False, ignores case in matching (default: False)

**Column Detection:** Looks for `playerName`, `name`, `Player`, or `player_name`

**Example:**
```python
filter_obj = PlayerNameFilter('LeBron', case_sensitive=False)
result = filter_obj.apply(data)
```

### PlayerIDFilter

Filter players by exact ID match.

```python
PlayerIDFilter(
    player_ids: Union[List[int], List[str], int, str]
)
```

**Parameters:**
- `player_ids`: Single ID or list of IDs (int or str)

**Column Detection:** Looks for `playerID`, `id`, `playerId`, or `player_id`

**Example:**
```python
filter_obj = PlayerIDFilter([101, 102, 103])
result = filter_obj.apply(data)
```

### PlayerIDFromCSVFilter

Load player IDs from CSV file and filter.

```python
PlayerIDFromCSVFilter(
    csv_path: Union[str, Path],
    id_column: str = 'playerID'
)
```

**Parameters:**
- `csv_path`: Path to CSV file
- `id_column`: Column name in CSV containing IDs (default: 'playerID')

**Raises:**
- `FileNotFoundError`: If CSV file not found
- `ValueError`: If specified column not found in CSV

**Example:**
```python
filter_obj = PlayerIDFromCSVFilter('players.csv', id_column='player_id')
result = filter_obj.apply(data)
```

### PlayerFilterRegistry

Compose multiple filters with AND or OR logic.

#### Methods

**add_filter(filter_obj: PlayerFilter) -> PlayerFilterRegistry**
- Add a filter instance directly

**add_name_filter(names, case_sensitive=False) -> PlayerFilterRegistry**
- Add name filter

**add_id_filter(player_ids) -> PlayerFilterRegistry**
- Add ID filter

**add_csv_filter(csv_path, id_column='playerID') -> PlayerFilterRegistry**
- Add CSV-based ID filter

**apply(data: pd.DataFrame) -> pd.DataFrame**
- Apply all filters sequentially (AND logic)

**apply_any(data: pd.DataFrame) -> pd.DataFrame**
- Apply all filters with union (OR logic)

**clear() -> PlayerFilterRegistry**
- Remove all filters

**Example:**
```python
registry = PlayerFilterRegistry()
registry.add_name_filter('LeBron').add_id_filter([101])
result = registry.apply(data)
```

## CSV Format Examples

### Standard Format

```csv
playerID
101
102
103
```

### Custom Column Name

```csv
player_id,player_name
101,LeBron James
102,Kevin Durant
103,Giannis Antetokounmpo
```

Load with:
```python
PlayerIDFromCSVFilter('players.csv', id_column='player_id')
```

### Multiple Columns (uses specified column only)

```csv
playerID,salary,team
101,11000,LAL
102,10500,PHX
103,10000,MIL
```

Load with:
```python
PlayerIDFromCSVFilter('players.csv')  # Uses 'playerID' column
```

## Integration with Backtesting

Use filters in walk-forward backtesting:

```python
from src.walk_forward_backtest import WalkForwardBacktest
from src.filters import PlayerFilterRegistry

# Create filter registry
filters = PlayerFilterRegistry()
filters.add_csv_filter('elite_players.csv')

# Pass to backtest
backtest = WalkForwardBacktest(
    filters=filters,
    # ... other parameters
)

results = backtest.run()
```

## Error Handling

All filters validate input and provide clear error messages:

```python
from src.filters import PlayerIDFromCSVFilter

# File not found
try:
    filter_obj = PlayerIDFromCSVFilter('/missing/file.csv')
except FileNotFoundError as e:
    print(f"Error: {e}")

# Column not found
try:
    filter_obj = PlayerIDFromCSVFilter('data.csv', id_column='nonexistent')
except ValueError as e:
    print(f"Error: {e}")

# Name column not found in data
try:
    filter_obj = PlayerNameFilter('LeBron')
    result = filter_obj.apply(data_without_name_column)
except ValueError as e:
    print(f"Error: {e}")
```

## Performance Considerations

- **Name filtering**: O(n) string comparison. Partial matches use regex.
- **ID filtering**: O(n) with set lookup. Optimal for large lists.
- **CSV filtering**: O(1) loading for CSV, O(n) for data filtering.
- **Registry**: Sequential application. Filters earlier in chain reduce data for later filters.

### Optimization Tip

Order filters by selectivity (most restrictive first):

```python
# Good: IDs are most selective, applied first
registry = (PlayerFilterRegistry()
            .add_id_filter([101, 102])           # Reduces from 500 to 2
            .add_name_filter('James'))           # Then filter by name

# Less efficient: Name filter applied first
registry = (PlayerFilterRegistry()
            .add_name_filter('James')            # Reduces from 500 to 50
            .add_id_filter([101, 102]))          # Then filter IDs
```

## Testing

Run the test suite:

```bash
pytest tests/filters/test_player_filters.py -v
```

All 25 tests pass, covering:
- Exact and partial name matching
- Case sensitivity
- Multiple ID filtering
- CSV file loading
- Filter composition (AND/OR logic)
- Error handling
- Edge cases (missing columns, empty results, null values)
