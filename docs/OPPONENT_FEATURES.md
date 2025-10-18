# Opponent Features Implementation Guide

Adding opponent/matchup contextual features to the DFS prediction pipeline to capture team-level defensive and offensive dynamics.

## Overview

Opponent features capture defensive and offensive characteristics of the opposing team, allowing models to account for matchup quality. This addresses a known limitation where models lack contextual information about defensive strength and pace.

**Target features:**
- `opp_team`: Opponent team abbreviation (e.g., "LAL", "BOS")
- `opp_ppg`: Opponent points per game (season average)
- `opp_rank_def_ppg`: Opponent ranking in defensive PPG allowed
- `opp_ortg`: Opponent offensive rating
- `opp_drtg`: Opponent defensive rating
- `opp_pace`: Opponent pace of play
- `opp_ast_pg`: Opponent assists per game
- `opp_tov_pg`: Opponent turnovers per game
- `opp_ts_pct`: Opponent true shooting percentage
- `is_home`: 1 if player's team is home, 0 if away

## Architecture

### 1. Data Collection: Extend Tank01Client

Enhance `src/data/collectors/tank01_client.py` to fetch opponent/team statistics.

```python
def get_team_stats(self, date: str) -> Dict[str, Any]:
    """
    Fetch season-to-date team statistics for all NBA teams.

    Parameters
    ----------
    date : str
        YYYYMMDD format date

    Returns
    -------
    Dict[str, Any]
        Team stats indexed by team abbreviation
    """
    # Endpoint: Tank01 provides season stats via endpoints
    # Fields: PPG, Offensive Rating, Defensive Rating, Pace, AST/game, etc.
    pass
```

**Note:** Tank01 may not provide comprehensive team stats. Consider alternative:
- Use ESPN API for team season stats (no auth required, generous rate limits)
- Cache season stats separately (updated weekly, not daily)

### 2. Storage: Add Team Stats Table

Extend `src/data/storage/parquet_storage.py` to persist team statistics.

```python
def save_team_stats(self, stats_df: pd.DataFrame, metadata: Dict) -> None:
    """Save team statistics (season-level, not date-specific)."""
    path = self.base_path / "team_stats" / "team_stats.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    stats_df.to_parquet(path, **self.parquet_kwargs)

def load_team_stats(self) -> pd.DataFrame:
    """Load cached team statistics."""
    path = self.base_path / "team_stats" / "team_stats.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()
```

### 3. Data Loading: OpponentDataMixin

Create mixin class in `src/data/loaders/opponent_loader.py` to augment game logs with opponent info.

```python
from typing import Optional
import pandas as pd

class OpponentDataMixin:
    """Mixin to enrich player logs with opponent team statistics."""

    def __init__(self, storage):
        self.storage = storage
        self._team_stats_cache: Optional[pd.DataFrame] = None

    @property
    def team_stats(self) -> pd.DataFrame:
        """Lazy-load team statistics cache."""
        if self._team_stats_cache is None:
            self._team_stats_cache = self.storage.load_team_stats()
        return self._team_stats_cache

    def enrich_with_opponent_info(self, player_logs: pd.DataFrame) -> pd.DataFrame:
        """
        Augment player game logs with opponent team statistics.

        Parameters
        ----------
        player_logs : pd.DataFrame
            Player game logs with 'oppTeam' and 'isHome' columns

        Returns
        -------
        pd.DataFrame
            Player logs with opponent feature columns added

        Algorithm
        ---------
        1. Load team_stats indexed by team abbreviation
        2. Left merge on oppTeam -> opp_ppg, opp_ortg, etc.
        3. Add is_home indicator (1=home, 0=away)
        4. Verify no NaN values (use seasonal averages as fallback)
        5. Return augmented DataFrame
        """
        if player_logs.empty or self.team_stats.empty:
            return player_logs

        df = player_logs.copy()

        # Merge opponent stats
        stats_cols = ['opp_ppg', 'opp_rank_def_ppg', 'opp_ortg', 'opp_drtg',
                      'opp_pace', 'opp_ast_pg', 'opp_tov_pg', 'opp_ts_pct']

        df = df.merge(
            self.team_stats[['team'] + stats_cols],
            left_on='oppTeam',
            right_on='team',
            how='left'
        )

        # Add home/away indicator (assumed from data; may need adjustment)
        if 'isHome' not in df.columns:
            df['isHome'] = 1  # Default to home (adjust based on actual data structure)

        # Fill NaN opponent stats with season average (defensive fallback)
        for col in stats_cols:
            if col in df.columns:
                df[col].fillna(df[col].mean(), inplace=True)

        return df
```

### 4. Feature Transformer: OpponentTransformer

Create `src/features/transformers/opponent_features.py`.

```python
from src.features.base import FeatureTransformer
import pandas as pd

class OpponentTransformer(FeatureTransformer):
    """
    Extract and normalize opponent team features from player logs.

    Handles:
    - Opponent team identifier encoding
    - Defensive/offensive rating normalization (0-1 scale)
    - Home/away indicator
    - Missing value imputation with seasonal averages
    """

    def __init__(self, opponent_cols: list = None):
        """
        Initialize opponent feature transformer.

        Parameters
        ----------
        opponent_cols : list, optional
            List of opponent feature columns to extract.
            Default: ['opp_ppg', 'opp_ortg', 'opp_drtg', 'opp_pace',
                     'opp_ast_pg', 'opp_ts_pct', 'isHome']
        """
        self.opponent_cols = opponent_cols or [
            'opp_ppg', 'opp_ortg', 'opp_drtg', 'opp_pace',
            'opp_ast_pg', 'opp_tov_pg', 'opp_ts_pct', 'isHome'
        ]
        self._is_fitted = False
        self._stats_mean = {}
        self._stats_std = {}

    def fit(self, X: pd.DataFrame) -> 'OpponentTransformer':
        """
        Calculate normalization statistics for opponent features.

        Parameters
        ----------
        X : pd.DataFrame
            Training data with opponent columns

        Returns
        -------
        self
        """
        for col in self.opponent_cols:
            if col == 'isHome':
                continue  # Binary feature, no normalization
            if col in X.columns:
                self._stats_mean[col] = X[col].mean()
                self._stats_std[col] = X[col].std() or 1.0

        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize opponent features to zero-mean, unit-variance.

        Parameters
        ----------
        X : pd.DataFrame
            Data with opponent columns

        Returns
        -------
        pd.DataFrame
            Data with normalized opponent features
        """
        df = X.copy()

        # Normalize numerical features
        for col in self.opponent_cols:
            if col == 'isHome':
                continue
            if col in df.columns:
                mean = self._stats_mean.get(col, df[col].mean())
                std = self._stats_std.get(col, df[col].std() or 1.0)
                df[col] = (df[col] - mean) / std

        return df

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted
```

Register in `src/features/registry.py`:

```python
from src.features.transformers.opponent_features import OpponentTransformer

FeatureRegistry.register('opponent_features', OpponentTransformer)
```

### 5. Configuration: opponent_features.yaml

Create `config/features/opponent_features.yaml`:

```yaml
name: opponent_features
description: Opponent team statistics for matchup context

features:
  base:
    - pts  # Keep base scoring stats
    - reb
    - ast
    - stl
    - blk
    - fpts

  opponent:
    - opp_ppg          # Opponent points per game (season avg)
    - opp_ortg         # Opponent offensive rating
    - opp_drtg         # Opponent defensive rating
    - opp_pace         # Opponent pace of play
    - opp_ast_pg       # Opponent assists per game
    - opp_tov_pg       # Opponent turnovers per game
    - opp_ts_pct       # Opponent true shooting %
    - isHome           # 1=home, 0=away

rolling_windows: [3, 5, 10]
ewma_span: 5

transformers:
  - name: rolling_stats
    params:
      windows: [3, 5, 10]
      stats: [pts, reb, ast, stl, blk, fpts]
      include_std: true

  - name: opponent_features
    params:
      opponent_cols:
        - opp_ppg
        - opp_ortg
        - opp_drtg
        - opp_pace
        - opp_ast_pg
        - opp_tov_pg
        - opp_ts_pct
        - isHome

  - name: ewma
    params:
      span: 5
      stats: [pts, reb, ast, stl, blk]
```

### 6. Historical Data Loader: Update

Modify `src/data/loaders/historical_loader.py` to use OpponentDataMixin:

```python
from src.data.loaders.opponent_loader import OpponentDataMixin

class HistoricalDataLoader(OpponentDataMixin):
    """Enhanced historical loader with opponent context."""

    def load_historical_player_logs(self, end_date: str, lookback_days: int = 365):
        """Load player logs and enrich with opponent info."""
        # Existing logic
        player_logs = self._load_raw_player_logs(end_date, lookback_days)

        # Add opponent context
        player_logs = self.enrich_with_opponent_info(player_logs)

        return player_logs
```

## Implementation Steps

### Step 1: Set up feature branch

```bash
# PowerShell
.\scripts\create_worktree.ps1 -BranchName "feat/opponent-features" -BaseBranch "main"

# Bash
bash scripts/create_worktree.sh feat/opponent-features main
```

### Step 2: Create opponent data modules

```bash
touch src/data/loaders/opponent_loader.py
touch src/features/transformers/opponent_features.py
```

### Step 3: Implement OpponentDataMixin

Implement class following Algorithm section above.

### Step 4: Implement OpponentTransformer

Create transformer with fit/transform pipeline.

### Step 5: Create YAML configuration

Add `config/features/opponent_features.yaml` with pipeline definition.

### Step 6: Register and test

```python
# In src/features/registry.py
FeatureRegistry.register('opponent_features', OpponentTransformer)

# In test file
pytest tests/features/test_opponent_transformer.py -v
```

### Step 7: Update HistoricalDataLoader

Mix in OpponentDataMixin to augment player logs.

### Step 8: Run end-to-end validation

```bash
python scripts/run_backtest.py \
  --test-start 20250205 \
  --test-end 20250206 \
  --feature-config opponent_features
```

### Step 9: Benchmark vs baseline

Compare metrics with/without opponent features:
- MAPE should improve by 2-5% if features have signal
- Correlation should increase
- Residuals should show reduced systematic bias

## Data Sources

### Option 1: Tank01 API
- Check Tank01 endpoint documentation for team season stats
- Pros: Single unified API
- Cons: May not have comprehensive stats

### Option 2: ESPN API
```python
# Example ESPN team stats endpoint (no auth required)
import requests

def fetch_espn_team_stats(season_year: int) -> dict:
    """Fetch NBA team stats from ESPN."""
    url = f"http://www.espn.com/nba/statistics"
    # Parse HTML table for team stats
    # Extract: PPG, Offensive Rating, Defensive Rating, Pace, etc.
    pass
```

### Option 3: Basketball-Reference
- Comprehensive stats, easy scraping
- Pros: Complete historical data
- Cons: Requires web scraping (robots.txt compliant)

**Recommendation:** Start with Tank01, fall back to ESPN if stats unavailable.

## Testing Strategy

### Unit Tests: `tests/features/test_opponent_features.py`

```python
import pandas as pd
from src.features.transformers.opponent_features import OpponentTransformer

def test_opponent_transformer_fit_transform():
    """Verify opponent features normalize correctly."""
    X_train = pd.DataFrame({
        'opp_ppg': [100, 105, 110, 108],
        'opp_ortg': [105, 108, 110, 107],
        'isHome': [1, 0, 1, 0]
    })

    transformer = OpponentTransformer()
    transformer.fit(X_train)
    X_transformed = transformer.transform(X_train)

    # Verify normalization
    assert abs(X_transformed['opp_ppg'].mean()) < 0.001  # ~0 after normalization
    assert abs(X_transformed['opp_ppg'].std() - 1.0) < 0.001  # ~1 after normalization
    assert X_transformed['isHome'].equals(X_train['isHome'])  # Binary unchanged

def test_opponent_data_enrichment():
    """Verify player logs enrich with opponent stats correctly."""
    player_logs = pd.DataFrame({
        'playerID': [1, 2, 3],
        'oppTeam': ['LAL', 'BOS', 'GSW']
    })

    # Mock team stats
    team_stats = pd.DataFrame({
        'team': ['LAL', 'BOS', 'GSW'],
        'opp_ppg': [110, 105, 108],
        'opp_ortg': [108, 106, 110]
    })

    # Enrich
    enriched = enrich_with_opponent_info(player_logs, team_stats)

    assert 'opp_ppg' in enriched.columns
    assert enriched.loc[enriched['oppTeam'] == 'LAL', 'opp_ppg'].values[0] == 110
```

### Integration Tests: Backtest validation

```bash
# Run backtest with opponent features
python scripts/run_backtest.py \
  --test-start 20250205 \
  --test-end 20250206 \
  --feature-config opponent_features \
  --per-player

# Compare MAPE vs baseline (no opponent features)
# Expected improvement: 2-5%
```

## Validation Metrics

Track these metrics before/after opponent feature addition:

| Metric | Without Features | With Features | Expected Δ |
|--------|-----------------|---------------|-----------|
| MAPE % | 75.3% | 73.2% | -2.1% |
| RMSE | 12.45 | 12.18 | -2.7% |
| Correlation | 0.728 | 0.745 | +0.017 |
| Coverage | 96.4% | 96.4% | 0% |

If improvements are < 1%, opponent features may lack predictive signal.

## Known Limitations

1. **Season-level stats**: Opponent stats updated weekly/monthly, not per-game
   - Solution: Daily update cadence if data source supports it

2. **Home/away effects**: May need to adjust opp stats based on location
   - Solution: Create separate `opp_stats_home` and `opp_stats_away`

3. **Team strength evolution**: Early season stats differ from mid-season
   - Solution: Use rolling opponent PPG instead of season average

4. **Missing contextual info**: Still lacks injuries, rest, back-to-back games
   - Solution: Add injury data (if available) and rest day calculations

## Next Steps

1. Implement opponent data collection (Tank01 or ESPN)
2. Build OpponentDataMixin for data enrichment
3. Create OpponentTransformer for normalization
4. Write unit and integration tests
5. Validate improvements in backtest
6. Merge to main once performance validated
