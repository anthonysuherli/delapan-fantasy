from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).parent.parent.parent
DB_PATH = os.getenv('NBA_DFS_DB_PATH', str(PROJECT_ROOT / 'nba_dfs.db'))
DATA_DIR = os.getenv('NBA_DFS_DATA_DIR', str(PROJECT_ROOT / 'data'))
MODEL_DIR = Path(DATA_DIR) / 'models'
PER_PLAYER_MODEL_DIR = MODEL_DIR / 'per_player'
PER_SLATE_MODEL_DIR = MODEL_DIR / 'per_slate'
TRAINING_INPUTS_DIR = MODEL_DIR / 'training_inputs'
PER_PLAYER_TRAINING_INPUTS_DIR = TRAINING_INPUTS_DIR / 'by_player'
PER_SLATE_TRAINING_INPUTS_DIR = TRAINING_INPUTS_DIR / 'by_slate'
