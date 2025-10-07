import sys
from pathlib import Path
import logging

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from src.walk_forward_backtest import WalkForwardBacktest

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def test_per_player_models():
    print("="*80)
    print("TESTING PER-PLAYER MODEL GENERATION")
    print("="*80)

    db_path = str(repo_root / 'nba_dfs.db')

    backtest = WalkForwardBacktest(
        db_path=db_path,
        start_date='20240115',
        end_date='20240115',
        lookback_days=90,
        model_type='xgboost',
        model_params={
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'random_state': 42
        },
        rolling_window_sizes=[3, 5, 10],
        output_dir='data/backtest_results',
        per_player_models=True,
        min_player_games=10
    )

    print("\nRunning backtest for single date to test model saving...")
    results = backtest.run()

    if 'error' not in results:
        print("\n" + "="*80)
        print("BACKTEST COMPLETED SUCCESSFULLY")
        print("="*80)
        print(f"Slates processed: {results['num_slates']}")
        print(f"Players evaluated: {results['total_players_evaluated']:.0f}")

        models_dir = Path('data/models')
        if models_dir.exists():
            model_files = list(models_dir.rglob('*.pkl'))
            metadata_files = list(models_dir.rglob('*.json'))

            print(f"\nModels saved: {len(model_files)}")
            print(f"Metadata files: {len(metadata_files)}")

            if model_files:
                print(f"\nSample model paths:")
                for model_file in model_files[:5]:
                    print(f"  {model_file}")
        else:
            print("\nWARNING: No models directory found")
    else:
        print(f"\nERROR: {results['error']}")

if __name__ == '__main__':
    test_per_player_models()
