import sys
from pathlib import Path

repo_root = Path(__file__).parent
sys.path.insert(0, str(repo_root))

from src.walk_forward_backtest import WalkForwardBacktest

backtest = WalkForwardBacktest(
    db_path='nba_dfs.db',
    start_date='20240115',
    end_date='20240117',
    lookback_days=90,
    model_type='xgboost',
    model_params={
        'n_estimators': 200,
        'max_depth': 6,
        'learning_rate': 0.1,
        'random_state': 42
    },
    rolling_window_sizes=[3, 5, 10],
    output_dir='data/backtest_results',
    per_player_models=True,
    min_player_games=10
)

print("Running backtest with per-player models...")
print("Models will be saved to: data/models/YYYY/MM/DD/")
print("")

results = backtest.run()

if 'error' not in results:
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"Slates: {results['num_slates']}")
    print(f"Mean MAPE: {results['mean_mape']:.2f}%")
    print(f"Mean RMSE: {results['mean_rmse']:.2f}")
    print(f"Mean Correlation: {results['mean_correlation']:.3f}")

    models_dir = Path('data/models')
    if models_dir.exists():
        model_files = list(models_dir.rglob('*.pkl'))
        print(f"\nTotal models saved: {len(model_files)}")
        print(f"\nModel locations:")
        for year_dir in sorted(models_dir.iterdir()):
            if year_dir.is_dir():
                for month_dir in sorted(year_dir.iterdir()):
                    if month_dir.is_dir():
                        for day_dir in sorted(month_dir.iterdir()):
                            if day_dir.is_dir():
                                count = len(list(day_dir.glob('*.pkl')))
                                if count > 0:
                                    print(f"  {year_dir.name}/{month_dir.name}/{day_dir.name}: {count} models")
else:
    print(f"ERROR: {results['error']}")
