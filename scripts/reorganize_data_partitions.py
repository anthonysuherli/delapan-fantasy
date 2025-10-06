import os
import re
import shutil
from pathlib import Path


def reorganize_player_logs(base_path):
    player_logs_dir = base_path / 'player_logs'
    if not player_logs_dir.exists():
        return

    pattern = re.compile(r'player_logs_(\d{4})(\d{2})(\d{2})_(.+)\.parquet')

    for file in player_logs_dir.glob('player_logs_*.parquet'):
        match = pattern.match(file.name)
        if match:
            year, month, day, game_id = match.groups()
            new_dir = player_logs_dir / year / month / day
            new_dir.mkdir(parents=True, exist_ok=True)
            new_path = new_dir / f'{game_id}.parquet'
            print(f'Moving {file} -> {new_path}')
            shutil.move(str(file), str(new_path))


def reorganize_date_only_files(base_path, subdirs):
    for subdir in subdirs:
        data_dir = base_path / subdir
        if not data_dir.exists():
            continue

        pattern = re.compile(rf'{subdir}_(\d{{4}})(\d{{2}})(\d{{2}})\.parquet')

        for file in data_dir.glob(f'{subdir}_*.parquet'):
            match = pattern.match(file.name)
            if match:
                year, month, day = match.groups()
                new_dir = data_dir / year / month
                new_dir.mkdir(parents=True, exist_ok=True)
                new_path = new_dir / f'{day}.parquet'
                print(f'Moving {file} -> {new_path}')
                shutil.move(str(file), str(new_path))


def reorganize_games(base_path):
    games_dir = base_path / 'games'
    if not games_dir.exists():
        return

    pattern = re.compile(r'games_(\d{4})(\d{2})(\d{2})\.parquet')

    for file in games_dir.glob('games_*.parquet'):
        match = pattern.match(file.name)
        if match:
            year, month, day = match.groups()
            new_dir = games_dir / year / month
            new_dir.mkdir(parents=True, exist_ok=True)
            new_path = new_dir / f'{day}.parquet'
            print(f'Moving {file} -> {new_path}')
            shutil.move(str(file), str(new_path))


def reorganize_injuries(base_path):
    injuries_dir = base_path / 'injuries'
    if not injuries_dir.exists():
        return

    pattern = re.compile(r'injuries_(\d{4})(\d{2})(\d{2})_to_(\d{4})(\d{2})(\d{2})\.parquet')

    for file in injuries_dir.glob('injuries_*.parquet'):
        match = pattern.match(file.name)
        if match:
            year_start, month_start, day_start, year_end, month_end, day_end = match.groups()
            new_dir = injuries_dir / year_start / month_start
            new_dir.mkdir(parents=True, exist_ok=True)
            new_path = new_dir / f'{day_start}_to_{year_end}{month_end}{day_end}.parquet'
            print(f'Moving {file} -> {new_path}')
            shutil.move(str(file), str(new_path))


def main():
    base_path = Path('C:/Users/antho/OneDrive/Documents/Repositories/delapan-fantasy/data')

    print('Reorganizing player logs...')
    reorganize_player_logs(base_path)

    print('\nReorganizing date-only files...')
    reorganize_date_only_files(base_path, [
        'dfs_salaries',
        'projections',
        'schedule',
        'betting_odds'
    ])

    print('\nReorganizing games...')
    reorganize_games(base_path)

    print('\nReorganizing injuries...')
    reorganize_injuries(base_path)

    print('\nReorganization complete.')


if __name__ == '__main__':
    main()
