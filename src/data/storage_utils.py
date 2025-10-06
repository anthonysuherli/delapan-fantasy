from pathlib import Path
from typing import Optional
from datetime import datetime


def get_partitioned_path(
    base_dir: str,
    date: str,
    filename: Optional[str] = None,
    date_format: str = '%Y%m%d'
) -> Path:
    """
    Generate partitioned file path based on date.

    Args:
        base_dir: Base directory for data type
        date: Date string in YYYYMMDD or datetime format
        filename: Optional filename (defaults to day.parquet)
        date_format: Format of date string

    Returns:
        Path object with year/month/day structure

    Examples:
        get_partitioned_path('data/dfs_salaries', '20250119')
        -> data/dfs_salaries/2025/01/19.parquet

        get_partitioned_path('data/player_logs', '20211019', 'BKN@MIL.parquet')
        -> data/player_logs/2021/10/19/BKN@MIL.parquet
    """
    if isinstance(date, str):
        dt = datetime.strptime(date, date_format)
    else:
        dt = date

    year = dt.strftime('%Y')
    month = dt.strftime('%m')
    day = dt.strftime('%d')

    base = Path(base_dir)
    partitioned_dir = base / year / month

    if filename is None:
        filename = f'{day}.parquet'

    if filename.endswith('.parquet') and '/' not in filename and '@' not in filename:
        return partitioned_dir / filename
    else:
        return partitioned_dir / day / filename


def get_all_files_in_date_range(
    base_dir: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    pattern: str = '*.parquet',
    date_format: str = '%Y%m%d'
) -> list[Path]:
    """
    Get all files in date range from partitioned structure.

    Args:
        base_dir: Base directory for data type
        start_date: Start date in YYYYMMDD format (None = all)
        end_date: End date in YYYYMMDD format (None = all)
        pattern: File pattern to match
        date_format: Format of date strings

    Returns:
        List of Path objects matching criteria
    """
    base = Path(base_dir)

    if not base.exists():
        return []

    all_files = []

    if start_date is None and end_date is None:
        return list(base.rglob(pattern))

    start_dt = datetime.strptime(start_date, date_format) if start_date else None
    end_dt = datetime.strptime(end_date, date_format) if end_date else None

    for year_dir in sorted(base.glob('*')):
        if not year_dir.is_dir() or not year_dir.name.isdigit():
            continue

        for month_dir in sorted(year_dir.glob('*')):
            if not month_dir.is_dir() or not month_dir.name.isdigit():
                continue

            for day_item in sorted(month_dir.glob('*')):
                try:
                    if day_item.is_file() and day_item.match(pattern):
                        file_date = datetime(
                            int(year_dir.name),
                            int(month_dir.name),
                            int(day_item.stem.split('_')[0])
                        )
                        if (start_dt is None or file_date >= start_dt) and \
                           (end_dt is None or file_date <= end_dt):
                            all_files.append(day_item)

                    elif day_item.is_dir() and day_item.name.isdigit():
                        day_dir = day_item
                        file_date = datetime(
                            int(year_dir.name),
                            int(month_dir.name),
                            int(day_dir.name)
                        )
                        if (start_dt is None or file_date >= start_dt) and \
                           (end_dt is None or file_date <= end_dt):
                            all_files.extend(day_dir.glob(pattern))

                except (ValueError, IndexError):
                    continue

    return sorted(all_files)
