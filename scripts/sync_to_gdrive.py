import argparse
import shutil
import logging
from pathlib import Path
from typing import List

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def sync_directory(source: Path, dest: Path, file_pattern: str = '*.parquet') -> int:
    """
    Sync files from source to destination.

    Args:
        source: Source directory
        dest: Destination directory
        file_pattern: File pattern to match

    Returns:
        Number of files copied
    """
    if not source.exists():
        logger.warning(f"Source directory does not exist: {source}")
        return 0

    dest.mkdir(parents=True, exist_ok=True)

    files = list(source.glob(file_pattern))
    copied = 0

    for file_path in files:
        dest_file = dest / file_path.name

        if dest_file.exists():
            if file_path.stat().st_mtime <= dest_file.stat().st_mtime:
                logger.debug(f"Skipping {file_path.name} (already up to date)")
                continue

        logger.info(f"Copying {file_path.name}")
        shutil.copy2(file_path, dest_file)
        copied += 1

    return copied


def sync_project_to_gdrive(
    local_dir: str,
    gdrive_dir: str,
    include_db: bool = True
) -> None:
    """
    Sync NBA DFS data to Google Drive.

    Code is cloned from GitHub in Colab, only data is synced to Drive.

    Args:
        local_dir: Local project directory
        gdrive_dir: Google Drive mount point
        include_db: If True, also sync SQLite database
    """
    local_root = Path(local_dir)
    gdrive_root = Path(gdrive_dir) / 'nba_dfs'

    logger.info(f"Syncing data from {local_root} to {gdrive_root}")
    logger.info("(Code will be cloned from GitHub in Colab)")

    gdrive_root.mkdir(parents=True, exist_ok=True)

    logger.info("\nSyncing data files...")
    data_dirs = ['box_scores', 'dfs_salaries', 'betting_odds', 'schedule', 'injuries', 'projections']

    total_copied = 0
    for data_type in data_dirs:
        source = local_root / 'data' / 'inputs' / data_type
        dest = gdrive_root / 'data' / 'inputs' / data_type

        if source.exists():
            copied = sync_directory(source, dest)
            logger.info(f"  {data_type}: {copied} files copied")
            total_copied += copied
        else:
            logger.warning(f"  {data_type}: directory not found")

    logger.info(f"\nData sync complete: {total_copied} files copied")

    if include_db:
        db_file = local_root / 'nba_dfs.db'
        if db_file.exists():
            dest_db = gdrive_root / 'nba_dfs.db'
            logger.info("\nSyncing database...")
            shutil.copy2(db_file, dest_db)
            logger.info(f"  Database copied: {db_file.stat().st_size / (1024**2):.1f} MB")
        else:
            logger.warning("\nDatabase not found (optional)")

    logger.info(f"\n{'='*60}")
    logger.info("Sync complete!")
    logger.info(f"{'='*60}")
    logger.info(f"Data location: {gdrive_root / 'data'}")
    logger.info("\nNext steps:")
    logger.info("1. Go to https://colab.research.google.com/")
    logger.info("2. Upload notebooks/colab_backtest.ipynb")
    logger.info("3. Run all cells (code auto-clones from GitHub)")


def main():
    parser = argparse.ArgumentParser(
        description='Sync NBA DFS data to Google Drive (code comes from GitHub)'
    )

    parser.add_argument(
        '--local-dir',
        type=str,
        default='.',
        help='Local project directory (default: current directory)'
    )

    parser.add_argument(
        '--gdrive-dir',
        type=str,
        required=True,
        help='Google Drive mount point (e.g., /content/drive/MyDrive or G:/MyDrive)'
    )

    parser.add_argument(
        '--no-db',
        action='store_true',
        help='Skip syncing SQLite database'
    )

    args = parser.parse_args()

    sync_project_to_gdrive(
        local_dir=args.local_dir,
        gdrive_dir=args.gdrive_dir,
        include_db=not args.no_db
    )


if __name__ == '__main__':
    main()
