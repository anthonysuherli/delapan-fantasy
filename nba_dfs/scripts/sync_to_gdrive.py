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
    data_only: bool = False
) -> None:
    """
    Sync NBA DFS project to Google Drive.

    Args:
        local_dir: Local project directory
        gdrive_dir: Google Drive mount point
        data_only: If True, only sync data files
    """
    local_root = Path(local_dir)
    gdrive_root = Path(gdrive_dir) / 'nba_dfs'

    logger.info(f"Syncing from {local_root} to {gdrive_root}")

    gdrive_root.mkdir(parents=True, exist_ok=True)

    logger.info("Syncing data files...")
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

    logger.info(f"Data sync complete: {total_copied} files copied")

    if not data_only:
        logger.info("Syncing source code...")

        src_dirs = ['src', 'config', 'scripts']
        for dir_name in src_dirs:
            source = local_root / dir_name
            dest = gdrive_root / dir_name

            if source.exists():
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.copytree(source, dest)
                logger.info(f"  {dir_name}: copied")
            else:
                logger.warning(f"  {dir_name}: directory not found")

        config_files = ['requirements.txt', 'README.md', 'CLAUDE.md']
        for file_name in config_files:
            source = local_root / file_name
            dest = gdrive_root / file_name

            if source.exists():
                shutil.copy2(source, dest)
                logger.info(f"  {file_name}: copied")

        logger.info("Source code sync complete")

    logger.info(f"\nSync complete! Project ready at: {gdrive_root}")
    logger.info("Next steps:")
    logger.info("1. Upload notebooks/colab_backtest.ipynb to Google Colab")
    logger.info("2. Mount Google Drive in Colab")
    logger.info("3. Run the notebook")


def main():
    parser = argparse.ArgumentParser(description='Sync NBA DFS project to Google Drive')

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
        '--data-only',
        action='store_true',
        help='Sync only data files (skip source code)'
    )

    args = parser.parse_args()

    sync_project_to_gdrive(
        local_dir=args.local_dir,
        gdrive_dir=args.gdrive_dir,
        data_only=args.data_only
    )


if __name__ == '__main__':
    main()
