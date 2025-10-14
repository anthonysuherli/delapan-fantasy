from pathlib import Path
from typing import Union


def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, creating it if necessary.

    Args:
        path: Directory path to ensure exists

    Returns:
        Path object to directory
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_parent_directory(file_path: Union[str, Path]) -> Path:
    """
    Ensure parent directory of file exists, creating it if necessary.

    Args:
        file_path: File path whose parent directory to ensure exists

    Returns:
        Path object to file
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    return file_path
