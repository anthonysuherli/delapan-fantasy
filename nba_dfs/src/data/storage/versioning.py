import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Callable
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class DataVersionManager:
    """Track and manage data schema versions"""

    CURRENT_VERSION = '1.0.0'

    def __init__(self, base_dir: str = 'data/inputs'):
        """
        Initialize data version manager.

        Args:
            base_dir: Base directory for data storage
        """
        self.base_dir = Path(base_dir)
        self.metadata_dir = self.base_dir / '_metadata'
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

        self.migrations: Dict[str, Dict[str, Callable]] = {}

    def _get_metadata_path(self, data_type: str, identifier: str) -> Path:
        """
        Get path to metadata file.

        Args:
            data_type: Type of data
            identifier: Unique identifier

        Returns:
            Path to metadata file
        """
        return self.metadata_dir / data_type / f"{identifier}_metadata.json"

    def save_with_version(
        self,
        data: pd.DataFrame,
        data_type: str,
        identifier: str,
        version: str = None,
        additional_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Save data with version metadata.

        Args:
            data: DataFrame to save
            data_type: Type of data
            identifier: Unique identifier
            version: Data schema version (defaults to CURRENT_VERSION)
            additional_metadata: Optional additional metadata

        Returns:
            Metadata dictionary
        """
        version = version or self.CURRENT_VERSION

        metadata = {
            'data_type': data_type,
            'identifier': identifier,
            'version': version,
            'schema': {
                'columns': list(data.columns),
                'dtypes': {col: str(dtype) for col, dtype in data.dtypes.items()},
                'shape': data.shape
            },
            'created_at': datetime.now().isoformat(),
            'row_count': len(data),
            'column_count': len(data.columns)
        }

        if additional_metadata:
            metadata.update(additional_metadata)

        metadata_path = self._get_metadata_path(data_type, identifier)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.debug(f"Saved metadata for {data_type}/{identifier} (v{version})")
        return metadata

    def load_metadata(self, data_type: str, identifier: str) -> Optional[Dict[str, Any]]:
        """
        Load metadata for data.

        Args:
            data_type: Type of data
            identifier: Unique identifier

        Returns:
            Metadata dictionary or None if not found
        """
        metadata_path = self._get_metadata_path(data_type, identifier)

        if not metadata_path.exists():
            return None

        try:
            with open(metadata_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to load metadata {metadata_path}: {e}")
            return None

    def get_version(self, data_type: str, identifier: str) -> Optional[str]:
        """
        Get version of stored data.

        Args:
            data_type: Type of data
            identifier: Unique identifier

        Returns:
            Version string or None if metadata not found
        """
        metadata = self.load_metadata(data_type, identifier)
        return metadata.get('version') if metadata else None

    def register_migration(
        self,
        data_type: str,
        from_version: str,
        to_version: str,
        migration_func: Callable[[pd.DataFrame], pd.DataFrame]
    ):
        """
        Register migration function for schema version upgrade.

        Args:
            data_type: Type of data this migration applies to
            from_version: Source version
            to_version: Target version
            migration_func: Function that takes DataFrame and returns migrated DataFrame
        """
        if data_type not in self.migrations:
            self.migrations[data_type] = {}

        migration_key = f"{from_version}->{to_version}"
        self.migrations[data_type][migration_key] = migration_func

        logger.info(f"Registered migration for {data_type}: {migration_key}")

    def migrate(
        self,
        data: pd.DataFrame,
        data_type: str,
        from_version: str,
        to_version: str
    ) -> pd.DataFrame:
        """
        Migrate data from one version to another.

        Args:
            data: DataFrame to migrate
            data_type: Type of data
            from_version: Source version
            to_version: Target version

        Returns:
            Migrated DataFrame

        Raises:
            ValueError: If migration path not found
        """
        migration_key = f"{from_version}->{to_version}"

        if data_type not in self.migrations:
            raise ValueError(f"No migrations registered for {data_type}")

        if migration_key not in self.migrations[data_type]:
            raise ValueError(f"No migration found: {data_type} {migration_key}")

        migration_func = self.migrations[data_type][migration_key]
        migrated_data = migration_func(data)

        logger.info(f"Migrated {data_type} from {from_version} to {to_version}")
        return migrated_data

    def load_with_migration(
        self,
        data: pd.DataFrame,
        data_type: str,
        identifier: str,
        target_version: str = None
    ) -> pd.DataFrame:
        """
        Load data and apply migrations if needed.

        Args:
            data: DataFrame to potentially migrate
            data_type: Type of data
            identifier: Unique identifier
            target_version: Target version (defaults to CURRENT_VERSION)

        Returns:
            DataFrame migrated to target version
        """
        target_version = target_version or self.CURRENT_VERSION
        current_version = self.get_version(data_type, identifier)

        if current_version is None:
            logger.warning(f"No version metadata found for {data_type}/{identifier}")
            return data

        if current_version == target_version:
            return data

        logger.info(f"Migrating {data_type}/{identifier} from {current_version} to {target_version}")
        return self.migrate(data, data_type, current_version, target_version)

    def validate_schema(self, data: pd.DataFrame, data_type: str, identifier: str) -> bool:
        """
        Validate data schema against stored metadata.

        Args:
            data: DataFrame to validate
            data_type: Type of data
            identifier: Unique identifier

        Returns:
            True if schema matches, False otherwise
        """
        metadata = self.load_metadata(data_type, identifier)

        if metadata is None:
            logger.warning(f"No metadata found for validation: {data_type}/{identifier}")
            return False

        stored_columns = set(metadata['schema']['columns'])
        current_columns = set(data.columns)

        if stored_columns != current_columns:
            missing = stored_columns - current_columns
            extra = current_columns - stored_columns
            logger.warning(
                f"Schema mismatch for {data_type}/{identifier}. "
                f"Missing: {missing}, Extra: {extra}"
            )
            return False

        return True
