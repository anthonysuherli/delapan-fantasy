import pytest
import pandas as pd
import tempfile
import shutil
from src.data.storage.versioning import DataVersionManager


class TestDataVersionManager:
    """Tests for DataVersionManager"""

    @pytest.fixture
    def temp_version_manager(self):
        """Create temporary version manager"""
        temp_dir = tempfile.mkdtemp()
        manager = DataVersionManager(base_dir=temp_dir)
        yield manager
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def sample_data(self):
        """Create sample DataFrame"""
        return pd.DataFrame({
            'playerID': ['p1', 'p2', 'p3'],
            'salary': [8000, 7500, 6000],
            'position': ['PG', 'SG', 'SF']
        })

    def test_save_with_version(self, temp_version_manager, sample_data):
        """Test saving data with version metadata"""
        metadata = temp_version_manager.save_with_version(
            sample_data,
            'dfs_salaries',
            '20240101',
            version='1.0.0'
        )

        assert metadata['version'] == '1.0.0'
        assert metadata['data_type'] == 'dfs_salaries'
        assert metadata['identifier'] == '20240101'
        assert metadata['row_count'] == 3
        assert metadata['column_count'] == 3
        assert 'created_at' in metadata

    def test_load_metadata(self, temp_version_manager, sample_data):
        """Test loading metadata"""
        temp_version_manager.save_with_version(
            sample_data,
            'dfs_salaries',
            '20240101'
        )

        metadata = temp_version_manager.load_metadata('dfs_salaries', '20240101')

        assert metadata is not None
        assert metadata['version'] == temp_version_manager.CURRENT_VERSION
        assert metadata['data_type'] == 'dfs_salaries'

    def test_load_metadata_not_found(self, temp_version_manager):
        """Test loading metadata for non-existent data"""
        metadata = temp_version_manager.load_metadata('dfs_salaries', '20240101')

        assert metadata is None

    def test_get_version(self, temp_version_manager, sample_data):
        """Test getting version of stored data"""
        temp_version_manager.save_with_version(
            sample_data,
            'dfs_salaries',
            '20240101',
            version='1.5.0'
        )

        version = temp_version_manager.get_version('dfs_salaries', '20240101')

        assert version == '1.5.0'

    def test_register_migration(self, temp_version_manager):
        """Test registering migration function"""
        def migrate_v1_to_v2(df):
            df['new_column'] = 0
            return df

        temp_version_manager.register_migration(
            'dfs_salaries',
            '1.0.0',
            '2.0.0',
            migrate_v1_to_v2
        )

        assert 'dfs_salaries' in temp_version_manager.migrations
        assert '1.0.0->2.0.0' in temp_version_manager.migrations['dfs_salaries']

    def test_migrate_data(self, temp_version_manager, sample_data):
        """Test migrating data between versions"""
        def migrate_v1_to_v2(df):
            df['new_column'] = 'added'
            return df

        temp_version_manager.register_migration(
            'dfs_salaries',
            '1.0.0',
            '2.0.0',
            migrate_v1_to_v2
        )

        migrated_data = temp_version_manager.migrate(
            sample_data,
            'dfs_salaries',
            '1.0.0',
            '2.0.0'
        )

        assert 'new_column' in migrated_data.columns
        assert all(migrated_data['new_column'] == 'added')

    def test_migrate_no_migration_found(self, temp_version_manager, sample_data):
        """Test migration fails when no migration registered"""
        with pytest.raises(ValueError, match="No migration found"):
            temp_version_manager.migrate(
                sample_data,
                'dfs_salaries',
                '1.0.0',
                '2.0.0'
            )

    def test_validate_schema_match(self, temp_version_manager, sample_data):
        """Test schema validation when schemas match"""
        temp_version_manager.save_with_version(
            sample_data,
            'dfs_salaries',
            '20240101'
        )

        is_valid = temp_version_manager.validate_schema(
            sample_data,
            'dfs_salaries',
            '20240101'
        )

        assert is_valid is True

    def test_validate_schema_mismatch(self, temp_version_manager, sample_data):
        """Test schema validation when schemas don't match"""
        temp_version_manager.save_with_version(
            sample_data,
            'dfs_salaries',
            '20240101'
        )

        different_data = pd.DataFrame({
            'playerID': ['p1'],
            'different_column': [100]
        })

        is_valid = temp_version_manager.validate_schema(
            different_data,
            'dfs_salaries',
            '20240101'
        )

        assert is_valid is False

    def test_save_with_additional_metadata(self, temp_version_manager, sample_data):
        """Test saving with additional metadata"""
        additional = {
            'source': 'Tank01 API',
            'processed_by': 'test_user'
        }

        metadata = temp_version_manager.save_with_version(
            sample_data,
            'dfs_salaries',
            '20240101',
            additional_metadata=additional
        )

        assert metadata['source'] == 'Tank01 API'
        assert metadata['processed_by'] == 'test_user'

    def test_load_with_migration_no_migration_needed(self, temp_version_manager, sample_data):
        """Test loading data when versions match (no migration)"""
        temp_version_manager.save_with_version(
            sample_data,
            'dfs_salaries',
            '20240101',
            version='1.0.0'
        )

        result = temp_version_manager.load_with_migration(
            sample_data,
            'dfs_salaries',
            '20240101',
            target_version='1.0.0'
        )

        pd.testing.assert_frame_equal(result, sample_data)

    def test_load_with_migration_applies_migration(self, temp_version_manager, sample_data):
        """Test loading data applies migration when needed"""
        def migrate_v1_to_v2(df):
            df['migrated'] = True
            return df

        temp_version_manager.register_migration(
            'dfs_salaries',
            '1.0.0',
            '2.0.0',
            migrate_v1_to_v2
        )

        temp_version_manager.save_with_version(
            sample_data,
            'dfs_salaries',
            '20240101',
            version='1.0.0'
        )

        result = temp_version_manager.load_with_migration(
            sample_data,
            'dfs_salaries',
            '20240101',
            target_version='2.0.0'
        )

        assert 'migrated' in result.columns
