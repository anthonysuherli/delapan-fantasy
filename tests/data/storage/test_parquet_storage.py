import pytest
import pandas as pd
from pathlib import Path
import tempfile
import shutil
from src.data.storage.parquet_storage import ParquetStorage


class TestParquetStorage:
    """Tests for ParquetStorage implementation"""

    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage for testing"""
        temp_dir = tempfile.mkdtemp()
        storage = ParquetStorage(base_dir=temp_dir)
        yield storage
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        return pd.DataFrame({
            'playerID': ['player1', 'player2', 'player3'],
            'playerName': ['John Doe', 'Jane Smith', 'Bob Johnson'],
            'salary': [8000, 7500, 6000],
            'position': ['PG', 'SG', 'SF']
        })

    def test_save_and_load(self, temp_storage, sample_data):
        """Test basic save and load operations"""
        temp_storage.save(sample_data, 'dfs_salaries', '20240101')

        assert temp_storage.exists('dfs_salaries', '20240101')

        loaded_data = temp_storage.load('dfs_salaries', {'start_date': '20240101', 'end_date': '20240101'})

        assert not loaded_data.empty
        assert len(loaded_data) == len(sample_data)
        assert list(loaded_data.columns) == list(sample_data.columns)

    def test_load_with_date_range(self, temp_storage, sample_data):
        """Test loading data with date range filters"""
        temp_storage.save(sample_data, 'dfs_salaries', '20240101')
        temp_storage.save(sample_data, 'dfs_salaries', '20240102')
        temp_storage.save(sample_data, 'dfs_salaries', '20240103')

        loaded_data = temp_storage.load(
            'dfs_salaries',
            {'start_date': '20240101', 'end_date': '20240102'}
        )

        assert len(loaded_data) == len(sample_data) * 2

    def test_exists(self, temp_storage, sample_data):
        """Test exists method"""
        assert not temp_storage.exists('dfs_salaries', '20240101')

        temp_storage.save(sample_data, 'dfs_salaries', '20240101')

        assert temp_storage.exists('dfs_salaries', '20240101')

    def test_save_betting_odds(self, temp_storage):
        """Test save_betting_odds convenience method"""
        odds_data = pd.DataFrame({
            'gameID': ['game1', 'game2'],
            'spread': [-5.5, 3.0],
            'total': [220.5, 215.0]
        })

        temp_storage.save_betting_odds(odds_data.to_dict('records'), '20240101')

        assert temp_storage.exists('betting_odds', '20240101')

    def test_save_dfs_salaries(self, temp_storage, sample_data):
        """Test save_dfs_salaries convenience method"""
        temp_storage.save_dfs_salaries(sample_data.to_dict('records'), '20240101')

        assert temp_storage.exists('dfs_salaries', '20240101')

    def test_load_empty_date_range(self, temp_storage):
        """Test loading data when no files exist in range"""
        loaded_data = temp_storage.load(
            'dfs_salaries',
            {'start_date': '20240101', 'end_date': '20240102'}
        )

        assert loaded_data.empty

    def test_save_teams(self, temp_storage):
        """Test save_teams special case (no date partition)"""
        teams_data = pd.DataFrame({
            'teamID': ['LAL', 'BOS', 'GSW'],
            'teamName': ['Lakers', 'Celtics', 'Warriors'],
            'city': ['Los Angeles', 'Boston', 'Golden State']
        })

        temp_storage.save_teams(teams_data.to_dict('records'))

        teams_path = Path(temp_storage.base_dir) / 'teams' / 'teams.parquet'
        assert teams_path.exists()

    def test_load_data_method(self, temp_storage, sample_data):
        """Test load_data convenience method"""
        temp_storage.save(sample_data, 'dfs_salaries', '20240101')
        temp_storage.save(sample_data, 'dfs_salaries', '20240102')

        loaded_data = temp_storage.load_data(
            'dfs_salaries',
            start_date='20240101',
            end_date='20240102'
        )

        assert len(loaded_data) == len(sample_data) * 2

    def test_save_dict_data(self, temp_storage):
        """Test saving dictionary data (auto-conversion to DataFrame)"""
        dict_data = {
            'playerID': ['player1'],
            'salary': [8000]
        }

        temp_storage.save(dict_data, 'dfs_salaries', '20240101')

        loaded_data = temp_storage.load('dfs_salaries', {'start_date': '20240101', 'end_date': '20240101'})

        assert not loaded_data.empty
        assert 'playerID' in loaded_data.columns
        assert 'salary' in loaded_data.columns
