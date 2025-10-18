"""Tests for player name and ID filtering"""

import pytest
import pandas as pd
import tempfile
from pathlib import Path

from src.filters import (
    PlayerNameFilter,
    PlayerIDFilter,
    PlayerIDFromCSVFilter,
    PlayerFilterRegistry,
)


@pytest.fixture
def sample_player_data():
    """Sample player data for testing"""
    return pd.DataFrame({
        'playerID': [101, 102, 103, 104, 105],
        'playerName': ['LeBron James', 'Kevin Durant', 'Giannis Antetokounmpo', 'Luka Doncic', 'Jayson Tatum'],
        'team': ['LAL', 'PHX', 'MIL', 'DAL', 'BOS'],
        'salary': [11000, 10500, 10000, 9500, 9000],
    })


class TestPlayerNameFilter:
    """Test PlayerNameFilter functionality"""

    def test_exact_name_match(self, sample_player_data):
        """Test exact name matching"""
        filter_obj = PlayerNameFilter('LeBron James')
        result = filter_obj.apply(sample_player_data)

        assert len(result) == 1
        assert result.iloc[0]['playerName'] == 'LeBron James'

    def test_partial_name_match(self, sample_player_data):
        """Test partial name matching"""
        filter_obj = PlayerNameFilter('LeBron')
        result = filter_obj.apply(sample_player_data)

        assert len(result) == 1
        assert 'LeBron' in result.iloc[0]['playerName']

    def test_multiple_names(self, sample_player_data):
        """Test filtering by multiple names"""
        filter_obj = PlayerNameFilter(['LeBron James', 'Kevin Durant'])
        result = filter_obj.apply(sample_player_data)

        assert len(result) == 2
        assert set(result['playerName']) == {'LeBron James', 'Kevin Durant'}

    def test_case_insensitive(self, sample_player_data):
        """Test case-insensitive matching"""
        filter_obj = PlayerNameFilter('lebron', case_sensitive=False)
        result = filter_obj.apply(sample_player_data)

        assert len(result) == 1

    def test_case_sensitive(self, sample_player_data):
        """Test case-sensitive matching"""
        filter_obj = PlayerNameFilter('lebron', case_sensitive=True)
        result = filter_obj.apply(sample_player_data)

        assert len(result) == 0

    def test_no_matches(self, sample_player_data):
        """Test when no names match"""
        filter_obj = PlayerNameFilter('Nonexistent Player')
        result = filter_obj.apply(sample_player_data)

        assert len(result) == 0

    def test_missing_name_column(self):
        """Test error handling for missing name column"""
        data = pd.DataFrame({'id': [1, 2], 'team': ['LAL', 'BOS']})
        filter_obj = PlayerNameFilter('LeBron')

        with pytest.raises(ValueError, match="No name column found"):
            filter_obj.apply(data)


class TestPlayerIDFilter:
    """Test PlayerIDFilter functionality"""

    def test_single_id(self, sample_player_data):
        """Test filtering by single ID"""
        filter_obj = PlayerIDFilter(101)
        result = filter_obj.apply(sample_player_data)

        assert len(result) == 1
        assert result.iloc[0]['playerID'] == 101

    def test_multiple_ids(self, sample_player_data):
        """Test filtering by multiple IDs"""
        filter_obj = PlayerIDFilter([101, 102, 103])
        result = filter_obj.apply(sample_player_data)

        assert len(result) == 3
        assert set(result['playerID']) == {101, 102, 103}

    def test_string_ids(self, sample_player_data):
        """Test filtering with string IDs"""
        filter_obj = PlayerIDFilter(['101', '102'])
        result = filter_obj.apply(sample_player_data)

        assert len(result) == 2

    def test_no_matches(self, sample_player_data):
        """Test when no IDs match"""
        filter_obj = PlayerIDFilter([999, 1000])
        result = filter_obj.apply(sample_player_data)

        assert len(result) == 0

    def test_missing_id_column(self):
        """Test error handling for missing ID column"""
        data = pd.DataFrame({'name': ['LeBron', 'Kevin'], 'team': ['LAL', 'PHX']})
        filter_obj = PlayerIDFilter(101)

        with pytest.raises(ValueError, match="No ID column found"):
            filter_obj.apply(data)


class TestPlayerIDFromCSVFilter:
    """Test PlayerIDFromCSVFilter functionality"""

    def test_csv_filter_basic(self, sample_player_data):
        """Test loading and filtering from CSV"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write('playerID\n101\n103\n105\n')
            csv_path = f.name

        try:
            filter_obj = PlayerIDFromCSVFilter(csv_path)
            result = filter_obj.apply(sample_player_data)

            assert len(result) == 3
            assert set(result['playerID']) == {101, 103, 105}
        finally:
            Path(csv_path).unlink()

    def test_csv_filter_custom_column(self, sample_player_data):
        """Test CSV filter with custom column name"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write('player_id,name\n101,LeBron\n102,Kevin\n')
            csv_path = f.name

        try:
            filter_obj = PlayerIDFromCSVFilter(csv_path, id_column='player_id')
            result = filter_obj.apply(sample_player_data)

            assert len(result) == 2
        finally:
            Path(csv_path).unlink()

    def test_csv_file_not_found(self):
        """Test error handling for missing CSV file"""
        with pytest.raises(FileNotFoundError):
            PlayerIDFromCSVFilter('/nonexistent/path.csv')

    def test_csv_column_not_found(self):
        """Test error handling for missing column in CSV"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write('id,name\n101,LeBron\n')
            csv_path = f.name

        try:
            with pytest.raises(ValueError, match="Column 'playerID' not found"):
                PlayerIDFromCSVFilter(csv_path, id_column='playerID')
        finally:
            Path(csv_path).unlink()

    def test_csv_with_missing_values(self, sample_player_data):
        """Test CSV filter handling missing/null values"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write('playerID\n101\n\n103\n')
            csv_path = f.name

        try:
            filter_obj = PlayerIDFromCSVFilter(csv_path)
            result = filter_obj.apply(sample_player_data)

            assert len(result) == 2
            assert set(result['playerID']) == {101, 103}
        finally:
            Path(csv_path).unlink()


class TestPlayerFilterRegistry:
    """Test PlayerFilterRegistry functionality"""

    def test_add_single_filter(self, sample_player_data):
        """Test adding single filter"""
        registry = PlayerFilterRegistry()
        registry.add_name_filter('LeBron James')

        result = registry.apply(sample_player_data)
        assert len(result) == 1

    def test_add_multiple_filters_and_logic(self, sample_player_data):
        """Test multiple filters with AND logic (sequential application)"""
        registry = PlayerFilterRegistry()
        registry.add_name_filter('James')  # Matches LeBron James
        registry.add_id_filter([101, 102])  # Only 101 is LeBron

        result = registry.apply(sample_player_data)
        assert len(result) == 1
        assert result.iloc[0]['playerID'] == 101

    def test_apply_any_or_logic(self, sample_player_data):
        """Test apply_any with OR logic"""
        registry = PlayerFilterRegistry()
        registry.add_name_filter('LeBron')
        registry.add_id_filter([103])  # Giannis

        result = registry.apply_any(sample_player_data)
        assert len(result) == 2
        assert set(result['playerID']) == {101, 103}

    def test_add_csv_filter(self, sample_player_data):
        """Test adding CSV filter to registry"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write('playerID\n102\n104\n')
            csv_path = f.name

        try:
            registry = PlayerFilterRegistry()
            registry.add_csv_filter(csv_path)

            result = registry.apply(sample_player_data)
            assert len(result) == 2
            assert set(result['playerID']) == {102, 104}
        finally:
            Path(csv_path).unlink()

    def test_clear_filters(self, sample_player_data):
        """Test clearing all filters"""
        registry = PlayerFilterRegistry()
        registry.add_name_filter('LeBron').add_id_filter([101])
        registry.clear()

        result = registry.apply(sample_player_data)
        assert len(result) == len(sample_player_data)

    def test_method_chaining(self, sample_player_data):
        """Test method chaining"""
        result = (PlayerFilterRegistry()
                  .add_name_filter('James')
                  .add_id_filter([101])
                  .apply(sample_player_data))

        assert len(result) == 1

    def test_empty_registry(self, sample_player_data):
        """Test empty registry returns all data"""
        registry = PlayerFilterRegistry()
        result = registry.apply(sample_player_data)

        assert len(result) == len(sample_player_data)

    def test_registry_repr(self):
        """Test registry string representation"""
        registry = PlayerFilterRegistry()
        registry.add_name_filter('LeBron')
        registry.add_id_filter([101])

        repr_str = repr(registry)
        assert 'PlayerFilterRegistry' in repr_str
        assert '2 filters' in repr_str
