import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.features.transformers.rolling_stats import RollingStatsTransformer
from src.features.transformers.rolling_minmax import RollingMinMaxTransformer
from src.features.transformers.ewma import EWMATransformer
from src.features.transformers.target import TargetTransformer
from src.features.transformers.injury import InjuryTransformer


@pytest.fixture
def sample_player_data():
    """Create sample player game log data"""
    dates = [(datetime(2024, 1, 1) + timedelta(days=i)).strftime('%Y%m%d')
             for i in range(10)]

    data = {
        'playerID': ['player1'] * 10,
        'gameDate': dates,
        'pts': [20, 25, 18, 30, 22, 28, 24, 26, 21, 29],
        'reb': [8, 10, 7, 12, 9, 11, 8, 10, 9, 11],
        'ast': [5, 6, 4, 8, 5, 7, 6, 6, 5, 7],
        'stl': [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
        'blk': [0, 1, 0, 1, 1, 1, 0, 1, 0, 1],
        'mins': [32, 35, 30, 38, 33, 36, 34, 35, 32, 37],
        'fpts': [40.0, 50.0, 35.0, 60.0, 45.0, 55.0, 42.0, 52.0, 38.0, 58.0]
    }

    return pd.DataFrame(data)


@pytest.fixture
def multi_player_data():
    """Create sample data for multiple players"""
    dates = [(datetime(2024, 1, 1) + timedelta(days=i)).strftime('%Y%m%d')
             for i in range(5)]

    data = {
        'playerID': ['player1'] * 5 + ['player2'] * 5,
        'gameDate': dates * 2,
        'pts': [20, 25, 18, 30, 22, 15, 20, 17, 25, 19],
        'reb': [8, 10, 7, 12, 9, 5, 7, 6, 9, 7],
        'ast': [5, 6, 4, 8, 5, 3, 4, 3, 5, 4],
        'stl': [1, 2, 1, 2, 1, 0, 1, 0, 1, 1],
        'blk': [0, 1, 0, 1, 1, 0, 0, 0, 1, 0],
        'mins': [32, 35, 30, 38, 33, 28, 30, 27, 32, 29],
        'fpts': [40.0, 50.0, 35.0, 60.0, 45.0, 25.0, 35.0, 28.0, 45.0, 32.0]
    }

    return pd.DataFrame(data)


class TestRollingStatsTransformer:
    """Test rolling statistics transformer"""

    def test_initialization(self):
        transformer = RollingStatsTransformer(windows=[3, 5], stats=['pts', 'reb'])
        assert transformer.windows == [3, 5]
        assert transformer.stats == ['pts', 'reb']
        assert transformer.include_std is True

    def test_fit(self, sample_player_data):
        transformer = RollingStatsTransformer()
        result = transformer.fit(sample_player_data)
        assert transformer.is_fitted
        assert result is transformer

    def test_transform_creates_ma_columns(self, sample_player_data):
        transformer = RollingStatsTransformer(windows=[3], stats=['pts'])
        transformer.fit(sample_player_data)
        result = transformer.transform(sample_player_data)

        assert 'rolling_pts_mean_3' in result.columns

    def test_transform_creates_std_columns(self, sample_player_data):
        transformer = RollingStatsTransformer(windows=[3], stats=['pts'])
        transformer.fit(sample_player_data)
        result = transformer.transform(sample_player_data)

        assert 'rolling_pts_std_3' in result.columns

    def test_transform_without_std(self, sample_player_data):
        transformer = RollingStatsTransformer(
            windows=[3],
            stats=['pts'],
            include_std=False
        )
        transformer.fit(sample_player_data)
        result = transformer.transform(sample_player_data)

        assert 'rolling_pts_mean_3' in result.columns
        assert 'rolling_pts_std_3' not in result.columns

    def test_rolling_calculation_correctness(self, sample_player_data):
        transformer = RollingStatsTransformer(windows=[3], stats=['pts'])
        transformer.fit(sample_player_data)
        result = transformer.transform(sample_player_data)

        expected_ma3_idx3 = (20 + 25 + 18) / 3
        assert abs(result['rolling_pts_mean_3'].iloc[3] - expected_ma3_idx3) < 0.01

    def test_multi_player_grouping(self, multi_player_data):
        transformer = RollingStatsTransformer(windows=[3], stats=['pts'])
        transformer.fit(multi_player_data)
        result = transformer.transform(multi_player_data)

        player1_data = result[result['playerID'] == 'player1']
        player2_data = result[result['playerID'] == 'player2']

        assert 'rolling_pts_mean_3' in result.columns
        assert len(player1_data) == 5
        assert len(player2_data) == 5

    def test_transform_without_fit_raises_error(self, sample_player_data):
        transformer = RollingStatsTransformer()
        with pytest.raises(ValueError, match="has not been fitted"):
            transformer.transform(sample_player_data)

    def test_missing_required_columns(self):
        transformer = RollingStatsTransformer()
        transformer.fit(pd.DataFrame({'col1': [1, 2, 3]}))

        with pytest.raises(ValueError, match="must contain"):
            transformer.transform(pd.DataFrame({'col1': [1, 2, 3]}))


class TestRollingMinMaxTransformer:
    """Test rolling min/max transformer"""

    def test_initialization(self):
        transformer = RollingMinMaxTransformer(windows=[3, 5], stats=['pts'])
        assert transformer.windows == [3, 5]
        assert transformer.stats == ['pts']

    def test_fit(self, sample_player_data):
        transformer = RollingMinMaxTransformer()
        result = transformer.fit(sample_player_data)
        assert transformer.is_fitted
        assert result is transformer

    def test_transform_creates_min_max_columns(self, sample_player_data):
        transformer = RollingMinMaxTransformer(windows=[3], stats=['pts'])
        transformer.fit(sample_player_data)
        result = transformer.transform(sample_player_data)

        assert 'pts_min3' in result.columns
        assert 'pts_max3' in result.columns

    def test_min_max_calculation_correctness(self, sample_player_data):
        transformer = RollingMinMaxTransformer(windows=[3], stats=['pts'])
        transformer.fit(sample_player_data)
        result = transformer.transform(sample_player_data)

        expected_min3_idx2 = min([20, 25, 18])
        expected_max3_idx2 = max([20, 25, 18])

        assert result['pts_min3'].iloc[2] == expected_min3_idx2
        assert result['pts_max3'].iloc[2] == expected_max3_idx2

    def test_multi_player_grouping(self, multi_player_data):
        transformer = RollingMinMaxTransformer(windows=[3], stats=['pts'])
        transformer.fit(multi_player_data)
        result = transformer.transform(multi_player_data)

        assert 'pts_min3' in result.columns
        assert 'pts_max3' in result.columns

    def test_transform_without_fit_raises_error(self, sample_player_data):
        transformer = RollingMinMaxTransformer()
        with pytest.raises(ValueError, match="has not been fitted"):
            transformer.transform(sample_player_data)


class TestEWMATransformer:
    """Test EWMA transformer"""

    def test_initialization(self):
        transformer = EWMATransformer(span=5, stats=['pts'])
        assert transformer.span == 5
        assert transformer.stats == ['pts']

    def test_default_span(self):
        transformer = EWMATransformer()
        assert transformer.span == 3

    def test_fit(self, sample_player_data):
        transformer = EWMATransformer()
        result = transformer.fit(sample_player_data)
        assert transformer.is_fitted
        assert result is transformer

    def test_transform_creates_ewma_columns(self, sample_player_data):
        transformer = EWMATransformer(span=3, stats=['pts'])
        transformer.fit(sample_player_data)
        result = transformer.transform(sample_player_data)

        assert 'ewma_pts_3' in result.columns

    def test_ewma_naming_convention(self, sample_player_data):
        transformer = EWMATransformer(span=5, stats=['pts'])
        transformer.fit(sample_player_data)
        result = transformer.transform(sample_player_data)

        assert 'ewma_pts_5' in result.columns

    def test_multi_player_grouping(self, multi_player_data):
        transformer = EWMATransformer(span=3, stats=['pts'])
        transformer.fit(multi_player_data)
        result = transformer.transform(multi_player_data)

        assert 'ewma_pts_3' in result.columns

    def test_transform_without_fit_raises_error(self, sample_player_data):
        transformer = EWMATransformer()
        with pytest.raises(ValueError, match="has not been fitted"):
            transformer.transform(sample_player_data)


class TestTargetTransformer:
    """Test target variable transformer"""

    def test_initialization(self):
        transformer = TargetTransformer(target_col='fpts', shift_periods=-1)
        assert transformer.target_col == 'fpts'
        assert transformer.shift_periods == -1

    def test_default_initialization(self):
        transformer = TargetTransformer()
        assert transformer.target_col == 'fpts'
        assert transformer.shift_periods == -1

    def test_fit(self, sample_player_data):
        transformer = TargetTransformer()
        result = transformer.fit(sample_player_data)
        assert transformer.is_fitted
        assert result is transformer

    def test_transform_creates_target_column(self, sample_player_data):
        transformer = TargetTransformer()
        transformer.fit(sample_player_data)
        result = transformer.transform(sample_player_data)

        assert 'target' in result.columns

    def test_target_shift_correctness(self, sample_player_data):
        transformer = TargetTransformer(target_col='fpts', shift_periods=-1)
        transformer.fit(sample_player_data)
        result = transformer.transform(sample_player_data)

        assert result['target'].iloc[0] == sample_player_data['fpts'].iloc[1]
        assert result['target'].iloc[1] == sample_player_data['fpts'].iloc[2]
        assert pd.isna(result['target'].iloc[-1])

    def test_multi_player_grouping(self, multi_player_data):
        transformer = TargetTransformer()
        transformer.fit(multi_player_data)
        result = transformer.transform(multi_player_data)

        player1_data = result[result['playerID'] == 'player1']
        player2_data = result[result['playerID'] == 'player2']

        assert player1_data['target'].iloc[0] == multi_player_data[
            multi_player_data['playerID'] == 'player1'
        ]['fpts'].iloc[1]

        assert player2_data['target'].iloc[0] == multi_player_data[
            multi_player_data['playerID'] == 'player2'
        ]['fpts'].iloc[1]

    def test_transform_without_fit_raises_error(self, sample_player_data):
        transformer = TargetTransformer()
        with pytest.raises(ValueError, match="has not been fitted"):
            transformer.transform(sample_player_data)

    def test_missing_target_column_raises_error(self, sample_player_data):
        transformer = TargetTransformer(target_col='nonexistent')
        transformer.fit(sample_player_data)

        with pytest.raises(ValueError, match="Target column"):
            transformer.transform(sample_player_data)


@pytest.fixture
def sample_injury_data():
    """Create sample injury data"""
    data = {
        'playerID': ['player1', 'player2', 'player3'],
        'designation': ['Out', 'Questionable', 'Day-To-Day'],
        'injDate': ['20250305', '20250305', '20250305'],
        'injReturnDate': ['20250307', '20250306', '20250308'],
        'description': [
            'Out with knee injury',
            'Questionable with ankle sprain',
            'Day-to-day with illness'
        ]
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_player_roster():
    """Create sample player roster for injury merging"""
    data = {
        'playerID': ['player1', 'player2', 'player3', 'player4'],
        'playerName': ['Player One', 'Player Two', 'Player Three', 'Player Four'],
        'team': ['LAL', 'BOS', 'GSW', 'MIA'],
        'pos': ['PG', 'SG', 'SF', 'PF'],
        'salary': [9000, 8500, 7500, 6500]
    }
    return pd.DataFrame(data)


class TestInjuryTransformer:
    """Test injury data transformer"""

    def test_initialization(self):
        transformer = InjuryTransformer()
        assert transformer.name == 'injury'

    def test_fit(self, sample_player_roster):
        transformer = InjuryTransformer()
        result = transformer.fit(sample_player_roster)
        assert transformer.is_fitted
        assert result is transformer

    def test_transform_with_injuries(self, sample_player_roster, sample_injury_data):
        transformer = InjuryTransformer()
        transformer.fit(sample_player_roster)
        result = transformer.transform(sample_player_roster, injuries=sample_injury_data)

        assert 'injury_status' in result.columns
        assert 'injury_designation' in result.columns
        assert 'injury_description' in result.columns
        assert 'is_injured' in result.columns
        assert 'is_out' in result.columns
        assert 'is_questionable' in result.columns
        assert 'is_doubtful' in result.columns
        assert 'is_day_to_day' in result.columns

    def test_transform_without_injuries(self, sample_player_roster):
        transformer = InjuryTransformer()
        transformer.fit(sample_player_roster)
        result = transformer.transform(sample_player_roster, injuries=None)

        assert result['injury_status'].eq('Healthy').all()
        assert result['is_injured'].eq(0).all()
        assert result['is_out'].eq(0).all()

    def test_injury_flag_out(self, sample_player_roster, sample_injury_data):
        transformer = InjuryTransformer()
        transformer.fit(sample_player_roster)
        result = transformer.transform(sample_player_roster, injuries=sample_injury_data)

        player1_row = result[result['playerID'] == 'player1'].iloc[0]
        assert player1_row['is_injured'] == 1
        assert player1_row['is_out'] == 1
        assert player1_row['injury_designation'] == 'Out'

    def test_injury_flag_questionable(self, sample_player_roster, sample_injury_data):
        transformer = InjuryTransformer()
        transformer.fit(sample_player_roster)
        result = transformer.transform(sample_player_roster, injuries=sample_injury_data)

        player2_row = result[result['playerID'] == 'player2'].iloc[0]
        assert player2_row['is_injured'] == 1
        assert player2_row['is_questionable'] == 1
        assert player2_row['injury_designation'] == 'Questionable'

    def test_injury_flag_day_to_day(self, sample_player_roster, sample_injury_data):
        transformer = InjuryTransformer()
        transformer.fit(sample_player_roster)
        result = transformer.transform(sample_player_roster, injuries=sample_injury_data)

        player3_row = result[result['playerID'] == 'player3'].iloc[0]
        assert player3_row['is_injured'] == 1
        assert player3_row['is_day_to_day'] == 1
        assert player3_row['injury_designation'] == 'Day-To-Day'

    def test_healthy_player(self, sample_player_roster, sample_injury_data):
        transformer = InjuryTransformer()
        transformer.fit(sample_player_roster)
        result = transformer.transform(sample_player_roster, injuries=sample_injury_data)

        player4_row = result[result['playerID'] == 'player4'].iloc[0]
        assert player4_row['is_injured'] == 0
        assert player4_row['injury_status'] == 'Healthy'

    def test_merge_injuries_convenience_method(self, sample_player_roster, sample_injury_data):
        transformer = InjuryTransformer()
        result = transformer.merge_injuries(sample_player_roster, sample_injury_data)

        assert 'injury_status' in result.columns
        assert len(result) == len(sample_player_roster)

    def test_transform_without_fit_raises_error(self, sample_player_roster):
        transformer = InjuryTransformer()
        with pytest.raises(ValueError, match="has not been fitted"):
            transformer.transform(sample_player_roster)

    def test_missing_playerid_column_raises_error(self):
        transformer = InjuryTransformer()
        df = pd.DataFrame({'name': ['Test']})
        transformer.fit(df)

        with pytest.raises(ValueError, match="must contain 'playerID'"):
            transformer.transform(df)
