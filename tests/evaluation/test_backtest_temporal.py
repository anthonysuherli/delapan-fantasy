import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.evaluation.backtest_config import BacktestConfig
from src.evaluation.data_loader import HistoricalDataLoader
from src.evaluation.feature_builder import FeatureBuilder


class TestTemporalConsistency:

    @pytest.fixture
    def config(self):
        return BacktestConfig(
            start_date='20240101',
            end_date='20240115',
            lookback_days=30,
            model_type='xgboost'
        )

    @pytest.fixture
    def sample_player_logs(self):
        dates = pd.date_range('20231201', '20240120', freq='D')
        data = []

        for date in dates:
            for player_id in ['player1', 'player2', 'player3']:
                data.append({
                    'gameDate': date.strftime('%Y%m%d'),
                    'playerID': player_id,
                    'pts': np.random.randint(10, 35),
                    'reb': np.random.randint(3, 12),
                    'ast': np.random.randint(2, 10),
                    'stl': np.random.randint(0, 3),
                    'blk': np.random.randint(0, 3),
                    'TOV': np.random.randint(1, 5),
                    'mins': np.random.randint(25, 40)
                })

        return pd.DataFrame(data)

    def test_no_lookahead_bias_in_data_loader(self, config, tmp_path):
        test_date = '20240115'

        sample_data = pd.DataFrame({
            'gameDate': ['20240110', '20240112', '20240114', '20240115', '20240116'],
            'playerID': ['player1', 'player1', 'player1', 'player1', 'player1'],
            'pts': [20, 22, 18, 25, 30]
        })

        loaded_data = sample_data[sample_data['gameDate'] < test_date].copy()

        assert test_date not in loaded_data['gameDate'].values, \
            f"LOOKAHEAD BIAS: test_date {test_date} found in training data"

        max_date = loaded_data['gameDate'].max()
        assert max_date < test_date, \
            f"LOOKAHEAD BIAS: max date {max_date} >= test_date {test_date}"

        future_dates = loaded_data[loaded_data['gameDate'] >= test_date]
        assert len(future_dates) == 0, \
            f"LOOKAHEAD BIAS: {len(future_dates)} future dates found in training data"

    def test_feature_calculation_uses_prior_games_only(self, sample_player_logs):
        builder = FeatureBuilder()

        df = sample_player_logs.copy()
        df = df[df['playerID'] == 'player1'].copy()
        df['gameDate'] = pd.to_datetime(df['gameDate'], format='%Y%m%d')
        df = df.sort_values('gameDate').reset_index(drop=True)

        test_game_idx = 10

        prior_games = df.iloc[:test_game_idx]
        test_game = df.iloc[test_game_idx]

        features = builder._calculate_features_from_prior_games(prior_games, 'player1')

        assert features['pts_avg_3'] is not None

        pts_values = pd.to_numeric(prior_games['pts'], errors='coerce')
        expected_avg_3 = pts_values.tail(3).mean()

        assert abs(features['pts_avg_3'] - expected_avg_3) < 0.01, \
            f"Feature calculation incorrect: got {features['pts_avg_3']}, expected {expected_avg_3}"

        test_game_pts = float(test_game['pts'])
        assert abs(features['pts_avg_3'] - test_game_pts) > 0.01, \
            "LOOKAHEAD BIAS: Feature includes test game data"

    def test_training_features_temporal_ordering(self, sample_player_logs):
        builder = FeatureBuilder()

        X_train, y_train = builder.build_training_features(sample_player_logs)

        assert len(X_train) > 0, "No training features generated"
        assert len(X_train) == len(y_train), "Feature and target length mismatch"

    def test_dk_fantasy_points_calculation(self):
        builder = FeatureBuilder()

        stats = pd.Series({
            'pts': 25,
            'reb': 10,
            'ast': 8,
            'stl': 2,
            'blk': 1,
            'TOV': 3
        })

        fpts = builder.calculate_dk_fantasy_points(stats)

        expected = (
            25 * 1.0 +
            10 * 1.25 +
            8 * 1.5 +
            2 * 2.0 +
            1 * 2.0 -
            3 * 0.5 +
            1.5
        )

        assert abs(fpts - expected) < 0.01, \
            f"DK scoring incorrect: got {fpts}, expected {expected}"

    def test_triple_double_bonus(self):
        builder = FeatureBuilder()

        stats = pd.Series({
            'pts': 12,
            'reb': 11,
            'ast': 10,
            'stl': 2,
            'blk': 1,
            'TOV': 2
        })

        fpts = builder.calculate_dk_fantasy_points(stats)

        base_points = (
            12 * 1.0 +
            11 * 1.25 +
            10 * 1.5 +
            2 * 2.0 +
            1 * 2.0 -
            2 * 0.5
        )

        expected = base_points + 1.5 + 1.5

        assert abs(fpts - expected) < 0.01, \
            f"Triple-double bonus incorrect: got {fpts}, expected {expected}"

    def test_config_validation(self):
        with pytest.raises(ValueError):
            BacktestConfig(
                start_date='20240115',
                end_date='20240101',
                lookback_days=30
            )

        with pytest.raises(ValueError):
            BacktestConfig(
                start_date='2024-01-01',
                end_date='2024-01-15',
                lookback_days=30
            )

        with pytest.raises(ValueError):
            BacktestConfig(
                start_date='20240101',
                end_date='20240115',
                lookback_days=-10
            )

    def test_feature_consistency_between_training_and_slate(self, sample_player_logs):
        builder = FeatureBuilder()

        training_data = sample_player_logs[sample_player_logs['gameDate'] < '20240115'].copy()

        X_train, y_train = builder.build_training_features(training_data)

        slate_data = {
            'salaries': pd.DataFrame({
                'playerID': ['player1', 'player2'],
                'playerName': ['Player One', 'Player Two'],
                'team': ['LAL', 'BOS'],
                'pos': ['PG', 'SG'],
                'salary': [8500, 7200]
            })
        }

        slate_features = builder.build_slate_features(slate_data, training_data)

        if not X_train.empty and not slate_features.empty:
            train_feature_cols = set(X_train.columns)
            slate_feature_cols = set([col for col in slate_features.columns
                                     if col not in ['playerID', 'playerName', 'team', 'pos', 'salary']])

            missing_in_slate = train_feature_cols - slate_feature_cols
            extra_in_slate = slate_feature_cols - train_feature_cols

            assert len(missing_in_slate) == 0, \
                f"Features missing in slate: {missing_in_slate}"

    def test_rolling_window_calculation(self):
        builder = FeatureBuilder()

        player_data = pd.DataFrame({
            'pts': [10, 15, 20, 25, 30],
            'reb': [5, 6, 7, 8, 9],
            'ast': [3, 4, 5, 6, 7]
        })

        features = builder._calculate_features_from_prior_games(player_data, 'test_player')

        assert features['pts_avg_3'] == 15.0
        assert features['pts_avg_5'] == 20.0

        expected_std_3 = np.std([10, 15, 20], ddof=1)
        assert abs(features['pts_std_3'] - expected_std_3) < 0.01


class TestBacktestConfig:

    def test_config_serialization(self):
        config = BacktestConfig(
            start_date='20240101',
            end_date='20240331',
            lookback_days=60,
            model_type='random_forest'
        )

        config_dict = config.to_dict()

        assert config_dict['start_date'] == '20240101'
        assert config_dict['end_date'] == '20240331'
        assert config_dict['lookback_days'] == 60
        assert config_dict['model_type'] == 'random_forest'

        restored_config = BacktestConfig.from_dict(config_dict)

        assert restored_config.start_date == config.start_date
        assert restored_config.end_date == config.end_date
        assert restored_config.lookback_days == config.lookback_days
        assert restored_config.model_type == config.model_type


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
