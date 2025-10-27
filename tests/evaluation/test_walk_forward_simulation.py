"""
Tests for walk-forward backtest simulation framework.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil

from src.evaluation.walk_forward_simulation import WalkForwardSimulation


@pytest.fixture
def temp_data_dir():
    """Create temporary data directory with mock data."""
    temp_dir = tempfile.mkdtemp()
    data_dir = Path(temp_dir) / 'data' / 'inputs'
    data_dir.mkdir(parents=True, exist_ok=True)

    # Create mock parquet files
    # TODO: Add mock data creation

    yield temp_dir

    # Cleanup
    shutil.rmtree(temp_dir)


class TestWalkForwardSimulation:
    """Tests for WalkForwardSimulation class."""

    def test_init(self):
        """Test simulation initialization."""
        sim = WalkForwardSimulation(
            data_dir='data',
            feature_config='default_features',
            model_config='xgboost_default',
            verbose=False
        )

        assert sim.feature_config_name == 'default_features'
        assert sim.model_config_name == 'xgboost_default'
        assert sim.verbose is False
        assert len(sim.daily_results) == 0
        assert len(sim.lineup_results) == 0

    def test_score_lineups(self):
        """Test lineup scoring against actual results."""
        sim = WalkForwardSimulation(verbose=False)

        # Create mock lineup
        lineups = [{
            'lineup_id': 1,
            'projected_points': 300.0,
            'total_salary': 48000,
            'players': [
                {'playerID': 'player1', 'name': 'Player 1', 'predicted': 40.0, 'salary': 6000},
                {'playerID': 'player2', 'name': 'Player 2', 'predicted': 35.0, 'salary': 5500},
            ]
        }]

        # Create mock actual results
        actual_results = pd.DataFrame({
            'playerID': ['player1', 'player2'],
            'name': ['Player 1', 'Player 2'],
            'fpts': [45.0, 30.0]
        })

        # Score lineups
        scores = sim._score_lineups(lineups, actual_results)

        assert len(scores) == 1
        assert scores[0]['lineup_id'] == 1
        assert scores[0]['actual_points'] == 75.0  # 45 + 30
        assert scores[0]['projected_points'] == 300.0
        assert scores[0]['error'] == -225.0  # 75 - 300
        assert scores[0]['missing_players'] == 0

    def test_score_lineups_missing_players(self):
        """Test lineup scoring with missing players."""
        sim = WalkForwardSimulation(verbose=False)

        lineups = [{
            'lineup_id': 1,
            'projected_points': 300.0,
            'total_salary': 48000,
            'players': [
                {'playerID': 'player1', 'name': 'Player 1', 'predicted': 40.0, 'salary': 6000},
                {'playerID': 'player2', 'name': 'Player 2', 'predicted': 35.0, 'salary': 5500},
                {'playerID': 'player3', 'name': 'Player 3', 'predicted': 30.0, 'salary': 5000},
            ]
        }]

        # Only have actual results for 2 players
        actual_results = pd.DataFrame({
            'playerID': ['player1', 'player2'],
            'name': ['Player 1', 'Player 2'],
            'fpts': [45.0, 30.0]
        })

        scores = sim._score_lineups(lineups, actual_results)

        assert len(scores) == 1
        assert scores[0]['actual_points'] == 75.0  # Only counted players with results
        assert scores[0]['missing_players'] == 1

    def test_aggregate_results_empty(self):
        """Test result aggregation with no data."""
        sim = WalkForwardSimulation(verbose=False)
        results = sim._aggregate_results()

        assert results == {}

    def test_aggregate_results(self):
        """Test result aggregation with mock data."""
        sim = WalkForwardSimulation(verbose=False)

        # Add mock daily results
        sim.daily_results = [
            {
                'date': '20250201',
                'num_lineups': 1,
                'avg_actual_points': 250.0,
                'avg_projected_points': 280.0,
                'avg_error': -30.0,
                'min_actual_points': 250.0,
                'max_actual_points': 250.0,
                'lineup_scores': []
            },
            {
                'date': '20250202',
                'num_lineups': 1,
                'avg_actual_points': 270.0,
                'avg_projected_points': 260.0,
                'avg_error': 10.0,
                'min_actual_points': 270.0,
                'max_actual_points': 270.0,
                'lineup_scores': []
            }
        ]

        # Add mock lineup results
        sim.lineup_results = [
            {
                'date': '20250201',
                'lineup_id': 1,
                'projected_points': 280.0,
                'actual_points': 250.0,
                'error': -30.0,
                'players': []
            },
            {
                'date': '20250202',
                'lineup_id': 1,
                'projected_points': 260.0,
                'actual_points': 270.0,
                'error': 10.0,
                'players': []
            }
        ]

        # Aggregate
        results = sim._aggregate_results()

        assert results['num_slates'] == 2
        assert results['total_lineups'] == 2
        assert results['date_range']['start'] == '20250201'
        assert results['date_range']['end'] == '20250202'

        metrics = results['aggregate_metrics']
        assert metrics['avg_actual_points'] == 260.0  # (250 + 270) / 2
        assert metrics['avg_projected_points'] == 270.0  # (280 + 260) / 2
        assert metrics['avg_error'] == -10.0  # (-30 + 10) / 2
        assert metrics['min_actual_points'] == 250.0
        assert metrics['max_actual_points'] == 270.0


class TestBacktestReport:
    """Tests for BacktestReport class."""

    def test_init_loads_results(self):
        """Test report initialization loads results JSON."""
        # Create temporary results file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            import json
            results = {
                'num_slates': 2,
                'total_lineups': 2,
                'date_range': {'start': '20250201', 'end': '20250202'},
                'aggregate_metrics': {
                    'avg_actual_points': 260.0,
                    'avg_projected_points': 270.0,
                    'avg_error': -10.0,
                    'mae': 20.0,
                    'rmse': 22.36,
                    'min_actual_points': 250.0,
                    'max_actual_points': 270.0
                },
                'daily_breakdown': [
                    {
                        'date': '20250201',
                        'num_lineups': 1,
                        'avg_actual_points': 250.0,
                        'avg_projected_points': 280.0,
                        'avg_error': -30.0
                    }
                ],
                'lineup_results': [
                    {
                        'date': '20250201',
                        'lineup_id': 1,
                        'projected_points': 280.0,
                        'actual_points': 250.0,
                        'error': -30.0
                    }
                ]
            }
            json.dump(results, f)
            temp_path = f.name

        from src.evaluation.backtest_report import BacktestReport

        try:
            report = BacktestReport(temp_path)

            assert report.results['num_slates'] == 2
            assert len(report.daily_df) == 1
            assert len(report.lineup_df) == 1

        finally:
            Path(temp_path).unlink()

    def test_generate_summary(self):
        """Test summary generation."""
        # Create temporary results file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            import json
            results = {
                'num_slates': 2,
                'total_lineups': 2,
                'date_range': {'start': '20250201', 'end': '20250202'},
                'aggregate_metrics': {
                    'avg_actual_points': 260.0,
                    'avg_projected_points': 270.0,
                    'avg_error': -10.0,
                    'mae': 20.0,
                    'rmse': 22.36,
                    'min_actual_points': 250.0,
                    'max_actual_points': 270.0
                },
                'daily_breakdown': [
                    {
                        'date': '20250201',
                        'num_lineups': 1,
                        'avg_actual_points': 250.0,
                        'avg_projected_points': 280.0,
                        'avg_error': -30.0
                    }
                ],
                'lineup_results': [
                    {
                        'date': '20250201',
                        'lineup_id': 1,
                        'projected_points': 280.0,
                        'actual_points': 250.0,
                        'error': -30.0
                    }
                ]
            }
            json.dump(results, f)
            temp_path = f.name

        from src.evaluation.backtest_report import BacktestReport

        try:
            report = BacktestReport(temp_path)
            summary = report.generate_summary()

            assert 'WALK-FORWARD BACKTEST SUMMARY' in summary
            assert 'Slates simulated: 2' in summary
            assert 'Avg actual points: 260.00' in summary
            assert 'Avg projected points: 270.00' in summary

        finally:
            Path(temp_path).unlink()
