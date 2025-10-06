import json
from typing import Dict, List, Optional, Any
from pathlib import Path


class BacktestConfig:

    def __init__(
        self,
        start_date: str,
        end_date: str,
        lookback_days: int = 90,
        model_type: str = 'xgboost'
    ):
        self.start_date = start_date
        self.end_date = end_date
        self.lookback_days = lookback_days
        self.model_type = model_type

        self.model_params = {
            'max_depth': 6,
            'learning_rate': 0.05,
            'n_estimators': 200,
            'min_child_weight': 5,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }

        self.features_to_use = [
            'pts_avg_3', 'pts_avg_5', 'pts_avg_10',
            'reb_avg_3', 'reb_avg_5',
            'ast_avg_3', 'ast_avg_5',
            'mins_avg_5', 'usage_rate',
            'pts_ewma', 'pts_std_5',
            'fpts_avg_3', 'fpts_avg_5', 'fpts_avg_10',
            'fpts_ewma', 'fpts_std_5'
        ]

        self.cash_lineups_per_slate = 3
        self.gpp_lineups_per_slate = 150

        self.min_training_games = 500

        self.save_daily_results = True
        self.output_dir = 'data/backtest_results'

        self._validate()

    def _validate(self):
        from datetime import datetime

        if len(self.start_date) != 8 or not self.start_date.isdigit():
            raise ValueError(f"start_date must be in YYYYMMDD format, got: {self.start_date}")

        if len(self.end_date) != 8 or not self.end_date.isdigit():
            raise ValueError(f"end_date must be in YYYYMMDD format, got: {self.end_date}")

        try:
            start_dt = datetime.strptime(self.start_date, '%Y%m%d')
            end_dt = datetime.strptime(self.end_date, '%Y%m%d')

            if start_dt >= end_dt:
                raise ValueError(f"start_date must be before end_date")
        except ValueError as e:
            raise ValueError(f"Invalid date format: {str(e)}")

        if self.lookback_days < 1:
            raise ValueError(f"lookback_days must be positive, got: {self.lookback_days}")

        if self.model_type not in ['xgboost', 'random_forest', 'linear', 'ensemble']:
            raise ValueError(f"Unsupported model_type: {self.model_type}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            'start_date': self.start_date,
            'end_date': self.end_date,
            'lookback_days': self.lookback_days,
            'model_type': self.model_type,
            'model_params': self.model_params,
            'features': self.features_to_use,
            'cash_lineups_per_slate': self.cash_lineups_per_slate,
            'gpp_lineups_per_slate': self.gpp_lineups_per_slate,
            'min_training_games': self.min_training_games,
            'output_dir': self.output_dir
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'BacktestConfig':
        config = cls(
            start_date=config_dict['start_date'],
            end_date=config_dict['end_date'],
            lookback_days=config_dict.get('lookback_days', 90),
            model_type=config_dict.get('model_type', 'xgboost')
        )

        if 'model_params' in config_dict:
            config.model_params.update(config_dict['model_params'])

        if 'features' in config_dict:
            config.features_to_use = config_dict['features']

        if 'cash_lineups_per_slate' in config_dict:
            config.cash_lineups_per_slate = config_dict['cash_lineups_per_slate']

        if 'gpp_lineups_per_slate' in config_dict:
            config.gpp_lineups_per_slate = config_dict['gpp_lineups_per_slate']

        if 'min_training_games' in config_dict:
            config.min_training_games = config_dict['min_training_games']

        if 'output_dir' in config_dict:
            config.output_dir = config_dict['output_dir']

        return config

    @classmethod
    def from_json(cls, json_path: str) -> 'BacktestConfig':
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    def to_json(self, json_path: str):
        Path(json_path).parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
