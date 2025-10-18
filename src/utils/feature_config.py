"""
Feature configuration loader.

Loads feature configuration from YAML files and builds feature pipelines.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, List, Tuple
import logging

logger = logging.getLogger(__name__)


class FeatureConfig:
    """Feature configuration loader and manager."""

    def __init__(self, config_path: str):
        """
        Initialize feature configuration.

        Args:
            config_path: Path to YAML config file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)

        logger.info(f"Loaded feature config: {config.get('name', 'Unknown')}")
        return config

    @property
    def name(self) -> str:
        """Get config name."""
        return self.config.get('name', 'Unknown')

    @property
    def description(self) -> str:
        """Get config description."""
        return self.config.get('description', '')

    @property
    def version(self) -> str:
        """Get config version."""
        return self.config.get('version', '1.0.0')

    @property
    def stats(self) -> List[str]:
        """Get list of stats to use for features."""
        return self.config.get('stats', [])

    @property
    def rolling_windows(self) -> List[int]:
        """Get rolling window sizes."""
        return self.config.get('rolling_windows', [3, 5, 10])

    @property
    def ewma_span(self) -> int:
        """Get EWMA span parameter."""
        return self.config.get('ewma_span', 5)

    @property
    def metadata_cols(self) -> List[str]:
        """Get list of metadata columns (non-feature columns)."""
        return self.config.get('metadata_cols', [
            'playerID', 'playerName', 'team', 'pos', 'gameDate',
            'fpts', 'target', 'pts', 'reb', 'ast', 'stl', 'blk', 'TOV', 'mins'
        ])

    @property
    def categorical_features(self) -> List[str]:
        """Get list of categorical features."""
        return self.config.get('categorical_features', [])

    @property
    def numeric_features(self) -> List[str]:
        """
        Get list of explicitly defined numeric features.
        If empty, numeric features are determined dynamically.
        """
        return self.config.get('numeric_features', [])

    @property
    def transformers(self) -> List[Dict[str, Any]]:
        """Get transformer configurations."""
        return self.config.get('transformers', [])

    def get_transformer_params(self, transformer_type: str) -> Dict[str, Any]:
        """
        Get parameters for a specific transformer type.

        Args:
            transformer_type: Type of transformer (e.g., 'rolling_stats', 'ewma')

        Returns:
            Dictionary of parameters for the transformer
        """
        for transformer in self.transformers:
            if transformer.get('type') == transformer_type:
                return transformer.get('params', {})
        return {}

    def build_pipeline(self, feature_pipeline_class):
        """
        Build a feature pipeline from configuration.

        Args:
            feature_pipeline_class: FeaturePipeline class to instantiate

        Returns:
            Configured FeaturePipeline instance
        """
        from src.features.transformers.rolling_stats import RollingStatsTransformer
        from src.features.transformers.ewma import EWMATransformer
        from src.features.transformers.target import TargetTransformer
        from src.features.transformers.injury import InjuryTransformer

        pipeline = feature_pipeline_class()

        for transformer_config in self.transformers:
            transformer_type = transformer_config.get('type')
            params = transformer_config.get('params', {})

            if transformer_type == 'rolling_stats':
                windows = params.get('windows', self.rolling_windows)
                include_std = params.get('include_std', True)

                transformer = RollingStatsTransformer(
                    windows=windows,
                    stats=self.stats,
                    include_std=include_std
                )
                pipeline.add(transformer)
                logger.info(f"Added RollingStatsTransformer: windows={windows}, stats={len(self.stats)}, include_std={include_std}")

            elif transformer_type == 'ewma':
                span = params.get('span', self.ewma_span)

                transformer = EWMATransformer(
                    span=span,
                    stats=self.stats
                )
                pipeline.add(transformer)
                logger.info(f"Added EWMATransformer: span={span}, stats={len(self.stats)}")

            elif transformer_type == 'target':
                target_col = params.get('target_col', 'fpts')
                shift_periods = params.get('shift_periods', -1)

                transformer = TargetTransformer(
                    target_col=target_col,
                    shift_periods=shift_periods
                )
                pipeline.add(transformer)
                logger.info(f"Added TargetTransformer: target_col={target_col}, shift_periods={shift_periods}")

            elif transformer_type == 'injury':
                transformer = InjuryTransformer()
                pipeline.add(transformer)
                logger.info(f"Added InjuryTransformer")

            else:
                logger.warning(f"Unknown transformer type: {transformer_type}")

        return pipeline

    def get_feature_columns(self, df) -> Tuple[List[str], List[str]]:
        """
        Get categorical and numeric feature columns from DataFrame.

        Args:
            df: DataFrame to extract feature columns from

        Returns:
            Tuple of (categorical_cols, numeric_cols)
        """
        all_cols = list(df.columns)

        # Categorical features defined in config
        categorical_cols = [col for col in self.categorical_features if col in all_cols]

        # If numeric features explicitly defined, use those
        if self.numeric_features:
            numeric_cols = [col for col in self.numeric_features if col in all_cols]
        else:
            # Otherwise, infer numeric features
            numeric_cols = [
                col for col in all_cols
                if col not in self.metadata_cols
                and col not in categorical_cols
                and df[col].dtype in ['int64', 'float64', 'float32', 'bool']
            ]

        return categorical_cols, numeric_cols

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'name': self.name,
            'description': self.description,
            'version': self.version,
            'stats': self.stats,
            'rolling_windows': self.rolling_windows,
            'ewma_span': self.ewma_span,
            'metadata_cols': self.metadata_cols,
            'categorical_features': self.categorical_features,
            'numeric_features': self.numeric_features,
            'transformers': self.transformers
        }

    def __repr__(self) -> str:
        """String representation."""
        return f"FeatureConfig(name='{self.name}', version={self.version}, stats={len(self.stats)})"


def load_feature_config(config_name: str = 'default_features') -> FeatureConfig:
    """
    Load a feature configuration by name.

    Args:
        config_name: Name of config file (without .yaml extension)

    Returns:
        FeatureConfig instance

    Raises:
        FileNotFoundError: If config file not found
    """
    from pathlib import Path

    repo_root = Path(__file__).parent.parent.parent
    config_path = repo_root / 'config' / 'features' / f'{config_name}.yaml'

    return FeatureConfig(str(config_path))
