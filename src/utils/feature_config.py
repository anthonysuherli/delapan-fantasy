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
        from src.features.transformers.opponent_stats import OpponentStatsTransformer

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

            elif transformer_type == 'opponent_stats':
                features = params.get('features', None)
                lookback_days = params.get('lookback_days', 365)
                recent_games_window = params.get('recent_games_window', 10)
                
                transformer = OpponentStatsTransformer(
                    features=features,
                    lookback_days=lookback_days,
                    recent_games_window=recent_games_window
                )
                pipeline.add(transformer)
                feature_list = features if features else transformer.default_features
                logger.info(f"Added OpponentStatsTransformer: {len(feature_list)} features, lookback={lookback_days}d")

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
    Load a feature configuration by name or combine multiple configurations.

    Args:
        config_name: Name of config file (without .yaml extension) or
                    comma-separated list of config names to combine
                    Examples: 
                    - 'default_features'
                    - 'base_features,opponent_features'
                    - 'default_features,opponent_features,custom_features'

    Returns:
        FeatureConfig instance (merged if multiple configs specified)

    Raises:
        FileNotFoundError: If any config file not found
    """
    from pathlib import Path

    repo_root = Path(__file__).parent.parent.parent
    config_dir = repo_root / 'config' / 'features'
    
    # Check if multiple configs specified
    config_names = [name.strip() for name in config_name.split(',')]
    
    if len(config_names) == 1:
        # Single config - existing behavior
        config_path = config_dir / f'{config_names[0]}.yaml'
        return FeatureConfig(str(config_path))
    else:
        # Multiple configs - merge them
        logger.info(f"Combining {len(config_names)} feature configurations: {config_names}")
        return _merge_feature_configs(config_names, config_dir)


def _merge_feature_configs(config_names: List[str], config_dir: Path) -> FeatureConfig:
    """
    Merge multiple feature configurations into a single combined config.
    
    Args:
        config_names: List of configuration names to merge
        config_dir: Directory containing config files
        
    Returns:
        Combined FeatureConfig instance
    """
    # Load all configs
    configs = []
    for name in config_names:
        config_path = config_dir / f'{name}.yaml'
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        configs.append(FeatureConfig(str(config_path)))
    
    # Create merged config data
    merged_data = _create_merged_config_data(configs)
    
    # Create temporary merged config file
    import tempfile
    import yaml
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(merged_data, f, default_flow_style=False)
        temp_path = f.name
    
    # Create FeatureConfig from merged data
    merged_config = FeatureConfig(temp_path)
    
    # Clean up temp file
    import os
    os.unlink(temp_path)
    
    return merged_config


def _create_merged_config_data(configs: List[FeatureConfig]) -> Dict[str, Any]:
    """
    Create merged configuration data from multiple FeatureConfig instances.
    
    Merging rules:
    - name: Combined from all config names
    - description: Combined from all descriptions  
    - version: Use latest version
    - stats: Union of all stats lists (no duplicates)
    - rolling_windows: Use from first config (they should be consistent)
    - ewma_span: Use from first config
    - metadata_cols: Union of all metadata columns
    - categorical_features: Union of all categorical features
    - numeric_features: Union of all numeric features
    - transformers: Concatenate all transformer lists (order matters)
    """
    if not configs:
        raise ValueError("No configs to merge")
    
    if len(configs) == 1:
        return configs[0].config
    
    # Combine names and descriptions
    names = [config.name for config in configs]
    descriptions = [config.description for config in configs if config.description]
    
    # Use first config as base
    base_config = configs[0]
    
    merged = {
        'name': f"Combined: {' + '.join(names)}",
        'description': f"Merged configuration: {' | '.join(descriptions)}",
        'version': '1.0.0',
        'rolling_windows': base_config.rolling_windows,
        'ewma_span': base_config.ewma_span
    }
    
    # Merge stats (union with preserved order)
    all_stats = []
    seen_stats = set()
    for config in configs:
        for stat in config.stats:
            if stat not in seen_stats:
                all_stats.append(stat)
                seen_stats.add(stat)
    merged['stats'] = all_stats
    
    # Merge metadata columns
    all_metadata = []
    seen_metadata = set()
    for config in configs:
        for col in config.metadata_cols:
            if col not in seen_metadata:
                all_metadata.append(col)
                seen_metadata.add(col)
    merged['metadata_cols'] = all_metadata
    
    # Merge categorical features
    all_categorical = []
    seen_categorical = set()
    for config in configs:
        for feature in config.categorical_features:
            if feature not in seen_categorical:
                all_categorical.append(feature)
                seen_categorical.add(feature)
    merged['categorical_features'] = all_categorical
    
    # Merge numeric features
    all_numeric = []
    seen_numeric = set()
    for config in configs:
        for feature in config.numeric_features:
            if feature not in seen_numeric:
                all_numeric.append(feature)
                seen_numeric.add(feature)
    merged['numeric_features'] = all_numeric
    
    # Merge transformers (concatenate in order)
    all_transformers = []
    for config in configs:
        all_transformers.extend(config.transformers)
    merged['transformers'] = all_transformers
    
    logger.info(f"Merged config created:")
    logger.info(f"  Stats: {len(merged['stats'])} total")
    logger.info(f"  Categorical features: {len(merged['categorical_features'])}")
    logger.info(f"  Transformers: {len(merged['transformers'])}")
    
    return merged
