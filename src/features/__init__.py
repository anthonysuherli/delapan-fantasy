from src.features.base import FeatureTransformer
from src.features.pipeline import FeaturePipeline
from src.features.registry import FeatureRegistry, registry
from src.features.transformers.rolling_stats import RollingStatsTransformer
from src.features.transformers.rolling_minmax import RollingMinMaxTransformer
from src.features.transformers.ewma import EWMATransformer
from src.features.transformers.target import TargetTransformer

registry.register('rolling_stats', RollingStatsTransformer)
registry.register('rolling_minmax', RollingMinMaxTransformer)
registry.register('ewma', EWMATransformer)
registry.register('target', TargetTransformer)

__all__ = [
    'FeatureTransformer',
    'FeaturePipeline',
    'FeatureRegistry',
    'registry',
    'RollingStatsTransformer',
    'RollingMinMaxTransformer',
    'EWMATransformer',
    'TargetTransformer'
]
