from src.features.base import FeatureTransformer
from src.features.pipeline import FeaturePipeline
from src.features.registry import FeatureRegistry, registry
from src.features.transformers.rolling_stats import RollingStatsTransformer
from src.features.transformers.rolling_minmax import RollingMinMaxTransformer
from src.features.transformers.ewma import EWMATransformer
from src.features.transformers.target import TargetTransformer
from src.features.transformers.injury import InjuryTransformer
from src.features.transformers.opponent_stats import OpponentStatsTransformer

registry.register('rolling_stats', RollingStatsTransformer)
registry.register('rolling_minmax', RollingMinMaxTransformer)
registry.register('ewma', EWMATransformer)
registry.register('target', TargetTransformer)
registry.register('injury', InjuryTransformer)
registry.register('opponent_stats', OpponentStatsTransformer)

__all__ = [
    'FeatureTransformer',
    'FeaturePipeline',
    'FeatureRegistry',
    'registry',
    'RollingStatsTransformer',
    'RollingMinMaxTransformer',
    'EWMATransformer',
    'TargetTransformer',
    'InjuryTransformer',
    'OpponentStatsTransformer'
]
