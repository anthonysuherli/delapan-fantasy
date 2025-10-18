from src.evaluation.backtest.validator import Validator
from src.evaluation.metrics.accuracy import MAPEMetric, RMSEMetric, MAEMetric, CorrelationMetric
from src.evaluation.metrics.registry import MetricRegistry, registry

try: 
    registry.register('mape', MAPEMetric)
except: 
    pass
try: 
    registry.register('rmse', RMSEMetric)
except: 
    pass
try: 
    registry.register('mae', MAEMetric)
except: 
    pass
try: 
    registry.register('correlation', CorrelationMetric)
except: 
    pass


__all__ = [
    'Validator',
    'MAPEMetric',
    'RMSEMetric',
    'MAEMetric',
    'CorrelationMetric',
    'MetricRegistry',
    'registry'
]
