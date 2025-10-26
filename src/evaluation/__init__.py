from src.evaluation.metrics.accuracy import MAPEMetric, RMSEMetric, MAEMetric, CorrelationMetric
from src.evaluation.metrics.registry import MetricRegistry, registry
from src.evaluation.walk_forward_simulation import WalkForwardSimulation
from src.evaluation.backtest_report import BacktestReport

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
    'WalkForwardSimulation',
    'BacktestReport',
    'MAPEMetric',
    'RMSEMetric',
    'MAEMetric',
    'CorrelationMetric',
    'MetricRegistry',
    'registry'
]
