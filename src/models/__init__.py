from src.models.base import BaseModel
from src.models.xgboost_model import XGBoostModel
from src.models.random_forest_model import RandomForestModel
from src.models.registry import ModelRegistry, registry

try: 
    registry.register('xgboost', XGBoostModel)
except: 
    pass

try: 
    registry.register('random_forest', RandomForestModel)
except: 
    pass

__all__ = [
    'BaseModel',
    'XGBoostModel',
    'RandomForestModel',
    'ModelRegistry',
    'registry'
]
