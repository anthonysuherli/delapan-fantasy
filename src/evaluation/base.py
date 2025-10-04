from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import pandas as pd


class MetricCalculator(ABC):

    @abstractmethod
    def calculate(self, actual: pd.Series, predicted: pd.Series) -> float:
        pass


class BacktestStrategy(ABC):

    @abstractmethod
    def run(
        self,
        start_date: str,
        end_date: str,
        **kwargs
    ) -> Dict[str, Any]:
        pass

    @abstractmethod
    def get_results(self) -> pd.DataFrame:
        pass


class ValidationStrategy(ABC):

    @abstractmethod
    def split(
        self,
        data: pd.DataFrame,
        date_column: str
    ) -> List[tuple[pd.DataFrame, pd.DataFrame]]:
        pass
