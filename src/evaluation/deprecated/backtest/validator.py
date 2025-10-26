import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
from ..metrics.base import BaseMetric


class Validator:
    """Reusable walk-forward validation"""

    def __init__(self, model, metrics: List[BaseMetric]):
        """
        Initialize validator.

        Args:
            model: Model instance with train() and predict() methods
            metrics: List of metrics to evaluate
        """
        self.model = model
        self.metrics = metrics

    def validate(
        self,
        train_data: Tuple[pd.DataFrame, pd.Series],
        test_data: Tuple[pd.DataFrame, pd.Series]
    ) -> Dict[str, float]:
        """
        Perform single validation iteration.

        Args:
            train_data: Tuple of (X_train, y_train)
            test_data: Tuple of (X_test, y_test)

        Returns:
            Dictionary mapping metric names to values
        """
        X_train, y_train = train_data
        X_test, y_test = test_data

        self.model.train(X_train, y_train)

        y_pred = self.model.predict(X_test)

        results = {}
        for metric in self.metrics:
            results[metric.name] = metric.calculate(y_test.values, y_pred)

        return results

    def walk_forward_validate(
        self,
        data_splits: List[Tuple[Tuple[pd.DataFrame, pd.Series], Tuple[pd.DataFrame, pd.Series]]]
    ) -> List[Dict[str, float]]:
        """
        Perform walk-forward validation across multiple splits.

        Args:
            data_splits: List of (train_data, test_data) tuples

        Returns:
            List of results dictionaries, one per split
        """
        results = []
        for train_data, test_data in data_splits:
            split_results = self.validate(train_data, test_data)
            results.append(split_results)
        return results
