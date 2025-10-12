from abc import ABC, abstractmethod
import pandas as pd
from typing import List, Dict, Any


class BaseConstraint(ABC):
    """Abstract base for lineup constraints"""

    @abstractmethod
    def is_satisfied(self, lineup: Dict[str, Any]) -> bool:
        """
        Check if lineup satisfies this constraint.

        Args:
            lineup: Lineup to validate

        Returns:
            True if constraint is satisfied
        """
        pass


class BaseOptimizer(ABC):
    """Abstract base for lineup optimizers"""

    def __init__(self, constraints: List[BaseConstraint]):
        """
        Initialize optimizer with constraints.

        Args:
            constraints: List of constraints to enforce
        """
        self.constraints = constraints

    @abstractmethod
    def optimize(
        self,
        projections: pd.DataFrame,
        num_lineups: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Generate optimal lineups.

        Args:
            projections: Player projections with columns for salary, projection, position
            num_lineups: Number of lineups to generate

        Returns:
            List of lineup dictionaries

        Raises:
            ValueError: If no valid lineups can be generated
        """
        pass

    def validate_lineup(self, lineup: Dict[str, Any]) -> bool:
        """
        Check if lineup satisfies all constraints.

        Args:
            lineup: Lineup to validate

        Returns:
            True if all constraints are satisfied
        """
        return all(c.is_satisfied(lineup) for c in self.constraints)
