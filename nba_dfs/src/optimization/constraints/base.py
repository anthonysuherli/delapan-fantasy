from abc import ABC, abstractmethod
from typing import Dict, Any


class Constraint(ABC):
    """Abstract constraint class"""

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

    @abstractmethod
    def get_description(self) -> str:
        """
        Get human-readable description of constraint.

        Returns:
            Description string
        """
        pass
