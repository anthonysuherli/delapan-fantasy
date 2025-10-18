from typing import Dict, Any
from .base import Constraint


class SalaryCapConstraint(Constraint):
    """Enforce DraftKings salary cap"""

    def __init__(self, max_salary: int = 50000):
        """
        Initialize salary cap constraint.

        Args:
            max_salary: Maximum total salary allowed (default 50000 for DraftKings)
        """
        self.max_salary = max_salary

    def is_satisfied(self, lineup: Dict[str, Any]) -> bool:
        """
        Check if lineup satisfies salary cap.

        Args:
            lineup: Lineup dictionary with 'total_salary' key

        Returns:
            True if total salary is within cap
        """
        return lineup.get('total_salary', 0) <= self.max_salary

    def get_description(self) -> str:
        """Get human-readable description"""
        return f"Total salary must not exceed ${self.max_salary}"


class RosterSizeConstraint(Constraint):
    """Enforce DraftKings roster size"""

    def __init__(self, roster_size: int = 8):
        """
        Initialize roster size constraint.

        Args:
            roster_size: Required number of players (default 8 for DraftKings NBA)
        """
        self.roster_size = roster_size

    def is_satisfied(self, lineup: Dict[str, Any]) -> bool:
        """
        Check if lineup has correct roster size.

        Args:
            lineup: Lineup dictionary with 'players' key

        Returns:
            True if lineup has exactly roster_size players
        """
        return len(lineup.get('players', [])) == self.roster_size

    def get_description(self) -> str:
        """Get human-readable description"""
        return f"Lineup must contain exactly {self.roster_size} players"


class PositionConstraint(Constraint):
    """Enforce DraftKings position requirements"""

    def __init__(self):
        """
        Initialize position constraint for DraftKings NBA.

        DraftKings NBA requirements:
        - 1-2 PG
        - 1-2 SG
        - 1-2 SF
        - 1-2 PF
        - 1-2 C
        - 1 G (PG or SG)
        - 1 F (SF or PF)
        - 1 UTIL (any position)
        """
        self.position_requirements = {
            'PG': (1, 2),
            'SG': (1, 2),
            'SF': (1, 2),
            'PF': (1, 2),
            'C': (1, 2)
        }

    def is_satisfied(self, lineup: Dict[str, Any]) -> bool:
        """
        Check if lineup satisfies position requirements.

        Args:
            lineup: Lineup dictionary with 'players' key containing position info

        Returns:
            True if position requirements are met
        """
        players = lineup.get('players', [])
        if len(players) != 8:
            return False

        position_counts = {}
        for player in players:
            pos = player.get('position', '')
            if '/' in pos:
                pos = pos.split('/')[0]
            position_counts[pos] = position_counts.get(pos, 0) + 1

        for pos, (min_req, max_req) in self.position_requirements.items():
            count = position_counts.get(pos, 0)
            if count < min_req or count > max_req:
                return False

        return True

    def get_description(self) -> str:
        """Get human-readable description"""
        return "Lineup must satisfy DraftKings position requirements (PG, SG, SF, PF, C, G, F, UTIL)"
