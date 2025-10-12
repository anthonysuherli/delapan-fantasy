import pandas as pd
import numpy as np
from typing import List, Dict, Any
from ..base import BaseOptimizer, BaseConstraint


class LinearProgramOptimizer(BaseOptimizer):
    """
    Linear programming optimizer for cash game lineups.

    Uses PuLP to maximize projected points subject to constraints.
    """

    def __init__(self, constraints: List[BaseConstraint], salary_cap: int = 50000):
        """
        Initialize LP optimizer.

        Args:
            constraints: List of constraints to enforce
            salary_cap: Maximum total salary (default 50000 for DraftKings)
        """
        super().__init__(constraints)
        self.salary_cap = salary_cap

    def optimize(
        self,
        projections: pd.DataFrame,
        num_lineups: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Generate optimal lineups using linear programming.

        Args:
            projections: DataFrame with columns: playerID, playerName, salary, projection, position
            num_lineups: Number of lineups to generate

        Returns:
            List of lineup dictionaries

        Raises:
            ValueError: If no valid lineups can be generated
        """
        try:
            from pulp import LpMaximize, LpProblem, LpVariable, lpSum, LpStatus
        except ImportError:
            raise ImportError("PuLP is required for linear programming optimization. Install with: pip install pulp")

        lineups = []

        for lineup_num in range(num_lineups):
            problem = LpProblem(f"DFS_Lineup_{lineup_num}", LpMaximize)

            player_vars = {}
            for idx, row in projections.iterrows():
                player_vars[idx] = LpVariable(f"player_{idx}", cat='Binary')

            objective = lpSum([
                player_vars[idx] * row['projection']
                for idx, row in projections.iterrows()
            ])
            problem += objective

            problem += lpSum([player_vars[idx] for idx in player_vars]) == 8, "Roster_Size"

            problem += lpSum([
                player_vars[idx] * row['salary']
                for idx, row in projections.iterrows()
            ]) <= self.salary_cap, "Salary_Cap"

            problem.solve()

            if LpStatus[problem.status] != 'Optimal':
                if lineup_num == 0:
                    raise ValueError("No valid lineup could be generated")
                break

            selected_players = []
            total_salary = 0
            total_projection = 0

            for idx, var in player_vars.items():
                if var.varValue == 1:
                    row = projections.iloc[idx]
                    selected_players.append({
                        'playerID': row['playerID'],
                        'playerName': row.get('playerName', ''),
                        'position': row.get('position', ''),
                        'salary': row['salary'],
                        'projection': row['projection']
                    })
                    total_salary += row['salary']
                    total_projection += row['projection']

            lineup = {
                'players': selected_players,
                'total_salary': total_salary,
                'total_projection': total_projection
            }

            lineups.append(lineup)

            if num_lineups > 1:
                for idx in player_vars:
                    if player_vars[idx].varValue == 1:
                        projections = projections.drop(idx)

        return lineups
