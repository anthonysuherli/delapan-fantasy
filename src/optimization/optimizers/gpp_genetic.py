"""
GPP Genetic Algorithm Optimizer.

Tournament-focused lineup optimizer using genetic algorithms.
Optimizes for high-ceiling projections and low ownership to maximize
tournament EV through differentiation.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from ..base import BaseOptimizer


class GPPGeneticOptimizer(BaseOptimizer):
    """
    Genetic algorithm optimizer for GPP tournaments.

    Uses evolutionary approach to generate diverse, high-ceiling lineups:
    - Fitness based on ceiling projections (90th percentile)
    - Ownership penalty to favor contrarian plays
    - Population diversity through crossover and mutation
    - Multi-entry tournament optimization
    """

    def __init__(
        self,
        constraints: List,
        salary_cap: int = 50000,
        population_size: int = 100,
        generations: int = 50,
        mutation_rate: float = 0.15,
        crossover_rate: float = 0.7,
        ownership_weight: float = 0.3,
        diversity_weight: float = 0.2,
        elite_fraction: float = 0.1,
        random_seed: Optional[int] = None
    ):
        """
        Initialize GPP genetic algorithm optimizer.

        Parameters
        ----------
        constraints : List
            DraftKings constraints (roster size, positions, etc)
        salary_cap : int
            Maximum salary allowed (default 50000)
        population_size : int
            Number of lineups in each generation (default 100)
        generations : int
            Number of evolutionary iterations (default 50)
        mutation_rate : float
            Probability of random player swap (default 0.15)
        crossover_rate : float
            Probability of combining parent lineups (default 0.7)
        ownership_weight : float
            Weight for ownership penalty [0-1] (default 0.3)
        diversity_weight : float
            Weight for lineup uniqueness [0-1] (default 0.2)
        elite_fraction : float
            Fraction of top lineups preserved each generation (default 0.1)
        random_seed : int, optional
            Random seed for reproducibility
        """
        super().__init__(constraints)
        self.salary_cap = salary_cap
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.ownership_weight = ownership_weight
        self.diversity_weight = diversity_weight
        self.elite_fraction = elite_fraction

        if random_seed is not None:
            np.random.seed(random_seed)

        # Position requirements for DraftKings NBA
        self.position_slots = ['PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'UTIL']
        self.position_eligibility = {
            'G': ['PG', 'SG'],
            'F': ['SF', 'PF'],
            'UTIL': ['PG', 'SG', 'SF', 'PF', 'C']
        }

    def optimize(
        self,
        projections: pd.DataFrame,
        num_lineups: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Generate optimal GPP lineups using genetic algorithm.

        Parameters
        ----------
        projections : pd.DataFrame
            Player projections with columns:
            - playerID: Unique player identifier
            - name: Player name
            - salary: DraftKings salary
            - predicted: Expected fantasy points (median)
            - ceiling: High-end projection (90th percentile)
            - ownership: Projected ownership percentage (optional)
            - allValidPositions: List of eligible positions
        num_lineups : int
            Number of unique lineups to generate (default 1)

        Returns
        -------
        List[Dict[str, Any]]
            List of optimized lineups with players and metadata
        """
        # Validate required columns
        required_cols = ['playerID', 'name', 'salary', 'allValidPositions']
        missing = [col for col in required_cols if col not in projections.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Use ceiling if available, otherwise predicted
        if 'ceiling' in projections.columns:
            projections['optimization_target'] = projections['ceiling']
        else:
            projections['optimization_target'] = projections['predicted']

        # Default ownership to uniform if not provided
        if 'ownership' not in projections.columns:
            projections['ownership'] = 100.0 / len(projections)

        self.projections = projections.copy()

        # Convert positions to standardized format
        self.projections['positions'] = self.projections['allValidPositions'].apply(
            lambda x: x if isinstance(x, list) else [x]
        )

        # Initialize population
        population = self._initialize_population()

        # Evolve population
        for generation in range(self.generations):
            # Calculate fitness for all lineups
            fitness_scores = [self._fitness(lineup) for lineup in population]

            # Select elite lineups
            elite_size = max(1, int(self.population_size * self.elite_fraction))
            elite_indices = np.argsort(fitness_scores)[-elite_size:]
            elite_lineups = [population[i] for i in elite_indices]

            # Generate new population
            new_population = elite_lineups.copy()

            while len(new_population) < self.population_size:
                # Tournament selection
                parent1 = self._tournament_select(population, fitness_scores)
                parent2 = self._tournament_select(population, fitness_scores)

                # Crossover
                if np.random.random() < self.crossover_rate:
                    child = self._crossover(parent1, parent2)
                else:
                    child = parent1.copy()

                # Mutation
                if np.random.random() < self.mutation_rate:
                    child = self._mutate(child)

                # Add to population if valid
                if self._is_valid_lineup(child):
                    new_population.append(child)

            population = new_population[:self.population_size]

        # Select top N unique lineups
        fitness_scores = [self._fitness(lineup) for lineup in population]
        top_indices = np.argsort(fitness_scores)[-num_lineups * 3:]  # Get extras for deduplication

        # Deduplicate lineups
        unique_lineups = []
        seen_lineups = set()

        for idx in reversed(top_indices):
            lineup = population[idx]
            lineup_key = tuple(sorted(lineup))

            if lineup_key not in seen_lineups:
                seen_lineups.add(lineup_key)
                unique_lineups.append(self._format_lineup(lineup))

                if len(unique_lineups) >= num_lineups:
                    break

        if len(unique_lineups) < num_lineups:
            raise ValueError(
                f"Could not generate {num_lineups} unique lineups. "
                f"Generated {len(unique_lineups)}. Try reducing num_lineups or "
                f"increasing population_size."
            )

        return unique_lineups

    def _initialize_population(self) -> List[List[str]]:
        """
        Generate initial random population of valid lineups.

        Returns
        -------
        List[List[str]]
            Population of lineups (each lineup is list of playerIDs)
        """
        population = []
        attempts = 0
        max_attempts = self.population_size * 100

        while len(population) < self.population_size and attempts < max_attempts:
            lineup = self._random_lineup()
            if self._is_valid_lineup(lineup):
                population.append(lineup)
            attempts += 1

        if len(population) < self.population_size:
            raise ValueError(
                f"Could not initialize population of size {self.population_size}. "
                f"Only generated {len(population)} valid lineups. "
                f"Check player pool and constraints."
            )

        return population

    def _random_lineup(self) -> List[str]:
        """
        Generate random lineup satisfying position and salary constraints.

        Returns
        -------
        List[str]
            List of 8 playerIDs
        """
        lineup = []
        available_players = self.projections.copy()

        # Fill each position slot
        for slot in self.position_slots:
            # Get eligible positions for this slot
            if slot in self.position_eligibility:
                eligible_positions = self.position_eligibility[slot]
            else:
                eligible_positions = [slot]

            # Filter available players by position
            candidates = available_players[
                available_players['positions'].apply(
                    lambda positions: any(pos in positions for pos in eligible_positions)
                )
            ]

            # Filter by remaining salary
            current_salary = sum(
                self.projections[self.projections['playerID'] == pid]['salary'].values[0]
                for pid in lineup
            )
            remaining_salary = self.salary_cap - current_salary
            candidates = candidates[candidates['salary'] <= remaining_salary]

            if len(candidates) == 0:
                # No valid candidates, start over
                return self._random_lineup()

            # Weighted random selection (favor high ceiling)
            weights = candidates['optimization_target'].values
            weights = weights / weights.sum()

            selected = np.random.choice(candidates['playerID'].values, p=weights)
            lineup.append(selected)

            # Remove selected player from available pool
            available_players = available_players[available_players['playerID'] != selected]

        return lineup

    def _fitness(self, lineup: List[str]) -> float:
        """
        Calculate fitness score for lineup.

        Fitness = ceiling_points - ownership_penalty + diversity_bonus

        Parameters
        ----------
        lineup : List[str]
            List of playerIDs

        Returns
        -------
        float
            Fitness score (higher is better)
        """
        lineup_df = self.projections[self.projections['playerID'].isin(lineup)]

        # Base fitness: sum of ceiling projections
        ceiling_points = lineup_df['optimization_target'].sum()

        # Ownership penalty: higher ownership = lower fitness
        total_ownership = lineup_df['ownership'].sum()
        ownership_penalty = self.ownership_weight * total_ownership

        # Diversity bonus: reward lineups with variance
        projection_std = lineup_df['optimization_target'].std()
        diversity_bonus = self.diversity_weight * projection_std

        fitness = ceiling_points - ownership_penalty + diversity_bonus

        return fitness

    def _tournament_select(
        self,
        population: List[List[str]],
        fitness_scores: List[float],
        tournament_size: int = 3
    ) -> List[str]:
        """
        Select parent using tournament selection.

        Parameters
        ----------
        population : List[List[str]]
            Current population
        fitness_scores : List[float]
            Fitness scores for population
        tournament_size : int
            Number of candidates in tournament (default 3)

        Returns
        -------
        List[str]
            Selected parent lineup
        """
        tournament_indices = np.random.choice(
            len(population),
            size=tournament_size,
            replace=False
        )
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_index = tournament_indices[np.argmax(tournament_fitness)]

        return population[winner_index].copy()

    def _crossover(self, parent1: List[str], parent2: List[str]) -> List[str]:
        """
        Combine two parent lineups through crossover.

        Parameters
        ----------
        parent1 : List[str]
            First parent lineup
        parent2 : List[str]
            Second parent lineup

        Returns
        -------
        List[str]
            Child lineup
        """
        # Single-point crossover
        crossover_point = np.random.randint(1, len(parent1))

        child = parent1[:crossover_point] + parent2[crossover_point:]

        # Remove duplicates by keeping first occurrence
        seen = set()
        unique_child = []
        for player_id in child:
            if player_id not in seen:
                seen.add(player_id)
                unique_child.append(player_id)

        # If we have fewer than 8 players, fill from both parents
        if len(unique_child) < 8:
            all_parent_players = parent1 + parent2
            for player_id in all_parent_players:
                if player_id not in seen and len(unique_child) < 8:
                    seen.add(player_id)
                    unique_child.append(player_id)

        # If still not enough, fill randomly
        while len(unique_child) < 8:
            available = self.projections[
                ~self.projections['playerID'].isin(unique_child)
            ]
            if len(available) == 0:
                break
            random_player = np.random.choice(available['playerID'].values)
            unique_child.append(random_player)

        return unique_child[:8]

    def _mutate(self, lineup: List[str]) -> List[str]:
        """
        Mutate lineup by swapping random player.

        Parameters
        ----------
        lineup : List[str]
            Original lineup

        Returns
        -------
        List[str]
            Mutated lineup
        """
        mutated = lineup.copy()

        # Select random position to mutate
        mutate_idx = np.random.randint(len(mutated))
        removed_player = mutated[mutate_idx]

        # Get position requirements for this slot
        slot = self.position_slots[mutate_idx]
        if slot in self.position_eligibility:
            eligible_positions = self.position_eligibility[slot]
        else:
            eligible_positions = [slot]

        # Find eligible replacements
        candidates = self.projections[
            (self.projections['positions'].apply(
                lambda positions: any(pos in positions for pos in eligible_positions)
            )) &
            (~self.projections['playerID'].isin(mutated))
        ]

        # Filter by salary (swap should maintain salary feasibility)
        removed_salary = self.projections[
            self.projections['playerID'] == removed_player
        ]['salary'].values[0]

        current_salary = sum(
            self.projections[self.projections['playerID'] == pid]['salary'].values[0]
            for pid in mutated
        )
        remaining_salary = self.salary_cap - (current_salary - removed_salary)

        candidates = candidates[candidates['salary'] <= remaining_salary]

        if len(candidates) > 0:
            # Weighted selection favoring high ceiling
            weights = candidates['optimization_target'].values
            weights = weights / weights.sum()
            new_player = np.random.choice(candidates['playerID'].values, p=weights)
            mutated[mutate_idx] = new_player

        return mutated

    def _is_valid_lineup(self, lineup: List[str]) -> bool:
        """
        Check if lineup satisfies all constraints.

        Parameters
        ----------
        lineup : List[str]
            List of playerIDs

        Returns
        -------
        bool
            True if lineup is valid
        """
        if len(lineup) != 8:
            return False

        if len(set(lineup)) != 8:  # Check for duplicates
            return False

        # Check salary
        lineup_df = self.projections[self.projections['playerID'].isin(lineup)]
        total_salary = lineup_df['salary'].sum()

        if total_salary > self.salary_cap:
            return False

        # Check position eligibility for each slot
        for idx, slot in enumerate(self.position_slots):
            player_id = lineup[idx]
            player = self.projections[self.projections['playerID'] == player_id].iloc[0]

            if slot in self.position_eligibility:
                eligible_positions = self.position_eligibility[slot]
            else:
                eligible_positions = [slot]

            if not any(pos in player['positions'] for pos in eligible_positions):
                return False

        return True

    def _format_lineup(self, lineup: List[str]) -> Dict[str, Any]:
        """
        Format lineup as output dictionary.

        Parameters
        ----------
        lineup : List[str]
            List of playerIDs

        Returns
        -------
        Dict[str, Any]
            Formatted lineup with metadata
        """
        lineup_df = self.projections[self.projections['playerID'].isin(lineup)]

        players = []
        for idx, player_id in enumerate(lineup):
            player = lineup_df[lineup_df['playerID'] == player_id].iloc[0]
            players.append({
                'playerID': player_id,
                'name': player['name'],
                'position': self.position_slots[idx],
                'salary': player['salary'],
                'predicted': player.get('predicted', player['optimization_target']),
                'ceiling': player.get('ceiling', player['optimization_target']),
                'ownership': player.get('ownership', 0.0)
            })

        return {
            'players': players,
            'total_salary': lineup_df['salary'].sum(),
            'total_predicted': lineup_df.get('predicted', lineup_df['optimization_target']).sum(),
            'total_ceiling': lineup_df['optimization_target'].sum(),
            'avg_ownership': lineup_df['ownership'].mean(),
            'total_ownership': lineup_df['ownership'].sum()
        }
