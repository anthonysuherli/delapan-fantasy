"""Player filtering implementations for name and ID-based filtering"""

import pandas as pd
from pathlib import Path
from typing import List, Set, Union
from .base import PlayerFilter


class PlayerNameFilter(PlayerFilter):
    """Filter players by exact or partial name match"""

    def __init__(self, names: Union[List[str], str], case_sensitive: bool = False):
        """
        Initialize player name filter.

        Parameters
        ----------
        names : Union[List[str], str]
            Player name(s) to filter for. Can be single name or list of names.
        case_sensitive : bool, optional
            Whether name matching is case sensitive. Default is False.

        Notes
        -----
        Partial matches are supported (e.g., "LeBron" matches "LeBron James")
        """
        super().__init__("PlayerNameFilter")
        self.names = [names] if isinstance(names, str) else names
        self.case_sensitive = case_sensitive

    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Filter players by name.

        Parameters
        ----------
        data : pd.DataFrame
            Player data with 'playerName' or 'name' column

        Returns
        -------
        pd.DataFrame
            Filtered DataFrame containing only matching players

        Raises
        ------
        ValueError
            If required name column not found in data
        """
        # Determine name column
        name_col = 'longName'

        # Apply case insensitivity if needed
        filter_names = self.names
        data_copy = data.copy()

        if not self.case_sensitive:
            filter_names = [name.lower() for name in self.names]
            data_copy[name_col] = data_copy[name_col].str.lower()

        # Filter by partial name match
        mask = data_copy[name_col].isin(filter_names) | \
               data_copy[name_col].str.contains('|'.join(filter_names), regex=True, na=False)

        return data.iloc[mask.values]


class PlayerIDFilter(PlayerFilter):
    """Filter players by exact playerID match"""

    def __init__(self, player_ids: Union[List[int], List[str], int, str]):
        """
        Initialize player ID filter.

        Parameters
        ----------
        player_ids : Union[List[int], List[str], int, str]
            Player ID(s) to filter for. Can be single ID or list of IDs.
            Converted to strings for flexible matching.
        """
        super().__init__("PlayerIDFilter")
        self.player_ids = set(str(pid) for pid in ([player_ids] if isinstance(player_ids, (int, str)) else player_ids))

    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Filter players by playerID.

        Parameters
        ----------
        data : pd.DataFrame
            Player data with 'playerID' or 'id' column

        Returns
        -------
        pd.DataFrame
            Filtered DataFrame containing only matching players

        Raises
        ------
        ValueError
            If required ID column not found in data
        """
        # Determine ID column
        id_col = self._get_id_column(data)

        # Convert data IDs to strings for matching
        mask = data[id_col].astype(str).isin(self.player_ids)

        return data[mask]

    @staticmethod
    def _get_id_column(data: pd.DataFrame) -> str:
        """Identify ID column in DataFrame"""
        for col in ['playerID', 'id', 'playerId', 'player_id']:
            if col in data.columns:
                return col
        raise ValueError(f"No ID column found. Available columns: {data.columns.tolist()}")


class PlayerIDFromCSVFilter(PlayerFilter):
    """Filter players using playerIDs from CSV file"""

    def __init__(self, csv_path: Union[str, Path], id_column: str = 'playerID'):
        """
        Initialize filter from CSV file containing player IDs.

        Parameters
        ----------
        csv_path : Union[str, Path]
            Path to CSV file containing player IDs
        id_column : str, optional
            Name of column in CSV containing player IDs. Default is 'playerID'.

        Raises
        ------
        FileNotFoundError
            If CSV file does not exist
        ValueError
            If specified id_column not found in CSV
        """
        super().__init__("PlayerIDFromCSVFilter")

        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        # Load CSV and extract player IDs
        df = pd.read_csv(csv_path)

        if id_column not in df.columns:
            raise ValueError(
                f"Column '{id_column}' not found in CSV. Available columns: {df.columns.tolist()}"
            )

        self.player_ids = set(str(pid) for pid in df[id_column].dropna())
        self.csv_path = csv_path

    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Filter players by IDs loaded from CSV.

        Parameters
        ----------
        data : pd.DataFrame
            Player data with 'playerID' or 'id' column

        Returns
        -------
        pd.DataFrame
            Filtered DataFrame containing only matching players
        """
        id_col = self._get_id_column(data)
        mask = data[id_col].astype(str).isin(self.player_ids)
        return data[mask]

    @staticmethod
    def _get_id_column(data: pd.DataFrame) -> str:
        """Identify ID column in DataFrame"""
        for col in ['playerID', 'id', 'playerId', 'player_id']:
            if col in data.columns:
                return col
        raise ValueError(f"No ID column found. Available columns: {data.columns.tolist()}")


class PlayerFilterRegistry:
    """Registry for managing and composing multiple player filters"""

    def __init__(self):
        """Initialize empty filter registry"""
        self.filters: List[PlayerFilter] = []

    def add_filter(self, player_filter: PlayerFilter) -> 'PlayerFilterRegistry':
        """
        Add filter to registry.

        Parameters
        ----------
        player_filter : PlayerFilter
            Filter instance to add

        Returns
        -------
        PlayerFilterRegistry
            Self for method chaining
        """
        self.filters.append(player_filter)
        return self

    def add_name_filter(self, names: Union[List[str], str], case_sensitive: bool = False) -> 'PlayerFilterRegistry':
        """
        Add name filter to registry.

        Parameters
        ----------
        names : Union[List[str], str]
            Player name(s) to filter
        case_sensitive : bool, optional
            Case sensitivity for name matching

        Returns
        -------
        PlayerFilterRegistry
            Self for method chaining
        """
        self.add_filter(PlayerNameFilter(names, case_sensitive))
        return self

    def add_id_filter(self, player_ids: Union[List[int], List[str], int, str]) -> 'PlayerFilterRegistry':
        """
        Add ID filter to registry.

        Parameters
        ----------
        player_ids : Union[List[int], List[str], int, str]
            Player ID(s) to filter

        Returns
        -------
        PlayerFilterRegistry
            Self for method chaining
        """
        self.add_filter(PlayerIDFilter(player_ids))
        return self

    def add_csv_filter(self, csv_path: Union[str, Path], id_column: str = 'playerID') -> 'PlayerFilterRegistry':
        """
        Add CSV-based ID filter to registry.

        Parameters
        ----------
        csv_path : Union[str, Path]
            Path to CSV file with player IDs
        id_column : str, optional
            Column name in CSV containing IDs

        Returns
        -------
        PlayerFilterRegistry
            Self for method chaining
        """
        self.add_filter(PlayerIDFromCSVFilter(csv_path, id_column))
        return self

    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all filters sequentially (AND logic).

        Parameters
        ----------
        data : pd.DataFrame
            Player data to filter

        Returns
        -------
        pd.DataFrame
            Data after applying all filters in sequence
        """
        result = data
        for player_filter in self.filters:
            result = player_filter.apply(result)
        return result

    def apply_any(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all filters with OR logic (union of all filter results).

        Parameters
        ----------
        data : pd.DataFrame
            Player data to filter

        Returns
        -------
        pd.DataFrame
            Union of all filter results with duplicates removed
        """
        if not self.filters:
            return data

        result_dfs = [player_filter.apply(data) for player_filter in self.filters]
        combined = pd.concat(result_dfs, ignore_index=False)
        return combined.drop_duplicates()

    def clear(self) -> 'PlayerFilterRegistry':
        """
        Clear all filters from registry.

        Returns
        -------
        PlayerFilterRegistry
            Self for method chaining
        """
        self.filters.clear()
        return self

    def __repr__(self) -> str:
        return f"PlayerFilterRegistry({len(self.filters)} filters: {[f.name for f in self.filters]})"
