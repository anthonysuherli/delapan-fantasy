"""
Contest Manager for handling DraftKings contests and lineup associations
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
from datetime import datetime

logger = logging.getLogger(__name__)


class ContestManager:
    """
    Manage DraftKings contests and lineup-to-contest associations
    """

    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize ContestManager.

        Parameters
        ----------
        storage_path : Optional[str]
            Path to store contest data
        """
        if storage_path:
            self.storage_path = Path(storage_path)
        else:
            self.storage_path = Path(__file__).parent.parent.parent / "data" / "contests"

        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.contests = {}
        self.lineup_associations = {}

    def load_contests_from_api(self, contests_data: List[Dict]) -> pd.DataFrame:
        """
        Load contest data from DraftKings API response.

        Parameters
        ----------
        contests_data : List[Dict]
            List of contest dictionaries from API

        Returns
        -------
        pd.DataFrame
            Processed contest data
        """
        contests_df = pd.DataFrame(contests_data)

        # Standardize columns
        column_mapping = {
            'contest_id': 'contestID',
            'name': 'contestName',
            'entry_fee': 'entryFee',
            'total_payouts': 'totalPrize',
            'max_entries': 'maxEntries',
            'entries': 'currentEntries',
            'game_type': 'gameType',
            'is_guaranteed': 'isGuaranteed',
            'starts_at': 'startTime'
        }

        # Rename columns if they exist
        for new_col, old_col in column_mapping.items():
            if old_col in contests_df.columns:
                contests_df[new_col] = contests_df[old_col]

        # Classify contest types
        contests_df['contest_category'] = contests_df.apply(
            self._classify_contest, axis=1
        )

        # Store in memory
        for _, contest in contests_df.iterrows():
            self.contests[contest.get('contestID', contest.name)] = contest.to_dict()

        logger.info(f"Loaded {len(contests_df)} contests")

        return contests_df

    def _classify_contest(self, contest: pd.Series) -> str:
        """
        Classify contest type based on characteristics.

        Parameters
        ----------
        contest : pd.Series
            Contest data

        Returns
        -------
        str
            Contest category
        """
        name = str(contest.get('contestName', '')).lower()
        entry_fee = contest.get('entryFee', 0)
        max_entries = contest.get('maxEntries', 1)

        # Check for specific contest types
        if 'gpp' in name or 'tournament' in name or entry_fee >= 100:
            return 'gpp_tournament'
        elif '50/50' in name or 'fifty' in name or 'double' in name:
            return 'cash_game'
        elif 'single entry' in name or max_entries == 1:
            return 'single_entry'
        elif max_entries > 1:
            return 'multi_entry'
        else:
            return 'unknown'

    def get_contest_config(self, contest_id: str) -> str:
        """
        Get appropriate configuration file for a contest.

        Parameters
        ----------
        contest_id : str
            Contest ID

        Returns
        -------
        str
            Path to configuration file
        """
        contest = self.contests.get(contest_id, {})
        category = contest.get('contest_category', 'cash_game')

        config_dir = Path(__file__).parent.parent.parent / "config" / "contests"
        config_path = config_dir / f"{category}.json"

        if not config_path.exists():
            config_path = config_dir / "cash_game.json"

        return str(config_path)

    def filter_contests(self,
                        min_entry_fee: Optional[float] = None,
                        max_entry_fee: Optional[float] = None,
                        contest_types: Optional[List[str]] = None,
                        guaranteed_only: bool = False) -> pd.DataFrame:
        """
        Filter contests based on criteria.

        Parameters
        ----------
        min_entry_fee : Optional[float]
            Minimum entry fee
        max_entry_fee : Optional[float]
            Maximum entry fee
        contest_types : Optional[List[str]]
            List of contest types to include
        guaranteed_only : bool
            Only include guaranteed contests

        Returns
        -------
        pd.DataFrame
            Filtered contests
        """
        contests_df = pd.DataFrame(list(self.contests.values()))

        if contests_df.empty:
            return contests_df

        # Apply filters
        if min_entry_fee is not None:
            contests_df = contests_df[contests_df['entryFee'] >= min_entry_fee]

        if max_entry_fee is not None:
            contests_df = contests_df[contests_df['entryFee'] <= max_entry_fee]

        if contest_types:
            contests_df = contests_df[
                contests_df['contest_category'].isin(contest_types)
            ]

        if guaranteed_only:
            contests_df = contests_df[contests_df.get('isGuaranteed', False)]

        return contests_df

    def associate_lineups_with_contests(self,
                                         lineups: List[Dict],
                                         contest_ids: List[str]) -> List[Dict]:
        """
        Associate generated lineups with specific contests.

        Parameters
        ----------
        lineups : List[Dict]
            List of lineup dictionaries
        contest_ids : List[str]
            List of contest IDs

        Returns
        -------
        List[Dict]
            Lineups with contest associations
        """
        associations = []

        for lineup in lineups:
            lineup_id = f"lineup_{lineup.get('lineup_num', 0)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # Store association
            self.lineup_associations[lineup_id] = {
                'lineup': lineup,
                'contest_ids': contest_ids,
                'created_at': datetime.now().isoformat()
            }

            # Add contest info to lineup
            lineup_with_contests = lineup.copy()
            lineup_with_contests['lineup_id'] = lineup_id
            lineup_with_contests['contests'] = []

            for contest_id in contest_ids:
                if contest_id in self.contests:
                    contest_info = {
                        'contest_id': contest_id,
                        'contest_name': self.contests[contest_id].get('contestName'),
                        'entry_fee': self.contests[contest_id].get('entryFee'),
                        'contest_type': self.contests[contest_id].get('contest_category')
                    }
                    lineup_with_contests['contests'].append(contest_info)

            associations.append(lineup_with_contests)

        return associations

    def generate_entry_file(self,
                            lineups: List[Dict],
                            contest_id: str,
                            output_path: str) -> Path:
        """
        Generate DraftKings entry file for contest.

        Parameters
        ----------
        lineups : List[Dict]
            List of lineups to enter
        contest_id : str
            Contest ID
        output_path : str
            Output file path

        Returns
        -------
        Path
            Path to generated file
        """
        output_path = Path(output_path)

        # Create DraftKings CSV format
        rows = []
        for lineup in lineups:
            row = {'contest_id': contest_id, 'contest_name': '', 'entry_fee': ''}

            # Add contest info if available
            if contest_id in self.contests:
                row['contest_name'] = self.contests[contest_id].get('contestName', '')
                row['entry_fee'] = self.contests[contest_id].get('entryFee', '')

            # Add players in DraftKings format
            positions = ['PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'UTIL']
            player_dict = {p['position']: p for p in lineup['players']}

            for i, pos in enumerate(positions, 1):
                if pos in player_dict:
                    player = player_dict[pos]
                    row[f'P{i}'] = f"{player['playerName']} ({player['playerID']})"
                else:
                    row[f'P{i}'] = ''

            rows.append(row)

        # Save to CSV
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)

        logger.info(f"Generated entry file for contest {contest_id}: {output_path}")

        return output_path

    def calculate_exposure(self) -> pd.DataFrame:
        """
        Calculate player exposure across all lineups.

        Returns
        -------
        pd.DataFrame
            Player exposure data
        """
        all_players = {}
        total_lineups = len(self.lineup_associations)

        if total_lineups == 0:
            return pd.DataFrame()

        for lineup_id, association in self.lineup_associations.items():
            lineup = association['lineup']
            for player in lineup['players']:
                player_id = player['playerID']
                if player_id not in all_players:
                    all_players[player_id] = {
                        'playerName': player['playerName'],
                        'team': player['team'],
                        'salary': player['salary'],
                        'count': 0,
                        'total_projection': 0
                    }

                all_players[player_id]['count'] += 1
                all_players[player_id]['total_projection'] += player['projected_fpts']

        # Calculate exposure percentages
        exposure_data = []
        for player_id, data in all_players.items():
            exposure_data.append({
                'playerID': player_id,
                'playerName': data['playerName'],
                'team': data['team'],
                'salary': data['salary'],
                'exposure': data['count'] / total_lineups * 100,
                'avg_projection': data['total_projection'] / data['count'],
                'lineup_count': data['count']
            })

        exposure_df = pd.DataFrame(exposure_data)
        exposure_df = exposure_df.sort_values('exposure', ascending=False)

        return exposure_df

    def save_session(self, session_name: str):
        """
        Save current session data to disk.

        Parameters
        ----------
        session_name : str
            Name for the session
        """
        session_path = self.storage_path / f"{session_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        session_data = {
            'session_name': session_name,
            'created_at': datetime.now().isoformat(),
            'contests': self.contests,
            'lineup_associations': self.lineup_associations
        }

        with open(session_path, 'w') as f:
            json.dump(session_data, f, indent=2)

        logger.info(f"Saved session to {session_path}")

    def load_session(self, session_path: str):
        """
        Load session data from disk.

        Parameters
        ----------
        session_path : str
            Path to session file
        """
        session_path = Path(session_path)

        if not session_path.exists():
            session_path = self.storage_path / session_path.name

        with open(session_path, 'r') as f:
            session_data = json.load(f)

        self.contests = session_data.get('contests', {})
        self.lineup_associations = session_data.get('lineup_associations', {})

        logger.info(f"Loaded session from {session_path}")