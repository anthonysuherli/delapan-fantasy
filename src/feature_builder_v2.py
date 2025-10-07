import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)


class FeatureBuilder:

    @staticmethod
    def calculate_dk_fantasy_points(stats: pd.Series) -> float:
        pts = float(stats.get('pts', 0) or 0)
        reb = float(stats.get('reb', 0) or 0)
        ast = float(stats.get('ast', 0) or 0)
        stl = float(stats.get('stl', 0) or 0)
        blk = float(stats.get('blk', 0) or 0)
        tov = float(stats.get('TOV', 0) or 0)

        fpts = (
            pts * 1.0 +
            reb * 1.25 +
            ast * 1.5 +
            stl * 2.0 +
            blk * 2.0 -
            tov * 0.5
        )

        if pts >= 10 and reb >= 10:
            fpts += 1.5
        elif pts >= 10 and ast >= 10:
            fpts += 1.5
        elif reb >= 10 and ast >= 10:
            fpts += 1.5

        triple_double = sum([pts >= 10, reb >= 10, ast >= 10, stl >= 10, blk >= 10]) >= 3
        if triple_double:
            fpts += 1.5

        return round(fpts, 2)

    def build_training_features(
        self,
        training_data: pd.DataFrame,
        window_sizes: List[int] = [3, 5, 10]
    ) -> Tuple[pd.DataFrame, pd.Series]:
        logger.info("Building training features with strict temporal ordering")

        if training_data.empty:
            logger.warning("Empty training data provided")
            return pd.DataFrame(), pd.Series()

        df = training_data.copy()
        df['gameDate'] = pd.to_datetime(df['gameDate'], format='%Y%m%d', errors='coerce')
        df = df.sort_values(['playerID', 'gameDate']).reset_index(drop=True)

        if 'fpts' not in df.columns:
            logger.info("Calculating DK fantasy points for training data")
            df['fpts'] = df.apply(self.calculate_dk_fantasy_points, axis=1)

        numeric_cols = ['pts', 'reb', 'ast', 'stl', 'blk', 'TOV', 'mins', 'fpts', 'usage']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        logger.info("Calculating rolling features for each player")
        feature_rows = []
        players_processed = 0

        for player_id, player_df in df.groupby('playerID'):
            player_df = player_df.sort_values('gameDate').reset_index(drop=True)

            for idx in range(len(player_df)):
                if idx < min(window_sizes):
                    continue

                prior_games = player_df.iloc[:idx]

                features = self._calculate_features_from_prior_games(
                    prior_games,
                    player_id=player_id,
                    window_sizes=window_sizes
                )

                features['target'] = player_df.iloc[idx]['fpts']
                features['gameDate'] = player_df.iloc[idx]['gameDate']

                feature_rows.append(features)

            players_processed += 1
            if players_processed % 100 == 0:
                logger.debug(f"Processed {players_processed} players")

        if not feature_rows:
            logger.warning("No features generated from training data")
            return pd.DataFrame(), pd.Series()

        features_df = pd.DataFrame(feature_rows)
        features_df = features_df.dropna(subset=['target'])

        feature_cols = [col for col in features_df.columns
                       if col not in ['target', 'gameDate', 'playerID']]

        X = features_df[feature_cols].fillna(0)
        y = features_df['target']

        logger.info(f"Built training features: {len(X)} samples, {len(feature_cols)} features")

        return X, y

    def build_slate_features(
        self,
        slate_data: Dict[str, Any],
        training_data: pd.DataFrame,
        window_sizes: List[int] = [3, 5, 10]
    ) -> pd.DataFrame:
        logger.info("Building features for slate players")

        if slate_data['salaries'].empty:
            logger.warning("No salaries data for slate")
            return pd.DataFrame()

        salaries_df = slate_data['salaries'].copy()

        if 'playerID' not in salaries_df.columns:
            logger.error("playerID column missing from salaries data")
            return pd.DataFrame()

        if 'longName' in salaries_df.columns and 'playerName' not in salaries_df.columns:
            salaries_df['playerName'] = salaries_df['longName']

        training_data = training_data.copy()
        training_data['gameDate'] = pd.to_datetime(training_data['gameDate'], format='%Y%m%d', errors='coerce')

        numeric_cols = ['pts', 'reb', 'ast', 'stl', 'blk', 'TOV', 'mins', 'usage']
        for col in numeric_cols:
            if col in training_data.columns:
                training_data[col] = pd.to_numeric(training_data[col], errors='coerce')

        if 'fpts' not in training_data.columns:
            training_data['fpts'] = training_data.apply(self.calculate_dk_fantasy_points, axis=1)

        slate_features = []

        for _, player_row in salaries_df.iterrows():
            player_id = player_row['playerID']

            player_history = training_data[training_data['playerID'] == player_id].copy()
            player_history = player_history.sort_values('gameDate')

            if len(player_history) < min(window_sizes):
                logger.debug(f"Insufficient history for player {player_id} ({len(player_history)} games)")
                continue

            features = self._calculate_features_from_prior_games(
                player_history,
                player_id=player_id,
                window_sizes=window_sizes
            )

            features['playerID'] = player_id
            features['playerName'] = player_row.get('playerName', '')
            features['team'] = player_row.get('team', '')
            features['pos'] = player_row.get('pos', '')
            features['salary'] = player_row.get('salary', 0)

            slate_features.append(features)

        if not slate_features:
            logger.warning("No features generated for slate")
            return pd.DataFrame()

        features_df = pd.DataFrame(slate_features)

        logger.info(f"Built slate features for {len(features_df)} players")

        return features_df

    def _calculate_features_from_prior_games(
        self,
        prior_games: pd.DataFrame,
        player_id: str,
        window_sizes: List[int] = [3, 5, 10]
    ) -> Dict[str, float]:
        features = {'playerID': player_id}

        if prior_games.empty:
            return features

        stat_cols = ['pts', 'reb', 'ast', 'stl', 'blk', 'TOV', 'mins', 'fpts', 'usage']

        for col in stat_cols:
            if col not in prior_games.columns:
                continue

            values = pd.to_numeric(prior_games[col], errors='coerce').dropna()

            if len(values) == 0:
                continue

            for window in window_sizes:
                if len(values) >= window:
                    features[f'{col}_avg_{window}'] = values.tail(window).mean()
                    features[f'{col}_std_{window}'] = values.tail(window).std()
                    features[f'{col}_min_{window}'] = values.tail(window).min()
                    features[f'{col}_max_{window}'] = values.tail(window).max()
                else:
                    features[f'{col}_avg_{window}'] = values.mean()
                    features[f'{col}_std_{window}'] = values.std()
                    features[f'{col}_min_{window}'] = values.min()
                    features[f'{col}_max_{window}'] = values.max()

            features[f'{col}_ewma'] = values.ewm(span=5, adjust=False).mean().iloc[-1]

        features['games_played'] = len(prior_games)

        return features
