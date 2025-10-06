import time
import requests
from typing import Dict, Any, Optional
from functools import lru_cache
from datetime import datetime, timedelta

from .endpoints import get_endpoint_config, validate_endpoint_params


class Tank01Client:

    def __init__(
        self,
        api_key: str,
        base_url: str = 'https://tank01-fantasy-stats.p.rapidapi.com',
        cache_ttl: int = 60,
        rate_limit: int = 500
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.cache_ttl = cache_ttl
        self.rate_limit = rate_limit

        self.request_count = 0
        self.request_reset_time = datetime.now() + timedelta(days=30)

        self.headers = {
            'X-RapidAPI-Key': self.api_key,
            'X-RapidAPI-Host': 'tank01-fantasy-stats.p.rapidapi.com'
        }

    def _make_request(
        self,
        endpoint_name: str,
        params: Optional[Dict[str, Any]] = None,
        max_retries: int = 3
    ) -> Dict[str, Any]:
        config = get_endpoint_config(endpoint_name)

        url = config.url
        retry_count = 0
        backoff_time = 1

        while retry_count < max_retries:
            try:
                response = requests.get(
                    url,
                    headers=self.headers,
                    params=params,
                    timeout=30
                )

                self.request_count += 1

                if response.status_code == 429:
                    retry_count += 1
                    if retry_count >= max_retries:
                        raise Exception("Rate limit exceeded after retries")

                    time.sleep(backoff_time)
                    backoff_time *= 2
                    continue

                response.raise_for_status()

                data = response.json()

                return data

            except requests.exceptions.Timeout:
                retry_count += 1
                if retry_count >= max_retries:
                    raise Exception(f"Request timeout after {max_retries} retries")
                time.sleep(backoff_time)
                backoff_time *= 2

            except requests.exceptions.RequestException as e:
                raise Exception(f"Request failed: {str(e)}")

        raise Exception("Max retries exceeded")

    def _build_params(self, param_map: Dict[str, str], **kwargs) -> Dict[str, Any]:
        return {param_map[k]: v for k, v in kwargs.items() if k in param_map and v is not None}

    def get_daily_schedule(self, game_date: str) -> Dict[str, Any]:
        """
        Get NBA schedule for a specific date (YYYYMMDD).
        :param game_date: Date in YYYYMMDD format.
        :return: API response as dict.
        """
        params = {'gameDate': game_date}
        return self._make_request('daily_schedule', params)

    def get_betting_odds(self, **kwargs) -> Dict[str, Any]:
        param_map = {'game_date': 'gameDate', 'game_id': 'gameID', 'item_format': 'itemFormat'}
        return self._make_request('nba_betting_odds', self._build_params(param_map, **kwargs))

    def get_box_score(self, game_id: str, **kwargs) -> Dict[str, Any]:
        param_map = {
            'fantasy_points': 'fantasyPoints', 'pts': 'pts', 'stl': 'stl', 'blk': 'blk',
            'reb': 'reb', 'ast': 'ast', 'tov': 'TOV', 'mins': 'mins',
            'double_double': 'doubleDouble', 'triple_double': 'tripleDouble', 'quad_double': 'quadDouble'
        }
        params = {'gameID': game_id, **self._build_params(param_map, **kwargs)}
        return self._make_request('game_box_score', params)

    def get_teams(self, **kwargs) -> Dict[str, Any]:
        param_map = {
            'schedules': 'schedules', 'rosters': 'rosters', 'stats_to_get': 'statsToGet',
            'top_performers': 'topPerformers', 'team_stats': 'teamStats'
        }
        return self._make_request('teams', self._build_params(param_map, **kwargs))

    def get_team_roster(self, **kwargs) -> Dict[str, Any]:
        param_map = {
            'team_abv': 'teamAbv', 'team_id': 'teamID',
            'archive_date': 'archiveDate', 'stats_to_get': 'statsToGet'
        }
        return self._make_request('team_roster', self._build_params(param_map, **kwargs))

    def get_injuries(self, **kwargs) -> Dict[str, Any]:
        param_map = {'team_abv': 'teamAbv'}
        return self._make_request('injury_list_history', self._build_params(param_map, **kwargs))

    def get_player_info(self, **kwargs) -> Dict[str, Any]:
        param_map = {
            'player_id': 'playerID', 'player_name': 'playerName', 'stats_to_get': 'statsToGet'
        }
        return self._make_request('player_info', self._build_params(param_map, **kwargs))

    def get_player_list(self) -> Dict[str, Any]:
        return self._make_request('player_list')

    def get_team_schedule(self, **kwargs) -> Dict[str, Any]:
        param_map = {'team_abv': 'teamAbv', 'team_id': 'teamID', 'season': 'season'}
        return self._make_request('team_schedule', self._build_params(param_map, **kwargs))

    def get_dfs_salaries(self, date: str, **kwargs) -> Dict[str, Any]:
        param_map = {'lineup_type': 'lineupType'}
        params = {'date': date, **self._build_params(param_map, **kwargs)}
        return self._make_request('dfs_salaries', params)

    def get_adp(self, **kwargs) -> Dict[str, Any]:
        param_map = {
            'adp_date': 'adpDate', 'combine_guards': 'combineGuards',
            'combine_forwards': 'combineForwards', 'combine_fc': 'combineFC'
        }
        return self._make_request('adp', self._build_params(param_map, **kwargs))

    def get_depth_charts(self, **kwargs) -> Dict[str, Any]:
        param_map = {'team_abv': 'teamAbv'}
        return self._make_request('depth_charts', self._build_params(param_map, **kwargs))

    def get_news(self, **kwargs) -> Dict[str, Any]:
        param_map = {
            'player_id': 'playerID', 'team_id': 'teamID', 'team_abv': 'teamAbv',
            'top_news': 'topNews', 'recent_news': 'recentNews',
            'fantasy_news': 'fantasyNews', 'max_items': 'maxItems'
        }
        return self._make_request('top_news_headlines', self._build_params(param_map, **kwargs))

    def get_projections(self, num_of_days: int, **kwargs) -> Dict[str, Any]:
        param_map = {
            'pts': 'pts', 'reb': 'reb', 'tov': 'TOV', 'stl': 'stl', 'blk': 'blk',
            'ast': 'ast', 'mins': 'mins', 'double_double': 'doubleDouble',
            'triple_double': 'tripleDouble', 'quad_double': 'quadDouble'
        }
        params = {'numOfDays': num_of_days, **self._build_params(param_map, **kwargs)}
        return self._make_request('fantasy_point_projections', params)

    def get_player_game_logs(self, player_id: str, **kwargs) -> Dict[str, Any]:
        param_map = {
            'game_id': 'gameID', 'season': 'season', 'number_of_games': 'numberOfGames',
            'fantasy_points': 'fantasyPoints', 'pts': 'pts', 'reb': 'reb',
            'stl': 'stl', 'blk': 'blk', 'ast': 'ast', 'tov': 'TOV', 'mins': 'mins',
            'double_double': 'doubleDouble', 'triple_double': 'tripleDouble', 'quad_double': 'quadDouble'
        }
        params = {'playerID': player_id, **self._build_params(param_map, **kwargs)}
        return self._make_request('player_game_logs', params)

    def get_scores_only(self, **kwargs) -> Dict[str, Any]:
        param_map = {
            'game_date': 'gameDate', 'game_id': 'gameID',
            'top_performers': 'topPerformers', 'lineups': 'lineups'
        }
        return self._make_request('scores_only', self._build_params(param_map, **kwargs))

    def get_request_count(self) -> int:
        return self.request_count

    def get_remaining_requests(self) -> int:
        return max(0, self.rate_limit - self.request_count)
