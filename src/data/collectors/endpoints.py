from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class EndpointConfig:
    name: str
    url: str
    required_params: List[str]
    optional_params: List[str]
    description: str


TANK01_ENDPOINTS = {
    'nba_betting_odds': EndpointConfig(
        name='nba_betting_odds',
        url='https://tank01-fantasy-stats.p.rapidapi.com/getNBABettingOdds',
        required_params=[],
        optional_params=['gameDate', 'gameID', 'itemFormat'],
        description='Get betting odds. Requires gameDate (YYYYMMDD) OR gameID. Optional: itemFormat (list/map, defaults to map).'
    ),

    'game_box_score': EndpointConfig(
        name='game_box_score',
        url='https://tank01-fantasy-stats.p.rapidapi.com/getNBABoxScore',
        required_params=['gameID'],
        optional_params=['fantasyPoints', 'pts', 'stl', 'blk', 'reb', 'ast', 'TOV', 'mins', 'doubleDouble', 'tripleDouble', 'quadDouble'],
        description='Get detailed box score for a specific game. Format: YYYYMMDD_AWAY@HOME. Optional fantasy scoring params.'
    ),

    'teams': EndpointConfig(
        name='teams',
        url='https://tank01-fantasy-stats.p.rapidapi.com/getNBATeams',
        required_params=[],
        optional_params=['schedules', 'rosters', 'statsToGet', 'topPerformers', 'teamStats'],
        description='Get all NBA teams with IDs and abbreviations. Optional: schedules, rosters, statsToGet (averages/totals), topPerformers, teamStats.'
        ),

    'general_game_info': EndpointConfig(
        name='general_game_info',
        url='https://tank01-fantasy-stats.p.rapidapi.com/getNBAGameInfo',
        required_params=['gameID'],
        optional_params=[],
        description='Get general information for a specific game. Format: YYYYMMDD_AWAY@HOME'
    ),

    'daily_schedule': EndpointConfig(
        name='daily_schedule',
        url='https://tank01-fantasy-stats.p.rapidapi.com/getNBAGamesForDate',
        required_params=['gameDate'],
        optional_params=[],
        description='Get NBA schedule for a specific date (YYYYMMDD).'
    ),

    'team_roster': EndpointConfig(
        name='team_roster',
        url='https://tank01-fantasy-stats.p.rapidapi.com/getNBATeamRoster',
        required_params=[],
        optional_params=['teamAbv', 'teamID', 'archiveDate', 'statsToGet'],
        description='Get team roster. Requires teamAbv OR teamID. Optional: archiveDate (YYYYMMDD), statsToGet (averages/totals, not with archiveDate).'
    ),

    'injury_list_history': EndpointConfig(
        name='injury_list_history',
        url='https://tank01-fantasy-stats.p.rapidapi.com/getNBAInjuryList',
        required_params=[],
        optional_params=['teamAbv', 'playerID', 'injDate', 'beginningInjDate', 'endInjDate', 'year', 'numberOfDays'],
        description='Get injury list. Optional: playerID, injDate (YYYYMMDD), beginningInjDate (YYYYMMDD), endInjDate (YYYYMMDD), year (YYYY), numberOfDays (1-30).'
    ),

    'player_info': EndpointConfig(
        name='player_info',
        url='https://tank01-fantasy-stats.p.rapidapi.com/getNBAPlayerInfo',
        required_params=[],
        optional_params=['playerID', 'playerName', 'statsToGet'],
        description='Get player information. Requires playerID OR playerName. Optional: statsToGet (averages/totals, current season only).'
    ),

    'player_list': EndpointConfig(
        name='player_list',
        url='https://tank01-fantasy-stats.p.rapidapi.com/getNBAPlayerList',
        required_params=[],
        optional_params=[],
        description='Get list of all NBA players.'
    ),

    'team_schedule': EndpointConfig(
        name='team_schedule',
        url='https://tank01-fantasy-stats.p.rapidapi.com/getNBATeamSchedule',
        required_params=[],
        optional_params=['teamAbv', 'teamID', 'season'],
        description='Get team schedule. Requires teamAbv OR teamID. Optional: season (YYYY, 2022+, defaults to current).'
    ),

    'dfs_salaries': EndpointConfig(
        name='dfs_salaries',
        url='https://tank01-fantasy-stats.p.rapidapi.com/getNBADFS',
        required_params=['gameDate'],
        optional_params=['lineupType'],
        description='Get DFS salaries. Required: date (YYYYMMDD). Optional: lineupType (DraftKings/FanDuel/Yahoo).'
    ),

    'adp': EndpointConfig(
        name='adp',
        url='https://tank01-fantasy-stats.p.rapidapi.com/getNBAADP',
        required_params=[],
        optional_params=['adpDate', 'combineGuards', 'combineForwards', 'combineFC'],
        description='Get Average Draft Position. Optional: adpDate (YYYYMMDD, since 20240924), combineGuards, combineForwards, combineFC.'
    ),

    'depth_charts': EndpointConfig(
        name='depth_charts',
        url='https://tank01-fantasy-stats.p.rapidapi.com/getNBADepthCharts',
        required_params=[],
        optional_params=['teamAbv'],
        description='Get depth charts. Optional: teamAbv to filter by team.'
    ),

    'top_news_headlines': EndpointConfig(
        name='top_news_headlines',
        url='https://tank01-fantasy-stats.p.rapidapi.com/getNBANews',
        required_params=[],
        optional_params=['playerID', 'teamID', 'teamAbv', 'topNews', 'recentNews', 'fantasyNews', 'maxItems'],
        description='Get NBA news. Optional: playerID, teamID, teamAbv, topNews, recentNews, fantasyNews, maxItems.'
    ),

    'fantasy_point_projections': EndpointConfig(
        name='fantasy_point_projections',
        url='https://tank01-fantasy-stats.p.rapidapi.com/getNBAProjections',
        required_params=['numOfDays'],
        optional_params=['pts', 'reb', 'TOV', 'stl', 'blk', 'ast', 'mins', 'doubleDouble', 'tripleDouble', 'quadDouble'],
        description='Get fantasy point projections. Required: numOfDays (7 or 14). Optional: pts, reb, TOV, stl, blk, ast, mins, doubleDouble, tripleDouble, quadDouble.'
    ),

    'player_game_logs': EndpointConfig(
        name='player_game_logs',
        url='https://tank01-fantasy-stats.p.rapidapi.com/getNBAGamesForPlayer',
        required_params=['playerID'],
        optional_params=['gameID', 'season', 'numberOfGames', 'fantasyPoints', 'pts', 'reb', 'stl', 'blk', 'ast', 'TOV', 'mins', 'doubleDouble', 'tripleDouble', 'quadDouble'],
        description='Get game logs for specific player. Required: playerID. Optional: gameID, season (YYYY), numberOfGames, fantasy scoring params.'
    ),

    'scores_only': EndpointConfig(
        name='scores_only',
        url='https://tank01-fantasy-stats.p.rapidapi.com/getNBAScoresOnly',
        required_params=[],
        optional_params=['gameDate', 'gameID', 'topPerformers', 'lineups'],
        description='Get game scores. Requires gameDate (YYYYMMDD) OR gameID. Optional: topPerformers, lineups (current date only).'
    ),
}


def get_endpoint_config(endpoint_name: str) -> Optional[EndpointConfig]:
    return TANK01_ENDPOINTS.get(endpoint_name)


def list_endpoints() -> List[str]:
    return list(TANK01_ENDPOINTS.keys())


def validate_endpoint_params(endpoint_name: str, params: Dict) -> tuple[bool, Optional[str]]:
    config = get_endpoint_config(endpoint_name)
    if not config:
        return False, f"Unknown endpoint: {endpoint_name}"

    if config.required_params:
        missing = [p for p in config.required_params if p not in params]
        if missing:
            return False, f"Missing required parameters: {missing}"

    return True, None
