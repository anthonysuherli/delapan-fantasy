import pandas as pd


def calculate_dk_fantasy_points(stats: pd.Series) -> float:
    """
    Calculate DraftKings fantasy points from box score stats.

    Scoring:
    - Points: 1.0
    - Rebounds: 1.25
    - Assists: 1.5
    - Steals: 2.0
    - Blocks: 2.0
    - Turnovers: -0.5
    - Double-double: +1.5
    - Triple-double: +3.0 (includes double-double bonus)

    Args:
        stats: Series containing player stats (pts, reb, ast, stl, blk, TOV)

    Returns:
        Fantasy points as float
    """
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

    double_double = sum([pts >= 10, reb >= 10, ast >= 10, stl >= 10, blk >= 10]) >= 2
    if double_double:
        fpts += 1.5

    triple_double = sum([pts >= 10, reb >= 10, ast >= 10, stl >= 10, blk >= 10]) >= 3
    if triple_double:
        fpts += 1.5

    return round(fpts, 2)
