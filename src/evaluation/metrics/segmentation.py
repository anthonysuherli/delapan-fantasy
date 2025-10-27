"""
Segmented performance metrics for salary tiers and positions.

This module provides functions to analyze prediction performance broken down
by salary tiers, player positions, and cross-combinations.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional


def analyze_by_salary(
    results_df: pd.DataFrame,
    bins: List[float] = None,
    labels: List[str] = None
) -> pd.DataFrame:
    """
    Analyze prediction performance by salary tier.

    Parameters
    ----------
    results_df : pd.DataFrame
        Results with columns: playerID, salary, actual, predicted, abs_error, pct_error
    bins : List[float], optional
        Salary bin edges. Default: [0, 5000, 7000, 9000, 15000]
    labels : List[str], optional
        Tier labels. Default: ['$3-5k', '$5-7k', '$7-9k', '$9k+']

    Returns
    -------
    pd.DataFrame
        Metrics grouped by salary tier with columns:
        Count, Avg Actual, Avg Predicted, MAE, MAPE, Avg Salary
    """
    if bins is None:
        bins = [0, 5000, 7000, 9000, 15000]
    if labels is None:
        labels = ['$3-5k', '$5-7k', '$7-9k', '$9k+']

    df = results_df.copy()
    df['salary_tier'] = pd.cut(df['salary'], bins=bins, labels=labels)

    tier_metrics = df.groupby('salary_tier').agg({
        'playerID': 'count',
        'actual': 'mean',
        'predicted': 'mean',
        'abs_error': 'mean',
        'pct_error': 'mean',
        'salary': 'mean'
    }).round(2)

    tier_metrics.columns = ['Count', 'Avg Actual', 'Avg Predicted', 'MAE', 'MAPE', 'Avg Salary']

    return tier_metrics


def analyze_by_position(
    results_df: pd.DataFrame,
    position_col: str = 'primary_position'
) -> pd.DataFrame:
    """
    Analyze prediction performance by player position.

    Parameters
    ----------
    results_df : pd.DataFrame
        Results with columns: playerID, primary_position, actual, predicted, abs_error, pct_error, salary
    position_col : str
        Name of the position column

    Returns
    -------
    pd.DataFrame
        Metrics grouped by position with columns:
        Count, Avg Actual, Avg Predicted, MAE, MAPE, Avg Salary
    """
    position_metrics = results_df.groupby(position_col).agg({
        'playerID': 'count',
        'actual': 'mean',
        'predicted': 'mean',
        'abs_error': 'mean',
        'pct_error': 'mean',
        'salary': 'mean'
    }).round(2)

    position_metrics.columns = ['Count', 'Avg Actual', 'Avg Predicted', 'MAE', 'MAPE', 'Avg Salary']
    position_metrics = position_metrics.sort_values('Count', ascending=False)

    return position_metrics


def cross_analysis(
    results_df: pd.DataFrame,
    tier_col: str = 'salary_tier',
    position_col: str = 'primary_position'
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create cross-tabulation of MAPE and counts by salary tier × position.

    Parameters
    ----------
    results_df : pd.DataFrame
        Results with salary_tier and position columns
    tier_col : str
        Name of the salary tier column
    position_col : str
        Name of the position column

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (cross_tab_mape, cross_tab_count)
        - cross_tab_mape: Mean MAPE for each tier × position combination
        - cross_tab_count: Player count for each tier × position combination
    """
    cross_tab_mape = pd.crosstab(
        results_df[tier_col],
        results_df[position_col],
        values=results_df['pct_error'],
        aggfunc='mean'
    ).round(1)

    cross_tab_count = pd.crosstab(
        results_df[tier_col],
        results_df[position_col],
        values=results_df['pct_error'],
        aggfunc='count'
    ).fillna(0).astype(int)

    return cross_tab_mape, cross_tab_count


def get_best_segments(
    results_df: pd.DataFrame,
    metric: str = 'pct_error',
    min_count: int = 3
) -> Dict[str, pd.DataFrame]:
    """
    Find best-performing salary × position combinations.

    Parameters
    ----------
    results_df : pd.DataFrame
        Results with salary_tier and primary_position columns
    metric : str
        Metric to minimize (default: pct_error for MAPE)
    min_count : int
        Minimum player count to include segment

    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary with 'best' and 'worst' DataFrames containing
        top/bottom segments with tier, pos, metric, count columns
    """
    if 'salary_tier' not in results_df.columns:
        bins = [0, 5000, 7000, 9000, 15000]
        labels = ['$3-5k', '$5-7k', '$7-9k', '$9k+']
        results_df = results_df.copy()
        results_df['salary_tier'] = pd.cut(results_df['salary'], bins=bins, labels=labels)

    cross_tab_mape, cross_tab_count = cross_analysis(results_df)

    # Flatten cross-tabulations
    flat_data = []
    for tier in cross_tab_mape.index:
        for pos in cross_tab_mape.columns:
            if not pd.isna(cross_tab_mape.loc[tier, pos]):
                count = cross_tab_count.loc[tier, pos]
                if count >= min_count:
                    flat_data.append({
                        'tier': tier,
                        'pos': pos,
                        'mape': cross_tab_mape.loc[tier, pos],
                        'count': count
                    })

    flat_df = pd.DataFrame(flat_data)

    if len(flat_df) == 0:
        return {'best': pd.DataFrame(), 'worst': pd.DataFrame()}

    return {
        'best': flat_df.nsmallest(5, 'mape'),
        'worst': flat_df.nlargest(5, 'mape')
    }


def summary_report(results_df: pd.DataFrame) -> str:
    """
    Generate text summary of segmented performance.

    Parameters
    ----------
    results_df : pd.DataFrame
        Results with salary, position, and error columns

    Returns
    -------
    str
        Formatted text report
    """
    tier_metrics = analyze_by_salary(results_df)
    position_metrics = analyze_by_position(results_df)
    segments = get_best_segments(results_df)

    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("SEGMENTED PERFORMANCE ANALYSIS")
    report_lines.append("=" * 60)

    # Salary tier summary
    report_lines.append("\nSALARY TIER PERFORMANCE:")
    for tier in tier_metrics.index:
        mape = tier_metrics.loc[tier, 'MAPE']
        count = int(tier_metrics.loc[tier, 'Count'])
        report_lines.append(f"  {tier}: {mape:.1f}% MAPE ({count} players)")

    # Position summary
    report_lines.append("\nPOSITION PERFORMANCE (Top 5):")
    for pos in position_metrics.index[:5]:
        mape = position_metrics.loc[pos, 'MAPE']
        count = int(position_metrics.loc[pos, 'Count'])
        report_lines.append(f"  {pos}: {mape:.1f}% MAPE ({count} players)")

    # Best segments
    if len(segments['best']) > 0:
        report_lines.append("\nBEST SALARY × POSITION SEGMENTS:")
        for _, row in segments['best'].head(3).iterrows():
            report_lines.append(
                f"  {row['tier']} {row['pos']}: {row['mape']:.1f}% MAPE (n={row['count']})"
            )

    report_lines.append("=" * 60)

    return "\n".join(report_lines)
