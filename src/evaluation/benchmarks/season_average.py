"""Season average benchmark for comparison with ML models"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class SeasonAverageBenchmark:
    """
    Simple benchmark that predicts player fantasy points based on season average.
    
    This provides a baseline for comparing ML model performance.
    """
    
    def __init__(self, min_games: int = 5):
        """
        Initialize benchmark.
        
        Args:
            min_games: Minimum games required to calculate average
        """
        self.min_games = min_games
        self.player_averages: Dict[str, float] = {}
        self.player_game_counts: Dict[str, int] = {}
    
    def fit(self, historical_data: pd.DataFrame) -> 'SeasonAverageBenchmark':
        """
        Calculate player season averages from historical data.
        
        Args:
            historical_data: DataFrame with 'playerID' and 'fpts' columns
            
        Returns:
            self for method chaining
        """
        if 'playerID' not in historical_data.columns or 'fpts' not in historical_data.columns:
            raise ValueError("historical_data must contain 'playerID' and 'fpts' columns")
        
        # Calculate averages per player
        player_stats = historical_data.groupby('playerID').agg({
            'fpts': ['mean', 'count']
        })
        
        # Flatten column names
        player_stats.columns = ['avg_fpts', 'game_count']
        player_stats = player_stats.reset_index()
        
        # Filter by minimum games
        qualified = player_stats[player_stats['game_count'] >= self.min_games]
        
        # Store averages
        self.player_averages = qualified.set_index('playerID')['avg_fpts'].to_dict()
        self.player_game_counts = qualified.set_index('playerID')['game_count'].to_dict()
        
        logger.info(f"Fitted benchmark for {len(self.player_averages)} players "
                   f"(min_games={self.min_games})")
        
        return self
    
    def predict(self, slate_data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate predictions for slate using season averages.
        
        Args:
            slate_data: DataFrame with 'playerID' column
            
        Returns:
            DataFrame with added 'benchmark_pred' column
        """
        result = slate_data.copy()
        result['benchmark_pred'] = result['playerID'].map(self.player_averages).fillna(0)
        
        coverage = (result['benchmark_pred'] > 0).sum()
        logger.info(f"Benchmark coverage: {coverage}/{len(result)} players "
                   f"({coverage/len(result)*100:.1f}%)")
        
        return result
    
    def compare_with_model(
        self,
        actual: pd.Series,
        model_pred: pd.Series,
        benchmark_pred: pd.Series
    ) -> Dict:
        """
        Compare model predictions with benchmark.
        
        Args:
            actual: Actual fantasy points
            model_pred: Model predictions
            benchmark_pred: Benchmark predictions
            
        Returns:
            Dictionary with comparison metrics
        """
        from src.evaluation.metrics.accuracy import MAPEMetric, RMSEMetric, MAEMetric
        
        mape_metric = MAPEMetric()
        rmse_metric = RMSEMetric()
        mae_metric = MAEMetric()
        
        # Calculate metrics for both
        model_mape = mape_metric.calculate(actual, model_pred)
        benchmark_mape = mape_metric.calculate(actual, benchmark_pred)
        
        model_rmse = rmse_metric.calculate(actual, model_pred)
        benchmark_rmse = rmse_metric.calculate(actual, benchmark_pred)
        
        model_mae = mae_metric.calculate(actual, model_pred)
        benchmark_mae = mae_metric.calculate(actual, benchmark_pred)
        
        # Calculate improvements
        mape_improvement = benchmark_mape - model_mape
        rmse_improvement = benchmark_rmse - model_rmse
        mae_improvement = benchmark_mae - model_mae
        
        summary = f"""
Benchmark Comparison
{'='*60}

Model Performance:
  MAPE: {model_mape:.2f}%
  RMSE: {model_rmse:.2f}
  MAE:  {model_mae:.2f}

Benchmark Performance:
  MAPE: {benchmark_mape:.2f}%
  RMSE: {benchmark_rmse:.2f}
  MAE:  {benchmark_mae:.2f}

Improvement (positive = model better):
  MAPE: {mape_improvement:+.2f}% 
  RMSE: {rmse_improvement:+.2f}
  MAE:  {mae_improvement:+.2f}
"""
        
        return {
            'model_mape': model_mape,
            'model_rmse': model_rmse,
            'model_mae': model_mae,
            'benchmark_mape': benchmark_mape,
            'benchmark_rmse': benchmark_rmse,
            'benchmark_mae': benchmark_mae,
            'mape_improvement': mape_improvement,
            'rmse_improvement': rmse_improvement,
            'mae_improvement': mae_improvement,
            'summary': summary
        }
    
    def compare_by_salary_tier(
        self,
        results: pd.DataFrame,
        salary_tiers: List[float]
    ) -> pd.DataFrame:
        """
        Compare model vs benchmark performance by salary tier.
        
        Args:
            results: DataFrame with 'salary', 'actual', 'model_pred', 'benchmark_pred'
            salary_tiers: List of salary boundaries for bins
            
        Returns:
            DataFrame with per-tier comparison metrics
        """
        from src.evaluation.metrics.accuracy import MAPEMetric, RMSEMetric, MAEMetric
        
        mape_metric = MAPEMetric()
        rmse_metric = RMSEMetric()
        mae_metric = MAEMetric()
        
        # Create salary tiers
        results = results.copy()
        results['salary_tier'] = pd.cut(
            results['salary'],
            bins=salary_tiers,
            labels=['Low', 'Mid', 'High', 'Elite'][:len(salary_tiers)-1]
        )
        
        tier_results = []
        
        for tier in results['salary_tier'].unique():
            if pd.isna(tier):
                continue
            
            tier_data = results[results['salary_tier'] == tier]
            
            if len(tier_data) == 0:
                continue
            
            # Calculate metrics for both
            model_mape = mape_metric.calculate(tier_data['actual'], tier_data['model_pred'])
            benchmark_mape = mape_metric.calculate(tier_data['actual'], tier_data['benchmark_pred'])
            
            model_rmse = rmse_metric.calculate(tier_data['actual'], tier_data['model_pred'])
            benchmark_rmse = rmse_metric.calculate(tier_data['actual'], tier_data['benchmark_pred'])
            
            model_mae = mae_metric.calculate(tier_data['actual'], tier_data['model_pred'])
            benchmark_mae = mae_metric.calculate(tier_data['actual'], tier_data['benchmark_pred'])
            
            tier_results.append({
                'salary_tier': tier,
                'count': len(tier_data),
                'model_mape': model_mape,
                'benchmark_mape': benchmark_mape,
                'mape_improvement': benchmark_mape - model_mape,
                'model_rmse': model_rmse,
                'benchmark_rmse': benchmark_rmse,
                'rmse_improvement': benchmark_rmse - model_rmse,
                'model_mae': model_mae,
                'benchmark_mae': benchmark_mae,
                'mae_improvement': benchmark_mae - model_mae
            })
        
        return pd.DataFrame(tier_results)
