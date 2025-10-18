"""
Chart analysis and conclusions generator for backtest reports.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional


class ChartAnalyzer:
    """
    Generate insights and conclusions from backtest chart data.
    """

    def __init__(self, results: Dict[str, Any]):
        self.results = results

    def analyze_error_distribution(self) -> str:
        """Analyze error distribution and generate conclusions."""
        if 'all_predictions' not in self.results or self.results['all_predictions'].empty:
            return "No data available for error distribution analysis."

        all_preds = self.results['all_predictions']
        errors = all_preds['projected_fpts'] - all_preds['actual_fpts']
        abs_errors = np.abs(errors)

        mean_error = errors.mean()
        std_error = errors.std()
        skewness = errors.skew()
        overest_pct = (errors > 0).sum() / len(errors) * 100

        # Key statistics
        median_abs_error = abs_errors.median()
        q1_abs_error = abs_errors.quantile(0.25)
        q3_abs_error = abs_errors.quantile(0.75)

        conclusions = []

        # Bias analysis
        if abs(mean_error) < 1.0:
            conclusions.append(f"✓ Model shows minimal bias with mean error of {mean_error:+.2f} fpts, indicating balanced predictions.")
        elif mean_error > 0:
            conclusions.append(f"⚠ Model tends to overestimate by {mean_error:.2f} fpts on average ({overest_pct:.1f}% of predictions overestimated).")
        else:
            conclusions.append(f"⚠ Model tends to underestimate by {abs(mean_error):.2f} fpts on average ({100-overest_pct:.1f}% of predictions underestimated).")

        # Distribution symmetry
        if abs(skewness) < 0.5:
            conclusions.append(f"✓ Error distribution is approximately symmetric (skewness={skewness:.2f}), suggesting consistent model behavior across prediction ranges.")
        elif skewness > 0:
            conclusions.append(f"⚠ Error distribution is right-skewed (skewness={skewness:.2f}), indicating occasional large overestimates.")
        else:
            conclusions.append(f"⚠ Error distribution is left-skewed (skewness={skewness:.2f}), indicating occasional large underestimates.")

        # Error magnitude
        conclusions.append(f"Median absolute error is {median_abs_error:.2f} fpts (IQR: {q1_abs_error:.2f} to {q3_abs_error:.2f} fpts).")

        if median_abs_error < 5:
            conclusions.append("✓ Model demonstrates strong accuracy for typical predictions.")
        elif median_abs_error < 8:
            conclusions.append("~ Model shows moderate accuracy, with room for improvement.")
        else:
            conclusions.append("✗ Model has substantial prediction errors that should be addressed.")

        return '<br>'.join(conclusions)

    def analyze_salary_tier_performance(self) -> str:
        """Analyze performance across salary tiers."""
        if 'tier_comparison' not in self.results:
            return "No tier comparison data available."

        tier_df = self.results['tier_comparison']
        conclusions = []

        # Find best and worst performing tiers
        best_tier = tier_df.loc[tier_df['model_mape'].idxmin()]
        worst_tier = tier_df.loc[tier_df['model_mape'].idxmax()]

        conclusions.append(f"Best performance: **{best_tier['salary_tier']}** tier with {best_tier['model_mape']:.1f}% MAPE ({best_tier['count']} players).")
        conclusions.append(f"Worst performance: **{worst_tier['salary_tier']}** tier with {worst_tier['model_mape']:.1f}% MAPE ({worst_tier['count']} players).")

        # Elite tier analysis
        elite_tier = tier_df[tier_df['salary_tier'] == 'Elite']
        if not elite_tier.empty:
            elite_mape = elite_tier.iloc[0]['model_mape']
            elite_improvement = elite_tier.iloc[0]['mape_improvement']

            if elite_mape < 35:
                conclusions.append(f"✓ Elite players ($8k+) predicted with {elite_mape:.1f}% MAPE - strong performance on high-value targets.")
            else:
                conclusions.append(f"⚠ Elite players have {elite_mape:.1f}% MAPE - consider adding contextual features (usage, matchups).")

            if elite_improvement > 5:
                conclusions.append(f"✓ Model outperforms benchmark by {elite_improvement:+.1f}% on elite players.")
            elif elite_improvement < -5:
                conclusions.append(f"✗ Model underperforms benchmark by {abs(elite_improvement):.1f}% on elite players - priority improvement area.")

        # Low tier analysis
        low_tier = tier_df[tier_df['salary_tier'] == 'Low']
        if not low_tier.empty:
            low_mape = low_tier.iloc[0]['model_mape']
            if low_mape > 100:
                conclusions.append(f"⚠ Low-salary players have {low_mape:.1f}% MAPE - high volatility in low-minute, low-output scenarios.")

        return '<br>'.join(conclusions)

    def analyze_correlation(self) -> str:
        """Analyze prediction correlation."""
        corr = self.results.get('model_mean_correlation', 0)
        conclusions = []

        if corr > 0.8:
            conclusions.append(f"✓ Excellent correlation (r={corr:.3f}) - model captures the relationship between features and outcomes very well.")
        elif corr > 0.7:
            conclusions.append(f"✓ Strong correlation (r={corr:.3f}) - model meets the target threshold for predictive reliability.")
        elif corr > 0.6:
            conclusions.append(f"~ Moderate correlation (r={corr:.3f}) - model has predictive power but below target threshold.")
        else:
            conclusions.append(f"✗ Weak correlation (r={corr:.3f}) - model struggles to capture the underlying patterns. Feature engineering recommended.")

        # R² interpretation
        r_squared = corr ** 2
        conclusions.append(f"Model explains {r_squared*100:.1f}% of the variance in actual fantasy points.")

        if corr < 0.7:
            conclusions.append("**Recommendations:** Add contextual features (opponent strength, pace, rest days), injury impact factors, or recent form indicators.")

        return '<br>'.join(conclusions)

    def analyze_calibration(self) -> str:
        """Analyze model calibration quality."""
        if 'all_predictions' not in self.results or self.results['all_predictions'].empty:
            return "No calibration data available."

        all_preds = self.results['all_predictions']
        conclusions = []

        # Calculate calibration across prediction ranges
        all_preds_sorted = all_preds.sort_values('projected_fpts')
        n_bins = 5
        bin_size = len(all_preds_sorted) // n_bins

        calibration_errors = []
        for i in range(n_bins):
            start_idx = i * bin_size
            end_idx = start_idx + bin_size if i < n_bins - 1 else len(all_preds_sorted)
            bin_data = all_preds_sorted.iloc[start_idx:end_idx]

            pred_mean = bin_data['projected_fpts'].mean()
            actual_mean = bin_data['actual_fpts'].mean()
            calib_error = abs(pred_mean - actual_mean) / actual_mean * 100
            calibration_errors.append(calib_error)

        mean_calib_error = np.mean(calibration_errors)
        max_calib_error = np.max(calibration_errors)

        if mean_calib_error < 5:
            conclusions.append(f"✓ Excellent calibration (mean error: {mean_calib_error:.1f}%) - predictions are well-calibrated across all ranges.")
        elif mean_calib_error < 10:
            conclusions.append(f"✓ Good calibration (mean error: {mean_calib_error:.1f}%) - minor systematic biases across ranges.")
        elif mean_calib_error < 15:
            conclusions.append(f"~ Moderate calibration issues (mean error: {mean_calib_error:.1f}%) - some ranges show systematic bias.")
        else:
            conclusions.append(f"✗ Poor calibration (mean error: {mean_calib_error:.1f}%) - significant systematic biases. Consider calibration techniques.")

        if max_calib_error > 20:
            conclusions.append(f"⚠ Maximum calibration error of {max_calib_error:.1f}% in one range - investigate specific prediction bins for bias.")

        conclusions.append("**Interpretation:** A well-calibrated model means predicted values match actual values across the full range (low, medium, high scorers).")

        return '<br>'.join(conclusions)

    def analyze_residuals(self) -> str:
        """Analyze residual patterns."""
        if 'all_predictions' not in self.results or self.results['all_predictions'].empty:
            return "No residual data available."

        all_preds = self.results['all_predictions']
        residuals = all_preds['actual_fpts'] - all_preds['projected_fpts']
        conclusions = []

        # Check for heteroscedasticity (variance changes with predicted value)
        low_pred = all_preds[all_preds['projected_fpts'] < all_preds['projected_fpts'].quantile(0.33)]
        high_pred = all_preds[all_preds['projected_fpts'] > all_preds['projected_fpts'].quantile(0.67)]

        low_var = (low_pred['actual_fpts'] - low_pred['projected_fpts']).var()
        high_var = (high_pred['actual_fpts'] - high_pred['projected_fpts']).var()
        var_ratio = high_var / low_var if low_var > 0 else 1.0

        if 0.5 < var_ratio < 2.0:
            conclusions.append("✓ Residuals show consistent variance across prediction ranges (homoscedasticity).")
        elif var_ratio >= 2.0:
            conclusions.append(f"⚠ Residuals show increasing variance for high predictions ({var_ratio:.1f}x higher) - model less confident on high scorers.")
        else:
            conclusions.append(f"⚠ Residuals show decreasing variance for high predictions - potential overfitting on high scorers.")

        # Normality test
        from scipy import stats
        _, p_value = stats.normaltest(residuals.dropna())

        if p_value > 0.05:
            conclusions.append("✓ Residuals approximately normally distributed - assumption for linear models satisfied.")
        else:
            conclusions.append("⚠ Residuals deviate from normal distribution - consider robust modeling approaches or outlier treatment.")

        conclusions.append("**Interpretation:** Ideal residuals are randomly scattered around zero with constant variance, indicating the model has captured all systematic patterns.")

        return '<br>'.join(conclusions)

    def analyze_position_performance(self) -> str:
        """Analyze performance by position."""
        if 'all_predictions' not in self.results or self.results['all_predictions'].empty:
            return "No position data available."

        all_preds = self.results['all_predictions']
        if 'position' not in all_preds.columns:
            return "Position data not available."

        position_stats = all_preds.groupby('position').agg({
            'projected_fpts': lambda x: np.abs(x - all_preds.loc[x.index, 'actual_fpts']).mean(),
            'playerID': 'count'
        }).reset_index()
        position_stats.columns = ['position', 'mae', 'count']
        position_stats = position_stats.sort_values('mae')

        conclusions = []
        best_pos = position_stats.iloc[0]
        worst_pos = position_stats.iloc[-1]

        conclusions.append(f"Most accurate: **{best_pos['position']}** (MAE: {best_pos['mae']:.2f} fpts, n={int(best_pos['count'])}).")
        conclusions.append(f"Least accurate: **{worst_pos['position']}** (MAE: {worst_pos['mae']:.2f} fpts, n={int(worst_pos['count'])}).")

        mae_range = worst_pos['mae'] - best_pos['mae']
        if mae_range > 5:
            conclusions.append(f"⚠ Large variation ({mae_range:.1f} fpts) in accuracy across positions - consider position-specific features or models.")
        else:
            conclusions.append(f"✓ Consistent accuracy across positions (range: {mae_range:.1f} fpts) - position-agnostic model performs well.")

        return '<br>'.join(conclusions)

    def analyze_minutes_impact(self) -> str:
        """Analyze how minutes played affects prediction accuracy."""
        if 'all_predictions' not in self.results or self.results['all_predictions'].empty:
            return "No minutes data available."

        all_preds = self.results['all_predictions']
        if 'actual_mins' not in all_preds.columns:
            return "Minutes data not available."

        # Segment by minutes
        low_min = all_preds[all_preds['actual_mins'] < 20]
        high_min = all_preds[all_preds['actual_mins'] >= 30]

        if len(low_min) == 0 or len(high_min) == 0:
            return "Insufficient data for minutes analysis."

        low_mae = np.abs(low_min['projected_fpts'] - low_min['actual_fpts']).mean()
        high_mae = np.abs(high_min['projected_fpts'] - high_min['actual_fpts']).mean()

        conclusions = []
        conclusions.append(f"Low minutes (<20): MAE = {low_mae:.2f} fpts (n={len(low_min)}).")
        conclusions.append(f"High minutes (≥30): MAE = {high_mae:.2f} fpts (n={len(high_min)}).")

        if low_mae > high_mae * 1.5:
            conclusions.append(f"⚠ Low-minute players are significantly harder to predict ({low_mae/high_mae:.1f}x higher error) - consider filtering or separate modeling.")
        elif high_mae > low_mae * 1.5:
            conclusions.append(f"⚠ High-minute players show higher errors - unexpected. Check for outliers or missing contextual features.")
        else:
            conclusions.append("✓ Consistent accuracy across minute ranges.")

        conclusions.append("**Recommendation:** For DFS optimization, consider minimum minutes thresholds to filter unpredictable low-usage players.")

        return '<br>'.join(conclusions)

    def analyze_model_vs_benchmark(self) -> str:
        """Analyze model improvement over benchmark."""
        improvement = self.results.get('mape_improvement', 0)
        model_mape = self.results.get('model_mean_mape', 0)
        bench_mape = self.results.get('benchmark_mean_mape', 0)

        conclusions = []

        if improvement > 10:
            conclusions.append(f"✓ **Strong outperformance:** Model beats benchmark by {improvement:.1f}% MAPE - substantial value added.")
        elif improvement > 5:
            conclusions.append(f"✓ **Moderate outperformance:** Model improves on benchmark by {improvement:.1f}% MAPE.")
        elif improvement > 0:
            conclusions.append(f"~ **Slight outperformance:** Model marginally better ({improvement:.1f}% improvement) - further optimization recommended.")
        elif improvement > -5:
            conclusions.append(f"⚠ **Slight underperformance:** Model performs {abs(improvement):.1f}% worse than benchmark - investigate feature engineering.")
        else:
            conclusions.append(f"✗ **Significant underperformance:** Model {abs(improvement):.1f}% worse than simple season averages - model architecture review needed.")

        # Statistical significance
        if 'statistical_test' in self.results:
            p_value = self.results['statistical_test']['p_value']
            if p_value < 0.05:
                conclusions.append(f"✓ Difference is **statistically significant** (p={p_value:.4f}) - results are reliable, not due to chance.")
            else:
                conclusions.append(f"⚠ Difference is **not statistically significant** (p={p_value:.4f}) - results could be due to random variation.")

        return '<br>'.join(conclusions)