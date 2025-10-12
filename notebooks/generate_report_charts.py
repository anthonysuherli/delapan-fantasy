import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from src.data.storage.sqlite_storage import SQLiteStorage
from src.data.loaders.historical_loader import HistoricalDataLoader
from src.models.xgboost_model import XGBoostModel
from src.features.pipeline import FeaturePipeline
from src.features.transformers.rolling_stats import RollingStatsTransformer
from src.features.transformers.ewma import EWMATransformer
from src.utils.fantasy_points import calculate_dk_fantasy_points
from src.evaluation.metrics.accuracy import MAPEMetric, RMSEMetric, CorrelationMetric
from src.utils.plot_styling import (
    apply_modern_style,
    MODERN_COLORS,
    PALETTE_GRADIENT,
    PALETTE_COOL,
    add_correlation_annotation,
    add_value_labels,
    create_gradient_cmap
)

apply_modern_style()

DB_PATH = repo_root / 'nba_dfs.db'
TARGET_DATE = '20250205'
ROLLING_WINDOWS = [3, 5, 10]
EWMA_SPAN = 5
MIN_GAMES = 10
OUTPUT_DIR = repo_root / 'notebooks' / 'report_charts'
OUTPUT_DIR.mkdir(exist_ok=True)

print(f"Loading data from {DB_PATH}...")
storage = SQLiteStorage(str(DB_PATH))
loader = HistoricalDataLoader(storage)

TRAIN_START = HistoricalDataLoader.get_season_start_date(TARGET_DATE)
target_dt = datetime.strptime(TARGET_DATE, '%Y%m%d')
train_end_dt = target_dt - timedelta(days=1)
TRAIN_END = train_end_dt.strftime('%Y%m%d')

print("Loading training data...")
training_data = loader.load_historical_player_logs(start_date=TRAIN_START, end_date=TRAIN_END)

if 'plusMinus' in training_data.columns:
    training_data['plusMinus'] = training_data['plusMinus'].apply(lambda x: int(x) if pd.notna(x) else 0)

print(f"Loaded {len(training_data)} training samples")

print("Loading target slate...")
slate_data = loader.load_slate_data(TARGET_DATE)
salaries_df = slate_data.get('dfs_salaries', pd.DataFrame())

if not salaries_df.empty and 'platform' in salaries_df.columns:
    salaries_df = salaries_df[salaries_df['platform'].str.lower() == 'draftkings']

print("Building features...")
df = training_data.copy()
df['gameDate'] = pd.to_datetime(df['gameDate'], format='%Y%m%d', errors='coerce')
df = df.sort_values(['playerID', 'gameDate'])

if 'plusMinus' in df.columns:
    df['plusMinus'] = df['plusMinus'].apply(lambda x: int(x) if pd.notna(x) else 0)

if 'fpts' not in df.columns:
    df['fpts'] = df.apply(calculate_dk_fantasy_points, axis=1)

df['target'] = df.groupby('playerID')['fpts'].shift(-1)

numeric_stat_cols = ['pts', 'reb', 'ast', 'stl', 'blk', 'TOV', 'mins', 'PF', 'fga', 'fgm', 'fgp', 'fta', 'ftm', 'ftp', 'tptfga', 'tptfgm', 'tptfgp', 'OffReb', 'DefReb', 'usage', 'plusMinus']

pipeline = FeaturePipeline()
pipeline.add(RollingStatsTransformer(
    windows=ROLLING_WINDOWS,
    stats=numeric_stat_cols,
    include_std=True
))
pipeline.add(EWMATransformer(
    span=EWMA_SPAN,
    stats=numeric_stat_cols
))

df = pipeline.fit_transform(df)
df = df.drop(columns=numeric_stat_cols)
df = df.dropna(subset=['target'])

metadata_cols = ['playerID', 'playerName', 'team', 'pos', 'gameDate', 'fpts', 'target', 'pts', 'reb', 'ast', 'stl', 'blk', 'TOV', 'mins']
feature_cols = [col for col in df.columns if col not in metadata_cols and df[col].dtype in ['int64', 'float64', 'bool']]

player_game_counts = df.groupby('playerID').size()
qualified_players = player_game_counts[player_game_counts >= MIN_GAMES].index
df_qualified = df[df['playerID'].isin(qualified_players)]

print(f"Training {len(qualified_players)} player models...")
player_models = {}
training_stats = []

for i, player_id in enumerate(qualified_players, 1):
    player_data = df_qualified[df_qualified['playerID'] == player_id]
    X_player = player_data[feature_cols].fillna(0)
    y_player = player_data['target']

    if len(X_player) < MIN_GAMES:
        continue

    model = XGBoostModel()
    model.train(X_player, y_player)
    player_models[player_id] = model

    training_stats.append({
        'playerID': player_id,
        'num_games': len(X_player),
        'mean_target': y_player.mean(),
        'std_target': y_player.std()
    })

    if i % 100 == 0:
        print(f"  Trained {i}/{len(qualified_players)} models")

training_stats_df = pd.DataFrame(training_stats)

print("Generating slate features...")
if 'longName' in salaries_df.columns and 'playerName' not in salaries_df.columns:
    salaries_df['playerName'] = salaries_df['longName']

training_data_for_slate = training_data.copy()
training_data_for_slate['gameDate'] = pd.to_datetime(training_data_for_slate['gameDate'], format='%Y%m%d', errors='coerce')

if 'fpts' not in training_data_for_slate.columns:
    training_data_for_slate['fpts'] = training_data_for_slate.apply(calculate_dk_fantasy_points, axis=1)

training_features = pipeline.transform(training_data_for_slate)

slate_features = []

for _, player_row in salaries_df.iterrows():
    player_id = player_row['playerID']
    player_features = training_features[training_features['playerID'] == player_id]

    if len(player_features) < min(ROLLING_WINDOWS):
        continue

    last_row = player_features.iloc[-1]

    features = {
        'playerID': player_id,
        'playerName': player_row.get('playerName', ''),
        'team': player_row.get('team', ''),
        'pos': player_row.get('pos', ''),
        'salary': player_row.get('salary', 0)
    }

    for col in last_row.index:
        if col not in metadata_cols and training_features[col].dtype in ['int64', 'float64', 'bool']:
            features[col] = last_row[col]

    slate_features.append(features)

slate_features_df = pd.DataFrame(slate_features)

print("Generating predictions...")
X_slate = slate_features_df.reindex(columns=feature_cols, fill_value=0).fillna(0)

projections_list = []

for idx, row in slate_features_df.iterrows():
    player_id = row['playerID']

    if player_id in player_models:
        model = player_models[player_id]
        X_player = X_slate.iloc[[idx]]
        prediction = model.predict(X_player)[0]
    else:
        prediction = np.nan

    projections_list.append({
        'playerID': player_id,
        'playerName': row['playerName'],
        'team': row['team'],
        'pos': row['pos'],
        'salary': row['salary'],
        'projected_fpts': prediction
    })

projections_df = pd.DataFrame(projections_list)
projections_df['salary'] = pd.to_numeric(projections_df['salary'], errors='coerce')

print("Loading actuals...")
filters = {'start_date': TARGET_DATE, 'end_date': TARGET_DATE}
actuals_df = storage.load('box_scores', filters)

if not actuals_df.empty:
    actuals_df['actual_fpts'] = actuals_df.apply(calculate_dk_fantasy_points, axis=1)

    if 'longName' in actuals_df.columns and 'playerName' not in actuals_df.columns:
        actuals_df['playerName'] = actuals_df['longName']

    if 'plusMinus' in actuals_df.columns:
        actuals_df['plusMinus'] = actuals_df['plusMinus'].apply(lambda x: int(x) if pd.notna(x) else 0)

merged = projections_df[projections_df['projected_fpts'].notna()].merge(
    actuals_df[['playerID', 'actual_fpts']],
    on='playerID',
    how='inner'
)

merged['error'] = merged['actual_fpts'] - merged['projected_fpts']
merged['abs_error'] = merged['error'].abs()

mape_metric = MAPEMetric()
rmse_metric = RMSEMetric()
corr_metric = CorrelationMetric()

y_true = merged['actual_fpts'].values
y_pred = merged['projected_fpts'].values

mape = mape_metric.calculate(y_true, y_pred)
rmse = rmse_metric.calculate(y_true, y_pred)
corr = corr_metric.calculate(y_true, y_pred)

merged['salary_tier'] = pd.cut(merged['salary'], bins=[0, 4000, 6000, 8000, 15000], labels=['Low', 'Mid', 'High', 'Elite'])

training_stats_merge = merged.merge(
    training_stats_df[['playerID', 'num_games']],
    on='playerID',
    how='left'
)

print("\nGenerating charts...")

print("1. Training games distribution...")
fig, ax = plt.subplots(figsize=(12, 6))
n, bins, patches = ax.hist(training_stats_df['num_games'], bins=30, edgecolor='white', linewidth=1.5)

cmap = create_gradient_cmap(PALETTE_COOL)
norm = plt.Normalize(vmin=bins.min(), vmax=bins.max())
for bin_val, patch in zip(bins, patches):
    patch.set_facecolor(cmap(norm(bin_val)))
    patch.set_alpha(0.85)

mean_val = training_stats_df['num_games'].mean()
ax.axvline(mean_val, color=MODERN_COLORS['danger'], linestyle='--', linewidth=3, label=f"Mean: {mean_val:.1f}", zorder=10)

ax.set_xlabel('Number of Training Games', fontsize=13, fontweight='bold')
ax.set_ylabel('Number of Players', fontsize=13, fontweight='bold')
ax.set_title('Distribution of Training Games Per Player', fontsize=16, fontweight='bold', pad=20)
ax.legend(fontsize=12, loc='upper right', framealpha=0.95)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / '01_training_games_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

print("2. Predicted vs Actual scatter plot...")
fig, ax = plt.subplots(figsize=(11, 9))

scatter = ax.scatter(
    merged['projected_fpts'],
    merged['actual_fpts'],
    c=merged['actual_fpts'],
    cmap=create_gradient_cmap(PALETTE_GRADIENT),
    alpha=0.7,
    s=100,
    edgecolors='white',
    linewidth=1.2
)

max_val = max(merged['actual_fpts'].max(), merged['projected_fpts'].max())
ax.plot([0, max_val], [0, max_val], color=MODERN_COLORS['danger'], linestyle='--', linewidth=3, label='Perfect Prediction', zorder=5)

cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
cbar.set_label('Actual Fantasy Points', fontsize=12, fontweight='bold')

ax.set_xlabel('Projected Fantasy Points', fontsize=13, fontweight='bold')
ax.set_ylabel('Actual Fantasy Points', fontsize=13, fontweight='bold')
ax.set_title(f'Predicted vs Actual Fantasy Points', fontsize=16, fontweight='bold', pad=20)

add_correlation_annotation(ax, merged['projected_fpts'].values, merged['actual_fpts'].values, loc='upper left')

ax.legend(fontsize=12, loc='lower right', framealpha=0.95)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / '02_predicted_vs_actual.png', dpi=300, bbox_inches='tight')
plt.close()

print("3. Error distribution histogram...")
fig, ax = plt.subplots(figsize=(12, 6))

n, bins, patches = ax.hist(merged['error'], bins=35, edgecolor='white', linewidth=1.5)

colors = [MODERN_COLORS['success'] if b < 0 else MODERN_COLORS['danger'] for b in bins[:-1]]
for patch, color in zip(patches, colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

mean_error = merged['error'].mean()
median_error = merged['error'].median()

ax.axvline(0, color='#212529', linestyle='-', linewidth=3, label='Zero Error', zorder=10)
ax.axvline(mean_error, color=MODERN_COLORS['warning'], linestyle='--', linewidth=2.5, label=f"Mean: {mean_error:.2f}", zorder=10)
ax.axvline(median_error, color=MODERN_COLORS['purple'], linestyle='--', linewidth=2.5, label=f"Median: {median_error:.2f}", zorder=10)

ax.set_xlabel('Error (Actual - Predicted)', fontsize=13, fontweight='bold')
ax.set_ylabel('Frequency', fontsize=13, fontweight='bold')
ax.set_title('Prediction Error Distribution', fontsize=16, fontweight='bold', pad=20)
ax.legend(fontsize=12, loc='upper right', framealpha=0.95)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / '03_error_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

print("4. Error by salary tier boxplot...")
fig, ax = plt.subplots(figsize=(12, 7))

tier_order = ['Low', 'Mid', 'High', 'Elite']
tier_data = [merged[merged['salary_tier'] == tier]['abs_error'].dropna() for tier in tier_order]

bp = ax.boxplot(
    tier_data,
    labels=tier_order,
    patch_artist=True,
    widths=0.6,
    medianprops=dict(color='#212529', linewidth=3),
    boxprops=dict(facecolor=MODERN_COLORS['primary'], alpha=0.7, edgecolor='white', linewidth=2),
    whiskerprops=dict(color='#495057', linewidth=2, linestyle='--'),
    capprops=dict(color='#495057', linewidth=2),
    flierprops=dict(marker='o', markerfacecolor=MODERN_COLORS['danger'], markersize=7, alpha=0.6, markeredgecolor='white', markeredgewidth=1)
)

for patch, color in zip(bp['boxes'], PALETTE_GRADIENT):
    patch.set_facecolor(color)
    patch.set_alpha(0.75)

ax.set_xlabel('Salary Tier', fontsize=13, fontweight='bold', labelpad=12)
ax.set_ylabel('Absolute Error (Fantasy Points)', fontsize=13, fontweight='bold', labelpad=12)
ax.set_title('Prediction Error Distribution by Salary Tier', fontsize=16, fontweight='bold', pad=20)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / '04_error_by_salary_tier.png', dpi=300, bbox_inches='tight')
plt.close()

print("5. Error vs training games scatter...")
fig, ax = plt.subplots(figsize=(12, 7))

scatter = ax.scatter(
    training_stats_merge['num_games'],
    training_stats_merge['abs_error'],
    c=training_stats_merge['abs_error'],
    cmap=create_gradient_cmap([MODERN_COLORS['success'], MODERN_COLORS['warning'], MODERN_COLORS['danger']]),
    alpha=0.7,
    s=90,
    edgecolors='white',
    linewidth=1.2
)

z = np.polyfit(training_stats_merge['num_games'].dropna(), training_stats_merge['abs_error'].dropna(), 1)
p = np.poly1d(z)
x_line = np.linspace(training_stats_merge['num_games'].min(), training_stats_merge['num_games'].max(), 100)
ax.plot(x_line, p(x_line), color=MODERN_COLORS['navy'], linestyle='--', linewidth=2.5, label='Trend Line', zorder=5)

cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
cbar.set_label('Absolute Error', fontsize=12, fontweight='bold')

ax.set_xlabel('Number of Training Games', fontsize=13, fontweight='bold', labelpad=12)
ax.set_ylabel('Absolute Error (Fantasy Points)', fontsize=13, fontweight='bold', labelpad=12)
ax.set_title('Model Error vs Training Data Volume', fontsize=16, fontweight='bold', pad=20)

add_correlation_annotation(ax, training_stats_merge['num_games'].values, training_stats_merge['abs_error'].values, loc='upper right')

ax.legend(fontsize=12, loc='upper left', framealpha=0.95)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / '05_error_vs_training_games.png', dpi=300, bbox_inches='tight')
plt.close()

print("6. Salary tier performance metrics...")
tier_stats = merged.groupby('salary_tier', observed=False).agg({
    'playerID': 'count',
    'projected_fpts': 'mean',
    'actual_fpts': 'mean',
    'abs_error': 'mean',
    'pct_error': 'mean'
}).rename(columns={'playerID': 'count'})

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

x_pos = np.arange(len(tier_stats))
bar_width = 0.65

bars1 = axes[0].bar(x_pos, tier_stats['pct_error'], width=bar_width, color=PALETTE_WARM, edgecolor='white', linewidth=2, alpha=0.85)
axes[0].axhline(30, color=MODERN_COLORS['danger'], linestyle='--', linewidth=3, label='30% Target', zorder=10)
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels(tier_stats.index, fontsize=12, fontweight='bold')
axes[0].set_ylabel('MAPE (%)', fontsize=13, fontweight='bold', labelpad=12)
axes[0].set_xlabel('Salary Tier', fontsize=13, fontweight='bold', labelpad=12)
axes[0].set_title('Mean Absolute Percentage Error by Salary Tier', fontsize=14, fontweight='bold', pad=15)
axes[0].legend(fontsize=11, loc='upper right', framealpha=0.95)
axes[0].yaxis.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
axes[0].set_axisbelow(True)
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)
add_value_labels(axes[0], spacing=2, format_str='{:.1f}%')

bars2 = axes[1].bar(x_pos, tier_stats['abs_error'], width=bar_width, color=PALETTE_COOL, edgecolor='white', linewidth=2, alpha=0.85)
axes[1].set_xticks(x_pos)
axes[1].set_xticklabels(tier_stats.index, fontsize=12, fontweight='bold')
axes[1].set_ylabel('Mean Absolute Error (Fantasy Points)', fontsize=13, fontweight='bold', labelpad=12)
axes[1].set_xlabel('Salary Tier', fontsize=13, fontweight='bold', labelpad=12)
axes[1].set_title('Mean Absolute Error by Salary Tier', fontsize=14, fontweight='bold', pad=15)
axes[1].yaxis.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
axes[1].set_axisbelow(True)
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)
add_value_labels(axes[1], spacing=0.3, format_str='{:.2f}')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '06_salary_tier_metrics.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"\nAll charts saved to {OUTPUT_DIR}/")
print("\nChart files created:")
for f in sorted(OUTPUT_DIR.glob('*.png')):
    print(f"  - {f.name}")
