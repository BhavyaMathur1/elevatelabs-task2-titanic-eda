"""
===============================================================
  Task 2: Exploratory Data Analysis (EDA) — Titanic Dataset
  Tools : Pandas, Matplotlib, Seaborn
===============================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 0. GLOBAL STYLE
# ─────────────────────────────────────────────
sns.set_theme(style='darkgrid', palette='muted')
plt.rcParams.update({
    'figure.facecolor': '#0d1117',
    'axes.facecolor':   '#161b22',
    'axes.edgecolor':   '#30363d',
    'axes.labelcolor':  '#c9d1d9',
    'xtick.color':      '#8b949e',
    'ytick.color':      '#8b949e',
    'text.color':       '#c9d1d9',
    'grid.color':       '#21262d',
    'figure.dpi':       120,
})
ACCENT   = '#58a6ff'
SUCCESS  = '#3fb950'
WARNING  = '#d29922'
DANGER   = '#f85149'
PURPLE   = '#bc8cff'

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
df = pd.read_csv('titanic.csv')
print("=" * 60)
print("  TITANIC — EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# ─────────────────────────────────────────────
# 2. BASIC INFO
# ─────────────────────────────────────────────
print(f"\n📐 Shape : {df.shape[0]} rows × {df.shape[1]} columns")
print("\n📋 Data Types:\n", df.dtypes.to_string())
print(f"\n🔍 Missing Values:\n{df.isnull().sum().to_string()}")
print(f"\n📊 Duplicates: {df.duplicated().sum()}")

# ─────────────────────────────────────────────
# 3. SUMMARY STATISTICS
# ─────────────────────────────────────────────
print("\n" + "─" * 60)
print("  SUMMARY STATISTICS")
print("─" * 60)
print(df.describe(include='all').T.to_string())

# ─────────────────────────────────────────────
# 4. HELPER — save figure
# ─────────────────────────────────────────────
def save(fig, name, tight=True):
    if tight:
        fig.tight_layout()
    fig.savefig(f'plots/{name}.png', bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✅  Saved plots/{name}.png")

import os
os.makedirs('plots', exist_ok=True)

# ─────────────────────────────────────────────
# 5. SURVIVAL OVERVIEW
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Survival Overview', fontsize=16, fontweight='bold',
             color='white', y=1.02)

# 5a. Survival count
counts = df['survived'].value_counts()
bars = axes[0].bar(['Did Not Survive', 'Survived'], counts.values,
                   color=[DANGER, SUCCESS], edgecolor='#30363d', linewidth=0.8)
axes[0].set_title('Survival Count', color='white')
axes[0].set_ylabel('Count', color='white')
for bar, val in zip(bars, counts.values):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                 str(val), ha='center', fontsize=12, color='white')

# 5b. Survival rate by sex
survival_sex = df.groupby('sex')['survived'].mean() * 100
colors_sex = [SUCCESS if s > 50 else DANGER for s in survival_sex.values]
bars2 = axes[1].bar(survival_sex.index, survival_sex.values,
                    color=colors_sex, edgecolor='#30363d')
axes[1].set_title('Survival Rate by Sex (%)', color='white')
axes[1].set_ylabel('Survival Rate (%)', color='white')
axes[1].set_ylim(0, 100)
for bar, val in zip(bars2, survival_sex.values):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f'{val:.1f}%', ha='center', color='white')

# 5c. Survival rate by class
survival_class = df.groupby('pclass')['survived'].mean() * 100
bars3 = axes[2].bar([f'Class {c}' for c in survival_class.index],
                    survival_class.values,
                    color=[ACCENT, WARNING, DANGER], edgecolor='#30363d')
axes[2].set_title('Survival Rate by Pclass (%)', color='white')
axes[2].set_ylabel('Survival Rate (%)', color='white')
axes[2].set_ylim(0, 100)
for bar, val in zip(bars3, survival_class.values):
    axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f'{val:.1f}%', ha='center', color='white')

save(fig, '01_survival_overview')

# ─────────────────────────────────────────────
# 6. HISTOGRAMS — numeric features
# ─────────────────────────────────────────────
numeric_cols = ['age', 'fare', 'sibsp', 'parch']
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Histograms of Numeric Features', fontsize=16,
             fontweight='bold', color='white')

for ax, col in zip(axes.flatten(), numeric_cols):
    data = df[col].dropna()
    ax.hist(data, bins=30, color=ACCENT, edgecolor='#0d1117', alpha=0.85)
    ax.axvline(data.mean(),   color=SUCCESS, linestyle='--', linewidth=1.5,
               label=f'Mean={data.mean():.1f}')
    ax.axvline(data.median(), color=WARNING, linestyle=':',  linewidth=1.5,
               label=f'Median={data.median():.1f}')
    ax.set_title(col.capitalize(), color='white')
    ax.set_xlabel(col, color='white')
    ax.set_ylabel('Frequency', color='white')
    ax.legend(fontsize=8, framealpha=0.3)

save(fig, '02_histograms')

# ─────────────────────────────────────────────
# 7. BOXPLOTS
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Boxplots — Age & Fare by Survival', fontsize=16,
             fontweight='bold', color='white')

palette = {0: DANGER, 1: SUCCESS}

for ax, col in zip(axes, ['age', 'fare']):
    data_0 = df[df['survived'] == 0][col].dropna()
    data_1 = df[df['survived'] == 1][col].dropna()
    bp = ax.boxplot([data_0, data_1],
                    patch_artist=True,
                    labels=['Did Not Survive', 'Survived'],
                    medianprops=dict(color='white', linewidth=2))
    colors = [DANGER, SUCCESS]
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    for whisker in bp['whiskers']:
        whisker.set_color('#8b949e')
    for cap in bp['caps']:
        cap.set_color('#8b949e')
    for flier in bp['fliers']:
        flier.set(marker='o', color='#8b949e', alpha=0.4, markersize=3)

    ax.set_title(col.capitalize(), color='white')
    ax.set_ylabel(col, color='white')

save(fig, '03_boxplots')

# ─────────────────────────────────────────────
# 8. CORRELATION MATRIX
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 7))
fig.suptitle('Correlation Matrix', fontsize=16,
             fontweight='bold', color='white')

corr = df[['survived', 'pclass', 'age', 'sibsp', 'parch', 'fare']].corr()
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
            ax=ax, linewidths=0.5, linecolor='#0d1117',
            cbar_kws={'shrink': 0.8}, vmin=-1, vmax=1)
ax.set_title('', color='white')

save(fig, '04_correlation_matrix')
print("\n  Key correlations with 'survived':")
print(corr['survived'].sort_values(ascending=False).to_string())

# ─────────────────────────────────────────────
# 9. PAIRPLOT (subset)
# ─────────────────────────────────────────────
pair_df = df[['survived', 'pclass', 'age', 'fare']].dropna().copy()
pair_df['survived'] = pair_df['survived'].map({0: 'No', 1: 'Yes'})

pp = sns.pairplot(pair_df, hue='survived', diag_kind='kde',
                  palette={'No': DANGER, 'Yes': SUCCESS},
                  plot_kws=dict(alpha=0.5, s=15),
                  diag_kws=dict(fill=True, alpha=0.5))
pp.figure.suptitle('Pairplot — Key Features', y=1.02,
                   fontsize=14, fontweight='bold', color='white')
pp.figure.savefig('plots/05_pairplot.png', bbox_inches='tight',
                  facecolor='#0d1117')
plt.close()
print("  ✅  Saved plots/05_pairplot.png")

# ─────────────────────────────────────────────
# 10. AGE DISTRIBUTION BY CLASS & SEX
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Age Distribution — Class & Sex', fontsize=16,
             fontweight='bold', color='white')

class_palette = {1: ACCENT, 2: WARNING, 3: DANGER}
for pclass, color in class_palette.items():
    data = df[df['pclass'] == pclass]['age'].dropna()
    axes[0].hist(data, bins=25, alpha=0.5, color=color,
                 label=f'Class {pclass}', edgecolor='#0d1117')
axes[0].set_title('Age by Passenger Class', color='white')
axes[0].set_xlabel('Age', color='white')
axes[0].set_ylabel('Count', color='white')
axes[0].legend()

for sex, color in [('male', ACCENT), ('female', PURPLE)]:
    data = df[df['sex'] == sex]['age'].dropna()
    axes[1].hist(data, bins=25, alpha=0.5, color=color,
                 label=sex.capitalize(), edgecolor='#0d1117')
axes[1].set_title('Age by Sex', color='white')
axes[1].set_xlabel('Age', color='white')
axes[1].set_ylabel('Count', color='white')
axes[1].legend()

save(fig, '06_age_distribution')

# ─────────────────────────────────────────────
# 11. FARE DISTRIBUTION (log scale + skewness)
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Fare Distribution — Skewness Detection', fontsize=16,
             fontweight='bold', color='white')

fare = df['fare'].dropna()
skew_val = fare.skew()

axes[0].hist(fare, bins=40, color=WARNING, edgecolor='#0d1117', alpha=0.8)
axes[0].set_title(f'Fare (Raw)  |  Skewness = {skew_val:.2f}', color='white')
axes[0].set_xlabel('Fare', color='white')
axes[0].set_ylabel('Count', color='white')

log_fare = np.log1p(fare)
axes[1].hist(log_fare, bins=40, color=SUCCESS, edgecolor='#0d1117', alpha=0.8)
axes[1].set_title(f'log(Fare+1)  |  Skewness = {log_fare.skew():.2f}',
                  color='white')
axes[1].set_xlabel('log(Fare + 1)', color='white')
axes[1].set_ylabel('Count', color='white')

save(fig, '07_fare_skewness')
print(f"\n  Fare skewness (raw)      : {skew_val:.3f}  → highly right-skewed")
print(f"  Fare skewness (log+1)    : {log_fare.skew():.3f}  → much more normal")

# ─────────────────────────────────────────────
# 12. EMBARKATION ANALYSIS
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Embarkation Analysis', fontsize=16,
             fontweight='bold', color='white')

embark_counts = df['embarked'].value_counts()
axes[0].bar(embark_counts.index, embark_counts.values,
            color=[ACCENT, WARNING, SUCCESS], edgecolor='#0d1117')
axes[0].set_title('Passenger Count by Port', color='white')
axes[0].set_xlabel('Port (S=Southampton, C=Cherbourg, Q=Queenstown)',
                   color='white', fontsize=8)
axes[0].set_ylabel('Count', color='white')

surv_emb = df.groupby('embarked')['survived'].mean() * 100
axes[1].bar(surv_emb.index, surv_emb.values,
            color=[ACCENT, WARNING, SUCCESS], edgecolor='#0d1117')
axes[1].set_title('Survival Rate by Port (%)', color='white')
axes[1].set_xlabel('Port', color='white')
axes[1].set_ylabel('Survival Rate (%)', color='white')
axes[1].set_ylim(0, 100)

save(fig, '08_embarkation')

# ─────────────────────────────────────────────
# 13. MISSING VALUE HEATMAP
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
fig.suptitle('Missing Values Heatmap', fontsize=16,
             fontweight='bold', color='white')

missing = df.isnull()
sns.heatmap(missing, yticklabels=False, cbar=False,
            cmap=['#21262d', DANGER], ax=ax)
ax.set_title('Yellow = Missing', color='white')

save(fig, '09_missing_values')

# ─────────────────────────────────────────────
# 14. FINAL SUMMARY PRINT
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  KEY INSIGHTS")
print("=" * 60)
surv_rate = df['survived'].mean() * 100
print(f"  1. Overall survival rate      : {surv_rate:.1f}%")

female_surv = df[df['sex']=='female']['survived'].mean() * 100
male_surv   = df[df['sex']=='male']['survived'].mean()   * 100
print(f"  2. Female survival rate       : {female_surv:.1f}%")
print(f"     Male survival rate         : {male_surv:.1f}%")

for c in [1, 2, 3]:
    r = df[df['pclass']==c]['survived'].mean() * 100
    print(f"  3. Class {c} survival rate        : {r:.1f}%")

print(f"  4. Fare is highly right-skewed (skew={fare.skew():.2f}), log transform helps")
print(f"  5. ~{df['age'].isnull().mean()*100:.0f}% of age values are missing — imputation needed")
print(f"  6. Strong negative correlation: pclass ↔ survived ({corr.loc['pclass','survived']:.2f})")
print(f"     Positive correlation: fare ↔ survived ({corr.loc['fare','survived']:.2f})")
print("\n  All plots saved to ./plots/")
print("=" * 60)
