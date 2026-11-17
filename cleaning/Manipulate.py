"""
Day 7: Cleaning, Pset B. Which cleaning strategy recovers the truth?
Data: IBM HR Attrition (MonthlyIncome is skewed, Age is symmetric)
Answer lines, written BEFORE running anything:

Q1. Random gaps (15% and 30% of MonthlyIncome blanked at random).
    Prediction: filling with the median or the job-role median will land the
    mean within 2% of the true mean, and filling with the mean will land closer
    still. If the median fill is within 2%, I'll recommend it for skewed columns.

Q2. Gaps that are not random (only the highest earners blanked).
    Prediction: every fill will pull the mean BELOW the true mean, because the
    fill values come only from the people who remain, and no strategy will fix
    it. If that's right, I'll treat missing high earners as a data-collection
    problem, not a cleaning one.

Q3. Spread (standard deviation after any fill).
    Prediction: the spread will shrink for every fill, because many rows now
    share one value, so it understates the real variation. If the shrink is
    bigger than 5%, I'll report it next to the mean.

Q4. Outliers (a few incomes multiplied by 10, plus a few impossible ages).
    Prediction: capping at the quartile fence will leave the mean closest to
    the truth, leaving them will be furthest, and removing will land in between.
    If so, I'll cap possible-but-extreme values and remove impossible ones.
"""

import numpy as np
import pandas as pd

df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")
truth = df.copy()   # never changed; every damaged copy starts from it

true_income = truth['MonthlyIncome']
true_age = truth['Age']


def pct_err(value, true_value):
    return (value - true_value) / true_value * 100


def score(series, true_series):
    """Percentage error of mean, median and spread against the truth."""
    return {
        'mean%': pct_err(series.mean(), true_series.mean()),
        'median%': pct_err(series.median(), true_series.median()),
        'std%': pct_err(series.std(), true_series.std()),
        'rows': len(series),
    }


def quartile_fences(series):
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr


# Truth
print("TRUTH")
print(f"  MonthlyIncome: mean={true_income.mean():.1f}  median={true_income.median():.1f}  std={true_income.std():.1f}")
print(f"  Age:           mean={true_age.mean():.1f}  median={true_age.median():.1f}  std={true_age.std():.1f}")

# Damage: missing values (seed set once, so the order of these steps matters)
np.random.seed(2)

def damage_random(fraction):
    d = truth.copy()
    rows = np.random.choice(d.index, size=int(fraction * len(d)), replace=False)
    d.loc[rows, 'MonthlyIncome'] = np.nan
    return d

df_15 = damage_random(0.15)
df_30 = damage_random(0.30)

df_top = truth.copy()
top_cutoff = true_income.quantile(0.85)
df_top.loc[df_top['MonthlyIncome'] >= top_cutoff, 'MonthlyIncome'] = np.nan

damaged = {"15% random": df_15, "30% random": df_30, "Top 15% missing": df_top}

print("\nDAMAGE CHECK")
for label, d in damaged.items():
    n = d['MonthlyIncome'].isna().sum()
    print(f"  {label}: {n} missing ({n / len(d) * 100:.1f}%)")
print(f"  Top-earner copy max: {df_top['MonthlyIncome'].max():.0f} (original max {true_income.max():.0f})")


# Strategies for missing values (every fill value comes from the damaged copy)
def fill_strategies(d):
    col = d['MonthlyIncome']
    role_median = d.groupby('JobRole')['MonthlyIncome'].transform('median')
    return {
        'Drop': col.dropna(),
        'Fill mean': col.fillna(col.mean()),
        'Fill median': col.fillna(col.median()),
        'Fill role median': col.fillna(role_median),
    }

fill_rows = {}
for label, d in damaged.items():
    for strategy, series in fill_strategies(d).items():
        fill_rows[(label, strategy)] = score(series, true_income)

print("\nMISSING VALUES: % error against the truth (MonthlyIncome)")
print(pd.DataFrame(fill_rows).T.round(2))


# Damage: outliers
df_outlier = truth.copy()
outlier_rows = np.random.choice(df_outlier.index, size=10, replace=False)
df_outlier.loc[outlier_rows, 'MonthlyIncome'] *= 10

df_old = truth.copy()
old_rows = np.random.choice(df_old.index, size=5, replace=False)
df_old.loc[old_rows, 'Age'] = 150


def outlier_treatments(series):
    lo, hi = quartile_fences(series)   # fences come from the damaged column
    return {
        'Leave': series,
        'Cap': series.clip(lo, hi),
        'Remove': series[(series >= lo) & (series <= hi)],
    }

for title, col, damaged_df, true_series in [
    ("OUTLIERS: incomes x10 (10 planted)", 'MonthlyIncome', df_outlier, true_income),
    ("OUTLIERS: impossible ages of 150 (5 planted)", 'Age', df_old, true_age),
]:
    rows = {name: score(series, true_series)
            for name, series in outlier_treatments(damaged_df[col]).items()}
    print(f"\n{title}: % error against the truth")
    print(pd.DataFrame(rows).T.round(2))
    print(f"  Planted max: {damaged_df[col].max():.0f}, true max: {true_series.max():.0f}")


"""
DAY 7 VERDICT (single random draw, seed 2; rerun with other seeds before trusting the ranking)

Q1. Random gaps: filling by each job role's median stayed within 2% of the true
    mean. A plain median fill missed my 2% bar because the column is skewed.
    Action: to recover the mean of a skewed column, fill by group median. Use a
    plain median fill only when the median is the statistic I care about.

Q2. Top earners missing: every strategy left the mean about 25% too low and
    halved the spread. No fill can recover values that are missing because they
    are high.
    Action: report the bias and treat it as a data-collection problem.

Q3. Spread: every single-value fill shrank it (7% at 15% gaps, 13 to 15% at 30%).
    Fill by role median shrank it least.
    Action: report the spread next to the mean whenever a column has been filled.

Q4. Outliers: capping was best for incomes. Removing failed my prediction because
    the fence also caught real high earners (116 rows removed, not the 10 planted).
    For the five impossible ages, removing was best.
    Action: cap extreme-but-possible values, remove only clearly impossible ones,
    and always check how many rows a fence actually removes.

What would change my mind: a different seed flipping the ranking.
"""