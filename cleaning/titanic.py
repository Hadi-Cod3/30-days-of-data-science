"""
Titanic Dataset: cleaning and descriptive statistics (Day 6)

Decision rules, written before running anything:
1. Drop any column with more than 40% of its values missing.
2. When the mean and median disagree, fill gaps with the median.
3. If an outlier is impossible, fix or remove it. If it is extreme but possible, keep it.
4. Drop identifier columns that are not useful for analysis.
"""
import pandas as pd

df = pd.read_csv("Titanic-Dataset.csv")
raw = df.copy()      # untouched, kept for the before-and-after comparison
clean = df.copy()    # all changes happen here

# Missing values per column, before any cleaning
missing_pct = (raw.isna().mean() * 100).round(1).sort_values(ascending=False)

# Flags come first, because they are built from values that get changed or dropped
clean['had_a_cabin_recorded'] = clean['Cabin'].notna()
clean['zero_fare'] = clean['Fare'] == 0

# Drop Cabin (77% missing) and the identifier columns
clean = clean.drop(columns=['Cabin', 'xPassengerId', 'Name', 'Ticket'])

# Fill Age with the median and Embarked with the most common port
clean['Age'] = clean['Age'].fillna(clean['Age'].median())
clean['Embarked'] = clean['Embarked'].fillna(clean['Embarked'].mode()[0])

# Fare outliers using the quartile fence (fares cannot be negative, so no lower fence)
fare = clean['Fare']
q1, q3 = fare.quantile(0.25), fare.quantile(0.75)
iqr = q3 - q1
upper_bound = q3 + 1.5 * iqr
outliers = clean[fare > upper_bound]
non_first_class_outliers = outliers[outliers['Pclass'] != 1]
zeros = clean[clean['zero_fare']]

# Before and after
cols = ['Age', 'Fare']
before = raw[cols].agg(['mean', 'median', 'std']).round(1)
after = clean[cols].agg(['mean', 'median', 'std']).round(1)

print("Missing values before cleaning (%):")
print(missing_pct[missing_pct > 0])
print("\nShape after cleaning:", clean.shape)
print("Missing values after cleaning:", clean.isna().sum().sum())

print("\nBefore cleaning:")
print(before)
print("\nAfter cleaning:")
print(after)

print(f"\nFare: median {fare.median():.1f}, mean {fare.mean():.1f}")
print(f"Fence: Q1 {q1:.1f}, Q3 {q3:.1f}, IQR {iqr:.1f}, upper bound {upper_bound:.1f}")
print(f"Outliers above the fence: {len(outliers)}")
print(outliers['Pclass'].value_counts())

print("\nNon-first-class outliers:")
print(non_first_class_outliers[['SibSp', 'Parch', 'Fare', 'Pclass']])

print(f"\nZero fares: {len(zeros)}")
print(zeros[['SibSp', 'Parch', 'Fare', 'Pclass']])

"""
DECISION TREE: how each column was handled

Is the column an identifier (no analytical use)?
    YES -> Drop it (PassengerId, Name, Ticket)
    NO  -> Is more than 40% of the column missing?
        YES -> Drop it, but keep a flag of whether a value existed (Cabin, 77.1%)
        NO  -> Is any of the column missing?
            YES -> Do the mean and median disagree?
                YES -> Fill with the median
                NO  -> Median is still safe, so fill with the median (Age, 19.9%)
                (Text column: fill with the most common value, Embarked)
            NO  -> Are there extreme values (quartile fence)?
                YES -> Is the value impossible?
                    YES -> Fix it or remove the row
                    NO  -> Keep it (Fare: 116 above the fence, 104 first class,12 group tickets)
                NO  -> Leave it alone
    Zero fares (15): possible but ambiguous, so kept and flagged, never filled

VERDICT: The choice most likely to change a conclusion is filling Age with the
median: the average barely moved (29.7 to 29.4), but the spread fell from 14.5
to 13.0, so any analysis of how age varies across passengers would look tighter
than the real data.
"""