import pandas as pd

df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

# Task 1: meet the data
print(df.shape)
print(df.dtypes)
print("Missing values:", df.isna().sum().sum())
print("Duplicate rows:", df.duplicated().sum())

# Q1. Share who left by job satisfaction and work-life balance
# Q2. Share who left, overtime vs no overtime
# Q3. Share who left by stock option level
# Q4. Share who left by income band (low, middle, high)
# Q5. Years since last promotion, left vs stayed

# new columns
df['left'] = df['Attrition'] == 'Yes'
df['income_band'] = pd.qcut(df['MonthlyIncome'], 3, labels=['Low', 'Middle', 'High'])

print("Overall share who left:", round(df['left'].mean() * 100, 1))

# Q1
print((df.groupby('JobSatisfaction')['left'].mean() * 100).round(1))
print(df.groupby('JobSatisfaction').size())
print((df.groupby('WorkLifeBalance')['left'].mean() * 100).round(1))
print(df.groupby('WorkLifeBalance').size())

# Q2
print((df.groupby('OverTime')['left'].mean() * 100).round(1))
print(df.groupby('OverTime').size())

# Q3
print((df.groupby('StockOptionLevel')['left'].mean() * 100).round(1))
print(df.groupby('StockOptionLevel').size())

# Q4
print((df.groupby('income_band')['left'].mean() * 100).round(1))
print(df.groupby('income_band').size())

# Q5
print(df.groupby('left')['YearsSinceLastPromotion'].mean().round(1))

# merge: a small lookup table typed by hand (made-up details)
departments = pd.DataFrame({
    'Department': ['Sales', 'Research & Development', 'Human Resources'],
    'Location': ['Lagos', 'Ibadan', 'Abuja'],
    'Budget': ['High', 'Medium', 'Low']
})

# left join keeps every employee; each department appears once in the lookup
merged = df.merge(departments, on='Department', how='left')
print("Rows before and after merge:", df.shape[0], merged.shape[0])

# a question only the merged table can answer
print((merged.groupby('Location')['left'].mean() * 100).round(1))

# one answer worked out two ways: share who left among overtime workers
way1 = df.groupby('OverTime')['left'].mean()['Yes']
way2 = df[df['OverTime'] == 'Yes']['left'].mean()
print("Two ways:", round(way1 * 100, 1), round(way2 * 100, 1))