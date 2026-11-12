import pandas as pd

df = pd.read_csv("EPL_2025_2026_data.csv")

small = df[['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']].rename(columns={
            'HomeTeam':    'home',
            'AwayTeam':    'away',
            'FTHG':   'hg',
            'FTAG':   'ag',
            'FTR':       'res'
        }).copy()

# new columns
small['total_goals'] = small['hg'] + small['ag']
small['goal_diff'] = (small['hg'] - small['ag']).abs()

# filters
high_scoring = small[small['total_goals'] >= 4]
home_wins = small[small['res'] == 'H']

team = 'Chelsea'
matches = small[(small['home'] == team) | (small['away'] == team)]

# ten biggest wins
biggest_wins = small.sort_values("goal_diff", ascending=False).head(10)

# home goals per team, highest first
home_goals = small.groupby('home')['hg'].sum().sort_values(ascending=False)

# share of home wins, draws and away wins
result_share = small['res'].value_counts(normalize=True) * 100

# my own question: which teams win the most of their home games?
small['home_win'] = small['res'] == 'H'
home_win_rate = small.groupby('home')['home_win'].mean().sort_values(ascending=False) * 100

# printing
print("Shape:", df.shape)
print("Teams:", df['HomeTeam'].nunique())
print(small.dtypes)
print(small.isna().sum())

print("High scoring matches:", len(high_scoring))
print("Home wins:", len(home_wins))
print(team, "matches:", len(matches))

print(biggest_wins)
print(home_goals)
print(result_share.round(1))
print(home_win_rate.round(1))
