import pandas as pd

# input/output files
input_file = "imf_economic_pillar_score_monthly.csv"
output_file = "dashboard_overall_scores.csv"
snapshot_output = "dashboard_score_snapshot.csv"

# load monthly pillar score data
try:
    df = pd.read_csv(input_file, parse_dates=['date'])
    print(f"✔ Successfully loaded {input_file}")
except FileNotFoundError:
    print(f"!! ERROR: '{input_file}' not found. !!")
    exit()

# calculate 12 month trfailing average score
print("Calculating trailing 12-month average scores...")
df.sort_values(by=['country', 'date'], inplace=True)
df['trailing_12m_avg'] = df.groupby('country')['imf_economic_pillar_score'].transform(
    lambda x: x.rolling(window=12, min_periods=1).mean()
)

# get most recent score per country
print("Extracting the most recent score for each country...")
latest_scores = df.groupby('country').last().reset_index()
overall_scores = latest_scores[['country', 'trailing_12m_avg']].rename(
    columns={'trailing_12m_avg': 'overall_risk_score'}
)

# transform to 0-100 score
print("Converting scores to a 0–100 scale...")
min_score = overall_scores['overall_risk_score'].min()
max_score = overall_scores['overall_risk_score'].max()

overall_scores['risk_score_0_to_100'] = (
    (overall_scores['overall_risk_score'] - min_score) / (max_score - min_score) * 100
).round(1)

# assign risk categories
print("Assigning risk categories...")

def assign_category(score):
    if score >= 70:
        return 'High Risk'
    elif score >= 30:
        return 'Medium Risk'
    else:
        return 'Low Risk'

overall_scores['risk_category'] = overall_scores['risk_score_0_to_100'].apply(assign_category)

# export to csv
overall_scores.to_csv(output_file, index=False)
print(f"Process complete! Dashboard-ready data saved to: {output_file}")

# export snapshot of top/bottom countries
top5 = overall_scores.sort_values('risk_score_0_to_100', ascending=False).head(5)
bottom5 = overall_scores.sort_values('risk_score_0_to_100').head(5)
snapshot = pd.concat([top5, bottom5])
snapshot.to_csv(snapshot_output, index=False)
print(f"✔ Snapshot saved to: {snapshot_output}")

# Preview Final Output
print("\nTop 5 highest risk countries:")
print(top5[['country', 'risk_score_0_to_100', 'risk_category']])

print("\nTop 5 lowest risk countries:")
print(bottom5[['country', 'risk_score_0_to_100', 'risk_category']])
