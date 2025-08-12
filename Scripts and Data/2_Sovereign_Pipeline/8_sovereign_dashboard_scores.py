import pandas as pd

# Load the Sovereign Pillar Data
input_file = "sovereign_pillar_scores.csv"  # adjust path if needed
df = pd.read_csv(input_file, parse_dates=["date"], low_memory=False)

# Standardize Country Column Name
if "Country" in df.columns:
    df.rename(columns={"Country": "country"}, inplace=True)

# Get Latest Entry Per Country
df_latest = df.sort_values(["country", "date"]).groupby("country", as_index=False).tail(1)

# Calculate 0–100 Scores from pc1
s = pd.to_numeric(df_latest["pc1"], errors="coerce")
mn, mx = s.min(), s.max()

if pd.isna(mn) or pd.isna(mx) or mx == mn:
    df_latest["risk_score_0_to_100"] = 50.0  # fallback if all values identical
else:
    df_latest["risk_score_0_to_100"] = (s - mn) / (mx - mn) * 100.0

# Save to New File
output_file = "sovereign_dashboard_overall_scores_final.csv"
df_latest[["country", "risk_score_0_to_100"]].round(2).to_csv(output_file, index=False)

print(f"Saved sovereign 0–100 dashboard scores → {output_file} (rows={len(df_latest)})")
print(df_latest[["country", "risk_score_0_to_100"]].head())
