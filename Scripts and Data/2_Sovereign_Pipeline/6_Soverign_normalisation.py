import os
import pandas as pd

# PATHS
in_path = r'C:\Users\cdye\OneDrive\Desktop\GE Heathcare\Data\Sovereign Ratings\Sovereign Scripts\sovereign_monthly_z.csv'
out_dir = r"C:\Users\cdye\OneDrive\Desktop\GE Heathcare\Data\Sovereign Ratings\Sovereign Scripts"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "Sovereign_monthly_normal.csv")

# load yearly sovereign ratings
df = pd.read_csv(in_path)

# Ensure 'Year' column exists and is numeric
if 'Year' not in df.columns:
    raise ValueError("Expected 'Year' column not found in input file.")

df['Year'] = pd.to_numeric(df['Year'], errors='coerce')

# Pivot so each agency becomes a column, with rows = (Year, Country)
df_pivot = df.pivot_table(
    index='Year',
    values=['Moody', 'S&P', 'Fitch'],
    aggfunc='mean'  # or 'first' depending on your data format
)

# Convert index to datetime (Jan 1 of each year)
df_pivot.index = pd.to_datetime(df_pivot.index.astype(str) + "-01-01", format="%Y-%m-%d")
df_pivot.index.name = 'date'

# Reindex to monthly frequency and forward-fill
monthly_index = pd.date_range("2006-01-01", "2025-05-01", freq="MS")
df_monthly = df_pivot.reindex(monthly_index).asfreq('MS').ffill()

# Calculate monthly cross-sectional z-scores
mu = df_monthly.mean(axis=1)
sigma = df_monthly.std(axis=1, ddof=0)
df_z = df_monthly.sub(mu, axis=0).div(sigma, axis=0)

# Rename columns
df_z.columns = [f"{col}_z" for col in df_z.columns]

# Save result
df_z.to_csv(out_path, index_label='date')
print("✔ Saved monthly sovereign z-scores to:", out_path)

