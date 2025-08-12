import pandas as pd
import numpy as np

INPUT_FILE = "allianz_normalised_scores.csv"  # cols: date, country, zscore_grade, zscore_risk
OUTPUT_FILE = "allianz_monthly_z.csv"

df = pd.read_csv(INPUT_FILE, parse_dates=["date"])
need = {"date","country","zscore_grade","zscore_risk"}
missing = need - set(df.columns)
if missing:
    raise ValueError(f"Missing columns in {INPUT_FILE}: {sorted(missing)}")

df["country"] = df["country"].astype(str)
df = df.sort_values(["country","date"])

min_date, max_date = df["date"].min(), df["date"].max()
all_months = pd.date_range(min_date, max_date, freq="MS")

expanded = []
for c, g in df.groupby("country"):
    g = g.set_index("date").reindex(all_months)
    g["country"] = c
    # forward-fill quarterly → monthly
    g["zscore_grade"] = g["zscore_grade"].ffill()
    g["zscore_risk"] = g["zscore_risk"].ffill()
    expanded.append(g.reset_index().rename(columns={"index":"date"}))

out = pd.concat(expanded, ignore_index=True)
# Drop any rows where both inputs still NaN (no history)
out = out.dropna(subset=["zscore_grade","zscore_risk"], how="all")

out.to_csv(OUTPUT_FILE, index=False)
print(f"Saved monthly-expanded z-scores {OUTPUT_FILE} (rows={len(out)}, countries={out['country'].nunique()})")
print(f"Dates: {out['date'].min().date()} {out['date'].max().date()}")
