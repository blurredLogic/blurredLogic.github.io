import pandas as pd
import numpy as np

INPUT_FILE = "allianz_pca_monthly.csv"
OUT_FLAT  = "allianz_flat_series_last12.csv"
OUT_TIES  = "allianz_ties_last12.csv"

df = pd.read_csv(INPUT_FILE, parse_dates=["date"])
latest = df["date"].max()
start = (latest - pd.offsets.MonthBegin(11)).normalize()

last12 = df[df["date"].between(start, latest)].copy()
print(f"Window checked: {start.date()} → {latest.date()}")

# per-country std in last 12 months
agg = last12.groupby("country").agg(
    grade_std=("zscore_grade","std"),
    risk_std=("zscore_risk","std"),
    pillar_std=("trade_payment_pillar_score","std")
).reset_index()

flat = agg[(agg["pillar_std"].fillna(0) < 1e-6)]
flat.to_csv(OUT_FLAT, index=False)
print(f"Countries with ~ zero pillar variance in last 12 months: {len(flat)} (saved {OUT_FLAT})")

# Compute trailing 12m average and find tied groups
last12 = last12.sort_values(["country","date"])
last12["trail12"] = last12.groupby("country")["trade_payment_pillar_score"].transform(
    lambda x: x.rolling(window=12, min_periods=12).mean()
)

latest_trail = last12.dropna(subset=["trail12"]).groupby("country").tail(1).copy()
if latest_trail.empty:
    print("No countries have 12 valid months; falling back to last value")
    latest_trail = last12.groupby("country").tail(1).copy()
    latest_trail["trail12"] = latest_trail["trade_payment_pillar_score"]

# group by rounded trailing avg to find ties
latest_trail["trail12_r6"] = latest_trail["trail12"].round(6)
ties = latest_trail.groupby("trail12_r6").size().reset_index(name="n").sort_values("n", ascending=False)
ties = ties[ties["n"] > 1]
ties.to_csv(OUT_TIES, index=False)
print(f"Tie groups found: {len(ties)} (saved {OUT_TIES})")
print(ties.head(10))
