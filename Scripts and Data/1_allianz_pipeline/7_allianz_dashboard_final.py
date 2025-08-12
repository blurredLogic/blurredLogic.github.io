import pandas as pd
import numpy as np

INPUT_FILE   = "allianz_pca_monthly.csv"
DASHBOARD_OUT = "allianz_dashboard_data.csv"

df = pd.read_csv(INPUT_FILE, parse_dates=["date"])
df = df.sort_values(["country","date"])

# strict trailing 12-month average
df["trail12"] = df.groupby("country")["trade_payment_pillar_score"].transform(
    lambda x: x.rolling(window=12, min_periods=12).mean()
)

# If nobody has 12 months yet, fallback to last value
latest = df.dropna(subset=["trail12"]).groupby("country").tail(1).copy()
if latest.empty:
    latest = df.groupby("country").tail(1).copy()
    latest["trail12"] = latest["trade_payment_pillar_score"]

# 0–100 scaling on latest month (strict 12m)
mn, mx = latest["trail12"].min(), latest["trail12"].max()
latest["risk_score_0_to_100"] = 50.0 if mx == mn else (latest["trail12"] - mn) / (mx - mn) * 100

# Merge both views
dash = latest[["country","trail12","risk_score_0_to_100"]].merge(
    latest_ewm, on="country", how="left"
).rename(columns={
    "trail12":"overall_risk_score", 
    "risk_score_0_to_100":"risk_score_0_to_100"
})

dash.to_csv(DASHBOARD_OUT, index=False)
print(f"Saved dashboard {DASHBOARD_OUT} (rows={len(dash)})")
print("\nTop 5 (strict 12m 0–100):")
print(dash.sort_values("risk_score_0_to_100", ascending=False).head())
