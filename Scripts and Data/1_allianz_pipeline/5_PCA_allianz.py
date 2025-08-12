import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

INPUT_FILE = "allianz_normalised_scores.csv"         
MONTHLY_PILLAR_OUT = "trade_payment_pillar_score_monthly.csv"
DASHBOARD_OUT = "allianz_dashboard_data.csv"

# 1) Load
df = pd.read_csv(INPUT_FILE, parse_dates=["date"])
need = {"date","country","zscore_grade","zscore_risk"}
missing = need - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns in {INPUT_FILE}: {sorted(missing)}")

df["country"] = df["country"].astype(str)
df = df.sort_values(["country","date"])

# 2) Expand to full monthly calendar per country
min_date, max_date = df["date"].min(), df["date"].max()
all_months = pd.date_range(min_date, max_date, freq="MS")

expanded = []
for c, g in df.groupby("country", sort=False):
    g = g.set_index("date").reindex(all_months)
    g["country"] = c
    # forward-fill the two z-scores
    g["zscore_grade"] = g["zscore_grade"].ffill()
    g["zscore_risk"]  = g["zscore_risk"].ffill()
    expanded.append(g.reset_index().rename(columns={"index":"date"}))

dfm = pd.concat(expanded, ignore_index=True)
# drop rows where both still NaN (no history)
dfm = dfm.dropna(subset=["zscore_grade","zscore_risk"], how="all").sort_values(["date","country"])

# 3) Cross-sectional PCA per month (PC1 -> sign align -> re-zscore)
def pca_month(g: pd.DataFrame) -> pd.DataFrame:
    X = g[["zscore_grade","zscore_risk"]].fillna(0.0)
    Xs = StandardScaler().fit_transform(X)  # harmless if already z-scores
    pca = PCA(n_components=1)
    pc1 = pca.fit_transform(Xs).ravel()
    # sign align so higher PC1 = higher risk
    sumz = X.sum(axis=1).to_numpy()
    corr = np.corrcoef(pc1, sumz)[0,1] if (pc1.std()>0 and sumz.std()>0) else 0.0
    if corr < 0:
        pc1 = -pc1
    # re-z within month
    mu, sd = pc1.mean(), pc1.std(ddof=0)
    pc1z = (pc1 - mu) / sd if sd != 0 else np.zeros_like(pc1)
    out = g[["date","country"]].copy()
    out["trade_payment_pillar_score"] = pc1z
    return out

monthly_list = []
for dt, g in dfm.groupby("date", sort=True):
    if len(g) >= 3:  
        monthly_list.append(pca_month(g))

if not monthly_list:
    raise RuntimeError("No monthly PCA results produced. Check input coverage and dates.")

monthly = pd.concat(monthly_list, ignore_index=True).sort_values(["country","date"])
monthly.to_csv(MONTHLY_PILLAR_OUT, index=False)
print(f" Saved monthly pillar → {MONTHLY_PILLAR_OUT} (rows={len(monthly)})")

# Trailing 12-month average (strict), with safe fallback if nobody has 12 months yet
monthly["trailing_12m_avg"] = (
    monthly.groupby("country")["trade_payment_pillar_score"]
    .transform(lambda x: x.rolling(window=12, min_periods=12).mean())
)

latest = (
    monthly.dropna(subset=["trailing_12m_avg"])
    .groupby("country", as_index=False)
    .tail(1)
    .copy()
)

# If strict 12m window yields nothing (e.g., very short history), fallback to last value
if latest.empty:
    latest = monthly.groupby("country", as_index=False).tail(1).copy()
    latest["trailing_12m_avg"] = latest["trade_payment_pillar_score"]

# Min–Max 0–100 scaling on latest trailing averages
mn, mx = latest["trailing_12m_avg"].min(), latest["trailing_12m_avg"].max()
if mx > mn:
    latest["risk_score_0_to_100"] = (latest["trailing_12m_avg"] - mn) / (mx - mn) * 100.0
else:
    latest["risk_score_0_to_100"] = 50.0  # degenerate case when all equal

dashboard = latest[["country","trailing_12m_avg","risk_score_0_to_100"]].rename(
    columns={"trailing_12m_avg":"overall_risk_score"}
)
dashboard.to_csv(DASHBOARD_OUT, index=False)
print(f" Saved dashboard → {DASHBOARD_OUT} (rows={len(dashboard)})")

# view top 5 for sanity
print("\nTop 5 (dashboard 0–100):")
print(dashboard.sort_values("risk_score_0_to_100", ascending=False).head())
