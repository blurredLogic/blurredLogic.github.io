"""
Sovereign Risk Pillar (PCA)
- Input: monthly panel with z-scores per agency (e.g., 'Moody_z','S&P_z','Fitch_z')
- Method: cross-sectional PCA per month; keep rows with >=2/3 indicators; mean-impute 0 (z-scored);
          sign-align PC1 (higher = higher risk via corr with sum of z's);
          re-normalise PC1 per month to z-scores for pillar comparability.
- Outputs:
  1) sovereign_pillar_scores.csv         (date, Country, Alpha3, pc1_raw, pc1, pc1_rank)
  2) sovereign_pca_variance.csv          (date, pc1_explained_variance_ratio)
  3) sovereign_pca_loadings.csv          (date, loadings for PC1 on each input column)
"""

import sys
import argparse
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

def run_pca_for_month(g: pd.DataFrame, z_cols: list[str]) -> tuple[pd.DataFrame, float, pd.DataFrame]:
    """
    Run PCA for a single month (cross-section across countries).
    Returns:
      scores_df: columns [date, Country, Alpha3, pc1_raw, pc1, pc1_rank]
      evr: explained variance ratio for PC1
      loadings_df: one row with PC1 loadings for each column in z_cols
    """
    # Keep rows with at least 2 non-null values among the z-score columns
    valid_mask = g[z_cols].notna().sum(axis=1) >= 2
    m = g.loc[valid_mask].copy()
    if m.shape[0] < 3:
        # Not enough rows to do a meaningful PCA
        return (pd.DataFrame(columns=["date","Country","Alpha3","pc1_raw","pc1","pc1_rank"]),
                np.nan,
                pd.DataFrame(columns=z_cols))

    # Mean-impute remaining gaps with 0, valid because inputs are z-scored
    X = m[z_cols].fillna(0.0).to_numpy()

    pca = PCA(n_components=1, svd_solver="full")
    pc1_raw = pca.fit_transform(X).ravel()

    # Sign alignment: correlate with sum of z's
    sum_z = m[z_cols].sum(axis=1).to_numpy()
    corr = np.corrcoef(pc1_raw, sum_z)[0, 1] if np.std(pc1_raw) > 0 and np.std(sum_z) > 0 else 0.0
    flipped = corr < 0
    if flipped:
        pc1_raw = -pc1_raw
        components = -pca.components_.copy()
    else:
        components = pca.components_.copy()

    # Monthly re-normalisation of PC1 to z-score
    mean_ = pc1_raw.mean()
    std_  = pc1_raw.std(ddof=0)
    pc1 = np.zeros_like(pc1_raw) if std_ == 0 else (pc1_raw - mean_) / std_

    # Rank within month: 1 = highest risk (largest pc1)
    # argsort twice trick for ranking; invert sign for descending
    order = (-pc1).argsort().argsort() + 1

    scores = m[["date","Country","Alpha3"]].copy()
    scores["pc1_raw"] = pc1_raw
    scores["pc1"] = pc1
    scores["pc1_rank"] = order

    evr = float(pca.explained_variance_ratio_[0])
    loadings = pd.Series(components[0], index=z_cols, name="loading").to_frame().T

    return scores, evr, loadings

def main():
    ap = argparse.ArgumentParser(description="Sovereign PCA pillar builder")
    ap.add_argument("--input",  default="sovereign_monthly_z.csv",
                    help="Path to monthly z-score panel (from ARIMA pipeline)")
    ap.add_argument("--out_scores",  default="sovereign_pillar_scores.csv",
                    help="Output CSV for per-country monthly scores")
    ap.add_argument("--out_variance", default="sovereign_pca_variance.csv",
                    help="Output CSV for per-month explained variance")
    ap.add_argument("--out_loadings", default="sovereign_pca_loadings.csv",
                    help="Output CSV for per-month PC1 loadings")
    args = ap.parse_args()

    # Load input
    df = pd.read_csv(args.input, parse_dates=["date"])

    # Detect which z-score columns exist among the expected set
    expected = ["Moody_z", "S&P_z", "Fitch_z"]
    z_cols = [c for c in expected if c in df.columns]
    if len(z_cols) < 2:
        raise ValueError(f"Need at least 2 z-score columns present among {expected}. Found: {z_cols}")

    # Sort for stable grouping
    df = df.sort_values(["date", "Country", "Alpha3"]).reset_index(drop=True)

    # Run monthly PCA
    scores_all, evr_rows, loadings_rows = [], [], []
    for dt, g in df.groupby("date", sort=True):
        scores, evr, loadings = run_pca_for_month(g, z_cols=z_cols)
        if not scores.empty:
            scores_all.append(scores)
            evr_rows.append({"date": dt, "pc1_explained_variance_ratio": evr})
            lrow = loadings.copy()
            lrow["date"] = dt
            loadings_rows.append(lrow)

    # Concatenate results
    scores_all = pd.concat(scores_all, ignore_index=True) if scores_all else pd.DataFrame(
        columns=["date","Country","Alpha3","pc1_raw","pc1","pc1_rank"]
    )
    evr_df = pd.DataFrame(evr_rows, columns=["date","pc1_explained_variance_ratio"]).sort_values("date")
    loadings_df = (pd.concat(loadings_rows, ignore_index=True).sort_values("date")
                   if loadings_rows else pd.DataFrame(columns=["date"] + z_cols))

    # Save
    scores_all.to_csv(args.out_scores, index=False)
    evr_df.to_csv(args.out_variance, index=False)
    loadings_df.to_csv(args.out_loadings, index=False)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Error:", e, file=sys.stderr)
        sys.exit(1)
