import os
import pandas as pd

# import datasets
IMF_DIR  = r"C:\Users\cdye\OneDrive\Desktop\GE Heathcare\Data\IMF Data\economic_pillar_pipeline"
OUT_DIR  = r"C:\Users\cdye\OneDrive\Desktop\GE Heathcare\Data\IMF Data\economic_pillar_pipeline"
os.makedirs(OUT_DIR, exist_ok=True)

# IMF files
files = {
    "GDP_growth":   "IMF_gross_domestic_product_constant_prices_percent_change_wide.csv",
    "CAB_pct_GDP":  "IMF_current_account_balance_percent_of_gdp_wide.csv",
    "Rev_pct_GDP":  "IMF_general_government_revenue_percent_of_gdp_wide.csv",
    "Inflation_Avg":"IMF_inflation_average_consumer_prices_index_wide.csv",
    "Inflation_EOP":"IMF_inflation_end_of_period_consumer_prices_index_wide.csv",
    "Debt":         "IMF_general_government_gross_debt_national_currency_wide.csv",
    "Spend_pct_GDP":"IMF_general_government_total_expenditure_percent_of_gdp_wide.csv",
    "Unemployment": "IMF_unemployment_rate_percent_of_total_labor_force_wide.csv"
}

# Which to invert so that higher always equals higher risk
invert = {"GDP_growth", "CAB_pct_GDP", "Rev_pct_GDP"}

# Monthly index
monthly_idx = pd.date_range("2006-01-01", "2025-05-01", freq="MS")

for name, fname in files.items():
    # Load annual panel
    path_in = os.path.join(IMF_DIR, fname)
    df = pd.read_csv(path_in, index_col=0)

    # Robust index parsing
    idx = df.index.astype(str)
    if idx.str.match(r"^\d{4}$").all():
        # pure “YYYY”
        df.index = pd.to_datetime(idx + "-12-31", format="%Y-%m-%d")
    else:
        # already full dates, let pandas infer
        df.index = pd.to_datetime(idx, infer_datetime_format=True)

    df.index.name = 'date'

    # Upsample to monthly and forward-fill
    df_m = df.reindex(monthly_idx).asfreq('MS').ffill()

    # Invert where needed
    if name in invert:
        df_m = -df_m

    # Cross-sectional z-score
    mu    = df_m.mean(axis=1)
    sigma = df_m.std(axis=1, ddof=0)
    df_z  = df_m.sub(mu, axis=0).div(sigma, axis=0)
    df_z.columns = [f"{c}_z" for c in df_z.columns]

    # Save normalized IMF panel
    out_path = os.path.join(OUT_DIR, f"{name}_monthly_z.csv")
    df_z.to_csv(out_path, index_label='date')
    print(f"{name}: upsampled & normalized {out_path}")
