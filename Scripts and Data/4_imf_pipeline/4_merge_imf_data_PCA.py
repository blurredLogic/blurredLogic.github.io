import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# imoport datasets
in_dir = r"C:\Users\cdye\OneDrive\Desktop\GE Heathcare\Data\IMF Data\economic_pillar_pipeline"
output_file_monthly = "imf_economic_pillar_score_monthly.csv"

INDICATOR_FILES = {
    "CAB": "CAB_pct_GDP_monthly_z.csv", "Debt": "Debt_monthly_z.csv",
    "GDP_growth": "GDP_growth_monthly_z.csv", "Inflation_Avg": "Inflation_Avg_monthly_z.csv",
    "Inflation_EOP": "Inflation_EOP_monthly_z.csv", "Revenue": "Rev_pct_GDP_monthly_z.csv",
    "Spending": "Spend_pct_GDP_monthly_z.csv", "Unemployment": "Unemployment_monthly_z.csv"
}
MIN_INDICATORS_REQUIRED = 5

# load clean and merge
print("Step 1: Loading, Cleaning, and Merging Data")
all_dfs = []
for short_name, filename in INDICATOR_FILES.items():
    full_path = os.path.join(in_dir, filename)
    try:
        df = pd.read_csv(full_path, index_col='date', parse_dates=True)
        df.columns = [f"{col.replace('_z', '')}_{short_name}_z" for col in df.columns]
        all_dfs.append(df)
        print(f"Loaded and renamed: {filename}")
    except FileNotFoundError:
        print(f"  - Warning: File '{filename}' was not found at {full_path}. Skipping.")
master_df = pd.concat(all_dfs, axis=1, join='outer')
print("All files merged successfully.")

# reshape and filter
print("Step 2: Reshaping and Filtering Data")
df_long = master_df.reset_index().melt(id_vars='date', var_name='country_indicator', value_name='z_score')
df_long[['country', 'indicator', 'z']] = df_long['country_indicator'].str.rsplit('_', n=2, expand=True)

print(f"Filtering: Keeping only country-months with at least {MIN_INDICATORS_REQUIRED} indicators...")
indicator_counts = df_long.groupby(['date', 'country'])['z_score'].count()
valid_indices = indicator_counts[indicator_counts >= MIN_INDICATORS_REQUIRED].index
df_filtered = df_long[df_long.set_index(['date', 'country']).index.isin(valid_indices)]

# pivot and prepare for PCA
df_pivot = df_filtered.pivot_table(index=['date', 'country'], columns='indicator', values='z_score')
df_pivot.fillna(0, inplace=True)
print("✔ Data filtered and prepared for PCA.")

# cross sectional PCA
print("Step 3: Running PCA on the filtered data")
results_list = []
explained_variances = []

for date, group in df_pivot.groupby(level='date'):
    if len(group) < 2: continue
    features = group.columns.tolist()
    if len(features) < 2: continue
    
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(group)
    pca = PCA(n_components=1)
    pillar_score = pca.fit_transform(x_scaled)

    # Sign check
    correlation = np.corrcoef(group.sum(axis=1), pillar_score.flatten())[0, 1]
    if correlation < 0:
        pillar_score *= -1

    explained_var = pca.explained_variance_ratio_[0]
    explained_variances.append({'date': date, 'explained_variance': explained_var})

    result_df = pd.DataFrame({
        'date': date,
        'country': group.index.get_level_values('country'),
        'imf_economic_pillar_score': pillar_score.flatten()
    })
    results_list.append(result_df)

if not results_list:
    print("!! FATAL ERROR: No valid monthly results were generated.")
    exit()

df_with_pca = pd.concat(results_list, ignore_index=True)

# Re-z-score final pillar scores
df_with_pca['imf_economic_pillar_score'] = df_with_pca.groupby('date')['imf_economic_pillar_score'].transform(
    lambda x: (x - x.mean()) / x.std()
)
print("✔ Monthly pillar scores calculated.")

# save PCA for score data
df_with_pca.to_csv(output_file_monthly, index=False)
print(f"Final IMF Economic Pillar score saved to: {output_file_monthly}")

# save explained variance log
pd.DataFrame(explained_variances).to_csv("explained_variance_by_month.csv", index=False)
print("✔ Saved explained variance log to explained_variance_by_month.csv")

# snapshot for loadings
print("PCA Loadings & Top/Bottom Country Scores: 2024-01-01")
sample_date = pd.to_datetime("2024-01-01")

if sample_date in df_pivot.index.get_level_values("date"):
    sample_group = df_pivot.loc[sample_date]
    if len(sample_group) >= 2:
        x_scaled = StandardScaler().fit_transform(sample_group)
        pca = PCA(n_components=1)
        pca.fit(x_scaled)

        # Factor loadings
        loadings = pca.components_[0]
        if np.sum(loadings) < 0:
            loadings *= -1
        loadings_df = pd.DataFrame({
            'Indicator': sample_group.columns,
            'Loading': loadings
        }).sort_values('Loading', key=abs, ascending=False)
        loadings_df.to_csv("pca_loadings_2024_01.csv", index=False)
        print("Saved PCA loadings to pca_loadings_2024_01.csv")

        # Top/bottom 5 country scores
        snapshot = df_with_pca[df_with_pca['date'] == sample_date]
        top5 = snapshot.nlargest(5, 'imf_economic_pillar_score')
        bottom5 = snapshot.nsmallest(5, 'imf_economic_pillar_score')
        pd.concat([top5, bottom5]).to_csv("pillar_score_snapshot_2024_01.csv", index=False)
        print("Saved top/bottom country snapshot to pillar_score_snapshot_2024_01.csv")

print("\nAll steps complete.")
