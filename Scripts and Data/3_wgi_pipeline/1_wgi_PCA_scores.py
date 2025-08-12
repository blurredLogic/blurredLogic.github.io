import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os

# Load Dataset
in_dir = r'C:\Users\cdye\OneDrive\Desktop\GE Heathcare\Data\WGI Data'
output_file_monthly = "wgi_governance_pillar_score_monthly.csv"

# This dictionary maps a short name to each of the normalized data files
INDICATOR_FILES = {
    "CC": "WGI_cc_wide_Z_Score_Full.csv",
    "GE": "WGI_ge_wide_Z_Score_Full.csv",
    "PV": "WGI_pv_wide_Z_Score_Full.csv",
    "RL": "WGI_rl_wide_Z_Score_Full.csv",
    "RQ": "WGI_rq_wide_Z_Score_Full.csv",
    "VA": "WGI_va_wide_Z_Score_Full.csv"
}

# load clean and merge datasets
print("--- Step 1: Loading, Cleaning, and Merging WGI Data ---")
all_dfs = []
for short_name, filename in INDICATOR_FILES.items():
    full_path = os.path.join(in_dir, filename)
    try:
        df = pd.read_csv(full_path, index_col='date', parse_dates=True)
        # Add a suffix to each country column to identify the indicator
        df.columns = [f"{col.replace('_z', '')}_{short_name}_z" for col in df.columns]
        all_dfs.append(df)
        print(f"Loaded and renamed: {filename}")
    except FileNotFoundError:
        print(f"  - Warning: File '{filename}' was not found at {full_path}. Skipping.")

# Merge all dataframes on their common date index
master_df = pd.concat(all_dfs, axis=1, join='outer')
print("All files merged successfully.")

# reshape and prepare for PCA
print("Step 2: Reshaping and Preparing Data for PCA")
df_long = master_df.reset_index().melt(id_vars='date', var_name='country_indicator', value_name='z_score')
df_long[['country', 'indicator', 'z']] = df_long['country_indicator'].str.rsplit('_', n=2, expand=True)
df_pivot = df_long.pivot_table(index=['date', 'country'], columns='indicator', values='z_score')
df_pivot.fillna(0, inplace=True) # Fill any missing values with the mean (0)
print("Data reshaped and prepared for PCA.")

# RUN CROSS-SECTIONAL PCA
print("Step 3: Running PCA on the filtered data")
results_list = []
for date, group in df_pivot.groupby(level='date'):
    if len(group) < 2: continue
    
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(group)
    pca = PCA(n_components=1)
    pillar_score = pca.fit_transform(x_scaled)

    # IMPORTANT: For WGI, a higher score is better
    # align the score so that a higher final score means higher risk.
    correlation = np.corrcoef(group.sum(axis=1), pillar_score.flatten())[0, 1]
    if correlation > 0: # If the correlation is positive, flip the sign
        pillar_score *= -1
    
    result_df = pd.DataFrame({
        'date': date,
        'country': group.index.get_level_values('country'),
        'governance_risk_pillar_score': pillar_score.flatten()
    })
    results_list.append(result_df)

if not results_list:
    print("!! FATAL ERROR: No valid monthly results were generated.!!")
    exit()

df_with_pca = pd.concat(results_list, ignore_index=True)

# Re-z-score the final pillar score for interpretability
df_with_pca['governance_risk_pillar_score'] = df_with_pca.groupby('date')['governance_risk_pillar_score'].transform(
    lambda x: (x - x.mean()) / x.std()
)
print("Monthly pillar scores calculated.")

# SAVE THE FINAL DATA
df_with_pca.to_csv(output_file_monthly, index=False)
print(f"Process complete! Final Governance Pillar score saved to: {output_file_monthly}")

# PCA ANALYSIS FOR REPORT
print("\n" + "="*50)
print("PCA Analysis for Report (Sample Date: 2022-01-01)")
try:
    # WGI data is often yearly, so choosing a year start date
    sample_date = pd.to_datetime('2022-01-01')
    data_sample = df_pivot.loc[sample_date]

    features = data_sample.columns.tolist()
    
    if len(data_sample) > 1 and len(features) > 1:
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(data_sample)
        pca = PCA(n_components=1)
        pca.fit(x_scaled)

        # Explained Variance
        explained_variance_ratio = pca.explained_variance_ratio_[0]
        print(f"Explained Variance:")
        print(f"The first principal component captures {explained_variance_ratio:.1%} of the total variance.")

        # Factor Loadings
        loadings = pca.components_[0]
        # Align sign for interpretability (so higher loading = higher contribution to risk)
        if np.sum(loadings) > 0:
            loadings *= -1
            
        print(f"Factor Loadings (Top Drivers):")
        print("Shows which indicators most strongly influence the pillar score.")
        
        loadings_df = pd.DataFrame({'Indicator': features, 'Loading': loadings})
        loadings_df['Abs_Loading'] = loadings_df['Loading'].abs()
        print(loadings_df.sort_values('Abs_Loading', ascending=False).drop('Abs_Loading', axis=1))

except KeyError:
    print(f"\nCould not perform analysis: No data found for the sample date '2022-01-01'.")
except Exception as e:
    print(f"\nAn error occurred during PCA analysis: {e}")