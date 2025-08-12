import pandas as pd
import numpy as np
from scipy.stats import zscore

# Load Processed Allianz Data
df_long = pd.read_csv("monthly_imputed_country_data.csv")
df_long['date'] = pd.to_datetime(df_long['date'])

print("Calculating cross-sectional z-scores...")

# Clean Country Grade Letters 
df_long['country_grade_letter'] = df_long['country_grade_letter'].str.strip().str.upper()

# Map Country Grade Letters to Numeric Risk Score
grade_mapping = {
    'AAA': 1, 'AA+': 2, 'AA': 3, 'AA-': 4,
    'A+': 5, 'A': 6, 'A-': 7,
    'BBB+': 8, 'BBB': 9, 'BBB-': 10,
    'BB+': 11, 'BB': 12, 'BB-': 13,
    'B+': 14, 'B': 15, 'B-': 16,
    'CCC+': 17, 'CCC': 18, 'CCC-': 19,
    'CC': 20, 'C': 21, 'D': 22,
    'NR': np.nan, 'NA': np.nan, 'N/A': np.nan
}

df_long['country_grade_numeric'] = df_long['country_grade_letter'].map(grade_mapping)

# Z-score by Month
df_long['zscore_grade'] = df_long.groupby('date')['country_grade_numeric'].transform(zscore)
df_long['zscore_risk'] = df_long.groupby('date')['short_term_risk_value'].transform(zscore)

# Drop Missing Values (to Ensure Clean PCA Input)
df_long = df_long.dropna(subset=['country_grade_letter', 'zscore_grade', 'zscore_risk'])

# Output Z-score Summary for Sanity Check
print("\nZ-Score Summary (first few rows):")
print(df_long[['date', 'country', 'zscore_grade', 'zscore_risk']].head())

# Pivot to Wide Format for PCA
print("Reshaping data into a single wide-format file for z-scores...")
df_wide = df_long.pivot(index='date', columns='country_code', values=['zscore_grade', 'zscore_risk'])

# Save Results
df_long.to_csv("allianz_normalised_scores.csv", index=False)
df_wide.to_csv("allianz_zscore_wide.csv")

print("✔ Saved PCA-ready data to:")
print(" - allianz_normalised_scores.csv (long format)")
print(" - allianz_zscore_wide.csv (wide format)")
