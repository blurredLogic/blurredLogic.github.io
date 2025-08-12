import pandas as pd
import numpy as np

# Load the Supabase-Exported File
df = pd.read_csv('allianz_combined_data.csv', encoding='latin1')

# Clean and Prepare
df.replace('missing', np.nan, inplace=True)
df.dropna(subset=['value'], inplace=True)

# Rename for clarity and consistency
df.rename(columns={
    'country_name': 'country',
    'indicator': 'metric'
}, inplace=True)

# Convert 'date' column to datetime format
df['date'] = pd.to_datetime(df['date'])

# Pivot the Data to Have Metrics in Columns
df_wide = df.pivot_table(
    index=['date', 'country', 'country_code'],
    columns='metric',
    values='value',
    aggfunc='first'
).reset_index()

# Feature Engineering
# Extract numeric from short-term risk e.g. "3 (Sensitive)" → 3
df_wide['short_term_risk_value'] = df_wide['short_term_risk_level'].str.extract(r'(\d)').astype(float)

# Copy grade letter for clarity
df_wide['country_grade_letter'] = df_wide['country_grade']

# Reorder final columns
df_final = df_wide[[ 
    'date', 'country', 'country_code',
    'country_grade', 'country_grade_letter',
    'short_term_risk_level', 'short_term_risk_value'
]]

# Save Result
df_final.to_csv('monthly_imputed_country_data.csv', index=False)

print("Processed Allianz data saved to 'monthly_imputed_country_data.csv'")
print(df_final.head())
