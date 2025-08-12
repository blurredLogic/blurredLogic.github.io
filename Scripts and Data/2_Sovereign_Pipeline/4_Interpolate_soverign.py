import pandas as pd

# Load the coverage-filtered dataset
input_path = r"C:\Users\cdye\OneDrive\Desktop\GE Heathcare\Data\Sovereign Ratings\Sovereign Scripts\sovereign_ratings_filtered.csv"
df = pd.read_csv(input_path)

# Define the full year range
years = list(range(2006, 2025))  # 2006 through 2024


# Interpolate ≤2-year gaps & truncate leading NaNs
records = []

# Loop over each country
for (iso, country), grp in df.groupby(['Alpha3', 'Country']):
    # Reindex so we have one row per year
    grp2 = grp.set_index('Year').reindex(years)
    
    # Interpolate interior gaps up to 2 years
    grp_interp = grp2.interpolate(method='linear', limit=2, limit_area='inside')
    
    # Truncate any leading NaNs (start at first real data)
    first_valid = grp_interp.apply(lambda s: s.first_valid_index())
    for col in grp_interp.columns:
        grp_interp[col] = grp_interp[col].loc[first_valid[col]:]
    
    # Restore metadata columns
    grp_interp['Alpha3'] = iso
    grp_interp['Country'] = country
    grp_interp = grp_interp.reset_index().rename(columns={'index': 'Year'})
    
    records.append(grp_interp)

# Combine all countries back into one DataFrame
df_interp = pd.concat(records, ignore_index=True)


# Save the interpolated & truncated series

output_path = r"C:\Users\cdye\OneDrive\Desktop\GE Heathcare\Data\Sovereign Ratings\Sovereign Scripts\sovereign_ratings_filtered_interpolated.csv"
df_interp.to_csv(output_path, index=False)
print(f"Saved interpolated panel to {output_path}")
