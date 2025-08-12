import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer  
from sklearn.impute import IterativeImputer


# Load required dataset
input_path = r"C:\Users\cdye\OneDrive\Desktop\GE Heathcare\Data\Sovereign Ratings\Sovereign Scripts\sovereign_ratings_final_mapped_1.csv"
df = pd.read_csv(input_path)

# Define constants
rating_cols = ['Moody_num', 'S&P_num', 'Fitch_num']
years = list(range(2006, 2025))  # force full 2006–2024 span

# Initialize MICE imputer
imp = IterativeImputer(max_iter=10, random_state=0, sample_posterior=False)

out = []
for (iso, country), grp in df.groupby(['Alpha3','Country']):
    # Force full-year index
    panel = grp.set_index('Year')[rating_cols].reindex(years)

    # Linear interpolate small gaps (≤2 years)
    panel = panel.infer_objects().interpolate(method='linear', limit=2, limit_area='inside')

    # Truncate leading NaNs (start at first real value)
    firsts = panel.apply(lambda s: s.first_valid_index())
    for col in panel:
        panel[col] = panel[col].loc[firsts[col]:]

    # MICE impute across the three series
    non_empty = panel.columns[panel.notna().any()].tolist()
    if len(non_empty) >= 2:
        imputed_part = imp.fit_transform(panel[non_empty])
        imputed_df = pd.DataFrame(imputed_part, index=panel.index, columns=non_empty)
        panel.loc[:, non_empty] = imputed_df

    # Reattach metadata
    panel['Alpha3'] = iso
    panel['Country'] = country
    panel = panel.reset_index().rename(columns={'index': 'Year'})
    out.append(panel)

df_imputed = pd.concat(out, ignore_index=True)

# Save the fully-imputed panel
output_path = r"C:\Users\cdye\OneDrive\Desktop\GE Heathcare\Data\Sovereign Ratings\Sovereign Scripts\sovereign_ratings_mice_imputed.csv"
df_imputed.to_csv(output_path, index=False)
print(f"Saved interpolated + MICE-imputed panel with forced years to {output_path}")
