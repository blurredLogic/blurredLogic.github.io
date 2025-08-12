import pandas as pd


# 1) Load the fully‐imputed dataset
input_path = r"C:\Users\cdye\OneDrive\Desktop\GE Heathcare\Data\Sovereign Ratings\Sovereign Scripts\sovereign_ratings_mice_imputed.csv"
df = pd.read_csv(input_path)

# 2) Melt to long form
long = df.melt(
    id_vars=['Alpha3','Country','Year'],
    value_vars=['Moody_num','S&P_num','Fitch_num'],
    var_name='Agency',
    value_name='Value'
)
# Drop the "_num" suffix so Agency = "Moody","S&P","Fitch"
long['Agency'] = long['Agency'].str.replace('_num','', regex=False)

# 3) Define conflict list
conflicts = {'SYR','UKR','IRQ','AFG','RUS'}

# 4) Per‐series keep‐test
def keep_series(group):
    iso, ag = group.name  # tuple (Alpha3, Agency)
    # 70% coverage over last 10 years (2015–2024)
    rec = group[group.Year.between(2015,2024)]
    rec_cov = rec.Value.notna().mean()
    # 70% coverage overall (2006–2024)
    ovl = group[group.Year.between(2006,2024)]
    ovl_cov = ovl.Value.notna().mean()
    # Conflict rule: if conflict country, drop if >15% missing
    miss = group.Value.isna().mean() if iso in conflicts else 0
    return (rec_cov >= 0.70) and (ovl_cov >= 0.70) and (miss <= 0.15)

# Apply to each (Alpha3, Agency)
keep_mask = (
    long
      .groupby(['Alpha3','Agency'])
      .apply(keep_series)
      .reset_index(name='keep')
)

# 5) Filter & pivot back to wide
filtered = (
    long
      .merge(
          keep_mask.loc[keep_mask.keep, ['Alpha3','Agency']],
          on=['Alpha3','Agency']
      )
      .pivot_table(
          index=['Alpha3','Country','Year'],
          columns='Agency',
          values='Value'
      )
      .reset_index()
)

# 6) Save the coverage‐filtered panel
output_path = r"C:\Users\cdye\OneDrive\Desktop\GE Heathcare\Data\Sovereign Ratings\Sovereign Scripts\sovereign_ratings_filtered.csv"
filtered.to_csv(output_path, index=False)
print("Saved coverage‐filtered panel to 03_filtered.csv")
