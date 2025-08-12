import pandas as pd

# Load raw data
df = pd.read_csv(r'C:\Users\cdye\OneDrive\Desktop\GE Heathcare\Data\Sovereign Ratings\Sovereign Scripts\sovereign_ratings_final.csv')

# Standardize agency names
df['Agency'] = (
    df['Agency']
      .str.replace("’", "'", regex=False)    # curly -> straight
      .replace({"Moody's": "Moody",            # straight -> Moody
                "Moody’s": "Moody"})          # catch any leftover
)

# Parse date -> Year
df['Date'] = pd.to_datetime(
    df['Long term Rating Foreign currency Date'],
    dayfirst=True, errors='coerce'
)
df['Year'] = df['Date'].dt.year

# Extract rating code
df['RatingCode'] = df['Long term Rating Foreign currency Rating(Outlook)'] \
    .str.extract(r'^([A-Za-z0-9\+\-]+)')

# Pivot to one row per (Alpha3,Country,Year)
df_panel = (
    df[['Alpha3','Country','Year','Agency','RatingCode']]
      .dropna(subset=['Year','RatingCode'])
      .pivot_table(
         index=['Alpha3','Country','Year'],
         columns='Agency',
         values='RatingCode',
         aggfunc='last'
      )
      .reset_index()
)

# Map letter‐grade -> numeric
moody_scale    = ['Aaa','Aa1','Aa2','Aa3','A1','A2','A3',
                  'Baa1','Baa2','Baa3','Ba1','Ba2','Ba3',
                  'B1','B2','B3','Caa1','Caa2','Caa3','Ca','C']
fitch_sp_scale = ['AAA','AA+','AA','AA-','A+','A','A-','BBB+','BBB','BBB-',
                  'BB+','BB','BB-','B+','B','B-','CCC+','CCC','CCC-','CC','C','RD','D']
moody_map    = {c:i+1 for i,c in enumerate(moody_scale)}
fitch_sp_map = {c:i+1 for i,c in enumerate(fitch_sp_scale)}

for ag in ["Moody","S&P","Fitch"]:
    if ag in df_panel.columns:
        m = moody_map if ag=="Moody" else fitch_sp_map
        df_panel[f"{ag}_num"] = df_panel[ag].map(m).astype(float)

# Save intermediate
df_panel.to_csv(r'C:\Users\cdye\OneDrive\Desktop\GE Heathcare\Data\Sovereign Ratings\Sovereign Scripts\sovereign_ratings_final_mapped_1.csv', index=False)
print("Mapped saved to sovereign_ratings_final_mapped.csv")
