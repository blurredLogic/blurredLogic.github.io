import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime

# Load the Interpolated Annual Ratings
input_path = "sovereign_ratings_filtered_interpolated.csv"
df = pd.read_csv(input_path)

# Melt to Long Format
long = df.melt(
    id_vars=['Alpha3', 'Country', 'Year'],
    value_vars=['Moody', 'S&P', 'Fitch'],
    var_name='Agency',
    value_name='Value'
)

# Forecast 2025 via ARIMA(1,1,1) for Series With Enough History
def forecast_one(series):
    series = series.dropna()
    if len(series) < 10:
        return np.nan
    try:
        model = ARIMA(series, order=(1, 1, 1)).fit()
        return model.forecast(1).iloc[0]
    except Exception:
        return np.nan

records = []
for (iso, country, agency), group in long.groupby(['Alpha3', 'Country', 'Agency']):
    ts = group.set_index('Year')['Value'].sort_index()
    forecast = forecast_one(ts)
    if not np.isnan(forecast):
        records.append({
            'Alpha3': iso,
            'Country': country,
            'Year': 2025,
            'Agency': agency,
            'Value': forecast
        })

df_2025 = pd.DataFrame(records)

# Combine Forecast with Historical Data
long_ext = pd.concat([long, df_2025], ignore_index=True)

#  Convert to Monthly Format
# Add 'date' column as Jan 1 of each year
long_ext['date'] = pd.to_datetime(long_ext['Year'].astype(str) + "-01-01")

# Forward fill each year to monthly rows
monthly_records = []
for (iso, country, agency), group in long_ext.groupby(['Alpha3', 'Country', 'Agency']):
    group = group.sort_values('date').set_index('date')
    monthly_index = pd.date_range(start="2006-01-01", end="2025-05-01", freq='MS')
    group_monthly = group[['Value']].reindex(monthly_index).ffill()
    group_monthly['Alpha3'] = iso
    group_monthly['Country'] = country
    group_monthly['Agency'] = agency
    monthly_records.append(group_monthly.reset_index())

df_monthly = pd.concat(monthly_records, ignore_index=True)
df_monthly.rename(columns={'index': 'date'}, inplace=True)

# Pivot to Wide Format (Moody, S&P, Fitch as columns)
df_pivot = df_monthly.pivot_table(
    index=['date', 'Country', 'Alpha3'],
    columns='Agency',
    values='Value'
).reset_index()

# Cross-Sectional Z-Score Per Month
print("Calculating monthly cross-sectional z-scores...")
df_pivot['date'] = pd.to_datetime(df_pivot['date'])

# Create z-score columns for each agency
for agency in ['Moody', 'S&P', 'Fitch']:
    df_pivot[f'{agency}_z'] = (
        df_pivot.groupby('date')[agency]
        .transform(lambda x: (x - x.mean()) / x.std(ddof=0))
    )

# Save Final dataset for PCA 
output_path = "sovereign_monthly_z.csv"
df_pivot.to_csv(output_path, index=False)
print(f"Saved monthly panel with z-scores to: {output_path}")

# Summary for Sanity Check
print("Sample:")
print(df_pivot[['date', 'Country', 'Moody_z', 'S&P_z', 'Fitch_z']].dropna().head())
