import os
import pandas as pd
import numpy as np
import supabase
from supabase import create_client, Client


# Enter appropriate Supabase Credentials
SUPABASE_URL = "https://bwmtbpfkylxvyvsdepvl.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImJ3bXRicGZreWx4dnl2c2RlcHZsIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc0ODk2MTQxNCwiZXhwIjoyMDY0NTM3NDE0fQ.8q_3dYYPsoJVOXWhT-61LXA-RYnET--sPKkUQi9WMe4"

# Define the location of the data file
WEO_FILE_PATH = "imf_weo_data.csv"

# List iof required Indicators
INDICATORS_TO_EXTRACT = [
    {'descriptor': 'Gross domestic product, constant prices', 'units': 'Percent change', 'db_col': 'gdp_growth_pct'},
    {'descriptor': 'Gross domestic product, current prices', 'units': 'U.S. dollars', 'db_col': 'nominal_gdp_usd_billions'},
    {'descriptor': 'Gross domestic product, current prices', 'units': 'Purchasing power parity; international dollars', 'db_col': 'gdp_ppp_billions'},
    {'descriptor': 'Inflation, average consumer prices', 'units': 'Index', 'db_col': 'inflation_avg_index'},
    {'descriptor': 'Inflation, end of period consumer prices', 'units': 'Index', 'db_col': 'inflation_eop_index'},
    {'descriptor': 'General government gross debt', 'units': 'National currency', 'db_col': 'govt_debt_nat_currency_billions'},
    {'descriptor': 'General government revenue', 'units': 'Percent of GDP', 'db_col': 'govt_revenue_pct_gdp'},
    {'descriptor': 'General government total expenditure', 'units': 'Percent of GDP', 'db_col': 'govt_spending_pct_gdp'},
    {'descriptor': 'Current account balance', 'units': 'Percent of GDP', 'db_col': 'current_account_pct_gdp'},
    {'descriptor': 'Unemployment rate', 'units': 'Percent of total labor force', 'db_col': 'unemployment_rate_pct'}
]

def get_supabase_client() -> Client:
    """Initializes and returns the Supabase client."""
    print("Connecting to Supabase...")
    return create_client(SUPABASE_URL, SUPABASE_KEY)

def process_weo_data() -> pd.DataFrame:
    """
    Loads WEO data, filters for specific indicators, and transforms the data.
    """
    print(f"Reading WEO data from '{WEO_FILE_PATH}'...")

    if not os.path.exists(WEO_FILE_PATH):
        print(f"ERROR: Data file not found at '{WEO_FILE_PATH}'.")
        return pd.DataFrame()

    try:
        df = pd.read_csv(WEO_FILE_PATH, thousands=',', encoding='latin-1')
    except Exception as e:
        print(f"ERROR: Failed to read CSV file. Error: {e}")
        return pd.DataFrame()

    df.columns = df.columns.str.strip()
    df['Subject Descriptor'] = df['Subject Descriptor'].str.strip()
    df['Units'] = df['Units'].str.strip()
    print("Successfully read and cleaned file.\n")
    
    all_indicators_df_list = []
    for indicator in INDICATORS_TO_EXTRACT:
        df_filtered = df[
            (df['Subject Descriptor'] == indicator['descriptor']) & 
            (df['Units'] == indicator['units'])
        ].copy()
        
        if not df_filtered.empty:
            df_filtered['indicator_code'] = indicator['db_col']
            all_indicators_df_list.append(df_filtered)

    if not all_indicators_df_list:
        print("ERROR: Could not find any of the specified indicators.")
        return pd.DataFrame()

    df_combined = pd.concat(all_indicators_df_list, ignore_index=True)
    df_combined.rename(columns={'ISO': 'country_code'}, inplace=True)
    
    year_columns = [col for col in df_combined.columns if str(col).isnumeric()]
    df_long = df_combined.melt(
        id_vars=['country_code', 'indicator_code'],
        value_vars=year_columns,
        var_name='period',
        value_name='value'
    )

    df_long.dropna(subset=['value'], inplace=True)
    df_long = df_long[pd.to_numeric(df_long['value'], errors='coerce').notna()]
    df_long['value'] = pd.to_numeric(df_long['value'])
    df_long['period'] = df_long['period'].astype(str)

    df_wide = df_long.pivot_table(
        index=['country_code', 'period'],
        columns='indicator_code',
        values='value'
    ).reset_index()
    

    codes_to_exclude = ['TWN', 'UVK', 'WBG']
    original_rows = len(df_wide)
    df_wide = df_wide[~df_wide['country_code'].isin(codes_to_exclude)]
    print(f"Successfully processed data. {original_rows - len(df_wide)} rows for problematic country codes were excluded.")
    print(f"Final row count for upload: {len(df_wide)}")
    
    # Replace pandas' Not-a-Number (NaN) with Python's None, which becomes 'null' in JSON.
    df_for_upload = df_wide.replace({np.nan: None})

    return df_for_upload

def main():
    """Main function to run the full data pipeline."""
    
    weo_data_to_upload = process_weo_data()
    
    if weo_data_to_upload.empty:
        print("No data processed. Exiting.")
        return

    data_dict = weo_data_to_upload.to_dict(orient='records')
    
    supabase = get_supabase_client()
    table_name = 'imf_weo_data_final'
    print(f"Upserting {len(data_dict)} records to '{table_name}' table. This may take a moment...")
    
    try:
        supabase.table(table_name).upsert(data_dict).execute()
        print(f"SUCCESS: Data has been uploaded to '{table_name}'!")
    except Exception as e:
        print(f"ERROR during Supabase upsert: {e}")

if __name__ == "__main__":
    main()