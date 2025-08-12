import pandas as pd
from supabase import create_client, Client
import os

# configuration
SUPABASE_URL = "https://bwmtbpfkylxvyvsdepvl.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImJ3bXRicGZreWx4dnl2c2RlcHZsIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc0ODk2MTQxNCwiZXhwIjoyMDY0NTM3NDE0fQ.8q_3dYYPsoJVOXWhT-61LXA-RYnET--sPKkUQi9WMe4"
OUTPUT_DIR = r"C:\Users\cdye\OneDrive\Desktop\GE Heathcare\Data\IMF Data\economic_pillar_pipeline"

# List of IMF table names
IMF_TABLES = [
    "imf_general_government_gross_debt_national_currency_wide",
    "imf_current_account_balance_percent_of_gdp_wide",
    "imf_general_government_revenue_percent_of_gdp_wide",
    "imf_general_government_total_expenditure_percent_of_gdp_wide",
    "imf_gross_domestic_product_constant_prices_percent_change_wide",
    "imf_gross_domestic_product_current_prices_purchasing_power_pari",
    "imf_gross_domestic_product_current_prices_u_s_dollars_wide",
    "imf_inflation_average_consumer_prices_index_wide",
    "imf_inflation_end_of_period_consumer_prices_index_wide",
    "imf_unemployment_rate_percent_of_total_labor_force_wide",
]

# initialise client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
print("Supabase client connected.")

# export each tables
for table in IMF_TABLES:
    try:
        print(f"Downloading: {table}")
        response = supabase.table(table).select("*").execute()

        if response.data:
            df = pd.DataFrame(response.data)
            output_path = os.path.join(OUTPUT_DIR, f"{table}.csv")
            df.to_csv(output_path, index=False)
            print(f"Saved to: {output_path}")
        else:
            print(f"No data found in {table}")

    except Exception as e:
        print(f"ERROR downloading {table}: {e}")

print("All tables processed.")
