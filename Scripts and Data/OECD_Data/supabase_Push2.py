import os
import pandas as pd
from supabase import create_client, Client
from datetime import datetime
import numpy as np
from dotenv import load_dotenv
load_dotenv()

pd.options.display.max_columns = None
pd.set_option('display.expand_frame_repr', False)

def fix_dates(date):
    if len(date) == 4:
        return datetime.strptime(date, "%Y").strftime("%Y-01-01")
    elif len(date) == 7:
        return datetime.strptime(date, "%Y-%m").strftime("%Y-%m-01")
    else:
        return date

url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")

supabase: Client = create_client(url, key)

datasets = ['industry_prod.csv', 'BCI.csv', 'GDP.csv', 'GDP_growth.csv', 'CCI.csv', 'CLI.csv', 'GovDebt.csv', 'unemployment.csv', 'CPI.csv']
growth = pd.read_csv(datasets[3])
growth["Quarter"] = growth["TIME_PERIOD"].str.split("-").str[1]
growth["TIME_PERIOD"] = (pd.PeriodIndex(growth["TIME_PERIOD"].str.replace("-",""), freq="Q").to_timestamp())
growth.to_csv(datasets[3][0], index=False)

country_code_fetch = supabase.table("country_region_IBAN").select("iso_alpha3").execute()
valid_codes = [code['iso_alpha3'] for code in country_code_fetch.data]

columns_to_keep = [
    "STRUCTURE_NAME", "REF_AREA", "Reference area", "Frequency of observation",
    "MEASURE", "Measure", "ADJUSTMENT", "Adjustment", "TRANSFORMATION", "Transformation",
    "Calculation methodology", "TIME_PERIOD", "OBS_VALUE", "OBS_STATUS", "Observation status",
    "UNIT_MULT", "Unit multiplier", "DECIMALS", "Decimals", "Quarter"
]

combined_oecd = pd.DataFrame()

for data in datasets:
    dataset = pd.read_csv(data)
    filtered_cols = [col for col in columns_to_keep if col in dataset.columns]
    dataset = dataset[filtered_cols]
    if "TIME_PERIOD" in dataset.columns:
        dataset["TIME_PERIOD"] = dataset["TIME_PERIOD"].astype(str)
        dataset["TIME_PERIOD"] = dataset["TIME_PERIOD"].apply(fix_dates)
    dataset = dataset[dataset["REF_AREA"].isin(valid_codes)].reset_index(drop=True)
    combined_oecd = pd.concat([combined_oecd, dataset], ignore_index=True)

combined_oecd.reset_index(drop=True, inplace=True)
combined_oecd.insert(0, 'ID', combined_oecd.index + 1)

data_json = combined_oecd.astype("object").replace({np.nan: None}).to_dict(orient="records")
batch_size = 500
for i in range(0, len(data_json), batch_size):
    records = data_json[i:i+batch_size]
    push = supabase.table("oecd_data").insert(records).execute()