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

datasets = [["industry_prod.csv","industry_production"], ["BCI.csv","business_confidence_index"],["GDP.csv","gross_domestic_product"],["GDP_growth.csv","gdp_growth"],["CCI.csv","consumer_confidence_index"],["CLI.csv","composite_leading_indicator"],
            ["GovDebt.csv","government_debt"],["unemployment.csv","unemployment"],["CPI.csv","consumer_price_index"]]

growth = pd.read_csv(datasets[3][0])
growth["Quarter"] = growth["TIME_PERIOD"].str.split("-").str[1]
growth["TIME_PERIOD"] = (pd.PeriodIndex(growth["TIME_PERIOD"].str.replace("-",""), freq="Q").to_timestamp())
growth.to_csv(datasets[3][0], index=False)

country_code_fetch = supabase.table("country_region_IBAN").select("iso_alpha3").execute()
valid_codes = [code['iso_alpha3'] for code in country_code_fetch.data]

for dataset in datasets:
    data = pd.read_csv(dataset[0])
    if "TIME_PERIOD" in data.columns:
        data["TIME_PERIOD"] = data["TIME_PERIOD"].astype(str)
        data["TIME_PERIOD"] = data["TIME_PERIOD"].apply(fix_dates)
    data = data[data["REF_AREA"].isin(valid_codes)].reset_index(drop=True)
    data = data.replace({np.nan: None})
    data_json = data.to_dict(orient="records")
    batch_size = 500
    for i in range(0, len(data_json), batch_size):
        records = data_json[i:i+batch_size]
        push = supabase.table(dataset[1]).insert(records).execute()