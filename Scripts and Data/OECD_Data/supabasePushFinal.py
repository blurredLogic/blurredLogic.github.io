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


def generate_create_table_sql(df, table_name):
    pg_type_map = {
        'int64': 'BIGINT',
        'float64': 'DOUBLE PRECISION',
        'object': 'TEXT',
        'datetime64[ns]': 'DATE',
        'bool': 'BOOLEAN'
    }

    cols = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        pg_type = pg_type_map.get(dtype, 'TEXT')
        safe_col = col.replace(" ", "_").lower()
        cols.append(f'"{safe_col}" {pg_type}')

    cols_sql = ",\n  ".join(cols)
    create_sql = f'CREATE TABLE IF NOT EXISTS "{table_name}" (\n  {cols_sql}\n);'
    return create_sql

url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")

supabase: Client = create_client(url, key)

# complex_datasets = ['industry_prod.csv', 'CPI.csv']
# simple_datasets = ['BCI.csv', 'CCI.csv', 'CLI.csv', 'GovDebt.csv', 'unemployment.csv']

simple_datasets = ['BCI.csv', 'CCI.csv', 'CLI.csv', 'GovDebt.csv', 'unemployment.csv','industry_prod.csv', 'CPI.csv']

country_code_fetch = supabase.table("country_region_IBAN").select("iso_alpha3").execute()
valid_codes = [code['iso_alpha3'] for code in country_code_fetch.data]

country_fetch = supabase.table("country_region_IBAN").select("iso_alpha3, country").execute()
if hasattr(country_fetch, "data"):
    code_map = {entry["iso_alpha3"]: entry["country"].replace(" ", "_").lower() for entry in country_fetch.data}
else:
    code_map = {}


# for data in complex_datasets:
#     dataset = pd.read_csv(data)
#     dataset = dataset[dataset["REF_AREA"].isin(valid_codes)].reset_index(drop=True)
#     if "TIME_PERIOD" in dataset.columns:
#         dataset["TIME_PERIOD"] = dataset["TIME_PERIOD"].astype(str)
#         dataset["TIME_PERIOD"] = dataset["TIME_PERIOD"].apply(fix_dates)
#     dataset['date'] = pd.to_datetime(dataset['TIME_PERIOD'], format='%Y-%m')
#
#     dataset['series'] = dataset['REF_AREA'] + '_' + dataset['TRANSFORMATION'].replace({
#         '_Z': 'IDX',
#         'G1': 'G1',
#         'GY': 'GY'
#         })
#     pivot = dataset.pivot(index='date', columns='series', values='OBS_VALUE')
#     print(pivot.head())


for data in simple_datasets:
    dataset = pd.read_csv(data)
    dataset = dataset[dataset["REF_AREA"].isin(valid_codes)].reset_index(drop=True)
    if "TIME_PERIOD" in dataset.columns:
        dataset["TIME_PERIOD"] = dataset["TIME_PERIOD"].astype(str)
        dataset["TIME_PERIOD"] = dataset["TIME_PERIOD"].apply(fix_dates)
    if "Unit of measure" in dataset.columns:
        dataset = dataset[dataset["Unit of measure"].isin(
            ["Index", "Percentage of GDP", "Percentage of labour force in the same subgroup"])].reset_index(drop=True)
    dataset['date'] = dataset['TIME_PERIOD']
        #pd.to_datetime(dataset['TIME_PERIOD'], format='%Y-%m'))
    dataset["country"] = dataset["REF_AREA"].map(code_map)
    dataset['series'] = dataset['country']
    pivot = dataset.pivot(index='date', columns='series', values='OBS_VALUE')
    pivot.columns.name = None
    pivot = pivot.reset_index()
    data_json = pivot.astype("object").replace({np.nan: None}).to_dict(orient="records")
    table_name = data.split("/")[-1].replace(".csv","").lower()
    #print(generate_create_table_sql(pivot, table_name))
    batch_size = 500
    for i in range(0, len(data_json), batch_size):
        records = data_json[i:i + batch_size]
        push = supabase.table(table_name).insert(records).execute()






