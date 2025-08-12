import pandas as pd

def data_types(columns):
    if pd.api.types.is_integer_dtype(columns):
        return "INT"
    elif pd.api.types.is_float_dtype(columns):
        return "NUMERIC"
    elif pd.api.types.is_bool_dtype(columns):
        return "BOOL"
    elif pd.api.types.is_datetime64_any_dtype(columns):
        return "DATE"
    else:
        return "TEXT"

datasets = ["industry_prod.csv", "BCI.csv","GDP.csv","GDP_growth.csv","CCI.csv","CLI.csv",
            "GovDebt.csv","unemployment.csv","CPI.csv"]

for dataset in datasets:
    df = pd.read_csv(dataset)

    print(f"\n{dataset}:")
    for column in df.columns:
        sql_type = data_types(df[column])
        print(f"  {column}: {sql_type}")