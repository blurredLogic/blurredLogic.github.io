import requests
import pandas as pd
from io import StringIO
import time

pd.options.display.max_columns = None
pd.set_option('display.expand_frame_repr', False)


datasets = [
    ("https://sdmx.oecd.org/public/rest/data/OECD.SDD.STES,DSD_KEI@DF_KEI,4.0/.M.PRVM..BTE..?startPeriod=2006-01&dimensionAtObservation=AllDimensions&format=csvfilewithlabels",
     "industry_prod.csv"),
    ("https://sdmx.oecd.org/public/rest/data/OECD.SDD.STES,DSD_STES@DF_CLI,/.M.BCICP...AA...H?startPeriod=2006-01&dimensionAtObservation=AllDimensions&format=csvfilewithlabels", "BCI.csv"),
    ("https://sdmx.oecd.org/public/rest/data/OECD.SDD.STES,DSD_STES@DF_CLI,/.M.CCICP...AA...H?startPeriod=2006-01&dimensionAtObservation=AllDimensions&format=csvfilewithlabels", "CCI.csv"),
    ("https://sdmx.oecd.org/public/rest/data/OECD.SDD.STES,DSD_STES@DF_CLI,4.1/AUS+FRA+DEU+ITA+JPN+KOR+MEX+ESP+TUR+GBR+USA+WXOECD+BRA+CHN+IND+IDN+ZAF+A5M+CAN.M.LI...NOR.IX..H?startPeriod=2006-01&dimensionAtObservation=AllDimensions&format=csvfilewithlabels", "CLI.csv"),
    ("https://sdmx.oecd.org/public/rest/data/OECD.GOV.GIP,DSD_GOV@DF_GOV_PF_YU,/A..GGD.PT_B1GQ...?startPeriod=2007&dimensionAtObservation=AllDimensions&format=csvfilewithlabels", "GovDebt.csv"),
    ("https://sdmx.oecd.org/public/rest/data/OECD.SDD.TPS,DSD_LFS@DF_IALFS_INDIC,1.0/.UNE_LF_M...Y._T.Y_GE15..M?startPeriod=2006-01&dimensionAtObservation=AllDimensions&format=csvfilewithlabels", "unemployment.csv"),
    ("https://sdmx.oecd.org/public/rest/data/OECD.SDD.TPS,DSD_PRICES@DF_PRICES_ALL,1.0/.M.N.CPI.._T.N.GY+_Z?startPeriod=2006-01&dimensionAtObservation=AllDimensions&format=csvfilewithlabels", "CPI.csv")
]


delay = 5


for i, (url, filename) in enumerate(datasets):
    print(f"Fetching dataset {i + 1} from {url}")
    try:
        response = requests.get(url)
        df = pd.read_csv(StringIO(response.text))
        columns_to_drop = [
            "STRUCTURE", "STRUCTURE_ID", "ACTION", "FREQ", "UNIT_MEASURE", "SECTOR",
            "Institutional sector", "Edition", "EDITION", "CATEGORY", "Category",
            "Time period", "Observation value", "PRICE_BASE", "Price base",
            "BASE_PER", "Base period"
        ]
        df.drop(columns=columns_to_drop, inplace=True, errors='ignore')
        df.insert(0, 'ID', range(1, len(df) + 1))
        df.to_csv(filename, index=False)
        print(f"Saved to {filename}. First few rows:")
        print(df.head())
    except Exception as e:
        print(f"Failed to fetch or save dataset {i + 1}: {e}")

    if i < len(datasets) - 1:
        time.sleep(delay)