import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import re
import unicodedata

# names for ISO-3 codes
CODE_TO_NAME = {
    'AFG':'Afghanistan','ALB':'Albania','DZA':'Algeria','AND':'Andorra','AGO':'Angola',
    'ARG':'Argentina','ARM':'Armenia','AUS':'Australia','AUT':'Austria','AZE':'Azerbaijan',
    'BHS':'Bahamas','BHR':'Bahrain','BGD':'Bangladesh','BRB':'Barbados','BLR':'Belarus',
    'BEL':'Belgium','BLZ':'Belize','BEN':'Benin','BTN':'Bhutan','BOL':'Bolivia',
    'BIH':'Bosnia and Herzegovina','BWA':'Botswana','BRA':'Brazil','BRN':'Brunei Darussalam',
    'BGR':'Bulgaria','BFA':'Burkina Faso','BDI':'Burundi','KHM':'Cambodia','CMR':'Cameroon',
    'CAN':'Canada','CAF':'Central African Republic','TCD':'Chad','CHL':'Chile','CHN':'China',
    'COL':'Colombia','COM':'Comoros','COG':'Congo (Brazzaville)','COD':'Congo (Kinshasa)',
    'CRI':'Costa Rica',"CIV":"Cote d'Ivoire",'HRV':'Croatia','CUB':'Cuba','CYP':'Cyprus',
    'CZE':'Czech Republic','DNK':'Denmark','DJI':'Djibouti','DOM':'Dominican Republic',
    'ECU':'Ecuador','EGY':'Egypt','SLV':'El Salvador','GNQ':'Equatorial Guinea','ERI':'Eritrea',
    'EST':'Estonia','SWZ':'Eswatini','ETH':'Ethiopia','FJI':'Fiji','FIN':'Finland','FRA':'France',
    'GAB':'Gabon','GMB':'Gambia, The','GEO':'Georgia','DEU':'Germany','GHA':'Ghana','GRC':'Greece',
    'GTM':'Guatemala','GIN':'Guinea','GNB':'Guinea-Bissau','GUY':'Guyana','HTI':'Haiti',
    'HND':'Honduras','HKG':'Hong Kong','HUN':'Hungary','ISL':'Iceland','IND':'India',
    'IDN':'Indonesia','IRN':'Iran','IRQ':'Iraq','IRL':'Ireland','ISR':'Israel','ITA':'Italy',
    'JAM':'Jamaica','JPN':'Japan','JOR':'Jordan','KAZ':'Kazakhstan','KEN':'Kenya',
    'KWT':'Kuwait','KGZ':'Kyrgyzstan','LAO':'Laos','LVA':'Latvia','LBN':'Lebanon','LSO':'Lesotho',
    'LBR':'Liberia','LBY':'Libya','LIE':'Liechtenstein','LTU':'Lithuania','LUX':'Luxembourg',
    'MAC':'Macau','MDG':'Madagascar','MWI':'Malawi','MYS':'Malaysia','MDV':'Maldives',
    'MLI':'Mali','MLT':'Malta','MRT':'Mauritania','MUS':'Mauritius','MEX':'Mexico',
    'MDA':'Moldova','MCO':'Monaco','MNG':'Mongolia','MNE':'Montenegro','MAR':'Morocco',
    'MOZ':'Mozambique','MMR':'Myanmar','NAM':'Namibia','NPL':'Nepal','NLD':'Netherlands',
    'NZL':'New Zealand','NIC':'Nicaragua','NER':'Niger','NGA':'Nigeria','MKD':'North Macedonia',
    'NOR':'Norway','OMN':'Oman','PAK':'Pakistan','PAN':'Panama','PNG':'Papua New Guinea',
    'PRY':'Paraguay','PER':'Peru','PHL':'Philippines','POL':'Poland','PRT':'Portugal',
    'PRI':'Puerto Rico','QAT':'Qatar','ROU':'Romania','RUS':'Russia','RWA':'Rwanda',
    'SAU':'Saudi Arabia','SEN':'Senegal','SRB':'Serbia','SLE':'Sierra Leone','SGP':'Singapore',
    'SVK':'Slovakia','SVN':'Slovenia','SOM':'Somalia','ZAF':'South Africa','KOR':'South Korea',
    'ESP':'Spain','LKA':'Sri Lanka','SDN':'Sudan','SUR':'Suriname','SWE':'Sweden',
    'CHE':'Switzerland','SYR':'Syria','TWN':'Taiwan','TJK':'Tajikistan','TZA':'Tanzania',
    'THA':'Thailand','TGO':'Togo','TTO':'Trinidad and Tobago','TUN':'Tunisia','TUR':'Turkey',
    'TKM':'Turkmenistan','UGA':'Uganda','UKR':'Ukraine','ARE':'United Arab Emirates',
    'GBR':'United Kingdom','USA':'United States','URY':'Uruguay','UZB':'Uzbekistan',
    'VEN':'Venezuela','VNM':'Vietnam','YEM':'Yemen','ZMB':'Zambia','ZWE':'Zimbabwe',

    # corrections
    'ABW':'Aruba','ADO':'Andorra','ATG':'Antigua and Barbuda','CPV':'Cabo Verde','CYM':'Cayman Islands',
    'DMA':'Dominica','FSM':'Micronesia, Fed. Sts.','GRD':'Grenada','GUF':'French Guiana','KIR':'Kiribati',
    'KNA':'St. Kitts and Nevis','KSV':'Kosovo','LCA':'St. Lucia','MHL':'Marshall Islands','NRU':'Nauru',
    'PLW':'Palau','PRK':'North Korea','SLB':'Solomon Islands','STP':'Sao Tome and Principe',
    'SYC':'Seychelles','TON':'Tonga','TUV':'Tuvalu','VCT':'St. Vincent and the Grenadines',
    'VUT':'Vanuatu','WBG':'West Bank and Gaza','WSM':'Samoa',
    'ROM':'Romania','TMP':'Timor-Leste','ZAR':'Congo (Kinshasa)'
}

# names for cleaning
ALIAS_TO_NAME = {
    # Côte d'Ivoire
    "cote divoire":"Cote d'Ivoire","cote d'ivoire":"Cote d'Ivoire","ivory coast":"Cote d'Ivoire",
    # Eswatini
    "swaziland":"Eswatini","eswatini (fmr. swaziland)":"Eswatini",
    # Congo DRC
    "democratic republic of the congo":"Congo (Kinshasa)","dr congo":"Congo (Kinshasa)",
    "drc":"Congo (Kinshasa)","congo, dem. rep.":"Congo (Kinshasa)","congo (drc)":"Congo (Kinshasa)",
    "zaire":"Congo (Kinshasa)",
    # Congo Rep.
    "republic of the congo":"Congo (Brazzaville)","congo, rep.":"Congo (Brazzaville)",
    # Korea
    "korea, rep.":"South Korea","republic of korea":"South Korea","south korea":"South Korea",
    "korea, dpr":"North Korea","democratic people's republic of korea":"North Korea","north korea":"North Korea",
    # Czechia
    "czechia":"Czech Republic",
    # North Macedonia
    "macedonia":"North Macedonia","macedonia, fyr":"North Macedonia","fyrom":"North Macedonia",
    # Myanmar
    "burma":"Myanmar",
    # Cabo Verde
    "cape verde":"Cabo Verde",
    # Timor-Leste
    "east timor":"Timor-Leste","timor leste":"Timor-Leste",
    # Taiwan/Hong Kong/Macau
    "taiwan, china":"Taiwan","taiwan province of china":"Taiwan",
    "hong kong sar, china":"Hong Kong",
    "macao sar, china":"Macau","macao":"Macau",
    # Palestine
    "west bank and gaza":"West Bank and Gaza","palestine":"West Bank and Gaza","palestinian territories":"West Bank and Gaza",
    # Moldova
    "republic of moldova":"Moldova","moldova, rep.":"Moldova",
    # Tanzania
    "united republic of tanzania":"Tanzania",
    # Venezuela
    "venezuela, rb":"Venezuela",
    # Gambia
    "gambia":"Gambia, The",
    # Bahamas / Micronesia
    "bahamas, the":"Bahamas","micronesia (federated states of)":"Micronesia, Fed. Sts.",
    # Lao PDR
    "lao pdr":"Laos","lao people's democratic republic":"Laos",
    # Russia / Syria / Iran
    "russian federation":"Russia","syrian arab republic":"Syria","iran, islamic rep.":"Iran","islamic republic of iran":"Iran",
    # UK/US/UAE
    "uk":"United Kingdom","u.k.":"United Kingdom","great britain":"United Kingdom",
    "usa":"United States","u.s.":"United States","u.s.a.":"United States",
    "uae":"United Arab Emirates",
    # St. → Saint (handled generically too, but keep explicit)
    "saint kitts and nevis":"St. Kitts and Nevis","saint lucia":"St. Lucia","saint vincent and the grenadines":"St. Vincent and the Grenadines",
}

# light normalisation helpers
def _strip_accents(text: str) -> str:
    return ''.join(c for c in unicodedata.normalize('NFKD', text) if not unicodedata.combining(c))

def _basic_clean(s: str) -> str:
    s = _strip_accents(s)
    s = s.replace("’", "'")
    s = re.sub(r"\s+", " ", s.strip())
    return s

def normalize_country(value: str) -> str:
    """Return a single canonical country name for names or ISO-3 codes."""
    if pd.isna(value):
        return value
    raw = str(value).strip()

    # ISO-3 code
    if len(raw) == 3 and raw.isalpha():
        name = CODE_TO_NAME.get(raw.upper())
        if name:
            return name

    s = _basic_clean(raw)
    key = s.lower()

    # standardize common "saint" variants to "St. "
    key = re.sub(r"^saint\s+", "st. ", key)

    # alias map
    if key in ALIAS_TO_NAME:
        return ALIAS_TO_NAME[key]

    # name/key cleanups
    generic = {
        "cote d ivoire": "Cote d'Ivoire",
    }
    if key in generic:
        return generic[key]

    # if nothing matched, return cleaned original with title/punctuation preserved
    # but keep known preferred punctuation e.g., Cote d'Ivoire
    return s

# monthly pillar inputs
PILLAR_CONFIG = {
    "economic": {
        "filename": "economic_pillar_score_monthly.csv",
        "country_col": "country",
        "score_col": "economic_risk_pillar_score",
    },
    "governance": {
        "filename": "wgi_governance_pillar_score_monthly.csv",
        "country_col": "country",
        "score_col": "governance_risk_pillar_score",
    },
    "sovereign": {
        "filename": "sovereign_pillar_score_monthly.csv",
        "country_col": "country",
        "score_col": "sovereign_risk_pillar_score",
    },
    "trade_payment": {
        "filename": "trade_payment_pillar_score_monthly.csv",
        "country_col": "country",
        "score_col": "trade_payment_pillar_score",
    },
}


# load and merge monthly pillars
pillar_dfs = []
for pillar, cfg in PILLAR_CONFIG.items():
    df = pd.read_csv(
        cfg["filename"],
        parse_dates=["date"],
        usecols=["date", cfg["country_col"], cfg["score_col"]],
    )
    # normalise country names first
    df[cfg["country_col"]] = df[cfg["country_col"]].apply(normalize_country)
    # standardise columns
    df.rename(
        columns={
            cfg["country_col"]: "country",
            cfg["score_col"]: f"{pillar}_score",
        },
        inplace=True,
    )
    pillar_dfs.append(df[["date", "country", f"{pillar}_score"]])

# merge all pillars
master = pillar_dfs[0]
for df in pillar_dfs[1:]:
    master = master.merge(df, on=["date", "country"], how="outer")

master.sort_values(["country", "date"], inplace=True)
pillar_cols = [f"{p}_score" for p in PILLAR_CONFIG.keys()]

CUT_OFF_DATE = pd.Timestamp('2025-01-01')  # keep strictly before Jan 2025
master = master[master['date'] <= CUT_OFF_DATE].copy()

# fill occasional gaps with monthly means (z-scores - 0 acts as mean imputation)
for col in pillar_cols:
    master[col] = master.groupby("date")[col].transform(lambda s: s.fillna(s.mean()))
    master[col] = master[col].fillna(0.0)

# validation and pca loadings
from pathlib import Path
OUTDIR = Path("validation_outputs")
OUTDIR.mkdir(exist_ok=True)

# Coverage & missingness on latest month
latest_date = master["date"].max()
latest_slice = master[master["date"] == latest_date].copy()

# how many pillars available per country (0–4)
latest_slice["pillar_count"] = latest_slice[pillar_cols].notna().sum(axis=1)

print("\n[VALIDATION] Latest date:", latest_date.date())
print("[VALIDATION] Pillar availability (countries by count):")
print(latest_slice["pillar_count"].value_counts().sort_index())

# export who has <3 pillars so they dont get a composite
lt3 = latest_slice.loc[latest_slice["pillar_count"] < 3, ["country"] + pillar_cols]
lt3.to_csv(OUTDIR / "countries_with_lt3_pillars_latest.csv", index=False)

# Pillar ranges and basic stats
desc = master[pillar_cols].describe().T
desc.to_csv(OUTDIR / "pillar_summary_stats_full_panel.csv")
print("\n[VALIDATION] Pillar summary stats (full panel) saved → pillar_summary_stats_full_panel.csv")

# Date alignment
date_counts = master.groupby("date").size().reset_index(name="rows")
date_counts.to_csv(OUTDIR / "rows_per_date.csv", index=False)

# PCA on latest month across pillars
#    - mean-impute within latest month so PCA has no NaNs
for c in pillar_cols:
    latest_slice[c] = latest_slice[c].fillna(latest_slice[c].mean())

# standardise pillars cross-sectionally (latest month), run PCA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
X = latest_slice[pillar_cols].to_numpy()
Xs = StandardScaler().fit_transform(X)

pca = PCA(n_components=len(pillar_cols))
pca.fit(Xs)

# Align PC1 sign so that higher → higher risk (correlate with sum of pillars)
pc1 = pca.components_[0, :]
if (Xs.sum(axis=1) @ (Xs @ pc1)) < 0:
    pca.components_[0, :] *= -1
    pc1 = pca.components_[0, :]

# explained variance
explained = pd.Series(pca.explained_variance_ratio_, index=[f"PC{i+1}" for i in range(len(pillar_cols))])
explained.to_csv(OUTDIR / "pca_explained_variance_latest.csv", header=["explained_variance_ratio"])
print("\n[VALIDATION] PCA explained variance (latest):")
print(explained)

# loadings contribution of each pillar to PC1
loadings_pc1 = pd.DataFrame({
    "pillar": pillar_cols,
    "loading_pc1": pc1
}).sort_values("loading_pc1", key=lambda s: s.abs(), ascending=False)
loadings_pc1.to_csv(OUTDIR / "pca_loadings_pc1_latest.csv", index=False)

print("\n[VALIDATION] PCA PC1 loadings (latest) saved pca_loadings_pc1_latest.csv")
print(loadings_pc1)

# Quick sanity snapshot: top/bottom risk by each pillar (latest month)
snap_list = []
for c in pillar_cols:
    top5 = latest_slice[["country", c]].sort_values(c, ascending=False).head(5).assign(metric=c, rank_group="top5")
    bot5 = latest_slice[["country", c]].sort_values(c, ascending=True).head(5).assign(metric=c, rank_group="bottom5")
    snap_list += [top5, bot5]
snap = pd.concat(snap_list, ignore_index=True)
snap.to_csv(OUTDIR / "pillar_top_bottom_latest.csv", index=False)
print("\n[VALIDATION] Saved top/bottom countries per pillar (latest) → pillar_top_bottom_latest.csv")

master.sort_values(["country", "date"], inplace=True)
pillar_cols = [f"{p}_score" for p in PILLAR_CONFIG.keys()]

# composite scores monthly
# Equal-weight (25% each)
master["CRI_equal_weights"] = master[pillar_cols].mean(axis=1)

# Four 40% focus scenarios
scenarios = {
    "CRI_economic_focus":  {"economic": 0.4, "governance": 0.2, "sovereign": 0.2, "trade_payment": 0.2},
    "CRI_governance_focus":{"economic": 0.2, "governance": 0.4, "sovereign": 0.2, "trade_payment": 0.2},
    "CRI_sovereign_focus": {"economic": 0.2, "governance": 0.2, "sovereign": 0.4, "trade_payment": 0.2},
    "CRI_trade_focus":     {"economic": 0.2, "governance": 0.2, "sovereign": 0.2, "trade_payment": 0.4},
}
for name, weights in scenarios.items():
    master[name] = sum(master[f"{p}_score"] * w for p, w in weights.items())

# C) PCA-based composite (monthly, cross-sectional)
pca_rows = []
for dt, grp in master.groupby("date", sort=True):
    X = grp[pillar_cols].to_numpy()
    # Scale across pillars
    Xs = StandardScaler().fit_transform(X)
    pca = PCA(n_components=1)
    pc1 = pca.fit_transform(Xs).ravel()
    # Align sign so higher - higher risk
    if np.corrcoef(pc1, Xs.sum(axis=1))[0, 1] < 0:
        pc1 = -pc1
    tmp = grp[["country"]].copy()
    tmp["date"] = dt
    tmp["CRI_pca_weights"] = pc1
    pca_rows.append(tmp)

if pca_rows:
    pcadf = pd.concat(pca_rows, ignore_index=True)
    master = master.merge(pcadf, on=["date", "country"], how="left")

# trailing 12 month averages for dashboard score
composite_cols = [c for c in master.columns if c.startswith("CRI_") and c not in ("CRI_pca_weights_0_to_100",)]

# trailing averages per country
for col in composite_cols:
    master[f"{col}_trailing_12m"] = (
        master.sort_values(["country", "date"])
              .groupby("country")[col]
              .transform(lambda s: s.rolling(window=12, min_periods=1).mean())
    )

# latest snapshot per country
latest = master.sort_values("date").groupby("country").tail(1).reset_index(drop=True)

# min–max scale each composite to 0–100
final = latest[["country"]].copy()
for col in composite_cols:
    avg_col = f"{col}_trailing_12m"
    mn = latest[avg_col].min()
    mx = latest[avg_col].max()
    if pd.notna(mn) and pd.notna(mx) and mx > mn:
        final[f"{col}_0_to_100"] = (latest[avg_col] - mn) / (mx - mn) * 100.0
    else:
        final[f"{col}_0_to_100"] = np.nan  # degenerate case

# save
final.to_csv("final_composite_risk_scores.csv", index=False)
print("Saved final_composite_risk_scores.csv")
print(final.head())

try:
    final = pd.read_csv("final_composite_risk_scores.csv")
    eq = final[["country", "CRI_equal_weights_0_to_100"]].dropna()
    eq.sort_values("CRI_equal_weights_0_to_100", ascending=False).head(10).to_csv(
        OUTDIR / "composite_equal_top10.csv", index=False
    )
    eq.sort_values("CRI_equal_weights_0_to_100", ascending=True).head(10).to_csv(
        OUTDIR / "composite_equal_bottom10.csv", index=False
    )
    print("\n[VALIDATION] Composite equal-weights top/bottom saved in validation_outputs/")
except Exception as e:
    print("[VALIDATION] Could not build composite snapshot:", e)
