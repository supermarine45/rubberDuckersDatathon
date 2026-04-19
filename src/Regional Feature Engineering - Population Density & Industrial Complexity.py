# Databricks notebook source


# COMMAND ----------

# DBTITLE 1,Install dependencies
# MAGIC %pip install eurostat --quiet

# COMMAND ----------

# DBTITLE 1,Configuration - Spanish NUTS region mappings
import pandas as pd
import numpy as np
import requests
import eurostat

# =============================================================================
# TARGET DATE RANGE for time-varying features
# Must match your electricity load data and weather table coverage
# =============================================================================
DATE_START = '2025-01-01'
DATE_END = '2026-02-28'

# Spanish Autonomous Communities (NUTS2 level) - this is typically the level
# at which electricity load data is available from REE/ENTSO-E
SPAIN_NUTS2 = {
    'ES11': 'Galicia',
    'ES12': 'Principado de Asturias',
    'ES13': 'Cantabria',
    'ES21': 'País Vasco',
    'ES22': 'Comunidad Foral de Navarra',
    'ES23': 'La Rioja',
    'ES24': 'Aragón',
    'ES30': 'Comunidad de Madrid',
    'ES41': 'Castilla y León',
    'ES42': 'Castilla-la Mancha',
    'ES43': 'Extremadura',
    'ES51': 'Cataluña',
    'ES52': 'Comunitat Valenciana',
    'ES53': 'Illes Balears',
    'ES61': 'Andalucía',
    'ES62': 'Región de Murcia',
    'ES63': 'Ciudad Autónoma de Ceuta',
    'ES64': 'Ciudad Autónoma de Melilla',
    'ES70': 'Canarias'
}

# DEGURBA-like urban/rural classification thresholds (inhabitants per km²)
# Based on Eurostat methodology:
#   - Densely populated (urban): > 1500 inh/km²
#   - Intermediate: 300 - 1500 inh/km²
#   - Thinly populated (rural): < 300 inh/km²
URBAN_THRESHOLD = 1500
INTERMEDIATE_THRESHOLD = 300

def classify_urban_rural(density):
    """Classify region based on population density (DEGURBA-like)."""
    if pd.isna(density):
        return 'unknown'
    elif density > URBAN_THRESHOLD:
        return 'urban'
    elif density > INTERMEDIATE_THRESHOLD:
        return 'intermediate'
    else:
        return 'rural'

print(f"Configured {len(SPAIN_NUTS2)} Spanish autonomous communities (NUTS2)")
print(f"Target date range: {DATE_START} to {DATE_END}")
print("Thresholds: Urban > {}, Intermediate > {}, Rural <= {}".format(
    URBAN_THRESHOLD, INTERMEDIATE_THRESHOLD, INTERMEDIATE_THRESHOLD))

# COMMAND ----------

# DBTITLE 1,Fetch Population & Urban/Rural Classification (Eurostat)
# =============================================================================
# POPULATION DENSITY by NUTS2 region
# Dataset: demo_r_d2jan (Population on 1 January by NUTS 2 region)
# Source: Eurostat - free, no API key required
#
# Strategy: Fetch total population, then divide by known land area (km²)
# to compute population density for urban/rural classification.
# =============================================================================

# Land area (km²) of Spanish autonomous communities (official INE/IGN values)
SPAIN_NUTS2_AREA_KM2 = {
    'ES11': 29_574,  # Galicia
    'ES12': 10_604,  # Asturias
    'ES13': 5_321,   # Cantabria
    'ES21': 7_234,   # País Vasco
    'ES22': 10_391,  # Navarra
    'ES23': 5_045,   # La Rioja
    'ES24': 47_720,  # Aragón
    'ES30': 8_028,   # Madrid
    'ES41': 94_226,  # Castilla y León
    'ES42': 79_461,  # Castilla-la Mancha
    'ES43': 41_635,  # Extremadura
    'ES51': 32_114,  # Cataluña
    'ES52': 23_255,  # Comunitat Valenciana
    'ES53': 4_992,   # Illes Balears
    'ES61': 87_268,  # Andalucía
    'ES62': 11_313,  # Murcia
    'ES63': 19,      # Ceuta
    'ES64': 13,      # Melilla
    'ES70': 7_447,   # Canarias
}

print("Fetching population data from Eurostat (demo_r_d2jan)...")

# Get population at NUTS2 level
pop_density_df = eurostat.get_data_df('demo_r_d2jan', flags=False)
print(f"Raw dataset shape: {pop_density_df.shape}")

# Identify the geo column (naming varies by eurostat package version)
geo_col = [c for c in pop_density_df.columns if 'geo' in c.lower()][0]

# Filter for Spanish NUTS2 regions, total sex, total age
spain_pop_raw = pop_density_df[
    (pop_density_df[geo_col].isin(SPAIN_NUTS2.keys())) &
    (pop_density_df['sex'] == 'T') &
    (pop_density_df['age'] == 'TOTAL')
].copy()

# Get the most recent year available
year_cols = [c for c in spain_pop_raw.columns if isinstance(c, str) and len(c) == 4 and c.isdigit()]
if not year_cols:
    year_cols = [c for c in spain_pop_raw.columns if str(c).isdigit()]
latest_year = sorted(year_cols)[-1]
print(f"Using latest available year: {latest_year}")

# Build clean DataFrame with population and compute density
pop_features = spain_pop_raw[[geo_col, latest_year]].copy()
pop_features.columns = ['nuts2_code', 'total_population_density_src']
pop_features['total_population_density_src'] = pd.to_numeric(
    pop_features['total_population_density_src'], errors='coerce'
)

# Compute population density (inhabitants per km²)
pop_features['area_km2'] = pop_features['nuts2_code'].map(SPAIN_NUTS2_AREA_KM2)
pop_features['population_density'] = (
    pop_features['total_population_density_src'] / pop_features['area_km2']
)

# Apply urban/rural classification based on DEGURBA thresholds
pop_features['urban_rural_class'] = pop_features['population_density'].apply(classify_urban_rural)
pop_features['region_name'] = pop_features['nuts2_code'].map(SPAIN_NUTS2)

# Keep only relevant columns
pop_features = pop_features[['nuts2_code', 'region_name', 'population_density', 'urban_rural_class']].copy()

print(f"\nPopulation density computed for {len(pop_features)} regions")
print(f"\nUrban/Rural distribution:")
print(pop_features['urban_rural_class'].value_counts())
display(pop_features.sort_values('population_density', ascending=False))

# COMMAND ----------

# DBTITLE 1,Fetch Total Population by Region (Eurostat)
# =============================================================================
# TOTAL POPULATION by NUTS2 region
# Dataset: demo_r_pjangrp3 (Population on 1 January by age, sex, NUTS 2)
# Source: Eurostat - free, no API key required
# =============================================================================

print("Fetching total population from Eurostat (demo_r_pjangrp3)...")

pop_total_df = eurostat.get_data_df('demo_r_pjangrp3', flags=False)
print(f"Raw dataset shape: {pop_total_df.shape}")

# Identify columns
geo_col = [c for c in pop_total_df.columns if 'geo' in c.lower()][0]

# Filter: Spain NUTS2, total sex, total age
spain_pop = pop_total_df[
    (pop_total_df[geo_col].isin(SPAIN_NUTS2.keys())) &
    (pop_total_df['sex'] == 'T') &
    (pop_total_df['age'] == 'TOTAL')
].copy()

# Get latest year
year_cols = [c for c in spain_pop.columns if isinstance(c, (int, float)) or 
             (isinstance(c, str) and len(c) == 4 and c.isdigit())]
latest_year = sorted(year_cols)[-1]
print(f"Using latest available year: {latest_year}")

# Build clean total population DataFrame
population_df = spain_pop[[geo_col, latest_year]].copy()
population_df.columns = ['nuts2_code', 'total_population']
population_df['total_population'] = pd.to_numeric(population_df['total_population'], errors='coerce')

print(f"Total population fetched for {len(population_df)} regions")
display(population_df.sort_values('total_population', ascending=False))

# COMMAND ----------

# DBTITLE 1,Fetch Tourism / Holiday Destination Indicator (Eurostat)
# =============================================================================
# TOURISM - Nights spent at tourist accommodation by NUTS2 region
# Dataset: tour_occ_nin2 (Nights spent at tourist accommodation by NUTS 2)
# Source: Eurostat - free, no API key required
#
# Strategy: Regions with high tourism nights per capita = holiday destinations
# =============================================================================

print("Fetching tourism data from Eurostat (tour_occ_nin2)...")

tourism_df = eurostat.get_data_df('tour_occ_nin2', flags=False)
print(f"Raw dataset shape: {tourism_df.shape}")

# Identify columns
geo_col = [c for c in tourism_df.columns if 'geo' in c.lower()][0]

# Filter for Spanish NUTS2, total accommodation (I551-I553), total origin
spain_tourism = tourism_df[
    tourism_df[geo_col].isin(SPAIN_NUTS2.keys())
].copy()

# Filter for total nights (all accommodation types, all countries of residence)
if 'nace_r2' in spain_tourism.columns:
    spain_tourism = spain_tourism[spain_tourism['nace_r2'] == 'I551-I553']
if 'c_resid' in spain_tourism.columns:
    spain_tourism = spain_tourism[spain_tourism['c_resid'] == 'TOTAL']
if 'unit' in spain_tourism.columns:
    spain_tourism = spain_tourism[spain_tourism['unit'] == 'NR']

# Get latest year
year_cols = [c for c in spain_tourism.columns if isinstance(c, (int, float)) or 
             (isinstance(c, str) and len(c) == 4 and c.isdigit())]
latest_year = sorted(year_cols)[-1]
print(f"Using latest available year: {latest_year}")

# Aggregate by region (in case multiple rows per region)
tourism_features = spain_tourism.groupby(geo_col)[latest_year].sum().reset_index()
tourism_features.columns = ['nuts2_code', 'tourism_nights']
tourism_features['tourism_nights'] = pd.to_numeric(tourism_features['tourism_nights'], errors='coerce')

# Compute tourism intensity (nights per capita) - join with population
tourism_features = tourism_features.merge(population_df[['nuts2_code', 'total_population']], on='nuts2_code', how='left')
tourism_features['tourism_nights_per_capita'] = (
    tourism_features['tourism_nights'] / tourism_features['total_population']
)

# Holiday destination flag: regions with tourism nights per capita > median * 1.5
median_tourism = tourism_features['tourism_nights_per_capita'].median()
tourism_features['is_holiday_destination'] = (
    tourism_features['tourism_nights_per_capita'] > median_tourism * 1.5
).astype(int)

print(f"\nTourism data fetched for {len(tourism_features)} regions")
print(f"Median tourism nights per capita: {median_tourism:.1f}")
print(f"Holiday destinations (above 1.5x median): {tourism_features['is_holiday_destination'].sum()} regions")
display(tourism_features[['nuts2_code', 'tourism_nights', 'tourism_nights_per_capita', 'is_holiday_destination']])

# COMMAND ----------

# DBTITLE 1,Fetch Industrial Index (Eurostat - Regional GVA by sector)
# =============================================================================
# INDUSTRIAL INDEX - Gross Value Added by sector at NUTS2 level
# Dataset: nama_10r_3gva (GVA at basic prices by NUTS 3 region)
#   Note: nama_10r_2gva was retired by Eurostat. The NUTS3 dataset also
#   contains NUTS2-level aggregates, so we filter for our NUTS2 codes.
# Source: Eurostat - free, no API key required
#
# Strategy: Compute industrial share = GVA_industry / GVA_total
# Higher values = more industrialized regions
# NACE sectors: B-E = Industry (except construction)
# =============================================================================

print("Fetching regional GVA data from Eurostat (nama_10r_3gva)...")

gva_df = eurostat.get_data_df('nama_10r_3gva', flags=False)
print(f"Raw dataset shape: {gva_df.shape}")

# Identify columns
geo_col = [c for c in gva_df.columns if 'geo' in c.lower()][0]

# Filter for Spanish NUTS2 regions (present as aggregates in the NUTS3 dataset)
spain_gva = gva_df[
    gva_df[geo_col].isin(SPAIN_NUTS2.keys())
].copy()

# Filter for current prices in million EUR
if 'unit' in spain_gva.columns:
    spain_gva = spain_gva[spain_gva['unit'] == 'CP_MEUR']

# Get latest year
year_cols = [c for c in spain_gva.columns if isinstance(c, (int, float)) or 
             (isinstance(c, str) and len(c) == 4 and c.isdigit())]
latest_year = sorted(year_cols)[-1]
print(f"Using latest available year: {latest_year}")

# Get total GVA and industrial GVA (NACE B-E)
if 'nace_r2' in spain_gva.columns:
    nace_col = 'nace_r2'
else:
    nace_col = [c for c in spain_gva.columns if 'nace' in c.lower()][0]

# Total GVA
gva_total = spain_gva[spain_gva[nace_col] == 'TOTAL'][[geo_col, latest_year]].copy()
gva_total.columns = ['nuts2_code', 'gva_total']
gva_total['gva_total'] = pd.to_numeric(gva_total['gva_total'], errors='coerce')

# Industrial GVA (B-E: Mining, Manufacturing, Energy, Water)
gva_industry = spain_gva[spain_gva[nace_col] == 'B-E'][[geo_col, latest_year]].copy()
gva_industry.columns = ['nuts2_code', 'gva_industry']
gva_industry['gva_industry'] = pd.to_numeric(gva_industry['gva_industry'], errors='coerce')

# Agricultural GVA (A) - store for reuse in Agricultural cell
gva_agri_raw = spain_gva[spain_gva[nace_col] == 'A'][[geo_col, latest_year]].copy()
gva_agri_raw.columns = ['nuts2_code', 'gva_agriculture']
gva_agri_raw['gva_agriculture'] = pd.to_numeric(gva_agri_raw['gva_agriculture'], errors='coerce')

# Compute industrial index (share of industry in total GVA)
industrial_features = gva_total.merge(gva_industry, on='nuts2_code', how='left')
industrial_features['industrial_index'] = (
    industrial_features['gva_industry'] / industrial_features['gva_total']
)

# Normalize to 0-1 scale (min-max across regions)
idx_min = industrial_features['industrial_index'].min()
idx_max = industrial_features['industrial_index'].max()
industrial_features['industrial_index_normalized'] = (
    (industrial_features['industrial_index'] - idx_min) / (idx_max - idx_min)
)

print(f"\nIndustrial index computed for {len(industrial_features)} regions")
print(f"Most industrialized: {industrial_features.loc[industrial_features['industrial_index'].idxmax(), 'nuts2_code']}")
print(f"Least industrialized: {industrial_features.loc[industrial_features['industrial_index'].idxmin(), 'nuts2_code']}")
display(industrial_features)

# COMMAND ----------

# DBTITLE 1,Fetch GDP per Capita by Region (Eurostat)
# =============================================================================
# GDP PER CAPITA by NUTS2 region
# Dataset: nama_10r_2gdp (GDP at current market prices by NUTS 2 regions)
# Source: Eurostat - free, no API key required
#
# Higher GDP per capita → more appliances, AC, heat pumps, EVs → higher load
# =============================================================================

print("Fetching GDP data from Eurostat (nama_10r_2gdp)...")

gdp_df = eurostat.get_data_df('nama_10r_2gdp', flags=False)
print(f"Raw dataset shape: {gdp_df.shape}")

# Identify columns
geo_col = [c for c in gdp_df.columns if 'geo' in c.lower()][0]

# Filter for Spanish NUTS2 regions and EUR_HAB (euro per inhabitant)
spain_gdp = gdp_df[
    gdp_df[geo_col].isin(SPAIN_NUTS2.keys())
].copy()

# Filter for GDP per capita unit (EUR per inhabitant)
if 'unit' in spain_gdp.columns:
    # Try EUR_HAB first (euro per inhabitant), then MIO_EUR
    if 'EUR_HAB' in spain_gdp['unit'].values:
        spain_gdp = spain_gdp[spain_gdp['unit'] == 'EUR_HAB']
        gdp_metric = 'gdp_per_capita_eur'
    elif 'MIO_EUR' in spain_gdp['unit'].values:
        spain_gdp = spain_gdp[spain_gdp['unit'] == 'MIO_EUR']
        gdp_metric = 'gdp_mio_eur'
    else:
        # Use whatever unit is available
        available_units = spain_gdp['unit'].unique()
        print(f"Available units: {available_units}")
        spain_gdp = spain_gdp[spain_gdp['unit'] == available_units[0]]
        gdp_metric = 'gdp_value'
else:
    gdp_metric = 'gdp_value'

# Get latest year
year_cols = [c for c in spain_gdp.columns if isinstance(c, str) and len(c) == 4 and c.isdigit()]
if not year_cols:
    year_cols = [c for c in spain_gdp.columns if str(c).isdigit()]
latest_year = sorted(year_cols)[-1]
print(f"Using latest available year: {latest_year} | Metric: {gdp_metric}")

# Build GDP DataFrame
gdp_features = spain_gdp[[geo_col, latest_year]].copy()
gdp_features.columns = ['nuts2_code', gdp_metric]
gdp_features[gdp_metric] = pd.to_numeric(gdp_features[gdp_metric], errors='coerce')

# If we got MIO_EUR, compute per capita ourselves
if gdp_metric == 'gdp_mio_eur':
    gdp_features = gdp_features.merge(population_df[['nuts2_code', 'total_population']], on='nuts2_code', how='left')
    gdp_features['gdp_per_capita_eur'] = (
        gdp_features['gdp_mio_eur'] * 1_000_000 / gdp_features['total_population']
    )
    gdp_features = gdp_features[['nuts2_code', 'gdp_per_capita_eur']].copy()
elif gdp_metric != 'gdp_per_capita_eur':
    gdp_features.columns = ['nuts2_code', 'gdp_per_capita_eur']

# Normalize to 0-1 scale
gmin = gdp_features['gdp_per_capita_eur'].min()
gmax = gdp_features['gdp_per_capita_eur'].max()
gdp_features['gdp_per_capita_normalized'] = (gdp_features['gdp_per_capita_eur'] - gmin) / (gmax - gmin)

print(f"\nGDP per capita fetched for {len(gdp_features)} regions")
print(f"Highest: {gdp_features.loc[gdp_features['gdp_per_capita_eur'].idxmax(), 'nuts2_code']} "
      f"({gdp_features['gdp_per_capita_eur'].max():,.0f} EUR)")
print(f"Lowest:  {gdp_features.loc[gdp_features['gdp_per_capita_eur'].idxmin(), 'nuts2_code']} "
      f"({gdp_features['gdp_per_capita_eur'].min():,.0f} EUR)")
display(gdp_features.sort_values('gdp_per_capita_eur', ascending=False))

# COMMAND ----------

# DBTITLE 1,Fetch Heating & Cooling Degree Days (Eurostat)
# =============================================================================
# HOURLY HEATING & COOLING DEGREE DAYS from existing hourly weather data
# Source: datathon.rubber_duckers.openmeteo_hourly_weather
#
# This table already has hourly temperature_2m for Spanish communities
# from 2025-01-01 to 2026-02-28. We compute HDD/CDD at hourly resolution:
#   hdd_hourly = max(0, 18°C - temperature_2m)   [EU standard base]
#   cdd_hourly = max(0, temperature_2m - 22°C)   [cooling threshold]
#
# Hourly resolution is more accurate than daily-mean-based methods because
# it captures within-day temperature swings (e.g., cold mornings + hot
# afternoons that cancel out in a daily average).
#
# No external API calls needed - data already in your lakehouse!
# Note: Only regions present in the weather table are included.
#       ES43 (Extremadura) and ES53 (Baleares) are NOT in the source data.
# =============================================================================

# Mapping from community_code in the weather table to NUTS2 codes
COMMUNITY_TO_NUTS2 = {
    'AN': 'ES61',  # Andalucía
    'AR': 'ES24',  # Aragón
    'AS': 'ES12',  # Asturias
    'CB': 'ES13',  # Cantabria
    'CE': 'ES63',  # Ceuta
    'CL': 'ES41',  # Castilla y León
    'CM': 'ES42',  # Castilla-la Mancha
    'CN': 'ES70',  # Canarias
    'CT': 'ES51',  # Cataluña
    'GA': 'ES11',  # Galicia
    'MC': 'ES62',  # Murcia
    'MD': 'ES30',  # Madrid
    'ML': 'ES64',  # Melilla
    'NC': 'ES22',  # Navarra
    'PV': 'ES21',  # País Vasco
    'RI': 'ES23',  # La Rioja
    'VC': 'ES52',  # Comunitat Valenciana
}

# HDD/CDD base temperatures
HDD_BASE_TEMP = 18.0  # Heating needed below this
CDD_BASE_TEMP = 22.0  # Cooling needed above this

print("Computing HOURLY HDD/CDD from datathon.rubber_duckers.openmeteo_hourly_weather...")
print(f"  HDD base: {HDD_BASE_TEMP}°C | CDD base: {CDD_BASE_TEMP}°C")
print(f"  Period: {DATE_START} to {DATE_END}")

# Query: hourly temperature with HDD/CDD computed per hour
hdd_cdd_query = f"""
SELECT 
    datetime_local,
    date,
    community_code,
    temperature_2m,
    GREATEST(0, {HDD_BASE_TEMP} - temperature_2m) AS hdd_hourly,
    GREATEST(0, temperature_2m - {CDD_BASE_TEMP}) AS cdd_hourly
FROM datathon.rubber_duckers.openmeteo_hourly_weather
WHERE date BETWEEN '{DATE_START}' AND '{DATE_END}'
ORDER BY community_code, datetime_local
"""

hdd_cdd_spark = spark.sql(hdd_cdd_query)
hdd_cdd_hourly = hdd_cdd_spark.toPandas()

# Map community_code to nuts2_code
hdd_cdd_hourly['nuts2_code'] = hdd_cdd_hourly['community_code'].map(COMMUNITY_TO_NUTS2)
hdd_cdd_hourly['datetime_local'] = pd.to_datetime(hdd_cdd_hourly['datetime_local'])
hdd_cdd_hourly['date'] = pd.to_datetime(hdd_cdd_hourly['date'])

print(f"\nQuery returned: {len(hdd_cdd_hourly)} rows "
      f"({hdd_cdd_hourly['community_code'].nunique()} regions x "
      f"{hdd_cdd_hourly['datetime_local'].nunique()} hours)")

# Report missing regions (no proxy/fabrication - just inform)
missing_nuts2 = set(SPAIN_NUTS2.keys()) - set(hdd_cdd_hourly['nuts2_code'].dropna().unique())
if missing_nuts2:
    print(f"\nRegions NOT in weather table (excluded, no proxy): {missing_nuts2}")

# --- Summary ---
print(f"\n{'='*60}")
print(f"Hourly HDD/CDD from existing hourly weather data:")
print(f"  Total records: {len(hdd_cdd_hourly)}")
print(f"  Regions: {hdd_cdd_hourly['nuts2_code'].nunique()}")
print(f"  Date range: {hdd_cdd_hourly['date'].min().date()} to {hdd_cdd_hourly['date'].max().date()}")
print(f"\n  Temperature: {hdd_cdd_hourly['temperature_2m'].mean():.1f}°C avg "
      f"(min {hdd_cdd_hourly['temperature_2m'].min():.1f}°C, max {hdd_cdd_hourly['temperature_2m'].max():.1f}°C)")
print(f"  HDD hourly:  {hdd_cdd_hourly['hdd_hourly'].mean():.2f} avg (max {hdd_cdd_hourly['hdd_hourly'].max():.1f})")
print(f"  CDD hourly:  {hdd_cdd_hourly['cdd_hourly'].mean():.2f} avg (max {hdd_cdd_hourly['cdd_hourly'].max():.1f})")

# Show monthly pattern for Madrid (aggregate hourly -> daily sums for comparison)
print(f"\nMadrid (ES30) monthly HDD/CDD pattern (summed from hourly):")
madrid = hdd_cdd_hourly[hdd_cdd_hourly['nuts2_code'] == 'ES30'].copy()
madrid['month'] = madrid['date'].dt.month
madrid_daily = madrid.groupby(['date', 'month']).agg(
    avg_temp=('temperature_2m', 'mean'),
    total_hdd=('hdd_hourly', 'sum'),
    total_cdd=('cdd_hourly', 'sum')
).reset_index()
madrid_monthly = madrid_daily.groupby('month').agg(
    avg_temp=('avg_temp', 'mean'),
    total_hdd=('total_hdd', 'sum'),
    total_cdd=('total_cdd', 'sum')
).round(1)
display(madrid_monthly)

# Keep backward compatibility: hdd_cdd_features (static annual summary)
# Sum all hourly values per region (total degree-hours over the period)
hdd_cdd_features = hdd_cdd_hourly.groupby('nuts2_code').agg(
    hdd=('hdd_hourly', 'sum'),
    cdd=('cdd_hourly', 'sum')
).reset_index()
for col in ['hdd', 'cdd']:
    cmin = hdd_cdd_features[col].min()
    cmax = hdd_cdd_features[col].max()
    hdd_cdd_features[f'{col}_normalized'] = (hdd_cdd_features[col] - cmin) / (cmax - cmin) if cmax > cmin else 0.5

# Also keep monthly aggregation for reference
hdd_cdd_monthly = hdd_cdd_hourly.copy()
hdd_cdd_monthly['year'] = hdd_cdd_monthly['date'].dt.year
hdd_cdd_monthly['month'] = hdd_cdd_monthly['date'].dt.month
hdd_cdd_monthly = hdd_cdd_monthly.groupby(['nuts2_code', 'year', 'month']).agg(
    hdd_monthly=('hdd_hourly', 'sum'),
    cdd_monthly=('cdd_hourly', 'sum')
).reset_index()

# COMMAND ----------

# DBTITLE 1,Fetch Agricultural GVA Share by Region (Eurostat)
# =============================================================================
# AGRICULTURAL SHARE - GVA from Agriculture / Total GVA at NUTS2
# Reuses gva_total and gva_agri_raw fetched in the Industrial Index cell
# (both from Eurostat nama_10r_2gva via REST API)
#
# Agricultural regions have seasonal electricity patterns (irrigation pumps,
# greenhouses, cold storage) that differ from urban/industrial profiles.
# This is annual data, applied as a constant regional feature.
# =============================================================================

print("Computing agricultural share from GVA data (reusing REST API fetch)...")

# gva_total and gva_agri_raw were already fetched in the Industrial Index cell
agri_features = gva_total.merge(gva_agri_raw, on='nuts2_code', how='left')
agri_features['agricultural_share'] = (
    agri_features['gva_agriculture'] / agri_features['gva_total']
)

# Normalize
amin = agri_features['agricultural_share'].min()
amax = agri_features['agricultural_share'].max()
agri_features['agricultural_share_normalized'] = (
    (agri_features['agricultural_share'] - amin) / (amax - amin)
)

print(f"Agricultural share computed for {len(agri_features)} regions")
print(f"Most agricultural: {agri_features.loc[agri_features['agricultural_share'].idxmax(), 'nuts2_code']}")
print(f"Least agricultural: {agri_features.loc[agri_features['agricultural_share'].idxmin(), 'nuts2_code']}")
display(agri_features[['nuts2_code', 'agricultural_share', 'agricultural_share_normalized']].sort_values('agricultural_share', ascending=False))

# COMMAND ----------

# DBTITLE 1,Combine all features into Regional Instrument DataFrame
# =============================================================================
# COMBINE ALL FEATURES INTO TWO SEPARATE TABLES
#
# Output 1: regional_features (STATIC) - annual socio-economic indicators
#           One row per region. Join on nuts2_code.
# Output 2: regional_features_temporal (TEMPORAL) - hourly time series only
#           datetime_local, date, nuts2_code, temperature_2m, hdd_hourly, cdd_hourly
#           No static columns duplicated here.
#
# At model training time, join both tables on nuts2_code:
#   temporal JOIN static USING (nuts2_code)
# =============================================================================

# ---- STATIC FEATURES (one row per region) ----
regional_features = pop_features[['nuts2_code', 'region_name', 'population_density', 'urban_rural_class']].copy()

regional_features = regional_features.merge(
    population_df[['nuts2_code', 'total_population']], on='nuts2_code', how='left')

tourism_cols = ['nuts2_code', 'is_holiday_destination']
if 'tourism_nights_per_capita' in tourism_features.columns:
    tourism_cols.insert(1, 'tourism_nights_per_capita')
regional_features = regional_features.merge(
    tourism_features[tourism_cols], on='nuts2_code', how='left')

ind_cols = ['nuts2_code', 'industrial_index']
if 'industrial_index_normalized' in industrial_features.columns:
    ind_cols.append('industrial_index_normalized')
regional_features = regional_features.merge(
    industrial_features[ind_cols], on='nuts2_code', how='left')

gdp_cols = ['nuts2_code', 'gdp_per_capita_eur']
if 'gdp_per_capita_normalized' in gdp_features.columns:
    gdp_cols.append('gdp_per_capita_normalized')
regional_features = regional_features.merge(
    gdp_features[gdp_cols], on='nuts2_code', how='left')

hdd_ccd_static_cols = ['nuts2_code']
for col in ['hdd', 'cdd', 'hdd_normalized', 'cdd_normalized']:
    if col in hdd_cdd_features.columns:
        hdd_ccd_static_cols.append(col)
regional_features = regional_features.merge(
    hdd_cdd_features[hdd_ccd_static_cols], on='nuts2_code', how='left')

agri_cols = ['nuts2_code', 'agricultural_share']
if 'agricultural_share_normalized' in agri_features.columns:
    agri_cols.append('agricultural_share_normalized')
regional_features = regional_features.merge(
    agri_features[agri_cols], on='nuts2_code', how='left')

urban_dummies = pd.get_dummies(regional_features['urban_rural_class'], prefix='is')
regional_features = pd.concat([regional_features, urban_dummies], axis=1)
for col in urban_dummies.columns:
    regional_features[col] = regional_features[col].astype(int)

# ---- TEMPORAL FEATURES (hourly, Jan 2025 - Feb 2026) ----
# Only real observed time-varying data — NO static columns merged in
regional_features_temporal = hdd_cdd_hourly[['datetime_local', 'date', 'nuts2_code', 'temperature_2m', 'hdd_hourly', 'cdd_hourly']].copy()
regional_features_temporal = regional_features_temporal.sort_values(['nuts2_code', 'datetime_local']).reset_index(drop=True)

# =============================================================================
# SUMMARY
# =============================================================================
print("=" * 70)
print("REGIONAL FEATURES — CLEAN SEPARATION")
print("=" * 70)

print(f"\n--- STATIC TABLE ({len(regional_features)} regions, {len(regional_features.columns)} cols) ---")
print(f"Columns: {list(regional_features.columns)}")

print(f"\n--- TEMPORAL TABLE ({len(regional_features_temporal)} rows, {len(regional_features_temporal.columns)} cols) ---")
print(f"Columns: {list(regional_features_temporal.columns)}")
print(f"Date range: {regional_features_temporal['date'].min().date()} to {regional_features_temporal['date'].max().date()}")
print(f"Regions: {regional_features_temporal['nuts2_code'].nunique()}")

print(f"\n--- JOIN AT TRAINING TIME ---")
print(f"  SELECT t.*, s.*")
print(f"  FROM regional_features_temporal t")
print(f"  JOIN regional_features_static s USING (nuts2_code)")
print("=" * 70)

print("\nStatic features preview:")
display(regional_features.head(5))

print("\nTemporal features preview (Madrid, 2025-01-01):")
display(
    regional_features_temporal[
        (regional_features_temporal['nuts2_code'] == 'ES30') &
        (regional_features_temporal['date'] == pd.Timestamp('2025-01-01'))
    ].head(10)
)

# COMMAND ----------

# DBTITLE 1,Save to Delta table in Unity Catalog
# =============================================================================
# SAVE REGIONAL FEATURES AS DELTA TABLES IN UNITY CATALOG
#
# Output tables:
#   1. regional_features_static   - 19 rows, socio-economic indicators per region
#   2. regional_features_temporal  - hourly weather time series (6 columns)
#
# Static and temporal are kept SEPARATE. Join on nuts2_code at query time.
# =============================================================================

# ---- CONFIGURE TARGET LOCATION ----
CATALOG = "datathon"
SCHEMA = "rubber_duckers"

TABLE_STATIC = f"{CATALOG}.{SCHEMA}.regional_features_static"
TABLE_TEMPORAL = f"{CATALOG}.{SCHEMA}.regional_features_temporal"

# ---- Convert pandas DataFrames to Spark ----
static_spark = spark.createDataFrame(regional_features)

temporal_save = regional_features_temporal.copy()
temporal_save['date'] = pd.to_datetime(temporal_save['date']).dt.date
temporal_spark = spark.createDataFrame(temporal_save)

# ---- Write as Delta tables (overwrite mode) ----
print(f"Writing static features to: {TABLE_STATIC}")
static_spark.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(TABLE_STATIC)
print(f"  ✓ {TABLE_STATIC} ({static_spark.count()} rows, {len(static_spark.columns)} columns)")
print(f"  Columns: {static_spark.columns}")

print(f"\nWriting temporal features to: {TABLE_TEMPORAL}")
temporal_spark.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(TABLE_TEMPORAL)
print(f"  ✓ {TABLE_TEMPORAL} ({temporal_spark.count()} rows, {len(temporal_spark.columns)} columns)")
print(f"  Columns: {temporal_spark.columns}")

print(f"\n{'='*70}")
print("Tables saved successfully!")
print(f"\nJoin example for model training:")
print(f"  SELECT t.datetime_local, t.nuts2_code, t.temperature_2m, t.hdd_hourly, t.cdd_hourly,")
print(f"         s.population_density, s.industrial_index, s.gdp_per_capita_eur, ...")
print(f"  FROM {TABLE_TEMPORAL} t")
print(f"  JOIN {TABLE_STATIC} s ON t.nuts2_code = s.nuts2_code")