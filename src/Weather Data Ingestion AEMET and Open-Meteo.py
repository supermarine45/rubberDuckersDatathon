# Databricks notebook source
# DBTITLE 1,AEMET Weather Data Ingestion
# MAGIC %md
# MAGIC # AEMET Weather Data Ingestion
# MAGIC
# MAGIC This notebook fetches **daily weather data** from the [AEMET OpenData API](https://opendata.aemet.es/) (Agencia Estatal de Meteorología — Spanish Meteorological Agency) for use in energy consumption prediction.
# MAGIC
# MAGIC **Purpose:** Enrich the client consumption dataset (`datathon.shared.client_consumption`) with weather features (temperature, precipitation, wind, humidity, sunshine hours) that are strong predictors of energy demand.
# MAGIC
# MAGIC **Coverage:**
# MAGIC - **Date range:** 2025-01-01 to 2026-02-28 (training + prediction periods)
# MAGIC - **Geography:** One representative AEMET weather station per autonomous community present in the dataset (18 communities)
# MAGIC - **Output:** Delta table `datathon.rubber_duckers.aemet_daily_weather`
# MAGIC
# MAGIC **Prerequisites:** A valid AEMET API key (free at https://opendata.aemet.es/centrodedescargas/altaUsuario)

# COMMAND ----------

# DBTITLE 1,Configure AEMET API Key
api_key = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJmcmFuc2lza3VzYWRyaWFuLmd1bmF3YW5AdXpoLmNoIiwianRpIjoiYmQwYjRlYmYtOTEwZi00YmFmLTliZjQtODMwM2ZjOWQ4YmE5IiwiaXNzIjoiQUVNRVQiLCJpYXQiOjE3NzY1MTY5NjUsInVzZXJJZCI6ImJkMGI0ZWJmLTkxMGYtNGJhZi05YmY0LTgzMDNmYzlkOGJhOSIsInJvbGUiOiIifQ.lYb_OyPcl9O39o3XlfqDweUsYUr-pJB_YCmmscxKe8c"

if not api_key:
    raise ValueError("Please paste your AEMET API key into the widget above before running subsequent cells.")
else:
    print(f"API key configured (length={len(api_key)})")

# COMMAND ----------

# DBTITLE 1,Imports
import requests
import pandas as pd
import json
import time
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

print("All imports loaded.")

# COMMAND ----------

# DBTITLE 1,Community Code to AEMET Station Mapping
# Query distinct community codes from the consumption dataset
community_codes_df = spark.sql("SELECT DISTINCT community_code FROM datathon.shared.client_consumption ORDER BY community_code")
community_codes = [row.community_code for row in community_codes_df.collect()]
print(f"Community codes in dataset ({len(community_codes)}): {community_codes}")

# Mapping: community_code -> (AEMET station indicativo, station description)
# Each station is a major airport or observatory in the autonomous community's capital/major city
COMMUNITY_STATION_MAP = {
    "AN": ("5783",  "Sevilla/Aeropuerto"),           # Andalucía
    "AR": ("9434",  "Zaragoza/Aeropuerto"),           # Aragón
    "AS": ("1249X", "Oviedo"),                         # Asturias
    "CB": ("1109",  "Santander/Aeropuerto"),           # Cantabria
    "CE": ("5000C", "Ceuta"),                           # Ceuta
    "CL": ("2422",  "Valladolid/Aeropuerto"),          # Castilla y León
    "CM": ("4121",  "Ciudad Real"),                     # Castilla-La Mancha
    "CN": ("C649I", "Gran Canaria/Aeropuerto"),        # Canarias
    "CT": ("0076",  "Barcelona/Aeropuerto"),            # Cataluña
    "EX": ("4452",  "Badajoz/Aeropuerto"),              # Extremadura
    "GA": ("1428",  "Santiago/Aeropuerto"),             # Galicia
    "MC": ("7178I", "Murcia"),                           # Murcia
    "MD": ("3129",  "Madrid/Retiro"),                   # Madrid
    "ML": ("6000A", "Melilla"),                          # Melilla
    "NC": ("9262",  "Pamplona/Aeropuerto"),             # Navarra
    "PV": ("1082",  "Bilbao/Aeropuerto"),               # País Vasco
    "RI": ("9170",  "Logroño/Aeropuerto"),              # La Rioja
    "VC": ("8416",  "Valencia/Aeropuerto"),             # Comunidad Valenciana
}

# Validate all community codes have a station
missing = [c for c in community_codes if c not in COMMUNITY_STATION_MAP]
if missing:
    print(f"WARNING: No station mapping for community codes: {missing}")
else:
    print(f"All {len(community_codes)} community codes mapped to AEMET stations.")

for code in community_codes:
    station_id, desc = COMMUNITY_STATION_MAP[code]
    print(f"  {code} -> {station_id} ({desc})")

# COMMAND ----------

# DBTITLE 1,AEMET API Helper Functions
AEMET_BASE_URL = "https://opendata.aemet.es/opendata/api"

def fetch_aemet_data(url, api_key, max_retries=3):
    """
    Makes a GET request to an AEMET API endpoint.
    AEMET returns an intermediate JSON with a 'datos' URL pointing to actual data.
    This function follows that redirect and returns the parsed data.
    """
    headers = {"api_key": api_key}

    for attempt in range(1, max_retries + 1):
        try:
            # Step 1: Call AEMET endpoint to get the intermediate response
            resp = requests.get(url, headers=headers, timeout=30)

            if resp.status_code == 429:  # Rate limited
                wait = 10 * attempt
                print(f"  Rate limited. Waiting {wait}s (attempt {attempt}/{max_retries})...")
                time.sleep(wait)
                continue

            if resp.status_code != 200:
                print(f"  HTTP {resp.status_code}: {resp.text[:200]}")
                if attempt < max_retries:
                    time.sleep(5 * attempt)
                    continue
                return None

            meta = resp.json()

            # Check for API-level errors
            if meta.get("estado") == 404:
                print(f"  No data available (404): {meta.get('descripcion', '')}")
                return None

            datos_url = meta.get("datos")
            if not datos_url:
                print(f"  No 'datos' URL in response: {meta}")
                return None

            # Step 2: Fetch actual data from the datos URL
            # AEMET returns data in ISO-8859-15 encoding
            data_resp = requests.get(datos_url, timeout=30)
            data_resp.encoding = "ISO-8859-15"

            if data_resp.status_code != 200:
                print(f"  Failed to fetch datos URL: HTTP {data_resp.status_code}")
                if attempt < max_retries:
                    time.sleep(5 * attempt)
                    continue
                return None

            return data_resp.json()

        except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
            print(f"  Error (attempt {attempt}/{max_retries}): {e}")
            if attempt < max_retries:
                time.sleep(5 * attempt)
            else:
                return None

    return None


def get_daily_weather(station_id, start_date, end_date, api_key):
    """
    Fetches daily climatological values for a station within a date range.
    AEMET API format: fechaini/YYYY-MM-DDTHH:MM:SSUTC/fechafin/YYYY-MM-DDTHH:MM:SSUTC
    """
    start_str = f"{start_date}T00:00:00UTC"
    end_str = f"{end_date}T23:59:59UTC"
    url = (
        f"{AEMET_BASE_URL}/valores/climatologicos/diarios/datos"
        f"/fechaini/{start_str}/fechafin/{end_str}/estacion/{station_id}"
    )
    return fetch_aemet_data(url, api_key)


print("Helper functions defined: fetch_aemet_data(), get_daily_weather()")

# COMMAND ----------

# DBTITLE 1,Fetch Weather Data for All Communities
# Date range covering training (Jan-Nov 2025) and prediction (Dec 2025 - Feb 2026)
DATE_START = datetime(2025, 1, 1)
DATE_END = datetime(2026, 2, 28)

all_records = []
errors = []

for community_code in community_codes:
    station_id, station_desc = COMMUNITY_STATION_MAP[community_code]
    print(f"\n{'='*60}")
    print(f"Fetching: {community_code} -> station {station_id} ({station_desc})")
    print(f"{'='*60}")

    # Chunk into monthly requests (AEMET limits to ~31 days per call)
    chunk_start = DATE_START
    community_records = 0

    while chunk_start <= DATE_END:
        # End of this chunk: last day of the month or DATE_END
        chunk_end = min(
            (chunk_start + relativedelta(months=1)) - timedelta(days=1),
            DATE_END
        )

        start_str = chunk_start.strftime("%Y-%m-%d")
        end_str = chunk_end.strftime("%Y-%m-%d")
        print(f"  Requesting {start_str} to {end_str}...", end=" ")

        data = get_daily_weather(station_id, start_str, end_str, api_key)

        if data and isinstance(data, list):
            for record in data:
                record["community_code"] = community_code
            all_records.extend(data)
            community_records += len(data)
            print(f"{len(data)} records")
        else:
            print("No data returned")
            errors.append((community_code, start_str, end_str))

        # Move to next month
        chunk_start = chunk_end + timedelta(days=1)

        # Rate limiting: respect AEMET's API limits
        time.sleep(1.5)

    print(f"  Total for {community_code}: {community_records} records")

print(f"\n{'='*60}")
print(f"DONE: {len(all_records)} total records collected")
if errors:
    print(f"Failed chunks ({len(errors)}):")
    for cc, s, e in errors:
        print(f"  {cc}: {s} to {e}")

# COMMAND ----------

# DBTITLE 1,Create Spark DataFrame and Save as Delta Table
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, DateType

if not all_records:
    raise RuntimeError("No records were collected. Check API key and errors above.")

# Convert to pandas DataFrame — keep ALL columns from AEMET
pdf = pd.DataFrame(all_records)
print(f"Raw DataFrame: {pdf.shape[0]} rows, {pdf.shape[1]} columns")
print(f"\nAll columns returned by AEMET:\n{sorted(pdf.columns.tolist())}")

# Columns that should stay as strings (identifiers, timestamps, names)
string_cols = {"fecha", "community_code", "indicativo", "nombre", "provincia",
               "altitud", "horatmin", "horatmax", "horaracha",
               "horaPresMax", "horaPresMin", "horaHrMax", "horaHrMin", "dir"}

# Convert Spanish decimal commas to periods for all numeric columns
for col in pdf.columns:
    if col not in string_cols:
        pdf[col] = (
            pdf[col]
            .astype(str)
            .str.replace(",", ".", regex=False)
            .str.strip()
        )
        # Handle "Ip" (trace precipitation), "Varias" (variable), "Acum" (accumulated), etc.
        pdf[col] = pd.to_numeric(pdf[col], errors="coerce")

print(f"\nProcessed DataFrame: {pdf.shape[0]} rows, {pdf.shape[1]} columns")
print(f"\nSample dtypes:\n{pdf.dtypes}")

# Convert to Spark DataFrame
sdf = spark.createDataFrame(pdf)

# Cast fecha to proper date type
sdf = sdf.withColumn("fecha", F.to_date(F.col("fecha"), "yyyy-MM-dd"))

# Drop rows with no date
sdf = sdf.filter(F.col("fecha").isNotNull())

print(f"\nFinal Spark DataFrame: {sdf.count()} rows")
sdf.printSchema()

# Save as Delta table with ALL columns
TABLE_NAME = "datathon.rubber_duckers.aemet_daily_weather"
sdf.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(TABLE_NAME)
print(f"\nSaved to {TABLE_NAME}")

# COMMAND ----------

# DBTITLE 1,Validate and Display Results
# Read back and validate
result_df = spark.table("datathon.rubber_duckers.aemet_daily_weather")

print(f"Total rows: {result_df.count()}")
print(f"Columns: {result_df.columns}")
print(f"Date range: {result_df.agg(F.min('fecha'), F.max('fecha')).collect()[0]}")
print(f"\nRows per community_code:")

display(
    result_df
    .groupBy("community_code")
    .agg(
        F.count("*").alias("row_count"),
        F.min("fecha").alias("min_date"),
        F.max("fecha").alias("max_date"),
        F.round(F.avg("tmed"), 1).alias("avg_temp_c")
    )
    .orderBy("community_code")
)

print("\nSample data:")
display(result_df.orderBy("community_code", "fecha").limit(20))

# COMMAND ----------

# DBTITLE 1,Open-Meteo Hourly Weather Data
# MAGIC %md
# MAGIC ---
# MAGIC # Open-Meteo: Hourly Historical Weather Data
# MAGIC
# MAGIC [Open-Meteo](https://open-meteo.com/) provides **free hourly historical weather data** with no API key required. This section fetches hourly data for the same 18 communities, covering Jan 2025 – Feb 2026.
# MAGIC
# MAGIC **Output:** Delta table `datathon.rubber_duckers.openmeteo_hourly_weather`
# MAGIC
# MAGIC **Variables fetched:** Temperature, humidity, dew point, apparent temperature, precipitation, rain, snowfall, snow depth, weather code, pressure, cloud cover (total/low/mid/high), wind speed & direction (10m/100m), wind gusts, ET₀, vapour pressure deficit, soil temperature & moisture, radiation (direct/diffuse/normal/shortwave/terrestrial), sunshine duration, and more.

# COMMAND ----------

# DBTITLE 1,Community Coordinates for Open-Meteo
# Coordinates for each community's representative city (matching AEMET stations)
COMMUNITY_COORDS = {
    "AN": {"lat": 37.39, "lon": -5.99, "city": "Sevilla"},
    "AR": {"lat": 41.66, "lon": -0.88, "city": "Zaragoza"},
    "AS": {"lat": 43.36, "lon": -5.85, "city": "Oviedo"},
    "CB": {"lat": 43.46, "lon": -3.82, "city": "Santander"},
    "CE": {"lat": 35.89, "lon": -5.32, "city": "Ceuta"},
    "CL": {"lat": 41.65, "lon": -4.72, "city": "Valladolid"},
    "CM": {"lat": 38.99, "lon": -3.93, "city": "Ciudad Real"},
    "CN": {"lat": 28.00, "lon": -15.41, "city": "Las Palmas"},
    "CT": {"lat": 41.39, "lon":  2.16, "city": "Barcelona"},
    "EX": {"lat": 38.88, "lon": -6.97, "city": "Badajoz"},
    "GA": {"lat": 42.88, "lon": -8.54, "city": "Santiago"},
    "MC": {"lat": 37.98, "lon": -1.13, "city": "Murcia"},
    "MD": {"lat": 40.41, "lon": -3.70, "city": "Madrid"},
    "ML": {"lat": 35.29, "lon": -2.94, "city": "Melilla"},
    "NC": {"lat": 42.81, "lon": -1.64, "city": "Pamplona"},
    "PV": {"lat": 43.26, "lon": -2.93, "city": "Bilbao"},
    "RI": {"lat": 42.47, "lon": -2.44, "city": "Logro\u00f1o"},
    "VC": {"lat": 39.47, "lon": -0.38, "city": "Valencia"},
}

print(f"Coordinates defined for {len(COMMUNITY_COORDS)} communities.")
for code, info in sorted(COMMUNITY_COORDS.items()):
    print(f"  {code} -> {info['city']} ({info['lat']}, {info['lon']})")

# COMMAND ----------

# DBTITLE 1,Fetch Open-Meteo Hourly Data for All Communities
import requests
import pandas as pd
import time

OPENMETEO_URL = "https://archive-api.open-meteo.com/v1/archive"

# ALL available hourly variables from Open-Meteo Historical API
HOURLY_VARS = [
    "temperature_2m", "relative_humidity_2m", "dew_point_2m",
    "apparent_temperature", "precipitation", "rain", "snowfall",
    "snow_depth", "weather_code", "pressure_msl", "surface_pressure",
    "cloud_cover", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high",
    "wind_speed_10m", "wind_speed_100m", "wind_direction_10m",
    "wind_direction_100m", "wind_gusts_10m",
    "et0_fao_evapotranspiration", "vapour_pressure_deficit",
    "soil_temperature_0cm", "soil_temperature_6cm",
    "soil_temperature_18cm", "soil_temperature_54cm",
    "soil_moisture_0_to_1cm", "soil_moisture_1_to_3cm",
    "soil_moisture_3_to_9cm", "soil_moisture_9_to_27cm",
    "soil_moisture_27_to_81cm",
    "shortwave_radiation", "direct_radiation", "diffuse_radiation",
    "direct_normal_irradiance", "global_tilted_irradiance",
    "terrestrial_radiation", "shortwave_radiation_instant",
    "direct_radiation_instant", "diffuse_radiation_instant",
    "direct_normal_irradiance_instant", "global_tilted_irradiance_instant",
    "terrestrial_radiation_instant",
    "sunshine_duration", "is_day",
]

START_DATE = "2025-01-01"
END_DATE = "2026-02-28"

all_hourly_dfs = []

for code in sorted(COMMUNITY_COORDS.keys()):
    info = COMMUNITY_COORDS[code]
    print(f"Fetching {code} ({info['city']})...", end=" ")

    params = {
        "latitude": info["lat"],
        "longitude": info["lon"],
        "start_date": START_DATE,
        "end_date": END_DATE,
        "hourly": ",".join(HOURLY_VARS),
        "timezone": "Europe/Madrid",
    }
    # Canarias is in Atlantic/Canary timezone
    if code == "CN":
        params["timezone"] = "Atlantic/Canary"

    for attempt in range(1, 4):
        try:
            resp = requests.get(OPENMETEO_URL, params=params, timeout=60)
            if resp.status_code == 200:
                data = resp.json()
                hourly = data.get("hourly", {})
                if hourly and "time" in hourly:
                    pdf = pd.DataFrame(hourly)
                    pdf["community_code"] = code
                    pdf["latitude"] = data.get("latitude")
                    pdf["longitude"] = data.get("longitude")
                    pdf["elevation"] = data.get("elevation")
                    all_hourly_dfs.append(pdf)
                    print(f"{len(pdf)} rows")
                else:
                    print("No hourly data in response")
                break
            elif resp.status_code == 429:
                wait = 10 * attempt
                print(f"Rate limited, waiting {wait}s...", end=" ")
                time.sleep(wait)
            else:
                print(f"HTTP {resp.status_code}: {resp.text[:150]}")
                if attempt < 3:
                    time.sleep(5)
                break
        except Exception as e:
            print(f"Error (attempt {attempt}): {e}")
            if attempt < 3:
                time.sleep(5)

    time.sleep(1)  # Be polite to the free API

if all_hourly_dfs:
    combined_pdf = pd.concat(all_hourly_dfs, ignore_index=True)
    print(f"\nTotal: {len(combined_pdf)} rows, {len(combined_pdf.columns)} columns")
    print(f"Columns: {sorted(combined_pdf.columns.tolist())}")
else:
    raise RuntimeError("No data collected from Open-Meteo. Check errors above.")

# COMMAND ----------

# DBTITLE 1,Save Open-Meteo Hourly Data as Delta Table
from pyspark.sql import functions as F

# ---- Step 1: Drop columns that are entirely null (not available for all locations) ----
null_counts = combined_pdf.isnull().sum()
total_rows = len(combined_pdf)
all_null_cols = null_counts[null_counts == total_rows].index.tolist()

if all_null_cols:
    print(f"Dropping {len(all_null_cols)} all-null columns: {all_null_cols}")
    combined_pdf = combined_pdf.drop(columns=all_null_cols)
else:
    print("No all-null columns found — all variables have data.")

# ---- Step 2: Report columns with partial nulls ----
partial_null = null_counts[(null_counts > 0) & (null_counts < total_rows)]
if not partial_null.empty:
    print(f"\nColumns with partial nulls (have data for most rows):")
    for col_name, n in partial_null.items():
        pct = n / total_rows * 100
        print(f"  {col_name}: {n} nulls ({pct:.1f}%)")

print(f"\nFinal columns ({len(combined_pdf.columns)}): {sorted(combined_pdf.columns.tolist())}")

# ---- Step 3: Convert to Spark DataFrame ----
sdf_hourly = spark.createDataFrame(combined_pdf)

# Parse the timestamp column
sdf_hourly = sdf_hourly.withColumn("time", F.to_timestamp("time"))

# Rename 'time' to 'datetime_local' for easy joining with consumption data
sdf_hourly = sdf_hourly.withColumnRenamed("time", "datetime_local")

# Add a date column for convenient joining with AEMET daily data
sdf_hourly = sdf_hourly.withColumn("date", F.to_date("datetime_local"))

# Drop rows with no timestamp
sdf_hourly = sdf_hourly.filter(F.col("datetime_local").isNotNull())

row_count = sdf_hourly.count()
print(f"\nSpark DataFrame: {row_count} rows, {len(sdf_hourly.columns)} columns")
print(f"\nDate range:")
sdf_hourly.select(F.min("datetime_local"), F.max("datetime_local")).show(truncate=False)
sdf_hourly.printSchema()

# ---- Step 4: Save as Delta table ----
TABLE_NAME = "datathon.rubber_duckers.openmeteo_hourly_weather"
sdf_hourly.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(TABLE_NAME)
print(f"\nSaved to {TABLE_NAME}")

# COMMAND ----------

# DBTITLE 1,Validate Open-Meteo Hourly Results
# Read the table fresh via SQL to avoid Photon column-mapping issues on wide tables
result_hourly = spark.sql("SELECT * FROM datathon.rubber_duckers.openmeteo_hourly_weather")

print(f"Total rows: {result_hourly.count()}")
print(f"Total columns: {len(result_hourly.columns)}")
print(f"Columns: {sorted(result_hourly.columns)}")

print(f"\nRows per community_code:")
display(
    spark.sql("""
        SELECT
            community_code,
            COUNT(*) AS row_count,
            MIN(datetime_local) AS min_datetime,
            MAX(datetime_local) AS max_datetime,
            ROUND(AVG(temperature_2m), 1) AS avg_temp_c,
            ROUND(AVG(wind_speed_10m), 1) AS avg_wind_kmh,
            ROUND(SUM(precipitation), 1) AS total_precip_mm
        FROM datathon.rubber_duckers.openmeteo_hourly_weather
        GROUP BY community_code
        ORDER BY community_code
    """)
)

print("\nSample data (first 10 rows):")
display(
    spark.sql("""
        SELECT * FROM datathon.rubber_duckers.openmeteo_hourly_weather
        ORDER BY community_code, datetime_local
        LIMIT 10
    """)
)