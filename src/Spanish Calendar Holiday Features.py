# Databricks notebook source
!pip install holidays

# COMMAND ----------

import pandas as pd
import holidays
from datetime import timedelta

def get_spanish_calendar_features(year=2026):
    # ── 1. All Spanish Autonomous Communities & Cities ──────────────────
    REGIONS = {
        'AN': 'Andalusia',       'AR': 'Aragon',          'AS': 'Asturias',
        'CB': 'Cantabria',       'CE': 'Ceuta',           'CL': 'Castile and León',
        'CM': 'Castile-La Mancha','CN': 'Canary Islands',  'CT': 'Catalonia',
        'EX': 'Extremadura',     'GA': 'Galicia',         'IB': 'Balearic Islands',
        'MC': 'Murcia',          'MD': 'Madrid',          'ML': 'Melilla',
        'NC': 'Navarre',         'PV': 'Basque Country',  'RI': 'La Rioja',
        'VC': 'Valencia',
    }
    region_codes = list(REGIONS.keys())

    # ── 2. Build calendars ─────────────────────────────────────────────
    es_nat = holidays.Spain(years=year)
    region_cals = {c: holidays.Spain(years=year, subdiv=c) for c in region_codes}

    # Bank extra closure dates
    bank_extra = set()
    bank_extra.add(pd.Timestamp(year, 12, 24))  # Christmas Eve
    bank_extra.add(pd.Timestamp(year, 12, 31))  # New Year's Eve
    gf = [d for d in es_nat if 'Good Friday' in es_nat.get(d, '')]
    if gf:
        bank_extra.add(pd.Timestamp(gf[0] - timedelta(days=1)))  # Holy Thursday

    # ── 3. Daily DataFrame (365 rows) ──────────────────────────────────
    df = pd.DataFrame({'date': pd.date_range(f'{year}-01-01', f'{year}-12-31')})

    # National holiday
    df['is_national_holiday'] = df['date'].apply(lambda d: 1 if d in es_nat else 0)
    df['national_holiday_name'] = df['date'].apply(lambda d: es_nat.get(d, ''))

    # ── 4. One-hot: regional holidays (holiday_XX) ─────────────────────
    for c in region_codes:
        cal = region_cals[c]
        df[f'holiday_{c}'] = df['date'].apply(lambda d, cal=cal: 1 if d in cal else 0)

    # ── 5. One-hot: bank holidays (bank_holiday_XX) ────────────────────
    for c in region_codes:
        cal = region_cals[c]
        df[f'bank_holiday_{c}'] = df['date'].apply(
            lambda d, cal=cal: 1 if (d in cal or d.weekday() == 6 or d in bank_extra) else 0
        )

    # National-level bank holiday
    df['is_bank_holiday'] = df['date'].apply(
        lambda d: 1 if (d in es_nat or d.weekday() == 6 or d in bank_extra) else 0
    )

    # ── 6. One-hot: bridge days (bridge_XX) ────────────────────────────
    for c in region_codes:
        hol = df[f'holiday_{c}'].values
        dow = df['date'].dt.weekday.values
        bridge = [0] * len(df)
        for i in range(len(df)):
            # Monday bridge if Tuesday is a holiday
            if dow[i] == 0 and i + 1 < len(df) and hol[i + 1] == 1:
                bridge[i] = 1
            # Friday bridge if Thursday is a holiday
            if dow[i] == 4 and i - 1 >= 0 and hol[i - 1] == 1:
                bridge[i] = 1
        df[f'bridge_{c}'] = bridge

    # ── 7. Aggregates ──────────────────────────────────────────────────
    hol_cols = [f'holiday_{c}' for c in region_codes]
    df['num_regions_holiday'] = df[hol_cols].sum(axis=1)

    # ── 8. Day-of-week one-hot + weekend ───────────────────────────────
    df['day_of_week'] = df['date'].dt.day_name()
    df = pd.get_dummies(df, columns=['day_of_week'], prefix='dow')
    df['is_weekend'] = df['date'].dt.weekday.isin([5, 6]).astype(int)

    # ── 9. Combined holiday flag (weekend OR national holiday) ─────────
    df['is_holiday'] = ((df['is_weekend'] == 1) | (df['is_national_holiday'] == 1)).astype(int)

    return df

# ── Execute ────────────────────────────────────────────────────────────
calendar_df = get_spanish_calendar_features(2026)
print(f"Shape: {calendar_df.shape}")

# ── Quick validation of is_holiday ─────────────────────────────────────
print(f"\nis_holiday breakdown:")
print(f"  Weekends only (not national):  {((calendar_df['is_weekend']==1) & (calendar_df['is_national_holiday']==0)).sum()}")
print(f"  National holidays on weekdays: {((calendar_df['is_weekend']==0) & (calendar_df['is_national_holiday']==1)).sum()}")
print(f"  Both (national on weekend):    {((calendar_df['is_weekend']==1) & (calendar_df['is_national_holiday']==1)).sum()}")
print(f"  Total is_holiday = 1:          {calendar_df['is_holiday'].sum()}")
print(f"  Working days (is_holiday = 0): {(calendar_df['is_holiday']==0).sum()}")

display(calendar_df[['date', 'is_national_holiday', 'national_holiday_name', 'is_weekend', 'is_holiday']].head(10))

# COMMAND ----------

# DBTITLE 1,Save calendar as Delta table
# Convert to Spark DataFrame and save as Delta table
table_name = "datathon.rubber_duckers.spanish_calendar_holidays_2026"  # Change catalog.schema.table as needed

spark_df = spark.createDataFrame(calendar_df)
spark_df.write.mode("overwrite").saveAsTable(table_name)

print(f"✓ Saved {spark_df.count()} rows to {table_name}")
display(spark.table(table_name).limit(5))