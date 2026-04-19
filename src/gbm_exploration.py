# Databricks notebook source
# MAGIC %md
# MAGIC # Energy Consumption Prediction: Exploration & Local Testing
# MAGIC
# MAGIC Use this notebook for **data exploration** and **local model testing**.
# MAGIC Changes here do **not** affect your submission — when you are happy with your
# MAGIC model, copy the `EnergyConsumptionModel` class back to the **submission notebook**.
# MAGIC
# MAGIC ## Workflow
# MAGIC 1. Explore the data below.
# MAGIC 2. Iterate on your model in this notebook using a train/test split (train: before 2025-12-01, test: Dec 2025 – Feb 2026).
# MAGIC 3. When satisfied, copy your `EnergyConsumptionModel` class (and its imports) to the submission notebook.
# MAGIC 4. Run the Submit cell in the submission notebook.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Imports

# COMMAND ----------

import pandas as pd
import numpy as np
from pyspark.sql import functions as F, Window

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Data
# MAGIC
# MAGIC Row-level security ensures you only see data **up to end of November 2025** interactively.
# MAGIC
# MAGIC **Tip:** The dataset is large. Keep data as PySpark DataFrames. If you want to
# MAGIC explore in pandas, use `df.limit(N).toPandas()` on a small sample.

# COMMAND ----------

df = spark.table("datathon.shared.client_consumption")
display(df)

# COMMAND ----------

print(f"Rows:    {df.count()}")
print(f"Clients: {df.select('client_id').distinct().count()}")
df.selectExpr("min(datetime_local) as min_dt", "max(datetime_local) as max_dt").show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Exploratory Data Analysis
# MAGIC
# MAGIC Feel free to add as many cells as you need here.
# MAGIC Use `df.limit(N).toPandas()` if you want to use pandas for exploration.

# COMMAND ----------

# Example: quick look at a small sample in pandas
pdf_sample = df.limit(10000).toPandas()
pdf_sample["datetime_local"] = pd.to_datetime(pdf_sample["datetime_local"])
pdf_sample.describe()

# COMMAND ----------

# Total consumption per 15 minutes per province
cons_15min = (
    df.withColumn("datetime_15min", F.window("datetime_local", "15 minutes").start)
      .groupBy("community_code", "datetime_15min")
      .agg(F.sum("active_kw").alias("total_active_kw"))
)
cons_15min.createOrReplaceTempView("consumption_15min_per_province")

# Total consumption per hour per province
cons_hour = (
    df.withColumn("datetime_hour", F.window("datetime_local", "1 hour").start)
      .groupBy("community_code", "datetime_hour")
      .agg(F.sum("active_kw").alias("total_active_kw"))
)
cons_hour.createOrReplaceTempView("consumption_hour_per_province")

# Total consumption per day per province
cons_day = (
    df.withColumn("date", F.to_date("datetime_local"))
      .groupBy("community_code", "date")
      .agg(F.sum("active_kw").alias("total_active_kw"))
)
cons_day.createOrReplaceTempView("consumption_day_per_province")

# Total consumption per week per province
cons_week = (
    df.withColumn("week_start", F.date_trunc("week", "datetime_local"))
      .groupBy("community_code", "week_start")
      .agg(F.sum("active_kw").alias("total_active_kw"))
)
cons_week.createOrReplaceTempView("consumption_week_per_province")

# Total consumption per month per province
cons_month = (
    df.withColumn("month_start", F.date_trunc("month", "datetime_local"))
      .groupBy("community_code", "month_start")
      .agg(F.sum("active_kw").alias("total_active_kw"))
)
cons_month.createOrReplaceTempView("consumption_month_per_province")

# COMMAND ----------

pdf_month = cons_month.orderBy("month_start").toPandas()
pdf_month["month_start"] = pd.to_datetime(pdf_month["month_start"])
pdf_month.pivot(index="month_start", columns="community_code", values="total_active_kw").plot(figsize=(12,6))

# COMMAND ----------

# Assign day names to each date and get sum of consumption per day per community
cons_day_with_dayname = (
    df.withColumn("date", F.to_date("datetime_local"))
      .withColumn(
          "day_name",
          F.date_format("date", "EEEE")
      )
      .groupBy("community_code", "date", "day_name")
      .agg(F.sum("active_kw").alias("total_active_kw"))
      .orderBy("community_code", "date")
)
display(cons_day_with_dayname)

# COMMAND ----------

# Sum total consumption for each weekday per community
weekday_sum = (
    cons_day_with_dayname
    .groupBy("community_code", "day_name")
    .agg(F.sum("total_active_kw").alias("sum_active_kw"))
    .orderBy("community_code", "day_name")
)

pdf_weekday_sum = weekday_sum.toPandas()
pivot = pdf_weekday_sum.pivot(index="community_code", columns="day_name", values="sum_active_kw")
pivot = pivot[["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]]
pivot.plot(kind="bar", figsize=(12,6), ylabel="Total Active kW", title="Total Consumption per Weekday per Community")

# COMMAND ----------

# Join all aggregation levels on community_code and timestamp/date keys
from pyspark.sql.functions import col, to_date, date_trunc, date_format, when, dense_rank, hour, minute, floor

# Prepare base 15-min intervals
base = cons_15min.withColumn("date", to_date("datetime_15min")) \
    .withColumn("hour", date_trunc("hour", "datetime_15min")) \
    .withColumn("week_start", date_trunc("week", "datetime_15min")) \
    .withColumn("month_start", date_trunc("month", "datetime_15min")) \
    .withColumn("day_name", date_format("date", "EEEE")) \
    .withColumn("hour_of_day", hour("datetime_15min")) \
    .withColumn("quarter_of_hour", floor(minute("datetime_15min") / 15))

# Label encode weekday: Monday=0, ..., Sunday=6
base = base.withColumn(
    "day_of_week",
    when(col("day_name") == "Monday", 0)
    .when(col("day_name") == "Tuesday", 1)
    .when(col("day_name") == "Wednesday", 2)
    .when(col("day_name") == "Thursday", 3)
    .when(col("day_name") == "Friday", 4)
    .when(col("day_name") == "Saturday", 5)
    .when(col("day_name") == "Sunday", 6)
    .otherwise(None)
)

# Label encode community_code
w_comm = Window.orderBy("community_code")
base = base.withColumn("community_code_label", dense_rank().over(w_comm) - 1)

# Join daily
feat = base.join(
    cons_day.withColumnRenamed("total_active_kw", "day_total_active_kw"),
    on=["community_code", "date"],
    how="left"
)

# Join hourly
feat = feat.join(
    cons_hour.withColumnRenamed("total_active_kw", "hour_total_active_kw")
        .withColumnRenamed("datetime_hour", "hour"),
    on=["community_code", "hour"],
    how="left"
)

# Join weekly
feat = feat.join(
    cons_week.withColumnRenamed("total_active_kw", "week_total_active_kw"),
    on=["community_code", "week_start"],
    how="left"
)

# Join monthly
feat = feat.join(
    cons_month.withColumnRenamed("total_active_kw", "month_total_active_kw"),
    on=["community_code", "month_start"],
    how="left"
)

# Final feature set: drop extra keys, keep all features and target
feature_cols = [
    "community_code", "community_code_label", "datetime_15min", "total_active_kw", "day_total_active_kw",
    "hour_total_active_kw", "week_total_active_kw", "month_total_active_kw", "day_of_week",
    "hour_of_day", "quarter_of_hour"
]

dataset = feat.select(*feature_cols)
display(dataset)

# COMMAND ----------

# MAGIC %pip install statsmodels
# MAGIC import statsmodels.api as sm
# MAGIC from sklearn.metrics import mean_absolute_error, r2_score
# MAGIC from sklearn.model_selection import train_test_split
# MAGIC
# MAGIC # Randomly sample 50,000 entries
# MAGIC pdf = dataset.dropna().orderBy(F.rand()).limit(50000).toPandas()
# MAGIC
# MAGIC # Prepare features and target: predict next day's 15min consumption
# MAGIC pdf = pdf.sort_values(["community_code", "datetime_15min"])
# MAGIC pdf["target"] = pdf.groupby("community_code")["total_active_kw"].shift(-96)  # 96 intervals = 1 day
# MAGIC pdf = pdf.dropna(subset=["target"])
# MAGIC
# MAGIC X = pdf[["community_code_label", "day_of_week", "hour_of_day", "quarter_of_hour" ]]
# MAGIC y = pdf["target"]
# MAGIC
# MAGIC # Train/test split (80/20)
# MAGIC X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# MAGIC
# MAGIC # Add constant for intercept
# MAGIC X_train_const = sm.add_constant(X_train)
# MAGIC X_test_const = sm.add_constant(X_test)
# MAGIC
# MAGIC # Fit OLS regression
# MAGIC ols_model = sm.OLS(y_train, X_train_const).fit()
# MAGIC
# MAGIC # Predict
# MAGIC y_train_pred = ols_model.predict(X_train_const)
# MAGIC y_test_pred = ols_model.predict(X_test_const)
# MAGIC
# MAGIC # Metrics
# MAGIC train_mae = mean_absolute_error(y_train, y_train_pred)
# MAGIC test_mae = mean_absolute_error(y_test, y_test_pred)
# MAGIC train_r2 = r2_score(y_train, y_train_pred)
# MAGIC test_r2 = r2_score(y_test, y_test_pred)
# MAGIC
# MAGIC print(ols_model.summary())
# MAGIC print(f"Train MAE: {train_mae:.4f}, R2: {train_r2:.4f}")
# MAGIC print(f"Test MAE: {test_mae:.4f}, R2: {test_r2:.4f}")

# COMMAND ----------

# MAGIC %pip install lightgbm
# MAGIC import lightgbm as lgb
# MAGIC from sklearn.model_selection import train_test_split
# MAGIC
# MAGIC
# MAGIC # Prepare features and target: predict next day's 15min consumption
# MAGIC pdf = dataset.dropna().orderBy(F.rand()).limit(50000).toPandas()
# MAGIC pdf = pdf.sort_values(["community_code", "datetime_15min"])
# MAGIC pdf["target"] = pdf.groupby("community_code")["total_active_kw"].shift(-96)
# MAGIC pdf = pdf.dropna(subset=["target"])
# MAGIC
# MAGIC X = pdf[["community_code_label", "quarter_of_hour"]]
# MAGIC y = pdf["target"]
# MAGIC
# MAGIC # Train/test split (80/20)
# MAGIC X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
# MAGIC
# MAGIC # LightGBM dataset
# MAGIC lgb_train = lgb.Dataset(X_train, y_train)
# MAGIC lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
# MAGIC
# MAGIC params = {
# MAGIC     "objective": "regression",
# MAGIC     "metric": "mae",
# MAGIC     "boosting_type": "gbdt",
# MAGIC     "learning_rate": 0.1,
# MAGIC     "num_leaves": 31,
# MAGIC     "verbose": -1,
# MAGIC     "seed": 20
# MAGIC }
# MAGIC
# MAGIC gbm = lgb.train(
# MAGIC     params,
# MAGIC     lgb_train,
# MAGIC     num_boost_round=100,
# MAGIC     valid_sets=[lgb_train, lgb_eval]
# MAGIC )
# MAGIC
# MAGIC y_train_pred = gbm.predict(X_train)
# MAGIC y_test_pred = gbm.predict(X_test)
# MAGIC
# MAGIC # Convert predictions and actuals to Spark DataFrames for MAE calculation
# MAGIC train_df = spark.createDataFrame(
# MAGIC     pd.DataFrame({"prediction": y_train_pred, "active_kw": y_train.values})
# MAGIC )
# MAGIC test_df = spark.createDataFrame(
# MAGIC     pd.DataFrame({"prediction": y_test_pred, "active_kw": y_test.values})
# MAGIC )
# MAGIC
# MAGIC train_mae = train_df.select(
# MAGIC     F.mean(F.abs(F.col("active_kw") - F.col("prediction")))
# MAGIC ).collect()[0][0]
# MAGIC
# MAGIC test_mae = test_df.select(
# MAGIC     F.mean(F.abs(F.col("active_kw") - F.col("prediction")))
# MAGIC ).collect()[0][0]
# MAGIC
# MAGIC train_r2 = r2_score(y_train, y_train_pred)
# MAGIC test_r2 = r2_score(y_test, y_test_pred)
# MAGIC
# MAGIC print(f"Train MAE: {train_mae:.4f}, R2: {train_r2:.4f}")
# MAGIC print(f"Test MAE: {test_mae:.4f}, R2: {test_r2:.4f}")

# COMMAND ----------

# DBTITLE 1,Improved model: lag features + external data + temporal split
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Aggregate to TOTAL 15-min consumption (matching submission target)
agg = (
    df.withColumn("datetime_15min", F.window("datetime_local", "15 minutes").start)
      .groupBy("datetime_15min")
      .agg(F.sum("active_kw").alias("total_kw"))
      .orderBy("datetime_15min")
)

# 2. Add lag & rolling features (the key missing signal)
w = Window.orderBy("datetime_15min")
intervals_per_day = 96      # 96 x 15min = 24h
intervals_per_week = 672    # 7 x 96

agg = agg.withColumn("lag_1d", F.lag("total_kw", intervals_per_day).over(w)) \
         .withColumn("lag_7d", F.lag("total_kw", intervals_per_week).over(w)) \
         .withColumn("lag_2d", F.lag("total_kw", 2 * intervals_per_day).over(w)) \
         .withColumn("lag_3d", F.lag("total_kw", 3 * intervals_per_day).over(w))

# Rolling means (7-day window)
w7d = Window.orderBy("datetime_15min").rowsBetween(-intervals_per_week, -1)
agg = agg.withColumn("rolling_mean_7d", F.avg("total_kw").over(w7d)) \
         .withColumn("rolling_std_7d", F.stddev("total_kw").over(w7d))

# 3. Time features
agg = agg.withColumn("hour", F.hour("datetime_15min")) \
         .withColumn("minute", F.minute("datetime_15min")) \
         .withColumn("day_of_week", F.dayofweek("datetime_15min")) \
         .withColumn("day_of_month", F.dayofmonth("datetime_15min")) \
         .withColumn("month", F.month("datetime_15min")) \
         .withColumn("is_weekend", F.when(F.dayofweek("datetime_15min").isin(1, 7), 1).otherwise(0))

# 4. Join external forecast data
demand_fc = spark.table("datathon.shared.demand_forecast") \
    .withColumnRenamed("value", "demand_forecast") \
    .withColumnRenamed("datetime_local", "datetime_15min") \
    .select("datetime_15min", "demand_forecast")

pv_fc = spark.table("datathon.shared.pv_production_forecast") \
    .withColumnRenamed("value", "pv_forecast") \
    .withColumnRenamed("datetime_local", "datetime_15min") \
    .select("datetime_15min", "pv_forecast")

wind_fc = spark.table("datathon.shared.wind_production_forecast") \
    .withColumnRenamed("value", "wind_forecast") \
    .withColumnRenamed("datetime_local", "datetime_15min") \
    .select("datetime_15min", "wind_forecast")

agg = agg.join(demand_fc, on="datetime_15min", how="left") \
         .join(pv_fc, on="datetime_15min", how="left") \
         .join(wind_fc, on="datetime_15min", how="left")

# 5. Temporal train/test split (NOT random!)
train_end = "2025-11-01"
agg_clean = agg.dropna()

train_pdf = agg_clean.filter(F.col("datetime_15min") < train_end).toPandas()
test_pdf = agg_clean.filter(F.col("datetime_15min") >= train_end).toPandas()

feature_cols = [
    "lag_1d", "lag_7d", "lag_2d", "lag_3d",
    "rolling_mean_7d", "rolling_std_7d",
    "hour", "minute", "day_of_week", "day_of_month", "month", "is_weekend",
    "demand_forecast", "pv_forecast", "wind_forecast"
]

X_train, y_train = train_pdf[feature_cols], train_pdf["total_kw"]
X_test, y_test = test_pdf[feature_cols], test_pdf["total_kw"]

print(f"Train size: {len(X_train):,} | Test size: {len(X_test):,}")
print(f"Train period: {train_pdf['datetime_15min'].min()} to {train_pdf['datetime_15min'].max()}")
print(f"Test period:  {test_pdf['datetime_15min'].min()} to {test_pdf['datetime_15min'].max()}")

# 6. Train LightGBM
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

params = {
    "objective": "regression",
    "metric": "mae",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 50,
    "verbose": -1,
    "seed": 42
}

gbm_v2 = lgb.train(
    params, lgb_train,
    num_boost_round=500,
    valid_sets=[lgb_train, lgb_eval],
)

y_train_pred = gbm_v2.predict(X_train)
y_test_pred = gbm_v2.predict(X_test)

print(f"\n--- Results ---")
print(f"Train MAE: {mean_absolute_error(y_train, y_train_pred):.2f}, R2: {r2_score(y_train, y_train_pred):.4f}")
print(f"Test  MAE: {mean_absolute_error(y_test, y_test_pred):.2f}, R2: {r2_score(y_test, y_test_pred):.4f}")

# Feature importance
importance = pd.DataFrame({
    "feature": feature_cols,
    "importance": gbm_v2.feature_importance("gain")
}).sort_values("importance", ascending=False)
print(f"\n--- Feature Importance (gain) ---")
print(importance.to_string(index=False))

# COMMAND ----------

# DBTITLE 1,Install torch
# MAGIC %pip install --force-reinstall typing_extensions>=4.10 torch

# COMMAND ----------

# # Fix for the typing_extensions error and ensure torch/sklearn are up to date
%pip install --upgrade typing_extensions>=4.5.0

# Restart Python to use the updated packages
dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,CVAE day-ahead forecast — no lag features
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from pyspark.sql import functions as F

# ═══════════════════════════════════════════════════════════════
# 1. DATA PREP — daily profiles (96 × 15-min) + conditions
# ═══════════════════════════════════════════════════════════════
agg = (
    df.withColumn("datetime_15min", F.window("datetime_local", "15 minutes").start)
      .groupBy("datetime_15min")
      .agg(F.sum("active_kw").alias("total_kw"))
      .orderBy("datetime_15min")
)

# Join external forecasts (available day-ahead — no leakage)
for tbl, alias in [("demand_forecast", "demand_fc"),
                   ("pv_production_forecast", "pv_fc"),
                   ("wind_production_forecast", "wind_fc")]:
    fc = (spark.table(f"datathon.shared.{tbl}")
          .withColumnRenamed("value", alias)
          .withColumnRenamed("datetime_local", "datetime_15min")
          .select("datetime_15min", alias))
    agg = agg.join(fc, "datetime_15min", "left")

# Temporal + interval index
agg = (agg
    .withColumn("date", F.to_date("datetime_15min"))
    .withColumn("interval_idx",
        (F.hour("datetime_15min") * 4 +
         F.floor(F.minute("datetime_15min") / 15)).cast("int"))
    .withColumn("day_of_week", F.dayofweek("date"))
    .withColumn("month", F.month("date"))
    .withColumn("is_weekend",
        F.when(F.dayofweek("date").isin(1, 7), 1).otherwise(0))
)

# Pivot to daily profiles — each row = 1 day, 96 cols
daily_profiles = (agg.groupBy("date")
    .pivot("interval_idx", list(range(96)))
    .agg(F.first("total_kw")))

# Daily conditioning features
daily_cond = agg.groupBy("date").agg(
    F.first("day_of_week").alias("day_of_week"),
    F.first("month").alias("month"),
    F.first("is_weekend").alias("is_weekend"),
    F.avg("demand_fc").alias("demand_fc_mean"),
    F.max("demand_fc").alias("demand_fc_max"),
    F.min("demand_fc").alias("demand_fc_min"),
    F.avg("pv_fc").alias("pv_fc_mean"),
    F.max("pv_fc").alias("pv_fc_max"),
    F.avg("wind_fc").alias("wind_fc_mean"),
    F.max("wind_fc").alias("wind_fc_max"),
)

daily = daily_profiles.join(daily_cond, "date", "inner").orderBy("date")
pdf_vae = daily.toPandas()

profile_cols = [str(i) for i in range(96)]
pdf_vae = pdf_vae.dropna(subset=profile_cols)

# One-hot encode calendar (fixed categories)
for d in range(1, 8):
    pdf_vae[f"dow_{d}"] = (pdf_vae["day_of_week"] == d).astype(np.float32)
for m in range(1, 13):
    pdf_vae[f"month_{m}"] = (pdf_vae["month"] == m).astype(np.float32)

cond_cols = (
    [f"dow_{d}" for d in range(1, 8)] +
    [f"month_{m}" for m in range(1, 13)] +
    ["is_weekend",
     "demand_fc_mean", "demand_fc_max", "demand_fc_min",
     "pv_fc_mean", "pv_fc_max",
     "wind_fc_mean", "wind_fc_max"]
)
pdf_vae[cond_cols] = pdf_vae[cond_cols].fillna(0)

profiles = pdf_vae[profile_cols].values.astype(np.float32)
conditions = pdf_vae[cond_cols].values.astype(np.float32)
dates_arr = pd.to_datetime(pdf_vae["date"]).values

# Temporal split
train_end = np.datetime64("2025-11-01")
tr = dates_arr < train_end
te = dates_arr >= train_end

profile_scaler = StandardScaler().fit(profiles[tr])
cond_scaler    = StandardScaler().fit(conditions[tr])

P_tr = profile_scaler.transform(profiles[tr]).astype(np.float32)
P_te = profile_scaler.transform(profiles[te]).astype(np.float32)
C_tr = cond_scaler.transform(conditions[tr]).astype(np.float32)
C_te = cond_scaler.transform(conditions[te]).astype(np.float32)

print(f"Train days: {tr.sum()}  |  Test days: {te.sum()}")
print(f"Profile dim: {P_tr.shape[1]}  |  Condition dim: {C_tr.shape[1]}")

# ═══════════════════════════════════════════════════════════════
# 2. CONDITIONAL VAE
# ═══════════════════════════════════════════════════════════════
class CVAE(nn.Module):
    def __init__(self, prof_d, cond_d, lat_d=24, hid_d=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(prof_d + cond_d, hid_d),
            nn.BatchNorm1d(hid_d), nn.LeakyReLU(0.2), nn.Dropout(0.1),
            nn.Linear(hid_d, hid_d // 2),
            nn.BatchNorm1d(hid_d // 2), nn.LeakyReLU(0.2),
        )
        self.fc_mu     = nn.Linear(hid_d // 2, lat_d)
        self.fc_logvar = nn.Linear(hid_d // 2, lat_d)
        self.decoder = nn.Sequential(
            nn.Linear(lat_d + cond_d, hid_d // 2),
            nn.BatchNorm1d(hid_d // 2), nn.LeakyReLU(0.2), nn.Dropout(0.1),
            nn.Linear(hid_d // 2, hid_d),
            nn.BatchNorm1d(hid_d), nn.LeakyReLU(0.2),
            nn.Linear(hid_d, prof_d),
        )
        self.lat_d = lat_d

    def encode(self, x, c):
        h = self.encoder(torch.cat([x, c], 1))
        return self.fc_mu(h), self.fc_logvar(h)

    def decode(self, z, c):
        return self.decoder(torch.cat([z, c], 1))

    def forward(self, x, c):
        mu, lv = self.encode(x, c)
        z = mu + torch.randn_like(lv) * torch.exp(0.5 * lv)
        return self.decode(z, c), mu, lv


def loss_fn(recon, target, mu, lv, beta=0.5):
    mse = nn.functional.mse_loss(recon, target, reduction="mean")
    kl  = -0.5 * torch.mean(1 + lv - mu.pow(2) - lv.exp())
    return mse + beta * kl, mse.item(), kl.item()

# ═══════════════════════════════════════════════════════════════
# 3. TRAINING
# ═══════════════════════════════════════════════════════════════
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

cvae = CVAE(P_tr.shape[1], C_tr.shape[1], lat_d=24, hid_d=256).to(device)
opt  = optim.Adam(cvae.parameters(), lr=1e-3, weight_decay=1e-5)
sch  = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=300)

loader = DataLoader(
    TensorDataset(torch.tensor(P_tr), torch.tensor(C_tr)),
    batch_size=32, shuffle=True, drop_last=True)

best_loss, patience, EPOCHS = float("inf"), 0, 300
for ep in range(EPOCHS):
    cvae.train()
    ep_loss = 0.0
    for bp, bc in loader:
        bp, bc = bp.to(device), bc.to(device)
        opt.zero_grad()
        recon, mu, lv = cvae(bp, bc)
        loss, rl, kl = loss_fn(recon, bp, mu, lv, beta=0.5)
        loss.backward()
        nn.utils.clip_grad_norm_(cvae.parameters(), 1.0)
        opt.step()
        ep_loss += loss.item() * len(bp)
    sch.step()
    avg = ep_loss / len(loader.dataset)
    if (ep + 1) % 50 == 0:
        print(f"Epoch {ep+1:3d}: loss={avg:.4f}  recon={rl:.4f}  kl={kl:.4f}")
    if avg < best_loss:
        best_loss, patience = avg, 0
        best_state = {k: v.clone() for k, v in cvae.state_dict().items()}
    else:
        patience += 1
        if patience >= 50:
            print(f"Early stop @ epoch {ep+1}")
            break

cvae.load_state_dict(best_state)

# ═══════════════════════════════════════════════════════════════
# 4. DAY-AHEAD PREDICTION (conditions only — no lags)
# ═══════════════════════════════════════════════════════════════
cvae.eval()
C_te_t = torch.tensor(C_te).to(device)

N_SAMPLES = 100
with torch.no_grad():
    preds = torch.zeros(len(C_te), P_tr.shape[1]).to(device)
    for _ in range(N_SAMPLES):
        z = torch.randn(len(C_te), cvae.lat_d).to(device)
        preds += cvae.decode(z, C_te_t)
    preds = (preds / N_SAMPLES).cpu().numpy()

pred_profiles = profile_scaler.inverse_transform(preds)
actual_profiles = profiles[te]

mae = mean_absolute_error(actual_profiles.ravel(), pred_profiles.ravel())
r2  = r2_score(actual_profiles.ravel(), pred_profiles.ravel())

print(f"\n{'='*55}")
print(f"  CVAE Day-Ahead Forecast  (Test: Nov 2025)")
print(f"  No lag features — calendar + external forecasts only")
print(f"{'='*55}")
print(f"  MAE : {mae:,.2f} kW")
print(f"  R²  : {r2:.4f}")
print(f"  Test intervals: {actual_profiles.size:,}")
print(f"{'='*55}")

# ── Visualise sample test days ───────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 8), sharex=True)
for i, ax in enumerate(axes.flat):
    if i < len(actual_profiles):
        ax.plot(range(96), actual_profiles[i], label="Actual", alpha=.8)
        ax.plot(range(96), pred_profiles[i], "--", label="CVAE", alpha=.8)
        day = pd.Timestamp(dates_arr[te][i])
        ax.set_title(day.strftime("%Y-%m-%d (%A)"))
        ax.legend(fontsize=8)
        ax.set_ylabel("Total kW")
axes[-1, 1].set_xlabel("15-min interval of day")
plt.suptitle("CVAE: Actual vs Predicted Daily Consumption Profiles", fontsize=14)
plt.tight_layout()
plt.show()

# COMMAND ----------

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

from pyspark.sql import functions as F

# COMMAND ----------

from pyspark.sql import Window as W

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print("Loading and preparing data with PySpark...")

consumption = spark.table("datathon.shared.client_consumption")
demand_fc   = spark.table("datathon.shared.demand_forecast").withColumnRenamed("value", "demand_fc")
pv_fc       = spark.table("datathon.shared.pv_production_forecast").withColumnRenamed("value", "pv_fc")
wind_fc     = spark.table("datathon.shared.wind_production_forecast").withColumnRenamed("value", "wind_fc")

# ── 1. Pre-sample 30 % of clients → cuts volume before the expensive pivot ──
sampled_clients = consumption.select("client_id").distinct().sample(False, 0.3, seed=42)
consumption = consumption.join(F.broadcast(sampled_clients), "client_id")

agg = (
    consumption
    .withColumn("dt15", F.window("datetime_local", "15 minutes").start)
    .groupBy("client_id", "community_code", "dt15")
    .agg(F.sum("active_kw").alias("total_kw"))
)

forecasts = (
    demand_fc.join(pv_fc, "datetime_local", "outer")
             .join(wind_fc, "datetime_local", "outer")
             .withColumnRenamed("datetime_local", "dt15"))
agg = agg.join(forecasts, "dt15", "left")

agg = (agg
    .withColumn("date", F.to_date("dt15"))
    .withColumn("interval_idx",
        (F.hour("dt15") * 4 + F.floor(F.minute("dt15") / 15)).cast("int"))
    .withColumn("day_of_week", F.dayofweek("date"))
    .withColumn("month", F.month("date"))
)

# ── 2. Pivot daily profiles per client (single pivot, no lags) ──
daily_profiles = (
    agg.groupBy("client_id", "community_code", "date")
    .pivot("interval_idx", list(range(96)))
    .agg(F.first("total_kw"))
).fillna(0)

daily_cond = agg.groupBy("client_id", "date").agg(
    F.first("day_of_week").alias("day_of_week"),
    F.first("month").alias("month"),
    F.avg("demand_fc").alias("demand_fc_mean"),
    F.avg("pv_fc").alias("pv_fc_mean"),
    F.avg("wind_fc").alias("wind_fc_mean"),
)

final_df = daily_profiles.join(daily_cond, ["client_id", "date"], "inner")

# ── 3. Cheap random sample (no full-sort orderBy!) ──
final_df = final_df.sample(False, 0.5, seed=42)
final_df = final_df.fillna(0, subset=["demand_fc_mean", "pv_fc_mean", "wind_fc_mean"])

# ── 4. Label encoding via dense_rank (pure SQL, no ML model) ──
final_df = (
    final_df
    .withColumn("client_idx",
        (F.dense_rank().over(W.orderBy("client_id")) - 1).cast("int"))
    .withColumn("community_idx",
        (F.dense_rank().over(W.orderBy("community_code")) - 1).cast("int"))
    .withColumn("is_train", (F.col("date") < "2025-11-01").cast("int"))
)

NUM_CLIENTS = final_df.agg(F.max("client_idx")).collect()[0][0] + 1
NUM_COMMUNITIES = final_df.agg(F.max("community_idx")).collect()[0][0] + 1

# ── 5. Collect compact dataset (no toPandas!) ──
profile_cols = [str(i) for i in range(96)]
cond_cols = ["demand_fc_mean", "pv_fc_mean", "wind_fc_mean"]
select_cols = (
    ["client_id"] + profile_cols + cond_cols +
    ["client_idx", "community_idx", "day_of_week", "month", "is_train"]
)

print("Collecting compact dataset...")
rows = final_df.select(*select_cols).collect()
print(f"Collected {len(rows):,} rows")

# ── 6. Convert to numpy arrays ──
n_p, n_c = len(profile_cols), len(cond_cols)

client_ids_arr = np.array([int(r[0]) for r in rows])
profiles_raw = np.array(
    [[float(r[1 + j]) for j in range(n_p)] for r in rows], dtype=np.float32)
conds_raw = np.array(
    [[float(r[1 + n_p + j]) for j in range(n_c)] for r in rows], dtype=np.float32)
cats = np.array(
    [[int(r[1 + n_p + n_c + j]) for j in range(4)] for r in rows], dtype=np.int64)
train_mask = np.array([bool(r[-1]) for r in rows])
test_mask = ~train_mask

# ── 7. Per-client profile scaling (fast on small numpy) ──
client_scalers = {}
P_scaled = np.zeros_like(profiles_raw)

for cid in np.unique(client_ids_arr):
    mask = client_ids_arr == cid
    tr = mask & train_mask
    if tr.sum() == 0:
        continue
    scaler = StandardScaler()
    scaler.fit(profiles_raw[tr])
    client_scalers[cid] = scaler
    P_scaled[mask] = scaler.transform(profiles_raw[mask])

cond_scaler = StandardScaler()
cond_scaler.fit(conds_raw[train_mask])
C_cont = cond_scaler.transform(conds_raw).astype(np.float32)

# Compatibility shim for Cell 23
pdf = pd.DataFrame({"client_id": client_ids_arr})

# ── 8. DataLoaders ──
def create_dataset(mask):
    return TensorDataset(
        torch.tensor(P_scaled[mask]),
        torch.tensor(C_cont[mask]),
        torch.tensor(cats[mask])
    )

train_loader = DataLoader(create_dataset(train_mask), batch_size=128, shuffle=True, drop_last=True)
test_loader = DataLoader(create_dataset(test_mask), batch_size=128, shuffle=False)

print(f"Train: {train_mask.sum():,} | Test: {test_mask.sum():,}")
print(f"Clients: {NUM_CLIENTS} | Communities: {NUM_COMMUNITIES}")

# COMMAND ----------

class GlobalCVAE(nn.Module):
    def __init__(self, num_clients, num_communities, lat_d=32):
        super().__init__()
        self.client_emb = nn.Embedding(num_clients, 16)
        self.comm_emb = nn.Embedding(num_communities, 4)
        self.dow_emb = nn.Embedding(8, 3)
        self.month_emb = nn.Embedding(13, 3)
        # No lags: cond = continuous(3) + embeddings(16+4+3+3)
        self.cond_dim = 3 + 26
        self.encoder = nn.Sequential(
            nn.Linear(96 + self.cond_dim, 256),
            nn.BatchNorm1d(256), nn.LeakyReLU(0.2), nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128), nn.LeakyReLU(0.2),
        )
        self.fc_mu = nn.Linear(128, lat_d)
        self.fc_logvar = nn.Linear(128, lat_d)
        self.decoder = nn.Sequential(
            nn.Linear(lat_d + self.cond_dim, 128),
            nn.BatchNorm1d(128), nn.LeakyReLU(0.2), nn.Dropout(0.1),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256), nn.LeakyReLU(0.2),
            nn.Linear(256, 96)
        )
        self.lat_d = lat_d

    def process_conditions(self, cont, cat):
        c_emb = self.client_emb(cat[:, 0])
        comm_emb = self.comm_emb(cat[:, 1])
        dow_emb = self.dow_emb(cat[:, 2])
        m_emb = self.month_emb(cat[:, 3])
        return torch.cat([cont, c_emb, comm_emb, dow_emb, m_emb], dim=1)

    def forward(self, target, cont, cat):
        cond = self.process_conditions(cont, cat)
        h = self.encoder(torch.cat([target, cond], dim=1))
        mu, lv = self.fc_mu(h), self.fc_logvar(h)
        z = mu + torch.randn_like(lv) * torch.exp(0.5 * lv)
        recon = self.decoder(torch.cat([z, cond], dim=1))
        return recon, mu, lv

    def sample(self, cont, cat):
        cond = self.process_conditions(cont, cat)
        z = torch.randn(cont.size(0), self.lat_d).to(cont.device)
        return self.decoder(torch.cat([z, cond], dim=1))


def loss_fn(recon, target, mu, lv, beta=0.1):
    mse = nn.functional.mse_loss(recon, target, reduction="mean")
    kl = -0.5 * torch.mean(1 + lv - mu.pow(2) - lv.exp())
    return mse + beta * kl, mse.item(), kl.item()

# COMMAND ----------

# Recover device after potential CUDA assert corruption
try:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        _ = torch.zeros(1, device="cuda")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
except RuntimeError:
    device = torch.device("cpu")
    print("CUDA unavailable after previous error, using CPU")

cvae = GlobalCVAE(NUM_CLIENTS, NUM_COMMUNITIES).to(device)
opt = optim.Adam(cvae.parameters(), lr=2e-3, weight_decay=1e-5)  # Increased LR for faster convergence
scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=30)  # Shorter scheduler

EPOCHS = 20  # Fewer epochs for speed
print(f"Starting Training for {EPOCHS} Epochs on {device}...")

for ep in range(EPOCHS):
    cvae.train()
    ep_loss = 0
    for target, cont, cat in train_loader:
        target, cont, cat = target.to(device, non_blocking=True), cont.to(device, non_blocking=True), cat.to(device, non_blocking=True)
        # Clamp indices to valid embedding ranges
        cat[:, 0].clamp_(0, NUM_CLIENTS - 1)
        cat[:, 1].clamp_(0, NUM_COMMUNITIES - 1)
        cat[:, 2].clamp_(0, 7)
        cat[:, 3].clamp_(0, 12)
        opt.zero_grad(set_to_none=True)
        recon, mu, lv = cvae(target, cont, cat)
        loss, mse, kl = loss_fn(recon, target, mu, lv, beta=0.05)
        loss.backward()
        nn.utils.clip_grad_norm_(cvae.parameters(), 1.0)
        opt.step()
        ep_loss += loss.item()
    scheduler.step()
    if (ep + 1) % 5 == 0 or ep == 0:
        avg_loss = ep_loss / len(train_loader)
        print(f"Epoch {ep+1:3d}/{EPOCHS} | Avg Total Loss: {avg_loss:.4f}")

print("\nGenerating Day-Ahead Forecasts for the Test Set...")
cvae.eval()
predictions = []
actuals = []
client_ids_test = pdf.loc[test_mask, 'client_id'].values

with torch.no_grad():
    for target, cont, cat in test_loader:
        target, cont, cat = target.to(device, non_blocking=True), cont.to(device, non_blocking=True), cat.to(device, non_blocking=True)
        # Clamp indices to valid embedding ranges
        cat[:, 0].clamp_(0, NUM_CLIENTS - 1)
        cat[:, 1].clamp_(0, NUM_COMMUNITIES - 1)
        cat[:, 2].clamp_(0, 7)
        cat[:, 3].clamp_(0, 12)
        N_SAMPLES = 3  # Fewer samples for speed
        batch_preds = torch.zeros_like(target)
        for _ in range(N_SAMPLES):
            batch_preds += cvae.sample(cont, cat)
        batch_preds /= N_SAMPLES
        predictions.append(batch_preds.cpu().numpy())
        actuals.append(target.cpu().numpy())

predictions = np.vstack(predictions)
actuals = np.vstack(actuals)

pred_kw = np.zeros_like(predictions)
actual_kw = np.zeros_like(actuals)

for i, c_id in enumerate(client_ids_test):
    scaler = client_scalers.get(c_id)
    if scaler is not None:
        pred_kw[i] = scaler.inverse_transform(predictions[i].reshape(1, -1))
        actual_kw[i] = scaler.inverse_transform(actuals[i].reshape(1, -1))
    else:
        pred_kw[i] = predictions[i]
        actual_kw[i] = actuals[i]

mae = mean_absolute_error(actual_kw.ravel(), pred_kw.ravel())
r2 = r2_score(actual_kw.ravel(), pred_kw.ravel())

# COMMAND ----------

print(f"\n{'='*55}")
print(f" GLOBAL CVAE DAY-AHEAD FORECAST RESULTS")
print(f"{'='*55}")
print(f" Global MAE : {mae:,.2f} kW")
print(f" Global R\u00b2  : {r2:.4f}")
print(f" Test Days  : {len(pred_kw):,}")
print(f"{'='*55}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define Your Model
# MAGIC
# MAGIC Iterate on the model here, then copy the final version to the **submission notebook**.
# MAGIC
# MAGIC The baseline below predicts each 15-minute interval using the value from
# MAGIC **7 days ago**, with the historical mean as a fallback.
# MAGIC
# MAGIC ### Important constraints
# MAGIC - During scoring, your `predict()` receives the **full dataset** (up to end of Feb 2026) as a **PySpark DataFrame**.
# MAGIC - `predict_start` and `predict_end` specify the period to generate predictions for.
# MAGIC - Your `predict()` must return a **PySpark DataFrame** with columns `datetime_15min` (timestamp) and `prediction` (double).
# MAGIC - `prediction` must be the **total** (sum) of `active_kw` across all clients for that interval.
# MAGIC - **Avoid `.toPandas()` on the full dataset.** Keep everything in PySpark.
# MAGIC - If using ML libraries (LightGBM, sklearn, etc.), do feature engineering in PySpark first, then
# MAGIC   `.toPandas()` only the **compact feature matrix**. Convert predictions back with `spark.createDataFrame()`.

# COMMAND ----------

# DBTITLE 1,OLS Feature Selection: all available predictors
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from pyspark.sql import functions as F

print("Building comprehensive feature matrix for OLS feature selection...")

# ── 1. Target: Total 15-min demand ─────────────────────────────────────
demand = (
    df.withColumn("datetime_15min", F.window("datetime_local", "15 minutes").start)
      .groupBy("datetime_15min")
      .agg(F.sum("active_kw").alias("demand_kw"))
      .orderBy("datetime_15min")
)
pdf_demand = demand.toPandas()
pdf_demand["datetime_15min"] = pd.to_datetime(pdf_demand["datetime_15min"])
merged = pdf_demand.sort_values("datetime_15min").set_index("datetime_15min")
print(f"Demand rows: {len(merged):,}")

# ── 2. OpenMeteo hourly weather → avg across communities, ffill to 15-min
openmeteo = spark.table("datathon.rubber_duckers.openmeteo_hourly_weather")
meteo_cols = [
    "temperature_2m", "relative_humidity_2m", "dew_point_2m",
    "apparent_temperature", "precipitation", "rain", "snowfall",
    "pressure_msl", "surface_pressure",
    "cloud_cover", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high",
    "wind_speed_10m", "wind_speed_100m", "wind_gusts_10m",
    "shortwave_radiation", "direct_radiation", "diffuse_radiation",
    "sunshine_duration", "vapour_pressure_deficit"
]
pdf_meteo = openmeteo.select("datetime_local", *meteo_cols).toPandas()
pdf_meteo["datetime_local"] = pd.to_datetime(pdf_meteo["datetime_local"])
for c in meteo_cols:
    pdf_meteo[c] = pd.to_numeric(pdf_meteo[c], errors="coerce")
pdf_meteo = (
    pdf_meteo.groupby("datetime_local")[meteo_cols]
    .mean().sort_index().resample("15min").ffill()
)
pdf_meteo.index.name = "datetime_15min"
merged = merged.join(pdf_meteo, how="left")
print(f"After OpenMeteo join: {merged.shape}")

# ── 3. AEMET daily weather → avg across stations ──────────────────────
aemet = spark.table("datathon.rubber_duckers.aemet_daily_weather")
aemet_cols = ["tmed", "prec", "tmin", "tmax", "velmedia", "racha",
              "sol", "presMax", "presMin", "hrMedia", "hrMax", "hrMin"]
pdf_aemet = aemet.select("fecha", *aemet_cols).toPandas()
pdf_aemet["fecha"] = pd.to_datetime(pdf_aemet["fecha"])
for c in aemet_cols:
    pdf_aemet[c] = pd.to_numeric(pdf_aemet[c], errors="coerce")
pdf_aemet = pdf_aemet.groupby("fecha")[aemet_cols].mean()
pdf_aemet.index.name = "date"

merged["date"] = merged.index.normalize()
merged = merged.join(pdf_aemet, on="date", how="left")

# ── 4. Regional temporal features → avg across regions ─────────────────
regional = spark.table("datathon.rubber_duckers.regional_features_temporal")
reg_cols = ["hdd_hourly", "cdd_hourly"]
pdf_reg = regional.select("date", *reg_cols).toPandas()
pdf_reg["date"] = pd.to_datetime(pdf_reg["date"])
for c in reg_cols:
    pdf_reg[c] = pd.to_numeric(pdf_reg[c], errors="coerce")
pdf_reg = pdf_reg.groupby("date")[reg_cols].mean()
merged = merged.join(pdf_reg, on="date", how="left")
merged = merged.drop(columns=["date"])

# ── 5. External forecasts (pv + wind, 15-min resolution) ──────────────
for tbl, alias in [("datathon.shared.pv_production_forecast", "pv_forecast"),
                   ("datathon.shared.wind_production_forecast", "wind_forecast")]:
    pdf_fc = spark.table(tbl).select(
        F.col("datetime_local").alias("datetime_15min"),
        F.col("value").alias(alias)
    ).toPandas()
    pdf_fc["datetime_15min"] = pd.to_datetime(pdf_fc["datetime_15min"])
    pdf_fc = pdf_fc.drop_duplicates("datetime_15min").set_index("datetime_15min")
    merged = merged.join(pdf_fc, how="left")
print(f"After all joins: {merged.shape}")

# ── 6. Time features ──────────────────────────────────────────────────
merged["hour"] = merged.index.hour + merged.index.minute / 60.0
merged["day_of_week"] = merged.index.dayofweek
merged["month"] = merged.index.month
merged["is_weekend"] = (merged.index.dayofweek >= 5).astype(int)
merged["hour_sin"] = np.sin(2 * np.pi * merged["hour"] / 24)
merged["hour_cos"] = np.cos(2 * np.pi * merged["hour"] / 24)

# ── 7. OLS on all candidate features ──────────────────────────────────
target_col = "demand_kw"
feature_candidates = [c for c in merged.columns if c != target_col]

ols_df = merged[feature_candidates + [target_col]].dropna()
print(f"OLS sample: {len(ols_df):,} rows after dropna")

# Subsample if very large
if len(ols_df) > 50000:
    ols_df = ols_df.sample(50000, random_state=42)

X = ols_df[feature_candidates]
y = ols_df[target_col]

# Standardize for comparable coefficients
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
X_const = sm.add_constant(X_scaled)

ols_model = sm.OLS(y, X_const).fit()

# ── 8. Report results ─────────────────────────────────────────────────
print("\n" + "="*70)
print("OLS FEATURE SELECTION RESULTS")
print("="*70)
print(f"R² = {ols_model.rsquared:.4f},  Adj R² = {ols_model.rsquared_adj:.4f}")
print(f"N  = {int(ols_model.nobs):,}")

results = pd.DataFrame({
    "feature": ols_model.params.index[1:],
    "std_coef": ols_model.params.values[1:],
    "|std_coef|": np.abs(ols_model.params.values[1:]),
    "p_value": ols_model.pvalues.values[1:],
    "sig": ["***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "" 
            for p in ols_model.pvalues.values[1:]]
}).sort_values("|std_coef|", ascending=False)

print(f"\nAll {len(results)} features ranked by |standardized coefficient|:")
print(results.to_string(index=False))

sig = results[results["p_value"] < 0.05]
print(f"\n{len(sig)} significant features (p < 0.05) out of {len(results)}")
print(f"\nRecommended features for model (significant, |coef| > median):")
median_coef = sig["|std_coef|"].median()
top_features = sig[sig["|std_coef|"] >= median_coef]["feature"].tolist()
print(top_features)

# COMMAND ----------

from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
import pandas as pd
from pyspark.sql import functions as F


class EnergyConsumptionModel:
    """
    Day-ahead energy consumption forecast using Gradient Boosting.
    Core: 8 exogenous + demand_forecast + demand lags + rolling mean.
    Forecasts resampled to 15-min to maximise training rows.
    """

    def __init__(self, n_estimators=600, max_depth=4, learning_rate=0.03,
                 subsample=0.7, min_samples_leaf=20):
        # Tuned parameters for better bias-variance tradeoff
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.min_samples_leaf = min_samples_leaf

    def predict(self, df, predict_start, predict_end):
        import warnings
        warnings.filterwarnings('ignore')

        # 1. Aggregate to total 15-min demand
        demand = (
            df.withColumn("datetime_15min",
                          F.window("datetime_local", "15 minutes").start)
              .groupBy("datetime_15min")
              .agg(F.sum("active_kw").alias("demand_kw"))
              .orderBy("datetime_15min")
        )

        # 2. Load external tables
        openmeteo = spark.table("datathon.rubber_duckers.openmeteo_hourly_weather")
        regional = spark.table("datathon.rubber_duckers.regional_features_temporal")
        calendar_tbl = spark.table("datathon.rubber_duckers.spanish_calendar_holidays_2026")

        # 3. Build demand pandas with datetime index
        pdf_demand = demand.toPandas()
        pdf_demand["datetime_15min"] = pd.to_datetime(pdf_demand["datetime_15min"])
        pdf_demand = pdf_demand.sort_values("datetime_15min").set_index("datetime_15min")

        # 3b. Demand lags & rolling stats
        lag_frames = {}
        for lag_days, name in [(7, "demand_lag_7d"), (1, "demand_lag_1d")]:
            tmp = pdf_demand[["demand_kw"]].copy()
            tmp.index = tmp.index + pd.Timedelta(days=lag_days)
            tmp = tmp.rename(columns={"demand_kw": name})
            lag_frames[name] = tmp

        # Rolling 7-day mean (shifted forward by 1 day to avoid leakage)
        roll = pdf_demand[["demand_kw"]].copy()
        roll["rolling_mean_7d"] = roll["demand_kw"].rolling(672, min_periods=96).mean()  # 672 = 7d * 96
        roll = roll[["rolling_mean_7d"]]
        roll.index = roll.index + pd.Timedelta(days=1)  # no leakage
        lag_frames["rolling_mean_7d"] = roll

        # 4. OpenMeteo hourly weather → avg across communities, ffill to 15-min
        weather_cols = [
            "dew_point_2m", "temperature_2m", "cloud_cover", "relative_humidity_2m"
        ]
        pdf_meteo = openmeteo.select("datetime_local", *weather_cols).toPandas()
        pdf_meteo["datetime_local"] = pd.to_datetime(pdf_meteo["datetime_local"])
        for c in weather_cols:
            pdf_meteo[c] = pd.to_numeric(pdf_meteo[c], errors="coerce")
        pdf_meteo = (
            pdf_meteo.groupby("datetime_local")[weather_cols]
            .mean().sort_index().resample("15min").ffill()
        )
        pdf_meteo.index.name = "datetime_15min"

        # 5. Regional temporal features
        reg_cols = ["hdd_hourly", "cdd_hourly"]
        pdf_reg = regional.select("date", *reg_cols).toPandas()
        pdf_reg["date"] = pd.to_datetime(pdf_reg["date"])
        for c in reg_cols:
            pdf_reg[c] = pd.to_numeric(pdf_reg[c], errors="coerce")
        pdf_reg = pdf_reg.groupby("date")[reg_cols].mean()

        # 6. Forecasts (hourly → resample to 15-min via ffill)
        forecast_dfs = {}
        for tbl, alias in [
            ("datathon.shared.demand_forecast", "demand_forecast"),
            ("datathon.shared.pv_production_forecast", "pv_forecast"),
            ("datathon.shared.wind_production_forecast", "wind_forecast")
        ]:
            pdf_fc = spark.table(tbl).select(
                F.col("datetime_local").alias("datetime_15min"),
                F.col("value").alias(alias)
            ).toPandas()
            pdf_fc["datetime_15min"] = pd.to_datetime(pdf_fc["datetime_15min"])
            pdf_fc = pdf_fc.drop_duplicates("datetime_15min").set_index("datetime_15min").sort_index()
            # Resample hourly → 15-min via forward-fill
            pdf_fc = pdf_fc.resample("15min").ffill()
            forecast_dfs[alias] = pdf_fc

        # 7. Calendar
        cal_cols = ["is_holiday", "is_weekend", "is_national_holiday"]
        pdf_cal = calendar_tbl.select("date", *cal_cols).toPandas()
        pdf_cal["date"] = pd.to_datetime(pdf_cal["date"])
        for c in cal_cols:
            pdf_cal[c] = pd.to_numeric(pdf_cal[c], errors="coerce")
        pdf_cal = pdf_cal.set_index("date")

        # 8. Merge onto demand axis
        merged = pdf_demand.copy()
        for name, frame in lag_frames.items():
            merged = merged.join(frame, how="left")
        merged = merged.join(pdf_meteo, how="left")
        for alias, pdf_fc in forecast_dfs.items():
            merged = merged.join(pdf_fc, how="left")

        merged["date"] = merged.index.normalize()
        merged = merged.join(pdf_reg, on="date", how="left")
        merged = merged.join(pdf_cal, on="date", how="left")
        merged = merged.drop(columns=["date"])

        # 9. Temporal / cyclical features
        merged["hour"] = merged.index.hour + merged.index.minute / 60.0
        merged["hour_sin"] = np.sin(2 * np.pi * merged["hour"] / 24)
        merged["hour_cos"] = np.cos(2 * np.pi * merged["hour"] / 24)
        merged["day_of_week"] = merged.index.dayofweek
        merged["month"] = merged.index.month
        merged["dow_sin"] = np.sin(2 * np.pi * merged["day_of_week"] / 7)
        merged["dow_cos"] = np.cos(2 * np.pi * merged["day_of_week"] / 7)
        merged["month_sin"] = np.sin(2 * np.pi * merged["month"] / 12)
        merged["month_cos"] = np.cos(2 * np.pi * merged["month"] / 12)

        _weekend_fallback = pd.Series(
            (merged.index.dayofweek >= 5).astype(int), index=merged.index
        )
        if "is_weekend" not in merged.columns:
            merged["is_weekend"] = _weekend_fallback
        else:
            merged["is_weekend"] = merged["is_weekend"].fillna(_weekend_fallback)

        for col in ["is_holiday", "is_national_holiday"]:
            if col in merged.columns:
                merged[col] = merged[col].fillna(0)
            else:
                merged[col] = 0

        # 10. Feature columns
        core_features = [
            "dew_point_2m", "temperature_2m", "hdd_hourly", "cdd_hourly",
            "pv_forecast", "cloud_cover", "wind_forecast", "relative_humidity_2m",
            "demand_forecast",
            "demand_lag_7d", "demand_lag_1d", "rolling_mean_7d"
        ]
        seasonal_features = [
            "is_weekend", "is_holiday", "hour", "day_of_week", "month",
            "hour_sin", "hour_cos", "dow_sin", "dow_cos",
            "month_sin", "month_cos"
        ]
        all_features = core_features + seasonal_features
        target_col = "demand_kw"

        # 11. Split
        ps = pd.Timestamp(predict_start)
        pe = pd.Timestamp(predict_end)

        train_mask = merged.index < ps
        pred_mask = (merged.index >= ps) & (merged.index < pe)

        train_df = merged.loc[train_mask, all_features + [target_col]].dropna()
        X_train = train_df[all_features].values
        y_train = train_df[target_col].values

        print(f"Training GB on {len(X_train):,} rows, {len(all_features)} features (before {predict_start})")

        # 12. Train Gradient Boosting
        gb = GradientBoostingRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            min_samples_leaf=self.min_samples_leaf,
            random_state=42
        )
        gb.fit(X_train, y_train)

        # 13. Predict
        pred_df = merged.loc[pred_mask].copy()

        if len(pred_df) > 0:
            train_medians = train_df[all_features].median()
            for col in all_features:
                pred_df[col] = pred_df[col].fillna(train_medians.get(col, 0))

            X_pred = pred_df[all_features].values
            pred_df["prediction"] = np.clip(gb.predict(X_pred), 0, None)
            result_pdf = pred_df[["prediction"]].reset_index()
            result_pdf.columns = ["datetime_15min", "prediction"]
        else:
            dt_range = pd.date_range(ps, pe, freq="15min", inclusive="left")
            grid = pd.DataFrame({"datetime_15min": dt_range}).set_index("datetime_15min")

            for name, frame in lag_frames.items():
                grid = grid.join(frame, how="left")
            grid = grid.join(pdf_meteo, how="left")
            for alias, pdf_fc in forecast_dfs.items():
                grid = grid.join(pdf_fc, how="left")
            grid["date"] = grid.index.normalize()
            grid = grid.join(pdf_reg, on="date", how="left")
            grid = grid.join(pdf_cal, on="date", how="left")
            grid = grid.drop(columns=["date"])

            grid["hour"] = grid.index.hour + grid.index.minute / 60.0
            grid["hour_sin"] = np.sin(2 * np.pi * grid["hour"] / 24)
            grid["hour_cos"] = np.cos(2 * np.pi * grid["hour"] / 24)
            grid["day_of_week"] = grid.index.dayofweek
            grid["month"] = grid.index.month
            grid["dow_sin"] = np.sin(2 * np.pi * grid["day_of_week"] / 7)
            grid["dow_cos"] = np.cos(2 * np.pi * grid["day_of_week"] / 7)
            grid["month_sin"] = np.sin(2 * np.pi * grid["month"] / 12)
            grid["month_cos"] = np.cos(2 * np.pi * grid["month"] / 12)
            grid["is_weekend"] = (grid.index.dayofweek >= 5).astype(int)
            grid["is_holiday"] = grid.get("is_holiday", pd.Series(0, index=grid.index)).fillna(0)
            grid["is_national_holiday"] = 0

            train_medians = train_df[all_features].median()
            for col in all_features:
                if col not in grid.columns:
                    grid[col] = train_medians.get(col, 0)
                grid[col] = grid[col].fillna(train_medians.get(col, 0))

            X_grid = grid[all_features].values
            grid["prediction"] = np.clip(gb.predict(X_grid), 0, None)
            result_pdf = grid[["prediction"]].reset_index()
            result_pdf.columns = ["datetime_15min", "prediction"]

        print(f"Predictions: {len(result_pdf):,} intervals")

        return spark.createDataFrame(result_pdf).select(
            F.col("datetime_15min").cast("timestamp"),
            F.col("prediction").cast("double")
        )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Local Validation: Train/Test Split
# MAGIC
# MAGIC The training set is everything **before 2025-12-01** and the test set covers
# MAGIC **2025-12-01 to 2026-02-28**.
# MAGIC
# MAGIC **Tip:** Change `val_start` and `val_end` to test on different periods
# MAGIC (e.g. November 2025). This lets you check whether your model
# MAGIC generalises well across time before spending a submission.

# COMMAND ----------

# Train/test split: hold out Dec 2025 – Feb 2026 as a local test set.
# The model receives all data up to end of Feb 2026 (including the test period) for
# feature engineering, but must only return predictions for intervals in [val_start, val_end).
val_start = "2025-11-01"
val_end = "2025-12-01"

model = EnergyConsumptionModel()
val_preds = model.predict(df, predict_start=val_start, predict_end=val_end)

# Actuals for the hold-out period
actuals_val = (
    df.filter(
        (F.col("datetime_local") >= val_start) & (F.col("datetime_local") < val_end)
    )
    .withColumn("datetime_15min", F.window("datetime_local", "15 minutes").start)
    .groupBy("datetime_15min")
    .agg(F.sum("active_kw").alias("active_kw"))
)

merged = val_preds.join(actuals_val, on="datetime_15min", how="inner")
local_mae = merged.select(
    F.mean(F.abs(F.col("active_kw") - F.col("prediction")))
).collect()[0][0]

if local_mae is not None:
    print(f"Local validation MAE ({val_start} to {val_end}): {local_mae:.4f}")
    print(f"Intervals predicted: {merged.count():,}")
else:
    print("WARNING: No predictions matched the hold-out period. Check your model.")

# COMMAND ----------

# Diagnostic: understand prediction quality
merged_pdf = merged.select("datetime_15min", "prediction", "active_kw").toPandas()
merged_pdf = merged_pdf.sort_values("datetime_15min").reset_index(drop=True)

print("=== Prediction vs Actual Summary ===")
print(f"Actual  mean: {merged_pdf['active_kw'].mean():,.0f} kW")
print(f"Predict mean: {merged_pdf['prediction'].mean():,.0f} kW")
bias = (merged_pdf['prediction'] - merged_pdf['active_kw']).mean()
print(f"Bias (pred-actual): {bias:,.0f} kW ({bias/merged_pdf['active_kw'].mean()*100:.1f}%)")
print(f"MAE: {(merged_pdf['prediction'] - merged_pdf['active_kw']).abs().mean():,.0f} kW")
print(f"\nActual  range: [{merged_pdf['active_kw'].min():,.0f}, {merged_pdf['active_kw'].max():,.0f}]")
print(f"Predict range: [{merged_pdf['prediction'].min():,.0f}, {merged_pdf['prediction'].max():,.0f}]")
print(f"Actual  std: {merged_pdf['active_kw'].std():,.0f} kW")
print(f"Predict std: {merged_pdf['prediction'].std():,.0f} kW")

# Hourly pattern
merged_pdf['hour'] = merged_pdf['datetime_15min'].dt.hour
hourly = merged_pdf.groupby('hour')[['active_kw', 'prediction']].mean()
print("\n=== Hourly Avg ===\n Hour  | Actual    | Predicted | Error")
for h in range(0, 24, 3):
    a, p = hourly.loc[h, 'active_kw'], hourly.loc[h, 'prediction']
    print(f"  {h:2d}   | {a:>9,.0f} | {p:>9,.0f} | {p-a:+,.0f} ({(p-a)/a*100:+.1f}%)")

# Day-of-week pattern
merged_pdf['dow'] = merged_pdf['datetime_15min'].dt.dayofweek
dow_names = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
daily = merged_pdf.groupby('dow')[['active_kw', 'prediction']].mean()
print("\n=== Day-of-Week Avg ===\n Day | Actual    | Predicted | Error")
for d in range(7):
    a, p = daily.loc[d, 'active_kw'], daily.loc[d, 'prediction']
    print(f"  {dow_names[d]} | {a:>9,.0f} | {p:>9,.0f} | {p-a:+,.0f} ({(p-a)/a*100:+.1f}%)")

# COMMAND ----------

# ═══ OVERFITTING DIAGNOSTIC ═══
# Compare error patterns to distinguish overfitting from systematic bias:
# - Overfitting → random/high-variance errors, some days good, some bad
# - Systematic bias → uniform errors, all days biased the same way

merged_pdf['date'] = merged_pdf['datetime_15min'].dt.date
merged_pdf['error'] = merged_pdf['prediction'] - merged_pdf['active_kw']
merged_pdf['pct_error'] = merged_pdf['error'] / merged_pdf['active_kw'] * 100

daily_stats = merged_pdf.groupby('date').agg(
    mae=('error', lambda x: x.abs().mean()),
    bias=('error', 'mean'),
    pct_bias=('pct_error', 'mean'),
    actual_mean=('active_kw', 'mean')
).reset_index()

print("═══ OVERFITTING DIAGNOSTIC ═══\n")

# Test 1: Per-day error consistency
print("Test 1: Per-day error consistency")
print(f"  Daily bias mean:  {daily_stats['pct_bias'].mean():+.1f}%")
print(f"  Daily bias std:    {daily_stats['pct_bias'].std():.1f}%")
print(f"  Daily bias range: [{daily_stats['pct_bias'].min():+.1f}%, "
      f"{daily_stats['pct_bias'].max():+.1f}%]")
print(f"  Days with positive bias: {(daily_stats['pct_bias'] > 0).sum()}/{len(daily_stats)}")
print(f"  Days with negative bias: {(daily_stats['pct_bias'] < 0).sum()}/{len(daily_stats)}")

# Test 2: Error by week (does error grow further from training?)
merged_pdf['days_from_start'] = (merged_pdf['datetime_15min'] - merged_pdf['datetime_15min'].min()).dt.days
weekly = merged_pdf.groupby(merged_pdf['days_from_start'] // 7).agg(
    pct_bias=('pct_error', 'mean'),
    mae=('error', lambda x: x.abs().mean())
).reset_index()

print(f"\nTest 2: Error drift over time (week-by-week)")
print(f"  Week | Bias     | MAE (kW)")
for _, row in weekly.iterrows():
    w = int(row['days_from_start'])
    print(f"    {w+1}  | {row['pct_bias']:+.1f}%   | {row['mae']:,.0f}")

# Test 3: Bias direction — overfitting produces mixed signs
all_under = (daily_stats['bias'] < 0).all()
print(f"\nTest 3: Bias direction")
print(f"  All {len(daily_stats)} days under-predicted? {all_under}")

# ── Verdict ──
bias_std = daily_stats['pct_bias'].std()
all_same_sign = all_under or (daily_stats['bias'] > 0).all()
drift = abs(weekly['pct_bias'].iloc[-1] - weekly['pct_bias'].iloc[0])

print(f"\n{'='*55}")
if all_same_sign and bias_std < 5.0:
    print("  VERDICT: NOT OVERFITTING")
    print("  Evidence: uniform bias across all days, low day-to-day")
    print("  variance, no temporal drift. This is a calibration issue.")
    print("  ")
    print("  Likely causes:")
    print("  1. log1p/expm1 Jensen's inequality: E[expm1(X)] < expm1(E[X])")
    print("     creates systematic under-prediction for right-skewed data")
    print("  2. z=0 deterministic inference: decoder trained with")
    print("     stochastic z but queried with z=0 → conservative output")
else:
    print("  VERDICT: POSSIBLE OVERFITTING")
    print(f"  Bias variance: {bias_std:.1f}%, drift: {drift:.1f}%")
print(f"{'='*55}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps
# MAGIC
# MAGIC When you are happy with your model's local MAE:
# MAGIC 1. **Copy** your `EnergyConsumptionModel` class (and any imports it needs) to the **submission notebook**.
# MAGIC 2. Run the **Submit** cell there.