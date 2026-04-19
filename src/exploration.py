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

# DBTITLE 1,Install torch
# MAGIC %pip install --force-reinstall typing_extensions>=4.10 torch

# COMMAND ----------

# Fix for the typing_extensions error and ensure torch/sklearn are up to date
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
print(f" Global MAE : {mae:,.2f}")
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

# DBTITLE 1,Local Model
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, TensorDataset
# from sklearn.preprocessing import StandardScaler
# import numpy as np
# import pandas as pd
# from pyspark.sql import functions as F, Window as W


# class EnergyConsumptionModel:
#     """
#     Day-ahead energy consumption forecast using an aggregate-level
#     Conditional VAE on total daily load profiles (96 × 15-min intervals).

#     Key features:
#     - Predicts total demand directly (no per-client scaling)
#     - Same-DOW lag profiles (7d and 14d) preserve weekday/weekend patterns
#     - Posterior-guided sampling anchors predictions to recent similar days
#     - is_weekend + calendar + external forecast conditioning
#     """

#     def __init__(self, epochs=100, latent_dim=32, batch_size=32,
#                  n_samples=10, lr=1e-3):
#         self.epochs = epochs
#         self.latent_dim = latent_dim
#         self.batch_size = batch_size
#         self.n_samples = n_samples
#         self.lr = lr

#     def predict(self, df, predict_start, predict_end):
#         """
#         Parameters
#         ----------
#         df : PySpark DataFrame  – datathon.shared.client_consumption
#         predict_start, predict_end : str – date range to predict

#         Returns
#         -------
#         PySpark DataFrame with columns datetime_15min (timestamp),
#         prediction (double)  – total demand across all clients.
#         """
#         import warnings
#         warnings.filterwarnings('ignore')

#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         print(f"Using device: {device}")

#         # ── 1. Load external day-ahead forecasts ────────────────────────
#         demand_fc = spark.table("datathon.shared.demand_forecast") \
#             .withColumnRenamed("value", "demand_fc")
#         pv_fc = spark.table("datathon.shared.pv_production_forecast") \
#             .withColumnRenamed("value", "pv_fc")
#         wind_fc = spark.table("datathon.shared.wind_production_forecast") \
#             .withColumnRenamed("value", "wind_fc")

#         # ── 2. Aggregate to total 15-min consumption (all clients) ──────
#         consumption = df.filter(
#             F.to_date("datetime_local") < predict_start
#         )

#         agg = (
#             consumption
#             .withColumn("dt15",
#                 F.window("datetime_local", "15 minutes").start)
#             .groupBy("dt15")
#             .agg(F.sum("active_kw").alias("total_kw"))
#         )

#         forecasts = (
#             demand_fc.join(pv_fc, "datetime_local", "outer")
#                      .join(wind_fc, "datetime_local", "outer")
#                      .withColumnRenamed("datetime_local", "dt15")
#         )
#         agg = agg.join(forecasts, "dt15", "left")

#         agg = (agg
#             .withColumn("date", F.to_date("dt15"))
#             .withColumn("interval_idx",
#                 (F.hour("dt15") * 4 + F.floor(F.minute("dt15") / 15))
#                 .cast("int"))
#             .withColumn("day_of_week", F.dayofweek("date"))
#             .withColumn("month", F.month("date"))
#         )

#         # ── 3. Pivot to 1 daily profile per day (96 intervals) ──────────
#         daily_profiles = (
#             agg.groupBy("date")
#             .pivot("interval_idx", list(range(96)))
#             .agg(F.first("total_kw"))
#         ).fillna(0)

#         daily_cond = agg.groupBy("date").agg(
#             F.first("day_of_week").alias("day_of_week"),
#             F.first("month").alias("month"),
#             F.avg("demand_fc").alias("demand_fc_mean"),
#             F.avg("pv_fc").alias("pv_fc_mean"),
#             F.avg("wind_fc").alias("wind_fc_mean"),
#             F.max("demand_fc").alias("demand_fc_max"),
#             F.min("demand_fc").alias("demand_fc_min"),
#         )

#         final_df = daily_profiles.join(daily_cond, "date", "inner")
#         final_df = final_df.fillna(0, subset=[
#             "demand_fc_mean", "pv_fc_mean", "wind_fc_mean",
#             "demand_fc_max", "demand_fc_min"
#         ])

#         # ── 4. Collect (small: ~300 rows) ───────────────────────────────
#         profile_cols = [str(i) for i in range(96)]
#         cond_cols = ["demand_fc_mean", "pv_fc_mean", "wind_fc_mean",
#                      "demand_fc_max", "demand_fc_min"]
#         select_cols = (
#             ["date"] + profile_cols + cond_cols +
#             ["day_of_week", "month"]
#         )

#         print("Collecting aggregate daily profiles...")
#         rows = final_df.orderBy("date").select(*select_cols).collect()
#         print(f"Collected {len(rows):,} daily profiles")

#         n_p = len(profile_cols)
#         n_c = len(cond_cols)

#         dates_arr = np.array([r[0] for r in rows])
#         profiles_raw = np.array(
#             [[float(r[1 + j]) for j in range(n_p)] for r in rows],
#             dtype=np.float32)
#         conds_raw = np.array(
#             [[float(r[1 + n_p + j]) for j in range(n_c)] for r in rows],
#             dtype=np.float32)
#         cats = np.array(
#             [[int(r[1 + n_p + n_c + j]) for j in range(2)] for r in rows],
#             dtype=np.int64)  # [day_of_week, month]

#         # ── 5. Same-DOW lag profiles (preserves weekday/weekend) ───────
#         N = len(profiles_raw)
#         lag7 = np.zeros_like(profiles_raw)
#         lag14 = np.zeros_like(profiles_raw)
#         for i in range(N):
#             # Exact same day-of-week: 7 days ago
#             lag7[i] = profiles_raw[i - 7] if i >= 7 \
#                 else profiles_raw[i]
#             # 14 days ago (same DOW)
#             lag14[i] = profiles_raw[i - 14] if i >= 14 \
#                 else profiles_raw[max(i - 7, 0)]

#         # ── 6. Scale profiles and conditions ────────────────────────────
#         profile_scaler = StandardScaler()
#         P_scaled = profile_scaler.fit_transform(profiles_raw)

#         lag7_scaled = profile_scaler.transform(lag7)
#         lag14_scaled = profile_scaler.transform(lag14)

#         cond_scaler = StandardScaler()
#         C_cont = cond_scaler.fit_transform(conds_raw).astype(np.float32)

#         # ── 7. Build condition matrix ───────────────────────────────────
#         # continuous(5) + lag7(96) + lag14(96) + DOW(7) + month(12)
#         # + is_weekend(1) = 217 dims
#         dow_onehot = np.zeros((N, 7), dtype=np.float32)
#         for i in range(N):
#             dow_onehot[i, cats[i, 0] - 1] = 1.0

#         month_onehot = np.zeros((N, 12), dtype=np.float32)
#         for i in range(N):
#             month_onehot[i, cats[i, 1] - 1] = 1.0

#         # is_weekend: PySpark dayofweek Sun=1, Sat=7
#         is_weekend = np.array(
#             [[1.0 if cats[i, 0] in (1, 7) else 0.0] for i in range(N)],
#             dtype=np.float32
#         )

#         all_cond = np.hstack([
#             C_cont,            # 5 forecast features
#             lag7_scaled,       # 96 same-DOW lag-7d
#             lag14_scaled,      # 96 same-DOW lag-14d
#             dow_onehot,        # 7 day of week
#             month_onehot,      # 12 month
#             is_weekend,        # 1 weekend flag
#         ]).astype(np.float32)
#         cond_dim = all_cond.shape[1]

#         # ── 8. DataLoader ───────────────────────────────────────────────
#         train_ds = TensorDataset(
#             torch.tensor(P_scaled.astype(np.float32)),
#             torch.tensor(all_cond),
#         )
#         train_loader = DataLoader(
#             train_ds, batch_size=self.batch_size,
#             shuffle=True, drop_last=True
#         )

#         # ── 9. Define aggregate CVAE ────────────────────────────────────
#         lat_d = self.latent_dim

#         class CVAE(nn.Module):
#             def __init__(self):
#                 super().__init__()
#                 self.encoder = nn.Sequential(
#                     nn.Linear(96 + cond_dim, 512),
#                     nn.BatchNorm1d(512), nn.LeakyReLU(0.2),
#                     nn.Dropout(0.15),
#                     nn.Linear(512, 256),
#                     nn.BatchNorm1d(256), nn.LeakyReLU(0.2),
#                 )
#                 self.fc_mu = nn.Linear(256, lat_d)
#                 self.fc_logvar = nn.Linear(256, lat_d)
#                 self.decoder = nn.Sequential(
#                     nn.Linear(lat_d + cond_dim, 256),
#                     nn.BatchNorm1d(256), nn.LeakyReLU(0.2),
#                     nn.Dropout(0.15),
#                     nn.Linear(256, 512),
#                     nn.BatchNorm1d(512), nn.LeakyReLU(0.2),
#                     nn.Linear(512, 96)
#                 )

#             def forward(self, target, cond):
#                 h = self.encoder(torch.cat([target, cond], dim=1))
#                 mu, lv = self.fc_mu(h), self.fc_logvar(h)
#                 z = mu + torch.randn_like(lv) * torch.exp(0.5 * lv)
#                 recon = self.decoder(torch.cat([z, cond], dim=1))
#                 return recon, mu, lv

#             def encode_mean(self, target, cond):
#                 """Encode to posterior mean (no sampling)."""
#                 h = self.encoder(torch.cat([target, cond], dim=1))
#                 return self.fc_mu(h)

#             def decode(self, z, cond):
#                 return self.decoder(torch.cat([z, cond], dim=1))

#             def sample(self, cond):
#                 z = torch.randn(cond.size(0), lat_d).to(cond.device)
#                 return self.decoder(torch.cat([z, cond], dim=1))

#         # ── 10. Train CVAE ──────────────────────────────────────────────
#         cvae = CVAE().to(device)
#         opt = optim.Adam(cvae.parameters(), lr=self.lr, weight_decay=1e-5)
#         scheduler = optim.lr_scheduler.CosineAnnealingLR(
#             opt, T_max=self.epochs
#         )

#         print(f"Training aggregate CVAE for {self.epochs} epochs "
#               f"on {device} (cond_dim={cond_dim})...")
#         for ep in range(self.epochs):
#             cvae.train()
#             ep_loss = 0
#             n_batches = 0
#             for target, cond in train_loader:
#                 target = target.to(device)
#                 cond = cond.to(device)
#                 opt.zero_grad(set_to_none=True)
#                 recon, mu, lv = cvae(target, cond)
#                 mse = nn.functional.mse_loss(recon, target)
#                 kl = -0.5 * torch.mean(
#                     1 + lv - mu.pow(2) - lv.exp()
#                 )
#                 beta = min(0.1, 0.001 * (ep + 1))
#                 loss = mse + beta * kl
#                 loss.backward()
#                 nn.utils.clip_grad_norm_(cvae.parameters(), 1.0)
#                 opt.step()
#                 ep_loss += loss.item()
#                 n_batches += 1
#             scheduler.step()
#             if (ep + 1) % 20 == 0 or ep == 0:
#                 avg = ep_loss / max(n_batches, 1)
#                 print(f"  Epoch {ep+1:3d}/{self.epochs} | "
#                       f"Loss: {avg:.6f}")

#         # ── 11. Generate predictions ────────────────────────────────────
#         cvae.eval()

#         # Forecast conditions for prediction dates
#         pred_dates = pd.date_range(
#             predict_start, predict_end, freq='D', inclusive='left'
#         )

#         pred_fc_rows = (
#             forecasts
#             .withColumn("date", F.to_date("dt15"))
#             .filter(
#                 (F.col("date") >= predict_start)
#                 & (F.col("date") < predict_end)
#             )
#             .groupBy("date").agg(
#                 F.avg("demand_fc").alias("demand_fc_mean"),
#                 F.avg("pv_fc").alias("pv_fc_mean"),
#                 F.avg("wind_fc").alias("wind_fc_mean"),
#                 F.max("demand_fc").alias("demand_fc_max"),
#                 F.min("demand_fc").alias("demand_fc_min"),
#             )
#         ).collect()

#         fc_by_date = {}
#         for r in pred_fc_rows:
#             fc_by_date[r["date"]] = np.array([
#                 float(r["demand_fc_mean"] or 0),
#                 float(r["pv_fc_mean"] or 0),
#                 float(r["wind_fc_mean"] or 0),
#                 float(r["demand_fc_max"] or 0),
#                 float(r["demand_fc_min"] or 0),
#             ], dtype=np.float32)

#         train_fc_median = np.nanmedian(conds_raw, axis=0)

#         # Keep a rolling buffer of recent profiles for lag conditioning
#         # Start with the last 14 days of training data
#         recent_profiles = list(profiles_raw[-14:])

#         print(f"Predicting {len(pred_dates)} days "
#               f"(posterior-guided sampling)...")

#         all_preds = {}
#         with torch.no_grad():
#             for date in pred_dates:
#                 d = date.date()
#                 # PySpark dayofweek: Sun=1..Sat=7
#                 spark_dow = (date.dayofweek + 2) % 7
#                 if spark_dow == 0:
#                     spark_dow = 7
#                 month = date.month
#                 is_wknd = 1.0 if spark_dow in (1, 7) else 0.0

#                 # Forecast conditions
#                 if d in fc_by_date:
#                     fc_raw = fc_by_date[d].reshape(1, -1)
#                 else:
#                     fc_raw = train_fc_median.reshape(1, -1)
#                 fc_scaled = cond_scaler.transform(
#                     fc_raw.astype(np.float32)
#                 ).astype(np.float32)

#                 # Same-DOW lag: 7 days ago and 14 days ago
#                 rp = np.array(recent_profiles)
#                 l7_raw = rp[-7].reshape(1, -1)   # exactly 7 days ago
#                 l14_raw = rp[-14].reshape(1, -1)  # exactly 14 days ago
#                 l7 = profile_scaler.transform(l7_raw).astype(np.float32)
#                 l14 = profile_scaler.transform(l14_raw).astype(np.float32)

#                 # One-hot calendar
#                 dow_oh = np.zeros((1, 7), dtype=np.float32)
#                 dow_oh[0, spark_dow - 1] = 1.0
#                 month_oh = np.zeros((1, 12), dtype=np.float32)
#                 month_oh[0, month - 1] = 1.0
#                 wknd = np.array([[is_wknd]], dtype=np.float32)

#                 cond_vec = np.hstack([
#                     fc_scaled, l7, l14, dow_oh, month_oh, wknd
#                 ])
#                 cond_t = torch.tensor(cond_vec).to(device)

#                 # Posterior-guided sampling:
#                 # Encode the lag-7d profile to get z_mean, then
#                 # sample near it (not from N(0,I)) to anchor to
#                 # recent similar-day patterns.
#                 l7_t = torch.tensor(l7).to(device)
#                 z_mean = cvae.encode_mean(l7_t, cond_t)

#                 batch_preds = torch.zeros(1, 96, device=device)
#                 for _ in range(self.n_samples):
#                     z = z_mean + 0.3 * torch.randn_like(z_mean)
#                     batch_preds += cvae.decode(z, cond_t)
#                 batch_preds /= self.n_samples

#                 # Inverse-scale to kW
#                 pred_profile = profile_scaler.inverse_transform(
#                     batch_preds.cpu().numpy()
#                 )[0]
#                 pred_profile = np.clip(pred_profile, 0, None)

#                 all_preds[d] = pred_profile
#                 recent_profiles.append(pred_profile)

#         # ── 12. Build result DataFrame ──────────────────────────────────
#         result_rows = []
#         for d in sorted(all_preds.keys()):
#             profile = all_preds[d]
#             base_ts = pd.Timestamp(d)
#             for interval in range(96):
#                 ts = base_ts + pd.Timedelta(minutes=15 * interval)
#                 result_rows.append((ts, float(profile[interval])))

#         result_pdf = pd.DataFrame(
#             result_rows, columns=["datetime_15min", "prediction"]
#         )
#         print(f"Predictions: {len(result_pdf):,} intervals")

#         return spark.createDataFrame(result_pdf).select(
#             F.col("datetime_15min").cast("timestamp"),
#             F.col("prediction").cast("double")
#         )

# COMMAND ----------

# DBTITLE 1,Import Model
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from pyspark.sql import functions as F, Window as W

class EnergyConsumptionModel:
    """
    Day-ahead energy consumption forecast using a per-community
    Conditional VAE on daily load profiles (96 × 15-min intervals).

    * ENHANCEMENTS APPLIED:
    - Community Aggregation: Forecasts at the community level instead of client level.
    - Robust Transformation: Log1p to handle zeros/outliers.
    - Full Intraday Forecasts: Conditions on the 96-step vectors of PV, Wind, and Grid Demand.
    - Non-leaking historical medians.
    - Deterministic Inference: Queries expected latent mean (Z=0) to remove stochastic noise.
    """

    def __init__(self, epochs=40, latent_dim=32,
                 batch_size=32, lr=1e-3):
        # sample_frac and n_samples removed: we aggregate 100% of clients 
        # and use deterministic (zero-sampled) expected inference.
        self.epochs = epochs
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.lr = lr

    def predict(self, df, predict_start, predict_end):
        import warnings
        warnings.filterwarnings('ignore')

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        spark = df.sparkSession

        # ── 1. Load external day-ahead forecasts ────────────────────────
        demand_fc = spark.table("datathon.shared.demand_forecast").withColumnRenamed("value", "demand_fc")
        pv_fc = spark.table("datathon.shared.pv_production_forecast").withColumnRenamed("value", "pv_fc")
        wind_fc = spark.table("datathon.shared.wind_production_forecast").withColumnRenamed("value", "wind_fc")

        forecasts = (
            demand_fc.join(pv_fc, "datetime_local", "outer")
                     .join(wind_fc, "datetime_local", "outer")
                     .withColumnRenamed("datetime_local", "dt15")
        )

        # Pivot forecasts to 96 intervals (288 continuous features total)
        fc_daily = (
            forecasts
            .withColumn("date", F.to_date("dt15"))
            .withColumn("interval_idx", (F.hour("dt15") * 4 + F.floor(F.minute("dt15") / 15)).cast("int"))
            .groupBy("date")
            .pivot("interval_idx", list(range(96)))
            .agg(
                F.first("demand_fc").alias("d"),
                F.first("pv_fc").alias("p"),
                F.first("wind_fc").alias("w")
            )
        ).fillna(0)
        
        fc_cols = [f"{i}_{f}" for i in range(96) for f in ['d', 'p', 'w']]

        # ── 2. Filter and Aggregate to Community Level ──────────────────
        print("Aggregating all clients to community level...")
        consumption = df.filter(F.to_date("datetime_local") < predict_start)

        agg = (
            consumption
            .withColumn("dt15", F.window("datetime_local", "15 minutes").start)
            .groupBy("community_code", "dt15")
            .agg(F.sum("active_kw").alias("total_kw"))
        )

        agg = (agg
            .withColumn("date", F.to_date("dt15"))
            .withColumn("interval_idx", (F.hour("dt15") * 4 + F.floor(F.minute("dt15") / 15)).cast("int"))
            .withColumn("day_of_week", F.dayofweek("date"))
            .withColumn("month", F.month("date"))
            .withColumn("is_weekend", F.when(F.dayofweek("date").isin([1, 7]), 1).otherwise(0))
        )

        # ── 3. Pivot daily profiles and historical baselines ────────────
        daily_profiles = (
            agg.groupBy("community_code", "date", "day_of_week", "month", "is_weekend")
            .pivot("interval_idx", list(range(96)))
            .agg(F.first("total_kw"))
        ).fillna(0)

        # Calculate robust median profile per community, per day of week using strictly historical data
        historical_baselines = (
            daily_profiles
            .groupBy("community_code", "day_of_week")
            .agg(*[F.expr(f"percentile_approx(`{i}`, 0.5)").alias(f"hist_{i}") for i in range(96)])
        )

        final_df = daily_profiles.join(historical_baselines, ["community_code", "day_of_week"], "left").fillna(0)
        final_df = final_df.join(fc_daily, "date", "inner")

        # ── 4. Label encoding via dense_rank ────────────────────────────
        final_df = (
            final_df
            .withColumn("community_idx", (F.dense_rank().over(W.orderBy("community_code")) - 1).cast("int"))
        )

        NUM_COMMUNITIES = final_df.agg(F.max("community_idx")).collect()[0][0] + 1
        print(f"Total communities tracked: {NUM_COMMUNITIES}")

        # ── 5. Collect compact dataset ──────────────────────────────────
        profile_cols = [str(i) for i in range(96)]
        hist_cols = [f"hist_{i}" for i in range(96)]
        cat_cols = ["community_idx", "day_of_week", "month", "is_weekend"]
        
        select_cols = ["community_code"] + profile_cols + hist_cols + fc_cols + cat_cols

        print("Collecting training data...")
        rows = final_df.select(*select_cols).collect()
        print(f"Collected {len(rows):,} rows")

        # ── 6. Convert to numpy ─────────────────────────────────────────
        n_p, n_h, n_f = len(profile_cols), len(hist_cols), len(fc_cols)

        comm_ids_arr = np.array([str(r[0]) for r in rows])
        
        profiles_raw = np.array([[float(r[1 + j]) for j in range(n_p)] for r in rows], dtype=np.float32)
        hist_raw = np.array([[float(r[1 + n_p + j]) for j in range(n_h)] for r in rows], dtype=np.float32)
        fc_raw = np.array([[float(r[1 + n_p + n_h + j]) for j in range(n_f)] for r in rows], dtype=np.float32)
        cats = np.array([[int(r[1 + n_p + n_h + n_f + j]) for j in range(len(cat_cols))] for r in rows], dtype=np.int64)

        # ── 7. Scaling ──────────────────────────────────────────────────
        comm_scalers = {}
        P_scaled = np.zeros_like(profiles_raw)
        H_scaled = np.zeros_like(hist_raw)

        for cid in np.unique(comm_ids_arr):
            mask = comm_ids_arr == cid
            if mask.sum() == 0: continue
            scaler = StandardScaler()
            
            # Log1p transformation handles massive outliers and avoids zero-variance explosions
            log_profiles = np.log1p(np.clip(profiles_raw[mask], 0, None))
            scaler.fit(log_profiles)
            comm_scalers[cid] = scaler
            
            P_scaled[mask] = scaler.transform(log_profiles)
            log_hist = np.log1p(np.clip(hist_raw[mask], 0, None))
            H_scaled[mask] = scaler.transform(log_hist)

        cond_scaler = StandardScaler()
        FC_scaled = cond_scaler.fit_transform(fc_raw).astype(np.float32)

        # Combine Continuous Conditions (Forecasts + Historical Baselines)
        C_cont = np.concatenate([FC_scaled, H_scaled], axis=1).astype(np.float32)

        # ── 8. DataLoader ───────────────────────────────────────────────
        train_ds = TensorDataset(torch.tensor(P_scaled), torch.tensor(C_cont), torch.tensor(cats))
        # Use drop_last=False unless the batch leaves exactly 1 item (which crashes BatchNorm1d)
        drop_last = (len(train_ds) % self.batch_size == 1)
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, drop_last=drop_last)

        # ── 9. Define Expanded CVAE ─────────────────────────────────────
        class CVAE(nn.Module):
            def __init__(self, num_communities, lat_d=32, cont_dim=384):
                super().__init__()
                self.comm_emb = nn.Embedding(num_communities, 16)
                self.dow_emb = nn.Embedding(8, 3)
                self.month_emb = nn.Embedding(13, 3)
                self.wend_emb = nn.Embedding(2, 2)
                
                self.cat_dim = 16 + 3 + 3 + 2
                self.cond_dim = cont_dim + self.cat_dim
                
                # Restored capacity: 512 -> 256 -> 128 to handle 400+ conditions without bottlenecking
                self.encoder = nn.Sequential(
                    nn.Linear(96 + self.cond_dim, 512),
                    nn.BatchNorm1d(512), nn.LeakyReLU(0.2), nn.Dropout(0.1),
                    nn.Linear(512, 256),
                    nn.BatchNorm1d(256), nn.LeakyReLU(0.2), nn.Dropout(0.1),
                    nn.Linear(256, 128),
                    nn.BatchNorm1d(128), nn.LeakyReLU(0.2)
                )
                self.fc_mu = nn.Linear(128, lat_d)
                self.fc_logvar = nn.Linear(128, lat_d)
                
                self.decoder = nn.Sequential(
                    nn.Linear(lat_d + self.cond_dim, 256),
                    nn.BatchNorm1d(256), nn.LeakyReLU(0.2), nn.Dropout(0.1),
                    nn.Linear(256, 512),
                    nn.BatchNorm1d(512), nn.LeakyReLU(0.2), nn.Dropout(0.1),
                    nn.Linear(512, 96)
                )
                self.lat_d = lat_d

            def process_conditions(self, cont, cat):
                co = self.comm_emb(cat[:, 0])
                dw = self.dow_emb(cat[:, 1])
                mo = self.month_emb(cat[:, 2])
                we = self.wend_emb(cat[:, 3])
                return torch.cat([cont, co, dw, mo, we], dim=1)

            def forward(self, target, cont, cat):
                cond = self.process_conditions(cont, cat)
                h = self.encoder(torch.cat([target, cond], dim=1))
                mu, lv = self.fc_mu(h), self.fc_logvar(h)
                z = mu + torch.randn_like(lv) * torch.exp(0.5 * lv)
                recon = self.decoder(torch.cat([z, cond], dim=1))
                return recon, mu, lv

            def sample(self, cont, cat):
                cond = self.process_conditions(cont, cat)
                # DETERMINISTIC INFERENCE: Exploit the VAE prior.
                # The expected value of the latent space is exactly zero.
                z = torch.zeros(cont.size(0), self.lat_d, device=cont.device)
                return self.decoder(torch.cat([z, cond], dim=1))

        # ── 10. Train CVAE ──────────────────────────────────────────────
        cont_dim_total = FC_scaled.shape[1] + H_scaled.shape[1]
        cvae = CVAE(NUM_COMMUNITIES, self.latent_dim, cont_dim=cont_dim_total).to(device)
        
        opt = optim.AdamW(cvae.parameters(), lr=self.lr, weight_decay=1e-3)
        scheduler = optim.lr_scheduler.OneCycleLR(
            opt, max_lr=self.lr, steps_per_epoch=max(1, len(train_loader)), epochs=self.epochs
        )

        print(f"Training CVAE for {self.epochs} epochs on {device}...")
        for ep in range(self.epochs):
            cvae.train()
            ep_loss = 0
            
            # KL Annealing over first 50% of epochs
            kl_weight = min(0.05, (ep / max(1, self.epochs * 0.5)) * 0.05)
            
            for target, cont, cat_b in train_loader:
                target = target.to(device)
                cont = cont.to(device)
                cat_b = cat_b.to(device)
                
                # Safety clamp categorical bounds
                cat_b[:, 0].clamp_(0, NUM_COMMUNITIES - 1)
                cat_b[:, 1].clamp_(0, 7)
                cat_b[:, 2].clamp_(0, 12)
                cat_b[:, 3].clamp_(0, 1)

                opt.zero_grad(set_to_none=True)
                recon, mu, lv = cvae(target, cont, cat_b)
                
                mse = nn.functional.mse_loss(recon, target)
                kl = -0.5 * torch.mean(1 + lv - mu.pow(2) - lv.exp())
                loss = mse + kl_weight * kl
                
                loss.backward()
                nn.utils.clip_grad_norm_(cvae.parameters(), 1.0)
                opt.step()
                scheduler.step()
                ep_loss += loss.item()
                
            if (ep + 1) % 10 == 0 or ep == 0:
                avg = ep_loss / max(1, len(train_loader))
                print(f"  Epoch {ep+1:3d}/{self.epochs} | Loss: {avg:.4f} | KL W: {kl_weight:.4f}")

        # ── 11. Inference Initialization ────────────────────────────────
        cvae.eval()

        comm_info = {}
        for i in range(len(comm_ids_arr)):
            cid = str(comm_ids_arr[i])
            if cid not in comm_info:
                comm_info[cid] = int(cats[i, 0])

        comm_ids_list = list(comm_info.keys())
        n_comms = len(comm_ids_list)
        comm_idxs = np.array([comm_info[c] for c in comm_ids_list], dtype=np.int64)

        # Extract pre-calculated baselines for inference
        baseline_rows = historical_baselines.collect()
        baseline_map = {}
        for r in baseline_rows:
            cid = str(r["community_code"])
            dow = int(r["day_of_week"])
            raw_hist = np.array([float(r[f"hist_{i}"] or 0.0) for i in range(96)])
            if cid in comm_scalers:
                log_hist = np.log1p(np.clip(raw_hist, 0, None))
                scaled_hist = comm_scalers[cid].transform(log_hist.reshape(1, -1))[0]
            else:
                scaled_hist = np.log1p(np.clip(raw_hist, 0, None))
            baseline_map[(cid, dow)] = scaled_hist

        # Fallback median if missing a specific day of week
        comm_median_fallback = {}
        for cid in comm_ids_list:
            available = [baseline_map[(cid, d)] for d in range(1, 8) if (cid, d) in baseline_map]
            if available:
                comm_median_fallback[cid] = np.mean(available, axis=0)
            else:
                comm_median_fallback[cid] = np.zeros(96, dtype=np.float32)

        # Forecast conditions mapping for future days
        pred_dates = pd.date_range(predict_start, predict_end, freq='D', inclusive='left')
        pred_fc_rows = fc_daily.filter(
            (F.col("date") >= predict_start) & (F.col("date") < predict_end)
        ).collect()

        fc_by_date = {r["date"]: [float(r[c] or 0) for c in fc_cols] for r in pred_fc_rows}
        train_fc_median = np.nanmedian(fc_raw, axis=0)

        # ── 12. Prediction Loop ──────────────────────────
        print(f"Predicting {len(pred_dates)} days × {n_comms} communities...")
        all_preds = {}
        
        with torch.no_grad():
            for date in pred_dates:
                d = date.date()
                spark_dow = (date.dayofweek + 2) % 7
                if spark_dow == 0: spark_dow = 7
                month = date.month
                is_weekend = 1 if spark_dow in [1, 7] else 0

                # 1. Fetch & scale exogenous forecasts
                fc = np.array(fc_by_date.get(d, train_fc_median), dtype=np.float32).reshape(1, -1)
                fc_scaled = cond_scaler.transform(fc).astype(np.float32)
                
                # 2. Replicate forecasts and get specific Day-of-Week historical baseline
                fc_t = torch.tensor(np.tile(fc_scaled, (n_comms, 1)), dtype=torch.float32).to(device)
                
                current_hists = np.zeros((n_comms, 96), dtype=np.float32)
                for i, cid in enumerate(comm_ids_list):
                    current_hists[i] = baseline_map.get((cid, spark_dow), comm_median_fallback[cid])
                
                hist_t = torch.tensor(current_hists, dtype=torch.float32).to(device)
                cont_t = torch.cat([fc_t, hist_t], dim=1)

                # 3. Categorical Variables
                cat_t = torch.tensor(np.column_stack([
                    comm_idxs,
                    np.full(n_comms, spark_dow, dtype=np.int64),
                    np.full(n_comms, month, dtype=np.int64),
                    np.full(n_comms, is_weekend, dtype=np.int64)
                ])).to(device)
                
                cat_t[:, 0].clamp_(0, NUM_COMMUNITIES - 1)
                cat_t[:, 1].clamp_(0, 7)
                cat_t[:, 2].clamp_(0, 12)
                cat_t[:, 3].clamp_(0, 1)

                # 4. Sample Model Output (DETERMINISTIC)
                batch_preds = cvae.sample(cont_t, cat_t)

                # 5. Inverse scale for final consumption output
                preds_np = batch_preds.cpu().numpy()
                total_profile = np.zeros(96)
                
                for j, cid in enumerate(comm_ids_list):
                    sc = comm_scalers.get(cid)
                    if sc is not None:
                        profile_log = sc.inverse_transform(preds_np[j:j+1])[0]
                        # Allow higher upper bounds for communities, clip only prevents inf
                        profile = np.expm1(np.clip(profile_log, 0, 30))
                    else:
                        profile = np.expm1(np.clip(preds_np[j], 0, 30))
                    total_profile += profile

                # Store summed prediction directly
                all_preds[d] = total_profile

        # ── 13. Build result DataFrame ──────────────────────────────────
        result_rows = []
        for d in sorted(all_preds.keys()):
            profile = all_preds[d]
            base_ts = pd.Timestamp(d)
            for interval in range(96):
                ts = base_ts + pd.Timedelta(minutes=15 * interval)
                result_rows.append((ts, float(profile[interval])))

        result_pdf = pd.DataFrame(result_rows, columns=["datetime_15min", "prediction"])
        print(f"Predictions generated: {len(result_pdf):,} intervals")

        return spark.createDataFrame(result_pdf).select(
            F.col("datetime_15min").cast("timestamp"),
            F.col("prediction").cast("double")
        )

# COMMAND ----------

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from pyspark.sql import functions as F, Window as W
import datetime

class EnergyConsumptionModel:
    """
    Day-ahead energy consumption forecast using a per-community
    Conditional VAE on daily load profiles (96 × 15-min intervals).

    * COMPLIANCE & PERFORMANCE ENHANCEMENTS APPLIED:
    - Bias Correction (No Log1p): Removed log1p transformation to eliminate Jensen's Inequality under-prediction bias. Community aggregates are stable enough for direct StandardScaler.
    - Fast-Adapting Baselines: Reduced recent historical baseline window from 90 to 30 days to prevent seasonal drift in early winter.
    - Strict Day-Ahead Timing: Training data explicitly cuts off at Day D-2.
    - Target MAE Directly: Uses Smooth L1 Loss with beta=0.1.
    - Ensembling: Trains 3 independent models and averages their predictions.
    """

    def __init__(self, epochs=60, latent_dim=16, batch_size=32, lr=1e-3, n_ensembles=3):
        self.epochs = epochs
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.lr = lr
        self.n_ensembles = n_ensembles

    def predict(self, df, predict_start, predict_end):
        import warnings
        warnings.filterwarnings('ignore')

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        spark = df.sparkSession

        # ── 1. Load external day-ahead forecasts ────────────────────────
        demand_fc = spark.table("datathon.shared.demand_forecast").withColumnRenamed("value", "demand_fc")
        pv_fc = spark.table("datathon.shared.pv_production_forecast").withColumnRenamed("value", "pv_fc")
        wind_fc = spark.table("datathon.shared.wind_production_forecast").withColumnRenamed("value", "wind_fc")

        forecasts = (
            demand_fc.join(pv_fc, "datetime_local", "outer")
                     .join(wind_fc, "datetime_local", "outer")
                     .withColumnRenamed("datetime_local", "dt15")
        )

        fc_daily = (
            forecasts
            .withColumn("date", F.to_date("dt15"))
            .withColumn("interval_idx", (F.hour("dt15") * 4 + F.floor(F.minute("dt15") / 15)).cast("int"))
            .groupBy("date")
            .pivot("interval_idx", list(range(96)))
            .agg(
                F.first("demand_fc").alias("d"),
                F.first("pv_fc").alias("p"),
                F.first("wind_fc").alias("w")
            )
        ).fillna(0)
        
        fc_cols = [f"{i}_{f}" for i in range(96) for f in ['d', 'p', 'w']]

        # ── 2. Filter and Aggregate to Community Level ──────────────────
        cutoff_date = (pd.to_datetime(predict_start) - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        
        print(f"Aggregating clients to community level (Excluding incomplete Day D-1: {cutoff_date})...")
        consumption = df.filter(F.to_date("datetime_local") < cutoff_date)

        agg = (
            consumption
            .withColumn("dt15", F.window("datetime_local", "15 minutes").start)
            .groupBy("community_code", "dt15")
            .agg(F.sum("active_kw").alias("total_kw"))
        )

        agg = (agg
            .withColumn("date", F.to_date("dt15"))
            .withColumn("interval_idx", (F.hour("dt15") * 4 + F.floor(F.minute("dt15") / 15)).cast("int"))
            .withColumn("day_of_week", F.dayofweek("date"))
            .withColumn("month", F.month("date"))
            .withColumn("is_weekend", F.when(F.dayofweek("date").isin([1, 7]), 1).otherwise(0))
        )

        # ── 3. Pivot daily profiles ─────────────────────────────────────
        daily_profiles = (
            agg.groupBy("community_code", "date", "day_of_week", "month", "is_weekend")
            .pivot("interval_idx", list(range(96)))
            .agg(F.first("total_kw"))
        ).fillna(0)

        # ── 4. Calculate Baselines (Handling Behavioral Changes) ────────
        global_baselines = (
            daily_profiles
            .groupBy("community_code", "day_of_week")
            .agg(*[F.expr(f"percentile_approx(`{i}`, 0.5)").alias(f"hist_global_{i}") for i in range(96)])
        )

        # FAST-ADAPTING BASELINE: 30 days instead of 90 to prevent seasonal drift
        recent_cutoff = (pd.to_datetime(predict_start) - pd.Timedelta(days=30)).strftime('%Y-%m-%d')
        recent_baselines = (
            daily_profiles.filter(F.col("date") >= recent_cutoff)
            .groupBy("community_code", "day_of_week")
            .agg(*[F.expr(f"percentile_approx(`{i}`, 0.5)").alias(f"hist_recent_{i}") for i in range(96)])
        )

        hist_select_cols = ["community_code", "day_of_week"] + [
            F.coalesce(F.col(f"hist_recent_{i}"), F.col(f"hist_global_{i}")).alias(f"hist_{i}")
            for i in range(96)
        ]
        
        historical_baselines = global_baselines.join(
            recent_baselines, ["community_code", "day_of_week"], "left"
        ).select(*hist_select_cols)

        final_df = daily_profiles.join(historical_baselines, ["community_code", "day_of_week"], "left").fillna(0)
        final_df = final_df.join(fc_daily, "date", "inner")

        # ── 5. Label encoding via dense_rank ────────────────────────────
        final_df = (
            final_df
            .withColumn("community_idx", (F.dense_rank().over(W.orderBy("community_code")) - 1).cast("int"))
        )

        NUM_COMMUNITIES = final_df.agg(F.max("community_idx")).collect()[0][0] + 1
        print(f"Total communities tracked: {NUM_COMMUNITIES}")

        # ── 6. Collect dataset ──────────────────────────────────────────
        profile_cols = [str(i) for i in range(96)]
        hist_cols = [f"hist_{i}" for i in range(96)]
        cat_cols = ["community_idx", "day_of_week", "month", "is_weekend"]
        
        select_cols = ["community_code", "date"] + profile_cols + hist_cols + fc_cols + cat_cols

        print("Collecting training data...")
        rows = final_df.select(*select_cols).collect()
        print(f"Collected {len(rows):,} rows")

        # ── 7. Convert to numpy ─────────────────────────────────────────
        n_p, n_h, n_f = len(profile_cols), len(hist_cols), len(fc_cols)

        comm_ids_arr = np.array([str(r[0]) for r in rows])
        dates_arr = np.array([r[1] for r in rows]) 
        
        idx_p = 2
        idx_h = idx_p + n_p
        idx_f = idx_h + n_h
        idx_c = idx_f + n_f

        profiles_raw = np.array([[float(r[idx_p + j] or 0) for j in range(n_p)] for r in rows], dtype=np.float32)
        hist_raw = np.array([[float(r[idx_h + j] or 0) for j in range(n_h)] for r in rows], dtype=np.float32)
        fc_raw = np.array([[float(r[idx_f + j] or 0) for j in range(n_f)] for r in rows], dtype=np.float32)
        cats = np.array([[int(r[idx_c + j] or 0) for j in range(len(cat_cols))] for r in rows], dtype=np.int64)

        # ── 8. Scaling ──────────────────────────────────────────────────
        comm_scalers = {}
        P_scaled = np.zeros_like(profiles_raw)
        H_scaled = np.zeros_like(hist_raw)

        for cid in np.unique(comm_ids_arr):
            mask = comm_ids_arr == cid
            if mask.sum() == 0: continue
            scaler = StandardScaler()
            
            # NO LOG1P: Direct scaling avoids Jensen's inequality bias for community aggregates
            scaler.fit(profiles_raw[mask])
            comm_scalers[cid] = scaler
            
            P_scaled[mask] = scaler.transform(profiles_raw[mask])
            H_scaled[mask] = scaler.transform(hist_raw[mask])

        fc_scaler = StandardScaler()
        FC_scaled = fc_scaler.fit_transform(fc_raw).astype(np.float32)
        
        C_cont = np.concatenate([FC_scaled, H_scaled], axis=1).astype(np.float32)

        # ── 9. DataLoader ───────────────────────────────────────────────
        train_ds = TensorDataset(torch.tensor(P_scaled), torch.tensor(C_cont), torch.tensor(cats))
        drop_last = (len(train_ds) % self.batch_size == 1)
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, drop_last=drop_last)

        # ── 10. Define Expanded CVAE ────────────────────────────────────
        class CVAE(nn.Module):
            def __init__(self, num_communities, lat_d=16, cont_dim=384):
                super().__init__()
                self.comm_emb = nn.Embedding(num_communities, 16)
                self.dow_emb = nn.Embedding(8, 3)
                self.month_emb = nn.Embedding(13, 3)
                self.wend_emb = nn.Embedding(2, 2)
                
                self.cat_dim = 16 + 3 + 3 + 2
                self.cond_dim = cont_dim + self.cat_dim
                
                self.encoder = nn.Sequential(
                    nn.Linear(96 + self.cond_dim, 512),
                    nn.BatchNorm1d(512), nn.LeakyReLU(0.2), nn.Dropout(0.2),
                    nn.Linear(512, 256),
                    nn.BatchNorm1d(256), nn.LeakyReLU(0.2), nn.Dropout(0.2),
                    nn.Linear(256, 128),
                    nn.BatchNorm1d(128), nn.LeakyReLU(0.2)
                )
                self.fc_mu = nn.Linear(128, lat_d)
                self.fc_logvar = nn.Linear(128, lat_d)
                
                self.decoder = nn.Sequential(
                    nn.Linear(lat_d + self.cond_dim, 256),
                    nn.BatchNorm1d(256), nn.LeakyReLU(0.2), nn.Dropout(0.2),
                    nn.Linear(256, 512),
                    nn.BatchNorm1d(512), nn.LeakyReLU(0.2), nn.Dropout(0.2),
                    nn.Linear(512, 96)
                )
                self.lat_d = lat_d

            def process_conditions(self, cont, cat):
                co = self.comm_emb(cat[:, 0])
                dw = self.dow_emb(cat[:, 1])
                mo = self.month_emb(cat[:, 2])
                we = self.wend_emb(cat[:, 3])
                return torch.cat([cont, co, dw, mo, we], dim=1)

            def forward(self, target, cont, cat):
                cond = self.process_conditions(cont, cat)
                h = self.encoder(torch.cat([target, cond], dim=1))
                mu, lv = self.fc_mu(h), self.fc_logvar(h)
                z = mu + torch.randn_like(lv) * torch.exp(0.5 * lv)
                recon = self.decoder(torch.cat([z, cond], dim=1))
                return recon, mu, lv

            def sample(self, cont, cat):
                cond = self.process_conditions(cont, cat)
                z = torch.zeros(cont.size(0), self.lat_d, device=cont.device)
                return self.decoder(torch.cat([z, cond], dim=1))

        # ── 11. Inference Initialization ────────────────────────────────
        comm_info = {}
        for i in range(len(comm_ids_arr)):
            cid = str(comm_ids_arr[i])
            if cid not in comm_info:
                comm_info[cid] = int(cats[i, 0])

        comm_ids_list = list(comm_info.keys())
        n_comms = len(comm_ids_list)
        comm_idxs = np.array([comm_info[c] for c in comm_ids_list], dtype=np.int64)

        baseline_rows = historical_baselines.collect()
        baseline_map = {}
        for r in baseline_rows:
            cid = str(r["community_code"])
            dow = int(r["day_of_week"])
            raw_hist = np.array([float(r[f"hist_{i}"] or 0.0) for i in range(96)])
            if cid in comm_scalers:
                scaled_hist = comm_scalers[cid].transform(raw_hist.reshape(1, -1))[0]
            else:
                scaled_hist = raw_hist
            baseline_map[(cid, dow)] = scaled_hist

        comm_median_fallback = {}
        for cid in comm_ids_list:
            available = [baseline_map[(cid, d)] for d in range(1, 8) if (cid, d) in baseline_map]
            if available:
                comm_median_fallback[cid] = np.mean(available, axis=0)
            else:
                comm_median_fallback[cid] = np.zeros(96, dtype=np.float32)

        pred_dates = pd.date_range(predict_start, predict_end, freq='D', inclusive='left')
        pred_fc_rows = fc_daily.filter(
            (F.col("date") >= predict_start) & (F.col("date") < predict_end)
        ).collect()

        fc_by_date = {r["date"]: [float(r[c] or 0) for c in fc_cols] for r in pred_fc_rows}
        train_fc_median = np.nanmedian(fc_raw, axis=0)

        # ── 12. Ensemble Training & Prediction ──────────────────────────
        print(f"\n--- Starting Ensemble Training ({self.n_ensembles} models) ---")
        all_ensemble_preds = []

        for ens_idx in range(self.n_ensembles):
            print(f"\n--- Training Model {ens_idx + 1}/{self.n_ensembles} ---")
            cvae = CVAE(NUM_COMMUNITIES, self.latent_dim, cont_dim=C_cont.shape[1]).to(device)
            
            opt = optim.AdamW(cvae.parameters(), lr=self.lr, weight_decay=1e-3)
            scheduler = optim.lr_scheduler.OneCycleLR(
                opt, max_lr=self.lr, steps_per_epoch=max(1, len(train_loader)), epochs=self.epochs
            )

            for ep in range(self.epochs):
                cvae.train()
                ep_loss = 0
                
                # KL Annealing over first 50% of epochs
                kl_weight = min(0.05, (ep / max(1, self.epochs * 0.5)) * 0.05)
                
                for target, cont, cat_b in train_loader:
                    target = target.to(device)
                    cont = cont.to(device)
                    cat_b = cat_b.to(device)
                    
                    cat_b[:, 0].clamp_(0, NUM_COMMUNITIES - 1)
                    cat_b[:, 1].clamp_(0, 7)
                    cat_b[:, 2].clamp_(0, 12)
                    cat_b[:, 3].clamp_(0, 1)

                    opt.zero_grad(set_to_none=True)
                    recon, mu, lv = cvae(target, cont, cat_b)
                    
                    loss_recon = nn.functional.smooth_l1_loss(recon, target, beta=0.1)
                    kl = -0.5 * torch.mean(1 + lv - mu.pow(2) - lv.exp())
                    loss = loss_recon + kl_weight * kl
                    
                    loss.backward()
                    nn.utils.clip_grad_norm_(cvae.parameters(), 1.0)
                    opt.step()
                    scheduler.step()
                    ep_loss += loss.item()
                    
                if (ep + 1) % 20 == 0 or ep == 0:
                    avg = ep_loss / max(1, len(train_loader))
                    print(f"  Epoch {ep+1:3d}/{self.epochs} | Loss (SmoothL1): {avg:.4f} | KL W: {kl_weight:.4f}")

            # Predict using the newly trained ensemble model
            cvae.eval()
            model_preds = {}
            
            with torch.no_grad():
                for date in pred_dates:
                    d = date.date()
                    spark_dow = (date.dayofweek + 2) % 7
                    if spark_dow == 0: spark_dow = 7
                    month = date.month
                    is_weekend = 1 if spark_dow in [1, 7] else 0

                    fc = np.array(fc_by_date.get(d, train_fc_median), dtype=np.float32).reshape(1, -1)
                    fc_scaled = fc_scaler.transform(fc).astype(np.float32)
                    fc_tiled = np.tile(fc_scaled, (n_comms, 1))
                    
                    current_hists = np.zeros((n_comms, 96), dtype=np.float32)
                    for i, cid in enumerate(comm_ids_list):
                        current_hists[i] = baseline_map.get((cid, spark_dow), comm_median_fallback[cid])

                    cond_raw_inf = np.concatenate([fc_tiled, current_hists], axis=1).astype(np.float32)
                    cont_t = torch.tensor(cond_raw_inf, dtype=torch.float32).to(device)

                    cat_t = torch.tensor(np.column_stack([
                        comm_idxs,
                        np.full(n_comms, spark_dow, dtype=np.int64),
                        np.full(n_comms, month, dtype=np.int64),
                        np.full(n_comms, is_weekend, dtype=np.int64)
                    ])).to(device)
                    
                    cat_t[:, 0].clamp_(0, NUM_COMMUNITIES - 1)
                    cat_t[:, 1].clamp_(0, 7)
                    cat_t[:, 2].clamp_(0, 12)
                    cat_t[:, 3].clamp_(0, 1)

                    batch_preds = cvae.sample(cont_t, cat_t)

                    preds_np = batch_preds.cpu().numpy()
                    total_profile = np.zeros(96)
                    
                    for j, cid in enumerate(comm_ids_list):
                        sc = comm_scalers.get(cid)
                        if sc is not None:
                            # Direct inverse transform, no expm1 required
                            profile_raw = sc.inverse_transform(preds_np[j:j+1])[0]
                            profile = np.clip(profile_raw, 0, None)
                        else:
                            profile = np.clip(preds_np[j], 0, None)
                            
                        total_profile += profile

                    model_preds[d] = total_profile
            
            all_ensemble_preds.append(model_preds)

        # ── 13. Average Ensemble Predictions ──────────────────────────
        all_preds = {}
        for d in all_ensemble_preds[0].keys():
            all_preds[d] = np.mean([preds[d] for preds in all_ensemble_preds], axis=0)

        # ── 14. Build result DataFrame ──────────────────────────────────
        result_rows = []
        for d in sorted(all_preds.keys()):
            profile = all_preds[d]
            base_ts = pd.Timestamp(d)
            for interval in range(96):
                ts = base_ts + pd.Timedelta(minutes=15 * interval)
                result_rows.append((ts, float(profile[interval])))

        result_pdf = pd.DataFrame(result_rows, columns=["datetime_15min", "prediction"])
        print(f"\nPredictions generated: {len(result_pdf):,} intervals")

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

# DBTITLE 1,Prediction Diagnostics
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

# DBTITLE 1,Overfitting Diagnostic
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