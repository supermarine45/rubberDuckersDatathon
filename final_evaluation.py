# Databricks notebook source
# MAGIC %md
# MAGIC # Energy Consumption Prediction: Submission
# MAGIC
# MAGIC ## Objective
# MAGIC Predict the **total (aggregated) energy consumption** across all clients for each
# MAGIC **15-minute interval** in **2026**.
# MAGIC
# MAGIC ## How It Works
# MAGIC 1. Add as many cells as you need **above** the Submit cell: install packages, import
# MAGIC    libraries, load additional data, do pre-computation, and define your `EnergyConsumptionModel` class.
# MAGIC 2. Run the **Submit** cell (the last cell). This triggers a scoring job that:
# MAGIC    - Re-runs **this entire notebook** with access to the full dataset (2025 + 2026)
# MAGIC    - Your `%pip install` commands, imports, and model class all run exactly as if you ran them yourself
# MAGIC    - Calls `model.predict()` to generate predictions for 2026
# MAGIC    - Computes your MAE and records it on the leaderboard
# MAGIC 3. Your **MAE score** and remaining submissions are printed once the job finishes.
# MAGIC
# MAGIC ## Model Contract
# MAGIC `predict(self, df, predict_start, predict_end)` receives a **PySpark DataFrame** with
# MAGIC **all** data (2025 + 2026):
# MAGIC - Columns: `client_id` (int), `datetime_local` (timestamp), `community_code` (string), `active_kw` (double)
# MAGIC - `predict_start` / `predict_end` define the prediction window
# MAGIC - `spark` is available as a global (you can use `spark.table()`, `spark.createDataFrame()`, etc.)
# MAGIC
# MAGIC It must return a **PySpark DataFrame** with exactly two columns:
# MAGIC - `datetime_15min` (timestamp): the 15-minute interval (floor of the timestamp)
# MAGIC - `prediction` (double): the **total** predicted `active_kw` across all clients for that interval
# MAGIC - One row per 15-minute interval in the prediction window
# MAGIC
# MAGIC ## Rules
# MAGIC - **Limited submissions** per team. Use the **exploration notebook** for local validation before submitting.
# MAGIC - Only **successful** submissions count towards the limit.
# MAGIC - **Do not modify the Submit cell** (the last cell). Everything else is yours to change.
# MAGIC
# MAGIC ## Performance Tips
# MAGIC - Use **PySpark** for all heavy data processing. Avoid `.toPandas()` on the full dataset.
# MAGIC - If using ML libraries (LightGBM, sklearn, etc.), do feature engineering in PySpark first, then
# MAGIC   `.toPandas()` only the **compact feature matrix** (e.g. aggregated 15-min intervals).
# MAGIC - Scoring has a **timeout (45 min)**. Keep the PySpark→pandas conversion for the very last step.
# MAGIC
# MAGIC ## Evaluation
# MAGIC **MAE (Mean Absolute Error)** on the aggregated 15-minute totals. Lower is better.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install Packages (optional)
# MAGIC
# MAGIC Add `%pip install` cells here if your model needs additional packages.
# MAGIC These will also be installed during scoring.

# COMMAND ----------

# Fix for the typing_extensions
%pip install --force-reinstall typing_extensions>=4.10 torch 
%pip install --upgrade typing_extensions>=4.5.0

# Restart Python to use the updated packages
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

# >>> Set your team name <<<
TEAM_NAME = "rubber_duckers"  # lowercase, no spaces, use underscores

# COMMAND ----------

# MAGIC %md
# MAGIC ## Imports

# COMMAND ----------

from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
import pandas as pd
from pyspark.sql import functions as F

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define Your Model
# MAGIC
# MAGIC The baseline below predicts each 15-minute interval using the value from
# MAGIC **7 days ago**, with the historical mean as a fallback. It gives you an initial score to beat.
# MAGIC
# MAGIC Feel free to add as many cells as you need above this one for pre-computation.
# MAGIC Everything above the Submit cell will be executed during scoring.

# COMMAND ----------

df = spark.table("datathon.shared.client_consumption")
display(df)

# COMMAND ----------

class EnergyConsumptionModel:
    """
    Day-ahead energy consumption forecast using Gradient Boosting.
    Core: 8 exogenous + demand_forecast + demand lags + rolling mean.
    Forecasts resampled to 15-min to maximise training rows.
    """

    def __init__(self, n_estimators=1000, max_depth=5, learning_rate=0.05,
                 subsample=0.8, min_samples_leaf=10):
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
# MAGIC ## Submit for Scoring
# MAGIC
# MAGIC **⚠️ DO NOT CHANGE THIS CELL ⚠️**
# MAGIC
# MAGIC When you run this cell interactively, it triggers the scoring job.
# MAGIC When the scoring job re-runs this notebook, this cell generates predictions
# MAGIC and writes them for evaluation.

# COMMAND ----------

# ============================================================
# ⚠️  DO NOT CHANGE THIS CELL — submission will break  ⚠️
# ============================================================

# Provided by the organizers. Do not change.
SCORING_JOB_ID = 659971041084731  # Set automatically during setup

# --- Internal mode detection (set by the scoring job) ---
dbutils.widgets.text("mode", "interactive")
_MODE = dbutils.widgets.get("mode").strip()

if _MODE == "score":
    # ---- Score mode: generate predictions and exit ----
    from pyspark.sql import functions as _F

    _predict_start = dbutils.widgets.get("predict_start").strip()
    _predict_end = dbutils.widgets.get("predict_end").strip()

    _full_df = spark.table("datathon.shared.client_consumption")
    _model = EnergyConsumptionModel()
    _predictions = _model.predict(_full_df, _predict_start, _predict_end)

    _predictions_table = "datathon.evaluation.submissions"
    (
        _predictions
        .withColumn("team_name", _F.lit(TEAM_NAME))
        .withColumn("submitted_at", _F.current_timestamp())
        .select("team_name", "datetime_15min", "prediction", "submitted_at")
        .write.mode("overwrite").saveAsTable(_predictions_table)
    )
    print(f"Wrote {_predictions.count():,} predictions to {_predictions_table}")
    dbutils.notebook.exit("ok")

# ---- Interactive mode: trigger the scoring job ----
import json
import datetime as dt
from databricks.sdk import WorkspaceClient

assert SCORING_JOB_ID is not None, "SCORING_JOB_ID has not been set. Ask the organisers."
assert TEAM_NAME != "my_team", "Please set your TEAM_NAME in the configuration cell before submitting."

_w = WorkspaceClient()
_submitter_email = _w.current_user.me().user_name
_notebook_path = (
    dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
)

print(f"Submitting as: {_submitter_email}")
print(f"Notebook: {_notebook_path}")

_job_run = _w.jobs.run_now(
    job_id=SCORING_JOB_ID,
    notebook_params={
        "team_name": TEAM_NAME,
        "submitter_email": _submitter_email,
        "notebook_path": _notebook_path,
    },
)
print("Job triggered. Waiting for scoring to finish (this may take a few minutes) ...")

try:
    _job_run = _job_run.result(timeout=dt.timedelta(minutes=50))
    _tasks = _w.jobs.get_run(_job_run.run_id).tasks
    _task_run_id = _tasks[0].run_id
    _output = _w.jobs.get_run_output(_task_run_id)
    _result = json.loads(_output.notebook_output.result)
except Exception as e:
    print(f"\nScoring job failed: {e}")
    _result = None

if _result and _result["status"] == "success":
    print(f"\n{'='*50}")
    print(f"  Team: {_result['team_name']}")
    print(f"  MAE:  {_result['mae']:.6f}")
    print(f"  Submissions remaining: {_result['submissions_remaining']}")
    print(f"{'='*50}")
elif _result:
    print(f"\nSubmission FAILED: {_result['message']}")
    if "submissions_remaining" in _result:
        print(f"Submissions remaining: {_result['submissions_remaining']}")