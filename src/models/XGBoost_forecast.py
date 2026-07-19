"""
XGBoost Forecasting Pipeline — 365-Day Horizon
Source: Appendix B — "Full XGBoost + Forecasting Pipeline"
Project: Forecasting the Air Quality Index using Statistical, Machine
         Learning and Deep Learning Models — A Comparative Study

Uses lag features (up to MAX_LAG days), a rolling mean, and cyclic
day-of-year / day-of-week features to iteratively forecast PM2.5 forward
one day at a time, feeding each new prediction back in as the next lag.
"""

from pathlib import Path
from datetime import timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
DATA_PATH = "data/processed/DL017_Imp.csv"
DATE_COL = "From.Date"
TARGET = "PM2.5_imputed"
FORECAST_DAYS = 365
MAX_LAG = 60           # lag window in days (try 30 for a shorter-memory model)
ROLL_WINDOW = 7         # rolling mean window in days
RANDOM_STATE = 42
OUT_DIR = Path("results/forecasts")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# 1) Load and parse dates robustly
# --------------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)
df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce", infer_datetime_format=True)
df = df.dropna(subset=[DATE_COL, TARGET]).sort_values(DATE_COL).reset_index(drop=True)


# --------------------------------------------------------------------------
# 2) Build lag features + rolling mean + seasonality
# --------------------------------------------------------------------------
ts = df[[DATE_COL, TARGET]].copy()

# lags
for lag in range(1, MAX_LAG + 1):
    ts[f"lag_{lag}"] = ts[TARGET].shift(lag)

# rolling mean of past ROLL_WINDOW days
ts[f"roll_mean_{ROLL_WINDOW}"] = ts[TARGET].rolling(
    window=ROLL_WINDOW, min_periods=1
).mean().shift(1)

# day-of-week and day-of-year (cyclic)
ts["doy"] = ts[DATE_COL].dt.dayofyear
ts["dow"] = ts[DATE_COL].dt.dayofweek

# cyclic transforms
ts["doy_sin"] = np.sin(2 * np.pi * ts["doy"] / 365.25)
ts["doy_cos"] = np.cos(2 * np.pi * ts["doy"] / 365.25)
ts["dow_sin"] = np.sin(2 * np.pi * ts["dow"] / 7.0)
ts["dow_cos"] = np.cos(2 * np.pi * ts["dow"] / 7.0)

# drop initial rows with NaN lags
ts_sup = ts.dropna().reset_index(drop=True)
print("Supervised rows:", len(ts_sup))


# --------------------------------------------------------------------------
# 3) Create X / y
# --------------------------------------------------------------------------
lag_cols = [f"lag_{l}" for l in range(1, MAX_LAG + 1)]
extra_cols = [f"roll_mean_{ROLL_WINDOW}", "doy_sin", "doy_cos", "dow_sin", "dow_cos"]
feature_cols = lag_cols + extra_cols

X = ts_sup[feature_cols].values
y = ts_sup[TARGET].values

# time-aware split (chronological — no shuffling)
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]


# --------------------------------------------------------------------------
# 4) Train model
# --------------------------------------------------------------------------
model = xgb.XGBRegressor(
    n_estimators=500, max_depth=6,
    learning_rate=0.05, subsample=0.8,
    colsample_bytree=0.8,
    random_state=RANDOM_STATE, n_jobs=4,
    objective="reg:squarederror"
)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=50)


# --------------------------------------------------------------------------
# 5) Evaluation
# --------------------------------------------------------------------------
def eval_print(y_true, y_pred, name):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{name}: samples={len(y_true)} RMSE={rmse:.3f} MAE={mae:.3f} R2={r2:.4f}")
    return rmse, mae, r2


y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
train_metrics = eval_print(y_train, y_train_pred, "Train")
test_metrics = eval_print(y_test, y_test_pred, "Test")

# residual standard deviation
resid = y_test - y_test_pred
resid_std = np.std(resid)
print("Test residual std:", resid_std)


# --------------------------------------------------------------------------
# 6) Iterative forecasting (walk the model forward day by day)
# --------------------------------------------------------------------------
last_rows = ts.tail(MAX_LAG).copy().reset_index(drop=True)
rolling_vals = list(last_rows[TARGET].values)
last_date = ts[DATE_COL].iloc[-1]

future_dates = []
future_preds = []


def build_features_from_window(rolling_vals, for_date):
    feat_lags = [rolling_vals[-i] for i in range(1, MAX_LAG + 1)]
    roll_mean = np.mean(rolling_vals[-ROLL_WINDOW:])
    doy = for_date.timetuple().tm_yday
    dow = for_date.weekday()
    doy_sin = np.sin(2 * np.pi * doy / 365.25)
    doy_cos = np.cos(2 * np.pi * doy / 365.25)
    dow_sin = np.sin(2 * np.pi * dow / 7.0)
    dow_cos = np.cos(2 * np.pi * dow / 7.0)
    feat = feat_lags + [roll_mean, doy_sin, doy_cos, dow_sin, dow_cos]
    return np.array(feat).reshape(1, -1)


for i in range(FORECAST_DAYS):
    next_date = last_date + timedelta(days=i + 1)
    Xf = build_features_from_window(rolling_vals, next_date)
    pred = model.predict(Xf)[0]
    future_dates.append(next_date)
    future_preds.append(pred)
    rolling_vals.append(pred)


# --------------------------------------------------------------------------
# 7) Save results
# --------------------------------------------------------------------------
forecast_df = pd.DataFrame({
    DATE_COL: future_dates,
    f"{TARGET}_forecast": future_preds
})
out_path = OUT_DIR / f"PM25_forecast_{FORECAST_DAYS}d_lag{MAX_LAG}_roll{ROLL_WINDOW}.csv"
forecast_df.to_csv(out_path, index=False)
print("Saved forecast to:", out_path)


# --------------------------------------------------------------------------
# 8) Plot forecast
# --------------------------------------------------------------------------
plt.figure(figsize=(14, 5))
plt.plot(ts[DATE_COL], ts[TARGET], label="Historical", linewidth=1.0)
plt.plot(forecast_df[DATE_COL], forecast_df[f"{TARGET}_forecast"],
          label="Forecast", linewidth=1.2, color="darkorange")
plt.legend()
plt.title(f"{TARGET} — {FORECAST_DAYS}-Day Forward Forecast")
plt.xlabel("Date")
plt.ylabel(f"{TARGET} (\u00b5g/m\u00b3)")
plt.tight_layout()
plt.show()

