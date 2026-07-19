"""
XGBoost Regression Pipeline — Train / Test Evaluation
Source: Appendix B — XGBoost Code
Project: Forecasting the Air Quality Index using Statistical, Machine
         Learning and Deep Learning Models — A Comparative Study

Trains an XGBoost regressor on the full engineered feature set (with a
preprocessing pipeline for numeric + categorical columns) and reports
train/test RMSE, MAE, and R^2. For the iterative 365-day lag-based
forecast used to generate forward predictions, see xgboost_forecast.py.
"""

import warnings
import inspect

import numpy as np
import pandas as pd
import sklearn
import joblib
import xgboost as xgb
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print("scikit-learn version:", sklearn.__version__)
print("xgboost version:", xgb.__version__)


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
FILE_PATH = "data/processed/DL017_Imp.csv"
TARGET_COL = "PM2.5_imputed"
RANDOM_STATE = 42

DENSE_OUTPUT = True
EARLY_STOPPING = 30
NUM_BOOST_ROUND = 1000
N_JOBS = 4


# --------------------------------------------------------------------------
# Load Data
# --------------------------------------------------------------------------
df = pd.read_csv(FILE_PATH)
print("Loaded:", FILE_PATH)
print("Shape before cleanup:", df.shape)


# --------------------------------------------------------------------------
# Drop all-null columns
# --------------------------------------------------------------------------
all_na_cols = df.columns[df.isna().all()].tolist()
if len(all_na_cols):
    print("Dropping columns with all-missing values:", all_na_cols)
    df = df.drop(columns=all_na_cols)

print("Shape after dropping all-NaN cols:", df.shape)


# --------------------------------------------------------------------------
# Ensure Target Exists
# --------------------------------------------------------------------------
if TARGET_COL not in df.columns:
    TARGET_COL = df.columns[-1]
    print("Target not found; using last column:", TARGET_COL)
else:
    print("Using target:", TARGET_COL)

X = df.drop(columns=[TARGET_COL]).copy()
y = df[TARGET_COL].copy()


# --------------------------------------------------------------------------
# Remove Invalid Target Values
# --------------------------------------------------------------------------
valid_mask = y.notna() & np.isfinite(y)
n_before = len(y)
n_invalid = (~valid_mask).sum()

if n_invalid > 0:
    print(f"Dropping {n_invalid} rows with invalid target (NaN/Inf).")
    X = X.loc[valid_mask].reset_index(drop=True)
    y = y.loc[valid_mask].reset_index(drop=True)
else:
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

print(f"Rows before: {n_before}, after removing invalid: {len(y)}")


# --------------------------------------------------------------------------
# Convert Bool Columns
# --------------------------------------------------------------------------
bool_cols = X.select_dtypes(include=["bool"]).columns.tolist()
if len(bool_cols):
    print("Converting bool to int:", bool_cols)
    X[bool_cols] = X[bool_cols].astype(int)


# --------------------------------------------------------------------------
# Convert Date Columns
# --------------------------------------------------------------------------
date_cols = [c for c in X.columns if "date" in c.lower() or "Date" in c]
for c in date_cols:
    try:
        X[c] = pd.to_datetime(X[c], errors="coerce")
        X[f"{c}_year"] = X[c].dt.year
        X[f"{c}_month"] = X[c].dt.month
        X[f"{c}_day"] = X[c].dt.day
        X[f"{c}_dow"] = X[c].dt.dayofweek
    except Exception:
        pass


# --------------------------------------------------------------------------
# Feature Types
# --------------------------------------------------------------------------
numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

print(f"Numeric: {len(numeric_cols)}, Categorical: {len(categorical_cols)}")


# --------------------------------------------------------------------------
# OneHotEncoder Wrapper (handles sklearn version differences)
# --------------------------------------------------------------------------
def make_onehot(handle_unknown="ignore", dense_output=True):
    try:
        return OneHotEncoder(handle_unknown=handle_unknown,
                              sparse_output=(not dense_output))
    except TypeError:
        try:
            return OneHotEncoder(handle_unknown=handle_unknown,
                                  sparse=(not dense_output))
        except TypeError:
            warnings.warn("Default OneHotEncoder used.")
            return OneHotEncoder(handle_unknown=handle_unknown)


ohe = make_onehot(dense_output=DENSE_OUTPUT)


# --------------------------------------------------------------------------
# Pipelines
# --------------------------------------------------------------------------
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", ohe)
])


# --------------------------------------------------------------------------
# ColumnTransformer
# --------------------------------------------------------------------------
transformers = []
if len(numeric_cols) > 0:
    transformers.append(("num", numeric_transformer, numeric_cols))
if len(categorical_cols) > 0:
    transformers.append(("cat", categorical_transformer, categorical_cols))

preprocessor = ColumnTransformer(transformers=transformers)


# --------------------------------------------------------------------------
# Train-Test Split
# --------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE
)

print("Train:", X_train.shape, " Test:", X_test.shape)


# --------------------------------------------------------------------------
# Fit Preprocessor
# --------------------------------------------------------------------------
preprocessor.fit(X_train)

X_train_trans = preprocessor.transform(X_train)
X_test_trans = preprocessor.transform(X_test)


# --------------------------------------------------------------------------
# Prepare XGBoost
# --------------------------------------------------------------------------
xgb_params = dict(
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=RANDOM_STATE,
    n_jobs=N_JOBS,
    objective="reg:squarederror"
)

reg = xgb.XGBRegressor(n_estimators=NUM_BOOST_ROUND, **xgb_params)


# --------------------------------------------------------------------------
# Train with Early Stopping
# --------------------------------------------------------------------------
sig = inspect.signature(reg.fit)
supports_early = "early_stopping_rounds" in sig.parameters
print("Supports early stopping?", supports_early)

if supports_early:
    reg.fit(
        X_train_trans, y_train, eval_set=[(X_test_trans, y_test)],
        early_stopping_rounds=EARLY_STOPPING, verbose=20
    )
else:
    print("Training using xgb.train fallback...")
    dtrain = xgb.DMatrix(X_train_trans, label=y_train)
    dval = xgb.DMatrix(X_test_trans, label=y_test)

    watchlist = [(dtrain, "train"), (dval, "eval")]

    bst = xgb.train(
        params={k: v for k, v in xgb_params.items() if k not in ["random_state", "n_jobs"]},
        dtrain=dtrain, num_boost_round=NUM_BOOST_ROUND, evals=watchlist,
        early_stopping_rounds=EARLY_STOPPING, verbose_eval=20
    )

    reg._Booster = bst
    reg._le = None


# --------------------------------------------------------------------------
# Predictions
# --------------------------------------------------------------------------
def safe_predict(model, X_input):
    try:
        return model.predict(X_input)
    except Exception:
        return model.predict(X_input.toarray())


y_train_pred = safe_predict(reg, X_train_trans)
y_test_pred = safe_predict(reg, X_test_trans)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def evaluate_metrics(y_true, y_pred, label="Set"):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"\n{label} Results:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  R2:   {r2:.4f}")
    return rmse, mae, r2


train_metrics = evaluate_metrics(y_train, y_train_pred, "Train")
test_metrics = evaluate_metrics(y_test, y_test_pred, "Test")


# --------------------------------------------------------------------------
# Save trained model + preprocessor (optional)
# --------------------------------------------------------------------------
# joblib.dump(reg, "results/xgboost_model.joblib")
# joblib.dump(preprocessor, "results/xgboost_preprocessor.joblib")

