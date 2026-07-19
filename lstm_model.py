"""
LSTM Forecasting Pipeline
Source: Appendix C — LSTM Code
Project: Forecasting the Air Quality Index using Statistical, Machine
         Learning and Deep Learning Models — A Comparative Study

Builds a sequence-to-one LSTM model (14-day lookback window) to forecast
PM2.5 / PM10, with chronological train/val/test evaluation and diagnostic
plots (actual vs predicted, residuals, error distribution, pseudo-ROC).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
PATH = "data/processed/DL017_Imp.csv"
DATE_COL = "From.Date"
VALID_TARGETS = ["PM2.5_imputed", "PM10_imputed"]

TARGET_COL = "PM2.5_imputed"   # choose one target
RUN_ALL_TARGETS = True

SEQ_LEN = 14
LAGS = [1, 2, 3, 7, 14]
TEST_RATIO = 0.15
VAL_RATIO = 0.10

# LSTM hyperparameters
LSTM_UNITS = 64
LSTM_DROPOUT = 0.2
BATCH_SIZE = 32
EPOCHS = 120


def evaluate(y_true, y_pred):
    y_true = np.array(y_true).ravel()
    y_pred = np.array(y_pred).ravel()
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"RMSE": rmse, "MAE": mae, "R2": r2}


# Load data
df_full = pd.read_csv(PATH)
if "Unnamed: 0" in df_full.columns:
    df_full = df_full.drop(columns=["Unnamed: 0"])
print("Raw shape:", df_full.shape)

targets_to_run = VALID_TARGETS if RUN_ALL_TARGETS else [TARGET_COL]
for t in targets_to_run:
    if t not in df_full.columns:
        raise KeyError(f"Target column '{t}' not found. Available: {list(df_full.columns)}")


def run_lstm_only(df, target_col):
    print("\n" + "=" * 60)
    print("Running LSTM pipeline for target:", target_col)
    df = df.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    df = df[~df[DATE_COL].isna()].copy()
    df = df[~df[target_col].isna()].copy()

    # calendar features
    df["dayofweek"] = df[DATE_COL].dt.dayofweek
    df["month"] = df[DATE_COL].dt.month

    # lag & rolling features
    for l in LAGS:
        df[f"{target_col}_lag{l}"] = df[target_col].shift(l)
    df[f"{target_col}_roll7_mean"] = df[target_col].rolling(
        window=7, min_periods=1
    ).mean().shift(1)
    df[f"{target_col}_roll14_mean"] = df[target_col].rolling(
        window=14, min_periods=1
    ).mean().shift(1)

    # optional extra features (co-pollutants, if present in this station's file)
    candidate_extra = ["NO", "NO2", "NOx", "NH3", "SO2", "CO", "Ozone",
                        "Benzene", "Toluene", "PM2.5", "PM10"]

    extra_features = [c for c in candidate_extra if c in df.columns and c != target_col]

    extra_rolls = [c for c in df.columns
                   if (target_col.split("_")[0] in c and "rollmean" in c and "imput" in c)]

    lstm_features = ([f"{target_col}_lag{l}" for l in LAGS] +
                      ["dayofweek", "month"] + extra_features + extra_rolls)
    lstm_features = [c for c in lstm_features if c in df.columns]
    print("LSTM feature count:", len(lstm_features))

    df_prep = df.dropna(subset=lstm_features + [target_col]).reset_index(drop=True)

    print("Prepared rows after dropna:", df_prep.shape[0])
    if df_prep.shape[0] < 200:
        print("Warning: small prepared rows. Consider reducing seq_len or checking missingness.")

    # sequence maker
    def make_sequences(df_local, feature_cols, target_col, seq_len=14):
        arr = df_local[feature_cols + [target_col]].values
        Xs, ys = [], []
        for i in range(seq_len, len(arr)):
            Xs.append(arr[i - seq_len:i, :len(feature_cols)])
            ys.append(arr[i, len(feature_cols)])
        return np.array(Xs), np.array(ys)

    df_prep = df_prep.sort_values(DATE_COL).reset_index(drop=True)
    X_all, y_all = make_sequences(df_prep, lstm_features, target_col, seq_len=SEQ_LEN)

    print("X_all shape:", X_all.shape, "y_all shape:", y_all.shape)
    if X_all.shape[0] == 0:
        raise ValueError("No sequences created — reduce seq_len or check data.")

    # sequential split (chronological — no shuffling)
    n = len(X_all)
    test_n = int(n * TEST_RATIO)
    val_n = int(n * VAL_RATIO)
    train_end = n - test_n - val_n
    val_end = n - test_n

    if train_end <= 0:
        raise ValueError("Train set size <= 0. Adjust test_ratio/val_ratio or ensure more data.")

    X_seq_train, X_seq_val, X_seq_test = X_all[:train_end], X_all[train_end:val_end], X_all[val_end:]
    y_seq_train, y_seq_val, y_seq_test = y_all[:train_end], y_all[train_end:val_end], y_all[val_end:]

    print("Train/Val/Test sizes:", len(X_seq_train), len(X_seq_val), len(X_seq_test))

    # scaling
    scaler_X = StandardScaler().fit(X_seq_train.reshape(-1, X_seq_train.shape[2]))
    scaler_y = StandardScaler().fit(y_seq_train.reshape(-1, 1))

    def scale_sequences(X_seq, scaler):
        n_samples, seq_l, n_feats = X_seq.shape
        X_reshaped = X_seq.reshape(-1, n_feats)
        X_scaled = scaler.transform(X_reshaped)
        return X_scaled.reshape(n_samples, seq_l, n_feats)

    X_seq_train_s = scale_sequences(X_seq_train, scaler_X)
    X_seq_val_s = scale_sequences(X_seq_val, scaler_X)
    X_seq_test_s = scale_sequences(X_seq_test, scaler_X)

    y_seq_train_s = scaler_y.transform(y_seq_train.reshape(-1, 1)).flatten()
    y_seq_val_s = scaler_y.transform(y_seq_val.reshape(-1, 1)).flatten()
    y_seq_test_s = scaler_y.transform(y_seq_test.reshape(-1, 1)).flatten()

    # build LSTM
    def build_lstm(seq_len, n_features, units=64, dropout=0.2):
        inp = layers.Input(shape=(seq_len, n_features))
        x = layers.LSTM(units, return_sequences=False)(inp)
        x = layers.Dropout(dropout)(x)
        x = layers.Dense(32, activation="relu")(x)
        out = layers.Dense(1)(x)
        model = models.Model(inp, out)
        model.compile(optimizer="adam", loss="mse")
        return model

    n_features = X_seq_train_s.shape[2]
    tf.keras.backend.clear_session()
    lstm_model = build_lstm(SEQ_LEN, n_features, units=LSTM_UNITS, dropout=LSTM_DROPOUT)

    es = callbacks.EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True)

    history = lstm_model.fit(
        X_seq_train_s, y_seq_train_s,
        validation_data=(X_seq_val_s, y_seq_val_s),
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        callbacks=[es], verbose=2
    )

    # predictions & inverse transform
    pred_train_s = lstm_model.predict(X_seq_train_s).flatten()
    pred_val_s = lstm_model.predict(X_seq_val_s).flatten()
    pred_test_s = lstm_model.predict(X_seq_test_s).flatten()

    pred_train = scaler_y.inverse_transform(pred_train_s.reshape(-1, 1)).flatten()
    pred_val = scaler_y.inverse_transform(pred_val_s.reshape(-1, 1)).flatten()
    pred_test = scaler_y.inverse_transform(pred_test_s.reshape(-1, 1)).flatten()

    # === Evaluations for train / val / test ===
    eval_train = evaluate(y_seq_train, pred_train)
    eval_val = evaluate(y_seq_val, pred_val)
    eval_test = evaluate(y_seq_test, pred_test)

    print("\nEvaluation (Train / Val / Test):")
    print(f"Train -> RMSE: {eval_train['RMSE']:.4f}, MAE: {eval_train['MAE']:.4f}, R2: {eval_train['R2']:.4f}")
    print(f"Val   -> RMSE: {eval_val['RMSE']:.4f}, MAE: {eval_val['MAE']:.4f}, R2: {eval_val['R2']:.4f}")
    print(f"Test  -> RMSE: {eval_test['RMSE']:.4f}, MAE: {eval_test['MAE']:.4f}, R2: {eval_test['R2']:.4f}")

    # save metrics
    metrics_df = pd.DataFrame({
        "split": ["train", "val", "test"],
        "RMSE": [eval_train["RMSE"], eval_val["RMSE"], eval_test["RMSE"]],
        "MAE": [eval_train["MAE"], eval_val["MAE"], eval_test["MAE"]],
        "R2": [eval_train["R2"], eval_val["R2"], eval_test["R2"]]
    })

    safe_name = target_col.replace(".", "").replace(" ", "_")
    metrics_csv = f"results/metrics/metrics_{safe_name}.csv"
    metrics_df.to_csv(metrics_csv, index=False)
    print("Saved metrics to:", metrics_csv)

    # === Plot Actual vs Predicted (Test) ===
    all_dates = df_prep[DATE_COL].iloc[SEQ_LEN:].reset_index(drop=True)
    train_dates = all_dates.iloc[:len(y_seq_train)].reset_index(drop=True)
    val_dates = all_dates.iloc[len(y_seq_train): len(y_seq_train) + len(y_seq_val)].reset_index(drop=True)
    test_dates = all_dates.iloc[len(y_seq_train) + len(y_seq_val):].reset_index(drop=True)

    # Plot test
    plt.figure(figsize=(14, 5))
    plt.plot(test_dates, y_seq_test, label="Actual", linewidth=1.5)
    plt.plot(test_dates, pred_test, label="LSTM_pred", linewidth=1.2)
    plt.legend()
    plt.title(f"{target_col} \u2014 Actual vs LSTM Prediction (Test Set)")
    plt.xlabel("Date")
    plt.ylabel(f"{target_col} concentration (\u00b5g/m\u00b3)")
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # residuals (test)
    plt.figure(figsize=(14, 4))
    plt.plot(test_dates, y_seq_test - pred_test, linewidth=1)
    plt.title(f"{target_col} \u2014 Residuals (Actual - LSTM Prediction) (Test)")
    plt.xlabel("Date")
    plt.ylabel("Residuals")
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Plot sample of train (last n_plot points)
    def plot_series(dates, actual, pred, title_suffix=""):
        plt.figure(figsize=(14, 4))
        plt.plot(dates, actual, label="Actual", linewidth=1.2)
        plt.plot(dates, pred, label="Predicted", linewidth=1.0)
        plt.legend()
        plt.title(title_suffix)
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    n_plot = min(200, len(y_seq_train))
    if n_plot > 0:
        plot_series(train_dates.iloc[-n_plot:], y_seq_train[-n_plot:], pred_train[-n_plot:],
                    title_suffix=f"{target_col} \u2014 Train Actual vs Predicted (last {n_plot})")

    if len(y_seq_val) > 0:
        plot_series(val_dates, y_seq_val, pred_val,
                    title_suffix=f"{target_col} \u2014 Val Actual vs Predicted")

    # save artifacts
    lstm_path = f"results/models/lstm_model_{safe_name}.h5"
    scalerX_path = f"results/models/scaler_X_{safe_name}.pkl"
    scalery_path = f"results/models/scaler_y_{safe_name}.pkl"

    lstm_model.save(lstm_path)
    joblib.dump(scaler_X, scalerX_path)
    joblib.dump(scaler_y, scalery_path)

    print("Saved:", lstm_path, scalerX_path, scalery_path)

    # ===== Additional Evaluation Visualizations =====
    import seaborn as sns

    # 1. Actual vs Predicted (Test)
    y_true = y_seq_test
    y_pred = pred_test

    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.6)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color="red", linestyle="--")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title(f"{target_col} \u2014 Actual vs Predicted (Test Set)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 2. Residual Plot
    residuals = y_true - y_pred
    plt.figure(figsize=(10, 5))
    plt.plot(y_true, residuals, "o", alpha=0.6)
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("Actual Values")
    plt.ylabel("Residuals (Actual - Predicted)")
    plt.title(f"{target_col} \u2014 Residual Plot (Test Set)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 3. Distribution of Errors
    sns.histplot(residuals, bins=30, kde=True)
    plt.title(f"{target_col} \u2014 Distribution of Prediction Errors (Residuals)")
    plt.xlabel("Residuals")
    plt.tight_layout()
    plt.show()

    # 4. (Optional) Pseudo ROC-AUC curve after binarizing target around the median
    from sklearn.metrics import roc_curve, auc

    threshold = np.median(y_true)  # you can choose a fixed value like 100 for PM2.5
    y_true_bin = (y_true > threshold).astype(int)
    y_pred_bin = y_pred

    fpr, tpr, _ = roc_curve(y_true_bin, y_pred_bin)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], color="red", linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{target_col} \u2014 Pseudo ROC Curve (Threshold={threshold:.2f})")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()

    return {
        "lstm_model": lstm_model,
        "scaler_X": scaler_X,
        "scaler_y": scaler_y,
        "y_test": y_seq_test,
        "pred_test_lstm": pred_test,
        "metrics": metrics_df,
        "history": history,
        "lstm_features": lstm_features,
        "df_prep": df_prep
    }


if __name__ == "__main__":
    results = {}
    for t in targets_to_run:
        results[t] = run_lstm_only(df_full, t)

    print("\nAll done (LSTM only).")
