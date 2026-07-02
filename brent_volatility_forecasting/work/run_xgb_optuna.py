import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler


ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "work"
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(exist_ok=True)

DATA_PATH = WORK / "veriseti.xlsx"
SEED = 42
N_TRIALS = 100


def metrics(y_true, y_pred):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "r2": r2}


def direction_accuracy(y_true, y_pred):
    return float((np.sign(y_true) == np.sign(y_pred)).mean())


df = pd.read_excel(DATA_PATH, sheet_name="data (1)")
df.columns = df.columns.str.strip()
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").set_index("Date")

cols = ["Brent_Petrol", "OVX", "GPRD", "GPRD_THREAT"]
for col in cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df = df.dropna(subset=cols)

train_ratio = 0.70
val_ratio = 0.15
n_raw = len(df)
t_end = int(n_raw * train_ratio)
v_end = int(n_raw * (train_ratio + val_ratio))

df_train = df.iloc[:t_end].copy()
df_val = df.iloc[t_end:v_end].copy()
df_test = df.iloc[v_end:].copy()

train_end_date = df_train.index[-1]
val_end_date = df_val.index[-1]

# Leakage-free replacement for non-causal wavelet denoising.
for part in (df_train, df_val, df_test):
    part["Brent_Petrol_clean"] = part["Brent_Petrol"].values
    part["GPRD_clean"] = part["GPRD"].values
    part["GPRD_THREAT_clean"] = part["GPRD_THREAT"].values

log_cols = ["GPRD_clean", "GPRD_THREAT_clean", "OVX"]
winsor_bounds_base = {}
for col in log_cols:
    log_vals = np.log1p(np.clip(df_train[col], 0, None))
    winsor_bounds_base[col] = (
        np.percentile(log_vals, 1.0),
        np.percentile(log_vals, 99.0),
    )


def apply_log_winsor(df_part):
    out = df_part.copy()
    for col in log_cols:
        log_vals = np.log1p(np.clip(df_part[col], 0, None))
        lo, hi = winsor_bounds_base[col]
        fin_col = col.replace("_clean", "_final") if "_clean" in col else col + "_final"
        out[fin_col] = np.clip(log_vals, lo, hi)
    return out


df_train = apply_log_winsor(df_train)
df_val = apply_log_winsor(df_val)
df_test = apply_log_winsor(df_test)
df_full = pd.concat([df_train, df_val, df_test]).sort_index()

# Target: next-day Brent log return.
df_full["Target"] = np.log(df_full["Brent_Petrol"].shift(-1) / df_full["Brent_Petrol"])

lag_max = 5
ema_spans = [5, 10, 20]
base_series = {
    "Brent_clean": df_full["Brent_Petrol_clean"],
    "GPRD": df_full["GPRD_final"],
    "GPRD_THREAT": df_full["GPRD_THREAT_final"],
    "OVX": df_full["OVX_final"],
}

for name, series in base_series.items():
    for lag in range(1, lag_max + 1):
        df_full[f"{name}_lag{lag}"] = series.shift(lag)
    for span in ema_spans:
        df_full[f"{name}_ema{span}"] = series.ewm(span=span, adjust=False).mean().shift(1)

brent_ret = np.log(df_full["Brent_Petrol"] / df_full["Brent_Petrol"].shift(1))
df_full["brent_vol5"] = brent_ret.rolling(5).std().shift(1)
df_full["brent_vol20"] = brent_ret.rolling(20).std().shift(1)
df_full["vol_ratio"] = df_full["brent_vol5"] / (df_full["brent_vol20"] + 1e-9)

roll_win = 60
ovx_raw = df_full["OVX_final"]
ovx_roll_mean = ovx_raw.rolling(roll_win).mean().shift(1)
ovx_roll_std = ovx_raw.rolling(roll_win).std().shift(1)
df_full["ovx_zscore"] = (ovx_raw.shift(1) - ovx_roll_mean) / (ovx_roll_std + 1e-9)
df_full["ovx_spike"] = (df_full["ovx_zscore"] > 2.0).astype(int)
ovx_ema20 = ovx_raw.ewm(span=20, adjust=False).mean().shift(1)
ovx_ema60 = ovx_raw.ewm(span=60, adjust=False).mean().shift(1)
df_full["ovx_high_regime"] = (ovx_ema20 > ovx_ema60).astype(int)
df_full["ovx_mean_revert"] = df_full["ovx_zscore"] * (-(ovx_raw.diff().shift(1)))

gprd_raw = df_full["GPRD_final"]
gprd_t_raw = df_full["GPRD_THREAT_final"]
df_full["gpr_threat_ratio"] = gprd_t_raw.shift(1) / (gprd_raw.shift(1) + 1e-9)
gprd_roll_mean = gprd_raw.rolling(roll_win).mean().shift(1)
gprd_roll_std = gprd_raw.rolling(roll_win).std().shift(1)
df_full["gprd_zscore"] = (gprd_raw.shift(1) - gprd_roll_mean) / (gprd_roll_std + 1e-9)
df_full["gprd_spike"] = (df_full["gprd_zscore"] > 2.0).astype(int)
gprd_ema5 = gprd_raw.ewm(span=5, adjust=False).mean().shift(1)
gprd_ema20 = gprd_raw.ewm(span=20, adjust=False).mean().shift(1)
df_full["gprd_momentum"] = gprd_ema5 - gprd_ema20
gprd_t_mean = gprd_t_raw.rolling(roll_win).mean().shift(1)
gprd_t_std = gprd_t_raw.rolling(roll_win).std().shift(1)
df_full["gprd_threat_zscore"] = (
    gprd_t_raw.shift(1) - gprd_t_mean
) / (gprd_t_std + 1e-9)

df_full["ovx_gprd_interact"] = df_full["OVX_final"].shift(1) * df_full["GPRD_final"].shift(1)
df_full["ovx_gprd_threat_interact"] = (
    df_full["OVX_final"].shift(1) * df_full["GPRD_THREAT_final"].shift(1)
)
df_full["regime_gprd_interact"] = df_full["ovx_high_regime"] * df_full["GPRD_final"].shift(1)
df_full["spike_ovx_interact"] = df_full["gprd_spike"] * df_full["OVX_final"].shift(1)

df_full["day_of_week"] = df_full.index.dayofweek
df_full["month"] = df_full.index.month
df_full["is_month_end"] = df_full.index.is_month_end.astype(int)
df_full["is_month_start"] = df_full.index.is_month_start.astype(int)

drop_cols = [
    "Brent_Petrol",
    "OVX",
    "GPRD",
    "GPRD_THREAT",
    "Brent_Petrol_clean",
    "GPRD_clean",
    "GPRD_THREAT_clean",
]
df_model = df_full.dropna().copy()
features = [
    c
    for c in df_model.columns
    if c not in drop_cols + ["Target", "GPRD_final", "GPRD_THREAT_final", "OVX_final"]
]

train = df_model.loc[:train_end_date]
val = df_model.loc[(df_model.index > train_end_date) & (df_model.index <= val_end_date)]
test = df_model.loc[df_model.index > val_end_date]

X_train = train[features]
y_train = train["Target"]
X_val = val[features]
y_val = val["Target"]
X_test = test[features]
y_test = test["Target"]

binary_feats = ["ovx_spike", "ovx_high_regime", "gprd_spike", "is_month_end", "is_month_start"]
continuous_feats = [f for f in features if f not in binary_feats]

winsor_bounds = {}
for col in continuous_feats:
    winsor_bounds[col] = (np.percentile(X_train[col], 1.0), np.percentile(X_train[col], 99.0))


def apply_winsor(frame):
    out = frame.copy()
    for col in continuous_feats:
        out[col] = np.clip(out[col], *winsor_bounds[col])
    return out


X_train_w = apply_winsor(X_train)
X_val_w = apply_winsor(X_val)
X_test_w = apply_winsor(X_test)

scaler = MinMaxScaler()
scaler.fit(X_train_w[continuous_feats])


def scale_df(frame):
    out = frame.copy()
    out[continuous_feats] = scaler.transform(frame[continuous_feats])
    return out


X_train_s = scale_df(X_train_w)
X_val_s = scale_df(X_val_w)
X_test_s = scale_df(X_test_w)

def objective(trial):
    params = {
        "objective": "reg:squarederror",
        "n_estimators": trial.suggest_int("n_estimators", 250, 2500),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.08, log=True),
        "max_depth": trial.suggest_int("max_depth", 1, 5),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.45, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.45, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 2.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 2.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.2, 30.0, log=True),
        "random_state": SEED,
        "n_jobs": 2,
        "eval_metric": "rmse",
        "early_stopping_rounds": 80,
    }
    trial_model = xgb.XGBRegressor(**params)
    trial_model.fit(X_train_s, y_train, eval_set=[(X_val_s, y_val)], verbose=False)
    pred_val_trial = trial_model.predict(X_val_s)
    trial.set_user_attr("best_iteration", int(getattr(trial_model, "best_iteration", params["n_estimators"])))
    return float(np.sqrt(mean_squared_error(y_val, pred_val_trial)))


optuna.logging.set_verbosity(optuna.logging.WARNING)
sampler = optuna.samplers.TPESampler(seed=SEED)
study = optuna.create_study(direction="minimize", sampler=sampler)
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

best_params = dict(study.best_params)
best_params.update(
    {
        "objective": "reg:squarederror",
        "random_state": SEED,
        "n_jobs": 2,
        "eval_metric": "rmse",
        "early_stopping_rounds": 80,
    }
)

model = xgb.XGBRegressor(**best_params)
model.fit(X_train_s, y_train, eval_set=[(X_val_s, y_val)], verbose=False)

pred_train = model.predict(X_train_s)
pred_val = model.predict(X_val_s)
pred_test = model.predict(X_test_s)

zero_test = np.zeros_like(y_test.values)
mean_test = np.full_like(y_test.values, y_train.mean(), dtype=float)

test_current_price = test["Brent_Petrol"].values
test_actual_next_price = test_current_price * np.exp(y_test.values)
test_pred_next_price = test_current_price * np.exp(pred_test)
test_zero_next_price = test_current_price
test_mean_next_price = test_current_price * np.exp(mean_test)

# Result interpretation note:
# This experiment predicts next-day log return. It must be compared against
# zero-return and train-mean baselines; high price-level R2 alone is misleading
# because Brent prices are highly persistent.
result = {
    "rows": {
        "raw": int(len(df)),
        "train": int(len(train)),
        "val": int(len(val)),
        "test": int(len(test)),
    },
    "date_ranges": {
        "train": [str(train.index.min().date()), str(train.index.max().date())],
        "val": [str(val.index.min().date()), str(val.index.max().date())],
        "test": [str(test.index.min().date()), str(test.index.max().date())],
    },
    "feature_count": int(len(features)),
    "optimization": {
        "n_trials": N_TRIALS,
        "best_validation_rmse": float(study.best_value),
        "best_params": study.best_params,
    },
    "best_iteration": int(getattr(model, "best_iteration", model.n_estimators)),
    "return_metrics": {
        "train": metrics(y_train, pred_train),
        "val": metrics(y_val, pred_val),
        "test": metrics(y_test, pred_test),
        "test_zero_return_baseline": metrics(y_test, zero_test),
        "test_train_mean_return_baseline": metrics(y_test, mean_test),
    },
    "direction_accuracy": {
        "test_xgboost": direction_accuracy(y_test.values, pred_test),
        "test_zero_return_baseline": direction_accuracy(y_test.values, zero_test),
        "test_train_mean_return_baseline": direction_accuracy(y_test.values, mean_test),
    },
    "price_metrics": {
        "test_xgboost": metrics(test_actual_next_price, test_pred_next_price),
        "test_random_walk_baseline": metrics(test_actual_next_price, test_zero_next_price),
        "test_train_mean_return_baseline": metrics(test_actual_next_price, test_mean_next_price),
    },
}

importance = pd.DataFrame(
    {
        "feature": features,
        "importance": model.feature_importances_,
    }
).sort_values("importance", ascending=False)

predictions = pd.DataFrame(
    {
        "Date": test.index,
        "Brent_t": test_current_price,
        "Actual_Brent_t_plus_1": test_actual_next_price,
        "Pred_Brent_t_plus_1": test_pred_next_price,
        "Actual_Log_Return": y_test.values,
        "Pred_Log_Return": pred_test,
    }
)

trials = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs", "state"))

importance.to_csv(OUTPUTS / "xgb_optuna_feature_importance.csv", index=False)
predictions.to_csv(OUTPUTS / "xgb_optuna_test_predictions.csv", index=False)
trials.to_csv(OUTPUTS / "xgb_optuna_trials.csv", index=False)
joblib.dump(model, OUTPUTS / "xgb_optuna_brent_model.pkl")
joblib.dump(scaler, OUTPUTS / "xgb_optuna_feature_scaler.pkl")

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(test.index, test_actual_next_price, label="Actual Brent t+1", lw=1.2)
ax.plot(test.index, test_pred_next_price, label="XGBoost prediction", lw=1.2)
ax.plot(test.index, test_zero_next_price, label="Random walk baseline", lw=1.0, alpha=0.8)
ax.set_title("Brent t+1: XGBoost vs Actual vs Random Walk")
ax.set_ylabel("USD / barrel")
ax.grid(True, alpha=0.25)
ax.legend()
fig.tight_layout()
fig.savefig(OUTPUTS / "xgb_optuna_test_predictions.png", dpi=160)
plt.close(fig)

with (OUTPUTS / "xgb_optuna_results.json").open("w", encoding="utf-8") as f:
    json.dump(result, f, indent=2)

print(json.dumps(result, indent=2))
print("\nTop 15 features:")
print(importance.head(15).to_string(index=False))
print(f"\nSaved: {OUTPUTS / 'xgb_optuna_results.json'}")
print(f"Saved: {OUTPUTS / 'xgb_optuna_test_predictions.csv'}")
print(f"Saved: {OUTPUTS / 'xgb_optuna_feature_importance.csv'}")
print(f"Saved: {OUTPUTS / 'xgb_optuna_trials.csv'}")
print(f"Saved: {OUTPUTS / 'xgb_optuna_test_predictions.png'}")
