import json
import random
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "work"
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(exist_ok=True)

DATA_PATH = WORK / "veriseti.xlsx"
SEED = 42
LOOKBACK = 30
BATCH_SIZE = 64
MAX_EPOCHS = 250
PATIENCE = 35
N_TRIALS = 40

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(2)


def metrics(y_true, y_pred):
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def direction_accuracy(y_true, y_pred):
    return float((np.sign(y_true) == np.sign(y_pred)).mean())


def prepare_tabular_data():
    df = pd.read_excel(DATA_PATH, sheet_name="data (1)")
    df.columns = df.columns.str.strip()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").set_index("Date")

    cols = ["Brent_Petrol", "OVX", "GPRD", "GPRD_THREAT"]
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=cols)

    n_raw = len(df)
    t_end = int(n_raw * 0.70)
    v_end = int(n_raw * 0.85)
    df_train = df.iloc[:t_end].copy()
    df_val = df.iloc[t_end:v_end].copy()
    df_test = df.iloc[v_end:].copy()
    train_end_date = df_train.index[-1]
    val_end_date = df_val.index[-1]

    for part in (df_train, df_val, df_test):
        part["Brent_Petrol_clean"] = part["Brent_Petrol"].values
        part["GPRD_clean"] = part["GPRD"].values
        part["GPRD_THREAT_clean"] = part["GPRD_THREAT"].values

    log_cols = ["GPRD_clean", "GPRD_THREAT_clean", "OVX"]
    base_bounds = {}
    for col in log_cols:
        log_vals = np.log1p(np.clip(df_train[col], 0, None))
        base_bounds[col] = (np.percentile(log_vals, 1.0), np.percentile(log_vals, 99.0))

    def apply_log_winsor(frame):
        out = frame.copy()
        for col in log_cols:
            log_vals = np.log1p(np.clip(frame[col], 0, None))
            lo, hi = base_bounds[col]
            fin_col = col.replace("_clean", "_final") if "_clean" in col else col + "_final"
            out[fin_col] = np.clip(log_vals, lo, hi)
        return out

    df_train = apply_log_winsor(df_train)
    df_val = apply_log_winsor(df_val)
    df_test = apply_log_winsor(df_test)
    df_full = pd.concat([df_train, df_val, df_test]).sort_index()

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
    df_full["gprd_threat_zscore"] = (gprd_t_raw.shift(1) - gprd_t_mean) / (gprd_t_std + 1e-9)

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

    binary_feats = ["ovx_spike", "ovx_high_regime", "gprd_spike", "is_month_end", "is_month_start"]
    continuous_feats = [f for f in features if f not in binary_feats]

    X_train = train[features]
    bounds = {}
    for col in continuous_feats:
        bounds[col] = (np.percentile(X_train[col], 1.0), np.percentile(X_train[col], 99.0))

    def apply_winsor(frame):
        out = frame.copy()
        for col in continuous_feats:
            out[col] = np.clip(out[col], *bounds[col])
        return out

    full_w = apply_winsor(df_model[features])
    train_w = full_w.loc[train.index]

    scaler = MinMaxScaler()
    scaler.fit(train_w[continuous_feats])
    full_s = full_w.copy()
    full_s[continuous_feats] = scaler.transform(full_w[continuous_feats])

    return df, df_model, full_s, features, train, val, test, scaler


def make_sequences(X_all, y_all, target_indexes, lookback):
    X_values = X_all.values.astype(np.float32)
    y_values = y_all.values.astype(np.float32)
    pos = pd.Series(np.arange(len(X_all)), index=X_all.index)

    sequences, targets, dates = [], [], []
    for dt in target_indexes:
        i = int(pos.loc[dt])
        if i - lookback + 1 < 0:
            continue
        sequences.append(X_values[i - lookback + 1 : i + 1])
        targets.append(y_values[i])
        dates.append(dt)
    return np.stack(sequences), np.array(targets, dtype=np.float32), pd.Index(dates)


class AttentionBiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=32, dropout=0.15):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.attn = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_size * 2, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        h, _ = self.lstm(x)
        scores = self.attn(h).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(h * weights.unsqueeze(-1), dim=1)
        out = self.head(self.dropout(context)).squeeze(-1)
        return out, weights


df_raw, df_model, X_scaled_all, features, train, val, test, scaler = prepare_tabular_data()
y_all = df_model["Target"]


def train_one_model(params, return_artifacts=False):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    lookback = int(params["lookback"])
    batch_size = int(params["batch_size"])
    max_epochs = int(params.get("max_epochs", MAX_EPOCHS))
    patience = int(params.get("patience", PATIENCE))

    X_train_seq, y_train, train_dates = make_sequences(X_scaled_all, y_all, train.index, lookback)
    X_val_seq, y_val, val_dates = make_sequences(X_scaled_all, y_all, val.index, lookback)
    X_test_seq, y_test, test_dates = make_sequences(X_scaled_all, y_all, test.index, lookback)

    y_mean = float(y_train.mean())
    y_std = float(y_train.std() + 1e-9)
    y_train_z = (y_train - y_mean) / y_std
    y_val_z = (y_val - y_mean) / y_std

    train_ds = TensorDataset(torch.tensor(X_train_seq), torch.tensor(y_train_z))
    generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=generator)
    val_x = torch.tensor(X_val_seq)
    loss_fn = nn.MSELoss()

    model = AttentionBiLSTM(
        input_size=len(features),
        hidden_size=int(params["hidden_size"]),
        dropout=float(params["dropout"]),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(params["lr"]),
        weight_decay=float(params["weight_decay"]),
    )

    best_val = np.inf
    best_state = None
    best_epoch = 0
    wait = 0
    history = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred, _ = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(params["clip_norm"]))
            optimizer.step()
            train_losses.append(float(loss.item()))

        model.eval()
        with torch.no_grad():
            val_pred_z, _ = model(val_x)
            val_loss = float(loss_fn(val_pred_z, torch.tensor(y_val_z)).item())
        history.append({"epoch": epoch, "train_loss": float(np.mean(train_losses)), "val_loss": val_loss})

        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_val_z, _ = model(val_x)
    pred_val = pred_val_z.numpy() * y_std + y_mean
    val_rmse = float(np.sqrt(mean_squared_error(y_val, pred_val)))

    if not return_artifacts:
        return val_rmse, best_epoch

    test_x = torch.tensor(X_test_seq)
    with torch.no_grad():
        pred_train_z, attn_train = model(torch.tensor(X_train_seq))
        pred_val_z, attn_val = model(val_x)
        pred_test_z, attn_test = model(test_x)

    return {
        "model": model,
        "lookback": lookback,
        "history": history,
        "best_epoch": best_epoch,
        "y_mean": y_mean,
        "y_std": y_std,
        "X_train_seq": X_train_seq,
        "X_val_seq": X_val_seq,
        "X_test_seq": X_test_seq,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "train_dates": train_dates,
        "val_dates": val_dates,
        "test_dates": test_dates,
        "pred_train": pred_train_z.numpy() * y_std + y_mean,
        "pred_val": pred_val_z.numpy() * y_std + y_mean,
        "pred_test": pred_test_z.numpy() * y_std + y_mean,
        "attn_test": attn_test.numpy(),
    }


def objective(trial):
    params = {
        "lookback": trial.suggest_categorical("lookback", [10, 20, 30, 45, 60]),
        "hidden_size": trial.suggest_categorical("hidden_size", [8, 16, 24, 32, 48, 64]),
        "dropout": trial.suggest_float("dropout", 0.0, 0.45),
        "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 5e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
        "clip_norm": trial.suggest_float("clip_norm", 0.3, 2.0),
        "max_epochs": 180,
        "patience": 25,
    }
    val_rmse, best_epoch_trial = train_one_model(params, return_artifacts=False)
    trial.set_user_attr("best_epoch", int(best_epoch_trial))
    return val_rmse


optuna.logging.set_verbosity(optuna.logging.WARNING)
sampler = optuna.samplers.TPESampler(seed=SEED)
study = optuna.create_study(direction="minimize", sampler=sampler)
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

best_params = dict(study.best_params)
best_params.update({"max_epochs": MAX_EPOCHS, "patience": PATIENCE})
artifacts = train_one_model(best_params, return_artifacts=True)

model = artifacts["model"]
LOOKBACK = artifacts["lookback"]
history = artifacts["history"]
best_epoch = artifacts["best_epoch"]
y_mean = artifacts["y_mean"]
y_std = artifacts["y_std"]
y_train = artifacts["y_train"]
y_val = artifacts["y_val"]
y_test = artifacts["y_test"]
train_dates = artifacts["train_dates"]
val_dates = artifacts["val_dates"]
test_dates = artifacts["test_dates"]
pred_train = artifacts["pred_train"]
pred_val = artifacts["pred_val"]
pred_test = artifacts["pred_test"]
attn_test = artifacts["attn_test"]

zero_test = np.zeros_like(y_test)
mean_test = np.full_like(y_test, y_mean)

test_rows = df_model.loc[test_dates]
test_current_price = test_rows["Brent_Petrol"].values
test_actual_next_price = test_current_price * np.exp(y_test)
test_pred_next_price = test_current_price * np.exp(pred_test)
test_zero_next_price = test_current_price
test_mean_next_price = test_current_price * np.exp(mean_test)

# Result interpretation note:
# This Attention BiLSTM also targets next-day log return. If RMSE/R2 do not beat
# the simple baselines, extra sequence complexity is not adding usable signal.
result = {
    "model": "AttentionBiLSTM_Optuna",
    "lookback": LOOKBACK,
    "optimization": {
        "n_trials": N_TRIALS,
        "best_validation_rmse": float(study.best_value),
        "best_params": study.best_params,
    },
    "rows": {
        "raw": int(len(df_raw)),
        "train_sequences": int(len(y_train)),
        "val_sequences": int(len(y_val)),
        "test_sequences": int(len(y_test)),
    },
    "date_ranges": {
        "train": [str(train_dates.min().date()), str(train_dates.max().date())],
        "val": [str(val_dates.min().date()), str(val_dates.max().date())],
        "test": [str(test_dates.min().date()), str(test_dates.max().date())],
    },
    "feature_count": int(len(features)),
    "best_epoch": int(best_epoch),
    "return_metrics": {
        "train": metrics(y_train, pred_train),
        "val": metrics(y_val, pred_val),
        "test": metrics(y_test, pred_test),
        "test_zero_return_baseline": metrics(y_test, zero_test),
        "test_train_mean_return_baseline": metrics(y_test, mean_test),
    },
    "direction_accuracy": {
        "test_attention_bilstm": direction_accuracy(y_test, pred_test),
        "test_zero_return_baseline": direction_accuracy(y_test, zero_test),
        "test_train_mean_return_baseline": direction_accuracy(y_test, mean_test),
    },
    "price_metrics": {
        "test_attention_bilstm": metrics(test_actual_next_price, test_pred_next_price),
        "test_random_walk_baseline": metrics(test_actual_next_price, test_zero_next_price),
        "test_train_mean_return_baseline": metrics(test_actual_next_price, test_mean_next_price),
    },
}

lag_attention = pd.DataFrame(
    {
        "lag_from_prediction_day": np.arange(LOOKBACK - 1, -1, -1),
        "mean_attention_weight": attn_test.mean(axis=0),
    }
)
predictions = pd.DataFrame(
    {
        "Date": test_dates,
        "Brent_t": test_current_price,
        "Actual_Brent_t_plus_1": test_actual_next_price,
        "Pred_Brent_t_plus_1": test_pred_next_price,
        "Actual_Log_Return": y_test,
        "Pred_Log_Return": pred_test,
    }
)
history_df = pd.DataFrame(history)
trials_df = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs", "state"))

predictions.to_csv(OUTPUTS / "attention_bilstm_optuna_test_predictions.csv", index=False)
lag_attention.to_csv(OUTPUTS / "attention_bilstm_optuna_lag_attention.csv", index=False)
history_df.to_csv(OUTPUTS / "attention_bilstm_optuna_history.csv", index=False)
trials_df.to_csv(OUTPUTS / "attention_bilstm_optuna_trials.csv", index=False)
torch.save(model.state_dict(), OUTPUTS / "attention_bilstm_optuna_model.pt")
joblib.dump(scaler, OUTPUTS / "attention_bilstm_optuna_feature_scaler.pkl")

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(test_dates, test_actual_next_price, label="Actual Brent t+1", lw=1.2)
ax.plot(test_dates, test_pred_next_price, label="Attention BiLSTM prediction", lw=1.2)
ax.plot(test_dates, test_zero_next_price, label="Random walk baseline", lw=1.0, alpha=0.8)
ax.set_title("Brent t+1: Attention BiLSTM vs Actual vs Random Walk")
ax.set_ylabel("USD / barrel")
ax.grid(True, alpha=0.25)
ax.legend()
fig.tight_layout()
fig.savefig(OUTPUTS / "attention_bilstm_optuna_test_predictions.png", dpi=160)
plt.close(fig)

fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(lag_attention["lag_from_prediction_day"], lag_attention["mean_attention_weight"])
ax.invert_xaxis()
ax.set_title("Mean Attention Weight by Lag on Test Set")
ax.set_xlabel("Lag from prediction day: 0 = current day")
ax.set_ylabel("Mean attention weight")
ax.grid(True, axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig(OUTPUTS / "attention_bilstm_optuna_lag_attention.png", dpi=160)
plt.close(fig)

with (OUTPUTS / "attention_bilstm_optuna_results.json").open("w", encoding="utf-8") as f:
    json.dump(result, f, indent=2)

print(json.dumps(result, indent=2))
print("\nMean attention by lag, top 10:")
print(lag_attention.sort_values("mean_attention_weight", ascending=False).head(10).to_string(index=False))
print(f"\nSaved: {OUTPUTS / 'attention_bilstm_optuna_results.json'}")
print(f"Saved: {OUTPUTS / 'attention_bilstm_optuna_test_predictions.csv'}")
print(f"Saved: {OUTPUTS / 'attention_bilstm_optuna_lag_attention.csv'}")
print(f"Saved: {OUTPUTS / 'attention_bilstm_optuna_trials.csv'}")
print(f"Saved: {OUTPUTS / 'attention_bilstm_optuna_test_predictions.png'}")
print(f"Saved: {OUTPUTS / 'attention_bilstm_optuna_lag_attention.png'}")
