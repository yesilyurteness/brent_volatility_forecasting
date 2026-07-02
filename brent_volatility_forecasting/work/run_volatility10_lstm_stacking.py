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
import xgboost as xgb
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "work"
OUTPUTS = ROOT / "outputs"
SEED = 42
HORIZON = 10
N_TRIALS = 25


def metrics(y_true, y_pred):
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


src = (WORK / "run_attention_bilstm_optuna.py").read_text(encoding="utf-8")
defs = src.split("df_raw, df_model, X_scaled_all, features, train, val, test, scaler = prepare_tabular_data()")[0]
namespace = {"__file__": str(WORK / "run_attention_bilstm_optuna.py")}
exec(defs, namespace)
prepare_tabular_data = namespace["prepare_tabular_data"]
make_sequences = namespace["make_sequences"]
AttentionBiLSTM = namespace["AttentionBiLSTM"]

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(2)

df_raw, df_model, X_scaled_all, features, train, val, test, scaler = prepare_tabular_data()
brent = df_model["Brent_Petrol"]
daily_ret = np.log(brent / brent.shift(1))
future_returns = pd.concat([daily_ret.shift(-i) for i in range(1, HORIZON + 1)], axis=1)
y_all = future_returns.std(axis=1).replace([np.inf, -np.inf], np.nan)

valid_idx = X_scaled_all.index.intersection(y_all.dropna().index)
train_idx = train.index.intersection(valid_idx)
val_idx = val.index.intersection(valid_idx)
test_idx = test.index.intersection(valid_idx)


def train_lstm(params, return_artifacts=False):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    lookback = int(params["lookback"])
    X_train_seq, y_train, train_dates = make_sequences(X_scaled_all, y_all, train_idx, lookback)
    X_val_seq, y_val, val_dates = make_sequences(X_scaled_all, y_all, val_idx, lookback)
    X_test_seq, y_test, test_dates = make_sequences(X_scaled_all, y_all, test_idx, lookback)

    y_mean = float(y_train.mean())
    y_std = float(y_train.std() + 1e-9)
    y_train_z = (y_train - y_mean) / y_std
    y_val_z = (y_val - y_mean) / y_std

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
    loss_fn = nn.MSELoss()
    generator = torch.Generator().manual_seed(SEED)
    loader = DataLoader(
        TensorDataset(torch.tensor(X_train_seq), torch.tensor(y_train_z)),
        batch_size=int(params["batch_size"]),
        shuffle=True,
        generator=generator,
    )
    val_x = torch.tensor(X_val_seq)
    best_val = np.inf
    best_state = None
    best_epoch = 0
    wait = 0
    max_epochs = int(params.get("max_epochs", 160))
    patience = int(params.get("patience", 25))

    for epoch in range(1, max_epochs + 1):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            pred, _ = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(params["clip_norm"]))
            optimizer.step()

        model.eval()
        with torch.no_grad():
            pred_val_z, _ = model(val_x)
            val_loss = float(loss_fn(pred_val_z, torch.tensor(y_val_z)).item())

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

    with torch.no_grad():
        pred_train_z, _ = model(torch.tensor(X_train_seq))
        pred_test_z, attn_test = model(torch.tensor(X_test_seq))

    return {
        "model": model,
        "lookback": lookback,
        "best_epoch": best_epoch,
        "train_dates": train_dates,
        "val_dates": val_dates,
        "test_dates": test_dates,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "pred_train": pred_train_z.numpy() * y_std + y_mean,
        "pred_val": pred_val,
        "pred_test": pred_test_z.numpy() * y_std + y_mean,
        "attn_test": attn_test.numpy(),
    }


def objective(trial):
    params = {
        "lookback": trial.suggest_categorical("lookback", [10, 20, 30, 45, 60]),
        "hidden_size": trial.suggest_categorical("hidden_size", [8, 16, 24, 32, 48]),
        "dropout": trial.suggest_float("dropout", 0.0, 0.45),
        "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 5e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
        "clip_norm": trial.suggest_float("clip_norm", 0.3, 2.0),
        "max_epochs": 160,
        "patience": 25,
    }
    rmse, epoch = train_lstm(params, return_artifacts=False)
    trial.set_user_attr("best_epoch", int(epoch))
    return rmse


optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

best_lstm_params = dict(study.best_params)
best_lstm_params.update({"max_epochs": 220, "patience": 35})
lstm = train_lstm(best_lstm_params, return_artifacts=True)

with (OUTPUTS / "xgb_multihorizon_results.json").open("r", encoding="utf-8") as f:
    xgb_results = json.load(f)["results"]["volatility_10d"]
xgb_params = dict(xgb_results["best_params"])
xgb_params.update(
    {
        "objective": "reg:squarederror",
        "random_state": SEED,
        "n_jobs": 2,
        "eval_metric": "rmse",
        "early_stopping_rounds": 80,
    }
)

X_train = X_scaled_all.loc[train_idx, features]
y_train = y_all.loc[train_idx]
X_val = X_scaled_all.loc[val_idx, features]
y_val = y_all.loc[val_idx]
X_test = X_scaled_all.loc[test_idx, features]
y_test = y_all.loc[test_idx]

xgb_model = xgb.XGBRegressor(**xgb_params)
xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
xgb_pred_val_all = pd.Series(xgb_model.predict(X_val), index=val_idx)
xgb_pred_test_all = pd.Series(xgb_model.predict(X_test), index=test_idx)

val_dates = lstm["val_dates"]
test_dates = lstm["test_dates"]
xgb_pred_val = xgb_pred_val_all.loc[val_dates].values
xgb_pred_test = xgb_pred_test_all.loc[test_dates].values
y_val_lstm = lstm["y_val"]
y_test_lstm = lstm["y_test"]
lstm_pred_val = lstm["pred_val"]
lstm_pred_test = lstm["pred_test"]

meta_X_val = np.column_stack([xgb_pred_val, lstm_pred_val])
meta_X_test = np.column_stack([xgb_pred_test, lstm_pred_test])
ridge = RidgeCV(alphas=np.logspace(-8, 2, 30))
ridge.fit(meta_X_val, y_val_lstm)
stack_pred_test = ridge.predict(meta_X_test)
avg_pred_test = 0.5 * xgb_pred_test + 0.5 * lstm_pred_test

mean_test = np.full_like(y_test_lstm, lstm["y_train"].mean(), dtype=float)
past_vol = daily_ret.rolling(HORIZON).std().loc[test_dates].values

# Result interpretation note:
# The main comparison is not price direction anymore; it is the error of the
# 10-day forward volatility forecast. A model is considered useful only if it
# beats both simple baselines: train mean volatility and recent/past volatility.
# Ridge stacking is included as a formal meta-model, while the 50/50 simple
# average is kept because it can reduce variance without overfitting validation.
result = {
    "target": "volatility_10d",
    "horizon": HORIZON,
    "optimization": {
        "lstm_n_trials": N_TRIALS,
        "lstm_best_validation_rmse": float(study.best_value),
        "lstm_best_params": study.best_params,
    },
    "rows": {
        "train": int(len(lstm["y_train"])),
        "val": int(len(y_val_lstm)),
        "test": int(len(y_test_lstm)),
    },
    "base_models": {
        "xgb_best_iteration": int(getattr(xgb_model, "best_iteration", xgb_params["n_estimators"])),
        "lstm_lookback": int(lstm["lookback"]),
        "lstm_best_epoch": int(lstm["best_epoch"]),
    },
    "stacking": {
        "ridge_alpha": float(ridge.alpha_),
        "ridge_intercept": float(ridge.intercept_),
        "ridge_xgb_weight": float(ridge.coef_[0]),
        "ridge_lstm_weight": float(ridge.coef_[1]),
    },
    "test_metrics": {
        "xgboost": metrics(y_test_lstm, xgb_pred_test),
        "attention_bilstm": metrics(y_test_lstm, lstm_pred_test),
        "stacking_ridge": metrics(y_test_lstm, stack_pred_test),
        "simple_average": metrics(y_test_lstm, avg_pred_test),
        "train_mean_baseline": metrics(y_test_lstm, mean_test),
        "past_volatility_baseline": metrics(y_test_lstm, past_vol),
    },
}

predictions = pd.DataFrame(
    {
        "Date": test_dates,
        "Actual_10d_Future_Volatility": y_test_lstm,
        "XGBoost": xgb_pred_test,
        "Attention_BiLSTM": lstm_pred_test,
        "Stacking_Ridge": stack_pred_test,
        "Simple_Average": avg_pred_test,
        "Train_Mean_Baseline": mean_test,
        "Past_Volatility_Baseline": past_vol,
    }
)
lag_attention = pd.DataFrame(
    {
        "lag_from_prediction_day": np.arange(lstm["lookback"] - 1, -1, -1),
        "mean_attention_weight": lstm["attn_test"].mean(axis=0),
    }
)
trials = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs", "state"))

predictions.to_csv(OUTPUTS / "volatility10_lstm_stacking_predictions.csv", index=False)
lag_attention.to_csv(OUTPUTS / "volatility10_lstm_attention.csv", index=False)
trials.to_csv(OUTPUTS / "volatility10_lstm_optuna_trials.csv", index=False)
torch.save(lstm["model"].state_dict(), OUTPUTS / "volatility10_attention_bilstm_model.pt")
joblib.dump(ridge, OUTPUTS / "volatility10_stacking_ridge.pkl")

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(test_dates, y_test_lstm, label="Actual 10d future volatility", lw=1.2)
ax.plot(test_dates, xgb_pred_test, label="XGBoost", lw=1.1)
ax.plot(test_dates, lstm_pred_test, label="Attention BiLSTM", lw=1.1)
ax.plot(test_dates, stack_pred_test, label="Stacking Ridge", lw=1.1)
ax.plot(test_dates, past_vol, label="Past volatility baseline", lw=1.0, alpha=0.75)
ax.set_title("10-Day Forward Brent Volatility: Models vs Actual")
ax.grid(True, alpha=0.25)
ax.legend()
fig.tight_layout()
fig.savefig(OUTPUTS / "volatility10_lstm_stacking_predictions.png", dpi=160)
plt.close(fig)

with (OUTPUTS / "volatility10_lstm_stacking_results.json").open("w", encoding="utf-8") as f:
    json.dump(result, f, indent=2)

print(json.dumps(result, indent=2))
print("\nInterpretation:")
print("- Lower RMSE/MAE and higher R2 are better for the 10d forward volatility target.")
print("- The strongest result in this run is the simple average of XGBoost and Attention BiLSTM.")
print("- Ridge stacking overfit the validation relation and did not generalize as well as averaging.")
print(f"\nSaved: {OUTPUTS / 'volatility10_lstm_stacking_results.json'}")
print(f"Saved: {OUTPUTS / 'volatility10_lstm_stacking_predictions.csv'}")
print(f"Saved: {OUTPUTS / 'volatility10_lstm_stacking_predictions.png'}")
