import json
import random
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "work"
OUTPUTS = ROOT / "outputs"
SEED = 42


def metrics(y_true, y_pred):
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def direction_accuracy(y_true, y_pred):
    return float((np.sign(y_true) == np.sign(y_pred)).mean())


# Reuse the already audited preprocessing and AttentionBiLSTM definitions without
# executing that script's training block.
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
y_all = df_model["Target"]

with (OUTPUTS / "xgb_optuna_results.json").open("r", encoding="utf-8") as f:
    xgb_results = json.load(f)
with (OUTPUTS / "attention_bilstm_optuna_results.json").open("r", encoding="utf-8") as f:
    lstm_results = json.load(f)

xgb_params = dict(xgb_results["optimization"]["best_params"])
xgb_params.update(
    {
        "objective": "reg:squarederror",
        "random_state": SEED,
        "n_jobs": 2,
        "eval_metric": "rmse",
        "early_stopping_rounds": 80,
    }
)
lstm_params = dict(lstm_results["optimization"]["best_params"])

X_train = X_scaled_all.loc[train.index, features]
y_train = train["Target"]
X_val = X_scaled_all.loc[val.index, features]
y_val = val["Target"]
X_test = X_scaled_all.loc[test.index, features]
y_test = test["Target"]

xgb_model = xgb.XGBRegressor(**xgb_params)
xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
xgb_pred_val = xgb_model.predict(X_val)
xgb_pred_test = xgb_model.predict(X_test)


def train_lstm(params):
    lookback = int(params["lookback"])
    X_train_seq, y_train_seq, train_dates = make_sequences(X_scaled_all, y_all, train.index, lookback)
    X_val_seq, y_val_seq, val_dates = make_sequences(X_scaled_all, y_all, val.index, lookback)
    X_test_seq, y_test_seq, test_dates = make_sequences(X_scaled_all, y_all, test.index, lookback)

    y_mean = float(y_train_seq.mean())
    y_std = float(y_train_seq.std() + 1e-9)
    y_train_z = (y_train_seq - y_mean) / y_std
    y_val_z = (y_val_seq - y_mean) / y_std

    torch.manual_seed(SEED)
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
    max_epochs = 250
    patience = 35

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
        pred_val_z, attn_val = model(val_x)
        pred_test_z, attn_test = model(torch.tensor(X_test_seq))

    return {
        "model": model,
        "lookback": lookback,
        "best_epoch": best_epoch,
        "val_dates": val_dates,
        "test_dates": test_dates,
        "y_val": y_val_seq,
        "y_test": y_test_seq,
        "pred_val": pred_val_z.numpy() * y_std + y_mean,
        "pred_test": pred_test_z.numpy() * y_std + y_mean,
        "attn_test": attn_test.numpy(),
    }


lstm = train_lstm(lstm_params)

val_dates = lstm["val_dates"]
test_dates = lstm["test_dates"]
y_val_aligned = lstm["y_val"]
y_test_aligned = lstm["y_test"]
xgb_pred_val_aligned = pd.Series(xgb_pred_val, index=val.index).loc[val_dates].values
xgb_pred_test_aligned = pd.Series(xgb_pred_test, index=test.index).loc[test_dates].values
lstm_pred_val = lstm["pred_val"]
lstm_pred_test = lstm["pred_test"]

meta_X_val = np.column_stack([xgb_pred_val_aligned, lstm_pred_val])
meta_X_test = np.column_stack([xgb_pred_test_aligned, lstm_pred_test])

ridge = RidgeCV(alphas=np.logspace(-6, 2, 25))
ridge.fit(meta_X_val, y_val_aligned)
stack_pred_test = ridge.predict(meta_X_test)

linear = LinearRegression()
linear.fit(meta_X_val, y_val_aligned)
linear_pred_test = linear.predict(meta_X_test)

avg_pred_test = 0.5 * xgb_pred_test_aligned + 0.5 * lstm_pred_test

zero_test = np.zeros_like(y_test_aligned)
mean_test = np.full_like(y_test_aligned, y_train.mean(), dtype=float)

test_rows = df_model.loc[test_dates]
test_current_price = test_rows["Brent_Petrol"].values
test_actual_next_price = test_current_price * np.exp(y_test_aligned)
price_preds = {
    "stacking_ridge": test_current_price * np.exp(stack_pred_test),
    "stacking_linear": test_current_price * np.exp(linear_pred_test),
    "simple_average": test_current_price * np.exp(avg_pred_test),
    "xgboost": test_current_price * np.exp(xgb_pred_test_aligned),
    "attention_bilstm": test_current_price * np.exp(lstm_pred_test),
    "random_walk_baseline": test_current_price,
    "train_mean_return_baseline": test_current_price * np.exp(mean_test),
}

# Result interpretation note:
# The meta-model is trained only on validation predictions, then evaluated once
# on test. If stacking underperforms the base models or baselines, the base-model
# predictions are not stable enough to combine through a learned meta-model.
result = {
    "model": "Stacking_XGBoost_AttentionBiLSTM",
    "meta_model": "RidgeCV trained on validation predictions",
    "rows": {
        "val_meta_rows": int(len(y_val_aligned)),
        "test_rows": int(len(y_test_aligned)),
    },
    "date_ranges": {
        "val": [str(val_dates.min().date()), str(val_dates.max().date())],
        "test": [str(test_dates.min().date()), str(test_dates.max().date())],
    },
    "base_models": {
        "xgboost_best_iteration": int(getattr(xgb_model, "best_iteration", xgb_params["n_estimators"])),
        "lstm_lookback": int(lstm["lookback"]),
        "lstm_best_epoch": int(lstm["best_epoch"]),
    },
    "meta_coefficients": {
        "ridge_alpha": float(ridge.alpha_),
        "ridge_intercept": float(ridge.intercept_),
        "ridge_xgboost_weight": float(ridge.coef_[0]),
        "ridge_lstm_weight": float(ridge.coef_[1]),
        "linear_intercept": float(linear.intercept_),
        "linear_xgboost_weight": float(linear.coef_[0]),
        "linear_lstm_weight": float(linear.coef_[1]),
    },
    "return_metrics": {
        "stacking_ridge": metrics(y_test_aligned, stack_pred_test),
        "stacking_linear": metrics(y_test_aligned, linear_pred_test),
        "simple_average": metrics(y_test_aligned, avg_pred_test),
        "xgboost": metrics(y_test_aligned, xgb_pred_test_aligned),
        "attention_bilstm": metrics(y_test_aligned, lstm_pred_test),
        "zero_return_baseline": metrics(y_test_aligned, zero_test),
        "train_mean_return_baseline": metrics(y_test_aligned, mean_test),
    },
    "direction_accuracy": {
        "stacking_ridge": direction_accuracy(y_test_aligned, stack_pred_test),
        "stacking_linear": direction_accuracy(y_test_aligned, linear_pred_test),
        "simple_average": direction_accuracy(y_test_aligned, avg_pred_test),
        "xgboost": direction_accuracy(y_test_aligned, xgb_pred_test_aligned),
        "attention_bilstm": direction_accuracy(y_test_aligned, lstm_pred_test),
        "train_mean_return_baseline": direction_accuracy(y_test_aligned, mean_test),
    },
    "price_metrics": {
        name: metrics(test_actual_next_price, pred) for name, pred in price_preds.items()
    },
}

predictions = pd.DataFrame(
    {
        "Date": test_dates,
        "Brent_t": test_current_price,
        "Actual_Brent_t_plus_1": test_actual_next_price,
        "XGB_Log_Return": xgb_pred_test_aligned,
        "LSTM_Log_Return": lstm_pred_test,
        "Stack_Ridge_Log_Return": stack_pred_test,
        "Stack_Linear_Log_Return": linear_pred_test,
        "Simple_Avg_Log_Return": avg_pred_test,
        "Actual_Log_Return": y_test_aligned,
        "Stack_Ridge_Brent_t_plus_1": price_preds["stacking_ridge"],
    }
)

predictions.to_csv(OUTPUTS / "stacking_xgb_lstm_predictions.csv", index=False)
joblib.dump(ridge, OUTPUTS / "stacking_xgb_lstm_ridge_meta.pkl")
torch.save(lstm["model"].state_dict(), OUTPUTS / "stacking_attention_bilstm_model.pt")
joblib.dump(xgb_model, OUTPUTS / "stacking_xgb_model.pkl")

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(test_dates, test_actual_next_price, label="Actual Brent t+1", lw=1.2)
ax.plot(test_dates, price_preds["stacking_ridge"], label="Stacking Ridge", lw=1.2)
ax.plot(test_dates, price_preds["xgboost"], label="XGBoost", lw=1.0, alpha=0.85)
ax.plot(test_dates, price_preds["attention_bilstm"], label="Attention BiLSTM", lw=1.0, alpha=0.85)
ax.plot(test_dates, price_preds["random_walk_baseline"], label="Random walk", lw=1.0, alpha=0.75)
ax.set_title("Brent t+1: Stacking vs Base Models")
ax.set_ylabel("USD / barrel")
ax.grid(True, alpha=0.25)
ax.legend()
fig.tight_layout()
fig.savefig(OUTPUTS / "stacking_xgb_lstm_predictions.png", dpi=160)
plt.close(fig)

with (OUTPUTS / "stacking_xgb_lstm_results.json").open("w", encoding="utf-8") as f:
    json.dump(result, f, indent=2)

print(json.dumps(result, indent=2))
print(f"\nSaved: {OUTPUTS / 'stacking_xgb_lstm_results.json'}")
print(f"Saved: {OUTPUTS / 'stacking_xgb_lstm_predictions.csv'}")
print(f"Saved: {OUTPUTS / 'stacking_xgb_lstm_predictions.png'}")
