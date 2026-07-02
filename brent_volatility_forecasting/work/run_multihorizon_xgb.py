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


ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "work"
OUTPUTS = ROOT / "outputs"
SEED = 42
N_TRIALS = 80


def metrics(y_true, y_pred):
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def direction_accuracy(y_true, y_pred):
    return float((np.sign(y_true) == np.sign(y_pred)).mean())


# Reuse the leakage-aware preprocessing/features from the LSTM script.
src = (WORK / "run_attention_bilstm_optuna.py").read_text(encoding="utf-8")
defs = src.split("df_raw, df_model, X_scaled_all, features, train, val, test, scaler = prepare_tabular_data()")[0]
namespace = {"__file__": str(WORK / "run_attention_bilstm_optuna.py")}
exec(defs, namespace)
prepare_tabular_data = namespace["prepare_tabular_data"]

df_raw, df_model, X_scaled_all, features, train_1d, val_1d, test_1d, scaler = prepare_tabular_data()

brent = df_model["Brent_Petrol"]
daily_ret = np.log(brent / brent.shift(1))

targets = {}
for h in [5, 10]:
    targets[f"return_{h}d"] = {
        "kind": "return",
        "horizon": h,
        "y": np.log(brent.shift(-h) / brent),
    }
    future_returns = pd.concat([daily_ret.shift(-i) for i in range(1, h + 1)], axis=1)
    targets[f"volatility_{h}d"] = {
        "kind": "volatility",
        "horizon": h,
        "y": future_returns.std(axis=1),
    }


def fit_xgb_for_target(name, target_spec):
    y_all = target_spec["y"].replace([np.inf, -np.inf], np.nan)
    valid_idx = X_scaled_all.index.intersection(y_all.dropna().index)

    train_idx = train_1d.index.intersection(valid_idx)
    val_idx = val_1d.index.intersection(valid_idx)
    test_idx = test_1d.index.intersection(valid_idx)

    X_train = X_scaled_all.loc[train_idx, features]
    y_train = y_all.loc[train_idx]
    X_val = X_scaled_all.loc[val_idx, features]
    y_val = y_all.loc[val_idx]
    X_test = X_scaled_all.loc[test_idx, features]
    y_test = y_all.loc[test_idx]

    def objective(trial):
        params = {
            "objective": "reg:squarederror",
            "n_estimators": trial.suggest_int("n_estimators", 250, 2200),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.08, log=True),
            "max_depth": trial.suggest_int("max_depth", 1, 5),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 25.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.45, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.45, 1.0),
            "gamma": trial.suggest_float("gamma", 0.0, 2.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 3.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.2, 50.0, log=True),
            "random_state": SEED,
            "n_jobs": 2,
            "eval_metric": "rmse",
            "early_stopping_rounds": 80,
        }
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        pred_val = model.predict(X_val)
        return float(np.sqrt(mean_squared_error(y_val, pred_val)))

    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

    params = dict(study.best_params)
    params.update(
        {
            "objective": "reg:squarederror",
            "random_state": SEED,
            "n_jobs": 2,
            "eval_metric": "rmse",
            "early_stopping_rounds": 80,
        }
    )
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    pred_train = model.predict(X_train)
    pred_val = model.predict(X_val)
    pred_test = model.predict(X_test)

    mean_test = np.full_like(y_test.values, y_train.mean(), dtype=float)
    zero_test = np.zeros_like(y_test.values)

    # Result interpretation note:
    # - For return targets, the model is useful only if it beats zero-return and
    #   train-mean baselines. Otherwise price-level plots can look deceptively good.
    # - For volatility targets, positive test R2 and lower RMSE than both train-mean
    #   and past-volatility baselines indicate usable predictive signal.
    out = {
        "target": name,
        "kind": target_spec["kind"],
        "horizon": target_spec["horizon"],
        "rows": {
            "train": int(len(train_idx)),
            "val": int(len(val_idx)),
            "test": int(len(test_idx)),
        },
        "date_ranges": {
            "train": [str(train_idx.min().date()), str(train_idx.max().date())],
            "val": [str(val_idx.min().date()), str(val_idx.max().date())],
            "test": [str(test_idx.min().date()), str(test_idx.max().date())],
        },
        "best_validation_rmse": float(study.best_value),
        "best_iteration": int(getattr(model, "best_iteration", params["n_estimators"])),
        "best_params": study.best_params,
        "metrics": {
            "train": metrics(y_train, pred_train),
            "val": metrics(y_val, pred_val),
            "test": metrics(y_test, pred_test),
            "test_train_mean_baseline": metrics(y_test, mean_test),
        },
    }

    predictions = pd.DataFrame(
        {
            "Date": test_idx,
            "Actual": y_test.values,
            "Pred": pred_test,
            "Train_Mean_Baseline": mean_test,
        }
    )

    if target_spec["kind"] == "return":
        current_price = df_model.loc[test_idx, "Brent_Petrol"].values
        actual_price = current_price * np.exp(y_test.values)
        pred_price = current_price * np.exp(pred_test)
        mean_price = current_price * np.exp(mean_test)
        rw_price = current_price
        out["metrics"]["test_zero_return_baseline"] = metrics(y_test, zero_test)
        out["direction_accuracy"] = {
            "xgboost": direction_accuracy(y_test.values, pred_test),
            "zero_return_baseline": direction_accuracy(y_test.values, zero_test),
            "train_mean_baseline": direction_accuracy(y_test.values, mean_test),
        }
        out["price_metrics"] = {
            "xgboost": metrics(actual_price, pred_price),
            "random_walk_baseline": metrics(actual_price, rw_price),
            "train_mean_baseline": metrics(actual_price, mean_price),
        }
        predictions["Actual_Brent_t_plus_h"] = actual_price
        predictions["Pred_Brent_t_plus_h"] = pred_price
        predictions["Random_Walk_Brent_t_plus_h"] = rw_price
    else:
        # Persistence baseline: recent realized volatility over the previous h days.
        h = target_spec["horizon"]
        past_vol = daily_ret.rolling(h).std().loc[test_idx].values
        out["metrics"]["test_past_volatility_baseline"] = metrics(y_test, past_vol)
        predictions["Past_Volatility_Baseline"] = past_vol

    importance = pd.DataFrame(
        {"feature": features, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)

    safe = name.replace("/", "_")
    predictions.to_csv(OUTPUTS / f"xgb_{safe}_predictions.csv", index=False)
    importance.to_csv(OUTPUTS / f"xgb_{safe}_feature_importance.csv", index=False)
    joblib.dump(model, OUTPUTS / f"xgb_{safe}_model.pkl")

    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.plot(test_idx, y_test.values, label="Actual", lw=1.2)
    ax.plot(test_idx, pred_test, label="XGBoost", lw=1.2)
    ax.plot(test_idx, mean_test, label="Train mean baseline", lw=1.0, alpha=0.8)
    ax.set_title(f"{name}: XGBoost vs Actual")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUTS / f"xgb_{safe}_predictions.png", dpi=160)
    plt.close(fig)

    return out, importance.head(15)


optuna.logging.set_verbosity(optuna.logging.WARNING)
all_results = {}
top_features = {}
for target_name, target_spec in targets.items():
    result, importance_head = fit_xgb_for_target(target_name, target_spec)
    all_results[target_name] = result
    top_features[target_name] = importance_head.to_dict(orient="records")

summary_rows = []
for target_name, result in all_results.items():
    row = {
        "target": target_name,
        "kind": result["kind"],
        "horizon": result["horizon"],
        "test_rmse": result["metrics"]["test"]["rmse"],
        "test_mae": result["metrics"]["test"]["mae"],
        "test_r2": result["metrics"]["test"]["r2"],
        "train_mean_rmse": result["metrics"]["test_train_mean_baseline"]["rmse"],
    }
    if result["kind"] == "return":
        row["zero_return_rmse"] = result["metrics"]["test_zero_return_baseline"]["rmse"]
        row["direction_accuracy"] = result["direction_accuracy"]["xgboost"]
        row["price_rmse"] = result["price_metrics"]["xgboost"]["rmse"]
        row["random_walk_price_rmse"] = result["price_metrics"]["random_walk_baseline"]["rmse"]
    else:
        row["past_volatility_rmse"] = result["metrics"]["test_past_volatility_baseline"]["rmse"]
    summary_rows.append(row)

summary = pd.DataFrame(summary_rows)
summary.to_csv(OUTPUTS / "xgb_multihorizon_summary.csv", index=False)
with (OUTPUTS / "xgb_multihorizon_results.json").open("w", encoding="utf-8") as f:
    json.dump({"results": all_results, "top_features": top_features}, f, indent=2)

print(summary.to_string(index=False))
print("\nInterpretation:")
print("- 5d/10d return targets are judged against zero-return and train-mean baselines.")
print("- 5d/10d volatility targets are judged against train-mean and past-volatility baselines.")
print("- In the current dataset, the 10d volatility target is the most defensible target.")
print(f"\nSaved: {OUTPUTS / 'xgb_multihorizon_results.json'}")
print(f"Saved: {OUTPUTS / 'xgb_multihorizon_summary.csv'}")
