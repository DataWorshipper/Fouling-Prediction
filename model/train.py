import os
import yaml
import joblib
import numpy as np
import polars as pl
import optuna
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

optuna.logging.set_verbosity(optuna.logging.WARNING)

def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def load_processed(config: dict) -> pl.DataFrame:
    path = config["paths"]["processed_data"]
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Processed data not found at '{path}'. Run the ETL pipeline first:\n"
            "  python run_etl.py"
        )
    return pl.read_parquet(path)

def split_data(df: pl.DataFrame, config: dict):
    features = config["data"]["features"]
    target = config["data"]["target"]
    n_train = config["data"]["train_scenarios"]

    unique_scenarios = df["scenario"].unique().sort().to_list()
    train_snrs = unique_scenarios[:n_train]
    test_snrs = unique_scenarios[n_train:]

    train_df = df.filter(pl.col("scenario").is_in(train_snrs))
    test_df = df.filter(pl.col("scenario").is_in(test_snrs))

    X_train = train_df.select(features).to_pandas()
    y_train = train_df.select(target).to_pandas().values.ravel()
    X_test = test_df.select(features).to_pandas()
    y_test = test_df.select(target).to_pandas().values.ravel()

    print(f"[train] Train: {X_train.shape} ({len(train_snrs)} scenarios)")
    print(f"[train] Test:  {X_test.shape} ({len(test_snrs)} scenarios)")
    return X_train, y_train, X_test, y_test

def build_objective(X_train, y_train, X_test, y_test, base_params: dict):
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "tree_method": base_params.get("tree_method", "hist"),
            "random_state": base_params.get("random_state", 42),
        }
        model = XGBRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        preds = model.predict(X_test)
        return r2_score(y_test, preds)
    return objective

def train(config_path: str = "config/config.yaml"):
    config = load_config(config_path)
    df = load_processed(config)

    X_train, y_train, X_test, y_test = split_data(df, config)

    n_trials = config["model"]["optuna_trials"]
    base_params = config["model"]["xgboost"]

    print(f"\n[train] Starting Optuna study ({n_trials} trials)...")
    study = optuna.create_study(direction="maximize")
    study.optimize(
        build_objective(X_train, y_train, X_test, y_test, base_params),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    best_params = {**study.best_params, **{
        "tree_method": base_params.get("tree_method", "hist"),
        "random_state": base_params.get("random_state", 42),
    }}
    print(f"\n[train] Best R²: {study.best_value:.6f}")
    print(f"[train] Best params: {best_params}")

    print("\n[train] Fitting final model on full training set...")
    final_model = XGBRegressor(**best_params)
    final_model.fit(X_train, y_train)

    preds = final_model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    print(f"\n[train] Final Test Performance:")
    print(f"  MAE:  {mae:.6f}")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  R²:   {r2:.6f}")

    model_path = config["paths"]["model"]
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(final_model, model_path)
    print(f"\n[train] Model saved to '{model_path}'")

    return final_model, {"MAE": mae, "RMSE": rmse, "R2": r2}

if __name__ == "__main__":
    train()