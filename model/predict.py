import os
import yaml
import joblib
import numpy as np
import pandas as pd
import polars as pl
from xgboost import XGBRegressor

def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def load_model(model_path: str) -> XGBRegressor:
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at '{model_path}'. Train it first:\n"
            "  python -c \"from model.train import train; train()\""
        )
    return joblib.load(model_path)

def load_scaler(scaler_path: str):
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(
            f"Scaler not found at '{scaler_path}'. Run the ETL pipeline first."
        )
    return joblib.load(scaler_path)

def batch_predict(df: pl.DataFrame, config_path: str = "config/config.yaml") -> np.ndarray:
    config = load_config(config_path)
    features = config["data"]["features"]
    model = load_model(config["paths"]["model"])
    X = df.select(features).to_pandas()
    return model.predict(X)

def predict_single(row: dict, config_path: str = "config/config.yaml") -> float:
    config = load_config(config_path)
    features = config["data"]["features"]
    model = load_model(config["paths"]["model"])

    missing = [f for f in features if f not in row]
    if missing:
        raise ValueError(f"Missing features in input: {missing}")

    X = pd.DataFrame([{f: row[f] for f in features}])
    prediction = model.predict(X)[0]
    return float(prediction)

if __name__ == "__main__":
    sample = {
        "flow_actual": 0.5,
        "T_in_actual_K": 0.2,
        "R_wall": -0.1,
        "target_Rf_lag_1": 0.3,
        "flow_lag_1": 0.5,
        "target_Rf_rolling_12h": 0.25,
        "hours_since_clean": 50.0,
    }
    result = predict_single(sample)
    print(f"Predicted Rf: {result:.6f}")