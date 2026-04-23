import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import yaml
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from model.predict import load_model, load_config

def evaluate(config_path: str = "config/config.yaml") -> dict:
    config = load_config(config_path)
    features = config["data"]["features"]
    target = config["data"]["target"]
    n_train = config["data"]["train_scenarios"]

    processed_path = config["paths"]["processed_data"]
    df = pl.read_parquet(processed_path)

    unique_scenarios = df["scenario"].unique().sort().to_list()
    test_snrs = unique_scenarios[n_train:]
    test_df = df.filter(pl.col("scenario").is_in(test_snrs))

    X_test = test_df.select(features).to_pandas()
    y_test = test_df.select(target).to_pandas().values.ravel()

    model = load_model(config["paths"]["model"])
    preds = model.predict(X_test)

    metrics = {
        "MAE": mean_absolute_error(y_test, preds),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, preds))),
        "R2": r2_score(y_test, preds),
    }

    print("\n=== Test Set Performance ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6f}")

    plt.figure(figsize=(8, 5))
    sample_idx = np.random.choice(len(y_test), size=min(5000, len(y_test)), replace=False)
    plt.scatter(y_test[sample_idx], preds[sample_idx], alpha=0.3, s=5)
    lims = [min(y_test.min(), preds.min()), max(y_test.max(), preds.max())]
    plt.plot(lims, lims, linestyle="--", linewidth=1)
    plt.xlabel("Actual Rf (standardised)")
    plt.ylabel("Predicted Rf (standardised)")
    plt.title(f"Predicted vs Actual  |  R²={metrics['R2']:.4f}")
    plt.tight_layout()
    plt.savefig("artifacts/eval_predicted_vs_actual.png", dpi=120)
    print("[evaluate] Plot saved to artifacts/eval_predicted_vs_actual.png")
    plt.show()

    return metrics

if __name__ == "__main__":
    evaluate()