import os
import yaml
import polars as pl

from etl.ingest import load_raw
from etl.transform import run_transform
from etl.feature_engineer import run_feature_engineering

def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def run_pipeline(config_path: str = "config/config.yaml", fit_scaler: bool = True) -> pl.DataFrame:
    config = load_config(config_path)

    print("=" * 50)
    print("STEP 1: Ingestion")
    print("=" * 50)
    df = load_raw(config)

    print("\n" + "=" * 50)
    print("STEP 2: Transform")
    print("=" * 50)
    df = run_transform(df, config, fit_scaler=fit_scaler)

    print("\n" + "=" * 50)
    print("STEP 3: Feature Engineering")
    print("=" * 50)
    df = run_feature_engineering(df)

    processed_path = config["paths"]["processed_data"]
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    df.write_parquet(processed_path)

    print(f"\n[pipeline] Processed data saved to '{processed_path}'")
    print(f"[pipeline] Final shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"[pipeline] Features available: {df.columns}")

    return df

if __name__ == "__main__":
    run_pipeline()