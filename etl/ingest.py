import os
import shutil
import polars as pl
import yaml
import kagglehub
def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def download_from_kaggle(dest_path: str) -> str:
    try:
        
        path = kagglehub.dataset_download(
            "fdavidsantillan/shell-and-tube-heat-exchanger-fouling-simulation"
        )
        for root, _, files in os.walk(path):
            for f in files:
                if f.endswith(".csv"):
                    src = os.path.join(root, f)
                    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                    shutil.copy(src, dest_path)
                    print(f"[ingest] Dataset copied to: {dest_path}")
                    return dest_path
        raise FileNotFoundError("CSV not found in kagglehub download.")
    except ImportError:
        raise ImportError(
            "kagglehub not installed. Run: pip install kagglehub\n"
            "Or manually place the CSV at the path specified in config.yaml"
        )

def load_raw(config: dict) -> pl.DataFrame:
    raw_path = config["paths"]["raw_data"]

    if not os.path.exists(raw_path):
        print(f"[ingest] Raw data not found at '{raw_path}'. Attempting kagglehub download...")
        download_from_kaggle(raw_path)

    df = pl.read_csv(raw_path)
    print(f"[ingest] Loaded {df.shape[0]:,} rows x {df.shape[1]} columns")
    return df

if __name__ == "__main__":
    cfg = load_config()
    df = load_raw(cfg)
    print(df.head(3))