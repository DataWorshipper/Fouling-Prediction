import polars as pl
import pandas as pd
import joblib
import os
from sklearn.preprocessing import StandardScaler

def rename_columns(df: pl.DataFrame, rename_map: dict) -> pl.DataFrame:
    existing = {k: v for k, v in rename_map.items() if k in df.columns}
    df = df.rename(existing)
    print(f"[transform] Renamed {len(existing)} columns.")
    return df

def convert_to_kelvin(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(
        (pl.col("T_in_actual_C") + 273.15).alias("T_in_actual_K")
    )
    print("[transform] Kelvin conversion applied.")
    return df

def drop_columns(df: pl.DataFrame, cols: list) -> pl.DataFrame:
    to_drop = [c for c in cols if c in df.columns]
    df = df.drop(to_drop)
    print(f"[transform] Dropped {len(to_drop)} columns: {to_drop}")
    return df

def scale_features(
    df: pl.DataFrame,
    untouchable_cols: list,
    scaler_path: str,
    fit: bool = True,
) -> pl.DataFrame:
    numeric_cols = [
        col for col in df.columns
        if df[col].dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32)
        and col not in untouchable_cols
    ]

    df_pd = df.to_pandas()

    if fit:
        scaler = StandardScaler()
        df_pd[numeric_cols] = scaler.fit_transform(df_pd[numeric_cols])
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        joblib.dump(scaler, scaler_path)
        print(f"[transform] Scaler fitted and saved to '{scaler_path}'.")
    else:
        scaler = joblib.load(scaler_path)
        cols_to_scale = [c for c in numeric_cols if c in scaler.feature_names_in_]
        df_pd[cols_to_scale] = scaler.transform(df_pd[cols_to_scale])
        print(f"[transform] Scaler loaded and applied from '{scaler_path}'.")

    return pl.from_pandas(df_pd)

def run_transform(df: pl.DataFrame, config: dict, fit_scaler: bool = True) -> pl.DataFrame:
    cfg_data = config["data"]

    df = rename_columns(df, cfg_data["rename_map"])
    df = convert_to_kelvin(df)
    df = drop_columns(df, cfg_data["drop_after_rename"])
    df = drop_columns(df, cfg_data["drop_after_vif"])
    df = scale_features(
        df,
        untouchable_cols=cfg_data["untouchable_cols"],
        scaler_path=config["paths"]["scaler"],
        fit=fit_scaler,
    )
    return df