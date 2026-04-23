import polars as pl

def add_lag_features(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns([
        pl.col("target_Rf").shift(1).over("scenario").alias("target_Rf_lag_1"),
        pl.col("flow_actual").shift(1).over("scenario").alias("flow_lag_1"),
    ])
    print("[feature_engineer] Lag features added.")
    return df

def add_rolling_features(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns([
        pl.col("target_Rf")
          .rolling_mean(window_size=12)
          .over("scenario")
          .alias("target_Rf_rolling_12h"),
    ])
    print("[feature_engineer] Rolling mean feature added.")
    return df

def add_hours_since_clean(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns([
        (
            pl.col("time_hours") -
            pl.when(pl.col("is_cleaning") == 1)
              .then(pl.col("time_hours"))
              .otherwise(None)
              .forward_fill()
              .over("scenario")
              .fill_null(df["time_hours"].min())
        ).alias("hours_since_clean")
    ])
    print("[feature_engineer] hours_since_clean feature added.")
    return df

def run_feature_engineering(df: pl.DataFrame) -> pl.DataFrame:
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_hours_since_clean(df)
    before = df.shape[0]
    df = df.drop_nulls()
    print(f"[feature_engineer] Dropped {before - df.shape[0]} null rows after lag creation.")
    return df