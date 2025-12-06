import pandas as pd
import numpy as np
from pathlib import Path

def load_data(path):
    full_path = Path(__file__).resolve().parent.parent / path
    return pd.read_csv(full_path)

def prepare_features(df):
    df["positivity_rate"] = df["Tests_positive"] / (
        df["Tests_positive"] + df["Tests_negative"]
    )

    df["harvest_per_km2"] = (
        df["Total_harvest"] / df["Area"]
    ).replace([np.inf, -np.inf], 0)

    df["tests_per_km2"] = (
        (df["Tests_positive"] + df["Tests_negative"]) / df["Area"]
    ).replace([np.inf, -np.inf], 0)

    df = df.fillna(0)

    y = df["Management_area_positive"].astype(int)

    drop_cols = [
        "Management_area_ID",
        "Management_area",
        "Administrative_area",
        "Latitude",
        "Longitude",
        "Season_year",
        "Area",
        "Management_area_positive",
    ]

    df_numeric = df.drop(columns=[c for c in drop_cols if c in df.columns])
    df_numeric = df_numeric.select_dtypes(include=["number"]).copy()
    X = df_numeric

    year_str = df["Season_year"].astype(str).str[:4]
    year_num = pd.to_numeric(year_str, errors="coerce")

    train_mask = year_num <= 2019
    val_mask   = year_num == 2020
    test_mask  = year_num >= 2021

    X_train, y_train = X[train_mask], y[train_mask]
    X_val,   y_val   = X[val_mask],   y[val_mask]
    X_test,  y_test  = X[test_mask],  y[test_mask]

    print("Data Split Summary")
    for name, ys in [
        ("Train", y_train),
        ("Val",   y_val),
        ("Test",  y_test),
    ]:
        pos_rate = ys.mean() if len(ys) > 0 else 0
        print(f"{name}: {len(ys)} rows | Pos rate: {pos_rate:.3f}")
    print("Done")

    return X_train, y_train, X_val, y_val, X_test, y_test
