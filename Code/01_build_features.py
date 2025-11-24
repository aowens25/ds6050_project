DATA_PATH = "../data/NA_wildlife_agency_CWD_surveillance_data_2000_2022_v2.csv"

def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    df = df.sort_values(by="Season_year").reset_index(drop=True)
    return df

def prepare_features(df):
    df["positivity_rate"] = df["Tests_positive"] / (
        df["Tests_positive"] + df["Tests_negative"]
    )
    df["harvest_per_km2"] = (df["Total_harvest"] / df["Area"]).replace([np.inf, -np.inf], 0)
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

    train_df = df[year_num <= 2020]
    val_df = df[year_num == 2021]
    test_df = df[year_num >= 2022]

    print("Data Split Summary")
    for name, subset in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        pos_rate = subset["Management_area_positive"].mean()
        print(f"{name}: {len(subset)} rows | Pos rate: {pos_rate:.3f}")
    print("Done")

    return X, y, train_df, val_df, test_df

df = load_data()
X, y, train_df, val_df, test_df = prepare_features(df)
