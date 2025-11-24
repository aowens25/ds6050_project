# Reload data so this cell is self-contained
df = load_data(DATA_PATH)
X_full, y_full, train_df, val_df, test_df = prepare_features(df)



# Define feature groups by name patterns
cols = X_full.columns

harvest_cols = [c for c in cols if "harvest" in c.lower()]
movement_cols = [
    c for c in cols
    if any(k in c.lower() for k in ["movement", "import_of_live_cervids"])
]
policy_cols = [
    c for c in cols
    if any(k in c.lower() for k in [
        "baiting", "feeding", "lures", "cervid_facilities",
        "hunting_enclosures", "processors", "taxidermists"
    ])
]
environment_cols = [
    c for c in cols
    if any(k in c.lower() for k in ["forest_cover", "streams", "clay"])
]

print("Feature group sizes:")
print(f"  harvest:      {len(harvest_cols)}")
print(f"  movement:     {len(movement_cols)}")
print(f"  policy / infra: {len(policy_cols)}")
print(f"  environment:  {len(environment_cols)}")


# Helper: 60 / 20 / 20 split
def split_60_20_20(X, y):
    n = len(X)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)
    X_train, X_val, X_test = X.iloc[:train_end], X.iloc[train_end:val_end], X.iloc[val_end:]
    y_train, y_val, y_test = y.iloc[:train_end], y.iloc[train_end:val_end], y.iloc[val_end:]
    return X_train, X_val, X_test, y_train, y_val, y_test



# Helper: run one RF ablation and summarize
def run_rf_ablation(name, drop_cols):
    if drop_cols:
        X = X_full.drop(columns=drop_cols)
    else:
        X = X_full.copy()

    X_train, X_val, X_test, y_train, y_val, y_test = split_60_20_20(X, y_full)

    rf_model = train_rf(X_train, y_train)
    y_val_prob = predict_rf(rf_model, X_val)
    y_test_prob = predict_rf(rf_model, X_test)

    rows, tau = summarize_model(name, y_val, y_val_prob, y_test, y_test_prob)
    print(f"\n[{name}] used {X.shape[1]} features, τ* (val F1-optimal) = {tau:.3f}")
    return rows


# Run ablations
ablation_specs = [
    ("RF_baseline_all_features", []),
    ("RF_drop_environment", environment_cols),
    ("RF_drop_movement", movement_cols),
    ("RF_drop_harvest", harvest_cols),
]

ablation_rows = []
for name, drop_cols in ablation_specs:
    ablation_rows.extend(run_rf_ablation(name, drop_cols))

ablation_df = pd.DataFrame(ablation_rows)
ablation_df = ablation_df[
    ["model", "split", "tau", "pr_auc", "roc_auc", "precision", "recall", "f1", "support"]
]

print("\nRandom Forest feature ablation results")
display(ablation_df.round(3))
