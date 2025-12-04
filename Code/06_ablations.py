import numpy as np
from sklearn.ensemble import RandomForestClassifier

def run_rf_ablation(name, drop_cols, X_train, y_train, X_val, y_val):
    X_train_mod = X_train.drop(columns=drop_cols, errors="ignore")
    X_val_mod = X_val.drop(columns=drop_cols, errors="ignore")

    rf = RandomForestClassifier(
        n_estimators=500,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train_mod, y_train)

    val_probs = rf.predict_proba(X_val_mod)[:, 1]
    return val_probs
