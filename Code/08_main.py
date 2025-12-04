import yaml
import importlib
import torch
from pathlib import Path

build_features = importlib.import_module("01_build_features")
train_logreg = importlib.import_module("02_train_logreg")
train_rf = importlib.import_module("03_train_rf")
train_mlp = importlib.import_module("04_train_mlp")
eval_models = importlib.import_module("05_eval_models")
ablations = importlib.import_module("06_ablations")
utils = importlib.import_module("07_utils")
visuals = importlib.import_module("09_visuals")

def load_config(path="config.yaml"):
    root = Path(__file__).resolve().parent.parent
    return yaml.safe_load(open(root / path, "r"))

def main():
    utils.set_global_seed(42)

    cfg = load_config()
    df = build_features.load_data(cfg["data"]["path"])
    X, y, train_df, val_df, test_df = build_features.prepare_features(df)

    n = len(X)
    a = int(n * 0.6)
    b = int(n * 0.8)

    X_train = X.iloc[:a]
    y_train = y.iloc[:a]

    X_val = X.iloc[a:b]
    y_val = y.iloc[a:b]

    X_test = X.iloc[b:]
    y_test = y.iloc[b:]

    lr_model, lr_scaler, lr_vt = train_logreg.train_logreg(X_train, y_train)
    _, y_val_prob_lr = train_logreg.predict_logreg(lr_model, lr_scaler, lr_vt, X_val)
    _, y_test_prob_lr = train_logreg.predict_logreg(lr_model, lr_scaler, lr_vt, X_test)

    rf_model = train_rf.train_rf(X_train, y_train)
    y_val_prob_rf = train_rf.predict_rf(rf_model, X_val)
    y_test_prob_rf = train_rf.predict_rf(rf_model, X_test)

    mlp_model, mlp_scaler, mlp_vt, y_test_prob_mlp, mlp_train_losses, mlp_val_losses = train_mlp.train_mlp(
        X_train, y_train, X_val, y_val, X_test, y_test
    )

    metrics_lr, tau_lr = eval_models.summarize_model(
        "logreg",
        y_val.astype(int),
        y_val_prob_lr,
        y_test.astype(int),
        y_test_prob_lr
    )

    metrics_rf, tau_rf = eval_models.summarize_model(
        "rf",
        y_val.astype(int),
        y_val_prob_rf,
        y_test.astype(int),
        y_test_prob_rf
    )

    metrics_mlp, tau_mlp = eval_models.summarize_model(
        "mlp",
        y_val.astype(int),
        None,
        y_test.astype(int),
        y_test_prob_mlp
    )

    visuals.run_all_visuals(
        y_test.astype(int),
        y_test_prob_lr,
        y_test_prob_rf,
        y_test_prob_mlp,
        mlp_train_losses,
        mlp_val_losses
    )

if __name__ == "__main__":
    main()
