import yaml
from pathlib import Path
import torch

# Import your module functions (each script already has its own imports)
from build_features import load_data, prepare_features
from train_logreg import train_logreg, predict_logreg
from train_rf import train_rf, predict_rf
from train_mlp import train_mlp
from eval_models import summarize_model
from ablations import run_rf_ablation
from utils import set_global_seed
import visuals

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    set_global_seed(42)

    # 1. Load config + data
    cfg = load_config()
    df = load_data(cfg["data"]["path"])
    X, y, train_df, val_df, test_df = prepare_features(df)

    # 2. Split data
    n = len(X)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)
    X_train, X_val, X_test = X.iloc[:train_end], X.iloc[train_end:val_end], X.iloc[val_end:]
    y_train, y_val, y_test = y.iloc[:train_end], y.iloc[train_end:val_end], y.iloc[val_end:]

    # 3. Train models
    print("\nTraining Logistic Regression...")
    logreg_model, logreg_scaler, logreg_vt = train_logreg(X_train, y_train)
    _, y_val_prob_lr = predict_logreg(logreg_model, logreg_scaler, logreg_vt, X_val)
    _, y_test_prob_lr = predict_logreg(logreg_model, logreg_scaler, logreg_vt, X_test)

    print("\nTraining Random Forest...")
    rf_model = train_rf(X_train, y_train)
    y_val_prob_rf = predict_rf(rf_model, X_val)
    y_test_prob_rf = predict_rf(rf_model, X_test)

    print("\nTraining MLP...")
    mlp_model, mlp_scaler, mlp_thresh, y_test_prob_mlp, mlp_train_losses, mlp_val_losses = train_mlp(
        X_train, y_train, X_val, y_val, X_test, y_test
    )

    # 4. Evaluate
    print("\nEvaluating Models...")
    models = {
        "Logistic Regression": (y_val_prob_lr, y_test_prob_lr),
        "Random Forest": (y_val_prob_rf, y_test_prob_rf),
        "MLP": (None, y_test_prob_mlp),  # Val probs recomputed if needed
    }

    for name, (val_probs, test_probs) in models.items():
        rows, tau = summarize_model(name, y_val, val_probs or y_val_prob_rf, y_test, test_probs)
        print(f"{name}: τ* = {tau:.3f}")

    # 5. Run visuals (plots training curves, comparisons, calibration, etc.)
    print("\nGenerating visualizations...")
    visuals.run_all_visuals(
        y_test,
        y_test_prob_lr,
        y_test_prob_rf,
        y_test_prob_mlp,
        mlp_train_losses,
        mlp_val_losses,
    )

    
    print("\nRunning Random Forest Ablations...")
    run_rf_ablation("RF_baseline_all_features", [])

if __name__ == "__main__":
    main()