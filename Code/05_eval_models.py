# Helper: summarize a model on val + test
def summarize_model(name, y_val, y_val_prob, y_test, y_test_prob):
    """
    Compute PR-AUC, ROC-AUC, and F1 / Precision / Recall for
    validation and test, using an F1-optimal threshold chosen
    on the validation set.

    Returns:
        rows: [val_row_dict, test_row_dict]
        tau: chosen threshold on validation
    """

    # Threshold-independent metrics
    pr_auc_val = average_precision_score(y_val, y_val_prob)
    pr_auc_test = average_precision_score(y_test, y_test_prob)
    roc_auc_val = roc_auc_score(y_val, y_val_prob)
    roc_auc_test = roc_auc_score(y_test, y_test_prob)

    # F1-optimal threshold on validation
    prec, rec, thr = precision_recall_curve(y_val, y_val_prob)
    f1s = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-8)

    if len(f1s) > 0:
        best_idx = int(np.argmax(f1s))
        tau = float(thr[best_idx])
    else:
        tau = 0.5  # fallback if something weird happens

    # Metrics at tau on val + test
    val_pred = (y_val_prob >= tau).astype(int)
    test_pred = (y_test_prob >= tau).astype(int)

    row_val = dict(
        model=name,
        split="val",
        tau=tau,
        pr_auc=pr_auc_val,
        roc_auc=roc_auc_val,
        precision=precision_score(y_val, val_pred, zero_division=0),
        recall=recall_score(y_val, val_pred, zero_division=0),
        f1=f1_score(y_val, val_pred, zero_division=0),
        support=int(y_val.sum()),
    )

    row_test = dict(
        model=name,
        split="test",
        tau=tau,
        pr_auc=pr_auc_test,
        roc_auc=roc_auc_test,
        precision=precision_score(y_test, test_pred, zero_division=0),
        recall=recall_score(y_test, test_pred, zero_division=0),
        f1=f1_score(y_test, test_pred, zero_division=0),
        support=int(y_test.sum()),
    )

    return [row_val, row_test], tau



# Load data and make a consistent 60 / 20 / 20 split

df = load_data(DATA_PATH)
X, y, train_df, val_df, test_df = prepare_features(df)

n = len(X)
train_end = int(n * 0.6)
val_end = int(n * 0.8)

X_train, X_val, X_test = (
    X.iloc[:train_end],
    X.iloc[train_end:val_end],
    X.iloc[val_end:],
)
y_train, y_val, y_test = (
    y.iloc[:train_end],
    y.iloc[train_end:val_end],
    y.iloc[val_end:],
)


# Train / score each model


# Logistic regression
logreg_model, logreg_scaler, logreg_vt = train_logreg(X_train, y_train)
_, y_val_prob_lr = predict_logreg(logreg_model, logreg_scaler, logreg_vt, X_val)
_, y_test_prob_lr = predict_logreg(logreg_model, logreg_scaler, logreg_vt, X_test)

# Random forest
rf_model = train_rf(X_train, y_train)
y_val_prob_rf = predict_rf(rf_model, X_val)
y_test_prob_rf = predict_rf(rf_model, X_test)

# MLP (improved tabular MLP)
mlp_model, mlp_scaler, mlp_threshold, mlp_test_probs = train_mlp(
    X_train, y_train, X_val, y_val, X_test, y_test
)

# Recompute validation probabilities for the trained MLP
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_val_proc = X_val.apply(pd.to_numeric, errors="coerce").fillna(0.0)
X_val_scaled = mlp_scaler.transform(X_val_proc)
X_val_t = torch.tensor(X_val_scaled, dtype=torch.float32).to(DEVICE)

with torch.no_grad():
    val_logits_mlp = mlp_model(X_val_t)
    y_val_prob_mlp = torch.sigmoid(val_logits_mlp).cpu().numpy()

y_test_prob_mlp = mlp_test_probs  # already probs on test from train_mlp


# Build comparison table

rows = []
taus = {}

for name, (vprob, tprob) in {
    "LogReg (scaled)": (y_val_prob_lr, y_test_prob_lr),
    "Random Forest": (y_val_prob_rf, y_test_prob_rf),
    "MLP (scaled)": (y_val_prob_mlp, y_test_prob_mlp),
}.items():
    model_rows, tau = summarize_model(name, y_val, vprob, y_test, tprob)
    rows.extend(model_rows)
    taus[name] = tau

metrics_df = pd.DataFrame(rows)
metrics_df = metrics_df[
    ["model", "split", "tau", "pr_auc", "roc_auc", "precision", "recall", "f1", "support"]
]

print("=== Model comparison (validation + test) ===")
display(metrics_df.round(3))

print("\nChosen F1-optimal thresholds by model:")
for name, tau in taus.items():
    print(f"  {name}: τ* = {tau:.3f}")
