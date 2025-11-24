def set_global_seed(seed=42):
    """
    Set random seeds for numpy, torch, and python's random
    so runs are as reproducible as possible.
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_60_20_20(X, y):
    """
    Split features and labels into 60% train, 20% val, 20% test
    using the original row order (time-respecting split).
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    n = len(X)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)

    X_train = X.iloc[:train_end]
    X_val   = X.iloc[train_end:val_end]
    X_test  = X.iloc[val_end:]

    y_train = y.iloc[:train_end]
    y_val   = y.iloc[train_end:val_end]
    y_test  = y.iloc[val_end:]

    return X_train, X_val, X_test, y_train, y_val, y_test


def summarize_model(name, y_val, y_val_prob, y_test, y_test_prob):
    """
    Compute PR-AUC, ROC-AUC, and F1 / Precision / Recall for
    validation and test, using an F1-optimal threshold chosen
    on the validation set.

    Returns:
        rows: list of two dicts (val row, test row)
        tau:  chosen threshold on validation
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
        tau = 0.5  # fallback

    # Metrics at tau
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
