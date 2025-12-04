from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)
import numpy as np

def summarize_model(name, y_val, val_probs, y_test, test_probs):
    results = {}

    if val_probs is not None:
        val_preds = (val_probs >= 0.5).astype(int)
        results["val_precision"] = precision_score(y_val, val_preds)
        results["val_recall"] = recall_score(y_val, val_preds)
        results["val_f1"] = f1_score(y_val, val_preds)
        results["val_auc"] = roc_auc_score(y_val, val_probs)
        results["val_pr_auc"] = average_precision_score(y_val, val_probs)
        tau = np.percentile(val_probs, 90)
    else:
        tau = None

    test_preds = (test_probs >= 0.5).astype(int)
    results["test_precision"] = precision_score(y_test, test_preds)
    results["test_recall"] = recall_score(y_test, test_preds)
    results["test_f1"] = f1_score(y_test, test_preds)
    results["test_auc"] = roc_auc_score(y_test, test_probs)
    results["test_pr_auc"] = average_precision_score(y_test, test_probs)

    return results, tau
