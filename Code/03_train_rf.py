def train_rf(X_train, y_train):
    X_train = X_train.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    return rf


def predict_rf(model, X):
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    probs = model.predict_proba(X)[:, 1]
    return probs


df = load_data(DATA_PATH)
X, y, train_df, val_df, test_df = prepare_features(df)

n = len(X)
train_end = int(n * 0.6)
val_end = int(n * 0.8)

X_train, X_val, X_test = X.iloc[:train_end], X.iloc[train_end:val_end], X.iloc[val_end:]
y_train, y_val, y_test = y.iloc[:train_end], y.iloc[train_end:val_end], y.iloc[val_end:]

rf_model = train_rf(X_train, y_train)

val_probs = predict_rf(rf_model, X_val)
val_pr_auc = average_precision_score(y_val, val_probs)
val_roc_auc = roc_auc_score(y_val, val_probs)

precisions, recalls, thresholds = precision_recall_curve(y_val, val_probs)
f1_scores = 2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1] + 1e-8)

if len(f1_scores) > 0:
    best_idx = int(np.argmax(f1_scores))
    best_thresh = float(thresholds[best_idx])
    val_f1 = float(f1_scores[best_idx])
    val_prec = float(precisions[best_idx])
    val_rec = float(recalls[best_idx])
else:
    best_thresh = 0.5
    val_pred = (val_probs >= best_thresh).astype(int)
    val_prec = precision_score(y_val, val_pred, zero_division=0)
    val_rec = recall_score(y_val, val_pred, zero_division=0)
    val_f1 = f1_score(y_val, val_pred, zero_division=0)

print("Validation (Random Forest):")
print(f"  PR-AUC = {val_pr_auc:.3f}, ROC-AUC = {val_roc_auc:.3f}")
print(f"  Best F1 = {val_f1:.3f} at threshold τ = {best_thresh:.3f}")
print(f"    Precision = {val_prec:.3f}, Recall = {val_rec:.3f}")

test_probs = predict_rf(rf_model, X_test)
test_pr_auc = average_precision_score(y_test, test_probs)
test_roc_auc = roc_auc_score(y_test, test_probs)
test_pred = (test_probs >= best_thresh).astype(int)
test_prec = precision_score(y_test, test_pred, zero_division=0)
test_rec = recall_score(y_test, test_pred, zero_division=0)
test_f1 = f1_score(y_test, test_pred, zero_division=0)

print(f"\nTest (Random Forest, τ = {best_thresh:.3f} from validation):")
print(f"  PR-AUC = {test_pr_auc:.3f}, ROC-AUC = {test_roc_auc:.3f}")
print(f"  F1 = {test_f1:.3f}  (Precision = {test_prec:.3f}, Recall = {test_rec:.3f})")
