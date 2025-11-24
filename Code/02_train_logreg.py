from sklearn.feature_selection import VarianceThreshold

def train_logreg(X_train, y_train):
    X_train = X_train.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    vt = VarianceThreshold(threshold=0.0)
    X_train_vt = vt.fit_transform(X_train)

    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train_vt)

    clf = LogisticRegression(penalty="l2", solver="liblinear")
    clf.fit(X_train_std, y_train)

    return clf, scaler, vt


def predict_logreg(model, scaler, vt, X):
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    X_vt = vt.transform(X)
    X_std = scaler.transform(X_vt)
    probs = model.predict_proba(X_std)[:, 1]
    preds = (probs >= 0.5).astype(int)
    return preds, probs


df = load_data(DATA_PATH)
X, y, train_df, val_df, test_df = prepare_features(df)

cut = int(len(X) * 0.8)
X_train, X_test = X.iloc[:cut], X.iloc[cut:]
y_train, y_test = y.iloc[:cut], y.iloc[cut:]

logreg_model, logreg_scaler, logreg_vt = train_logreg(X_train, y_train)
logreg_preds, logreg_probs = predict_logreg(logreg_model, logreg_scaler, logreg_vt, X_test)

precision = precision_score(y_test, logreg_preds)
recall = recall_score(y_test, logreg_preds)
f1 = f1_score(y_test, logreg_preds)
roc_auc = roc_auc_score(y_test, logreg_probs)
ap = average_precision_score(y_test, logreg_probs)
cm = confusion_matrix(y_test, logreg_preds)

print("Logistic Regression results:")
print(f"  Precision: {precision:.3f}")
print(f"  Recall:    {recall:.3f}")
print(f"  F1-score:  {f1:.3f}")
print(f"  ROC AUC:   {roc_auc:.3f}")
print(f"  PR AUC:    {ap:.3f}")
print("  Confusion matrix:")
print(cm)
