import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
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
