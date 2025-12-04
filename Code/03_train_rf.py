import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

def train_rf(X_train, y_train):
    X_train = X_train.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)
    return clf

def predict_rf(model, X):
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return model.predict_proba(X)[:, 1]
