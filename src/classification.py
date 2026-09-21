"""
Four-class molecular subtype classification from radiomic features alone,
benchmarked by five-fold stratified cross-validation.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix


def get_models(random_state: int = 42) -> dict:
    return {
        "RandomForest": RandomForestClassifier(n_estimators=300, max_depth=6, random_state=random_state, n_jobs=-1),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=random_state),
        "LogisticRegression": LogisticRegression(max_iter=3000, C=0.5),
        "SVM_RBF": SVC(kernel="rbf", probability=True, C=1.0, random_state=random_state),
    }


def cross_validated_benchmark(X: pd.DataFrame, y: pd.Series, n_folds: int = 5, random_state: int = 42):
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    rows = []
    best_proba, best_name, best_auc = None, None, -1
    for name, model in get_models(random_state).items():
        proba = cross_val_predict(model, X_scaled, y_enc, cv=skf, method="predict_proba")
        preds = np.argmax(proba, axis=1)
        auc = roc_auc_score(y_enc, proba, multi_class="ovr", average="macro")
        rows.append({
            "model": name,
            "auc_macro_ovr": auc,
            "accuracy": accuracy_score(y_enc, preds),
            "precision_macro": precision_score(y_enc, preds, average="macro"),
            "recall_macro": recall_score(y_enc, preds, average="macro"),
            "f1_macro": f1_score(y_enc, preds, average="macro"),
        })
        if auc > best_auc:
            best_auc, best_proba, best_name = auc, proba, name

    perf_df = pd.DataFrame(rows).sort_values("auc_macro_ovr", ascending=False).reset_index(drop=True)
    best_preds = np.argmax(best_proba, axis=1)
    cm = confusion_matrix(y_enc, best_preds)
    cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)
    return perf_df, cm_df, best_name
