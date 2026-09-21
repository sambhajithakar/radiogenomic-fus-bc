"""
Bootstrap feature-selection stability analysis for the radiomic feature set,
following the same bootstrap-stability methodology used elsewhere in this
manuscript series: refit a classifier on many resamples of the cohort,
compute permutation feature importance in each resample, and retain only
features that rank among the most important in a large majority of
resamples. Applied here to radiomic features rather than genes, to derive a
"robust radiomic signature" whose relevance to subtype classification is
not an artifact of one particular train/test split.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils import resample


def bootstrap_feature_stability(X: pd.DataFrame, y: pd.Series, top_k: int = 8, n_bootstrap: int = 100, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    le = LabelEncoder()
    y_enc = pd.Series(le.fit_transform(y), index=y.index)

    counts = pd.Series(0, index=X.columns, dtype=float)
    mean_importance = pd.Series(0.0, index=X.columns)

    for b in range(n_bootstrap):
        seed = int(rng.integers(0, 1_000_000))
        X_bs, y_bs = resample(X, y_enc, replace=True, stratify=y_enc, random_state=seed)
        scaler = StandardScaler()
        X_bs_scaled = pd.DataFrame(scaler.fit_transform(X_bs), columns=X.columns, index=X_bs.index)

        model = RandomForestClassifier(n_estimators=150, max_depth=6, random_state=seed, n_jobs=-1)
        model.fit(X_bs_scaled, y_bs)
        perm = permutation_importance(model, X_bs_scaled, y_bs, n_repeats=3, random_state=seed, scoring="accuracy")
        imp = pd.Series(perm.importances_mean, index=X.columns)
        mean_importance += imp

        top_feats = imp.sort_values(ascending=False).head(top_k).index
        counts.loc[top_feats] += 1

    stability = (counts / n_bootstrap).sort_values(ascending=False)
    mean_importance /= n_bootstrap
    out = pd.DataFrame({
        "stability_frequency": stability,
        "mean_importance": mean_importance.loc[stability.index],
    })
    return out


def derive_robust_signature(stability_df: pd.DataFrame, stability_threshold: float = 0.70) -> list[str]:
    return stability_df[stability_df["stability_frequency"] >= stability_threshold].index.tolist()
