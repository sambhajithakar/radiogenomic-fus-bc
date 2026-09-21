"""
Radiogenomic correlation analysis: Pearson correlation between each radiomic
feature and each driver gene's expression, with Benjamini-Hochberg false
discovery rate correction across all feature-gene pairs tested. This is the
step that turns a subtype classifier into a genuinely "radiogenomic" result
-- showing which specific imaging phenotypes track which specific
transcriptomic drivers, not just that imaging predicts a subtype label.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats


def _bh_correction(p_values: np.ndarray) -> np.ndarray:
    n = len(p_values)
    order = np.argsort(p_values)
    ranked = p_values[order]
    adj = ranked * n / (np.arange(n) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    out = np.empty(n)
    out[order] = adj
    return out


def radiomic_gene_correlations(rad: pd.DataFrame, gene: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rfeat in rad.columns:
        for gfeat in gene.columns:
            r, p = stats.pearsonr(rad[rfeat], gene[gfeat])
            rows.append({"radiomic_feature": rfeat, "gene": gfeat, "pearson_r": r, "p_value": p})
    df = pd.DataFrame(rows)
    df["p_adj"] = _bh_correction(df["p_value"].values)
    return df.sort_values("p_adj")
