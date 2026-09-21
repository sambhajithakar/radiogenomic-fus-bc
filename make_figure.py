import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "outputs"

cm = pd.read_csv(f"{OUT}/confusion_matrix_best_model.csv", index_col=0)
stab = pd.read_csv(f"{OUT}/radiomic_feature_stability.csv", index_col=0)
perf_full = pd.read_csv(f"{OUT}/classification_performance.csv")
perf_sig = pd.read_csv(f"{OUT}/signature_classification_performance.csv")
corr = pd.read_csv(f"{OUT}/radiomic_gene_correlations.csv")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# A: confusion matrix heatmap (best model, full feature set)
ax = axes[0, 0]
cm_norm = cm.div(cm.sum(axis=1), axis=0)
im = ax.imshow(cm_norm.values, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(len(cm.columns))); ax.set_xticklabels(cm.columns, rotation=30, ha="right", fontsize=8)
ax.set_yticks(range(len(cm.index))); ax.set_yticklabels(cm.index, fontsize=8)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, str(cm.values[i, j]), ha="center", va="center",
                 color="white" if cm_norm.values[i, j] > 0.5 else "black", fontsize=9)
ax.set_xlabel("Predicted subtype"); ax.set_ylabel("True subtype")
ax.set_title("(A) Confusion matrix (best model: LogisticRegression)")
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Row-normalized fraction")

# B: bootstrap stability
ax = axes[0, 1]
stab_sorted = stab.sort_values("stability_frequency", ascending=True)
colors = ["#2e8b57" if v >= 0.70 else "#b0b0b0" for v in stab_sorted["stability_frequency"]]
ax.barh(stab_sorted.index, stab_sorted["stability_frequency"], color=colors)
ax.axvline(0.70, ls="--", color="#c0392b", lw=1.2, label="Stability threshold (0.70)")
ax.set_xlabel("Bootstrap selection frequency (100 resamples)")
ax.set_title("(B) Radiomic feature stability")
ax.legend(fontsize=8, loc="lower right")
ax.tick_params(axis='y', labelsize=8)

# C: classification performance, full panel vs robust signature
ax = axes[1, 0]
models = perf_full["model"].tolist()
x = np.arange(len(models))
w = 0.35
full_auc = perf_full.set_index("model").loc[models, "auc_macro_ovr"].values
sig_auc = perf_sig.set_index("model").loc[models, "auc_macro_ovr"].values
ax.bar(x - w/2, full_auc, w, label="Full panel (12 radiomic features)", color="#4c72b0")
ax.bar(x + w/2, sig_auc, w, label="Robust signature (6 features)", color="#dd8452")
ax.set_xticks(x); ax.set_xticklabels(models, rotation=15, ha="right", fontsize=8)
ax.set_ylim(0.9, 1.0)
ax.set_ylabel("Macro-average AUC (one-vs-rest)")
ax.set_title("(C) Subtype classification: full panel vs. robust signature")
ax.legend(fontsize=8, loc="lower right")

# D: correlation heatmap, robust signature features x driver genes
ax = axes[1, 1]
sig_feats = stab[stab["stability_frequency"] >= 0.70].index.tolist()
genes = ["ESR1", "PGR", "GATA3", "FOXA1", "ERBB2", "GRB7", "MKI67", "KRT5", "KRT14", "EGFR"]
mat = corr.pivot(index="radiomic_feature", columns="gene", values="pearson_r").loc[sig_feats, genes]
im2 = ax.imshow(mat.values, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(genes))); ax.set_xticklabels(genes, rotation=45, ha="right", fontsize=8)
ax.set_yticks(range(len(sig_feats))); ax.set_yticklabels(sig_feats, fontsize=8)
for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        ax.text(j, i, f"{mat.values[i,j]:.2f}", ha="center", va="center",
                 color="white" if abs(mat.values[i, j]) > 0.6 else "black", fontsize=6.5)
ax.set_title("(D) Robust radiomic signature x driver-gene correlation (Pearson r)")
plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.04, label="Pearson r")

plt.tight_layout()
plt.savefig("fig1_radiogenomics.png", dpi=200, bbox_inches="tight")
print("saved fig1_radiogenomics.png")
