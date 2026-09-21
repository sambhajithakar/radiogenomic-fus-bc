"""
End-to-end radiogenomics pipeline: paired synthetic radiomics-transcriptomics
cohort -> multiclass molecular subtype classification from radiomics alone
-> bootstrap-stable explainable-AI radiomic feature selection -> radiomic-
gene correlation analysis linking imaging phenotypes to transcriptomic
drivers.

Usage:
    python main.py --synthetic
"""
from __future__ import annotations
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data_simulation import simulate_radiogenomic_cohort, RADIOMIC_FEATURES, DRIVER_GENES
from classification import cross_validated_benchmark
from stability import bootstrap_feature_stability, derive_robust_signature
from correlation import radiomic_gene_correlations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--out-dir", type=str, default="outputs")
    parser.add_argument("--n-bootstrap", type=int, default=100)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("[1/5] Simulating paired radiogenomic cohort...")
    rad, gene, labels, latent = simulate_radiogenomic_cohort(n_patients=300, random_state=42)
    print(f"  {rad.shape[0]} patients; subtype counts:\n{labels.value_counts().to_string()}")
    rad.to_csv(os.path.join(args.out_dir, "radiomic_features.csv"))
    gene.to_csv(os.path.join(args.out_dir, "driver_gene_expression.csv"))
    labels.to_csv(os.path.join(args.out_dir, "subtype_labels.csv"))

    print("[2/5] Benchmarking multiclass subtype classifiers on radiomic features (5-fold CV)...")
    perf_df, cm_df, best_name = cross_validated_benchmark(rad, labels)
    perf_df.to_csv(os.path.join(args.out_dir, "classification_performance.csv"), index=False)
    cm_df.to_csv(os.path.join(args.out_dir, "confusion_matrix_best_model.csv"))
    print(perf_df.to_string(index=False))
    print(f"  Best model: {best_name}")
    print(cm_df.to_string())

    print(f"[3/5] Bootstrap stability analysis of radiomic features ({args.n_bootstrap} resamples)...")
    stability_df = bootstrap_feature_stability(rad, labels, top_k=8, n_bootstrap=args.n_bootstrap)
    stability_df.to_csv(os.path.join(args.out_dir, "radiomic_feature_stability.csv"))
    signature = derive_robust_signature(stability_df, stability_threshold=0.70)
    print(f"  Robust radiomic signature ({len(signature)} features, >=70% bootstrap stability): {signature}")

    print("[4/5] Validating robust radiomic signature classification performance...")
    rad_sig = rad[signature]
    sig_perf_df, sig_cm_df, sig_best_name = cross_validated_benchmark(rad_sig, labels)
    sig_perf_df.to_csv(os.path.join(args.out_dir, "signature_classification_performance.csv"), index=False)
    print(sig_perf_df.to_string(index=False))

    print("[5/5] Radiomic-gene correlation analysis (radiogenomic mapping)...")
    corr_df = radiomic_gene_correlations(rad, gene)
    corr_df.to_csv(os.path.join(args.out_dir, "radiomic_gene_correlations.csv"), index=False)
    sig_corr = corr_df[corr_df["p_adj"] < 0.05]
    print(f"  {len(sig_corr)}/{len(corr_df)} radiomic-gene pairs significant after BH-FDR correction (p_adj<0.05)")
    print(sig_corr.head(20).to_string(index=False))

    print(f"\nDone. Outputs in: {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()
