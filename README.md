# Explainable Radiogenomic Fusion — Breast Cancer (Synthetic Pilot)

Code accompanying the manuscript *"Explainable Radiogenomic Fusion Links
MRI-Derived Radiomic Phenotypes to Transcriptomic Molecular Subtypes in
Breast Cancer"* (prepared for *Artificial Intelligence in Medicine*).

This repository validates the mechanics of an explainable radiogenomics
framework on a **synthetic, schema-matched paired radiomics–transcriptomics
cohort** (n = 300) constructed so that both imaging features and driver-gene
expression are generated from shared, subtype-characteristic latent biology
(ER, HER2 and proliferation activity). It is a pilot validation, not a
real-data study — see the manuscript's Limitations section and `main.py`'s
`--synthetic` flag.

## Pipeline

1. **Cohort simulation** (`src/data_simulation.py`) — generates 12 MRI
   radiomic features (shape, GLCM texture, DCE-MRI kinetics) and 10
   driver-gene expression values per patient from three latent molecular
   activity scores.
2. **Classification** (`src/classification.py`) — 5-fold cross-validated
   four-class molecular subtype classification from radiomic features alone
   (random forest, gradient boosting, logistic regression, SVM-RBF).
3. **Bootstrap stability** (`src/stability.py`) — 100-resample bootstrap
   permutation-importance analysis deriving a compact, reproducible robust
   radiomic signature.
4. **Radiomic–gene correlation** (`src/correlation.py`) — Pearson
   correlation (BH-FDR corrected) between every radiomic feature and every
   driver gene.

`main.py` orchestrates all four stages end to end and writes every result
table to `outputs/`. `make_figure.py` renders the four-panel summary figure
used in the manuscript (`fig1_radiogenomics.png`).

## Requirements

- Python 3.10+
- See `requirements.txt`

## Usage

```bash
pip install -r requirements.txt
python main.py --synthetic
python make_figure.py
```

This reproduces every number and the figure reported in the manuscript.
Outputs are written to `outputs/`:

| File | Contents |
|---|---|
| `radiomic_features.csv` | Simulated per-patient radiomic feature matrix |
| `driver_gene_expression.csv` | Simulated per-patient driver-gene expression |
| `subtype_labels.csv` | Simulated molecular subtype label per patient |
| `classification_performance.csv` | 5-fold CV performance, full 12-feature panel |
| `confusion_matrix_best_model.csv` | Confusion matrix, best-performing model |
| `radiomic_feature_stability.csv` | Bootstrap stability frequency per feature |
| `signature_classification_performance.csv` | 5-fold CV performance, 6-feature robust signature |
| `radiomic_gene_correlations.csv` | All 120 radiomic feature × gene Pearson correlations |

## Reproducibility

All randomness is seeded (`random_state=42` throughout); re-running
`python main.py --synthetic` regenerates the identical cohort and results.

## Status / next steps

This is a synthetic mechanics-validation pilot. The next step, described in
the manuscript, is applying this identical framework to real paired breast
MRI and transcriptomic data (e.g., TCGA-BRCA imaging linked via The Cancer
Imaging Archive, with expression from GEO or cBioPortal), substituting
`src/data_simulation.py` for a real radiomics-extraction step (e.g.,
PyRadiomics) and real gene-expression quantification.

## License

MIT (see `LICENSE`).

## Citation

If you use this code, please cite the accompanying manuscript (citation
details to be added once published/preprinted).
