"""
Synthetic paired radiomics-transcriptomics (radiogenomics) breast cancer
cohort generator.

Design: each simulated patient has three latent, continuous molecular
activity scores -- ER_activity, HER2_activity and PROLIFERATION_activity --
drawn from subtype-characteristic distributions that mirror the well-
documented intrinsic-subtype biology (luminal tumors ER-high; HER2-enriched
tumors HER2-high; basal-like tumors ER-low/proliferation-high) [Perou 2000;
Sorlie 2001; Parker 2009 PAM50; TCGA 2012 -- see manuscript references].
From these three latent scores we generate two observed, noisy views:

  1. A small panel of driver-gene expression values (ESR1, PGR, GATA3,
     FOXA1 for luminal biology; ERBB2, GRB7 for HER2 biology; MKI67, KRT5,
     KRT14, EGFR for proliferative/basal biology) -- the "genomics" side.
  2. A panel of twelve quantitative MRI radiomic features spanning the
     standard radiomics feature families used in breast MRI radiogenomics
     studies -- shape, first-order intensity, GLCM texture, and DCE-MRI
     kinetic features -- each generated as a noisy, partially redundant
     function of the same three latent scores plus feature-specific
     nuisance variance -- the "radiomics" side.

Because both views are generated from the same latent biology but through
different, independently noisy generative paths, this cohort has a known,
checkable ground truth for (a) how well radiomics alone can recover
molecular subtype, and (b) which specific radiomic features should
correlate with which specific driver genes -- exactly the two questions a
radiogenomics framework needs to answer, without requiring real paired
imaging-genomic data for this pilot validation.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

SUBTYPES = ["Luminal A", "Luminal B", "HER2-enriched", "Basal-like"]

DRIVER_GENES = ["ESR1", "PGR", "GATA3", "FOXA1", "ERBB2", "GRB7", "MKI67", "KRT5", "KRT14", "EGFR"]

RADIOMIC_FEATURES = [
    "shape_irregularity", "margin_spiculation", "size_mm",
    "T2_intensity_variance", "ADC_mean",
    "glcm_contrast", "glcm_entropy", "glcm_homogeneity",
    "wash_in_slope", "wash_out_ratio", "peak_enhancement", "enhancement_heterogeneity",
]

# Subtype-characteristic means for the three latent activity scores
# (ER_activity, HER2_activity, PROLIFERATION_activity), each roughly z-scored.
SUBTYPE_PROFILE = {
    "Luminal A":     dict(ER=1.3,  HER2=-0.5, PROLIF=-0.8),
    "Luminal B":     dict(ER=1.0,  HER2=-0.2, PROLIF=0.9),
    "HER2-enriched": dict(ER=-0.6, HER2=1.5,  PROLIF=0.7),
    "Basal-like":    dict(ER=-1.4, HER2=-0.4, PROLIF=1.2),
}
SUBTYPE_PRIOR = {"Luminal A": 0.40, "Luminal B": 0.20, "HER2-enriched": 0.15, "Basal-like": 0.25}


def simulate_radiogenomic_cohort(n_patients: int = 300, random_state: int = 42, latent_noise_sd: float = 0.55):
    rng = np.random.default_rng(random_state)
    subtypes = rng.choice(list(SUBTYPE_PRIOR.keys()), size=n_patients, p=list(SUBTYPE_PRIOR.values()))

    ER = np.array([SUBTYPE_PROFILE[s]["ER"] for s in subtypes]) + rng.normal(0, latent_noise_sd, n_patients)
    HER2 = np.array([SUBTYPE_PROFILE[s]["HER2"] for s in subtypes]) + rng.normal(0, latent_noise_sd, n_patients)
    PROLIF = np.array([SUBTYPE_PROFILE[s]["PROLIF"] for s in subtypes]) + rng.normal(0, latent_noise_sd, n_patients)

    patient_ids = [f"P{i:04d}" for i in range(n_patients)]

    # ---- Driver gene expression (log2-intensity-like), function of latent scores ----
    gene = pd.DataFrame(index=patient_ids)
    gene["ESR1"] = 6.0 + 1.1 * ER + rng.normal(0, 0.4, n_patients)
    gene["PGR"] = 6.0 + 0.9 * ER + rng.normal(0, 0.5, n_patients)
    gene["GATA3"] = 6.0 + 0.8 * ER - 0.2 * PROLIF + rng.normal(0, 0.4, n_patients)
    gene["FOXA1"] = 6.0 + 0.85 * ER + rng.normal(0, 0.4, n_patients)
    gene["ERBB2"] = 6.0 + 1.2 * HER2 + rng.normal(0, 0.4, n_patients)
    gene["GRB7"] = 6.0 + 1.0 * HER2 + rng.normal(0, 0.45, n_patients)
    gene["MKI67"] = 6.0 + 1.1 * PROLIF + rng.normal(0, 0.4, n_patients)
    gene["KRT5"] = 6.0 - 0.7 * ER + 0.5 * PROLIF + rng.normal(0, 0.5, n_patients)
    gene["KRT14"] = 6.0 - 0.6 * ER + 0.45 * PROLIF + rng.normal(0, 0.5, n_patients)
    gene["EGFR"] = 6.0 - 0.5 * ER + 0.6 * PROLIF + rng.normal(0, 0.5, n_patients)

    # ---- Radiomic features (mixed units, z-scored downstream), function of latent scores ----
    rad = pd.DataFrame(index=patient_ids)
    rad["shape_irregularity"] = 0.5 + 0.30 * PROLIF - 0.15 * ER + rng.normal(0, 0.35, n_patients)
    rad["margin_spiculation"] = 0.5 + 0.28 * PROLIF - 0.22 * ER + rng.normal(0, 0.35, n_patients)
    rad["size_mm"] = 22.0 + 2.0 * PROLIF + rng.normal(0, 6.5, n_patients)  # weak signal, mostly noise
    rad["T2_intensity_variance"] = 40.0 + 6.0 * PROLIF + rng.normal(0, 7.0, n_patients)
    rad["ADC_mean"] = 1.10 - 0.12 * PROLIF + rng.normal(0, 0.10, n_patients)  # restricted diffusion -> lower ADC
    rad["glcm_contrast"] = 15.0 + 3.2 * PROLIF + rng.normal(0, 3.0, n_patients)
    rad["glcm_entropy"] = 4.0 + 0.35 * PROLIF - 0.10 * ER + rng.normal(0, 0.35, n_patients)
    rad["glcm_homogeneity"] = 0.60 - 0.09 * PROLIF + rng.normal(0, 0.08, n_patients)
    rad["wash_in_slope"] = 3.0 + 0.55 * PROLIF + 0.35 * HER2 + rng.normal(0, 0.6, n_patients)
    rad["wash_out_ratio"] = 0.15 + 0.30 * HER2 + 0.10 * PROLIF + rng.normal(0, 0.12, n_patients)
    rad["peak_enhancement"] = 180.0 + 22.0 * HER2 + 8.0 * PROLIF + rng.normal(0, 18.0, n_patients)
    rad["enhancement_heterogeneity"] = 0.35 + 0.20 * PROLIF - 0.10 * ER + rng.normal(0, 0.10, n_patients)

    labels = pd.Series(subtypes, index=patient_ids, name="subtype")
    latent = pd.DataFrame({"ER_activity": ER, "HER2_activity": HER2, "PROLIF_activity": PROLIF}, index=patient_ids)

    return rad, gene, labels, latent
