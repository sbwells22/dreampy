"""
sensitivity_perez2022.py — dreampy vs dreampy sensitivity run.

Runs the full dreampy pipeline on the Perez 2022 SLE immune dataset while 
excluding the ImmVar samples to replicate the paper's analysis. Then compares 
concordance metrics against the validate_perez2022.py values (full dataset).

Pipeline order (canonical):
  pseudobulk -> filter_samples -> TMM -> filter_by_expr -> log2CPM -> voom ->
  fit (Satterthwaite) -> eBayes

Formula:  ~ sle + (1 | donor_id) + (1 | Processing_Cohort)
Dataset:  Perez et al. 2022 SLE
Assays:   ~10 Immune cell subtypes (auto-discovered from reference directory)

Outputs:
  - {assay_name}_py_sensitivity_results.csv: per-assay DE results.
  - sensitivity_concordance_report.csv: (logFC correlation, t-stat correlation, 
  Jaccard of FDR < 0.05 sets) per celltype against the full-cohort results.

Prerequisites:
  1. preprocess_perez2022.py has produced perez2022_filtered.h5ad
  2. validate_perez2022.py has produced *_py_*.csv files in DATADIR

Environment:
  INPUT_H5AD  — path to perez2022_filtered.h5ad (required)
  PEREZ2022_REF   — path to reference CSV directory (optional, defaults to
                     ../data/perez2022 relative to this file)
"""

import os
import numpy as np
import pandas as pd
import scanpy as sc
from pathlib import Path

import dreampy as dp

# =============================================================================
# Paths
# =============================================================================

# ── Defaults (edit these for interactive use) ───────────────────────────────
# Script lives at: dreampy/validation/python/sensitivity_perez2022.py
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent  # dreampy/

INPUT_H5AD = str(REPO_ROOT / "validation/data/perez2022/perez2022_filtered.h5ad")
REF_DIR = Path(
    os.environ.get(
        "PEREZ2022_REF",
        Path(__file__).parent.parent / "data" / "perez2022",
    )
)

# =============================================================================
# Pipeline constants — must match reference_perez2022.R exactly
# =============================================================================

FORMULA = "~ sle + (1 | donor_id) + (1 | Processing_Cohort)"
GROUPBY = ["author_cell_type", "sample_id"]
MIN_CELLS = 20
MIN_SAMPLES = 4
MIN_COUNT = 5
MIN_PROP = 0.4

# =============================================================================
# Assay auto-discovery from reference directory
# =============================================================================

if REF_DIR.is_dir():
    ASSAYS = sorted({
        p.name.replace("_py_results.csv", "")
        for p in REF_DIR.glob("*_py_results.csv")
    })
else:
    ASSAYS = []

# =============================================================================
# Runs the full dreampy pipeline ONCE for all ~10 assays.
# =============================================================================

def pipeline(INPUT_H5AD, REF_DIR):
    """
    Run the complete dreampy pipeline:
      pseudobulk -> filter_samples -> TMM -> filter_by_expr -> log2CPM -> voom ->
      fit (Satterthwaite) -> eBayes
    """
    if INPUT_H5AD is None or not Path(INPUT_H5AD).exists():
        raise FileNotFoundError("INPUT_H5AD not set or file not found")
    if len(ASSAYS) == 0:
        raise RuntimeError(f"No reference CSVs found in {REF_DIR}")

    full_metrics = pd.read_csv(REF_DIR / "perez2022_validation_metrics.csv")
    full_sample_counts = full_metrics.groupby("assay")["n_samples"].first().to_dict()

    # --- Load filtered data ---
    ad = sc.read_h5ad(INPUT_H5AD)

    ad = ad[ad.obs["study_cohort"] != "ImmVar"].copy()

    # --- Step 0: Pseudobulk aggregation ---
    pb = dp.aggregate_pseudobulk(ad, groupby=GROUPBY)

    # =====================================================================
    #  Pipeline: filter_samples -> TMM -> filter -> log2CPM -> voom 
    # -> fit_models (Satterthwaite) -> eBayes
    # =====================================================================

    pb = dp.filter_samples(pb, min_cells=MIN_CELLS, min_samples=MIN_SAMPLES)
    pb = dp.compute_tmm_factors(pb, assay_col="assays")
    assays_dict = dp.filter_by_expr(
        pb, assay_col="assays", min_count=MIN_COUNT, min_prop=MIN_PROP
    )

    concordance_rows = []
    for assay_name, assay_pb in assays_dict.items():
        n_samples_sens = assay_pb.n_obs
        assay_pb = dp.log2cpm(assay_pb)
        assay_pb = dp.estimate_weights(
            assay_pb, formula=FORMULA, reml=False, n_jobs=-1, span=0.5
        )
        fit = dp.fit_models(
            assay_pb,
            formula=FORMULA,
            n_jobs=-1,
            reml=True,
            assay_name=assay_name,
            ddf="Satterthwaite",
        )
        fit_eb = dp.ebayes(fit)
        sens = dp.get_results(fit_eb, assay_name=assay_name)
        sens.to_csv(REF_DIR / f"{assay_name}_py_sensitivity_results.csv")

        # --- Concordance against the full-cohort run ---
        full_path = REF_DIR / f"{assay_name}_py_results.csv"
        if not full_path.exists():
            print(f"WARN: no full-cohort results for {assay_name}, skipping concordance")
            continue
        full = pd.read_csv(full_path)
        assert sens["id"].is_unique, f"{assay_name}: duplicate genes in sensitivity results"
        assert full["id"].is_unique, f"{assay_name}: duplicate genes in full-cohort results"

        merged = sens.merge(full, on='id', suffixes=("_sens", "_full"))
        n_full_only = len(set(full["id"]) - set(sens["id"]))
        n_sens_only = len(set(sens["id"]) - set(full["id"]))

        sig_sens = set(merged.loc[merged["adj_p_val_sens"] < 0.05, "id"])
        sig_full = set(merged.loc[merged["adj_p_val_full"] < 0.05, "id"])
        union = sig_sens | sig_full
        jaccard = len(sig_sens & sig_full) / len(union) if union else np.nan

        concordance_rows.append({
            "assay": assay_name,
            "n_samples_full": full_sample_counts.get(assay_name, np.nan),
            "n_samples_sens": n_samples_sens,
            "n_shared_genes": len(merged),
            "n_full_only": n_full_only,
            "n_sens_only": n_sens_only,
            "logfc_corr": np.corrcoef(merged["logfc_sens"], merged["logfc_full"])[0, 1],
            "t_corr": np.corrcoef(merged["t_sens"], merged["t_full"])[0, 1],
            "jaccard_fdr05": jaccard,
            "n_sig_full": len(sig_full),
            "n_sig_sens": len(sig_sens),
        })

    pd.DataFrame(concordance_rows).to_csv(REF_DIR / f"sensitivity_concordance_report.csv")

if __name__ == "__main__":
    pipeline(INPUT_H5AD, REF_DIR)