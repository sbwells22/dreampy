"""
dreampy: Differential Expression Analysis for Single-Cell RNA-seq
=====================================================================

A Python implementation of the dreamlet methodology for differential
expression analysis of pseudobulk single-cell data.

Quick Start
-----------
>>> import dreampy as dp
>>>
>>> # Tier 3: Multi-assay DE (easiest)
>>> pb = dp.aggregate_pseudobulk(adata, groupby=["celltype", "donor"])
>>> results = dp.de(pb, "~ disease + (1|donor)", assay_col="assays")
>>> dp.top_table(results, coef="disease", fdr_scope="global")
>>>
>>> # Tier 2: Single-assay DE
>>> pb = dp.aggregate_pseudobulk(adata, groupby=["donor"])
>>> fit = dp.de_assay(pb, "~ disease + (1|donor)")
>>> dp.top_table(fit, coef="disease")
>>>
>>> # Tier 1: Granular control
>>> pb = dp.aggregate_pseudobulk(adata, groupby=["celltype", "donor"])
>>> pb = dp.filter_samples(pb, min_cells=5, min_samples=4)
>>> # ... subset to single assay ...
>>> counts = pb.X.T  # genes × samples
>>> lib_size = counts.sum(axis=0)
>>> norm_factors = dp.compute_tmm_factors(counts, lib_size)
>>> keep = dp.filter_by_expr(counts, lib_size, norm_factors)
>>> y = dp.log2cpm(counts[keep], lib_size, norm_factors)
>>> weights, trend = dp.estimate_weights(y, sigma, lib_size, norm_factors)
>>> fit = dp.fit_models(y, formula, pb.obs, weights=weights)
>>> fit = dp.eBayes(fit)
>>> dp.top_table(fit, coef="disease")

API Reference
-------------
Tier 1 (Granular):
    aggregate_pseudobulk    Aggregate single-cell to pseudobulk
    filter_samples          Filter samples by cell count
    compute_tmm_factors     TMM normalization factors
    filter_by_expr          Expression-based gene filtering
    log2cpm                 Log2 CPM transformation
    estimate_weights        Precision weight estimation
    fit_models              Fit gene-wise models
    eBayes                  Empirical Bayes moderation

Results:
    top_table               Extract DE results as DataFrame

Contrasts:
    make_contrasts          Create contrast matrix
    contrasts_fit           Apply contrasts to fitted model

Validation:
    validate_formula        Check formula syntax and compatibility

Types:
    DEFit                   Fitted model container
    DEControl               Analysis parameters
"""

__version__ = "0.1.0"

# =============================================================================
# Types
# =============================================================================
from .de_types import (
    DEControl,
    DEFit,
    FitStatus,
)

# Step 6: Empirical Bayes moderation
from .ebayes import ebayes

# Step 5: Model fitting
from .fitting import (
    fit_genes_parallel,
    fit_models,
)

# =============================================================================
# Validation
# =============================================================================
from .formula import (
    Formula,
    validate_formula,
)

# Step 1-3: Preprocessing (including sample filtering)
# Step 4: Weight estimation
from .preprocessing import (
    compute_tmm_factors,
    estimate_weights,
    filter_by_expr,
    filter_samples,
    log2cpm,
)

# =============================================================================
# Tier 1: Granular Components
# =============================================================================
# Step 0: Pseudobulk aggregation
from .pseudobulk import aggregate_pseudobulk

# =============================================================================
# Results Interface
# =============================================================================
from .results import get_results, get_results_multi_assay

# =============================================================================
# Public API
# =============================================================================
__all__ = [
    # Version
    "__version__",
    # Tier 3
    "de",
    # Tier 2
    "de_assay",
    # Tier 1 - Pseudobulk
    "aggregate_pseudobulk",
    # Tier 1 - Preprocessing
    "compute_tmm_factors",
    "filter_by_expr",
    "filter_samples",
    "log2cpm",
    "estimate_weights",
    # Tier 1 - Fitting
    "fit_models",
    "fit_genes_parallel",
    # Tier 1 - Moderation
    "ebayes",
    # Results
    "get_results",
    "get_results_multi_assay",
    # Contrasts
    "make_contrasts",
    "contrasts_fit",
    "validate_contrasts",
    # Validation
    "Formula",
    "validate_formula",
    # Types
    "DEFit",
    "DEControl",
    "FitStatus",
]
