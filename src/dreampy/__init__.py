"""
dreampy: pseudobulk mixed-model differential expression in Python
=================================================================

A Python port of the R dreamlet workflow for differential expression
analysis of pseudobulk single-cell RNA-seq data.

Quick start
-----------
>>> import scanpy as sc
>>> import dreampy as dp
>>>
>>> adata = sc.read_h5ad("my_data.h5ad")
>>>
>>> # Pseudobulk aggregation; the first groupby column defines the assays
>>> pb = dp.aggregate_pseudobulk(adata, groupby=["celltype", "donor_id", "site"])
>>>
>>> # Preprocessing
>>> pb = dp.filter_samples(pb, min_cells=5, min_samples=3)
>>> pb = dp.compute_tmm_factors(pb, assay_col="assays")
>>> assays = dp.filter_by_expr(pb, assay_col="assays")  # dict: assay name -> AnnData
>>>
>>> # Differential expression, one assay (cell type) at a time
>>> formula = "~ condition + (1|site)"
>>> for name, assay_pb in assays.items():
...     assay_pb = dp.log2cpm(assay_pb)
...     assay_pb = dp.estimate_weights(assay_pb, formula=formula)
...     fit = dp.fit_models(assay_pb, formula=formula)
...     fit = dp.ebayes(fit)
...     results = dp.get_results(fit, assay_name=name)

See ``notebooks/tutorial.ipynb`` for a complete walkthrough.

Public API
----------
Pseudobulk:
    aggregate_pseudobulk     Sum single-cell counts into pseudobulk samples

Preprocessing:
    filter_samples           Filter pseudobulk samples by cell count
    compute_tmm_factors      TMM normalization factors
    filter_by_expr           Expression-based gene filtering
    log2cpm                  log2 counts-per-million transformation
    estimate_weights         voom-style precision weights

Model fitting:
    fit_models               Fit gene-wise linear (mixed) models
    fit_genes_parallel       Lower-level parallel fitting on prebuilt matrices
    ebayes                   Empirical Bayes moderation

Results:
    get_results              Results table for one fit
    get_results_multi_assay  Combined results across assays, study-wide adjustment

Formula:
    Formula                  Formula parser
    validate_formula         Check formula syntax and metadata compatibility

Types:
    DEFit                    Fitted model container
    DEControl                Fitting configuration
    FitStatus                Per-gene fit status codes
"""

__version__ = "0.1.1"

from .de_types import DEControl, DEFit, FitStatus
from .ebayes import ebayes
from .fitting import fit_genes_parallel, fit_models
from .formula import Formula, validate_formula
from .preprocessing import (
    compute_tmm_factors,
    estimate_weights,
    filter_by_expr,
    filter_samples,
    log2cpm,
)
from .pseudobulk import aggregate_pseudobulk
from .results import get_results, get_results_multi_assay

__all__ = [
    "DEControl",
    "DEFit",
    "FitStatus",
    "Formula",
    "__version__",
    "aggregate_pseudobulk",
    "compute_tmm_factors",
    "ebayes",
    "estimate_weights",
    "filter_by_expr",
    "filter_samples",
    "fit_genes_parallel",
    "fit_models",
    "get_results",
    "get_results_multi_assay",
    "log2cpm",
    "validate_formula",
]