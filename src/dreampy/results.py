"""
Results Interface for Differential Expression Analysis for dreampy

This module provides functions to extract and format results from fitted
dream/dreamlet models.

Key functions:
- get_results(): Extract differentially expressed genes from a single fit
- get_results_multi_assay(): Combine results across cell types with global FDR

References:
- limma::topTable()
- variancePartition::topTable()
- dreamlet::topTable()
"""
from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy import stats

# Import DEFit for type hints
if TYPE_CHECKING:
    from .de_types import DEFit

__all__ = [
    "get_results",
    "get_results_multi_assay",
]


def get_results(
    fit: DEFit,
    coef: int | str | list | None = None,
    assay_name: str | None = None,
    number: float = np.inf,
    adjust_method: str = "BH",
    sort_by: str = "p",
    p_value: float = 1.0,
    lfc: float = 0.0,
) -> pd.DataFrame:
    """
    Extract differentially expressed genes from a fitted model.

    High-level wrapper that accepts a DEFit object and extracts the
    necessary arrays to compute results.

    Parameters
    ----------
    fit : DEFit
        Fitted model from fit_models(), optionally with ebayes() applied.
        Required fields: coefficients, stdev_unscaled, sigma, df_resid_eb,
        gene_names, coef_names.
        Optional fields (from eBayes): s2_post, df_total, lods, z_std.
    coef : int, str, or list, optional
        Which coefficient(s) to extract. Can be:
        - int: coefficient index
        - str: coefficient name (e.g., "treatment", "diagnosis_dcm")
        - list: multiple coefficients
        Default: all non-intercept coefficients.
    assay_name : str, optional
        Assay name to include in results. If None and fit.assay_name is set,
        uses fit.assay_name. If both are None, assay column is omitted.
    number : float
        Maximum number of genes to return. Default: np.inf (all genes).
    adjust_method : str
        Multiple testing correction method:
        - "BH": Benjamini-Hochberg FDR control (default)
        - "bonferroni": Bonferroni FWER control
        - "holm": Holm step-down FWER control
        - "none": No correction
    sort_by : str
        Sort results by:
        - "p": p-value (default)
        - "adj_p": adjusted p-value
        - "logfc": log-fold-change (absolute value)
        - "t": t-statistic (absolute value)
        - "b": B statistic / log-odds
        - "none": no sorting
    p_value : float
        P-value cutoff (on adjusted p-values). Default 1.0 (no filter).
    lfc : float
        Log-fold-change cutoff (absolute value). Default 0.0 (no filter).

    Returns
    -------
    results : pd.DataFrame
        DataFrame with columns (conditional on availability):
        - assay: assay name (if provided)
        - coef: coefficient name
        - gene: gene identifier
        - logfc: log-fold-change (coefficient estimate)
        - ave_expr: average expression (if available)
        - t: (moderated) t-statistic
        - p_value: raw p-value
        - adj_p_val: adjusted p-value
        - b: log-odds of DE (if eBayes applied)
        - z_std: standardized z-score (if eBayes applied)

    Raises
    ------
    ValueError
        If fit has no genes or required fields are missing/empty.

    Examples
    --------
    >>> # Basic usage after eBayes
    >>> fit = fit_models(Y, formula, metadata, gene_names=genes)
    >>> fit = ebayes(fit)
    >>> results = get_results(fit, coef="treatment", number=20)

    >>> # Get all genes for a specific coefficient
    >>> all_results = get_results(fit, coef="diagnosis_dcm")

    >>> # Filter by FDR and fold-change
    >>> sig_results = get_results(fit, coef="treatment", p_value=0.05, lfc=1.0)

    >>> # Include assay name in results
    >>> results = get_results(fit, coef="treatment", assay_name="myeloid")

    Notes
    -----
    If eBayes has been applied to the fit, moderated t-statistics are computed
    using the posterior variance estimates (s2_post) and total degrees of
    freedom (df_total). Otherwise, ordinary t-statistics are used.

    The b (log-odds) and z_std columns are only included if eBayes has been
    applied to the fit.
    """
    # Import DEFit here to avoid circular imports at module load time
    try:
        from .de_types import DEFit # noqa: F401
    except ImportError:
        pass

    # =================================================================
    # Step 1: Validate input DEFit
    # =================================================================

    n_genes = len(fit.gene_names)

    if n_genes == 0:
        raise ValueError("DEFit has no genes. Cannot extract results from empty fit.")

    # Check required fields
    if fit.coefficients.size == 0:
        raise ValueError("DEFit.coefficients is empty. Fit may have failed.")

    if fit.stdev_unscaled.size == 0:
        raise ValueError(
            "DEFit.stdev_unscaled is empty. "
            "Ensure fitting computed stdev_unscaled (Bug #12 fix required)."
        )

    if fit.sigma.size == 0:
        raise ValueError("DEFit.sigma is empty. Fit may have failed.")

    if fit.df_resid_eb.size == 0:
        raise ValueError(
            "DEFit.df_resid_eb is empty. Ensure fitting computed df_resid (Bug #12 fix required)."
        )

    if len(fit.coef_names) == 0:
        raise ValueError("DEFit.coef_names is empty.")

    # =================================================================
    # Step 2: Extract arrays from DEFit
    # =================================================================

    coefficients = fit.coefficients
    stdev_unscaled = fit.stdev_unscaled
    sigma = fit.sigma
    df_residual = fit.df_resid_eb  # Base df (may be overridden by df_total)
    gene_names = fit.gene_names
    coef_names = fit.coef_names

    # Optional fields - may be None if eBayes not applied or weights not estimated
    ave_expr = fit.ave_expr if fit.ave_expr is not None and fit.ave_expr.size > 0 else None
    s2_post = fit.s2_post  # None if eBayes not applied
    df_total = fit.df_total  # None if eBayes not applied
    lods = fit.lods  # None if eBayes not applied
    z_std = fit.z_std  # None if eBayes not applied

    # =================================================================
    # Step 3: Resolve assay name
    # =================================================================

    # Use fit.assay_name if assay_name param not provided
    if assay_name is None and hasattr(fit, "assay_name") and fit.assay_name is not None:
        assay_name = fit.assay_name

    # =================================================================
    # Step 4: Call low-level implementation
    # =================================================================

    return _get_results_arrays(
        coefficients=coefficients,
        stdev_unscaled=stdev_unscaled,
        sigma=sigma,
        df_residual=df_residual,
        gene_names=gene_names,
        coef_names=coef_names,
        ave_expr=ave_expr,
        s2_post=s2_post,
        df_total=df_total,
        lods=lods,
        z_std=z_std,
        coef=coef,
        assay_name=assay_name,
        number=number,
        adjust_method=adjust_method,
        sort_by=sort_by,
        p_value=p_value,
        lfc=lfc,
    )


def _get_results_arrays(
    coefficients: np.ndarray,
    stdev_unscaled: np.ndarray,
    sigma: np.ndarray,
    df_residual: np.ndarray,
    gene_names: np.ndarray,
    coef_names: list[str],
    ave_expr: np.ndarray | None = None,
    s2_post: np.ndarray | None = None,
    df_total: np.ndarray | None = None,
    lods: np.ndarray | None = None,
    z_std: np.ndarray | None = None,
    coef: int | str | list | None = None,
    assay_name: str | None = None,
    number: float = np.inf,
    adjust_method: str = "BH",
    sort_by: str = "p",
    p_value: float = 1.0,
    lfc: float = 0.0,
) -> pd.DataFrame:
    """
    Extract differentially expressed genes (low-level array interface).

    This is the low-level implementation that accepts raw numpy arrays.
    Most users should use get_results(fit, ...) instead.

    Parameters
    ----------
    coefficients : np.ndarray (n_genes, n_coef)
        Coefficient estimates (log-fold-changes)
    stdev_unscaled : np.ndarray (n_genes, n_coef)
        Unscaled standard errors (SE/sigma)
    sigma : np.ndarray (n_genes,)
        Residual standard deviations (before eBayes)
    df_residual : np.ndarray
        Residual degrees of freedom. Can be:
        - (n_genes,) scalar per gene (for eBayes df)
        - (n_genes, n_coef) per gene and coefficient (for Satterthwaite)
    gene_names : np.ndarray (n_genes,)
        Gene identifiers
    coef_names : List[str]
        Names of coefficients
    ave_expr : np.ndarray (n_genes,), optional
        Average expression values
    s2_post : np.ndarray (n_genes,), optional
        Posterior variance estimates from eBayes
    df_total : np.ndarray (n_genes,) or (n_genes, n_coef), optional
        Total df (df_prior + df_resid) from eBayes
    lods : np.ndarray (n_genes, n_coef), optional
        Log-odds of differential expression (B statistic)
    z_std : np.ndarray (n_genes, n_coef), optional
        Standardized z-scores from eBayes
    coef : int, str, or list, optional
        Which coefficient(s) to extract. Default: all non-intercept.
    assay_name : str, optional
        Assay name to include in results. If None, column omitted.
    number : float
        Maximum number of genes to return. Use np.inf for all.
    adjust_method : str
        Multiple testing correction: "BH", "bonferroni", "holm", "none"
    sort_by : str
        Sort by: "p", "adj_p", "logfc", "t", "b", "none"
    p_value : float
        P-value cutoff (after adjustment). Default 1.0 (no filter).
    lfc : float
        Log-fold-change cutoff (absolute value). Default 0.0.

    Returns
    -------
    results : pd.DataFrame
        Columns (conditional): assay, coef, gene, logfc, ave_expr, t,
        p_value, adj_p_val, b, z_std

    Notes
    -----
    The function computes moderated t-statistics if s2_post is provided,
    otherwise uses ordinary t-statistics.

    Multiple testing correction methods:
    - "BH": Benjamini-Hochberg FDR control (default)
    - "bonferroni": Bonferroni FWER control
    - "holm": Holm step-down FWER control
    - "none": No correction
    """
    n_genes, n_coef = coefficients.shape

    # Resolve coefficient selection
    coef_indices = _resolve_coef(coef, coef_names, n_coef)

    # Handle df_residual shape
    # Can be (n_genes,) or (n_genes, n_coef)
    df_residual = np.atleast_1d(df_residual)
    if df_residual.ndim == 1:
        # Broadcast to (n_genes, n_coef)
        df_residual_2d = np.broadcast_to(df_residual[:, np.newaxis], (n_genes, n_coef))
    else:
        df_residual_2d = df_residual

    # Check if eBayes was applied
    ebayes_applied = lods is not None and z_std is not None

    # Compute statistics for each selected coefficient
    results_list = []

    for coef_idx in coef_indices:
        coef_name = coef_names[coef_idx]

        # Extract per-coefficient data
        logfc = coefficients[:, coef_idx]
        se_unscaled = stdev_unscaled[:, coef_idx]
        df = df_residual_2d[:, coef_idx] if df_residual_2d.ndim == 2 else df_residual

        # Use df_total for t-test df if provided (eBayes case)
        if df_total is not None:
            # df_total may be (n_genes,) or (n_genes, n_coef)
            if df_total.ndim == 2:
                df_for_t = df_total[:, coef_idx]
            else:
                df_for_t = df_total
        else:
            df_for_t = df

        # Compute standard error
        if s2_post is not None:
            # Moderated: use posterior variance
            se = se_unscaled * np.sqrt(s2_post)
        else:
            # Ordinary: use sigma directly
            se = se_unscaled * sigma

        # Compute t-statistics
        with np.errstate(divide="ignore", invalid="ignore"):
            t_stat = logfc / se

        # Compute p-values
        # Two-tailed t-test
        with np.errstate(invalid="ignore"):
            p_val = 2 * stats.t.sf(np.abs(t_stat), df_for_t)

        # Handle edge cases
        p_val = np.clip(p_val, 0, 1)
        p_val = np.where(np.isfinite(t_stat), p_val, np.nan)

        # Build results dataframe for this coefficient
        df_result = pd.DataFrame(
            {
                "coef": coef_name,
                "id": gene_names,
                "logfc": logfc,
                "t": t_stat,
                "p_value": p_val,
            }
        )

        # Add assay if provided
        if assay_name is not None:
            df_result["assay"] = assay_name

        # Add average expression if available
        if ave_expr is not None:
            df_result["ave_expr"] = ave_expr

        # Add eBayes columns if available
        if ebayes_applied:
            df_result["b"] = lods[:, coef_idx]
            df_result["z_std"] = z_std[:, coef_idx]

        results_list.append(df_result)

    # Combine results from all coefficients
    if len(results_list) == 0:
        return pd.DataFrame()

    results = pd.concat(results_list, ignore_index=True)

    # Apply multiple testing correction
    valid_mask = np.isfinite(results["p_value"].values)
    adj_p = np.full(len(results), np.nan)

    if valid_mask.sum() > 0:
        valid_pvals = results.loc[valid_mask, "p_value"].values
        adj_p[valid_mask] = _adjust_pvalues(valid_pvals, method=adjust_method)

    results["adj_p_val"] = adj_p

    # Apply filters
    # LFC filter
    if lfc > 0:
        results = results[np.abs(results["logfc"]) >= lfc]

    # P-value filter (on adjusted p-values)
    if p_value < 1.0:
        results = results[results["adj_p_val"] <= p_value]

    # Sort
    if sort_by.lower() != "none" and len(results) > 0:
        sort_col, ascending = _get_sort_params(sort_by)
        if sort_col in results.columns:
            # For logfc and t, sort by absolute value
            if sort_col in ("logfc", "t"):
                results = results.iloc[np.argsort(-np.abs(results[sort_col].values))]
            else:
                results = results.sort_values(sort_col, ascending=ascending, na_position="last")

    # Limit number of results
    if number < len(results):
        results = results.head(int(number))

    # Reorder columns
    # Order: [assay], coef, gene, logfc, [ave_expr], t, p_value, adj_p_val, [b], [z_std]
    col_order = []
    if "assay" in results.columns:
        col_order.append("assay")
    col_order.extend(["coef", "id", "logfc"])
    if "ave_expr" in results.columns:
        col_order.append("ave_expr")
    col_order.extend(["t", "p_value", "adj_p_val"])
    if "b" in results.columns:
        col_order.append("b")
    if "z_std" in results.columns:
        col_order.append("z_std")

    # Only include columns that exist
    col_order = [c for c in col_order if c in results.columns]
    results = results[col_order]

    # Reset index
    results = results.reset_index(drop=True)

    return results


def get_results_multi_assay(
    fits: dict[str, DEFit],
    coef: str | list[str] | None = None,
    number: float = np.inf,
    adjust_method: str = "BH",
    sort_by: str = "p",
    p_value: float = 1.0,
    lfc: float = 0.0,
) -> pd.DataFrame:
    """
    Extract results from multiple assays with global FDR correction.

    This combines results from multiple cell types/assays and applies
    a study-wide multiple testing correction across all tests.

    Parameters
    ----------
    fits : Dict[str, DEFit]
        Dictionary mapping assay names to DEFit objects.
    coef : str or List[str], optional
        Which coefficient(s) to extract
    number : float
        Maximum number of genes per assay to include before combining.
        Default: np.inf (all genes).
    adjust_method : str
        Multiple testing correction method
    sort_by : str
        Sort criterion
    p_value : float
        Adjusted p-value cutoff
    lfc : float
        Log-fold-change cutoff

    Returns
    -------
    results : pd.DataFrame
        Combined results with columns: assay, coef, gene, logfc,
        ave_expr, t, p_value, adj_p_val, b, z_std
        (conditional columns included based on availability)

    Notes
    -----
    The global FDR procedure:
    1. Extract all results from each assay (without FDR correction)
    2. Combine into single table
    3. Apply BH/other correction across ALL tests

    This ensures that adj_p_val controls the FDR across the entire study,
    not just within each cell type.
    """
    all_results = []

    for dict_key, fit in fits.items():
        # Prefer fit.assay_name if set, else use dict key
        assay_name = (
            fit.assay_name
            if hasattr(fit, "assay_name") and fit.assay_name is not None
            else dict_key
        )

        # Extract results for this assay WITHOUT adjustment
        # We'll do global adjustment later
        try:
            df = get_results(
                fit=fit,
                coef=coef,
                assay_name=assay_name,
                number=np.inf,  # Get all genes
                adjust_method="none",  # No adjustment yet
                sort_by="none",  # Don't sort yet
                p_value=1.0,  # No p-value filter yet
                lfc=0.0,  # No LFC filter yet
            )

            if len(df) > 0:
                all_results.append(df)
        except Exception as e:
            # Skip failed assays with warning
            import warnings

            warnings.warn(f"Failed to extract results for assay {assay_name}: {e}", stacklevel=2)
            continue

    if len(all_results) == 0:
        return pd.DataFrame()

    # Combine all results
    results = pd.concat(all_results, ignore_index=True)

    # Apply global FDR correction
    valid_mask = np.isfinite(results["p_value"].values)
    adj_p = np.full(len(results), np.nan)

    if valid_mask.sum() > 0:
        valid_pvals = results.loc[valid_mask, "p_value"].values
        adj_p[valid_mask] = _adjust_pvalues(valid_pvals, method=adjust_method)

    results["adj_p_val"] = adj_p

    # Apply filters
    if lfc > 0:
        results = results[np.abs(results["logfc"]) >= lfc]

    if p_value < 1.0:
        results = results[results["adj_p_val"] <= p_value]

    # Sort
    if sort_by.lower() != "none" and len(results) > 0:
        sort_col, ascending = _get_sort_params(sort_by)
        if sort_col in results.columns:
            # For logfc and t, sort by absolute value
            if sort_col in ("logfc", "t"):
                results = results.iloc[np.argsort(-np.abs(results[sort_col].values))]
            else:
                results = results.sort_values(sort_col, ascending=ascending, na_position="last")

    # Limit results
    if number < len(results):
        results = results.head(int(number))

    # Reorder columns
    col_order = []
    if "assay" in results.columns:
        col_order.append("assay")
    col_order.extend(["coef", "id", "logfc"])
    if "ave_expr" in results.columns:
        col_order.append("ave_expr")
    col_order.extend(["t", "p_value", "adj_p_val"])
    if "b" in results.columns:
        col_order.append("b")
    if "z_std" in results.columns:
        col_order.append("z_std")

    col_order = [c for c in col_order if c in results.columns]
    results = results[col_order]

    return results.reset_index(drop=True)


# =============================================================================
# Helper Functions
# =============================================================================


def _resolve_coef(
    coef: int | str | list | None,
    coef_names: list[str],
    n_coef: int,
) -> list[int]:
    """
    Resolve coefficient specification to list of indices.

    Parameters
    ----------
    coef : int, str, list, or None
        Coefficient specification:
        - None: all non-intercept coefficients
        - int: single index
        - str: single name
        - list: multiple indices or names
    coef_names : List[str]
        Names of all coefficients
    n_coef : int
        Total number of coefficients

    Returns
    -------
    indices : List[int]
        List of coefficient indices to extract

    Raises
    ------
    ValueError
        If coefficient name not found or index out of range
    """
    if coef is None:
        # Default: all non-intercept coefficients
        # Skip any coefficient named "Intercept" or "(Intercept)"
        indices = []
        for i, name in enumerate(coef_names):
            if name.lower() not in ("intercept", "(intercept)"):
                indices.append(i)
        # If all are intercepts (unusual), return all
        if len(indices) == 0:
            indices = list(range(n_coef))
        return indices

    if isinstance(coef, (int, np.integer)):
        if coef < 0 or coef >= n_coef:
            raise ValueError(f"Coefficient index {coef} out of range [0, {n_coef})")
        return [int(coef)]

    if isinstance(coef, str):
        if coef in coef_names:
            return [coef_names.index(coef)]
        raise ValueError(f"Coefficient '{coef}' not found. Available: {coef_names}")

    if isinstance(coef, (list, tuple)):
        indices = []
        for c in coef:
            if isinstance(c, (int, np.integer)):
                if c < 0 or c >= n_coef:
                    raise ValueError(f"Coefficient index {c} out of range [0, {n_coef})")
                indices.append(int(c))
            elif isinstance(c, str):
                if c in coef_names:
                    indices.append(coef_names.index(c))
                else:
                    raise ValueError(f"Coefficient '{c}' not found. Available: {coef_names}")
            else:
                raise ValueError(f"Invalid coefficient type: {type(c)}")
        return indices

    raise ValueError(f"Invalid coef specification: {coef}")


def _adjust_pvalues(pvalues: np.ndarray, method: str = "BH") -> np.ndarray:
    """
    Adjust p-values for multiple testing.

    Parameters
    ----------
    pvalues : np.ndarray
        Raw p-values (1D array, should not contain NaN)
    method : str
        Adjustment method:
        - "BH": Benjamini-Hochberg FDR control
        - "bonferroni": Bonferroni FWER control
        - "holm": Holm step-down FWER control
        - "none": No adjustment

    Returns
    -------
    adj_p : np.ndarray
        Adjusted p-values

    Notes
    -----
    BH procedure:
    1. Sort p-values in ascending order
    2. For rank i (1-indexed), compute p_i * n / i
    3. Enforce monotonicity: adj_p[i] >= adj_p[i-1]
    4. Cap at 1.0
    """
    n = len(pvalues)
    if n == 0:
        return np.array([])

    method = method.upper()

    if method == "NONE":
        return pvalues.copy()

    if method == "BONFERRONI":
        return np.minimum(pvalues * n, 1.0)

    if method == "HOLM":
        # Holm step-down procedure
        order = np.argsort(pvalues)
        sorted_p = pvalues[order]

        # Adjusted p: p_i * (n - i + 1)
        ranks = np.arange(1, n + 1)
        adj_sorted = sorted_p * (n - ranks + 1)

        # Enforce monotonicity (cumulative maximum)
        adj_sorted = np.maximum.accumulate(adj_sorted)

        # Cap at 1
        adj_sorted = np.minimum(adj_sorted, 1.0)

        # Restore original order
        adj_p = np.empty(n)
        adj_p[order] = adj_sorted
        return adj_p

    if method == "BH":
        # Benjamini-Hochberg FDR control
        order = np.argsort(pvalues)
        sorted_p = pvalues[order]

        # Adjusted p: p_i * n / i
        ranks = np.arange(1, n + 1)
        adj_sorted = sorted_p * n / ranks

        # Enforce monotonicity from the end (cumulative minimum in reverse)
        adj_sorted = np.minimum.accumulate(adj_sorted[::-1])[::-1]

        # Cap at 1
        adj_sorted = np.minimum(adj_sorted, 1.0)

        # Restore original order
        adj_p = np.empty(n)
        adj_p[order] = adj_sorted
        return adj_p

    raise ValueError(
        f"Unknown adjustment method: {method}. Use 'BH', 'bonferroni', 'holm', or 'none'."
    )


def _get_sort_params(sort_by: str) -> tuple:
    """
    Get sort column and direction.

    Parameters
    ----------
    sort_by : str
        Sort option: "p", "adj_p", "logfc", "t", "b", "none"

    Returns
    -------
    tuple
        (column_name, ascending)
    """
    sort_by = sort_by.lower()

    if sort_by == "p":
        return "p_value", True  # ascending
    elif sort_by == "adj_p":
        return "adj_p_val", True  # ascending
    elif sort_by == "logfc":
        return "logfc", False  # descending by absolute value (handled separately)
    elif sort_by == "t":
        return "t", False  # descending by absolute value (handled separately)
    elif sort_by == "b":
        return "b", False  # descending (higher lods = more likely DE)
    else:
        return "p_value", True  # default
