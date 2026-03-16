"""
Pseudobulk aggregation module for dreampy.

This module provides functionality for aggregating single-cell expression data
into pseudobulk samples, following the methodology described in Hoffman & Roussos (2024).

Main Function
-------------
aggregate_pseudobulk
    Aggregate single-cell data to pseudobulk samples by summing counts
    within groups defined by metadata columns.

Notes
-----
**Input Requirements:**
- Input data should be raw integer counts. Non-integer values will trigger
  a warning and be truncated.

**Groupby Semantics:**
- First column in groupby defines assay labels (values become `assays` column)
- Remaining columns define sample replicates within each assay
- For single-assay analysis, add a constant column as the first groupby element
"""

import warnings

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy import sparse

# =============================================================================
# Validation Functions
# =============================================================================


def validate_columns(ad: AnnData, columns: list[str], context: str = "groupby") -> None:
    """
    Validate that specified columns exist in AnnData.obs.

    Parameters
    ----------
    ad : AnnData
        Single-cell expression object
    columns : List[str]
        Column names to validate
    context : str
        Context for error message (e.g., "groupby", "covariates")

    Raises
    ------
    ValueError
        If any columns are missing from ad.obs
    TypeError
        If columns is not a list
    """
    if not isinstance(columns, list):
        raise TypeError(f"{context} must be a list of column names, got {type(columns).__name__}")

    if len(columns) == 0:
        raise ValueError(f"{context} cannot be an empty list")

    available_cols = set(ad.obs.columns)
    missing_cols = [col for col in columns if col not in available_cols]

    if missing_cols:
        # Provide helpful suggestions for typos
        suggestions = {}
        for missing in missing_cols:
            similar = [
                c
                for c in available_cols
                if missing.lower() in c.lower() or c.lower() in missing.lower()
            ]
            if similar:
                suggestions[missing] = similar[:3]

        error_msg = f"Columns not found in ad.obs for {context}: {missing_cols}"
        if suggestions:
            error_msg += "\n\nDid you mean:"
            for missing, similar in suggestions.items():
                error_msg += f"\n  '{missing}' -> {similar}"
        error_msg += f"\n\nAvailable columns: {sorted(available_cols)}"

        raise ValueError(error_msg)


def validate_assay(ad: AnnData, assay: str | None) -> None:
    """
    Validate that specified assay/layer exists in AnnData.

    Parameters
    ----------
    ad : AnnData
        Single-cell expression object
    assay : str or None
        Layer name to validate. If None, validates that ad.X exists.

    Raises
    ------
    ValueError
        If assay is not found in ad.layers or if ad.X is None when assay is None
    """
    if assay is None:
        if ad.X is None:
            raise ValueError(
                "ad.X is None. Either set ad.X or specify an assay from ad.layers.\n"
                f"Available layers: {list(ad.layers.keys())}"
            )
    else:
        available_layers = list(ad.layers.keys())
        if assay not in available_layers:
            raise ValueError(
                f"Assay '{assay}' not found in ad.layers.\nAvailable layers: {available_layers}"
            )


# =============================================================================
# Core Aggregation Function
# =============================================================================


def aggregate_pseudobulk(ad: AnnData, groupby: list[str], layer: str | None = None) -> AnnData:
    """
    Aggregate single-cell data to pseudobulk samples.

    Performs pseudobulk aggregation by summing counts across cells within
    each group defined by the groupby columns. This follows the methodology
    described in Hoffman & Roussos (2024) dreamlet package.

    Parameters
    ----------
    ad : AnnData
        Single-cell expression object with raw integer counts.
        Non-integer values will trigger a warning and be truncated.
    groupby : List[str]
        Metadata columns to group by.

        The first column defines assay labels (values become the `assays`
        column in output). Remaining columns define sample replicates
        within each assay.

        Example: `groupby=["celltype", "donor"]` creates samples for each
        celltype-donor combination, with `assays` containing celltype values.

        .. note::
            If you want all samples in a single assay (e.g., pre-filtered
            data for one cell type), add a constant column first::

                ad.obs['assay'] = 'Tcells'
                pb = aggregate_pseudobulk(ad, groupby=['assay', 'donor'])

    layer : str, optional
        Layer containing count data. If None, uses ad.X.

    Returns
    -------
    pb : AnnData
        Pseudobulk AnnData object with:

        - **X**: Summed counts (samples × genes), integer dtype
        - **obs**: Sample metadata including:
          - `assays`: Assay labels (from first groupby column values)
          - `n_cells`: Number of cells aggregated per sample
          - `total_counts`: Sum of counts per sample
          - All constant metadata columns from input
        - **var**: Gene metadata (copied from input)

    Examples
    --------
    Multi-assay aggregation (typical use):

    >>> import dreampy as dp
    >>> pb = dp.aggregate_pseudobulk(
    ...     adata,
    ...     groupby=["celltype", "donor"],
    ...     layer="counts"
    ... )
    >>> pb.obs['assays'].unique()  # ['Bcell', 'Tcell', ...]
    >>> pb.shape  # (n_celltype_donor_combinations, n_genes)

    Single-assay aggregation (pre-filtered to one cell type):

    >>> tcells = adata[adata.obs['celltype'] == 'Tcell']
    >>> tcells.obs['assay'] = 'Tcell'  # Add constant column for assay label
    >>> pb = dp.aggregate_pseudobulk(tcells, groupby=["assay", "donor"])
    >>> pb.obs['assays'].unique()  # ['Tcell']
    >>> pb.shape  # (n_donors, n_genes)

    Notes
    -----
    **Input data requirements:**

    This function expects raw integer counts. If non-integer values are
    detected, a warning is issued and values are truncated to integers.
    Do not pass normalized or log-transformed data.

    **Groupby column order matters:**

    The first column in `groupby` determines the `assays` column in the
    output, which is used by downstream functions like `de()` to identify
    cell types for multi-assay analysis. Values from this column become
    the assay labels.

    If you want all samples in a single assay, add a constant column:
    ``ad.obs['assay'] = 'MyAssay'`` then include it as the first groupby column.

    **Downstream usage:**

    The returned AnnData can be passed directly to:
    - `de(pb, formula, assay_col="assays")` for multi-assay DE
    - `de_assay(pb[pb.obs['assays'] == 'Tcell'], formula)` for single-assay DE
    - Tier 1 functions after extracting `.X` and `.obs`

    See Also
    --------
    filter_samples : Filter pseudobulk samples by cell count
    de : Multi-assay differential expression
    de_assay : Single-assay differential expression
    """
    # Validation
    validate_columns(ad, groupby, context="groupby")
    validate_assay(ad, layer)

    obs = ad.obs[groupby].copy()
 
    # Get expression matrix
    if layer is None:
        X = ad.X
    else:
        X = ad.layers[layer]

    # Ensure sparse CSR format for efficient operations
    if not sparse.issparse(X):
        X_sparse = sparse.csr_matrix(X)
    else:
        X_sparse = X.tocsr() if not isinstance(X, sparse.csr_matrix) else X

    # Check for non-integer values and warn
    if hasattr(X_sparse, "data") and len(X_sparse.data) > 0:
        if not np.allclose(X_sparse.data, np.round(X_sparse.data)):
            warnings.warn(
                "Input contains non-integer values. aggregate_pseudobulk() expects "
                "raw counts. Non-integer values will be truncated to integers.",
                UserWarning, stacklevel=2
            )

    # Create group identifiers 
    group_ids = obs.astype(str).apply(tuple, axis=1)

    # Find unique groups and their indices
    group_indices, unique_groups = pd.factorize(group_ids, sort=True)
    n_groups = len(unique_groups)

    # Count the number of cells in each group
    group_sizes = np.bincount(group_indices, minlength=n_groups)

    # Build observation metadata
    obs_dict = {}
    for col in ad.obs.columns:
        obs_dict[col] = []
        is_constant = True

        for group_idx in range(n_groups):
            mask = group_indices == group_idx
            group_values = ad.obs.loc[mask, col].unique()

            if len(group_values) == 1:
                obs_dict[col].append(group_values[0])
            else:
                is_constant = False
                break

        if not is_constant:
            del obs_dict[col]

    # Add number of cells per group
    obs_dict["n_cells"] = group_sizes

    # Add assays column (always uses values from first groupby column)
    obs_dict["assays"] = obs_dict[groupby[0]]

    # Make obs and var DataFrames
    pb_obs = pd.DataFrame(obs_dict, index=unique_groups)
    pb_var = ad.var.copy()

    # Reorder pb_obs to place assays and groupby columns in front
    cols_reorder = (
        ["assays"] + groupby + [col for col in pb_obs.columns if col not in ["assays"] + groupby]
    )
    pb_obs = pb_obs.loc[:, cols_reorder].copy()

    # Create cell-to-group mapping as sparse matrix
    group_cell_matrix = sparse.csr_matrix(
        (np.ones(len(group_indices)), (group_indices, np.arange(len(group_indices)))),
        shape=(n_groups, len(group_indices)),
    )

    # Generate the sums via sparse matrix multiplication
    group_sums = group_cell_matrix @ X_sparse

    # Create the pseudobulk AnnData
    pb = AnnData(X=group_sums.astype(int), obs=pb_obs, var=pb_var)
    pb.obs["total_counts"] = np.asarray(pb.X.sum(axis=1)).flatten().astype(int)

    return pb