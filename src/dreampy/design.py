"""
Design Matrix Cleaning in dreampy

Functions for removing constant and redundant terms from formulas and design matrices.
Matches variancePartition/limma R implementations.

Key Functions (Public):
- remove_constant_terms(): Formula-level constant term removal
- drop_redundant_terms(): Formula-level collinearity detection via correlation
- drop_redundant_columns_qr(): Matrix-level QR-based rank deficiency detection
- clean_design_matrix(): Full pipeline combining all three steps

These are called BEFORE fitting models to ensure design matrix is full rank.
"""

import difflib
import re
import warnings
from collections.abc import Collection

import numpy as np
import pandas as pd
from scipy import linalg


def remove_constant_terms(
    formula: str,
    data: pd.DataFrame,
    min_unique: int = 2,
) -> tuple[str, list[str]]:
    """
    Remove constant terms from formula based on data.

    Parameters
    ----------
    formula : str
        Original formula string (e.g., "~ age + sex + (1|donor)")
    data : pd.DataFrame
        Metadata DataFrame with columns referenced in formula
    min_unique : int
        Minimum number of unique values required (default 2)

    Returns
    -------
    formula_modified : str
        Formula with constant terms removed
    dropped_warnings : List[str]
        List of warning messages describing dropped terms

    Notes
    -----
    A term is constant if:
    - All values in data are identical (numeric)
    - Factor has only one level observed
    - Factor has fewer than min_unique unique values

    The intercept is never removed.
    Random effects terms like (1|group) are checked for the grouping variable.

    Reference: variancePartition::removeConstantTerms()
    """
    dropped_warnings = []

    # Parse formula into components
    # Extract RHS only (handle "y ~ predictors" or "~ predictors")
    formula_body = formula.strip()
    if "~" in formula_body:
        _, formula_body = formula_body.split("~", 1)
        formula_body = formula_body.strip()

    # Split by + but preserve parenthetical terms
    terms = _split_formula_terms(formula_body)

    kept_terms = []
    for term in terms:
        term = term.strip()
        if not term:
            continue

        # Check if this is a random effect term (1|group)
        re_match = re.match(r"\(1\s*\|\s*(\w+)\)", term)
        if re_match:
            group_var = re_match.group(1)
            if group_var in data.columns:
                n_unique = data[group_var].nunique()
                if n_unique < min_unique:
                    dropped_warnings.append(
                        f"Dropped random effect (1|{group_var}): only {n_unique} level(s)"
                    )
                    continue
            kept_terms.append(term)
            continue

        # Check if term is a variable in data
        # Handle simple variable names
        var_name = term.strip()

        if var_name in data.columns:
            col = data[var_name]
            n_unique = col.nunique()

            if n_unique < min_unique:
                dropped_warnings.append(f"Dropped '{var_name}': only {n_unique} unique value(s)")
                continue

        kept_terms.append(term)

    # Reconstruct formula
    if kept_terms:
        formula_modified = "~ " + " + ".join(kept_terms)
    else:
        formula_modified = "~ 1"  # Intercept only

    return formula_modified, dropped_warnings


def drop_redundant_terms(
    formula: str,
    data: pd.DataFrame,
    tol: float = 1e-3,
    reference_coded_terms: Collection[str] = (),
) -> tuple[str, list[str]]:
    """
    Drop redundant (collinear) terms from formula using correlation detection.

    Detects co-linear fixed effects and drops the last one in each pair.
    Only columns that belong to different terms are compared. The indicator
    columns of a single categorical term are never tested against each other:
    the two indicators of a two-level factor in a no-intercept design have
    correlation -1 by construction, which is not redundancy.

    Parameters
    ----------
    formula : str
        Formula string (after removeConstantTerms)
    data : pd.DataFrame
        Metadata DataFrame
    tol : float
        Tolerance for correlation = 1 detection (default 1e-3)
        Terms with |r| > 1 - tol are considered redundant
    reference_coded_terms : Collection[str]
        Categorical terms that must always be reference coded (first level
        dropped), even in a no-intercept formula. clean_design_matrix() passes
        the random effect grouping variables here, because it temporarily
        lists them as fixed terms for this check and they must not be mistaken
        for the first fixed factor.

    Returns
    -------
    formula_modified : str
        Formula with redundant terms removed
    dropped_warnings : List[str]
        Descriptions of dropped terms with collinearity info

    Notes
    -----
    Algorithm from variancePartition::dropRedundantTerms():
    1. Extract fixed effect terms from formula
    2. Create design matrix from formula (fixed effects only)
    3. Compute correlation matrix of predictors
    4. For each pair with |r| > 1 - tol: drop the later term
    5. Return modified formula and warnings

    Random effects (1|group) are preserved without checking.

    Reference: variancePartition::dropRedundantTerms()
    """
    dropped_warnings = []

    # Parse formula - extract RHS only (handle "y ~ predictors" or "~ predictors")
    formula_body = formula.strip()
    if "~" in formula_body:
        _, formula_body = formula_body.split("~", 1)
        formula_body = formula_body.strip()

    terms = _split_formula_terms(formula_body)

    # Separate fixed and random effects
    fixed_terms = []
    random_terms = []

    for term in terms:
        term = term.strip()
        if not term:
            continue
        if re.match(r"\(.*\|.*\)", term):
            random_terms.append(term)
        else:
            fixed_terms.append(term)

    if len(fixed_terms) <= 1:
        # Nothing to compare
        if fixed_terms or random_terms:
            all_terms = fixed_terms + random_terms
            return "~ " + " + ".join(all_terms), []
        return "~ 1", []

    # Build design matrix for fixed effects
    try:
        X, col_names, col_to_term = _build_simple_design_matrix(
            fixed_terms, data, reference_coded_terms=reference_coded_terms
        )
    except (ValueError, TypeError) as e:
        # Can't build design matrix, return original
        warnings.warn(f"Could not build design matrix for redundancy check: {e}", stacklevel=2)
        all_terms = fixed_terms + random_terms
        return "~ " + " + ".join(all_terms), []

    if X.shape[1] <= 1:
        all_terms = fixed_terms + random_terms
        return "~ " + " + ".join(all_terms), []

    # Compute correlation matrix (skip intercept column if present)
    # Standardize columns
    X_std = X.copy()
    col_stds = np.std(X_std, axis=0, ddof=1)
    col_means = np.mean(X_std, axis=0)

    # Don't standardize constant columns (intercept)
    variable_cols = col_stds > 1e-10
    X_std[:, variable_cols] = (X_std[:, variable_cols] - col_means[variable_cols]) / col_stds[
        variable_cols
    ]

    # Compute correlation matrix
    n = X_std.shape[0]
    corr_matrix = X_std.T @ X_std / (n - 1)

    # Find highly correlated pairs
    keep_cols = np.ones(len(col_names), dtype=bool)

    for i in range(len(col_names)):
        if not keep_cols[i]:
            continue
        for j in range(i + 1, len(col_names)):
            if not keep_cols[j]:
                continue

            # Columns of the same term are not compared with each other
            if col_to_term[i] is not None and col_to_term[i] == col_to_term[j]:
                continue

            r = corr_matrix[i, j]
            if np.abs(r) > 1 - tol:
                # Drop the later term (j)
                keep_cols[j] = False
                dropped_warnings.append(
                    f"Dropped '{col_names[j]}': correlated with '{col_names[i]}' (r={r:.4f})"
                )

    # Map dropped columns back to parent terms
    terms_to_drop = set()
    for i, keep in enumerate(keep_cols):
        if not keep and col_to_term[i] is not None:
            terms_to_drop.add(col_to_term[i])

    kept_fixed = [t for t in fixed_terms if t not in terms_to_drop]

    # If all fixed terms were dropped, keep intercept
    if not kept_fixed:
        kept_fixed = ["1"]

    # Reconstruct formula
    all_kept = kept_fixed + random_terms
    formula_modified = "~ " + " + ".join(all_kept)

    return formula_modified, dropped_warnings


def drop_redundant_columns_qr(
    X: np.ndarray,
    col_names: list[str],
    tol: float = 1e-10,
) -> tuple[np.ndarray, list[str], list[str]]:
    """
    Drop rank-deficient columns using QR decomposition.

    This is the authoritative matrix-level check that catches
    any collinearity that formula-level methods missed.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n_samples, n_columns)
    col_names : List[str]
        Names for each column
    tol : float
        Tolerance for rank detection (default 1e-10)

    Returns
    -------
    X_clean : np.ndarray
        Design matrix with redundant columns removed
    col_names_clean : List[str]
        Updated column names
    dropped : List[str]
        Names of dropped columns

    Notes
    -----
    Uses pivoted QR decomposition (lme4's chkRank.drop.cols approach).
    If R[i,i] < tol * max(abs(diag(R))), column i is redundant.

    This provides a final safety net after formula-level pruning.

    Algorithm:
    1. Compute Q, R, P = qr(X, pivoting=True)
    2. Find effective rank from diagonal of R
    3. Keep only columns corresponding to non-zero diagonals
    4. Return cleaned matrix in original column order

    Reference: lme4's chkRank.drop.cols
    """
    n, p = X.shape

    if p == 0:
        return X, col_names, []

    if p == 1:
        # Single column - check if it's all zeros
        if np.all(np.abs(X) < tol):
            return np.empty((n, 0)), [], col_names
        return X, col_names, []

    # QR decomposition with column pivoting
    _Q, R, perm = linalg.qr(X, pivoting=True, mode="economic")

    # Get absolute diagonal of R
    diag_R = np.abs(np.diag(R))

    # Threshold for rank detection
    threshold = tol * np.max(diag_R) if len(diag_R) > 0 else tol

    # Determine effective rank
    rank_mask = diag_R >= threshold
    effective_rank = np.sum(rank_mask)

    if effective_rank == p:
        # Full rank - nothing to drop
        return X, col_names, []

    # Identify which original columns to keep
    # perm maps new column index -> original column index
    # Keep columns corresponding to non-zero diagonals
    keep_perm_indices = np.where(rank_mask)[0]
    keep_original_indices = perm[keep_perm_indices]

    # Sort to maintain original order
    keep_original_indices = np.sort(keep_original_indices)

    # Determine dropped columns
    all_indices = set(range(p))
    kept_indices = set(keep_original_indices)
    dropped_indices = sorted(all_indices - kept_indices)

    dropped = [col_names[i] for i in dropped_indices]
    col_names_clean = [col_names[i] for i in keep_original_indices]
    X_clean = X[:, keep_original_indices]

    return X_clean, col_names_clean, dropped


def clean_design_matrix(
    formula: str,
    data: pd.DataFrame,
    X: np.ndarray | None = None,
    col_names: list[str] | None = None,
    constant_tol: int = 2,
    corr_tol: float = 1e-3,
    qr_tol: float = 1e-10,
    verbose: bool = True,
) -> tuple[str, np.ndarray, list[str], list[str]]:
    """
    Full design matrix cleaning pipeline.

    Applies all three cleaning steps in order:
    1. remove_constant_terms() - formula level
    2. drop_redundant_terms() - formula level, correlation-based
    3. drop_redundant_columns_qr() - matrix level, QR-based

    Random effect grouping variables are included in collinearity checking.
    When a random effect is collinear with a fixed effect, the random effect
    is dropped (fixed effects take precedence).

    Parameters
    ----------
    formula : str
        Original formula string (e.g., "~ treatment + (1|donor)")
    data : pd.DataFrame
        Metadata DataFrame
    X : np.ndarray, optional
        Pre-built design matrix. If provided, RE collinearity checking is
        skipped (only QR check is performed). A warning is issued.
    col_names : List[str], optional
        Column names for X. Required if X is provided.
    constant_tol : int
        Minimum unique values for remove_constant_terms (default 2)
    corr_tol : float
        Correlation tolerance for drop_redundant_terms (default 1e-3)
    qr_tol : float
        QR tolerance for drop_redundant_columns_qr (default 1e-10)
    verbose : bool
        Print summary and warnings about dropped terms (default True)

    Returns
    -------
    formula_clean : str
        Cleaned formula with redundant terms removed
    X_clean : np.ndarray
        Cleaned design matrix (fixed effects only)
    col_names_clean : List[str]
        Column names after cleaning
    all_warnings : List[str]
        All warning messages from cleaning steps

    Notes
    -----
    Collinearity checking includes:
    - Fixed effect ↔ Fixed effect: drops later term
    - Fixed effect ↔ Random effect: drops the random effect
    - Random effect ↔ Random effect: drops later random effect

    Reference: variancePartition/limma R implementations
    """
    all_warnings = []

    # =========================================================================
    # HANDLE PRE-BUILT X (backward compatibility)
    # =========================================================================

    if X is not None:
        # Pre-built X provided - skip RE-aware preprocessing
        # We can't identify which columns are RE grouping vars
        if col_names is None:
            raise ValueError("col_names required when X is provided")

        all_warnings.append(
            "Pre-built X provided: skipping random effect collinearity checking. "
            "Only QR-based rank deficiency check will be performed."
        )

        # Only do QR check on provided matrix
        X_clean, col_names_clean, dropped = drop_redundant_columns_qr(X, col_names, tol=qr_tol)

        if dropped:
            for d in dropped:
                all_warnings.append(f"QR cleanup: dropped '{d}' (rank deficient)")

        if verbose:
            for w in all_warnings:
                warnings.warn(w, stacklevel=2)

        # Return formula unchanged (we can't clean it without knowing the mapping)
        return formula, X_clean, col_names_clean, all_warnings

    # =========================================================================
    # PHASE 1: PREPROCESSING - Flatten RE to fixed for collinearity checking
    # =========================================================================

    # Parse original formula - extract RHS only (handle "y ~ predictors" or "~ predictors")
    formula_body = formula.strip()
    if "~" in formula_body:
        _, formula_body = formula_body.split("~", 1)
        formula_body = formula_body.strip()

    original_terms = _split_formula_terms(formula_body)

    # Separate and track RE vs fixed terms
    fixed_terms = []
    re_grouping_vars = {}  # Maps grouping var -> original RE syntax, e.g. 'donor' -> '(1|donor)'

    for term in original_terms:
        term = term.strip()
        if not term:
            continue

        # Check if this is a random effect term using loose matching
        re_match = re.match(r"\(([^|]+)\|([^)]+)\)", term)
        if re_match:
            effect = re_match.group(1).strip()
            group_var = re_match.group(2).strip()

            # Validate only random intercepts supported
            if effect != "1":
                raise ValueError(
                    f"Only random intercepts supported, got '({effect}|{group_var})'. "
                    f"Random slopes like '({effect}|{group_var})' are not implemented."
                )

            # Check grouping variable exists in data
            if group_var not in data.columns:
                raise ValueError(
                    f"Random effect grouping variable '{group_var}' not found in metadata. "
                    f"Available columns: {list(data.columns)}"
                )

            re_grouping_vars[group_var] = term  # e.g., 'donor' -> '(1|donor)'
        else:
            fixed_terms.append(term)

    # If no RE terms, use original flow (slight optimization)
    if not re_grouping_vars:
        # No random effects - use existing flow unchanged
        formula_step1, warnings1 = remove_constant_terms(formula, data, min_unique=constant_tol)
        all_warnings.extend(warnings1)

        formula_step2, warnings2 = drop_redundant_terms(formula_step1, data, tol=corr_tol)
        all_warnings.extend(warnings2)

        # Build design matrix - extract RHS only (handle "y ~ predictors" or "~ predictors")
        formula_body_clean = formula_step2.strip()
        if "~" in formula_body_clean:
            _, formula_body_clean = formula_body_clean.split("~", 1)
            formula_body_clean = formula_body_clean.strip()

        terms_clean = _split_formula_terms(formula_body_clean)
        fixed_terms_clean = [t.strip() for t in terms_clean if t.strip()]

        try:
            X, col_names, _ = _build_simple_design_matrix(fixed_terms_clean, data)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Could not build design matrix: {e}") from e

        # QR-based cleanup
        X_clean, col_names_clean, dropped = drop_redundant_columns_qr(X, col_names, tol=qr_tol)

        if dropped:
            for d in dropped:
                all_warnings.append(f"QR cleanup: dropped '{d}' (rank deficient)")

        if verbose:
            n_fixed = len([t for t in fixed_terms_clean if t != "1" and t.lower() != "intercept"])
            print(f"Formula cleaned: {n_fixed} fixed effect(s), 0 random effect(s)")
            if all_warnings:
                for w in all_warnings:
                    warnings.warn(w, stacklevel=2)

        return formula_step2, X_clean, col_names_clean, all_warnings

    # =========================================================================
    # RE TERMS PRESENT - First remove constant RE, then check collinearity
    # =========================================================================

    # First, filter out RE grouping vars that only have 1 level (constant)
    # This must happen BEFORE collinearity checks to avoid misleading warnings
    constant_re = []
    for var, re_syntax in list(re_grouping_vars.items()):
        n_levels = data[var].nunique()
        if n_levels < constant_tol:
            constant_re.append(var)
            all_warnings.append(f"Dropped random effect {re_syntax}: only {n_levels} level(s)")
            del re_grouping_vars[var]

    # Now check RE-level collinearity for remaining RE terms
    # The existing drop_redundant_terms() has a bug with categorical variables:
    # it doesn't properly map dropped columns back to parent terms.
    # So we do term-level collinearity detection for RE grouping vars here.

    # Check RE ↔ RE collinearity (same grouping structure)
    re_vars_list = list(re_grouping_vars.keys())
    re_to_drop = set()

    for i, var1 in enumerate(re_vars_list):
        if var1 in re_to_drop:
            continue
        for j, var2 in enumerate(re_vars_list):
            if j <= i or var2 in re_to_drop:
                continue

            # Check if var1 and var2 have identical grouping structure
            # (i.e., each level of var1 maps to exactly one level of var2 and vice versa)
            vals1 = data[var1].values
            vals2 = data[var2].values

            # Two variables are collinear if they have identical group memberships
            # Check by seeing if sorting by one also sorts by the other
            pairs = list(zip(vals1, vals2))
            unique_pairs = set(pairs)
            unique_v1 = set(vals1)
            unique_v2 = set(vals2)

            # If |unique_pairs| == |unique_v1| == |unique_v2|, it's a 1:1 mapping (collinear)
            if len(unique_pairs) == len(unique_v1) == len(unique_v2):
                # var2 is collinear with var1, drop var2 (later one)
                re_to_drop.add(var2)
                all_warnings.append(
                    f"Dropped random effect {re_grouping_vars[var2]}: "
                    f"collinear with random effect {re_grouping_vars[var1]} (identical grouping structure)"
                )

    # Check RE ↔ Fixed collinearity
    # An RE grouping var is collinear with fixed effects if the grouping is
    # fully determined by the fixed effects (each combination of fixed effect
    # values maps to exactly one RE group)
    for re_var in re_vars_list:
        if re_var in re_to_drop:
            continue

        re_vals = data[re_var].values

        for fixed_term in fixed_terms:
            if fixed_term == "1" or fixed_term.lower() == "intercept":
                continue
            if fixed_term not in data.columns:
                continue

            # Skip constant fixed effects - they'll be handled by remove_constant_terms
            if data[fixed_term].nunique() < constant_tol:
                continue

            fixed_vals = data[fixed_term].values

            # Check if fixed_term determines re_var (1:1 mapping)
            pairs = list(zip(fixed_vals, re_vals))
            unique_pairs = set(pairs)
            unique_fixed = set(fixed_vals)
            unique_re = set(re_vals)

            if len(unique_pairs) == len(unique_fixed) == len(unique_re):
                # RE is determined by this fixed effect, drop RE
                re_to_drop.add(re_var)
                all_warnings.append(
                    f"Dropped random effect {re_grouping_vars[re_var]}: "
                    f"collinear with fixed effect '{fixed_term}'"
                )
                break

    # Remove dropped RE grouping vars
    for var in re_to_drop:
        del re_grouping_vars[var]

    # Build flattened formula: FIXED FIRST, then surviving RE grouping vars
    # This ensures RE is dropped (not fixed) when they're collinear
    flattened_terms = fixed_terms + list(re_grouping_vars.keys())

    if flattened_terms:
        flattened_formula = "~ " + " + ".join(flattened_terms)
    else:
        flattened_formula = "~ 1"

    # =========================================================================
    # PHASE 2: CALL EXISTING FUNCTIONS (unchanged)
    # =========================================================================

    # Step 1: Remove constant terms (now checks all vars including RE grouping vars)
    formula_step1, warnings1 = remove_constant_terms(
        flattened_formula, data, min_unique=constant_tol
    )
    all_warnings.extend(warnings1)

    # Step 2: Drop redundant terms - primarily for fixed effect collinearity
    # (RE collinearity already handled above at term level)
    # The grouping variables are listed as fixed terms only for this check, so
    # they are always reference coded, including in a no-intercept formula.
    formula_step2, warnings2 = drop_redundant_terms(
        formula_step1,
        data,
        tol=corr_tol,
        reference_coded_terms=set(re_grouping_vars),
    )
    all_warnings.extend(warnings2)

    # =========================================================================
    # PHASE 3: POSTPROCESSING - Convert RE grouping vars back to RE syntax
    # =========================================================================

    # Parse surviving terms from flattened formula
    # Extract RHS only (handle "y ~ predictors" or "~ predictors")
    formula_body_clean = formula_step2.strip()
    if "~" in formula_body_clean:
        _, formula_body_clean = formula_body_clean.split("~", 1)
        formula_body_clean = formula_body_clean.strip()

    surviving_terms = _split_formula_terms(formula_body_clean)

    # Separate into final fixed and RE terms
    final_fixed = []
    final_re = []

    for term in surviving_terms:
        term = term.strip()
        if not term:
            continue

        if term in re_grouping_vars:
            # This was an RE grouping var - convert back to RE syntax
            final_re.append(re_grouping_vars[term])
        else:
            final_fixed.append(term)

    # Enhance warnings to clarify when RE was dropped
    enhanced_warnings = []
    for w in all_warnings:
        enhanced = w

        # Check if warning mentions a dropped RE grouping var
        for var, re_syntax in re_grouping_vars.items():
            if f"Dropped '{var}'" in w:
                # Check what it was collinear with
                collinear_with_re = False
                for other_var, other_re in re_grouping_vars.items():
                    if other_var != var and f"correlated with '{other_var}'" in w:
                        # RE collinear with another RE
                        enhanced = w.replace(
                            f"Dropped '{var}'", f"Dropped random effect {re_syntax}"
                        ).replace(
                            f"correlated with '{other_var}'",
                            f"collinear with random effect {other_re}",
                        )
                        collinear_with_re = True
                        break

                if not collinear_with_re:
                    # RE collinear with fixed effect (or constant)
                    enhanced = w.replace(f"Dropped '{var}'", f"Dropped random effect {re_syntax}")
                break

        enhanced_warnings.append(enhanced)

    all_warnings = enhanced_warnings

    # Reconstruct final formula: fixed terms first, then RE terms
    final_parts = final_fixed + final_re

    if final_parts:
        formula_clean = "~ " + " + ".join(final_parts)
    else:
        # All terms dropped - this is an error
        raise ValueError(
            "Cannot fit model: no terms remaining after cleaning. "
            "All fixed effects and random effects were dropped due to "
            "constant values or collinearity."
        )

    # Check if we ended up with only intercept + RE (no real fixed effects)
    real_fixed = [t for t in final_fixed if t != "1" and t.lower() != "intercept"]
    if not real_fixed and final_re:
        all_warnings.append(
            "All fixed effects were dropped. Model will have intercept only "
            f"with random effects: {', '.join(final_re)}"
        )

    # Check if all RE were dropped
    if not final_re and re_grouping_vars:
        all_warnings.append("All random effects were dropped. Model will be fixed-effects only.")

    # =========================================================================
    # PHASE 4: BUILD FINAL DESIGN MATRIX (fixed effects only)
    # =========================================================================

    try:
        X, col_names, _ = _build_simple_design_matrix(final_fixed, data)
    except (ValueError, TypeError) as e:
        raise ValueError(f"Could not build design matrix: {e}") from e

    # QR-based cleanup (safety net)
    X_clean, col_names_clean, dropped = drop_redundant_columns_qr(X, col_names, tol=qr_tol)

    if dropped:
        for d in dropped:
            all_warnings.append(f"QR cleanup: dropped '{d}' (rank deficient)")

    # =========================================================================
    # VERBOSE OUTPUT
    # =========================================================================

    if verbose:
        n_fixed = len(real_fixed)
        n_re = len(final_re)
        print(f"Formula cleaned: {n_fixed} fixed effect(s), {n_re} random effect(s)")
        if all_warnings:
            for w in all_warnings:
                warnings.warn(w, stacklevel=2)

    return formula_clean, X_clean, col_names_clean, all_warnings


def build_model_matrices(
    formula: str,
    metadata: pd.DataFrame,
    verbose: bool = True,
    constant_tol: int = 2,
    corr_tol: float = 1e-3,
    qr_tol: float = 1e-10,
) -> tuple[
    np.ndarray, list[str], np.ndarray | None, list[int] | None, list[str] | None, str, list[str]
]:
    """
    Build all model matrices from a formula and metadata.

    This is the single entry point for design matrix construction. It calls
    clean_design_matrix() for the fixed effects, then builds the random effects
    indicator matrix Z from the cleaned formula. Called once per assay — never
    per gene.

    Parameters
    ----------
    formula : str
        Model formula, e.g. "~ diagnosis + (1|donor) + (1|batch)"
    metadata : pd.DataFrame
        Sample metadata with columns for all formula variables.
        Rows = samples.
    verbose : bool
        Print summary and warnings about dropped terms (default True)
    constant_tol, corr_tol, qr_tol
        Tolerances passed through to clean_design_matrix().

    Returns
    -------
    X : np.ndarray (n, p)
        Fixed effects design matrix (includes intercept by default)
    coef_names : List[str]
        Column names for X, e.g. ['(Intercept)', 'diagnosisdcm']
    Z : np.ndarray (n, q_total) or None
        Random effects indicator matrix. None for FE-only formulas.
    block_sizes : List[int] or None
        Number of groups per RE term, e.g. [23, 5]. None for FE-only.
    re_names : List[str] or None
        RE term names, e.g. ['donor', 'batch']. None for FE-only.
    formula_clean : str
        Cleaned formula after constant/collinearity removal
    all_warnings : List[str]
        All cleaning warnings

    Raises
    ------
    ValueError
        If the formula contains a term that cannot be used: a variable that is
        not a column of the metadata, an interaction, a transformation or
        function call, removal of a term other than the intercept, or a random
        effect other than a random intercept on a single metadata column.
        Also if any variable used by the formula has missing values. Nothing
        is dropped silently in either case.

    Notes
    -----
    The returned Z and block_sizes are constant for a given assay and should be
    passed to fit_lmm / fit_genes_parallel directly. They must NOT be rebuilt
    per gene.

    The no-intercept forms '~ 0 + x', '~ -1 + x' and '~ x - 1' are equivalent.
    """
    from .formula import Formula

    # Step 0: Reject formulas and metadata that cannot be fitted as written.
    # This runs before any cleaning, so that an unusable term raises an error
    # here and is never skipped further down.
    formula_str = formula if isinstance(formula, str) else formula.original
    formula_str = _validate_formula(formula_str, metadata)

    # Step 1: Clean formula and build X
    formula_clean, X, coef_names, all_warnings = clean_design_matrix(
        formula_str,
        metadata,
        verbose=verbose,
        constant_tol=constant_tol,
        corr_tol=corr_tol,
        qr_tol=qr_tol,
    )

    # Step 2: Parse cleaned formula for surviving RE terms
    f = Formula(formula_clean)

    if not f.has_random_effects():
        return X, coef_names, None, None, None, formula_clean, all_warnings

    # Step 3: Extract RE grouping variables and build Z
    re_names_out: list[str] = []
    Z_blocks: list[np.ndarray] = []
    block_sizes_out: list[int] = []
    n_obs = len(metadata)

    for effect, group_var in f.random_terms:
        if effect.strip() != "1":
            raise ValueError(f"Only random intercepts supported, got ({effect}|{group_var})")
        if group_var not in metadata.columns:
            raise ValueError(
                f"Grouping variable '{group_var}' not found in metadata. "
                f"Available columns: {list(metadata.columns)}"
            )

        groups = metadata[group_var].values
        unique_groups = np.unique(groups)
        n_groups = len(unique_groups)

        # Build indicator matrix for this term
        group_to_idx = {g: i for i, g in enumerate(unique_groups)}
        Z_block = np.zeros((n_obs, n_groups))
        for i, g in enumerate(groups):
            Z_block[i, group_to_idx[g]] = 1.0

        Z_blocks.append(Z_block)
        block_sizes_out.append(n_groups)
        re_names_out.append(group_var)

    Z = np.hstack(Z_blocks)

    return X, coef_names, Z, block_sizes_out, re_names_out, formula_clean, all_warnings


# =============================================================================
# Private helper functions
# =============================================================================


# Matches a trailing "- 1" on a formula term, e.g. "disease - 1" or "-1".
_TRAILING_MINUS_ONE = re.compile(r"(.*?)\s*-\s*1")

# Matches a random effect term "(effect | group)".
_RANDOM_EFFECT_TERM = re.compile(r"\(([^|]*)\|([^)]*)\)")


def _has_top_level_minus(term: str) -> bool:
    """True if the term contains a '-' that is not inside parentheses."""
    depth = 0
    for char in term:
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif char == "-" and depth == 0:
            return True
    return False


def _describe_unusable_term(term: str, columns: list[str]) -> str:
    """Explain why a fixed effect term that is not a metadata column cannot be used."""
    if _has_top_level_minus(term):
        return (
            f"'{term}': removing a term with '-' is not supported. "
            "Only '- 1' (no intercept) is recognised."
        )
    if "(" in term or ")" in term:
        return (
            f"'{term}': transformations and function calls are not supported. "
            "Compute the transformed variable as a new metadata column and use that column."
        )
    if ":" in term or "*" in term:
        return f"'{term}': interaction terms are not supported."
    if "|" in term:
        return f"'{term}': a random effect must be written in parentheses, for example '(1|donor)'."
    if "/" in term or "^" in term or "%" in term:
        return f"'{term}': the operators '/', '^' and '%in%' are not supported."

    message = f"'{term}' is not a column in the metadata."
    close = difflib.get_close_matches(term, columns, n=3, cutoff=0.6)
    if close:
        message += " Did you mean " + " or ".join(f"'{c}'" for c in close) + "?"
    return message


def _validate_formula(formula: str, data: pd.DataFrame) -> str:
    """
    Check that every term of a formula can be used, and normalise the formula.

    A term is usable if it is an intercept control ('0', '1', '-1',
    'intercept'), the name of a metadata column, or a random intercept
    '(1|group)' whose grouping variable is a metadata column. Any other term
    raises an error. Terms are never dropped.

    A trailing '- 1' on a term, as in '~ disease - 1', suppresses the
    intercept and is rewritten to the equivalent '~ 0 + disease'. A column
    whose own name contains '-' is matched as a column first, so a column
    called 'cell-type' is still usable.

    Every variable used by the formula is then checked for missing values.

    Parameters
    ----------
    formula : str
        Formula string, with or without a left-hand side
    data : pd.DataFrame
        Sample metadata

    Returns
    -------
    formula_normalised : str
        The same model, with any intercept suppression written as a leading
        '0'. Terms keep their original order.

    Raises
    ------
    ValueError
        If any term cannot be used. All unusable terms are reported together.
    ValueError
        If any variable used by the formula has missing values. All such
        variables are reported together, with the number of samples affected.
    """
    formula_body = formula.strip()
    if "~" in formula_body:
        _, formula_body = formula_body.split("~", 1)
        formula_body = formula_body.strip()

    columns = [str(c) for c in data.columns]
    column_set = set(columns)

    suppress_intercept = False
    kept_terms: list[str] = []
    used_variables: list[str] = []
    problems: list[str] = []

    for raw_term in _split_formula_terms(formula_body):
        term = raw_term.strip()

        # Strip any trailing "- 1", unless what we have is itself a column name
        while term not in column_set:
            match = _TRAILING_MINUS_ONE.fullmatch(term)
            if match is None:
                break
            suppress_intercept = True
            term = match.group(1).strip()

        if not term:
            continue

        if term == "0":
            suppress_intercept = True
            continue

        if term in column_set:
            kept_terms.append(term)
            used_variables.append(term)
            continue

        if term == "1" or term.lower() == "intercept":
            kept_terms.append(term)
            continue

        random_effect = _RANDOM_EFFECT_TERM.fullmatch(term)
        if random_effect is not None:
            effect = random_effect.group(1).strip()
            group_var = random_effect.group(2).strip()
            if effect != "1":
                problems.append(
                    f"'{term}': only random intercepts are supported, written '(1|{group_var})'. "
                    "Random slopes are not implemented."
                )
            elif group_var not in column_set:
                detail = _describe_unusable_term(group_var, columns)
                problems.append(f"'{term}': the grouping variable cannot be used. {detail}")
            else:
                # Canonical spelling, so later steps see one form of the term
                kept_terms.append(f"(1|{group_var})")
                used_variables.append(group_var)
            continue

        problems.append(_describe_unusable_term(term, columns))

    if problems:
        raise ValueError(
            f"Formula '{formula.strip()}' cannot be used:\n"
            + "\n".join(f"  - {problem}" for problem in problems)
            + f"\nAvailable metadata columns: {columns}"
        )

    # Missing values in any variable the formula uses
    missing = []
    for variable in dict.fromkeys(used_variables):  # unique, in formula order
        n_missing = int(data[variable].isna().sum())
        if n_missing > 0:
            missing.append(f"'{variable}' ({n_missing} of {len(data)} samples)")
    if missing:
        raise ValueError(
            "Missing values in metadata variables used by formula "
            f"'{formula.strip()}': " + ", ".join(missing) + ". "
            "Samples are not dropped automatically. Remove these samples from both the "
            "expression data and the metadata, or fill in the values, before fitting."
        )

    all_terms = (["0"] if suppress_intercept else []) + kept_terms
    if not all_terms:
        return "~ 1"
    return "~ " + " + ".join(all_terms)


def _split_formula_terms(formula_body: str) -> list[str]:
    """
    Split formula body by + while respecting parentheses.

    "a + b + (1|c)" -> ["a", "b", "(1|c)"]
    """
    terms = []
    current = []
    depth = 0

    for char in formula_body:
        if char == "(":
            depth += 1
            current.append(char)
        elif char == ")":
            depth -= 1
            current.append(char)
        elif char == "+" and depth == 0:
            term = "".join(current).strip()
            if term:
                terms.append(term)
            current = []
        else:
            current.append(char)

    # Don't forget last term
    term = "".join(current).strip()
    if term:
        terms.append(term)

    return terms


def _build_simple_design_matrix(
    terms: list[str],
    data: pd.DataFrame,
    reference_coded_terms: Collection[str] = (),
) -> tuple[np.ndarray, list[str], list[str | None]]:
    """
    Build design matrix from simple terms.

    Handles:
    - '1' or 'intercept': explicit intercept request (default behavior)
    - '0' or '-1': suppress default intercept
    - Numeric variables: direct inclusion
    - Categorical variables: indicator (dummy) encoding

    Standard behavior (matching R's model.matrix):
    - Intercept is added by default unless '0' or '-1' is present
    - '~ x' is equivalent to '~ 1 + x'
    - '~ 0 + x' or '~ -1 + x' means no intercept
    - If both '0' and '1' appear, '0' wins (intercept suppressed)

    Coding of categorical terms:
    - With an intercept, every categorical term is reference coded: the first
      level is dropped and absorbed into the intercept.
    - Without an intercept, the first categorical term in the formula gets one
      column per level, so each level has its own mean. Every later
      categorical term is reference coded.
    - Levels of a pandas Categorical that have no samples are removed before
      coding, so the reference is always the first level that is observed.

    Parameters
    ----------
    terms : List[str]
        Fixed effect terms, e.g. ['0', 'disease', 'age']
    data : pd.DataFrame
        Metadata DataFrame
    reference_coded_terms : Collection[str]
        Categorical terms that are always reference coded, even when they are
        the first categorical term of a no-intercept formula. Used for random
        effect grouping variables that are temporarily listed as fixed terms
        during collinearity checking.

    Returns
    -------
    X : np.ndarray
        Design matrix (n_samples, n_columns)
    col_names : List[str]
        Column names, e.g. ['(Intercept)', 'age', 'diagnosis_dcm', 'diagnosis_hcm']
    col_to_term : List[Optional[str]]
        Parent formula term for each column. None for intercept.
        e.g. [None, 'age', 'diagnosis', 'diagnosis']

    Raises
    ------
    ValueError
        If intercept is suppressed but no other valid terms exist
    """
    n = len(data)
    columns = []
    col_names = []
    col_to_term = []

    # Check if intercept should be suppressed ('0' or '-1' in terms)
    # '0' always wins over '1' if both present
    suppress_intercept = "0" in terms or "-1" in terms

    # Add intercept by default unless suppressed
    if not suppress_intercept:
        columns.append(np.ones(n))
        col_names.append("(Intercept)")
        col_to_term.append(None)

    # Without an intercept, exactly one categorical term carries a column for
    # every level. This flag is cleared once that term has been coded.
    full_coding_pending = suppress_intercept

    for term in terms:
        term = term.strip()

        # Skip intercept control terms (already handled above)
        if term in ("0", "-1", "1") or term.lower() == "intercept":
            continue

        if term not in data.columns:
            # Skip unknown terms
            continue

        col = data[term]

        if pd.api.types.is_numeric_dtype(col):
            columns.append(col.values.astype(float))
            col_names.append(term)
            col_to_term.append(term)
        else:
            # Categorical - indicator encode

            # Remove levels with no samples, so that the reference level is
            # the first level actually observed
            if isinstance(col.dtype, pd.CategoricalDtype):
                col = col.cat.remove_unused_categories()

            use_full_coding = full_coding_pending and term not in reference_coded_terms
            if use_full_coding:
                full_coding_pending = False

            dummies = pd.get_dummies(col, prefix=term, drop_first=not use_full_coding)
            for dummy_col in dummies.columns:
                columns.append(dummies[dummy_col].values.astype(float))
                col_names.append(dummy_col)
                col_to_term.append(term)

    if not columns:
        # No columns created - this happens when:
        # 1. suppress_intercept=True AND no valid variable terms, OR
        # 2. All terms were unknown/skipped
        if suppress_intercept:
            raise ValueError(
                "Cannot create design matrix: intercept suppressed ('0' or '-1' in formula) "
                "but no other valid terms exist. Use '~ 1' for intercept-only model."
            )
        # Fallback: return intercept-only (shouldn't reach here due to default intercept)
        return np.ones((n, 1)), ["(Intercept)"], [None]

    X = np.column_stack(columns)
    return X, col_names, col_to_term


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "build_model_matrices",
    "clean_design_matrix",
    "drop_redundant_columns_qr",
    "drop_redundant_terms",
    "remove_constant_terms",
]