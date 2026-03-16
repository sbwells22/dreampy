"""
Preprocessing functions for dreampy.

This module provides the Tier 1 preprocessing pipeline:
  - compute_tmm_factors(): TMM/RLE/quantile normalization
  - filter_by_expr(): Expression-based gene filtering
  - log2cpm(): Count transformation to log2 CPM
  - estimate_weights(): Precision weight estimation via mean-variance trend

These functions can be used standalone or composed into custom pipelines.
"""
from __future__ import annotations

import warnings
from typing import Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd
    from anndata import AnnData
    from .formula import Formula

import numpy as np
from scipy import sparse
from scipy.interpolate import interp1d

# =============================================================================
# Lowess Smoothing (ported from R)
# =============================================================================


def loess_auto(
    x: np.ndarray,
    y: np.ndarray,
    degree: int = 1,
    criterion: str = "aicc",
    span_range: tuple[float, float] = (0.05, 0.95),
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Loess with automatic smoothing parameter selection.

    This is a port of R's fANCOVA::loess.as() function, which is used by
    dreamlet's voomWithDreamWeights when span='auto'.

    Uses scikit-misc's loess which wraps the original Fortran code from
    Cleveland et al., matching R's loess() exactly.

    Parameters
    ----------
    x : np.ndarray
        Predictor values
    y : np.ndarray
        Response values
    degree : int
        Degree of local polynomial (1 = linear, 2 = quadratic). Default: 1
    criterion : str
        Criterion for selecting optimal span. 'aicc' (default) or 'gcv'
    span_range : tuple
        Range of span values to search. Default: (0.05, 0.95)

    Returns
    -------
    x_out : np.ndarray
        Sorted x values
    y_out : np.ndarray
        Fitted y values
    span : float
        Optimal span selected

    Notes
    -----
    Uses AICc (corrected Akaike Information Criterion) or GCV (Generalized
    Cross-Validation) to select the optimal smoothing parameter.

    AICc = log(sigma2) + 1 + 2*(2*(traceL+1)) / (n - traceL - 2)
    GCV = n*sigma2 / (n - traceL)^2

    where traceL is the trace of the hat matrix (effective df).
    """
    from scipy.optimize import minimize_scalar
    from skmisc.loess import loess

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(x)

    # Sort by x
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]

    def compute_loess_criterion(span):
        """
        Compute AICc or GCV criterion for a given span.

        Matches R's fANCOVA::loess.as opt.span function exactly.
        """
        if span <= span_range[0] or span > span_range[1]:
            return np.inf

        try:
            # Fit loess using skmisc (matches R's loess exactly)
            loess_fit = loess(x_sorted, y_sorted, degree=degree, span=span)
            loess_fit.fit()

            # Get trace of hat matrix (equivalent to R's trace.hat)
            traceL = loess_fit.outputs.enp  # equivalent number of parameters

            # Compute residuals
            residuals = y_sorted - loess_fit.outputs.fitted_values

            # Compute sigma^2 (R uses n-1 denominator)
            sigma2 = np.sum(residuals**2) / (n - 1)

            if criterion == "aicc":
                # AICc criterion - matches R exactly
                denom = n - traceL - 2
                if denom <= 0:
                    return np.inf
                crit = np.log(sigma2) + 1 + 2 * (2 * (traceL + 1)) / denom
            else:  # gcv
                # GCV criterion
                denom = n - traceL
                if denom <= 0:
                    return np.inf
                crit = n * sigma2 / (denom**2)

            return crit
        except ValueError:
            return np.inf

    # Optimize span using bounded optimization (matches R's optimize())
    result = minimize_scalar(
        compute_loess_criterion, bounds=span_range, method="bounded", options={"xatol": 1e-5}
    )

    optimal_span = result.x

    # Fit with optimal span
    loess_fit = loess(x_sorted, y_sorted, degree=degree, span=optimal_span)
    loess_fit.fit()
    y_fitted = loess_fit.outputs.fitted_values

    return x_sorted, y_fitted, optimal_span


def _lowest(
    x: np.ndarray, y: np.ndarray, xs: float, nleft: int, nright: int, userw: bool, rw: np.ndarray
) -> tuple[float, bool]:
    """
    Compute lowest fitted value at xs using points nleft:nright.

    This is a direct port of R's lowest() function from lowess.c

    Parameters
    ----------
    x, y : np.ndarray
        Input data (must be sorted by x)
    xs : float
        Point at which to evaluate
    nleft, nright : int
        Indices defining the window [nleft, nright]
    userw : bool
        Whether to use robustness weights
    rw : np.ndarray
        Robustness weights

    Returns
    -------
    ys : float
        Fitted value at xs
    ok : bool
        Whether fit was successful
    """
    n = len(x)
    x_range = x[-1] - x[0]
    h = max(xs - x[nleft], x[nright] - xs)
    h9 = 0.999 * h
    h1 = 0.001 * h

    # Compute weights
    a = 0.0
    w = np.zeros(n)

    j = nleft
    while j < n:
        w[j] = 0.0
        r = abs(x[j] - xs)
        if r <= h9:
            if r <= h1:
                w[j] = 1.0
            else:
                # Tricube weight: (1 - (r/h)^3)^3
                w[j] = (1.0 - (r / h) ** 3) ** 3
            if userw:
                w[j] *= rw[j]
            a += w[j]
        elif x[j] > xs:
            break
        j += 1

    # Rightmost point (may be greater than nright due to ties)
    nrt = j - 1

    if a <= 0:
        return 0.0, False

    # Normalize weights to sum to 1
    for j in range(nleft, nrt + 1):
        w[j] /= a

    if h > 0:
        # Weighted center of x values
        a = 0.0
        for j in range(nleft, nrt + 1):
            a += w[j] * x[j]

        b = xs - a
        c = 0.0
        for j in range(nleft, nrt + 1):
            c += w[j] * (x[j] - a) ** 2

        if np.sqrt(c) > 0.001 * x_range:
            # Points are spread out enough to compute slope
            b /= c
            for j in range(nleft, nrt + 1):
                w[j] *= b * (x[j] - a) + 1.0

    # Compute fitted value
    ys = 0.0
    for j in range(nleft, nrt + 1):
        ys += w[j] * y[j]

    return ys, True


def lowess_r(
    x: np.ndarray, y: np.ndarray, f: float = 2 / 3, nsteps: int = 3, delta: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Direct port of R's stats::lowess() function.

    This implementation exactly replicates R's lowess algorithm from
    src/library/stats/src/lowess.c to ensure numerical equivalence.

    Parameters
    ----------
    x : np.ndarray
        X values (predictor)
    y : np.ndarray
        Y values (response)
    f : float
        Smoother span - fraction of points to use in each local fit.
        Default: 2/3 (R's default). Use 0.5 for voom.
    nsteps : int
        Number of robustifying iterations. Default: 3 (R's default)
    delta : float, optional
        Skip distance for efficiency. Points within delta of each other
        are interpolated rather than fitted.
        Default: 0.01 * range(x) (R's default)

    Returns
    -------
    x_out : np.ndarray
        Sorted x values
    y_out : np.ndarray
        Smoothed y values

    Notes
    -----
    R code: l <- stats::lowess(sx, sy, f = span)

    Key differences from statsmodels.lowess:
    - Uses R's exact tricube weighting
    - Uses R's delta-based interpolation for efficiency
    - Uses R's biweight for robustness iterations
    - Handles ties the same way as R

    References
    ----------
    Cleveland, W.S. (1979) "Robust Locally Weighted Regression and
    Smoothing Scatterplots". Journal of the American Statistical
    Association 74:829-836.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    # Sort by x (R does this in lowess.R)
    o = np.argsort(x)
    x = x[o].copy()
    y = y[o].copy()

    n = len(x)
    if n < 2:
        return x, y.copy()

    # Default delta: 0.01 * range(x) - same as R
    if delta is None:
        delta = 0.01 * (x[-1] - x[0])

    ys = np.zeros(n)
    rw = np.ones(n)  # robustness weights
    res = np.zeros(n)  # residuals

    # ns = number of points in each local fit
    # at least 2, at most n
    ns = max(2, min(n, int(f * n + 1e-7)))

    # Robustness iterations
    for iteration in range(nsteps + 1):
        nleft = 0
        nright = ns - 1
        last = -1  # index of previous estimated point
        i = 0  # index of current point

        while True:
            # Move nleft, nright to right if radius decreases
            while nright < n - 1:
                d1 = x[i] - x[nleft]
                d2 = x[nright + 1] - x[i]
                if d1 > d2:
                    nleft += 1
                    nright += 1
                else:
                    break

            # Fitted value at x[i]
            ys[i], ok = _lowest(x, y, x[i], nleft, nright, iteration > 0, rw)
            if not ok:
                ys[i] = y[i]  # all weights zero - copy over value

            # Interpolate skipped points
            if last >= 0 and last < i - 1:
                denom = x[i] - x[last]
                for j in range(last + 1, i):
                    alpha = (x[j] - x[last]) / denom
                    ys[j] = alpha * ys[i] + (1.0 - alpha) * ys[last]

            # Last point actually estimated
            last = i

            # Skip points within delta (x coord of close points)
            cut = x[last] + delta
            i = last + 1
            while i < n:
                if x[i] > cut:
                    break
                if x[i] == x[last]:
                    ys[i] = ys[last]
                    last = i
                i += 1

            i = max(last + 1, i - 1)
            if last >= n - 1:
                break

        # Compute residuals
        res = y - ys

        # Last iteration - don't compute weights
        if iteration >= nsteps:
            break

        # Overall scale estimate
        sc = np.mean(np.abs(res))

        # Compute robustness weights (biweight)
        abs_res = np.abs(res).copy()

        # Compute cmad = 6 * median(abs_res)
        m1 = n // 2
        if n % 2 == 0:
            # Even n: average of two middle values
            abs_res_sorted = np.partition(abs_res, [m1 - 1, m1])
            cmad = 3.0 * (abs_res_sorted[m1 - 1] + abs_res_sorted[m1])
        else:
            # Odd n: middle value
            cmad = 6.0 * np.partition(abs_res, m1)[m1]

        if cmad < 1e-7 * sc:
            # Effectively zero - stop iterations
            break

        c9 = 0.999 * cmad
        c1 = 0.001 * cmad

        for i in range(n):
            r = np.abs(res[i])
            if r <= c1:
                rw[i] = 1.0
            elif r <= c9:
                rw[i] = (1.0 - (r / cmad) ** 2) ** 2  # biweight
            else:
                rw[i] = 0.0

    return x, ys


def lowess_smooth(
    x: np.ndarray, y: np.ndarray, span: str | float = 0.5
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit a lowess (locally weighted regression) curve.

    This is a wrapper that matches R's behavior:
    - When span is numeric: uses lowess_r() which ports R's stats::lowess()
    - When span is 'auto': uses loess_auto() which ports fANCOVA::loess.as()

    Parameters
    ----------
    x : np.ndarray
        X values (predictor)
    y : np.ndarray
        Y values (response)
    span : float or 'auto'
        Fraction of data to use for local regression (R's f parameter).
        If 'auto', uses AICc-based automatic selection (fANCOVA::loess.as).
        Default: 0.5

    Returns
    -------
    x_sorted : np.ndarray
        Sorted x values
    y_fitted : np.ndarray
        Fitted (smoothed) y values

    Notes
    -----
    When span='auto', this replicates dreamlet's behavior which uses
    fANCOVA::loess.as() for automatic smoothing parameter selection.

    When span='auto' and skmisc is not available, falls back to span=0.5
    with a warning.
    """
    if isinstance(span, str) and span == "auto":
        try:
            # Use fANCOVA::loess.as equivalent
            x_out, y_out, _optimal_span = loess_auto(x, y, degree=1, criterion="aicc")
            return x_out, y_out
        except ImportError:
            warnings.warn(
                "skmisc not available for automatic span selection. "
                "Falling back to span=0.5. Install with: pip install scikit-misc",
                stacklevel=2
            )
            return lowess_r(x, y, f=0.5, nsteps=3, delta=None)
    else:
        # Use R's lowess
        return lowess_r(x, y, f=float(span), nsteps=3, delta=None)


# =============================================================================
# Normalization (from normalization.py)
# =============================================================================


def _quantile_r(x: np.ndarray, p: float) -> float:
    """
    Compute quantile using R's default method (type 7).

    R's quantile type 7 uses linear interpolation between points.
    This matches R's quantile(x, p) exactly.
    """
    x_sorted = np.sort(x[~np.isnan(x)])
    n = len(x_sorted)

    if n == 0:
        return np.nan
    if n == 1:
        return x_sorted[0]

    h = (n - 1) * p
    lo = int(np.floor(h))
    hi = int(np.ceil(h))

    if lo == hi:
        return x_sorted[lo]

    return x_sorted[lo] + (h - lo) * (x_sorted[hi] - x_sorted[lo])


def compute_tmm_factors(
    data: Union[np.ndarray, sparse.csr_matrix, AnnData],
    lib_size: np.ndarray | None = None,
    method: str = "TMM",
    ref_column: int | None = None,
    log_ratio_trim: float = 0.3,
    sum_trim: float = 0.05,
    do_weighting: bool = True,
    acutoff: float = -1e10,
    p: float = 0.75,
    *,
    assay_col: str | None = "assays",
    inplace: bool = False,
    quiet: bool = False,
) -> Union[np.ndarray, AnnData, None]:
    """
    Compute normalization factors for RNA-seq count data.

    Ported from edgeR::calcNormFactors.default()

    Parameters
    ----------
    data : np.ndarray, sparse.csr_matrix, or AnnData
        Count data. If array: (genes × samples). If AnnData: pseudobulk object
        from aggregate_pseudobulk() with counts in .X (samples × genes).
    lib_size : np.ndarray, optional
        Library sizes (sum of counts per sample). If None, computed from data.
        Ignored when data is AnnData (computed automatically).
    method : {'TMM', 'RLE', 'upperquartile', 'none'}
        Normalization method to use. Default: 'TMM'
    ref_column : int, optional
        Which column to use as reference for TMM. If None, auto-selected.
        For AnnData with assay_col, reference is selected within each assay.
    log_ratio_trim : float
        Proportion of extreme log-ratios to trim (for TMM). Default: 0.3
    sum_trim : float
        Proportion of extreme sums to trim (for TMM). Default: 0.05
    do_weighting : bool
        Whether to use weighted average in TMM. Default: True
    acutoff : float
        Cutoff on absolute expression for TMM. Default: -1e10
    p : float
        Percentile for upperquartile method. Default: 0.75
    assay_col : str or None, default='assays'
        Column in .obs identifying assays. When provided, TMM is computed
        per-assay (recommended for multi-assay pseudobulk). Set to None
        for global normalization. Only used when data is AnnData.
    inplace : bool, default=False
        If True and data is AnnData, modify in place. If False, return copy.
        Ignored when data is array.
    quiet : bool, default=False
        If True, suppress progress messages. Only applies to AnnData input.

    Returns
    -------
    result : np.ndarray, AnnData, or None
        - If data is array: normalization factors (shape: n_samples)
        - If data is AnnData and inplace=False: AnnData with norm_factors in .obs
        - If data is AnnData and inplace=True: None (modifies in place)

    Notes
    -----
    Factors are scaled so they multiply to 1.0 across samples (per-assay when
    assay_col is provided).

    When data is AnnData, results are stored in:
    - pb.obs['lib_size']: Library sizes (sum of counts per sample)
    - pb.obs['norm_factors']: Normalization factors

    For multi-assay pseudobulk data, TMM should be computed per-assay to avoid
    cross-cell-type comparisons affecting normalization.

    Examples
    --------
    Array input (backward compatible):

    >>> counts = np.random.poisson(100, (1000, 10))
    >>> norm_factors = compute_tmm_factors(counts)
    >>> lib_size = counts.sum(axis=0)
    >>> effective_lib_size = lib_size * norm_factors

    AnnData input (recommended):

    >>> pb = dp.aggregate_pseudobulk(adata, groupby=["celltype", "donor"])
    >>> pb = dp.filter_samples(pb, min_cells=10)
    >>> pb = dp.compute_tmm_factors(pb, assay_col='assays')
    >>> # Now pb.obs has 'lib_size' and 'norm_factors' columns
    """
    # Check if input is AnnData
    is_anndata = hasattr(data, "X") and hasattr(data, "obs")

    if is_anndata:
        return _compute_tmm_factors_anndata(
            data,
            method=method,
            ref_column=ref_column,
            log_ratio_trim=log_ratio_trim,
            sum_trim=sum_trim,
            do_weighting=do_weighting,
            acutoff=acutoff,
            p=p,
            assay_col=assay_col,
            inplace=inplace,
            quiet=quiet,
        )
    else:
        # Array path (original implementation)
        return _compute_tmm_factors_array(
            data, lib_size, method, ref_column, log_ratio_trim, sum_trim, do_weighting, acutoff, p
        )


def _compute_tmm_factors_anndata(
    pb: AnnData,
    method: str = "TMM",
    ref_column: int | None = None,
    log_ratio_trim: float = 0.3,
    sum_trim: float = 0.05,
    do_weighting: bool = True,
    acutoff: float = -1e10,
    p: float = 0.75,
    assay_col: str | None = "assays",
    inplace: bool = False,
    quiet: bool = False,
) -> Optional[AnnData]:
    """Compute TMM factors for AnnData, optionally per-assay."""

    if not inplace:
        pb = pb.copy()

    # Get counts matrix (samples × genes in AnnData → genes × samples for TMM)
    counts = pb.X.T
    if hasattr(counts, "toarray"):
        counts = counts.toarray()
    counts = np.asarray(counts, dtype=np.float64)

    _n_genes, n_samples = counts.shape

    # Compute lib_size
    lib_size = counts.sum(axis=0)
    pb.obs["lib_size"] = lib_size

    # Initialize norm_factors
    norm_factors = np.ones(n_samples)

    if assay_col is None or assay_col not in pb.obs.columns:
        # Global normalization
        norm_factors = _compute_tmm_factors_array(
            counts, lib_size, method, ref_column, log_ratio_trim, sum_trim, do_weighting, acutoff, p
        )
        if not quiet:
            print(f"compute_tmm_factors: {method} normalization complete ({n_samples} samples)")
    else:
        # Per-assay normalization
        assays = pb.obs[assay_col].values
        unique_assays = np.unique(assays)

        for assay in unique_assays:
            mask = assays == assay
            assay_counts = counts[:, mask]
            assay_lib_size = lib_size[mask]

            assay_factors = _compute_tmm_factors_array(
                assay_counts,
                assay_lib_size,
                method,
                None,  # ref_column=None for per-assay
                log_ratio_trim,
                sum_trim,
                do_weighting,
                acutoff,
                p,
            )

            norm_factors[mask] = assay_factors

        if not quiet:
            print(
                f"compute_tmm_factors: {method} normalization complete ({len(unique_assays)} assays, {n_samples} samples)"
            )

    pb.obs["norm_factors"] = norm_factors

    if inplace:
        return None
    else:
        return pb


def _compute_tmm_factors_array(
    counts: np.ndarray,
    lib_size: np.ndarray | None = None,
    method: str = "TMM",
    ref_column: int | None = None,
    log_ratio_trim: float = 0.3,
    sum_trim: float = 0.05,
    do_weighting: bool = True,
    acutoff: float = -1e10,
    p: float = 0.75,
) -> np.ndarray:
    """Original array-based TMM computation."""

    if sparse.issparse(counts):
        raise TypeError("counts must be a dense numpy array. Call .toarray() first.")

    _n_genes, n_samples = counts.shape

    # Compute lib_size if not provided
    if lib_size is None:
        lib_size = counts.sum(axis=0)
    else:
        lib_size = np.asarray(lib_size, dtype=np.float64)

    if len(lib_size) != n_samples:
        raise ValueError(
            f"Length of lib_size ({len(lib_size)}) must match number of samples ({n_samples})"
        )

    # Remove all zero rows
    all_zero = (counts > 0).sum(axis=1) == 0
    if all_zero.any():
        counts = counts[~all_zero, :]

    # Handle degenerate cases
    if counts.shape[0] == 0 or n_samples == 1:
        method = "none"

    # Calculate factors based on method
    if method == "TMM":
        factors = _calc_factor_tmm(
            counts, lib_size, ref_column, log_ratio_trim, sum_trim, do_weighting, acutoff
        )
    elif method == "RLE":
        factors = _calc_factor_rle(counts, lib_size)
    elif method == "upperquartile":
        factors = _calc_factor_quantile(counts, lib_size, p)
    elif method == "none":
        factors = np.ones(n_samples)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Scale so factors multiply to 1
    factors = factors / np.exp(np.mean(np.log(factors)))

    return factors


def _calc_factor_tmm(
    counts: np.ndarray,
    lib_size: np.ndarray,
    ref_column: int | None = None,
    log_ratio_trim: float = 0.3,
    sum_trim: float = 0.05,
    do_weighting: bool = True,
    acutoff: float = -1e10,
) -> np.ndarray:
    """Calculate TMM normalization factors. Ported from edgeR."""
    from scipy.stats import rankdata

    _n_genes, n_samples = counts.shape

    # Auto-select reference column
    if ref_column is None:
        f75 = np.zeros(n_samples)
        for j in range(n_samples):
            f75[j] = _quantile_r(counts[:, j], 0.75)
        f75 = f75 / lib_size

        if np.median(f75) < 1e-20:
            ref_column = np.argmax(np.sum(np.sqrt(counts), axis=0))
        else:
            ref_column = np.argmin(np.abs(f75 - np.mean(f75)))

    factors = np.zeros(n_samples)

    for i in range(n_samples):
        obs = counts[:, i].astype(np.float64)
        ref = counts[:, ref_column].astype(np.float64)

        nO = lib_size[i]
        nR = lib_size[ref_column]

        with np.errstate(divide="ignore", invalid="ignore"):
            logR = np.log2((obs / nO) / (ref / nR))
            absE = (np.log2(obs / nO) + np.log2(ref / nR)) / 2
            v = (nO - obs) / nO / obs + (nR - ref) / nR / ref

        fin = np.isfinite(logR) & np.isfinite(absE) & (absE > acutoff)

        logR = logR[fin]
        absE = absE[fin]
        v = v[fin]

        if len(logR) == 0:
            factors[i] = 1.0
            continue

        if np.max(np.abs(logR)) < 1e-6:
            factors[i] = 1.0
            continue

        n = len(logR)
        loL = int(np.floor(n * log_ratio_trim)) + 1
        hiL = n + 1 - loL
        loS = int(np.floor(n * sum_trim)) + 1
        hiS = n + 1 - loS

        rank_logR = rankdata(logR, method="average")
        rank_absE = rankdata(absE, method="average")

        keep = (rank_logR >= loL) & (rank_logR <= hiL) & (rank_absE >= loS) & (rank_absE <= hiS)

        logR_keep = logR[keep]
        v_keep = v[keep]

        if do_weighting:
            with np.errstate(divide="ignore", invalid="ignore"):
                f = np.nansum(logR_keep / v_keep) / np.nansum(1.0 / v_keep)
        else:
            f = np.nanmean(logR_keep)

        if np.isnan(f):
            f = 0.0

        factors[i] = 2.0**f

    return factors


def _calc_factor_rle(counts: np.ndarray, lib_size: np.ndarray) -> np.ndarray:
    """Calculate RLE normalization factors. Ported from edgeR."""
    _n_genes, n_samples = counts.shape

    with np.errstate(divide="ignore"):
        log_counts = np.log(counts)
    log_counts[~np.isfinite(log_counts)] = np.nan
    gm = np.exp(np.nanmean(log_counts, axis=1))

    factors = np.zeros(n_samples)
    for j in range(n_samples):
        ratios = counts[:, j] / gm
        keep = gm > 0
        if np.sum(keep) > 0:
            factors[j] = np.median(ratios[keep])
        else:
            factors[j] = 1.0

    factors = factors / lib_size
    return factors


def _calc_factor_quantile(counts: np.ndarray, lib_size: np.ndarray, p: float = 0.75) -> np.ndarray:
    """Calculate quantile normalization factors. Ported from edgeR."""
    n_samples = counts.shape[1]
    factors = np.zeros(n_samples)

    for j in range(n_samples):
        factors[j] = _quantile_r(counts[:, j], p)

    if np.min(factors) == 0:
        warnings.warn("One or more quantiles are zero", stacklevel=2)

    factors = factors / lib_size
    return factors


# =============================================================================
# Gene Filtering (from gene_filtering.py)
# =============================================================================


def filter_by_expr(
    data: Union[np.ndarray, sparse.csr_matrix, AnnData],
    lib_size: np.ndarray | None = None,
    norm_factors: np.ndarray | None = None,
    min_count: int = 5,
    min_total_count: int = 15,
    large_n: int = 10,
    min_prop: float = 0.4,
    *,
    assay_col: str | None = "assays",
    inplace: bool = False,
    quiet: bool = False,
) -> Union[np.ndarray, AnnData, dict[str, AnnData], None]:
    """
    Filter genes by expression levels.

    Ported from edgeR::filterByExpr.default()

    A gene is retained if it has at least min_count CPM in at least
    min_prop fraction of the samples.

    Parameters
    ----------
    data : np.ndarray, sparse.csr_matrix, or AnnData
        Count data. If array: (genes × samples). If AnnData: pseudobulk object
        with counts in .X (samples × genes).
    lib_size : np.ndarray, optional
        Library sizes. If None, computed from data or taken from pb.obs['lib_size'].
    norm_factors : np.ndarray, optional
        Normalization factors. If None, taken from pb.obs['norm_factors'] or defaults to 1.0.
    min_count : int
        Minimum count to define "expressed" gene. Default: 5
    min_total_count : int
        Minimum total count across all samples. Default: 15
    large_n : int
        Threshold for "large" group size. Default: 10
    min_prop : float
        Minimum proportion of samples expressing gene. Default: 0.4
    assay_col : str or None, default='assays'
        Column in .obs identifying assays. When provided, filtering is done
        per-assay, returning a dict of AnnDatas (each may have different genes).
        Set to None for global filtering. Only used when data is AnnData.
    inplace : bool, default=False
        If True and data is AnnData with assay_col=None, modify in place.
        Ignored when assay_col is provided (always returns new objects).
    quiet : bool, default=False
        If True, suppress progress messages showing genes retained/lost.

    Returns
    -------
    result : np.ndarray, AnnData, Dict[str, AnnData], or None
        - If data is array: boolean mask (shape: n_genes)
        - If data is AnnData and assay_col=None: filtered AnnData (or None if inplace)
        - If data is AnnData and assay_col provided: Dict mapping assay name → filtered AnnData

    Notes
    -----
    The minimum sample size is calculated as:
    - If n_samples <= large_n: min_sample_size = n_samples
    - If n_samples > large_n: min_sample_size = large_n + (n_samples - large_n) * min_prop

    For multi-assay data, each assay is filtered independently. This means different
    assays may retain different genes. This is the expected behavior for dreamlet-style
    analysis where each cell type is analyzed separately.

    Examples
    --------
    Array input (backward compatible):

    >>> counts = np.random.poisson(10, (1000, 20))
    >>> keep = filter_by_expr(counts, min_count=5)
    >>> counts_filtered = counts[keep, :]

    AnnData input (single assay):

    >>> pb = dp.compute_tmm_factors(pb, assay_col=None)
    >>> pb = dp.filter_by_expr(pb, assay_col=None)

    AnnData input (multi-assay, returns dict):

    >>> pb = dp.compute_tmm_factors(pb, assay_col='assays')
    >>> assay_data = dp.filter_by_expr(pb, assay_col='assays')
    >>> # assay_data is Dict[str, AnnData], one per assay with different genes
    >>> tcells = assay_data['T_cells']
    """
    # Check if input is AnnData
    is_anndata = hasattr(data, "X") and hasattr(data, "obs")

    if is_anndata:
        return _filter_by_expr_anndata(
            data,
            lib_size=lib_size,
            norm_factors=norm_factors,
            min_count=min_count,
            min_total_count=min_total_count,
            large_n=large_n,
            min_prop=min_prop,
            assay_col=assay_col,
            inplace=inplace,
            quiet=quiet,
        )
    else:
        # Array path (original implementation)
        return _filter_by_expr_array(
            data, lib_size, norm_factors, min_count, min_total_count, large_n, min_prop
        )


def _filter_by_expr_anndata(
    pb: AnnData,
    lib_size: np.ndarray | None = None,
    norm_factors: np.ndarray | None = None,
    min_count: int = 5,
    min_total_count: int = 15,
    large_n: int = 10,
    min_prop: float = 0.4,
    assay_col: str | None = "assays",
    inplace: bool = False,
    quiet: bool = False,
) -> Union[AnnData, dict[str, AnnData], None]:
    """Filter genes in AnnData, optionally per-assay."""

    # Get lib_size and norm_factors from .obs if not provided
    if lib_size is None and "lib_size" in pb.obs.columns:
        lib_size = pb.obs["lib_size"].values
    if norm_factors is None and "norm_factors" in pb.obs.columns:
        norm_factors = pb.obs["norm_factors"].values

    n_genes_input = pb.n_vars

    if assay_col is None or assay_col not in pb.obs.columns:
        # Global filtering - return single AnnData
        # Get counts (samples × genes in AnnData → genes × samples for filter)
        counts = pb.X.T
        if hasattr(counts, "toarray"):
            counts = counts.toarray()
        counts = np.asarray(counts, dtype=np.float64)

        keep = _filter_by_expr_array(
            counts, lib_size, norm_factors, min_count, min_total_count, large_n, min_prop
        )

        n_kept = np.sum(keep)
        n_lost = n_genes_input - n_kept

        if n_kept == 0:
            warnings.warn(
                "filter_by_expr: No genes passed filtering criteria. "
                "Consider relaxing min_count or min_prop parameters.",
                UserWarning, stacklevel=2
            )

        if not quiet:
            print(
                f"filter_by_expr: {n_kept}/{n_genes_input} genes retained ({n_lost} filtered out)"
            )

        if inplace:
            pb._inplace_subset_var(keep)
            return None
        else:
            return pb[:, keep].copy()
    else:
        # Per-assay filtering - return dict of AnnDatas
        assays = pb.obs[assay_col].values
        unique_assays = np.unique(assays)

        result = {}
        if not quiet:
            print(f"filter_by_expr: filtering {len(unique_assays)} assays")

        for assay in unique_assays:
            mask = assays == assay
            assay_pb = pb[mask, :].copy()

            # Get assay-specific lib_size and norm_factors
            assay_lib_size = lib_size[mask] if lib_size is not None else None
            assay_norm_factors = norm_factors[mask] if norm_factors is not None else None

            # Get counts for this assay
            counts = assay_pb.X.T
            if hasattr(counts, "toarray"):
                counts = counts.toarray()
            counts = np.asarray(counts, dtype=np.float64)

            keep = _filter_by_expr_array(
                counts,
                assay_lib_size,
                assay_norm_factors,
                min_count,
                min_total_count,
                large_n,
                min_prop,
            )

            n_kept = np.sum(keep)
            n_lost = n_genes_input - n_kept

            if n_kept == 0:
                warnings.warn(
                    f"filter_by_expr: No genes passed filtering for assay '{assay}'. "
                    "Consider relaxing min_count or min_prop parameters.",
                    UserWarning, stacklevel=2
                )

            if not quiet:
                print(f"  {assay}: {n_kept}/{n_genes_input} genes retained ({n_lost} filtered out)")

            # Subset to kept genes
            result[assay] = assay_pb[:, keep].copy()

        return result


def _filter_by_expr_array(
    counts: np.ndarray,
    lib_size: np.ndarray | None = None,
    norm_factors: np.ndarray | None = None,
    min_count: int = 5,
    min_total_count: int = 15,
    large_n: int = 10,
    min_prop: float = 0.4,
) -> np.ndarray:
    """Original array-based filter_by_expr implementation."""
    # Convert to dense if sparse
    if sparse.issparse(counts):
        raise TypeError("counts must be a dense numpy array. Call .toarray() first.")

    _n_genes, n_samples = counts.shape

    # Compute lib_size if not provided
    if lib_size is None:
        lib_size = counts.sum(axis=0)
    else:
        lib_size = np.asarray(lib_size, dtype=np.float64)

    # Default norm_factors to 1
    if norm_factors is None:
        norm_factors = np.ones(n_samples)
    else:
        norm_factors = np.asarray(norm_factors, dtype=np.float64)

    # Compute effective library sizes
    effective_lib_size = lib_size * norm_factors

    # Determine minimum number of samples that should express gene
    if n_samples <= large_n:
        min_sample_size = n_samples
    else:
        min_sample_size = large_n + (n_samples - large_n) * min_prop

    # Compute CPM cutoff
    median_lib_size = np.median(effective_lib_size)
    cpm_cutoff = (min_count / median_lib_size) * 1e6

    # Compute CPM
    cpm_values = (counts / effective_lib_size[np.newaxis, :]) * 1e6

    # Filter by CPM and total count
    keep_cpm = np.sum(cpm_values >= cpm_cutoff, axis=1) >= (min_sample_size - 1e-14)
    keep_total = np.sum(counts, axis=1) >= (min_total_count - 1e-14)

    keep = keep_cpm & keep_total

    return keep


# =============================================================================
# Sample Filtering
# =============================================================================


def filter_samples(
    pb: AnnData,
    min_cells: int = 5,
    min_samples: int = 4,
    assay_col: str | None = "assays",
    n_cells_col: str = "n_cells",
    inplace: bool = False,
    quiet: bool = False,
) -> Optional[AnnData]:
    """
    Filter pseudobulk samples by minimum cell count.

    Removes samples with fewer than `min_cells` cells, then verifies
    that at least `min_samples` remain per assay (if `assay_col` provided)
    or globally. Assays that don't meet the threshold are entirely removed.

    Parameters
    ----------
    pb : AnnData
        Pseudobulk AnnData from `aggregate_pseudobulk()`.
    min_cells : int, default=5
        Minimum cells per sample to retain.
    min_samples : int, default=4
        Minimum samples required after filtering.
        - If `assay_col` is provided: applies per-assay
        - If `assay_col` is None: applies globally
    assay_col : str or None, default='assays'
        Column in pb.obs identifying assays. If provided, min_samples
        is checked per-assay and assays failing are removed entirely.
        Set to None for single-assay data.
    n_cells_col : str, default='n_cells'
        Column in pb.obs containing cell counts per sample.
    inplace : bool, default=False
        If True, modify pb in place. If False, return filtered copy.
    quiet : bool, default=False
        If False, print filtering summary.

    Returns
    -------
    AnnData or None
        Filtered pseudobulk data. Returns None if inplace=True.

    Raises
    ------
    ValueError
        If no samples/assays remain after filtering.

    Examples
    --------
    >>> import dreampy as dp
    >>> pb = dp.aggregate_pseudobulk(adata, groupby=["celltype", "donor"])
    >>>
    >>> # Filter with per-assay min_samples check (recommended)
    >>> pb_filtered = dp.filter_samples(pb, min_cells=10, min_samples=4)
    >>>
    >>> # Single-assay mode (no per-assay check)
    >>> pb_filtered = dp.filter_samples(pb, assay_col=None)
    >>>
    >>> # In-place modification
    >>> dp.filter_samples(pb, min_cells=10, inplace=True)

    Notes
    -----
    For multi-assay pseudobulk data, always use `assay_col` (default).
    This ensures each cell type has enough samples for reliable DE.
    Without it, assays with too few samples would pass through and
    cause problems in downstream analysis.

    See Also
    --------
    aggregate_pseudobulk : Creates pseudobulk with n_cells column
    filter_by_expr : Gene filtering by expression level
    """

    # Validate inputs
    if n_cells_col not in pb.obs.columns:
        raise ValueError(f"n_cells_col '{n_cells_col}' not found in pb.obs")

    if assay_col is not None and assay_col not in pb.obs.columns:
        raise ValueError(f"assay_col '{assay_col}' not found in pb.obs")

    n_cells = pb.obs[n_cells_col].values
    n_samples_input = pb.n_obs

    # Step 1: Filter by min_cells
    cells_mask = n_cells >= min_cells
    n_dropped_cells = n_samples_input - np.sum(cells_mask)

    if not quiet and n_dropped_cells > 0:
        print(f"filter_samples: {n_dropped_cells} samples dropped (n_cells < {min_cells})")

    # Step 2: Apply min_samples threshold
    if assay_col is None:
        # Global filtering (single-assay mode)
        sample_mask = cells_mask
        n_samples_retained = np.sum(sample_mask)

        if n_samples_retained < min_samples:
            raise ValueError(
                f"Only {n_samples_retained} samples have >= {min_cells} cells, "
                f"but min_samples={min_samples} required."
            )

        if not quiet:
            print(f"filter_samples: {n_samples_retained}/{n_samples_input} samples retained")

    else:
        # Per-assay filtering (multi-assay mode)
        assays = pb.obs[assay_col].values
        unique_assays = np.unique(assays)

        # Count samples per assay after min_cells filtering
        assays_to_keep = []
        assays_dropped = []

        for assay in unique_assays:
            assay_cells_mask = (assays == assay) & cells_mask
            n_assay_samples = np.sum(assay_cells_mask)

            if n_assay_samples >= min_samples:
                assays_to_keep.append(assay)
                if not quiet:
                    print(f"  {assay}: {n_assay_samples} samples retained")
            else:
                assays_dropped.append((assay, n_assay_samples))
                if not quiet:
                    print(f"  {assay}: DROPPED ({n_assay_samples} samples < {min_samples})")

        if len(assays_to_keep) == 0:
            raise ValueError(
                f"No assays have >= {min_samples} samples after filtering. "
                f"Dropped: {[f'{a} ({n})' for a, n in assays_dropped]}"
            )

        # Build final mask: passes min_cells AND belongs to kept assay
        sample_mask = cells_mask & np.isin(assays, assays_to_keep)

        if not quiet:
            n_retained = np.sum(sample_mask)
            print(
                f"filter_samples: {n_retained}/{n_samples_input} samples, "
                f"{len(assays_to_keep)} assays retained, {len(assays_dropped)} dropped"
            )

    # Apply filter
    if inplace:
        # Modify in place by subsetting
        pb._inplace_subset_obs(sample_mask)
        return None
    else:
        return pb[sample_mask].copy()


# =============================================================================
# Log2 CPM Transformation
# =============================================================================


def log2cpm(
    data: Union[np.ndarray, sparse.csr_matrix, AnnData],
    lib_size: np.ndarray | None = None,
    norm_factors: np.ndarray | None = None,
    prior_count: float = 0.5,
    *,
    inplace: bool = False,
    quiet: bool = False,
) -> Union[np.ndarray, AnnData, None]:
    """
    Transform count data to log2 counts per million (log2 CPM).

    Parameters
    ----------
    data : np.ndarray, sparse.csr_matrix, or AnnData
        Count data. If array: (genes × samples). If AnnData: pseudobulk object
        with counts in .X (samples × genes).
    lib_size : np.ndarray, optional
        Library sizes. If None, computed from data or taken from pb.obs['lib_size'].
    norm_factors : np.ndarray, optional
        Normalization factors. If None, taken from pb.obs['norm_factors'] or defaults to 1.0.
    prior_count : float
        Pseudocount added to avoid log(0). Default: 0.5
        This is scaled by library size following limma/edgeR behavior.
    inplace : bool, default=False
        If True and data is AnnData, replace .X in place. If False, return copy.
        Ignored when data is array.
    quiet : bool, default=False
        If True, suppress progress messages. Only applies to AnnData input.

    Returns
    -------
    result : np.ndarray, AnnData, or None
        - If data is array: log2 CPM matrix (genes × samples)
        - If data is AnnData and inplace=False: AnnData with .X replaced by log2 CPM
        - If data is AnnData and inplace=True: None (modifies in place)

    Notes
    -----
    The transformation follows limma/voom:

    1. Compute effective library size: lib_size * norm_factors
    2. Augment counts: counts + prior_count (constant, NOT library-scaled)
    3. Compute CPM: augmented_counts / (effective_lib_size + 1) * 1e6
    4. Log2 transform

    When data is AnnData, the original counts in .X are replaced with log2 CPM values.
    The original counts are stored in .layers['counts'] for reference.

    Examples
    --------
    Array input (backward compatible):

    >>> counts = np.random.poisson(100, (1000, 10))
    >>> norm_factors = compute_tmm_factors(counts)
    >>> y = log2cpm(counts, norm_factors=norm_factors)

    AnnData input:

    >>> pb = dp.compute_tmm_factors(pb, assay_col='assays')
    >>> pb = dp.filter_by_expr(pb, assay_col=None)  # or per-assay
    >>> pb = dp.log2cpm(pb)
    >>> # pb.X now contains log2 CPM, original counts in pb.layers['counts']
    """
    # Check if input is AnnData
    is_anndata = hasattr(data, "X") and hasattr(data, "obs")

    if is_anndata:
        return _log2cpm_anndata(
            data,
            lib_size=lib_size,
            norm_factors=norm_factors,
            prior_count=prior_count,
            inplace=inplace,
            quiet=quiet,
        )
    else:
        # Array path (original implementation)
        return _log2cpm_array(data, lib_size, norm_factors, prior_count)


def _looks_like_counts(arr: np.ndarray) -> bool:
    """
    Check if array values look like count data (non-negative integers).

    Handles both integer-typed arrays (checks non-negativity only) and
    float-typed arrays (checks non-negativity and integer-valuedness).
    Samples up to 10,000 finite values for efficiency.

    Parameters
    ----------
    arr : np.ndarray
        Data to check

    Returns
    -------
    bool
        True if data appears to be count data
    """
    if arr.size == 0:
        return True
    sample = arr.ravel()[:10000]
    finite = sample[np.isfinite(sample)]
    if len(finite) == 0:
        return True
    if np.any(finite < 0):
        return False
    if np.issubdtype(arr.dtype, np.integer):
        return True
    return np.allclose(finite, np.round(finite))


def _log2cpm_anndata(
    pb: AnnData,
    lib_size: np.ndarray | None = None,
    norm_factors: np.ndarray | None = None,
    prior_count: float = 0.5,
    inplace: bool = False,
    quiet: bool = False,
) -> Optional[AnnData]:
    """Transform AnnData counts to log2 CPM, replacing .X."""

    if not inplace:
        pb = pb.copy()

    # Store original counts in layers
    if "counts" not in pb.layers:
        pb.layers["counts"] = pb.X.copy()

    # Get lib_size and norm_factors from .obs if not provided
    if lib_size is None:
        if "lib_size" in pb.obs.columns:
            lib_size = pb.obs["lib_size"].values
        else:
            # Compute from counts
            if hasattr(pb.X, "toarray"):
                lib_size = np.asarray(pb.X.toarray().sum(axis=1)).flatten()
            else:
                lib_size = np.asarray(pb.X.sum(axis=1)).flatten()

    if norm_factors is None:
        if "norm_factors" in pb.obs.columns:
            norm_factors = pb.obs["norm_factors"].values
        else:
            norm_factors = np.ones(pb.n_obs)

    # Get counts (samples × genes in AnnData → genes × samples for transform)
    counts = pb.X.T
    if hasattr(counts, "toarray"):
        counts = counts.toarray()
    counts = np.asarray(counts, dtype=np.float64)

    # Warn if data does not look like counts
    if not _looks_like_counts(counts):
        warnings.warn(
            "log2cpm: pb.X does not appear to contain count data "
            "(values are not non-negative integers).",
            UserWarning, stacklevel=2
        )

    # Apply transformation
    log2_cpm = _log2cpm_array(counts, lib_size, norm_factors, prior_count)

    # Replace .X with transformed data (transpose back to samples × genes)
    pb.X = log2_cpm.T

    if not quiet:
        print(f"log2cpm: transformation complete ({pb.n_obs} samples, {pb.n_vars} genes)")

    if inplace:
        return None
    else:
        return pb


def _log2cpm_array(
    counts: np.ndarray,
    lib_size: np.ndarray | None = None,
    norm_factors: np.ndarray | None = None,
    prior_count: float = 0.5,
) -> np.ndarray:
    """Original array-based log2cpm implementation."""
    # Convert to dense if sparse
    if sparse.issparse(counts):
        raise TypeError("counts must be a dense numpy array. Call .toarray() first.")

    _n_genes, n_samples = counts.shape

    # Compute lib_size if not provided
    if lib_size is None:
        lib_size = counts.sum(axis=0).flatten()
    else:
        lib_size = np.asarray(lib_size, dtype=np.float64)

    # Default norm_factors to 1
    if norm_factors is None:
        norm_factors = np.ones(n_samples)
    else:
        norm_factors = np.asarray(norm_factors, dtype=np.float64)

    # Compute effective library sizes
    effective_lib_size = lib_size * norm_factors

    # Augment counts with constant prior count to avoid log(0)
    # This matches limma/voom behavior: counts + 0.5 (constant, not library-scaled)
    # See utilities.py augment_prior_count(..., scaled_by_lib=False)
    counts_aug = counts + prior_count

    # Transform to log2 CPM
    # Note: +1 to effective_lib_size matches R's voom behavior
    log2_cpm = np.log2((counts_aug / (effective_lib_size + 1)[np.newaxis, :]) * 1e6)

    return log2_cpm


# =============================================================================
# Weight Estimation
# =============================================================================


def estimate_weights(
    data: Union[np.ndarray, AnnData],
    formula: Union[str, Formula],
    metadata: Optional[pd.DataFrame] = None,
    lib_size: np.ndarray | None = None,
    *,
    norm_factors: np.ndarray | None = None,
    counts: np.ndarray | None = None,
    span: str | float = "auto",
    reml: bool = True,
    n_jobs: int = 1,
    diagnostics_surface: bool = False,
    diagnostics_path: bool = False,
    quiet: bool = False,
) -> Union[tuple[np.ndarray, dict], AnnData]:
    """
    Estimate voom precision weights from mean-variance trend.

    This is the KEY step that makes dreamlet/voom work. It:
    1. Fits an initial linear mixed model to each gene (no weights)
    2. Extracts residual variance (sigma) and fitted values
    3. Fits a lowess trend to the mean-variance relationship
    4. Computes observation-level precision weights

    Can be called in two modes (same operation, different I/O format):

    **Array mode:**
        weights, trend_info = estimate_weights(
            Y, formula, metadata, lib_size,
            norm_factors=norm_factors, counts=counts, span='auto'
        )

    **AnnData mode:**
        pb = estimate_weights(
            pb, formula, span='auto', n_jobs=4,
            diagnostics_surface=True
        )

    Parameters
    ----------
    data : AnnData or np.ndarray
        If AnnData: Pseudobulk with log2 CPM in .X (samples × genes)
        If ndarray: log2 CPM matrix (genes × samples)
    formula : str or Formula
        Model formula, e.g. "~ disease + (1|donor) + (1|batch)"
    metadata : pd.DataFrame, optional
        Array mode only: Sample metadata with formula variables.
        AnnData mode: reads from pb.obs
    lib_size : np.ndarray, optional
        Library sizes (samples,).
        AnnData mode: reads from pb.obs['lib_size'] if not provided
    norm_factors : np.ndarray, optional
        Normalization factors. Default 1.0.
        AnnData mode: reads from pb.obs['norm_factors'] if not provided
    counts : np.ndarray, optional
        Original counts for all-zero gene detection.
        AnnData mode: reads from pb.layers['counts'] if not provided
    span : float or 'auto'
        Lowess smoothing parameter. Default 'auto' uses AICc selection.
    reml : bool
        Use REML (True) or ML (False) for variance estimation. Default True.
        Dreamlet's default is False.
    n_jobs : int
        Number of parallel jobs for gene fitting. Default 1.
    diagnostics_surface : bool
        Compute log-likelihood surface for each gene. Default False.
    diagnostics_path : bool
        Record optimization path for each gene. Default False.
    quiet : bool
        Suppress progress messages. Default False.

    Returns
    -------
    If array mode: Tuple[np.ndarray, Dict]
        - weights: Precision weights (genes × samples)
        - trend_info: Dict with 'xy' and 'line' coordinates
    If AnnData mode: AnnData
        Input pb with added:
        - .layers['weights']: Precision weights (samples × genes)
        - .uns['voom_info']: Dict with Amean, sigma, fitted_vals, trend_xy,
          trend_line, span, converged, gene_diagnostics (if enabled)

    Examples
    --------
    >>> # Array mode
    >>> weights, trend = estimate_weights(y, "~ disease + (1|donor)", metadata, lib_size)

    >>> # AnnData mode
    >>> pb = dp.log2cpm(pb)
    >>> pb = dp.estimate_weights(pb, "~ disease + (1|donor)")
    >>> # Weights now in pb.layers['weights']
    """
    # Check if input is AnnData
    is_anndata = hasattr(data, "X") and hasattr(data, "obs")

    if is_anndata:
        return _estimate_weights_anndata(
            data,
            formula=formula,
            lib_size=lib_size,
            norm_factors=norm_factors,
            span=span,
            reml=reml,
            n_jobs=n_jobs,
            diagnostics_surface=diagnostics_surface,
            diagnostics_path=diagnostics_path,
            quiet=quiet,
        )
    else:
        # Array mode
        if metadata is None:
            raise ValueError(
                "metadata is required for array mode. "
                "For AnnData input, metadata is read from pb.obs automatically."
            )
        return _estimate_weights_array(
            Y=data,
            formula=formula,
            metadata=metadata,
            lib_size=lib_size,
            norm_factors=norm_factors,
            counts=counts,
            span=span,
            reml=reml,
            n_jobs=n_jobs,
            diagnostics_surface=diagnostics_surface,
            diagnostics_path=diagnostics_path,
            quiet=quiet,
        )


def _estimate_weights_array(
    Y: np.ndarray,
    formula: Union[str, Formula],
    metadata: pd.DataFrame,
    lib_size: np.ndarray,
    *,
    norm_factors: np.ndarray | None = None,
    counts: np.ndarray | None = None,
    span: str | float = "auto",
    reml: bool = True,
    n_jobs: int = 1,
    diagnostics_surface: bool = False,
    diagnostics_path: bool = False,
    quiet: bool = False,
) -> tuple[np.ndarray, dict]:
    """
    Array-mode implementation of estimate_weights.

    Parameters
    ----------
    Y : np.ndarray (genes × samples)
        Log2 CPM expression matrix
    formula : str or Formula
        Model formula
    metadata : pd.DataFrame
        Sample metadata (rows = samples)
    lib_size : np.ndarray (samples,)
        Library sizes
    norm_factors : np.ndarray (samples,), optional
        Normalization factors. Default 1.0.
    counts : np.ndarray (genes × samples), optional
        Original counts for all-zero gene detection
    span : float or 'auto'
        Lowess smoothing parameter
    reml : bool
        REML estimation
    n_jobs : int
        Parallel jobs
    diagnostics_surface, diagnostics_path : bool
        Diagnostic options
    quiet : bool
        Suppress messages

    Returns
    -------
    weights : np.ndarray (genes × samples)
        Precision weights
    trend_info : Dict
        Trend fitting information
    """

    # Late import to avoid circular dependency
    # Import design matrix builder
    from .design import build_model_matrices
    from .fitting import fit_genes_parallel

    # Validate inputs
    n_genes, n_samples = Y.shape

    if lib_size is None:
        raise ValueError("lib_size is required for array mode")

    lib_size = np.asarray(lib_size, dtype=np.float64)
    if len(lib_size) != n_samples:
        raise ValueError(f"lib_size length ({len(lib_size)}) must match n_samples ({n_samples})")

    if len(metadata) != n_samples:
        raise ValueError(f"metadata has {len(metadata)} rows but Y has {n_samples} samples")

    # Default norm_factors
    if norm_factors is None:
        norm_factors = np.ones(n_samples)
    else:
        norm_factors = np.asarray(norm_factors, dtype=np.float64)

    effective_lib_size = lib_size * norm_factors

    # Build all model matrices once for this assay
    formula_str = formula if isinstance(formula, str) else formula.original
    X, _coef_names, Z, block_sizes, re_names, formula_clean, cleaning_warnings = (
        build_model_matrices(formula_str, metadata, verbose=not quiet)
    )

    # Emit cleaning warnings if any
    for w in cleaning_warnings:
        if "dropped" in w.lower() or "collinear" in w.lower():
            warnings.warn(w, stacklevel=2)

    # Fit models to get sigma, Amean, fitted_values
    if not quiet:
        print(f"estimate_weights: Fitting {n_genes} genes with formula '{formula_clean}'...")

    results = fit_genes_parallel(
        Y=Y,
        X=X,
        Z=Z,
        block_sizes=block_sizes,
        weights_matrix=None,  # No weights on first pass
        re_names=re_names,
        reml=reml,
        n_jobs=n_jobs,
        backend="loky",
        verbose=not quiet,
        diagnostics_surface=diagnostics_surface,
        diagnostics_path=diagnostics_path,
    )

    # Extract results
    sigma = np.zeros(n_genes)
    Amean = np.zeros(n_genes)
    fitted_vals = np.zeros((n_genes, n_samples))
    converged = np.zeros(n_genes, dtype=bool)
    hit_lower_bound = np.zeros(n_genes, dtype=bool)
    gene_diagnostics = []

    # Per-RE arrays for theta and sigma_u, keyed by RE name.
    # For FE models (Z is None), re_names is empty and these stay empty dicts.
    theta_dict: dict[str, np.ndarray] = {}
    sigma_u_dict: dict[str, np.ndarray] = {}
    if re_names:
        for name in re_names:
            theta_dict[name] = np.zeros(n_genes)
            sigma_u_dict[name] = np.zeros(n_genes)

    for i, fit in enumerate(results):
        sigma[i] = fit["sigma"]
        Amean[i] = fit["Amean"]
        fitted_vals[i, :] = fit["fitted_values"]
        converged[i] = fit.get("converged", True)
        hit_lower_bound[i] = fit.get("hit_lower_bound", False)
        gene_diagnostics.append(fit.get("diagnostics", None))

        # Collect per-RE theta and sigma_u
        theta_val = fit.get("theta", {})
        if isinstance(theta_val, dict):
            for name, val in theta_val.items():
                if name in theta_dict:
                    theta_dict[name][i] = val
        sigma_u_val = fit.get("sigma_u", {})
        if isinstance(sigma_u_val, dict):
            for name, val in sigma_u_val.items():
                if name in sigma_u_dict:
                    sigma_u_dict[name][i] = val

    # Filter to converged genes with valid sigma
    keep_genes = np.isfinite(sigma) & (sigma > 0) & converged
    n_kept = np.sum(keep_genes)

    if not quiet:
        print(f"estimate_weights: {n_kept}/{n_genes} genes converged with valid sigma")

    if n_kept < 3:
        warnings.warn(f"Too few genes with valid model fits ({n_kept}). Returning uniform weights.",
                      stacklevel=2)
        return np.ones((n_genes, n_samples)), {}

    # Compute voom plot coordinates
    # sx = Amean + mean(log2(effective_lib_size + 1)) - log2(1e6)
    sx = Amean[keep_genes] + np.mean(np.log2(effective_lib_size + 1)) - np.log2(1e6)
    sy = np.sqrt(sigma[keep_genes])

    # Remove all-zero genes from trend fitting (if counts provided)
    if counts is not None:
        allzero = counts[keep_genes, :].sum(axis=1) == 0
        if np.any(allzero):
            sx_trend = sx[~allzero]
            sy_trend = sy[~allzero]
        else:
            sx_trend = sx
            sy_trend = sy
    else:
        sx_trend = sx
        sy_trend = sy

    # Fit lowess trend
    x_smooth, y_smooth = lowess_smooth(sx_trend, sy_trend, span=span)

    # Create interpolation function
    f_interp = interp1d(
        x_smooth,
        y_smooth,
        kind="linear",
        bounds_error=False,
        fill_value=(y_smooth[0], y_smooth[-1]),
    )

    # Compute precision weights for converged genes
    # fitted_cpm = 2^fitted_vals, fitted_count = fitted_cpm * (lib_size + 1) / 1e6
    fitted_cpm = 2 ** fitted_vals[keep_genes, :]
    fitted_count = (1e-6) * (fitted_cpm * (effective_lib_size + 1)[np.newaxis, :])
    fitted_logcount = np.log2(fitted_count)

    # w = 1 / f(fitted_logcount)^4
    weights_kept = 1.0 / (f_interp(fitted_logcount) ** 4)

    # Expand weights to all genes (non-converged get weight=1.0)
    weights = np.ones((n_genes, n_samples))
    weights[keep_genes, :] = weights_kept

    # Check for extreme weights
    max_weight = np.max(weights)
    if max_weight > 1e8:
        warnings.warn(
            f"Maximum precision weight is {max_weight:.2e}, suggesting poor smoothing. "
            "Consider adjusting span parameter.",
            stacklevel=2
        )

    trend_info = {
        "formula_original": formula_str,
        "formula_clean": formula_clean,
        "xy": {"x": sx, "y": sy},
        "line": {"x": x_smooth, "y": y_smooth},
        "Amean": Amean,
        "sigma": sigma,
        "fitted_vals": fitted_vals,
        "converged": converged,
        "hit_lower_bound": hit_lower_bound,
        "theta": theta_dict if theta_dict else None,
        "sigma_u": sigma_u_dict if sigma_u_dict else None,
        "gene_diagnostics": gene_diagnostics,
    }

    return weights, trend_info


def _estimate_weights_anndata(
    pb: AnnData,
    formula: Union[str, Formula],
    *,
    lib_size: np.ndarray | None = None,
    norm_factors: np.ndarray | None = None,
    span: str | float = "auto",
    reml: bool = True,
    n_jobs: int = 1,
    diagnostics_surface: bool = False,
    diagnostics_path: bool = False,
    quiet: bool = False,
) -> AnnData:
    """
    AnnData-mode implementation of estimate_weights.

    Fits models, computes weights, and stores results in the AnnData object.

    Parameters
    ----------
    pb : AnnData
        Pseudobulk data with log2 CPM in .X (samples × genes)
    formula : str or Formula
        Model formula
    lib_size : np.ndarray, optional
        Library sizes. If None, read from pb.obs['lib_size'] or computed.
    norm_factors : np.ndarray, optional
        Normalization factors. If None, read from pb.obs['norm_factors'].
    span : float or 'auto'
        Lowess smoothing parameter
    reml : bool
        REML estimation
    n_jobs : int
        Parallel jobs
    diagnostics_surface, diagnostics_path : bool
        Diagnostic options
    quiet : bool
        Suppress messages

    Returns
    -------
    AnnData
        Modified pb with .layers['weights'] and .uns['voom_info']
    """

    # Validate AnnData has required data
    if pb.X is None:
        raise ValueError("pb.X must contain log2 CPM values. Run log2cpm(pb) first.")

    n_samples, _n_genes = pb.shape

    # Extract Y matrix (genes × samples for fitting)
    if hasattr(pb.X, "toarray"):
        Y = pb.X.toarray().T  # (genes × samples)
    else:
        Y = pb.X.T  # (genes × samples)

    # Get metadata from pb.obs
    metadata = pb.obs

    # Get lib_size
    if lib_size is None:
        if "lib_size" in pb.obs.columns:
            lib_size = pb.obs["lib_size"].values
        elif "counts" in pb.layers:
            warnings.warn(
                "lib_size not found in pb.obs, computing from pb.layers['counts']. "
                "Note: lib_size should be computed BEFORE gene filtering for accurate "
                "normalization. Consider running compute_tmm_factors() earlier in the pipeline.",
                stacklevel=2
            )
            counts_matrix = pb.layers["counts"]
            if hasattr(counts_matrix, "toarray"):
                lib_size = np.asarray(counts_matrix.toarray().sum(axis=1)).flatten()
            else:
                lib_size = np.asarray(counts_matrix.sum(axis=1)).flatten()
        else:
            raise ValueError(
                "Cannot determine lib_size: not in pb.obs and pb.layers['counts'] not available. "
                "Run compute_tmm_factors(pb) first to populate pb.obs['lib_size']."
            )

    # Get norm_factors
    if norm_factors is None:
        if "norm_factors" in pb.obs.columns:
            norm_factors = pb.obs["norm_factors"].values
        else:
            norm_factors = np.ones(n_samples)

    # Get counts for all-zero gene detection
    if "counts" in pb.layers:
        counts_matrix = pb.layers["counts"]
        if hasattr(counts_matrix, "toarray"):
            counts = counts_matrix.toarray().T  # (genes × samples)
        else:
            counts = counts_matrix.T  # (genes × samples)
    else:
        counts = None

    # Call array implementation
    weights, trend_info = _estimate_weights_array(
        Y=Y,
        formula=formula,
        metadata=metadata,
        lib_size=lib_size,
        norm_factors=norm_factors,
        counts=counts,
        span=span,
        reml=reml,
        n_jobs=n_jobs,
        diagnostics_surface=diagnostics_surface,
        diagnostics_path=diagnostics_path,
        quiet=quiet,
    )

    # Store results in AnnData
    # Weights: transpose back to (samples × genes)
    pb.layers["weights"] = weights.T

    # Store voom info
    pb.uns["voom_info"] = {
        "formula_original": trend_info.get("formula_original", str(formula)),
        "formula_clean": trend_info.get("formula_clean", str(formula)),
        "Amean": trend_info["Amean"],
        "sigma": trend_info["sigma"],
        "fitted_vals": trend_info["fitted_vals"],
        "trend_xy": trend_info["xy"],
        "trend_line": trend_info["line"],
        "span": span,
        "converged": trend_info["converged"],
        "hit_lower_bound": trend_info["hit_lower_bound"],
        "theta": trend_info["theta"],
        "sigma_u": trend_info["sigma_u"],
        "gene_diagnostics": trend_info["gene_diagnostics"],
    }

    if not quiet:
        print("estimate_weights: Weights stored in .layers['weights']")
        print("estimate_weights: Voom info stored in .uns['voom_info']")

    return pb


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Normalization
    "compute_tmm_factors",
    # Gene filtering
    "filter_by_expr",
    # Transformation
    "log2cpm",
    # Weight estimation
    "estimate_weights",
    # Lowess smoothing (exposed for advanced use)
    "lowess_smooth",
]