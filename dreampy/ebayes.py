"""
Empirical Bayes Moderation in dreampy

Functions for empirical Bayes variance shrinkage and moderated statistics.
Matches limma::eBayes() and limma::squeezeVar() behavior.

Key Functions:
- trigamma_inverse(): Solve trigamma(y) = x via Newton-Raphson
- _fit_f_dist(): Fit scaled F-distribution to variances
- squeezeVar(): Shrink variances toward common prior
- eBayes(): Full empirical Bayes moderation pipeline
"""

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import digamma, gammaln, polygamma
from scipy.stats import t as t_dist

# Import DEFit for type hints - use TYPE_CHECKING to allow forward reference
# Actual import done inside ebayes() to avoid potential circular imports
if TYPE_CHECKING:
    from .de_types import DEFit


# =============================================================================
# Constants
# =============================================================================

# Variance floor constant (matches limma behavior)
VARIANCE_FLOOR_FACTOR = 1e-15  # Floor = VARIANCE_FLOOR_FACTOR * median(sigma^2)

__all__ = [
    "EBayesResult",
    "eBayes",
    "ebayes",
    "squeezeVar",
    "trigamma_inverse",
]


# =============================================================================
# Core Functions
# =============================================================================


def trigamma_inverse(x: np.ndarray) -> np.ndarray:
    """
    Solve trigamma(y) = x for y via Newton-Raphson.

    Parameters
    ----------
    x : np.ndarray
        Input values (positive). trigamma(y) approaches 0 as y→∞
        and approaches ∞ as y→0.

    Returns
    -------
    y : np.ndarray
        Values such that scipy.special.polygamma(1, y) ≈ x

    Notes
    -----
    Algorithm from limma::trigammaInverse():

    1. Handle edge cases:
       - x < 0: return NaN
       - x > 1e7: return 1/sqrt(x) (asymptotic)
       - x < 1e-6: return 1/x (asymptotic)

    2. Newton-Raphson iteration:
       y_new = y + trigamma(y) * (1 - trigamma(y)/x) / psigamma(y)

    3. Convergence: |y_new - y| / y < 1e-8

    Reference: limma/R/ebayes.R::trigammaInverse
    """
    x = np.asarray(x, dtype=np.float64)
    scalar_input = x.ndim == 0
    x = np.atleast_1d(x)

    y = np.empty_like(x)

    # Handle edge cases
    neg_mask = x < 0
    large_mask = x > 1e7
    small_mask = x < 1e-6
    normal_mask = ~neg_mask & ~large_mask & ~small_mask

    y[neg_mask] = np.nan
    y[large_mask] = 1.0 / np.sqrt(x[large_mask])
    y[small_mask] = 1.0 / x[small_mask]

    # Newton-Raphson for remaining values
    if np.any(normal_mask):
        x_valid = x[normal_mask]
        # Initial guess: 0.5 + 1/x
        y_iter = 0.5 + 1.0 / x_valid

        for _ in range(50):
            tri = polygamma(1, y_iter)  # trigamma
            psi = polygamma(2, y_iter)  # tetragamma (derivative of trigamma)

            # Newton step: y_new = y - f(y)/f'(y)
            # where f(y) = trigamma(y) - x
            # f'(y) = tetragamma(y)
            # Limma uses: y + tri*(1 - tri/x)/psi
            # which is equivalent to: y - (tri - x)/psi when simplified
            dif = tri * (1 - tri / x_valid) / psi
            y_iter = y_iter + dif

            if np.max(np.abs(dif / y_iter)) < 1e-8:
                break

        y[normal_mask] = y_iter

    return y[0] if scalar_input else y


def _apply_variance_floor(sigma_sq: np.ndarray) -> np.ndarray:
    """
    Apply variance floor matching limma exactly.

    Parameters
    ----------
    sigma_sq : np.ndarray
        Gene-wise variance estimates

    Returns
    -------
    sigma_sq_floored : np.ndarray
        Variances with floor applied

    Notes
    -----
    From limma: s2[s2 < 1e-15*median(s2)] <- 1e-15*median(s2)

    Edge case: If median of positive variances is 0 or no positive variances
    exist, return unchanged (no floor applicable).
    """
    positive_mask = sigma_sq > 0
    if not np.any(positive_mask):
        return sigma_sq.copy()

    median_var = np.median(sigma_sq[positive_mask])
    if median_var <= 0:
        return sigma_sq.copy()

    floor = VARIANCE_FLOOR_FACTOR * median_var
    return np.maximum(sigma_sq, floor)


def _logmdigamma(a: np.ndarray) -> np.ndarray:
    """
    Compute log(a) - digamma(a).

    This is a helper function used in fitFDistUnequalDF1.
    For large a, this approaches 1/(2a) asymptotically.

    Parameters
    ----------
    a : np.ndarray
        Input values (positive)

    Returns
    -------
    np.ndarray
        log(a) - digamma(a)

    Notes
    -----
    This function is numerically stable for all positive a.

    Reference: limma/R/fitFDist.R::logmdigamma
    """
    a = np.asarray(a, dtype=np.float64)
    return np.log(a) - digamma(a)


def _fit_f_dist(
    sigma_sq: np.ndarray,
    df: np.ndarray,
) -> tuple[float, float]:
    """
    Fit scaled F-distribution to observed variances.

    Parameters
    ----------
    sigma_sq : np.ndarray
        (n_genes,) observed residual variances (sigma^2)
    df : np.ndarray
        (n_genes,) residual degrees of freedom

    Returns
    -------
    s2_prior : float
        Prior variance (scale parameter)
    df_prior : float
        Prior degrees of freedom

    Notes
    -----
    Algorithm from limma::fitFDist():

    1. Filter: keep only genes with df > 0 and finite, positive variance
    2. Apply variance floor: 1e-15 * median
    3. Moment estimation:
       - E[log(s²)] = digamma(df/2) - log(df/2) + log(s²_prior)
       - Var[log(s²)] = trigamma(df/2) + trigamma(df_prior/2)
    4. Solve for df_prior using trigamma_inverse
    5. Compute s2_prior from first moment

    Edge case: If df_prior < 1e-6, set df_prior = 0 (no shrinkage)

    Reference: limma/R/ebayes.R::fitFDist
    """
    sigma_sq = np.asarray(sigma_sq)
    df = np.asarray(df)

    # Filter valid observations
    valid = (df > 0) & np.isfinite(sigma_sq) & (sigma_sq > 0)
    if np.sum(valid) < 2:
        return 1.0, 0.0  # No shrinkage possible

    s2 = sigma_sq[valid].copy()
    d = df[valid]

    # Apply variance floor
    s2 = _apply_variance_floor(s2)

    # Log-variance statistics
    log_s2 = np.log(s2)

    # Moment estimation
    # E[log(s²)] = ψ(d/2) - log(d/2) + log(s²_0)
    e_log_s2 = np.mean(log_s2)
    mean_digamma_term = np.mean(digamma(d / 2) - np.log(d / 2))

    # Var[log(s²)] = trigamma(d/2) + trigamma(d0/2)
    var_log_s2 = np.var(log_s2, ddof=1)
    mean_trigamma = np.mean(polygamma(1, d / 2))

    # Solve for df_prior
    target_trigamma = var_log_s2 - mean_trigamma
    if target_trigamma > 0:
        df_prior = 2 * trigamma_inverse(target_trigamma)
    else:
        df_prior = np.inf  # No shrinkage (variance is too low)

    # Handle edge cases
    if not np.isfinite(df_prior) or df_prior < 1e-6:
        # No meaningful shrinkage
        return np.median(s2), 0.0

    # Compute s2_prior from first moment
    log_s2_prior = e_log_s2 - mean_digamma_term + digamma(df_prior / 2) - np.log(df_prior / 2)
    s2_prior = np.exp(log_s2_prior)

    return float(s2_prior), float(df_prior)


def _fit_f_dist_unequal_df1(
    x: np.ndarray,
    df1: np.ndarray,
) -> tuple[float, float]:
    """
    Fit scaled F-distribution with unequal df1 values per observation.

    This is the profile likelihood method used when residual degrees of
    freedom vary across genes (typical for LMMs with trace-based rdf).

    Parameters
    ----------
    x : np.ndarray
        (n_genes,) observed residual variances (sigma^2)
    df1 : np.ndarray
        (n_genes,) residual degrees of freedom (can vary per gene)

    Returns
    -------
    s2_prior : float
        Prior variance (scale parameter)
    df_prior : float
        Prior degrees of freedom (df2 in limma notation)

    Notes
    -----
    Algorithm from limma::fitFDistUnequalDF1():

    1. Filter valid observations: x > 0, df1 >= 0.01
    2. Apply variance floor: xpos = max(x, 1e-12 * median(x))
    3. Compute transformed response: e = log(xpos) + logmdigamma(df1/2)
    4. Compute precision weights: w = 1 / trigamma(df1/2)
    5. Weighted mean: emean = sum(w * e) / sum(w)
    6. Optimize profile likelihood over df2:
       - Parameterize as par ∈ (0.5, 0.9998) where d2 = par / (1 - par)
       - Minimize -2 * log-likelihood of scaled F-distribution
    7. Recover scale: s20 = exp(emean - logmdigamma(d2))
    8. Return: scale = s20, df2 = 2 * d2

    The key difference from fitFDist is that this method:
    - Weights genes by 1/trigamma(df1/2) (more weight to genes with higher df)
    - Uses profile likelihood instead of moment matching
    - Properly handles varying df1 across genes

    Reference: limma/R/fitFDist.R::fitFDistUnequalDF1
    """
    x = np.asarray(x, dtype=np.float64)
    df1 = np.asarray(df1, dtype=np.float64)
    n = len(x)

    # Handle length mismatch
    if len(df1) == 1:
        df1 = np.full(n, df1[0])
    elif len(df1) != n:
        raise ValueError(f"x and df1 are different lengths: {n} vs {len(df1)}")

    # =========================================================================
    # Filter valid observations
    # =========================================================================
    # Mark invalid: NA, x <= 0, df1 < 0.01
    valid = np.isfinite(x) & (x > 0) & np.isfinite(df1) & (df1 >= 0.01)
    n_informative = np.sum(valid)

    if n_informative < 2:
        return np.nan, np.nan

    # Work with valid subset
    x_valid = x[valid]
    df1_valid = df1[valid]

    # =========================================================================
    # Apply variance floor (matching limma)
    # =========================================================================
    m = np.median(x_valid)
    xpos = np.maximum(x_valid, 1e-12 * m)

    # =========================================================================
    # Compute transformed response and weights
    # =========================================================================
    # d1 = df1 / 2 (half degrees of freedom, used in F-distribution parameterization)
    d1 = df1_valid / 2.0

    # z = log(xpos)
    z = np.log(xpos)

    # e = log(x) + logmdigamma(d1) = log(x) + log(d1) - digamma(d1)
    # This is the "normalized" log-variance that has mean = log(s20) + logmdigamma(d2)
    e = z + _logmdigamma(d1)

    # Precision weights: w = 1 / trigamma(d1)
    # Genes with higher df1 get more weight (more precise variance estimates)
    w = 1.0 / polygamma(1, d1)

    # =========================================================================
    # Weighted mean of e
    # =========================================================================
    emean = np.sum(w * e) / np.sum(w)

    # =========================================================================
    # Precompute for likelihood
    # =========================================================================
    # d1 * xpos is used repeatedly in the likelihood
    d1x = d1 * xpos

    # =========================================================================
    # Profile likelihood optimization
    # =========================================================================
    # We parameterize d2 = par / (1 - par) where par ∈ (0, 1)
    # This ensures d2 > 0 and allows bounded optimization
    # par = 0.5 => d2 = 1 (df2 = 2)
    # par = 0.9998 => d2 ≈ 4999 (df2 ≈ 9998)

    def minus_twice_log_lik(par: float) -> float:
        """
        Compute -2 * log-likelihood as function of par.

        The scaled F-distribution has density:
        f(x; df1, df2, s0) = F(x/s0; df1, df2) / s0

        where F is the standard F-distribution density.

        Log-likelihood for observation i:
        ll_i = log(Gamma(d1_i + d2)) - log(Gamma(d1_i)) - log(Gamma(d2))
               + d1_i * log(d1_i) + d2 * log(d2)
               - d1_i * log(s20) - d2 * log(s20)
               - (d1_i + d2) * log(d1_i * x_i / s20 + d2)

        After profiling out s20 using emean, the terms involving s20 simplify.
        """
        # Transform to d2 (half of df2)
        d2 = par / (1.0 - par)

        # Profile scale parameter: s20 = exp(emean - logmdigamma(d2))
        # This is derived from setting derivative of likelihood w.r.t. log(s20) to zero
        # Combined term: d2 * s20 = d2 * exp(emean - logmdigamma(d2))
        d2s20 = d2 * np.exp(emean - _logmdigamma(d2))

        # Log-likelihood terms (dropping constants that don't depend on d2)
        # ll = sum( -(d1+d2)*log(1 + d1*x/(d2*s20)) - d1*log(d2*s20)
        #           + lgamma(d1+d2) - lgamma(d2) )
        #
        # Note: lgamma(d1) terms are constant w.r.t. d2, so omitted

        ll_terms = (
            -(d1 + d2) * np.log1p(d1x / d2s20) - d1 * np.log(d2s20) + gammaln(d1 + d2) - gammaln(d2)
        )

        return -2.0 * np.sum(ll_terms)

    # Optimize over par in (0.5, 0.9998)
    # Lower bound 0.5 => df2 >= 2 (reasonable minimum)
    # Upper bound 0.9998 => df2 <= ~10000 (practical maximum)
    try:
        result = minimize_scalar(
            minus_twice_log_lik, bounds=(0.5, 0.9998), method="bounded", options={"xatol": 1e-10}
        )
        par_opt = result.x
    except (ValueError, RuntimeError):
        # Optimization failed - return NaN
        return np.nan, np.nan

    # =========================================================================
    # Recover parameters
    # =========================================================================
    d2_opt = par_opt / (1.0 - par_opt)

    # Scale parameter: s20 = exp(emean - logmdigamma(d2))
    s20 = np.exp(emean - _logmdigamma(d2_opt))

    # df_prior = 2 * d2 (convert from half-df to full df)
    df_prior = 2.0 * d2_opt

    # =========================================================================
    # Validation
    # =========================================================================
    if not np.isfinite(s20) or s20 <= 0:
        return np.nan, np.nan
    if not np.isfinite(df_prior) or df_prior <= 0:
        return np.nan, np.nan

    return float(s20), float(df_prior)


def squeezeVar(
    sigma_sq: np.ndarray,
    df: np.ndarray,
    robust: bool = False,
) -> tuple[np.ndarray, float, float]:
    """
    Squeeze gene-wise variances toward common prior.

    Parameters
    ----------
    sigma_sq : np.ndarray
        (n_genes,) observed residual variances
    df : np.ndarray
        (n_genes,) residual degrees of freedom (can be non-integer for LMMs)
    robust : bool
        If True, use robust fitting (winsorized variances). NOT YET IMPLEMENTED.

    Returns
    -------
    s2_post : np.ndarray
        (n_genes,) posterior variance estimates
    s2_prior : float
        Estimated prior variance
    df_prior : float
        Estimated prior degrees of freedom

    Notes
    -----
    Posterior variance is:
    s²_post = (df * s² + df_prior * s²_prior) / (df + df_prior)

    This is a weighted average, with more shrinkage when:
    - df is small (uncertain observed variance)
    - df_prior is large (strong prior)

    **Dispatch logic (matching limma::squeezeVar):**
    - If all df values are identical: use legacy fitFDist (moment matching)
    - If df values vary: use fitFDistUnequalDF1 (profile likelihood)

    This matters for LMMs where trace-based rdf varies per gene.

    Edge case handling:
    - df <= 0: s2_post = s2_prior (full shrinkage to prior)
    - df_prior = 0 or inf: s2_post = s2 (no shrinkage)

    Reference: limma/R/squeezeVar.R
    """
    sigma_sq = np.asarray(sigma_sq)
    df = np.asarray(df)

    if robust:
        raise NotImplementedError("Robust squeezeVar not yet implemented")

    # =========================================================================
    # Determine whether to use legacy mode (all df identical) or unequal df1
    # =========================================================================
    # This matches limma's logic:
    #   legacy <- identical(min(dfp), max(dfp))
    # where dfp = df[df > 0]

    df_valid = df[df > 0]

    if len(df_valid) < 2:
        # Not enough valid observations - return no shrinkage
        median_s2 = float(np.median(sigma_sq[sigma_sq > 0])) if np.any(sigma_sq > 0) else 1.0
        return sigma_sq.copy(), median_s2, 0.0

    # Check if all df values are identical (within numerical tolerance)
    df_min = np.min(df_valid)
    df_max = np.max(df_valid)
    legacy = df_min == df_max  # Use exact equality like R's identical()

    # =========================================================================
    # Fit F-distribution using appropriate method
    # =========================================================================
    if legacy:
        # All df are equal - use moment matching (faster, equivalent)
        s2_prior, df_prior = _fit_f_dist(sigma_sq, df)
    else:
        # df vary - use profile likelihood method
        s2_prior, df_prior = _fit_f_dist_unequal_df1(sigma_sq, df)

    # =========================================================================
    # Handle fitting failures
    # =========================================================================
    if np.isnan(s2_prior) or np.isnan(df_prior):
        # Fitting failed - return no shrinkage
        median_s2 = float(np.median(sigma_sq[sigma_sq > 0])) if np.any(sigma_sq > 0) else 1.0
        return sigma_sq.copy(), median_s2, 0.0

    # =========================================================================
    # Compute posterior variances
    # =========================================================================
    if df_prior == 0 or not np.isfinite(df_prior):
        # No shrinkage
        s2_post = sigma_sq.copy()
    elif np.isinf(df_prior):
        # Full shrinkage to prior
        s2_post = np.full_like(sigma_sq, s2_prior)
    else:
        # Normal case: weighted average
        # s2_post = (df * s2 + df_prior * s2_prior) / (df + df_prior)
        df_safe = np.maximum(df, 0)
        s2_post = (df_safe * sigma_sq + df_prior * s2_prior) / (df_safe + df_prior)
        # For genes with df <= 0, use prior
        s2_post = np.where(df <= 0, s2_prior, s2_post)

    return s2_post, s2_prior, df_prior


def tmixture_vector(
    tstat: np.ndarray,
    stdev_unscaled: np.ndarray,
    df: np.ndarray,
    proportion: float,
    v0_lim: tuple[float, float] | None = None,
) -> float:
    """
    Estimate coefficient prior variance from t-statistics using order statistics.

    Assumes DE genes have t-stats from sqrt(1 + v0/v1) * t(df), where v1 = stdev_unscaled^2
    and v0 (var.prior) is to be estimated.

    Parameters
    ----------
    tstat : np.ndarray
        (n_genes,) t-statistics for one coefficient
    stdev_unscaled : np.ndarray
        (n_genes,) unscaled standard errors for that coefficient
    df : np.ndarray
        (n_genes,) degrees of freedom
    proportion : float
        Expected proportion of DE genes
    v0_lim : tuple of (float, float), optional
        Bounds for v0 estimate

    Returns
    -------
    v0 : float
        Estimated coefficient prior variance, or np.nan if estimation fails

    Notes
    -----
    Reference: limma/R/ebayes.R::tmixture.vector
    """
    # Remove missing values
    valid = np.isfinite(tstat) & np.isfinite(stdev_unscaled) & np.isfinite(df)
    if not np.any(valid):
        return np.nan

    tstat = np.abs(tstat[valid].copy())
    stdev_unscaled = stdev_unscaled[valid].copy()
    df = df[valid].copy()

    # ntarget t-statistics will be used for estimation
    ngenes = len(tstat)
    ntarget = int(np.ceil(proportion / 2 * ngenes))
    if ntarget < 1:
        return np.nan

    # If ntarget is very small, ensure p at least matches selected proportion
    # This ensures ptarget < 1
    p = max(ntarget / ngenes, proportion)

    # Method requires equal df - standardize to MaxDF
    MaxDF = np.max(df)
    if MaxDF <= 0:
        return np.nan

    # Transform t-stats to common df using quantile mapping
    needs_transform = df < MaxDF
    if np.any(needs_transform):
        # P(T > |t|) at original df, using log for numerical stability
        tail_log_p = t_dist.logsf(tstat[needs_transform], df[needs_transform])
        # Transform to t-stat at MaxDF with same tail probability
        tstat[needs_transform] = t_dist.isf(np.exp(tail_log_p), MaxDF)
        df[needs_transform] = MaxDF

    # Select top ntarget statistics
    order = np.argsort(tstat)[::-1]  # descending
    top_idx = order[:ntarget]
    tstat_top = tstat[top_idx]
    v1 = stdev_unscaled[top_idx] ** 2

    # Compare to order statistics
    r = np.arange(1, ntarget + 1)
    p0 = 2 * t_dist.sf(tstat_top, MaxDF)  # Two-tailed p-value under null

    # Expected quantile under mixture
    ptarget = ((r - 0.5) / ngenes - (1 - p) * p0) / p

    # Estimate v0 for each top gene
    v0 = np.zeros(ntarget)
    pos = ptarget > p0
    if np.any(pos):
        # Inverse quantile at target p
        qtarget = t_dist.isf(ptarget[pos] / 2, MaxDF)
        # v0 such that sqrt(1 + v0/v1) * t_null = t_obs
        # => v0 = v1 * ((t_obs/qtarget)^2 - 1)
        v0[pos] = v1[pos] * ((tstat_top[pos] / qtarget) ** 2 - 1)

    # Apply limits if provided
    if v0_lim is not None:
        v0 = np.clip(v0, v0_lim[0], v0_lim[1])

    # Return mean of non-negative estimates
    v0 = np.maximum(v0, 0)
    return float(np.mean(v0))


def tmixture_matrix(
    tstat: np.ndarray,
    stdev_unscaled: np.ndarray,
    df: np.ndarray,
    proportion: float,
    v0_lim: tuple[float, float] | None = None,
) -> np.ndarray:
    """
    Estimate coefficient prior variance for each coefficient column.

    Parameters
    ----------
    tstat : np.ndarray
        (n_genes, n_coef) t-statistics
    stdev_unscaled : np.ndarray
        (n_genes, n_coef) unscaled standard errors
    df : np.ndarray
        (n_genes,) degrees of freedom
    proportion : float
        Expected proportion of DE genes
    v0_lim : tuple of (float, float), optional
        Bounds for v0 estimate

    Returns
    -------
    v0 : np.ndarray
        (n_coef,) estimated coefficient prior variances

    Notes
    -----
    Reference: limma/R/ebayes.R::tmixture.matrix
    """
    _n_genes, n_coef = tstat.shape
    v0 = np.zeros(n_coef)

    for j in range(n_coef):
        v0[j] = tmixture_vector(tstat[:, j], stdev_unscaled[:, j], df, proportion, v0_lim)

    return v0


@dataclass
class EBayesResult:
    """
    Results from empirical Bayes moderation.
    """

    # Prior estimates
    s2_prior: float
    df_prior: float

    # Posterior variances
    s2_post: np.ndarray
    df_total: np.ndarray

    # Moderated statistics
    t: np.ndarray  # Moderated t-statistics
    p_value: np.ndarray  # P-values from moderated t

    # Optional additional statistics
    z_std: np.ndarray | None = None  # Standardized z-scores
    lods: np.ndarray | None = None  # Log-odds of DE (B statistic)
    var_prior: np.ndarray | None = None  # (n_coef,) coefficient prior variance for B statistic


def eBayes(
    coefficients: np.ndarray,
    stdev_unscaled: np.ndarray,
    sigma_sq: np.ndarray,
    df_resid: np.ndarray,
    df_resid_test: np.ndarray | None = None,
    robust: bool = False,
    proportion: float = 0.01,
    stdev_coef_lim: tuple[float, float] = (0.1, 4.0),
) -> EBayesResult:
    """
    Apply empirical Bayes moderation to fitted models.

    Parameters
    ----------
    coefficients : np.ndarray
        (n_genes, n_coef) coefficient estimates
    stdev_unscaled : np.ndarray
        (n_genes, n_coef) unscaled standard deviations
    sigma_sq : np.ndarray
        (n_genes,) residual variances
    df_resid : np.ndarray
        (n_genes,) residual degrees of freedom for eBayes variance shrinkage
    df_resid_test : np.ndarray, optional
        (n_genes, n_coef) residual degrees of freedom for hypothesis testing.
        If None, uses df_resid broadcast to all coefficients.
        When Satterthwaite df is available, this provides per-coefficient df.
    robust : bool
        Use robust fitting (winsorized variances)
    proportion : float
        Expected proportion of DE genes (for B statistic)
    stdev_coef_lim : tuple of (float, float)
        Limits on coefficient standard deviation for B statistic estimation.
        Default (0.1, 4.0) matches limma.

    Returns
    -------
    EBayesResult
        Contains:
        - s2_prior, df_prior: prior estimates
        - s2_post: posterior variances
        - df_total: (n_genes, n_coef) min(df_prior + df_resid_test, df_pooled)
        - t: moderated t-statistics
        - p_value: p-values from moderated t
        - z_std: standardized z-scores (for cross-assay comparison)
        - lods: log-odds of differential expression
        - var_prior: (n_coef,) coefficient prior variances

    Notes
    -----
    The moderated t-statistic is:
        t = β / (sqrt(s²_post) * stdev_unscaled)

    P-values use t-distribution with df = df_total (per gene AND coefficient).

    Reference: limma/R/ebayes.R::eBayes
    """
    n_genes, n_coef = coefficients.shape

    # Squeeze variances (uses scalar df_resid per gene)
    s2_post, s2_prior, df_prior = squeezeVar(sigma_sq, df_resid, robust=robust)

    # Build df_resid_test if not provided
    # If provided, it's (n_genes, n_coef) with per-coefficient Satterthwaite df
    # If not provided, broadcast df_resid to all coefficients
    if df_resid_test is None:
        df_resid_test = np.tile(df_resid[:, np.newaxis], (1, n_coef))

    # Total degrees of freedom with pooled cap (Bug #23 fix)
    # limma: df.total <- pmin(df.residual + df.prior, sum(df.residual))
    # Now (n_genes, n_coef) shape
    df_total = df_resid_test + df_prior
    df_pooled = np.nansum(df_resid)  # Pool uses scalar df for shrinkage
    df_total = np.minimum(df_total, df_pooled)

    # Compute moderated t-statistics
    # t = beta / (sqrt(s2_post) * stdev_unscaled)
    # Broadcast: (n_genes,) -> (n_genes, 1) for element-wise ops
    se = np.sqrt(s2_post[:, np.newaxis]) * stdev_unscaled
    t = np.where(se > 0, coefficients / se, 0.0)

    # Compute p-values from moderated t
    # Each gene AND coefficient has its own df_total
    p_value = np.zeros_like(t)
    for i in range(n_genes):
        for j in range(n_coef):
            if df_total[i, j] > 0:
                p_value[i, j] = 2 * t_dist.sf(np.abs(t[i, j]), df=df_total[i, j])
            else:
                p_value[i, j] = np.nan

    # Compute standardized z-scores for cross-assay comparison
    z_std = _compute_z_std(t, df_total)

    # Compute coefficient prior variance for B statistic (Bug #21 fix)
    # limma: var.prior.lim <- stdev.coef.lim^2 / median(s2.prior)
    s2_prior_median = np.median(s2_prior) if np.ndim(s2_prior) > 0 else s2_prior
    if s2_prior_median > 0:
        var_prior_lim = (
            stdev_coef_lim[0] ** 2 / s2_prior_median,
            stdev_coef_lim[1] ** 2 / s2_prior_median,
        )
    else:
        var_prior_lim = None

    # For tmixture estimation, use scalar df per gene (df_resid + df_prior)
    # tmixture_matrix expects (n_genes,) df, not per-coefficient
    df_for_tmix = df_resid + df_prior
    df_for_tmix = np.minimum(df_for_tmix, df_pooled)
    var_prior = tmixture_matrix(t, stdev_unscaled, df_for_tmix, proportion, var_prior_lim)

    # Handle NA in var_prior - limma falls back to 1/s2_prior
    if np.any(np.isnan(var_prior)):
        fallback = 1.0 / s2_prior if s2_prior > 0 else 1.0
        var_prior = np.where(np.isnan(var_prior), fallback, var_prior)

    # Compute log-odds of DE (B statistic) using correct limma formula
    # Bug #54 fix: In R's variancePartition eBayes for MArrayLM2, lods is
    # computed inside limma::eBayes (Phase 2) using scalar rdf + df_prior,
    # NOT per-coefficient Satterthwaite df. The Satterthwaite df_total is
    # only used for t-statistics and p-values (Phase 3). Since lods is never
    # recomputed after Phase 3, it retains the scalar-based df.
    # df_for_tmix already holds the correct scalar df; broadcast to (n_genes, n_coef).
    df_total_lods = np.tile(df_for_tmix[:, np.newaxis], (1, n_coef))
    lods = _compute_lods(t, stdev_unscaled, df_total_lods, df_prior, proportion, var_prior)

    return EBayesResult(
        s2_prior=s2_prior,
        df_prior=df_prior,
        s2_post=s2_post,
        df_total=df_total,
        t=t,
        p_value=p_value,
        z_std=z_std,
        lods=lods,
        var_prior=var_prior,
    )


def _compute_z_std(
    t: np.ndarray,
    df: np.ndarray,
) -> np.ndarray:
    """
    Convert t-statistics to standardized z-scores.

    Parameters
    ----------
    t : np.ndarray
        (n_genes, n_coef) moderated t-statistics
    df : np.ndarray
        (n_genes, n_coef) degrees of freedom (per gene AND coefficient)

    Returns
    -------
    z : np.ndarray
        (n_genes, n_coef) standardized z-scores

    Notes
    -----
    z = sign(t) * sqrt(df * (exp(t²/df) - 1))

    This is the Wilson-Hilferty approximation that makes z-scores
    comparable across genes with different df.

    Reference: limma/R/ebayes.R::zscoreT
    """
    from scipy.stats import norm

    n_genes, n_coef = t.shape
    z = np.zeros_like(t)

    for i in range(n_genes):
        for j in range(n_coef):
            if df[i, j] > 0:
                # Convert t to p-value, then p-value to z
                p = 2 * t_dist.sf(np.abs(t[i, j]), df=df[i, j])
                # Avoid p = 0 or p = 1
                p = np.clip(p, 1e-15, 1 - 1e-15)
                z[i, j] = np.sign(t[i, j]) * (-norm.ppf(p / 2))
            else:
                z[i, j] = np.nan

    return z


def _compute_lods(
    t: np.ndarray,
    stdev_unscaled: np.ndarray,
    df_total: np.ndarray,
    df_prior: float,
    proportion: float,
    var_prior: np.ndarray,
) -> np.ndarray:
    """
    Compute log-odds of differential expression (B statistic).

    Parameters
    ----------
    t : np.ndarray
        (n_genes, n_coef) moderated t-statistics
    stdev_unscaled : np.ndarray
        (n_genes, n_coef) unscaled standard errors
    df_total : np.ndarray
        (n_genes, n_coef) total degrees of freedom
    df_prior : float
        Prior degrees of freedom
    proportion : float
        Expected proportion of DE genes
    var_prior : np.ndarray
        (n_coef,) coefficient prior variances from tmixture_matrix

    Returns
    -------
    lods : np.ndarray
        (n_genes, n_coef) log-odds of DE

    Notes
    -----
    Implements limma's exact B statistic formula:

    r = (stdev.unscaled^2 + var.prior) / stdev.unscaled^2
    kernel = (1+df.total)/2 * log((t^2+df.total) / (t^2/r+df.total))
    lods = log(proportion/(1-proportion)) - log(r)/2 + kernel

    Special case for df.prior > 1e6 (essentially infinite):
    kernel = t^2 * (1 - 1/r) / 2

    Reference: limma/R/ebayes.R::.ebayes lines 66-86
    """
    n_genes, n_coef = t.shape
    lods = np.zeros_like(t)

    # Prior odds
    log_prior_odds = np.log(proportion / (1 - proportion))

    # Compute variance ratio r for each gene and coefficient
    # r = (stdev_unscaled^2 + var_prior) / stdev_unscaled^2
    # var_prior is (n_coef,), broadcast to (n_genes, n_coef)
    stdev_sq = stdev_unscaled**2
    # Broadcast var_prior: (n_coef,) -> (1, n_coef) -> (n_genes, n_coef)
    r = (stdev_sq + var_prior[np.newaxis, :]) / stdev_sq
    # Handle division by zero
    r = np.where(stdev_sq > 0, r, 1.0)

    t2 = t**2

    # Check if df_prior is essentially infinite
    Infdf = df_prior > 1e6

    for i in range(n_genes):
        for j in range(n_coef):
            if df_total[i, j] > 0:
                if Infdf:
                    # Simplified kernel for infinite df.prior
                    # kernel = t^2 * (1 - 1/r) / 2
                    kernel = t2[i, j] * (1 - 1 / r[i, j]) / 2
                else:
                    # Full kernel formula
                    # kernel = (1+df.total)/2 * log((t^2+df.total) / (t^2/r+df.total))
                    kernel = (
                        (1 + df_total[i, j])
                        / 2
                        * np.log(
                            (t2[i, j] + df_total[i, j]) / (t2[i, j] / r[i, j] + df_total[i, j])
                        )
                    )

                # lods = log(p/(1-p)) - log(r)/2 + kernel
                lods[i, j] = log_prior_odds - np.log(r[i, j]) / 2 + kernel
            else:
                lods[i, j] = np.nan

    return lods


# =============================================================================
# High-Level Wrapper
# =============================================================================


def ebayes(
    fit: "DEFit",
    robust: bool = False,
    proportion: float = 0.01,
    stdev_coef_lim: tuple[float, float] = (0.1, 4.0),
) -> "DEFit":
    """
    Apply empirical Bayes moderation to a fitted model.

    High-level wrapper around eBayes() that accepts and returns DEFit.
    Returns a NEW DEFit object (does not modify input).

    Parameters
    ----------
    fit : DEFit
        Fitted model from fit_models(). Must have valid:
        - coefficients: (n_genes, n_coef) coefficient estimates
        - stdev_unscaled: (n_genes, n_coef) unscaled standard errors
        - sigma: (n_genes,) residual standard deviations
        - df_resid_eb: (n_genes,) residual degrees of freedom
    robust : bool, default=False
        Use robust fitting (winsorized variances).
        NOT YET IMPLEMENTED - will raise NotImplementedError.
    proportion : float, default=0.01
        Expected proportion of differentially expressed genes.
        Only affects the lods (B statistic), not t-statistics or p-values.
    stdev_coef_lim : tuple of (float, float), default=(0.1, 4.0)
        Limits on coefficient standard deviation for B statistic estimation.
        Matches limma default.

    Returns
    -------
    DEFit
        New DEFit object with empirical Bayes results populated:
        - s2_prior: Prior variance estimate
        - df_prior: Prior degrees of freedom
        - s2_post: (n_genes,) posterior variance estimates
        - df_total: (n_genes,) total df (min of df_prior + df_resid_eb, df_pooled)
        - t: (n_genes, n_coef) moderated t-statistics
        - p_value: (n_genes, n_coef) p-values from moderated t
        - z_std: (n_genes, n_coef) standardized z-scores
        - lods: (n_genes, n_coef) log-odds of DE (B statistic)
        - var_prior: (n_coef,) coefficient prior variances for B statistic

    Raises
    ------
    ValueError
        If fit has no genes, or if stdev_unscaled/df_resid_eb are all NaN
        (indicating Bug #12 fix was not applied).
    NotImplementedError
        If robust=True (not yet implemented).

    Examples
    --------
    >>> fit = fit_models(adata, formula, metadata)
    >>> fit = ebayes(fit)
    >>> results = top_table(fit, coef="treatment")

    Notes
    -----
    The moderated t-statistic is computed as:
        t = β / (sqrt(s²_post) * stdev_unscaled)

    where s²_post is the posterior variance estimate that shrinks
    gene-wise variances toward a common prior.

    Reference: limma/R/ebayes.R::eBayes
    """
    # Import DEFit here to avoid circular imports at module load time

    # =================================================================
    # Step 1: Validate input DEFit
    # =================================================================

    n_genes = len(fit.gene_names)

    if n_genes == 0:
        raise ValueError("DEFit has no genes. Cannot apply eBayes to empty fit.")

    # Check coefficients shape
    if fit.coefficients.size == 0:
        raise ValueError("DEFit.coefficients is empty. Fit may have failed.")

    # Check stdev_unscaled has real values
    if fit.stdev_unscaled.size == 0:
        raise ValueError(
            "DEFit.stdev_unscaled is empty. "
            "Ensure Bug #12 fix is applied (fitting.py must compute stdev_unscaled)."
        )

    if np.all(np.isnan(fit.stdev_unscaled)):
        raise ValueError(
            "DEFit.stdev_unscaled is all NaN. "
            "All gene fits may have failed, or Bug #12 fix is not applied."
        )

    # Check df_resid_eb has real values
    if fit.df_resid_eb.size == 0:
        raise ValueError(
            "DEFit.df_resid_eb is empty. "
            "Ensure Bug #12 fix is applied (fitting.py must compute df_resid)."
        )

    if np.all(np.isnan(fit.df_resid_eb)):
        raise ValueError(
            "DEFit.df_resid_eb is all NaN. "
            "All gene fits may have failed, or Bug #12 fix is not applied."
        )

    # Check sigma has some finite values
    if fit.sigma.size == 0:
        raise ValueError("DEFit.sigma is empty. Fit may have failed.")

    n_finite_sigma = np.sum(np.isfinite(fit.sigma))
    if n_finite_sigma == 0:
        raise ValueError("DEFit.sigma has no finite values. All gene fits may have failed.")

    # =================================================================
    # Step 2: Extract arrays from DEFit
    # =================================================================

    coefficients = fit.coefficients  # (n_genes, n_coef)
    stdev_unscaled = fit.stdev_unscaled  # (n_genes, n_coef)
    sigma_sq = fit.sigma**2  # Convert SD to variance!
    df_resid = fit.df_resid_eb  # (n_genes,) for variance shrinkage
    df_resid_test = fit.df_resid_test  # (n_genes, n_coef) for p-values

    # =================================================================
    # Step 3: Call low-level eBayes()
    # =================================================================

    result = eBayes(
        coefficients=coefficients,
        stdev_unscaled=stdev_unscaled,
        sigma_sq=sigma_sq,
        df_resid=df_resid,
        df_resid_test=df_resid_test,
        robust=robust,
        proportion=proportion,
        stdev_coef_lim=stdev_coef_lim,
    )

    # =================================================================
    # Step 4: Create NEW DEFit with results
    # =================================================================

    # Use dataclasses.replace() to create a copy with updated fields
    new_fit = replace(
        fit,
        s2_prior=result.s2_prior,
        df_prior=result.df_prior,
        s2_post=result.s2_post,
        df_total=result.df_total,
        t=result.t,
        p_value=result.p_value,
        z_std=result.z_std,
        lods=result.lods,
        var_prior=result.var_prior,  # (n_coef,) coefficient prior variances for B statistic
        eb_applied=True,
    )

    # =================================================================
    # Step 5: Return new DEFit
    # =================================================================

    return new_fit