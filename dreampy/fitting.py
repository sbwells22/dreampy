"""
Unified Linear Mixed Model Fitting for dreampy.

This module provides the core LMM fitting functionality supporting:
- Single random effect (k=1): ~ condition + (1|donor)
- Multiple crossed random effects (k>=1): ~ condition + (1|donor) + (1|batch)

The implementation follows lme4pureR's plsJSS algorithm for numerical
equivalence with R's lme4::lmer().

Mathematical Background
-----------------------
The linear mixed model is:

    y = Xβ + Zu + ε

where:
    - y is the n×1 response vector
    - X is the n×p fixed effects design matrix
    - β is the p×1 fixed effects coefficient vector
    - Z is the n×q random effects design matrix (block-structured for k>1)
    - u ~ N(0, σ²Λ'Λ) is the q×1 random effects vector
    - ε ~ N(0, σ²W⁻¹) is the n×1 residual vector
    - Λ is the relative covariance factor (block-diagonal for random intercepts)

For k random intercept terms:
    - Z = [Z₁ | Z₂ | ... | Zₖ] where Zᵢ is the indicator matrix for term i
    - Λ = diag(θ₁I, θ₂I, ..., θₖI) where θᵢ = σᵤᵢ/σₑ

References
----------
- Bates et al. (2015). "Fitting Linear Mixed-Effects Models Using lme4"
- lme4pureR: https://github.com/lme4/lme4pureR
- Hoffman & Roussos (2024). "dreamlet" R package
"""
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Literal, TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd
    from anndata import AnnData
    from .de_types import DEFit

import numpy as np
from numpy.linalg import LinAlgError, lstsq
from scipy.linalg import cho_solve, cholesky, solve_triangular
from scipy.optimize import minimize

# Optional dependencies
try:
    import pybobyqa

    PYBOBYQA_AVAILABLE = True
except ImportError:
    PYBOBYQA_AVAILABLE = False

try:
    from joblib import Parallel, delayed

    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

try:
    from threadpoolctl import threadpool_limits

    THREADPOOLCTL_AVAILABLE = True
except ImportError:
    THREADPOOLCTL_AVAILABLE = False

try:
    from tqdm import tqdm

    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

# Kenward-Roger degrees of freedom
try:
    from .kenward_roger import compute_kr_adjustment

    KENWARD_ROGER_AVAILABLE = True
except ImportError:
    try:
        from kenward_roger import compute_kr_adjustment

        KENWARD_ROGER_AVAILABLE = True
    except ImportError:
        KENWARD_ROGER_AVAILABLE = False


# =============================================================================
# Numerical Constants
# =============================================================================
#
# These constants define the numerical regime for theta (variance ratios)
# and related quantities. They form an ordered progression:
#
#   optimizer lower bound (0.0)
#     → _PARAM_FLOOR (1e-10)      : positivity guarantee for optimization
#     → _THETA_EFFECTIVELY_ZERO   : model class boundary (LMM → LM fallback)
#     → _THETA_SNAP_ZERO (1e-6)   : post-optimization snap to exact 0.0
#     → _BOUNDARY_TOL_DEFAULT     : boundary detection for restart logic
#
# The ordering is load-bearing: floor < effectively_zero < snap < boundary_tol.
# Violating it causes optimizer oscillation, RDF branch instability, and
# boundary detection errors.

# Positivity floor for theta and varpar during optimization.
# Prevents log(0) and division-by-zero in profile likelihood evaluation.
_PARAM_FLOOR = 1e-10

# Positivity floor for observation weights.
# Same value as _PARAM_FLOOR but semantically distinct — weights are data,
# not optimization parameters. lme4 assumes positive weights without clamping;
# we clamp defensively.
_WEIGHT_FLOOR = 1e-10

# Ridge regularization added to a matrix when Cholesky factorization fails.
# Also used as the "all theta ≈ 0" early-return threshold in compute_rdf.
_CHOLESKY_REG = 1e-8

# Model class boundary: below this, a RE term contributes negligibly and
# compute_rdf returns classical df (n - p). This is a statistical decision,
# not purely numerical — it determines when an LMM is treated as an LM.
_THETA_EFFECTIVELY_ZERO = 1e-8

# Post-optimization: snap theta values below this to exact 0.0.
# Also used to floor starting theta in restart-edge logic (starting below
# the snap threshold is pointless since the result would be snapped anyway).
_THETA_SNAP_ZERO = 1e-6

# Default tolerance for detecting boundary convergence (theta ≈ 0).
# Used in fit_lmm's restart-edge logic.
_BOUNDARY_TOL_DEFAULT = 1e-5

# Floor for within-group variance in method-of-moments initialization.
# Intentionally much larger than _PARAM_FLOOR — prevents extreme initial
# theta when var_within is near-zero (e.g., near-constant genes), which
# would bias the optimizer starting point.
_INIT_VARIANCE_FLOOR = 1e-6

# Verify ordering invariant. If this fires, someone broke the numerical regime.
assert _PARAM_FLOOR < _THETA_EFFECTIVELY_ZERO < _THETA_SNAP_ZERO < _BOUNDARY_TOL_DEFAULT, (
    "Numerical constants must satisfy: "
    "_PARAM_FLOOR < _THETA_EFFECTIVELY_ZERO < _THETA_SNAP_ZERO < _BOUNDARY_TOL_DEFAULT"
)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class OptimizationDiagnostics:
    """
    Complete optimization diagnostics for debugging convergence.

    This dataclass captures detailed information about the optimization
    process for a single gene's mixed model fit. Minimal fields are always
    populated for every LMM gene fit. Expensive fields are only populated
    when explicitly requested via diagnostics flags.

    Surface evaluation (diagnostics_surface=True) is currently supported
    for single-RE models (k=1) only — it evaluates deviance over a 1D
    grid of theta values. For k>1, surface evaluation is skipped.

    Path tracking (diagnostics_path=True) works for any k — it records
    the sequence of theta vectors visited by the optimizer, which can be
    overlaid on the deviance surface (k=1) or examined directly (k>1).

    Attributes
    ----------
    surface_thetas : np.ndarray, optional
        Grid of theta values where deviance was evaluated.
        Shape: (n_points,). Only populated if diagnostics_surface=True
        and k=1.
    surface_deviances : np.ndarray, optional
        Deviance values at each theta in surface_thetas.
        Only populated if diagnostics_surface=True and k=1.
    surface_grid_min : np.ndarray, optional
        Theta value with lowest deviance on the evaluation grid.
        Only populated if diagnostics_surface=True and k=1.
    theta_path : np.ndarray, optional
        Sequence of theta values visited during optimization.
        Shape: (n_steps, k) for k>1 or (n_steps,) for k=1.
        Only populated if diagnostics_path=True.
    deviance_path : np.ndarray, optional
        Deviance values at each theta in theta_path.
        Only populated if diagnostics_path=True.
    path_truncated : bool
        True if path recording hit MAX_PATH limit and was truncated.
    theta_init_used : np.ndarray
        Initial theta values used for optimization. Always populated.
    theta_optimal : np.ndarray
        Final optimized theta values. Always populated.
    convergence_reason : str, optional
        Message describing convergence status. Always populated.
    n_feval : int, optional
        Total number of function evaluations. Always populated.
    time_elapsed : float, optional
        Time taken for optimization in seconds. Always populated.
    hit_lower_bound : np.ndarray
        Boolean mask: True where theta_optimal < boundary_tol. Always populated.
    restart_edge_triggered : bool
        True if restart_edge recovery logic was invoked. Always populated.
    converged_to_grid_min : bool, optional
        True if theta_optimal is close to surface_grid_min.
        Only meaningful if surface was evaluated (k=1).
    """

    # Surface evaluation (only if diagnostics_surface=True)
    surface_thetas: np.ndarray | None = None
    surface_deviances: np.ndarray | None = None
    surface_grid_min: np.ndarray | None = None

    # Path tracking (only if diagnostics_path=True)
    theta_path: np.ndarray | None = None
    deviance_path: np.ndarray | None = None
    path_truncated: bool = False

    # Always populated for every LMM gene fit
    theta_init_used: np.ndarray | None = None
    theta_optimal: np.ndarray | None = None
    convergence_reason: str | None = None
    n_feval: int | None = None
    time_elapsed: float | None = None

    # Boundary behavior flags — always populated for every LMM gene fit
    hit_lower_bound: np.ndarray | None = None  # Per-component boolean array
    restart_edge_triggered: bool = False

    # Convergence check (only if surface evaluated)
    converged_to_grid_min: bool | None = None


@dataclass
class CholeskyFactors:
    """
    Cholesky factorization products at a given theta.

    These are the expensive intermediate matrices from the normal equations
    solve that multiple downstream consumers need (stdev_unscaled, rdf,
    Satterthwaite, KR). Computing them once in profile_loglik and passing
    them forward eliminates redundant Cholesky decompositions.

    All matrices are at the OPTIMAL theta (post-optimization).

    Attributes
    ----------
    L : np.ndarray (q_total, q_total)
        Lower Cholesky factor of Λ'Z'WZΛ + I
    RZX : np.ndarray (q_total, p)
        L⁻¹(Λ'Z'WX)
    RXtRX : np.ndarray (p, p)
        Downdated cross-product: X'WX - RZX'RZX
    L_RXtRX : np.ndarray (p, p) or None
        Lower Cholesky factor of RXtRX. None if Cholesky failed
        (β was solved via lstsq fallback).
    """

    L: np.ndarray
    RZX: np.ndarray
    RXtRX: np.ndarray
    L_RXtRX: np.ndarray | None = None


@dataclass
class ProfileResult:
    """
    Complete output from profile_loglik at a given theta.

    Bundles the deviance, estimated parameters, and Cholesky factors
    so that downstream consumers (stdev_unscaled, rdf) can reuse the
    expensive factorizations without recomputing them.

    Attributes
    ----------
    deviance : float
        Profiled deviance (-2 * log-likelihood)
    sigma_e_sq : float
        Profiled residual variance (pwrss / degFree)
    beta : np.ndarray (p,)
        Fixed effects estimates
    wrss : float
        Weighted residual sum of squares
    u : np.ndarray (q_total,)
        Spherical random effects (BLUPs)
    b : np.ndarray (q_total,)
        Actual random effects (b = Λu)
    factors : CholeskyFactors
        Cholesky factorization products for reuse
    """

    deviance: float
    sigma_e_sq: float
    beta: np.ndarray
    wrss: float
    u: np.ndarray
    b: np.ndarray
    factors: CholeskyFactors


@dataclass
class LMMResult:
    """
    Results from fitting a weighted linear mixed model.

    Supports FE-only (k=0), single (k=1) and multiple (k>1) random effects.
    For k=0 (from fit_lm), theta/sigma_u/block_sizes/re_names are empty.
    For backward compatibility, scalar accessors are provided when k=1.

    Attributes
    ----------
    theta : np.ndarray
        (k,) Optimized theta values (θᵢ = σᵤᵢ/σₑ)
    sigma_u : np.ndarray
        (k,) Random effect standard deviations
    sigma_e : float
        Residual standard deviation
    beta : np.ndarray
        (p,) Fixed effect coefficients
    u : np.ndarray
        (q_total,) Spherical random effects (BLUPs)
    b : np.ndarray
        (q_total,) Actual random effects (b = Λu)
    fitted_values : np.ndarray
        (n,) Xβ + Zb
    residuals : np.ndarray
        (n,) y - fitted
    converged : bool
        Optimization converged
    n_obs : int
        Number of observations
    n_groups : List[int]
        Number of groups per RE term
    n_fixed : int
        Number of fixed effects
    loglik : float
        Log-likelihood at optimum
    reml_loglik : float, optional
        REML log-likelihood (if REML=True)
    Amean : float
        Mean of response (for voom compatibility)
    n_iter : int
        Optimizer iterations
    message : str
        Convergence message
    hit_lower_bound : np.ndarray
        (k,) Boolean mask for boundary theta
    restart_edge_triggered : bool
        True if restart_edge was invoked
    re_names : List[str]
        Names of random effect terms
    block_sizes : List[int]
        Number of levels per RE term
    diagnostics : OptimizationDiagnostics, optional
        Full diagnostics if requested
    """

    # Variance components (array-valued for k>=1)
    theta: np.ndarray
    sigma_u: np.ndarray
    sigma_e: float

    # Fixed effects
    beta: np.ndarray

    # Random effects
    u: np.ndarray  # Spherical
    b: np.ndarray  # Actual (b = Λu)

    # Fitted values and residuals
    fitted_values: np.ndarray
    residuals: np.ndarray

    # Model information
    converged: bool
    n_obs: int
    n_groups: list[int]
    n_fixed: int

    # Likelihood
    loglik: float
    reml_loglik: float | None = None

    # For voom compatibility
    Amean: float = 0.0

    # Diagnostics (always present)
    n_iter: int = 0
    message: str = ""

    # Boundary flags (array for k>=1)
    hit_lower_bound: np.ndarray = field(default_factory=lambda: np.array([False]))
    restart_edge_triggered: bool = False

    # RE metadata
    re_names: list[str] = field(default_factory=list)
    block_sizes: list[int] = field(default_factory=list)

    # Full diagnostics object (only if requested)
    diagnostics: OptimizationDiagnostics | None = None

    # === Standard errors and degrees of freedom (for eBayes) ===
    stdev_unscaled: np.ndarray = field(default_factory=lambda: np.array([]))
    # (p,) sqrt(diag(inv(X'WX))) for LM, sqrt(diag(inv(RXtRX))) for LMM
    # "Unscaled" means NOT divided by sigma. Actual SE = sigma * stdev_unscaled

    df_resid: float = 0.0
    # Residual degrees of freedom (n - p for Phase 1 minimal implementation)

    df_satterthwaite: np.ndarray | None = None
    # (p,) Satterthwaite degrees of freedom per coefficient (if computed)

    # === Scalar accessors for k=1 backward compatibility ===

    @property
    def sigma_u_scalar(self) -> float:
        """Get sigma_u as scalar (for k=1 backward compat)."""
        if len(self.sigma_u) == 1:
            return float(self.sigma_u[0])
        raise ValueError("sigma_u_scalar only valid for k=1")

    @property
    def theta_scalar(self) -> float:
        """Get theta as scalar (for k=1 backward compat)."""
        if len(self.theta) == 1:
            return float(self.theta[0])
        raise ValueError("theta_scalar only valid for k=1")

    @property
    def n_groups_scalar(self) -> int:
        """Get n_groups as scalar (for k=1 backward compat)."""
        if len(self.n_groups) == 1:
            return self.n_groups[0]
        raise ValueError("n_groups_scalar only valid for k=1")

    @property
    def hit_lower_bound_any(self) -> bool:
        """True if ANY theta hit lower bound."""
        return bool(np.any(self.hit_lower_bound))

    @property
    def k(self) -> int:
        """Number of random effect terms."""
        return len(self.theta)

    @property
    def q_total(self) -> int:
        """Total random effect levels across all terms."""
        return sum(self.n_groups)

    @property
    def deviance(self) -> float:
        """Deviance = -2 * loglik."""
        return -2 * self.loglik


# REStructure and create_re_structure deleted in v16 (Item #6).
# Z and block_sizes are now built once per assay by
# design.build_model_matrices() and passed through.


# =============================================================================
# Precomputation
# =============================================================================


def precompute_weighted_products(
    y: np.ndarray,
    X: np.ndarray,
    Z: np.ndarray,
    weights: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Pre-compute weighted cross-products that don't depend on theta.

    This EXACTLY matches lme4pureR's initialization step:

    R code (plsJSS):
        sqrtW <- Diagonal(x=sqrt(weights))
        WX <- sqrtW %*% X
        Wy <- sqrtW %*% y
        ZtW <- Zt %*% sqrtW
        XtWX <- crossprod(WX)
        XtWy <- crossprod(WX, Wy)
        ZtWX <- ZtW %*% WX
        ZtWy <- ZtW %*% Wy

    Parameters
    ----------
    y : np.ndarray (n,)
        Response vector
    X : np.ndarray (n, p)
        Fixed effects design matrix
    Z : np.ndarray (n, q_total)
        Combined random effects design matrix
    weights : np.ndarray (n,)
        Prior weights (voom precision weights)

    Returns
    -------
    dict with pre-computed products
    """
    sqrtW = np.sqrt(weights)

    # W^{1/2} X and W^{1/2} y
    WX = sqrtW[:, np.newaxis] * X  # (n, p)
    Wy = sqrtW * y  # (n,)

    # Z' W^{1/2}
    ZtW_sqrt = Z.T * sqrtW[np.newaxis, :]  # (q_total, n)

    # Cross-products
    XtWX = WX.T @ WX  # = X' W X, (p, p)
    XtWy = WX.T @ Wy  # = X' W y, (p,)
    ZtWX = ZtW_sqrt @ WX  # = Z' W X, (q_total, p)
    ZtWy = ZtW_sqrt @ Wy  # = Z' W y, (q_total,)
    ZtWZ = ZtW_sqrt @ ZtW_sqrt.T  # = Z' W Z, (q_total, q_total)

    # Weight normalization term for deviance
    sum_log_weights = np.sum(np.log(weights))

    # Products needed by compute_rdf (hat matrix factors CL, CR)
    # ZtSqrtW = Z' @ diag(√W) — used for CL = L⁻¹ Λ' Z' √W
    # XtSqrtW = X' @ diag(√W) — used for CR = RX⁻ᵀ (X'√W - RZX' CL)
    # Note: ZtSqrtW is identical to ZtW_sqrt computed above
    XtSqrtW = X.T * sqrtW[np.newaxis, :]  # (p, n)

    return {
        "XtWX": XtWX,
        "XtWy": XtWy,
        "ZtWX": ZtWX,
        "ZtWy": ZtWy,
        "ZtWZ": ZtWZ,
        "sqrtW": sqrtW,
        "sum_log_weights": sum_log_weights,
        "ZtSqrtW": ZtW_sqrt,  # (q_total, n) — reuse, no recomputation
        "XtSqrtW": XtSqrtW,  # (p, n)
    }


# =============================================================================
# Core LMM Factorization and Solve
# =============================================================================


def _lambda_scale_rows(
    source: np.ndarray,
    theta: np.ndarray,
    block_sizes: list[int],
) -> np.ndarray:
    """
    Apply block-diagonal Λ scaling to rows of source.

    For each RE term i, rows in block i are scaled by θᵢ:
        result[block_i] = θᵢ * source[block_i]

    This is the single-loop Λ operation used for:
    - Λ'Z'WX  (2D matrix)
    - Λ'Z'Wy  (1D vector)
    - Λu → b  (1D vector)
    - Λ'Z'√W  (2D matrix)

    Parameters
    ----------
    source : np.ndarray
        (q_total,) vector or (q_total, m) matrix to scale.
    theta : np.ndarray (k,)
        Variance ratios, one per RE term.
    block_sizes : List[int]
        Number of levels per RE term.

    Returns
    -------
    np.ndarray
        Same shape as source, with rows scaled by their block's theta.
    """
    result = np.zeros_like(source)
    col = 0
    for theta_k, q_k in zip(theta, block_sizes):
        result[col : col + q_k] = theta_k * source[col : col + q_k]
        col += q_k
    return result


def _compute_cholesky_factors(
    theta: np.ndarray,
    block_sizes: list[int],
    precomputed: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Cholesky factors at given theta.

    This is Steps I-III of the lme4pureR algorithm: Λ-scaling of Z'WZ
    and Z'WX, Cholesky factorization of Λ'Z'WZΛ + I, and downdating
    of X'WX.

    Used directly by _compute_var_beta_at_varpar for perturbed-point
    evaluations, and internally by _lmm_core_solve for the full solve.

    Parameters
    ----------
    theta : np.ndarray (k,)
        Variance ratios θᵢ = σᵤᵢ/σₑ. Clamped to ≥_PARAM_FLOOR internally.
    block_sizes : List[int]
        Number of levels per RE term.
    precomputed : dict
        Weighted products (must contain XtWX, ZtWX, ZtWZ).

    Returns
    -------
    L : np.ndarray (q, q)
        Cholesky factor of Λ'Z'WZΛ + I
    RZX : np.ndarray (q, p)
        L⁻¹(Λ'Z'WX)
    RXtRX : np.ndarray (p, p)
        Downdated cross-product X'WX - RZX'RZX
    """
    q_total = sum(block_sizes)

    theta = np.maximum(theta, _PARAM_FLOOR)

    XtWX = precomputed["XtWX"]
    ZtWX = precomputed["ZtWX"]
    ZtWZ = precomputed["ZtWZ"]

    # Step I: Build Λ'Z'WZΛ + I
    LtZtWZL = np.zeros_like(ZtWZ)
    col_i = 0
    for _i, (theta_i, q_i) in enumerate(zip(theta, block_sizes)):
        col_j = 0
        for _j, (theta_j, q_j) in enumerate(zip(theta, block_sizes)):
            block = ZtWZ[col_i : col_i + q_i, col_j : col_j + q_j]
            LtZtWZL[col_i : col_i + q_i, col_j : col_j + q_j] = theta_i * theta_j * block
            col_j += q_j
        col_i += q_i

    LtZtWZL_plus_I = LtZtWZL + np.eye(q_total)

    try:
        L = cholesky(LtZtWZL_plus_I, lower=True)
    except LinAlgError:
        warnings.warn(
            "Unexpected loss of positive-definiteness in RE precision matrix "
            "(Λ'Z'WZΛ + I); adding ridge regularization. This may indicate "
            "numerical issues with the input data.",
            stacklevel=2,
        )
        LtZtWZL_plus_I += np.eye(q_total) * _CHOLESKY_REG
        L = cholesky(LtZtWZL_plus_I, lower=True)

    # Step II: Λ'Z'WX and RZX = L⁻¹(Λ'Z'WX)
    LambdatZtWX = _lambda_scale_rows(ZtWX, theta, block_sizes)

    RZX = solve_triangular(L, LambdatZtWX, lower=True)

    # Step III: RXtRX = X'WX - RZX'RZX
    RXtRX = XtWX - RZX.T @ RZX

    return L, RZX, RXtRX


@dataclass
class _CoreSolveResult:
    """
    Internal result from _lmm_core_solve.

    Contains all intermediate quantities needed by both profile_loglik
    (profiled deviance) and _deviance_at_varpar (unprofiled deviance).
    """

    factors: CholeskyFactors
    beta: np.ndarray
    u: np.ndarray
    b: np.ndarray
    wrss: float
    pwrss: float
    log_det_L: float
    log_det_RXtRX: float


def _lmm_core_solve(
    theta: np.ndarray,
    y: np.ndarray,
    X: np.ndarray,
    Z: np.ndarray,
    block_sizes: list[int],
    precomputed: dict,
    cholesky_tuple: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> _CoreSolveResult:
    """
    Core LMM solve at given theta: factorize, solve normal equations,
    compute random effects, residuals, and log determinants.

    This is the single implementation of the lme4pureR plsJSS algorithm
    (Steps I-VI minus the deviance formula). Called by:
    - profile_loglik: applies profiled deviance formula (σ² = pwrss/df)
    - _deviance_at_varpar: applies unprofiled deviance formula (σ² provided)

    Parameters
    ----------
    theta : np.ndarray (k,)
        Variance ratios. Clamped to ≥_PARAM_FLOOR in _compute_cholesky_factors.
    y : np.ndarray (n,)
        Response vector.
    X : np.ndarray (n, p)
        Fixed effects design matrix.
    Z : np.ndarray (n, q_total)
        Random effects design matrix.
    block_sizes : List[int]
        Number of levels per RE term.
    precomputed : dict
        Weighted products from precompute_weighted_products().
    cholesky_tuple : (L, RZX, RXtRX), optional
        Pre-computed Cholesky factors. When provided, skips
        _compute_cholesky_factors. Use for the optimal theta where
        factors are already available from a previous call.

    Returns
    -------
    _CoreSolveResult
        All quantities needed for deviance computation and downstream use.
    """
    _n = len(y)
    _p = X.shape[1]
    _q_total = Z.shape[1]

    theta = np.atleast_1d(theta).astype(np.float64)

    XtWy = precomputed["XtWy"]
    ZtWy = precomputed["ZtWy"]
    sqrtW = precomputed["sqrtW"]

    # Steps I-III: Cholesky factorization (or reuse)
    if cholesky_tuple is not None:
        L, RZX, RXtRX = cholesky_tuple
    else:
        L, RZX, RXtRX = _compute_cholesky_factors(theta, block_sizes, precomputed)

    # Compute Λ'Z'Wy
    LambdatZtWy = _lambda_scale_rows(ZtWy, theta, block_sizes)

    # cu = L⁻¹(Λ'Z'Wy)
    cu = solve_triangular(L, LambdatZtWy, lower=True)

    # RXtRy = X'Wy - RZX'cu
    RXtRy = XtWy - RZX.T @ cu

    # Solve for β via Cholesky of RXtRX
    L_RXtRX = None
    try:
        L_RXtRX = cholesky(RXtRX, lower=True)
        beta = cho_solve((L_RXtRX, True), RXtRy)
    except LinAlgError:
        beta, _, _, _ = lstsq(RXtRX, RXtRy, rcond=None)

    # Step IV: Spherical random effects u = L⁻ᵀ(cu - RZX @ β)
    u = solve_triangular(L.T, cu - RZX @ beta, lower=False)

    # Step V: Actual random effects b = Λu, residuals, WRSS
    b = _lambda_scale_rows(u, theta, block_sizes)

    fitted = X @ beta + Z @ b
    resid = y - fitted
    resid_weighted = sqrtW * resid
    wrss = np.dot(resid_weighted, resid_weighted)
    pwrss = wrss + np.dot(u, u)

    # Step VI: Log determinants
    log_det_L = np.sum(np.log(np.diag(L)))

    # log|RXtRX| (needed for REML; computed unconditionally since it's cheap)
    try:
        if L_RXtRX is None:
            L_RXtRX = cholesky(RXtRX, lower=True)
        log_det_RXtRX = 2 * np.sum(np.log(np.diag(L_RXtRX)))
    except LinAlgError:
        log_det_RXtRX = np.log(max(np.linalg.det(RXtRX), 1e-300))

    return _CoreSolveResult(
        factors=CholeskyFactors(L=L, RZX=RZX, RXtRX=RXtRX, L_RXtRX=L_RXtRX),
        beta=beta,
        u=u,
        b=b,
        wrss=wrss,
        pwrss=pwrss,
        log_det_L=log_det_L,
        log_det_RXtRX=log_det_RXtRX,
    )


# =============================================================================
# Profile Likelihood (Generalized for k>=1)
# =============================================================================


def profile_loglik(
    theta: np.ndarray,
    y: np.ndarray,
    X: np.ndarray,
    Z: np.ndarray,
    weights: np.ndarray,
    block_sizes: list[int],
    reml: bool = False,
    precomputed: dict | None = None,
) -> ProfileResult:
    """
    Compute profile deviance for k>=1 random intercept terms.

    This generalizes the lme4pureR plsJSS algorithm to handle:
    - k=1: Single random effect (backward compatible with weighted_lmm.py)
    - k>1: Multiple crossed random intercepts

    Parameters
    ----------
    theta : np.ndarray (k,)
        Vector of theta values, one per RE term (θᵢ = σᵤᵢ/σₑ)
    y : np.ndarray (n,)
        Response vector
    X : np.ndarray (n, p)
        Fixed effects design matrix
    Z : np.ndarray (n, q_total)
        Combined random effects design matrix
    weights : np.ndarray (n,)
        Prior weights
    block_sizes : List[int]
        Number of groups per RE term
    reml : bool
        Use REML (True) or ML (False)
    precomputed : dict, optional
        Pre-computed weighted products

    Returns
    -------
    ProfileResult
        Bundled results: deviance, sigma_e_sq, beta, wrss, u, b, and
        CholeskyFactors (L, RZX, RXtRX, L_RXtRX) for downstream reuse.
    """
    n = len(y)
    p = X.shape[1]
    k = len(block_sizes)

    # Ensure theta is array
    theta = np.atleast_1d(theta).astype(np.float64)

    if len(theta) != k:
        raise ValueError(f"theta length ({len(theta)}) must match number of RE terms ({k})")

    # Pre-compute if not provided
    if precomputed is None:
        precomputed = precompute_weighted_products(y, X, Z, weights)

    # Core solve: factorize, solve normal equations, compute residuals
    core = _lmm_core_solve(theta, y, X, Z, block_sizes, precomputed)

    # Degrees of freedom
    degFree = (n - p) if reml else n

    # Profile out σ²
    sigma_e_sq = core.pwrss / degFree

    # Profiled deviance formula
    if reml:
        logDet = 2 * core.log_det_L + core.log_det_RXtRX
        base_deviance = logDet + degFree * (
            1.0 + np.log(2.0 * np.pi * core.pwrss) - np.log(degFree)
        )
    else:
        base_deviance = 2 * core.log_det_L + degFree * (
            1.0 + np.log(2.0 * np.pi * core.pwrss) - np.log(degFree)
        )

    # Weight correction for R compatibility
    weight_correction = -precomputed.get("sum_log_weights", 0.0)
    deviance = base_deviance + weight_correction

    return ProfileResult(
        deviance=deviance,
        sigma_e_sq=sigma_e_sq,
        beta=core.beta,
        wrss=core.wrss,
        u=core.u,
        b=core.b,
        factors=core.factors,
    )


def compute_stdev_unscaled_lmm(
    factors: CholeskyFactors,
    p: int,
) -> np.ndarray:
    """
    Compute stdev_unscaled for LMM from pre-computed Cholesky factors.

    For LMMs, the covariance of beta is:
        Cov(β) = σ² * (RXtRX)^{-1}
    where RXtRX = X'WX - RZX'RZX is the downdated cross-product.

    Returns:
        stdev_unscaled = sqrt(diag((RXtRX)^{-1}))

    Parameters
    ----------
    factors : CholeskyFactors
        Pre-computed Cholesky factors from profile_loglik
    p : int
        Number of fixed effects

    Returns
    -------
    stdev_unscaled : np.ndarray (p,)
        sqrt(diag(inv(RXtRX))), or NaN if computation fails
    """
    # If profile_loglik already computed L_RXtRX, reuse it
    if factors.L_RXtRX is not None:
        try:
            L_RXtRX_inv = solve_triangular(factors.L_RXtRX, np.eye(p), lower=True)
            return np.sqrt(np.sum(L_RXtRX_inv**2, axis=0))
        except LinAlgError:
            pass  # Fall through to recomputation

    # L_RXtRX was None (β solved via lstsq) — try Cholesky on RXtRX
    try:
        L_RXtRX = cholesky(factors.RXtRX, lower=True)
        L_RXtRX_inv = solve_triangular(L_RXtRX, np.eye(p), lower=True)
        return np.sqrt(np.sum(L_RXtRX_inv**2, axis=0))
    except LinAlgError:
        pass

    # Fallback: pseudoinverse
    try:
        RXtRX_inv = np.linalg.pinv(factors.RXtRX)
        return np.sqrt(np.maximum(np.diag(RXtRX_inv), 0.0))
    except Exception:
        return np.full(p, np.nan)


def rdf_from_matrices(A: np.ndarray, B: np.ndarray) -> float:
    """
    Compute residual degrees of freedom from hat matrix factors.

    Given H = A'A + B'B where A and B are low rank, compute:
        rdf = n - 2*tr(H) + tr(H²)

    in O(qn + pn) instead of O(n²) by using eigendecomposition.

    This matches variancePartition::rdf_from_matrices() exactly.

    Parameters
    ----------
    A : np.ndarray (q, n)
        First factor matrix (CL from LMM)
    B : np.ndarray (p, n)
        Second factor matrix (CR from LMM)

    Returns
    -------
    rdf : float
        Residual degrees of freedom: n - 2*tr(H) + tr(H²)

    Notes
    -----
    Mathematical derivation:

    H = A'A + B'B  (n × n hat matrix)

    tr(H) = tr(A'A) + tr(B'B) = ||A||²_F + ||B||²_F

    tr(H²) = tr((A'A + B'B)²)
           = tr(A'A A'A) + 2*tr(A'A B'B) + tr(B'B B'B)

    Using cyclic property of trace:
    - tr(A'A A'A) = tr(AA' AA') = ||AA'||²_F = Σ(λ_i)² where λ_i are eigenvalues of AA'
    - tr(B'B B'B) = Σ(μ_j)² where μ_j are eigenvalues of BB'
    - tr(A'A B'B) = tr(BA' AB') = ||BA'||²_F

    Since eigenvalues of AA' equal squared singular values of A:
    - Σ(λ_i)² = Σ(σ_i⁴) where σ_i are singular values of A

    Algorithm (from R):
    1. Compute eigendecomposition of AA' → eigenvalues λ_i, eigenvectors U
    2. s_a² = λ_i (eigenvalues = squared singular values)
    3. Compute eigenvalues of BB' → μ_j
    4. s_b² = μ_j
    5. G = BA' @ U (orthogonal transformation preserves Frobenius norm)
    6. ||BA'||²_F = ||G||²_F = sum(G²)
    7. tr(H²) = sum(s_a⁴) + 2*sum(G²) + sum(s_b⁴)

    Reference: variancePartition R/rdf_functions.R::rdf_from_matrices
    """
    # Get dimensions
    q, n = A.shape
    p, n_check = B.shape

    if n != n_check:
        raise ValueError(f"A and B must have same number of columns: {n} != {n_check}")

    # Handle edge case: empty matrices
    if q == 0 and p == 0:
        return float(n)

    # Tolerance for numerical stability
    tol = 1e-12

    # =========================================================================
    # Eigendecomposition of AA' (q × q)
    # =========================================================================
    if q > 0:
        AAt = A @ A.T
        try:
            eigvals_A, U = np.linalg.eigh(AAt)
        except np.linalg.LinAlgError:
            # Fallback to SVD if eigendecomposition fails
            U, s, _ = np.linalg.svd(A, full_matrices=False)
            eigvals_A = s**2

        # Floor negative eigenvalues (numerical noise)
        eigvals_A = np.maximum(eigvals_A, 0.0)

        # Filter out tiny eigenvalues for stability
        valid_A = eigvals_A > tol
        eigvals_A_valid = eigvals_A[valid_A]
        U_valid = U[:, valid_A]

        # s_a² = eigenvalues of AA' (= squared singular values of A)
        s_a_sq = eigvals_A_valid
    else:
        s_a_sq = np.array([])
        U_valid = np.zeros((0, 0))

    # =========================================================================
    # Eigendecomposition of BB' (p × p) - values only
    # =========================================================================
    if p > 0:
        BBt = B @ B.T
        try:
            eigvals_B = np.linalg.eigvalsh(BBt)
        except np.linalg.LinAlgError:
            # Fallback to SVD
            s = np.linalg.svd(B, compute_uv=False)
            eigvals_B = s**2

        # Floor negative eigenvalues
        eigvals_B = np.maximum(eigvals_B, 0.0)

        # s_b² = eigenvalues of BB'
        s_b_sq = eigvals_B
    else:
        s_b_sq = np.array([])

    # =========================================================================
    # Cross-term: ||BA'||²_F = ||BA' @ U||²_F (U is orthogonal)
    # =========================================================================
    if q > 0 and p > 0 and U_valid.shape[1] > 0:
        # G = B @ A' @ U_valid
        G = B @ A.T @ U_valid
        cross_term = np.sum(G**2)
    else:
        cross_term = 0.0

    # =========================================================================
    # Compute rdf = n - 2*tr(H) + tr(H²)
    # =========================================================================

    # tr(H) = ||A||²_F + ||B||²_F = sum(s_a²) + sum(s_b²)
    tr_H = np.sum(s_a_sq) + np.sum(s_b_sq)

    # tr(H²) = sum(s_a⁴) + 2*||BA'||²_F + sum(s_b⁴)
    tr_H_sq = np.sum(s_a_sq**2) + 2.0 * cross_term + np.sum(s_b_sq**2)

    rdf = n - 2.0 * tr_H + tr_H_sq

    return float(rdf)


def compute_rdf(
    theta: np.ndarray,
    factors: CholeskyFactors,
    precomputed: dict,
    block_sizes: list[int],
    n: int,
    p: int,
) -> float:
    """
    Compute trace-based residual degrees of freedom for eBayes shrinkage.

    This computes the approximate residual df for a linear mixed model,
    matching variancePartition::rdf.merMod().

    For a linear model, rdf = n - p exactly.
    For a mixed model, rdf accounts for df "consumed" by random effects.

    Parameters
    ----------
    theta : np.ndarray (k,)
        Variance component ratios (σ_u / σ_e) at optimal values
    factors : CholeskyFactors
        Pre-computed L, RZX, RXtRX, L_RXtRX from profile_loglik
    precomputed : Dict
        Precomputed products (must include ZtSqrtW, XtSqrtW)
    block_sizes : List[int]
        Number of levels per random effect term
    n : int
        Number of observations
    p : int
        Number of fixed effects

    Returns
    -------
    rdf : float
        Residual degrees of freedom, capped at n - p (classical df)

    Notes
    -----
    The hat matrix for a weighted LMM is H = CL'CL + CR'CR where:

        CL = L⁻¹ @ Λ' @ Z' @ diag(√W)
        CR = RX⁻ᵀ @ (X' @ diag(√W) - RZX' @ CL)

    where L, RZX come from CholeskyFactors, and ZtSqrtW, XtSqrtW
    from precomputed.

    The residual df is: rdf = n - 2*tr(H) + tr(H²)
    This is computed efficiently using rdf_from_matrices(CL, CR).

    Reference: variancePartition R/rdf_functions.R::rdf.merMod
    """
    k = len(block_sizes)
    q_total = sum(block_sizes)

    # Classical df as upper bound
    classical_df = float(n - p)

    # Handle edge cases
    if k == 0 or q_total == 0:
        return max(classical_df, 0.0)

    theta = np.atleast_1d(theta).astype(np.float64)

    if len(theta) != k:
        raise ValueError(f"theta length ({len(theta)}) must match number of RE terms ({k})")

    # Floor theta for numerical stability
    theta = np.maximum(theta, _PARAM_FLOOR)

    # If all theta effectively zero, this is not a mixed model — return classical df
    if np.all(theta < _THETA_EFFECTIVELY_ZERO):
        return max(classical_df, 0.0)

    # =========================================================================
    # Extract precomputed ZtSqrtW and XtSqrtW
    # =========================================================================
    ZtSqrtW = precomputed["ZtSqrtW"]  # (q_total, n)
    XtSqrtW = precomputed["XtSqrtW"]  # (p, n)

    # =========================================================================
    # Compute Λ'Z'√W using block scaling (theta-dependent, can't cache)
    # =========================================================================
    LambdatZtSqrtW = _lambda_scale_rows(ZtSqrtW, theta, block_sizes)

    # =========================================================================
    # CL = L⁻¹ @ Λ' @ Z' @ diag(√W)  — reuse L from factors
    # =========================================================================
    try:
        CL = solve_triangular(factors.L, LambdatZtSqrtW, lower=True)  # (q_total, n)
    except LinAlgError:
        return max(classical_df, 0.0)

    # =========================================================================
    # CR = L_RXtRX⁻¹ @ (XtSqrtW - RZX' @ CL)  — reuse L_RXtRX, RZX
    # =========================================================================
    # Need L_RXtRX. Reuse from factors if available, else compute.
    L_RXtRX = factors.L_RXtRX
    if L_RXtRX is None:
        try:
            L_RXtRX = cholesky(factors.RXtRX, lower=True)
        except LinAlgError:
            # Downdating can produce near-singular RXtRX due to
            # floating-point cancellation — expected numerical behavior
            try:
                warnings.warn(
                    "Downdated cross-product (RXtRX) required ridge stabilization "
                    "for Cholesky factorization in residual df computation.",
                    stacklevel=2,
                )
                L_RXtRX = cholesky(factors.RXtRX + np.eye(p) * _CHOLESKY_REG, lower=True)
            except LinAlgError:
                return max(classical_df, 0.0)

    temp = XtSqrtW - factors.RZX.T @ CL  # (p, n)

    try:
        CR = solve_triangular(L_RXtRX, temp, lower=True)  # (p, n)
    except LinAlgError:
        return max(classical_df, 0.0)

    # =========================================================================
    # Compute rdf using efficient trace formula
    # =========================================================================
    rdf = rdf_from_matrices(CL, CR)

    # Cap at classical df (should never exceed, but numerical stability)
    rdf = min(rdf, classical_df)

    # Floor at 0 (shouldn't happen, but safety)
    rdf = max(rdf, 0.0)

    return rdf


# =============================================================================
# Starting Value Computation
# =============================================================================


def compute_starting_theta(
    y: np.ndarray,
    Z: np.ndarray,
    block_sizes: list[int],
    X: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute initial theta values using method of moments.

    Parameters
    ----------
    y : np.ndarray (n,)
        Response vector
    Z : np.ndarray (n, q_total)
        Random effects indicator matrix
    block_sizes : List[int]
        Number of groups per RE term
    X : np.ndarray, optional
        Fixed effects design matrix (for residual computation)

    Returns
    -------
    theta_init : np.ndarray (k,)
        Initial theta values, one per RE term
    """
    n = len(y)
    k = len(block_sizes)

    # Compute residuals from fixed effects if X provided
    if X is not None:
        try:
            beta_ols, _, _, _ = lstsq(X, y, rcond=None)
            resid = y - X @ beta_ols
        except (LinAlgError, ValueError):
            resid = y - y.mean()
    else:
        resid = y - y.mean()

    theta_init = np.zeros(k)
    ones_n = np.ones(n)

    col = 0
    for i in range(k):
        q_i = block_sizes[i]
        Z_block = Z[:, col : col + q_i]
        col += q_i

        n_groups = q_i

        if n_groups <= 1:
            theta_init[i] = 0.1
            continue

        # Group sums and sizes via Z columns
        group_sums = Z_block.T @ resid  # (q_i,)
        group_sizes = Z_block.T @ ones_n  # (q_i,)

        # Between-group variance
        group_means = group_sums / np.maximum(group_sizes, 1)
        overall_mean = resid.mean()

        var_between = np.sum(group_sizes * (group_means - overall_mean) ** 2) / (n_groups - 1)

        # Within-group variance via residual from group means
        fitted_group = Z_block @ group_means
        var_within = np.sum((resid - fitted_group) ** 2) / max(n - n_groups, 1)

        # Estimate variance component
        avg_group_size = n / n_groups
        sigma_u_sq = max(0, (var_between - var_within) / avg_group_size)
        sigma_e_sq = max(var_within, _INIT_VARIANCE_FLOOR)

        # theta = sigma_u / sigma_e
        theta_init[i] = np.sqrt(sigma_u_sq / sigma_e_sq) if sigma_e_sq > 0 else 0.5
        theta_init[i] = np.clip(theta_init[i], 0.01, 10.0)

    return theta_init


# =============================================================================
# Deviance Surface Evaluation (for diagnostics)
# =============================================================================


def evaluate_deviance_surface_1d(
    y: np.ndarray,
    X: np.ndarray,
    Z: np.ndarray,
    weights: np.ndarray,
    block_sizes: list[int],
    precomputed: dict,
    theta_range: tuple[float, float] = (0.0, 5.0),
    n_points: int = 100,
    reml: bool = False,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Evaluate deviance over a 1D grid of theta values (for k=1).

    Returns
    -------
    thetas : np.ndarray (n_points,)
    deviances : np.ndarray (n_points,)
    grid_min : float - theta at minimum deviance
    """
    thetas = np.linspace(theta_range[0], theta_range[1], n_points)

    deviances = np.array(
        [
            profile_loglik(
                np.array([theta]), y, X, Z, weights, block_sizes, reml=reml, precomputed=precomputed
            ).deviance
            for theta in thetas
        ]
    )

    grid_min = thetas[np.argmin(deviances)]

    return thetas, deviances, grid_min


# =============================================================================
# Satterthwaite Degrees of Freedom
# =============================================================================
# Consolidated from satterthwaite.py (Phase 2.4). These functions implement the
# lmerTest Satterthwaite approximation for denominator df in mixed models:
#   1. Parametrize on varpar = (θ₁,...,θₖ, σ)
#   2. Compute Hessian of deviance w.r.t. varpar via central finite differences
#   3. Invert: Cov(varpar) = 2 × H⁻¹
#   4. Delta method: Var(Var(β̂ⱼ)) = ∇ᵥₐᵣₚₐᵣ[Var(β̂ⱼ)]' × Cov(varpar) × ∇
#   5. Satterthwaite: df_j = 2 × Var(β̂ⱼ)² / Var(Var(β̂ⱼ))


def _deviance_at_varpar(
    varpar: np.ndarray,
    y: np.ndarray,
    X: np.ndarray,
    Z: np.ndarray,
    weights: np.ndarray,
    block_sizes: list[int],
    reml: bool,
    precomputed: dict,
    cholesky_tuple: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> float:
    """
    Compute deviance at given varpar = (theta, sigma) WITHOUT profiling sigma.

    This is the key difference from profile_loglik: we use the provided sigma
    instead of computing σ² = pwrss / df. Matches lmerTest's devfun_vp().

    Shares the core LMM solve with profile_loglik via _lmm_core_solve.

    Note: The weight correction term (-sum_log_weights) is intentionally omitted.
    It is a constant that does not depend on varpar, so it cancels in the
    Hessian (second derivatives) and gradient (first differences) and has no
    effect on Satterthwaite df. Omitting it means f0 disagrees with
    profile_loglik's deviance by this constant — harmless for computation.

    Parameters
    ----------
    cholesky_tuple : optional (L, RZX, RXtRX)
        Pre-computed Cholesky factors at this varpar. When provided,
        skips _compute_cholesky_factors. Use for the optimal varpar
        where factors are already available from profile_loglik.
    """
    n = len(y)
    p = X.shape[1]

    # Extract theta and sigma from varpar
    theta = varpar[:-1]
    sigma = varpar[-1]
    sigma_sq = sigma**2

    # Ensure positive
    theta = np.maximum(theta, _PARAM_FLOOR)
    sigma_sq = max(sigma_sq, _PARAM_FLOOR**2)

    # Core solve (shared with profile_loglik)
    core = _lmm_core_solve(theta, y, X, Z, block_sizes, precomputed, cholesky_tuple=cholesky_tuple)

    # Degrees of freedom
    degFree = (n - p) if reml else n

    # Deviance WITHOUT profiling sigma (uses provided σ² instead of pwrss/df)
    if reml:
        deviance = (
            2 * core.log_det_L
            + core.log_det_RXtRX
            + degFree * np.log(2 * np.pi * sigma_sq)
            + core.pwrss / sigma_sq
        )
    else:
        deviance = (
            2 * core.log_det_L + degFree * np.log(2 * np.pi * sigma_sq) + core.pwrss / sigma_sq
        )

    return deviance


def _compute_var_beta_at_varpar(
    varpar: np.ndarray,
    coef_idx: int,
    X: np.ndarray,
    Z: np.ndarray,
    weights: np.ndarray,
    block_sizes: list[int],
    precomputed: dict,
    cholesky_tuple: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> float:
    """
    Compute Var(β̂_j) at given varpar = (theta, sigma).

    Var(β̂_j) = σ² × (RXtRX⁻¹)_jj

    Parameters
    ----------
    cholesky_tuple : optional (L, RZX, RXtRX)
        Pre-computed Cholesky factors at this varpar. When provided,
        skips _compute_cholesky_factors. Use for the optimal varpar
        where factors are already available from profile_loglik.
    """
    p = X.shape[1]

    # Extract theta and sigma
    theta = varpar[:-1]
    sigma = varpar[-1]
    sigma_sq = sigma**2

    # Get Cholesky factors — reuse if provided
    if cholesky_tuple is not None:
        _L, _RZX, RXtRX = cholesky_tuple
    else:
        _L, _RZX, RXtRX = _compute_cholesky_factors(theta, block_sizes, precomputed)

    # Compute (RXtRX⁻¹)_jj via Cholesky
    try:
        L_RXtRX = cholesky(RXtRX, lower=True)
        # To get j-th diagonal of inverse: solve L_RXtRX @ x = e_j, then ||x||²
        e_j = np.zeros(p)
        e_j[coef_idx] = 1.0
        x = solve_triangular(L_RXtRX, e_j, lower=True)
        var_unscaled = np.sum(x**2)
    except LinAlgError:
        # Fallback to pseudoinverse
        try:
            RXtRX_inv = np.linalg.pinv(RXtRX)
            var_unscaled = RXtRX_inv[coef_idx, coef_idx]
        except (LinAlgError, np.linalg.LinAlgError):  # Issue #46: was bare except
            return np.nan

    return sigma_sq * var_unscaled


def _compute_varpar_vcov(
    varpar: np.ndarray,
    y: np.ndarray,
    X: np.ndarray,
    Z: np.ndarray,
    weights: np.ndarray,
    block_sizes: list[int],
    reml: bool,
    precomputed: dict,
    eps: float = 1e-4,
    cholesky_tuple_opt: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """
    Compute variance-covariance matrix of varpar via finite-difference Hessian.

    Matches lmerTest's approach:
        h <- numDeriv::hessian(func=devfun_vp, x=varpar_opt, ...)
        vcov_varpar <- 2 * h_inv

    Parameters
    ----------
    cholesky_tuple_opt : optional (L, RZX, RXtRX)
        Pre-computed Cholesky factors at the optimal varpar. Used only for
        the f0 (center point) deviance evaluation. Perturbed-point
        evaluations compute their own factors since theta/sigma differ.
    """
    m = len(varpar)  # k+1

    def deviance(vp, cholesky_tuple=None):
        vp = np.maximum(vp, _PARAM_FLOOR)
        return _deviance_at_varpar(
            vp, y, X, Z, weights, block_sizes, reml, precomputed, cholesky_tuple=cholesky_tuple
        )

    # Compute Hessian via central differences
    # f0 at the optimal point reuses pre-computed Cholesky factors
    hessian = np.zeros((m, m))
    f0 = deviance(varpar, cholesky_tuple=cholesky_tuple_opt)

    for i in range(m):
        for j in range(i, m):
            ei = np.zeros(m)
            ej = np.zeros(m)

            # Adaptive step size
            hi = eps * max(abs(varpar[i]), 1.0)
            hj = eps * max(abs(varpar[j]), 1.0)

            ei[i] = hi
            ej[j] = hj

            if i == j:
                # Diagonal: second derivative
                vp_plus = varpar + ei
                vp_minus = varpar - ei
                # Ensure positive
                vp_minus = np.maximum(vp_minus, _PARAM_FLOOR)

                f_plus = deviance(vp_plus)
                f_minus = deviance(vp_minus)
                hessian[i, i] = (f_plus - 2 * f0 + f_minus) / (hi * hi)
            else:
                # Off-diagonal: mixed partial
                f_pp = deviance(np.maximum(varpar + ei + ej, _PARAM_FLOOR))
                f_pm = deviance(np.maximum(varpar + ei - ej, _PARAM_FLOOR))
                f_mp = deviance(np.maximum(varpar - ei + ej, _PARAM_FLOOR))
                f_mm = deviance(np.maximum(varpar - ei - ej, _PARAM_FLOOR))
                hessian[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4 * hi * hj)
                hessian[j, i] = hessian[i, j]

    # Invert Hessian to get covariance
    # vcov = 2 * H⁻¹ (factor of 2 because deviance = -2 * loglik)
    try:
        varpar_vcov = 2.0 * np.linalg.inv(hessian)
    except np.linalg.LinAlgError:
        warnings.warn("Hessian is singular, using pseudoinverse", stacklevel=2)
        varpar_vcov = 2.0 * np.linalg.pinv(hessian)

    return varpar_vcov


def _compute_gradient_var_beta(
    varpar: np.ndarray,
    coef_idx: int,
    X: np.ndarray,
    Z: np.ndarray,
    weights: np.ndarray,
    block_sizes: list[int],
    precomputed: dict,
    eps: float = 1e-4,
) -> np.ndarray:
    """
    Compute gradient of Var(β̂_j) with respect to varpar.
    """
    m = len(varpar)
    gradient = np.zeros(m)

    for i in range(m):
        # Adaptive step size
        h = eps * max(abs(varpar[i]), 1.0)

        varpar_plus = varpar.copy()
        varpar_minus = varpar.copy()

        varpar_plus[i] += h
        varpar_minus[i] = max(varpar_minus[i] - h, _PARAM_FLOOR)

        var_plus = _compute_var_beta_at_varpar(
            varpar_plus, coef_idx, X, Z, weights, block_sizes, precomputed
        )
        var_minus = _compute_var_beta_at_varpar(
            varpar_minus, coef_idx, X, Z, weights, block_sizes, precomputed
        )

        gradient[i] = (var_plus - var_minus) / (varpar_plus[i] - varpar_minus[i])

    return gradient


def compute_satterthwaite_df_from_matrices(
    y: np.ndarray,
    X: np.ndarray,
    Z: np.ndarray,
    weights: np.ndarray,
    block_sizes: list[int],
    theta: np.ndarray,
    sigma: float,
    reml: bool = False,
    eps: float = 1e-4,
    precomputed: dict | None = None,
    cholesky_factors=None,
) -> np.ndarray:
    """
    Compute Satterthwaite df from pre-built matrices.

    Parameters
    ----------
    y : np.ndarray (n,)
        Response vector
    X : np.ndarray (n, p)
        Fixed effects design matrix
    Z : np.ndarray (n, q_total)
        Random effects design matrix
    weights : np.ndarray (n,)
        Observation weights
    block_sizes : List[int]
        Number of levels per RE term
    theta : np.ndarray (k,)
        Optimal theta values (sigma_u / sigma_e for each RE term)
    sigma : float
        Optimal residual standard deviation (sigma_e)
    reml : bool
        Whether REML was used for fitting
    eps : float
        Finite difference step size
    precomputed : dict, optional
        Pre-computed weighted products from precompute_weighted_products().
        When provided, skips redundant recomputation. Must contain at
        minimum: XtWX, XtWy, ZtWX, ZtWy, ZtWZ, sqrtW, sum_log_weights.
    cholesky_factors : CholeskyFactors, optional
        Pre-computed Cholesky factors at optimal theta from profile_loglik.
        When provided, eliminates redundant factorizations for evaluations
        at the optimal varpar (f0 in Hessian, Var(beta_j) for each j).
        Duck-typed: must have .L, .RZX, .RXtRX attributes.

    Returns
    -------
    df : np.ndarray (p,)
        Satterthwaite degrees of freedom for each fixed effect coefficient.
        Returns np.nan for coefficients where computation fails.
    """
    n = len(y)
    p = X.shape[1]
    k = len(block_sizes)

    # Validate inputs
    theta = np.atleast_1d(theta).astype(np.float64)
    if len(theta) != k:
        raise ValueError(f"theta has {len(theta)} elements but block_sizes has {k} elements")

    # Ensure weights are positive
    weights = np.maximum(weights, _WEIGHT_FLOOR)

    # Construct varpar = (theta, sigma)
    varpar = np.concatenate([theta, [sigma]])

    # Precompute weighted products — reuse from caller if available
    if precomputed is None:
        precomputed = precompute_weighted_products(y, X, Z, weights)

    # Build cholesky tuple from CholeskyFactors for optimal-point reuse.
    # Duck-typed: check for .L attribute rather than importing CholeskyFactors.
    cholesky_tuple_opt = None
    if cholesky_factors is not None and hasattr(cholesky_factors, "L"):
        cholesky_tuple_opt = (cholesky_factors.L, cholesky_factors.RZX, cholesky_factors.RXtRX)

    # Compute varpar covariance matrix
    # cholesky_tuple_opt is passed for the f0 (center point) evaluation.
    # Perturbed-point evaluations still compute their own factors.
    try:
        varpar_vcov = _compute_varpar_vcov(
            varpar,
            y,
            X,
            Z,
            weights,
            block_sizes,
            reml,
            precomputed,
            eps,
            cholesky_tuple_opt=cholesky_tuple_opt,
        )
    except Exception as e:
        warnings.warn(f"Failed to compute varpar_vcov: {e}", stacklevel=2)
        return np.full(p, np.nan)

    # Compute df for each coefficient
    df = np.zeros(p)

    for j in range(p):
        # Variance of this coefficient — reuse factors at optimal point
        var_j = _compute_var_beta_at_varpar(
            varpar,
            j,
            X,
            Z,
            weights,
            block_sizes,
            precomputed,
            cholesky_tuple=cholesky_tuple_opt,
        )

        if not np.isfinite(var_j) or var_j <= 0:
            df[j] = np.nan
            continue

        # Gradient of variance w.r.t. varpar
        grad_j = _compute_gradient_var_beta(varpar, j, X, Z, weights, block_sizes, precomputed, eps)

        # Variance of variance estimate via delta method
        # Var(Var(beta_j)) approx= grad' @ varpar_vcov @ grad
        var_var_j = grad_j @ varpar_vcov @ grad_j

        # Satterthwaite df
        if var_var_j > 1e-15:
            df_j = 2 * var_j**2 / var_var_j
        else:
            df_j = np.inf

        # Apply lower and upper bounds
        df[j] = min(max(df_j, 1.0), n - p)

    return df


# =============================================================================
# Main Fitting Function
# =============================================================================


def fit_lmm(
    y: np.ndarray,
    X: np.ndarray,
    Z: np.ndarray,
    block_sizes: list[int],
    weights: np.ndarray | None = None,
    re_names: list[str] | None = None,
    reml: bool = False,
    theta_init: np.ndarray | None = None,
    theta_bounds: tuple[float, float] = (0.0, 100.0),
    maxiter: int = 100,
    tol: float = 1e-8,
    restart_edge: bool = True,
    boundary_tol: float = _BOUNDARY_TOL_DEFAULT,
    rhobeg: float | None = None,
    rhobeg_scale: float = 0.5,
    maxfun: int | None = None,
    optimizer: str = "auto",
    diagnostics_surface: bool = False,
    diagnostics_path: bool = False,
    surface_n_points: int = 100,
    surface_theta_range: tuple[float, float] = (0.0, 5.0),
    compute_satterthwaite: bool = False,
    satterthwaite_eps: float = 1e-4,
) -> LMMResult:
    """
    Fit a weighted linear mixed model with k>=1 random intercept terms.

    Model: y = Xβ + Z₁u₁ + Z₂u₂ + ... + ε
    where uᵢ ~ N(0, σ²ᵤᵢ I), ε ~ N(0, σ²ₑ W⁻¹)

    This is the unified fitting function that handles both:
    - k=1: Single random effect (e.g., (1|donor))
    - k>1: Multiple random effects (e.g., (1|donor) + (1|batch))

    Parameters
    ----------
    y : np.ndarray (n,)
        Response variable
    X : np.ndarray (n, p)
        Fixed effects design matrix (should include intercept if desired)
    Z : np.ndarray (n, q_total)
        Random effects indicator matrix. Built once per assay by
        build_model_matrices(), not per gene.
    block_sizes : List[int]
        Number of groups per RE term, e.g. [23, 5] for two crossed RE.
        len(block_sizes) == k, sum(block_sizes) == q_total.
    weights : np.ndarray (n,), optional
        Observation-level precision weights. Default is all ones.
    re_names : List[str], optional
        Names for random effects. Default: ["RE_0", "RE_1", ...]
    reml : bool
        Use REML (True) or ML (False). Default False (dreamlet default).
    theta_init : np.ndarray (k,), optional
        Initial values for theta. If None, computed from data.
    theta_bounds : tuple
        Bounds for each theta. Default (0.0, 100.0).
    maxiter : int
        Maximum iterations for optimizer. Default 100.
    tol : float
        Convergence tolerance (rhoend for BOBYQA). Default 1e-8.
    restart_edge : bool
        If True, restart when theta hits boundary. Default True.
    boundary_tol : float
        Tolerance for detecting boundary. Default _BOUNDARY_TOL_DEFAULT.
    rhobeg : float, optional
        Initial trust region radius. None = auto.
    rhobeg_scale : float
        Scale factor for automatic rhobeg. Default 0.5.
    maxfun : int, optional
        Maximum function evaluations. None = maxiter * 10.
    optimizer : str
        'auto' (default), 'bobyqa', or 'lbfgsb'.
    diagnostics_surface : bool
        Evaluate deviance surface over a 1D theta grid (k=1 only;
        silently skipped for k>1). Default False.
    diagnostics_path : bool
        Track optimization path (works for any k). Default False.
    surface_n_points : int
        Grid points for surface. Default 100.
    surface_theta_range : tuple
        Range for surface. Default (0.0, 5.0).
    compute_satterthwaite : bool
        Compute Satterthwaite degrees of freedom. Default False.
        Adds ~5× overhead per gene but provides accurate df for mixed models.
    satterthwaite_eps : float
        Finite difference step size for Satterthwaite computation. Default 1e-4.

    Returns
    -------
    LMMResult
        Fitted model results

    Examples
    --------
    >>> # Single RE (k=1)
    >>> result = fit_lmm(y, X, Z, block_sizes=[23], weights=w)

    >>> # Multiple RE (k=2)
    >>> result = fit_lmm(y, X, Z, block_sizes=[23, 5], re_names=['donor', 'batch'])
    """
    # Constants for path tracking
    MAX_PATH = 500
    PATH_EPS_ABS = 1e-8
    PATH_EPS_REL = 1e-4

    n = len(y)
    p = X.shape[1]
    k = len(block_sizes)
    q_total = sum(block_sizes)

    if Z.shape != (n, q_total):
        raise ValueError(f"Z shape {Z.shape} inconsistent with n={n}, sum(block_sizes)={q_total}")

    # Default RE names
    if re_names is None:
        re_names = [f"RE_{i}" for i in range(k)]
    elif len(re_names) != k:
        raise ValueError(f"re_names length ({len(re_names)}) must match len(block_sizes) ({k})")

    # Default weights
    if weights is None:
        weights = np.ones(n)
    weights = np.maximum(weights, _WEIGHT_FLOOR)

    # Precompute weighted products
    precomputed = precompute_weighted_products(y, X, Z, weights)

    # Track whether we used a provided theta_init (warm-start)
    used_warm_start = theta_init is not None

    # Compute starting theta if not provided
    if theta_init is None:
        theta_init = compute_starting_theta(y, Z, block_sizes, X)
    theta_init = np.atleast_1d(theta_init).astype(np.float64)

    if len(theta_init) != k:
        raise ValueError(f"theta_init length ({len(theta_init)}) must match k ({k})")

    theta_init_used = theta_init.copy()

    # Compute rhobeg if not provided
    if rhobeg is None:
        _rhobeg = max(np.mean(theta_init) * rhobeg_scale, 0.1)
    else:
        _rhobeg = rhobeg

    # Compute maxfun if not provided
    if maxfun is None:
        _maxfun = maxiter * 10
    else:
        _maxfun = maxfun

    # Determine optimizer
    if optimizer == "auto":
        use_bobyqa = PYBOBYQA_AVAILABLE
    elif optimizer == "bobyqa":
        if not PYBOBYQA_AVAILABLE:
            raise ImportError("pybobyqa not installed. Install with: pip install pybobyqa")
        use_bobyqa = True
    elif optimizer == "lbfgsb":
        use_bobyqa = False
    else:
        raise ValueError(f"Unknown optimizer: {optimizer}. Use 'auto', 'bobyqa', or 'lbfgsb'")

    # Track diagnostics
    total_nfeval = 0
    restart_edge_was_triggered = False

    # Path tracking variables
    theta_path_list = []
    deviance_path_list = []
    path_truncated = False
    last_recorded_theta = None

    # Start timing (cheap — single time.time() call, always available in minimal diagnostics)
    start_time = time.time()

    # Objective function
    def objective(theta_vec):
        nonlocal path_truncated, last_recorded_theta

        theta_vec = np.atleast_1d(theta_vec)
        theta_vec = np.maximum(theta_vec, _PARAM_FLOOR)

        deviance = profile_loglik(
            theta_vec, y, X, Z, weights, block_sizes, reml=reml, precomputed=precomputed
        ).deviance

        # Path tracking with meaningful-change filtering
        if diagnostics_path and not path_truncated:
            should_record = False
            if last_recorded_theta is None:
                should_record = True
            else:
                abs_diff = np.max(np.abs(theta_vec - last_recorded_theta))
                rel_diff = abs_diff / max(np.max(np.abs(last_recorded_theta)), _PARAM_FLOOR)
                if abs_diff > PATH_EPS_ABS or rel_diff > PATH_EPS_REL:
                    should_record = True

            if should_record:
                if len(theta_path_list) < MAX_PATH:
                    theta_path_list.append(theta_vec.copy())
                    deviance_path_list.append(deviance)
                    last_recorded_theta = theta_vec.copy()
                else:
                    path_truncated = True

        return deviance

    def run_optimization(init_theta, current_rhobeg):
        nonlocal total_nfeval

        init_theta = np.maximum(init_theta, _THETA_SNAP_ZERO)
        bounds_lower = np.full(k, theta_bounds[0])
        bounds_upper = np.full(k, theta_bounds[1])

        if use_bobyqa:
            try:
                result = pybobyqa.solve(
                    objective,
                    x0=init_theta,
                    bounds=(bounds_lower, bounds_upper),
                    rhobeg=current_rhobeg,
                    rhoend=tol,
                    maxfun=_maxfun,
                    seek_global_minimum=False,
                    objfun_has_noise=False,
                    scaling_within_bounds=False,
                    do_logging=False,
                    print_progress=False,
                )

                theta_opt = np.maximum(result.x, 0.0)
                total_nfeval += result.nf
                success = result.flag in [result.EXIT_SUCCESS, result.EXIT_MAXFUN_WARNING]

                return theta_opt, success, result.nf, f"BOBYQA: flag={result.flag}"

            except Exception as e:
                warnings.warn(f"pybobyqa failed ({e}), falling back to scipy.minimize", stacklevel=2)

        # L-BFGS-B fallback
        result = minimize(
            objective,
            x0=init_theta,
            method="L-BFGS-B",
            bounds=[(theta_bounds[0], theta_bounds[1])] * k,
            options={"maxiter": maxiter, "ftol": tol, "gtol": tol},
        )

        total_nfeval += result.nfev
        theta_opt = np.maximum(result.x, 0.0)
        return theta_opt, result.success, result.nfev, str(result.message)

    # Run optimization
    theta_opt, converged, n_iter, message = run_optimization(theta_init, _rhobeg)

    # restart_edge logic: if we used warm-start and any theta hit boundary, restart fresh
    any_at_boundary = np.any(theta_opt < boundary_tol)
    if restart_edge and used_warm_start and any_at_boundary:
        fresh_theta = compute_starting_theta(y, Z, block_sizes, X)

        if np.any(fresh_theta > boundary_tol):
            fresh_rhobeg = (
                max(np.mean(fresh_theta) * rhobeg_scale, 0.1) if rhobeg is None else rhobeg
            )
            theta_opt2, converged2, n_iter2, msg2 = run_optimization(fresh_theta, fresh_rhobeg)

            # Use fresh result if it has fewer boundary components
            n_boundary_old = np.sum(theta_opt < boundary_tol)
            n_boundary_new = np.sum(theta_opt2 < boundary_tol)

            if n_boundary_new < n_boundary_old or (converged2 and not converged):
                theta_opt = theta_opt2
                converged = converged2
                n_iter = n_iter2
                message = f"restart_edge triggered; {msg2}"
                restart_edge_was_triggered = True

    # Clean up boundary cases
    theta_opt = np.where(theta_opt < _THETA_SNAP_ZERO, 0.0, theta_opt)

    # Compute boundary flags (per-component)
    hit_lower_bound = theta_opt < boundary_tol

    # Recompute at optimal theta — keep ProfileResult for downstream reuse
    opt_result = profile_loglik(
        theta_opt, y, X, Z, weights, block_sizes, reml=reml, precomputed=precomputed
    )
    deviance = opt_result.deviance
    sigma_e_sq = opt_result.sigma_e_sq
    beta = opt_result.beta
    u = opt_result.u
    b = opt_result.b
    factors = opt_result.factors

    # Bug #51: Override convergence for all-boundary theta with valid deviance.
    # When all theta components snap to 0.0 (no RE variance), BOBYQA may report
    # non-success because its trust region collapsed at the lower bounds.
    # This is not a failure — theta=0 IS the optimum. R's lmer handles this via
    # lmerControl(check.conv.singular = .makeCC("ignore", tol=1e-4)).
    # Only override when ALL components are zero (partial boundary with
    # converged=False could indicate genuine optimizer failure on nonzero components).
    if not converged and np.all(theta_opt == 0.0) and np.isfinite(deviance):
        converged = True

    # Compute variance components
    sigma_e = np.sqrt(sigma_e_sq)
    sigma_u = theta_opt * sigma_e

    # Fitted values
    fitted_values = X @ beta + Z @ b
    residuals = y - fitted_values

    # Log-likelihood
    loglik = -deviance / 2

    # Compute stdev_unscaled at optimal theta — reuse Cholesky factors
    stdev_unscaled = compute_stdev_unscaled_lmm(factors, p)

    # Residual degrees of freedom — reuse Cholesky factors
    try:
        df_resid = compute_rdf(
            theta=theta_opt,
            factors=factors,
            precomputed=precomputed,
            block_sizes=block_sizes,
            n=n,
            p=p,
        )
    except Exception as e:
        warnings.warn(f"Trace-based rdf computation failed: {e}. Using classical n-p.", stacklevel=2)
        df_resid = max(n - p, 0)

    # Compute Satterthwaite degrees of freedom if requested
    df_satterthwaite = None
    if compute_satterthwaite:
        try:
            df_satterthwaite = compute_satterthwaite_df_from_matrices(
                y=y,
                X=X,
                Z=Z,
                weights=weights,
                block_sizes=block_sizes,
                theta=theta_opt,
                sigma=sigma_e,
                reml=reml,
                eps=satterthwaite_eps,
                precomputed=precomputed,
                cholesky_factors=factors,
            )
        except Exception as e:
            warnings.warn(f"Satterthwaite df computation failed: {e}", stacklevel=2)
            df_satterthwaite = np.full(p, np.nan)

    # Build OptimizationDiagnostics — minimal fields always, expensive fields gated
    elapsed = time.time() - start_time

    # Expensive: surface evaluation (k=1 only) and path tracking
    surface_thetas = None
    surface_deviances = None
    surface_grid_min = None
    converged_to_grid_min = None
    theta_path = None
    deviance_path = None

    if diagnostics_surface or diagnostics_path:
        # Surface evaluation: k=1 only. Evaluates deviance over a 1D grid
        # of theta values for visual inspection of the optimization landscape.
        # For k>1, use diagnostics_path to inspect the optimizer trajectory.
        if diagnostics_surface and k == 1:
            surface_thetas, surface_deviances, surface_grid_min = evaluate_deviance_surface_1d(
                y,
                X,
                Z,
                weights,
                block_sizes,
                precomputed,
                theta_range=surface_theta_range,
                n_points=surface_n_points,
                reml=reml,
            )
            converged_to_grid_min = abs(theta_opt[0] - surface_grid_min) < 0.1

        # Build path arrays
        theta_path = np.array(theta_path_list) if theta_path_list else None
        deviance_path = np.array(deviance_path_list) if deviance_path_list else None

    diag_obj = OptimizationDiagnostics(
        # Expensive fields (None unless explicitly requested)
        surface_thetas=surface_thetas,
        surface_deviances=surface_deviances,
        surface_grid_min=np.atleast_1d(surface_grid_min)
        if surface_grid_min is not None
        else None,
        theta_path=theta_path,
        deviance_path=deviance_path,
        path_truncated=path_truncated,
        converged_to_grid_min=converged_to_grid_min,
        # Minimal fields (always populated)
        theta_init_used=theta_init_used,
        theta_optimal=theta_opt,
        convergence_reason=message,
        n_feval=total_nfeval,
        time_elapsed=elapsed,
        hit_lower_bound=hit_lower_bound,
        restart_edge_triggered=restart_edge_was_triggered,
    )

    return LMMResult(
        theta=theta_opt,
        sigma_u=sigma_u,
        sigma_e=sigma_e,
        beta=beta,
        u=u,
        b=b,
        fitted_values=fitted_values,
        residuals=residuals,
        converged=converged,
        n_obs=n,
        n_groups=block_sizes,
        n_fixed=p,
        loglik=loglik,
        reml_loglik=loglik if reml else None,
        Amean=np.mean(y),
        n_iter=n_iter,
        message=message,
        hit_lower_bound=hit_lower_bound,
        restart_edge_triggered=restart_edge_was_triggered,
        re_names=re_names,
        block_sizes=block_sizes,
        diagnostics=diag_obj,
        stdev_unscaled=stdev_unscaled,
        df_resid=float(df_resid),
        df_satterthwaite=df_satterthwaite,
    )


# =============================================================================
# Fixed Effects Fallback
# =============================================================================


def fit_lm(
    y: np.ndarray,
    X: np.ndarray,
    weights: np.ndarray | None = None,
) -> LMMResult:
    """
    Fit a weighted linear model (no random effects).

    This is the primary fitting function for FE-only formulas (no random
    effects in the formula). It matches R's limma::lmFit() behavior:
    weights are used as-is (no per-gene rescaling to mean 1).

    Weight convention
    -----------------
    R's dreamlet dispatches FE formulas to lmFit(), which uses raw voom
    weights. RE formulas go to run_lmm() → lmer(), which rescales weights
    per gene to mean 1 for consistent variance component estimation.

    dreampy mirrors this: fit_gene() calls fit_lm() with raw weights
    for FE formulas and fit_lmm() with rescaled weights for RE formulas.
    This ensures all genes entering eBayes have sigma/stdev_unscaled on a
    consistent scale within each population.

    Parameters
    ----------
    y : np.ndarray (n,)
        Response variable
    X : np.ndarray (n, p)
        Design matrix (should include intercept if desired)
    weights : np.ndarray (n,), optional
        Observation-level precision weights. Default is all ones.
        For the FE path these should be raw voom weights (NOT rescaled).

    Returns
    -------
    LMMResult
        Fitted model results (with theta=0, sigma_u=0)
    """
    n = len(y)
    p = X.shape[1]

    # Default weights
    if weights is None:
        weights = np.ones(n)
    weights = np.maximum(weights, _WEIGHT_FLOOR)

    # Weighted least squares: (X'WX)^{-1} X'Wy
    sqrtW = np.sqrt(weights)
    WX = sqrtW[:, np.newaxis] * X
    Wy = sqrtW * y

    XtWX = WX.T @ WX
    XtWy = WX.T @ Wy

    # Track whether Cholesky succeeded (needed for stdev_unscaled)
    L = None
    try:
        L = cholesky(XtWX, lower=True)
        beta = cho_solve((L, True), XtWy)
    except LinAlgError:
        beta, _, _, _ = lstsq(XtWX, XtWy, rcond=None)

    # Compute stdev_unscaled = sqrt(diag(inv(XtWX)))
    # If Cholesky succeeded: inv(XtWX) = L^{-T} @ L^{-1}
    # diag(inv(XtWX))_j = sum_i (L^{-1})_{i,j}^2
    if L is not None:
        try:
            L_inv = solve_triangular(L, np.eye(p), lower=True)
            stdev_unscaled = np.sqrt(np.sum(L_inv**2, axis=0))
        except LinAlgError:
            # Fallback: use pseudoinverse
            try:
                XtWX_inv = np.linalg.pinv(XtWX)
                stdev_unscaled = np.sqrt(np.maximum(np.diag(XtWX_inv), 0.0))
            except (LinAlgError, np.linalg.LinAlgError):
                stdev_unscaled = np.full(p, np.nan)
    else:
        # Cholesky failed, use pseudoinverse
        try:
            XtWX_inv = np.linalg.pinv(XtWX)
            stdev_unscaled = np.sqrt(np.maximum(np.diag(XtWX_inv), 0.0))
        except (LinAlgError, np.linalg.LinAlgError):
            stdev_unscaled = np.full(p, np.nan)

    # Residual degrees of freedom
    df_resid = max(n - p, 0)

    # Fitted values and residuals
    fitted_values = X @ beta
    residuals = y - fitted_values

    # Weighted residual sum of squares
    wtres = sqrtW * residuals
    wrss = np.sum(wtres**2)

    # Residual variance (unbiased estimator for sigma used by eBayes)
    df = n - p
    sigma_e_sq = wrss / df if df > 0 else wrss
    sigma_e = np.sqrt(sigma_e_sq)

    # ML log-likelihood: uses σ²_ML = wrss/n (MLE, not unbiased estimator)
    sigma_e_sq_ml = wrss / n if n > 0 else wrss
    loglik = -0.5 * n * (np.log(2 * np.pi * sigma_e_sq_ml) + 1) if n > 0 else 0.0

    # For LM (no random effects), Satterthwaite df = n - p for all coefficients
    df_satterthwaite = np.full(p, float(df_resid))

    return LMMResult(
        theta=np.array([]),
        sigma_u=np.array([]),
        sigma_e=sigma_e,
        beta=beta,
        u=np.array([]),
        b=np.array([]),
        fitted_values=fitted_values,
        residuals=residuals,
        converged=True,
        n_obs=n,
        n_groups=[],
        n_fixed=p,
        loglik=loglik,
        reml_loglik=None,
        Amean=np.mean(y),
        n_iter=0,
        message="WLS fit (no random effects)",
        hit_lower_bound=np.array([], dtype=bool),
        restart_edge_triggered=False,
        re_names=[],
        block_sizes=[],
        diagnostics=None,
        stdev_unscaled=stdev_unscaled,
        df_resid=float(df_resid),
        df_satterthwaite=df_satterthwaite,
    )


# =============================================================================
# High-Level Gene Fitting Interface
# =============================================================================


def fit_gene(
    y: np.ndarray,
    X: np.ndarray,
    Z: np.ndarray | None,
    block_sizes: list[int] | None,
    weights: np.ndarray | None = None,
    re_names: list[str] | None = None,
    reml: bool = False,
    ddf: Literal["Satterthwaite", "Kenward-Roger", "adaptive"] | None = "adaptive",
    **kwargs,
) -> dict:
    """
    Fit a single gene, dispatching to LMM or LM based on formula structure.

    This is the interface used by parallel fitting. It handles:
    - Dispatch: Z determines LMM vs LM path
    - Weight convention: RE path rescales to mean 1 (matching R's run_lmm),
      FE path uses raw weights (matching R's lmFit)
    - Converting LMMResult to dict format for aggregation

    Weight rescaling convention
    --------------------------
    R's dreamlet uses two distinct weight conventions:

    - RE formulas → run_lmm() → lmer(): weights rescaled per gene to mean 1
      for consistent variance component (θ = σ_u/σ_e) estimation.
    - FE formulas → lmFit(): raw voom weights, no rescaling.

    This matters because weight rescaling changes the absolute scale of
    sigma and stdev_unscaled (their product SE is invariant). eBayes
    pools sigma² across genes, so all genes must be on the same scale.
    Mixing rescaled and unrescaled genes would corrupt s2_prior.

    Parameters
    ----------
    y : np.ndarray (n,)
        Gene expression values
    X : np.ndarray (n, p)
        Design matrix
    Z : np.ndarray (n, q_total) or None
        Random effects indicator matrix. None → FE-only (LM path).
        Built once per assay by build_model_matrices().
    block_sizes : List[int] or None
        Number of groups per RE term. None → FE-only (LM path).
    weights : np.ndarray (n,), optional
        Precision weights (raw voom weights)
    re_names : List[str], optional
        RE names (required if Z is not None)
    reml : bool
        REML estimation
    ddf : str, optional
        Degrees of freedom method (only used for RE path):
        - "Satterthwaite": Use Satterthwaite approximation
        - "Kenward-Roger": Use Kenward-Roger small-sample correction
        - "adaptive": Use KR if n < 20, else Satterthwaite (matches dreamlet)
        - None: No df adjustment (classical n - p)
    **kwargs
        Passed to fit_lmm (RE path only)

    Returns
    -------
    dict with keys: sigma, Amean, fitted_values, converged, sigma_u, theta,
                   beta, u, hit_lower_bound, restart_edge_triggered, diagnostics,
                   stdev_unscaled, df_resid, df_satterthwaite, ddf_method
    """
    n = len(y)
    p = X.shape[1]

    # Dispatch based on formula structure, not on runtime exceptions.
    # This is the single source of truth: the formula determines the path.
    is_lmm = Z is not None

    # Handle weights: copy to avoid mutating caller's array
    if weights is None:
        w = np.ones(n)
    else:
        w = weights.copy()

    # =====================================================================
    # FE path: weighted least squares with raw weights (matches lmFit)
    # =====================================================================
    if not is_lmm:
        w = np.maximum(w, _WEIGHT_FLOOR)
        # No weight rescaling — matches R's lmFit which uses raw voom weights

        result = fit_lm(y=y, X=X, weights=w)

        # FE path: classical df, no Satterthwaite/KR
        return {
            "sigma": result.sigma_e,
            "Amean": result.Amean,
            "fitted_values": result.fitted_values,
            "converged": result.converged,
            "sigma_u": {},
            "theta": {},
            "beta": result.beta,
            "u": np.array([]),
            "hit_lower_bound": False,
            "hit_lower_bound_per_re": {},
            "restart_edge_triggered": False,
            "diagnostics": None,
            "re_names": [],
            "n_groups": [],
            "stdev_unscaled": result.stdev_unscaled,
            "df_resid": result.df_resid,
            "df_satterthwaite": None,
            "ddf_method": None,
        }

    # =====================================================================
    # RE path: linear mixed model with rescaled weights (matches run_lmm)
    # =====================================================================
    # Rescale weights to mean 1: required for consistent θ estimation
    # across genes. R's run_lmm() does this before calling lmer().
    w = w / np.mean(w)
    w = np.maximum(w, _WEIGHT_FLOOR)

    # Determine effective ddf method based on sample size
    effective_ddf = ddf
    if ddf == "adaptive":
        effective_ddf = "Kenward-Roger" if n < 20 else "Satterthwaite"

    # Determine whether to compute Satterthwaite in fit_lmm
    # (needed as fallback for KR or as primary method)
    compute_satt = effective_ddf in ("Satterthwaite", "Kenward-Roger")

    # Fit the mixed model. No fallback to fit_lm — if fit_lmm converges
    # to θ=0, compute_rdf and compute_stdev_unscaled_lmm both gracefully
    # reduce to their FE equivalents (verified numerically). If fit_lmm
    # genuinely fails, the gene gets NaN → FitStatus.FAILED in eBayes.
    try:
        result = fit_lmm(
            y=y,
            X=X,
            Z=Z,
            block_sizes=block_sizes,
            weights=w,
            re_names=re_names,
            reml=reml,
            compute_satterthwaite=compute_satt,
            **kwargs,
        )

        # Check if converged and sigma is valid
        if not result.converged or not np.isfinite(result.sigma_e) or result.sigma_e <= 0:
            return _make_failed_gene_dict(
                n,
                p,
                y,
                re_names,
                error="LMM did not converge or produced invalid sigma",
            )

    except Exception as e:
        return _make_failed_gene_dict(
            n,
            p,
            y,
            re_names,
            error=str(e),
        )

    # Get base values from fit
    stdev_unscaled = result.stdev_unscaled
    df_satterthwaite = result.df_satterthwaite
    ddf_method_used = effective_ddf

    # Apply Kenward-Roger adjustment if requested and available
    if effective_ddf == "Kenward-Roger" and KENWARD_ROGER_AVAILABLE:
        try:
            # Slice Z into per-term blocks
            Z_blocks = []
            col = 0
            for bs in block_sizes:
                Z_blocks.append(Z[:, col : col + bs])
                col += bs

            # Import compute_vcov_beta_from_fit
            try:
                from .kenward_roger import compute_vcov_beta_from_fit
            except ImportError:
                from kenward_roger import compute_vcov_beta_from_fit

            vcov_beta = compute_vcov_beta_from_fit(
                X=X,
                Z_blocks=Z_blocks,
                theta=result.theta,
                sigma_e=result.sigma_e,
                weights=w,
            )

            # Compute KR adjustment
            kr_result = compute_kr_adjustment(
                theta=result.theta,
                sigma_e=result.sigma_e,
                vcov_beta=vcov_beta,
                X=X,
                Z_blocks=Z_blocks,
                weights=w,
                reml=reml,
            )

            if kr_result.success:
                stdev_unscaled = kr_result.stdev_unscaled_adj
                df_satterthwaite = kr_result.df
                ddf_method_used = "Kenward-Roger"
            else:
                # KR failed, fall back to Satterthwaite
                ddf_method_used = f"Satterthwaite (KR fallback: {kr_result.fallback_reason})"

        except Exception as e:
            # KR computation failed, keep Satterthwaite values
            ddf_method_used = f"Satterthwaite (KR error: {str(e)[:50]})"

    elif effective_ddf == "Kenward-Roger" and not KENWARD_ROGER_AVAILABLE:
        ddf_method_used = "Satterthwaite (KR unavailable)"

    # Convert sigma_u array to dict for types.py compatibility
    sigma_u_dict = {name: result.sigma_u[i] for i, name in enumerate(result.re_names)}

    # Convert theta array to dict as well
    theta_dict = {name: result.theta[i] for i, name in enumerate(result.re_names)}

    return {
        "sigma": result.sigma_e,
        "Amean": result.Amean,
        "fitted_values": result.fitted_values,
        "converged": result.converged,
        "sigma_u": sigma_u_dict,
        "theta": theta_dict,
        "beta": result.beta,
        "u": result.u,
        "hit_lower_bound": result.hit_lower_bound_any,
        "hit_lower_bound_per_re": {
            name: result.hit_lower_bound[i] for i, name in enumerate(result.re_names)
        },
        "restart_edge_triggered": result.restart_edge_triggered,
        "diagnostics": result.diagnostics,
        "re_names": result.re_names,
        "n_groups": result.n_groups,
        "stdev_unscaled": stdev_unscaled,
        "df_resid": result.df_resid,
        "df_satterthwaite": df_satterthwaite,
        "ddf_method": ddf_method_used,
    }


def _make_failed_gene_dict(
    n: int,
    p: int,
    y: np.ndarray,
    re_names: list[str] | None,
    error: str,
) -> dict:
    """
    Construct a failed-gene result dict with NaN values.

    Used when fit_lmm raises an exception or produces invalid results.
    The dict has the same keys as a successful fit so that
    create_de_fit_from_results can process it uniformly.
    """
    return {
        "sigma": np.nan,
        "Amean": np.mean(y),
        "fitted_values": np.full(n, np.nan),
        "converged": False,
        "sigma_u": {},
        "theta": {},
        "beta": np.full(p, np.nan),
        "u": np.array([]),
        "error": error,
        "hit_lower_bound": False,
        "hit_lower_bound_per_re": {},
        "restart_edge_triggered": False,
        "diagnostics": None,
        "re_names": re_names if re_names else [],
        "n_groups": [],
        "stdev_unscaled": np.full(p, np.nan),
        "df_resid": np.nan,
        "df_satterthwaite": None,
        "ddf_method": None,
    }


def fit_genes_parallel(
    Y: np.ndarray,
    X: np.ndarray,
    Z: np.ndarray | None,
    block_sizes: list[int] | None,
    weights_matrix: np.ndarray | None = None,
    re_names: list[str] | None = None,
    reml: bool = False,
    n_jobs: int = -1,
    backend: str = "loky",
    verbose: bool = True,
    ddf: Literal["Satterthwaite", "Kenward-Roger", "adaptive"] | None = "adaptive",
    **kwargs,
) -> list[dict]:
    """
    Fit gene models in parallel using joblib.

    Dispatches each gene to fit_gene(), which routes to fit_lmm (RE) or
    fit_lm (FE) based on Z. Weight rescaling convention is
    handled internally by fit_gene (RE: rescaled to mean 1, FE: raw).

    Parameters
    ----------
    Y : np.ndarray (n_genes, n_samples)
        Gene expression matrix
    X : np.ndarray (n_samples, p)
        Design matrix
    Z : np.ndarray (n_samples, q_total) or None
        Random effects indicator matrix. None → FE-only fitting.
        Built once per assay by build_model_matrices().
    block_sizes : List[int] or None
        Number of groups per RE term. None → FE-only fitting.
    weights_matrix : np.ndarray (n_genes, n_samples), optional
        Per-gene precision weights (raw voom weights)
    re_names : List[str], optional
        RE names
    reml : bool
        REML estimation
    n_jobs : int
        Parallel jobs. -1 = all cores
    backend : str
        Joblib backend
    verbose : bool
        Show progress
    ddf : str, optional
        Degrees of freedom method (RE path only):
        - "Satterthwaite": Use Satterthwaite approximation
        - "Kenward-Roger": Use Kenward-Roger small-sample correction
        - "adaptive": Use KR if n < 20, else Satterthwaite (matches dreamlet)
        - None: No df adjustment
    **kwargs
        Passed to fit_gene → fit_lmm

    Returns
    -------
    List[Dict]
        List of fitted model results, one per gene (order preserved)
    """
    if not JOBLIB_AVAILABLE:
        raise ImportError(
            "joblib is required for parallel fitting. Install with: pip install joblib"
        )

    n_genes = Y.shape[0]

    # Pre-slice data
    y_slices = [Y[i, :].copy() for i in range(n_genes)]
    if weights_matrix is not None:
        w_slices = [weights_matrix[i, :].copy() for i in range(n_genes)]
    else:
        w_slices = [None] * n_genes

    # Define single-gene fitting function
    def _fit_one_gene(i, y, w):
        result = fit_gene(
            y=y,
            X=X,
            Z=Z,
            block_sizes=block_sizes,
            weights=w,
            re_names=re_names,
            reml=reml,
            ddf=ddf,
            **kwargs,
        )
        return i, result

    # Progress bar
    if TQDM_AVAILABLE and verbose:
        iterator = tqdm(range(n_genes), desc="Fitting genes")
    else:
        iterator = range(n_genes)
        if verbose:
            print(f"Fitting {n_genes} genes...")

    # Execute
    if n_jobs == 1:
        results_unordered = [_fit_one_gene(i, y_slices[i], w_slices[i]) for i in iterator]
    else:
        if THREADPOOLCTL_AVAILABLE:
            with threadpool_limits(limits=1, user_api="blas"):
                results_unordered = Parallel(n_jobs=n_jobs, backend=backend)(
                    delayed(_fit_one_gene)(i, y_slices[i], w_slices[i]) for i in iterator
                )
        else:
            if n_jobs != 1:
                warnings.warn(
                    "threadpoolctl not available. Parallel execution may cause "
                    "BLAS thread oversubscription.", stacklevel=2
                )
            results_unordered = Parallel(n_jobs=n_jobs, backend=backend)(
                delayed(_fit_one_gene)(i, y_slices[i], w_slices[i]) for i in iterator
            )

    # Sort by index
    results = [r for _, r in sorted(results_unordered, key=lambda x: x[0])]

    return results


# =============================================================================
# High-Level API: fit_models()
# =============================================================================


def fit_models(
    Y: np.ndarray | AnnData,
    formula: str,
    metadata: pd.DataFrame | None = None,
    weights: np.ndarray | None = None,
    gene_names: np.ndarray | None = None,
    assay_name: str | None = None,
    reml: bool = False,
    ddf: Literal["Satterthwaite", "Kenward-Roger", "adaptive"] | None = "adaptive",
    n_jobs: int = 1,
    backend: str = "loky",
    show_progress: bool = True,
) -> DEFit:
    """
    Fit gene-wise linear (mixed) models.

    This is the Tier 1 API for model fitting. It wraps fit_genes_parallel()
    with formula parsing and design matrix construction.

    Parameters
    ----------
    Y : np.ndarray or AnnData
        Expression matrix. If ndarray: (genes × samples), typically log2 CPM.
        If AnnData: expects (samples × genes) in .X, will transpose internally.
    formula : str
        Model formula, e.g., "~ condition + age + (1|donor)"
    metadata : pd.DataFrame, optional
        Sample metadata with formula variables as columns.
        If Y is AnnData and metadata is None, reads from Y.obs.
    weights : np.ndarray, optional
        Precision weights matrix (genes × samples). If None, uses unweighted fit.
        If Y is AnnData and weights is None, reads from Y.layers['weights'] if present.
    gene_names : np.ndarray, optional
        Gene identifiers (length must match Y.shape[0] for array, Y.shape[1] for AnnData).
        If None and Y is AnnData, reads from Y.var_names.
        If None and Y is array, uses numeric indices [0, 1, 2, ...].
    assay_name : str, optional
        Name of assay/cell type. Stored in DEFit for multi-assay pipelines.
    reml : bool
        Use REML estimation. Default: False (ML, matching dreamlet).
    ddf : str, optional
        Denominator degrees of freedom method:
        - "Satterthwaite": df adjustment only (default)
        - "Kenward-Roger": vcov AND df adjustment (small-sample correction)
        - "adaptive": Kenward-Roger if n_samples < 20, else Satterthwaite
          (matches dreamlet's ddf='adaptive')
        - None: No df adjustment, use classical n - p
    n_jobs : int
        Number of parallel jobs. Default: 1 (sequential).
        Use -1 for all available cores.
    backend : str
        Joblib backend. Default: 'loky'
    show_progress : bool
        Show progress bar. Default: True

    Returns
    -------
    DEFit
        Fitted model container with:
        - coefficients: (n_genes, n_coef) array
        - sigma: (n_genes,) array of residual SDs
        - sigma_u: Dict mapping RE names to (n_genes,) arrays
        - fit_status: (n_genes,) FitStatus enum array
        - assay_name: str if provided
        - ddf_method: str indicating which df method was used
        - etc.

    Notes
    -----
    This function:
    1. Parses the formula to extract fixed and random effects
    2. Builds the design matrix X from metadata
    3. Builds the random effects structure from grouping variables
    4. Fits gene-wise LMMs (if random effects) or LMs (if fixed only)
    5. Returns results in a DEFit container

    For more control, use fit_genes_parallel() directly.

    Example
    -------
    >>> # Array mode
    >>> from dreampy import log2cpm, compute_tmm_factors, fit_models
    >>> norm_factors = compute_tmm_factors(counts)
    >>> y = log2cpm(counts, norm_factors=norm_factors)
    >>> fit = fit_models(y, "~ condition + (1|donor)", metadata)

    >>> # AnnData mode
    >>> pb = estimate_weights(pb, "~ condition + (1|donor)")
    >>> fit = fit_models(pb, "~ condition + (1|donor)", assay_name="CD4_T")
    """

    # Import types module (avoid circular import)
    try:
        from .de_types import DEFit, create_de_fit_from_results
    except ImportError:
        # Standalone mode - try direct import
        try:
            from de_types import DEFit, create_de_fit_from_results
        except ImportError:
            DEFit = None
            create_de_fit_from_results = None

    # Import design matrix builder
    try:
        from .design import build_model_matrices
    except ImportError:
        from design import build_model_matrices

    # =================================================================
    # Handle AnnData input
    # =================================================================
    is_anndata = hasattr(Y, "X") and hasattr(Y, "obs")

    if is_anndata:
        adata = Y

        # Extract Y matrix (transpose from samples×genes to genes×samples)
        if hasattr(adata.X, "toarray"):
            Y = adata.X.toarray().T
        else:
            Y = np.asarray(adata.X).T

        # Extract metadata from .obs if not provided
        if metadata is None:
            metadata = adata.obs

        # Extract gene_names from .var_names if not provided
        if gene_names is None:
            gene_names = adata.var_names.values

        # Extract weights from .layers['weights'] if present and not provided
        if weights is None and "weights" in adata.layers:
            # Transpose from samples×genes to genes×samples
            if hasattr(adata.layers["weights"], "toarray"):
                weights = adata.layers["weights"].toarray().T
            else:
                weights = np.asarray(adata.layers["weights"]).T

    n_genes, n_samples = Y.shape

    # Validate metadata is provided (required for both modes)
    if metadata is None:
        raise ValueError(
            "metadata is required. For array input, pass metadata explicitly. "
            "For AnnData input, metadata is read from adata.obs."
        )

    if n_samples != len(metadata):
        raise ValueError(f"Y has {n_samples} samples but metadata has {len(metadata)} rows")

    # Validate gene_names if provided
    if gene_names is not None:
        gene_names = np.asarray(gene_names)
        if len(gene_names) != n_genes:
            raise ValueError(
                f"gene_names length ({len(gene_names)}) does not match "
                f"number of genes in Y ({n_genes})"
            )

    # Build all model matrices once for this assay
    formula_str = formula if isinstance(formula, str) else formula.original
    X, coef_names, Z, block_sizes, re_names, formula_clean, cleaning_warnings = (
        build_model_matrices(formula_str, metadata, verbose=show_progress)
    )

    # Emit cleaning warnings if any
    for w in cleaning_warnings:
        if "dropped" in w.lower() or "collinear" in w.lower():
            warnings.warn(w, stacklevel=2)

    # Default weights
    if weights is None:
        weights_matrix = np.ones((n_genes, n_samples))
    else:
        weights_matrix = weights

    # Fit all genes
    results = fit_genes_parallel(
        Y=Y,
        X=X,
        Z=Z,
        block_sizes=block_sizes,
        weights_matrix=weights_matrix,
        re_names=re_names,
        reml=reml,
        n_jobs=n_jobs,
        backend=backend,
        verbose=show_progress,
        ddf=ddf,
    )

    # Convert to DEFit if available
    if DEFit is not None:
        # Use provided gene_names or fall back to numeric indices
        if gene_names is None:
            gene_names_final = np.arange(n_genes)
        else:
            gene_names_final = gene_names
        sample_names = metadata.index.values if hasattr(metadata, "index") else np.arange(n_samples)

        return create_de_fit_from_results(
            gene_results=results,
            gene_names=gene_names_final,
            sample_names=sample_names,
            formula=formula_clean,
            design_matrix=X,
            coef_names=coef_names,
            method="lmm" if Z is not None else "lm",
            reml=reml,
            weights=weights_matrix if weights is not None else None,
            assay_name=assay_name,
        )

    # Return raw results if types module not available
    return results


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Data classes
    "LMMResult",
    "OptimizationDiagnostics",
    "CholeskyFactors",
    "ProfileResult",
    # High-level API
    "fit_models",
    # Core functions
    "fit_lmm",
    "fit_lm",
    "fit_gene",
    "fit_genes_parallel",
    # Satterthwaite
    "compute_satterthwaite_df_from_matrices",
    # Helpers
    "precompute_weighted_products",
    "profile_loglik",
    "compute_starting_theta",
]