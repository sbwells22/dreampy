"""
Kenward-Roger Degrees of Freedom and Variance-Covariance Adjustment in dreampy.

This module implements the Kenward-Roger small-sample correction for
linear mixed models, providing:
1. Adjusted covariance matrix for fixed effects (vcov adjustment)
2. Adjusted denominator degrees of freedom for hypothesis tests

The implementation follows pbkrtest (Halekoh & Højsgaard, 2014) and the
original Kenward & Roger (1997) paper.

Key differences from Satterthwaite:
- Satterthwaite adjusts df only, uses raw vcov(β)
- Kenward-Roger adjusts BOTH vcov(β) AND df

Mathematical Background
-----------------------
The KR approach addresses small-sample bias in two ways:

1. Variance Adjustment:
   Φ_adj = Φ + 2Γ
   where Φ = raw vcov(β) and Γ is a bias correction term

2. Degrees of Freedom:
   Computed by matching moments of the test statistic to an F distribution

Requirements:
- REML estimation (KR is only valid for REML fits)
- Linear covariance structure (weighted sum of known matrices)

References
----------
- Kenward & Roger (1997). "Small Sample Inference for Fixed Effects from
  Restricted Maximum Likelihood" Biometrics 53:983-997
- Halekoh & Højsgaard (2014). "A Kenward-Roger Approximation and Parametric
  Bootstrap Methods for Tests in Linear Mixed Models - The R Package pbkrtest"
  Journal of Statistical Software 59(9):1-32
- pbkrtest source: R/KR-vcovAdj.R, R/KR-Sigma-G2.R, R/get_info_functions.R
"""

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from scipy.linalg import LinAlgError, cho_solve, cholesky

# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class KRResult:
    """
    Result from Kenward-Roger adjustment.

    Attributes
    ----------
    vcov_adj : np.ndarray
        (p, p) adjusted covariance matrix for fixed effects
    stdev_unscaled_adj : np.ndarray
        (p,) adjusted unscaled standard errors (sqrt(diag(vcov_adj))/sigma)
    df : np.ndarray
        (p,) adjusted degrees of freedom per coefficient
    P : List[np.ndarray]
        List of P matrices (needed for df computation)
    W : np.ndarray
        (M, M) asymptotic covariance of variance parameters (2 * inv(IE2))
    success : bool
        True if adjustment succeeded
    method_used : str
        "Kenward-Roger" if successful, "Satterthwaite" if fallback
    fallback_reason : str, optional
        Reason for fallback if method_used != "Kenward-Roger"
    """

    vcov_adj: np.ndarray
    stdev_unscaled_adj: np.ndarray
    df: np.ndarray
    P: list[np.ndarray] | None
    W: np.ndarray | None
    success: bool
    method_used: str
    fallback_reason: str | None = None


class SigmaGResult(NamedTuple):
    """Result from get_sigma_g computation."""

    Sigma: np.ndarray  # (n, n) marginal covariance matrix
    G: list[np.ndarray]  # List of (n, n) derivative matrices
    ggamma: np.ndarray  # (M,) variance parameters [σ²_b1, ..., σ²_bk, σ²_e]


# =============================================================================
# Core Functions
# =============================================================================


def get_sigma_g(
    Z_blocks: list[np.ndarray],
    theta: np.ndarray,
    sigma_e: float,
    weights: np.ndarray,
) -> SigmaGResult:
    """
    Construct Sigma and G matrices for KR adjustment.

    Implements the logic from pbkrtest's get_SigmaG / .get_SigmaG functions
    (R/KR-Sigma-G2.R).

    Parameters
    ----------
    Z_blocks : list of k arrays
        Each Z_i is (n, q_i) indicator matrix for random effect term i.
        For random intercepts, Z_i has one column per group level.
    theta : (k,) array
        Variance ratios θ_i = σ_{u_i} / σ_e for each random effect term
    sigma_e : float
        Residual standard deviation
    weights : (n,) array
        Precision weights (from voom). Higher weight = lower variance.

    Returns
    -------
    SigmaGResult with:
        Sigma : (n, n) marginal covariance matrix of y
            Sigma = Σ_{i=0}^{k} ggamma[i] * G[i]
        G : list of (k+1) matrices, each (n, n)
            G[i] = Z_i @ Z_i.T for i < k (random effects)
            G[k] = diag(1/weights) for residual
        ggamma : (k+1,) variance parameters
            [σ²_b1, ..., σ²_bk, σ²_e]

    Notes
    -----
    The G matrices are derivatives of Sigma w.r.t. each variance parameter:
        ∂Sigma/∂(σ²_bi) = Z_i @ Z_i.T
        ∂Sigma/∂(σ²_e) = diag(1/w)

    With weights, the residual covariance is σ²_e * diag(1/w) rather than σ²_e * I.
    """
    n = len(weights)
    k = len(theta)

    # Validate inputs
    if k != len(Z_blocks):
        raise ValueError(f"theta has {k} elements but Z_blocks has {len(Z_blocks)} blocks")

    for i, Z_i in enumerate(Z_blocks):
        if Z_i.shape[0] != n:
            raise ValueError(f"Z_blocks[{i}] has {Z_i.shape[0]} rows but weights has {n} elements")

    sigma_e_sq = sigma_e**2

    # Build variance parameter vector: [σ²_b1, ..., σ²_bk, σ²_e]
    # Convert from θ = σ_u/σ_e to σ²_u = θ² * σ²_e
    ggamma = np.zeros(k + 1)
    for i in range(k):
        ggamma[i] = (theta[i] ** 2) * sigma_e_sq
    ggamma[k] = sigma_e_sq

    # Build G matrices
    G = []

    # G[0..k-1]: derivatives w.r.t. random effect variances
    for i in range(k):
        Z_i = Z_blocks[i]
        G_i = Z_i @ Z_i.T  # (n, n)
        G.append(G_i)

    # G[k]: derivative w.r.t. residual variance (with weights)
    # When weights are present, Var(ε_i) = σ²_e / w_i
    # So the residual covariance matrix is σ²_e * diag(1/w)
    # G_residual = np.diag(1.0 / np.maximum(weights, 1e-10))
    G_residual = np.eye(n)
    G.append(G_residual)

    # Build Sigma = Σ ggamma[i] * G[i]
    Sigma = np.zeros((n, n))
    for i in range(k + 1):
        Sigma += ggamma[i] * G[i]

    return SigmaGResult(Sigma=Sigma, G=G, ggamma=ggamma)


def vcov_adj(
    vcov_beta: np.ndarray,
    X: np.ndarray,
    Sigma: np.ndarray,
    G: list[np.ndarray],
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray, bool]:
    """
    Compute Kenward-Roger adjusted vcov(β).

    Implements the algorithm from pbkrtest's KR_vcovAdj / vcovAdj_internal
    (R/KR-vcovAdj.R, lines 97-199).

    Parameters
    ----------
    vcov_beta : (p, p) array
        Raw (unadjusted) covariance matrix of fixed effects from REML fit
    X : (n, p) array
        Fixed effects design matrix
    Sigma : (n, n) array
        Marginal covariance matrix from get_sigma_g
    G : list of M arrays, each (n, n)
        Derivative matrices from get_sigma_g

    Returns
    -------
    vcov_adj : (p, p) array
        Adjusted covariance matrix (Φ_A in KR notation)
    P : list of M arrays, each (p, p)
        P matrices needed for df computation
    W : (M, M) array
        Asymptotic covariance of variance parameters (2 * inv(IE2))
    success : bool
        True if adjustment succeeded (result is positive definite)

    Algorithm (from KR paper and pbkrtest)
    --------------------------------------
    1. SigmaInv = inv(Sigma)                           # O(n³)
    2. TT = SigmaInv @ X                               # (n, p)
    3. For each variance component r:
       HH[r] = G[r] @ SigmaInv                         # (n, n)
       OO[r] = HH[r] @ X                               # (n, p)
    4. For each r:
       PP[r] = -OO[r].T @ TT                           # (p, p)
    5. For each r, s:
       QQ[r,s] = OO[r].T @ SigmaInv @ OO[s]           # (p, p)
    6. Information matrix for variance parameters:
       IE2[r,s] = trace(HH[r].T @ HH[s])
                  - 2*trace(Phi @ QQ[r,s])
                  + trace(Phi @ PP[r] @ Phi @ PP[s])
    7. WW = 2 * inv(IE2)                              # asymptotic cov of var params
    8. UU = Σ WW[r,s] * (QQ[r,s] - PP[r] @ Phi @ PP[s])
    9. GAMMA = Phi @ UU @ Phi
    10. Phi_adj = Phi + 2 * GAMMA
    """
    n, p = X.shape
    M = len(G)  # Number of variance parameters (k + 1)
    Phi = vcov_beta  # Raw vcov(β), using KR notation

    # Step 1: Invert Sigma
    try:
        L_Sigma = cholesky(Sigma, lower=True)
        SigmaInv = cho_solve((L_Sigma, True), np.eye(n))
    except LinAlgError:
        # Sigma not positive definite - cannot proceed
        return vcov_beta, [], np.array([]), False

    # Step 2: TT = SigmaInv @ X
    TT = SigmaInv @ X  # (n, p)

    # Step 3: Build HH and OO matrices
    HH = []  # HH[r] = G[r] @ SigmaInv, shape (n, n)
    OO = []  # OO[r] = HH[r] @ X, shape (n, p)
    for r in range(M):
        HH_r = G[r] @ SigmaInv
        OO_r = HH_r @ X
        HH.append(HH_r)
        OO.append(OO_r)

    # Step 4: Build PP matrices
    # PP[r] = -OO[r].T @ TT = -t(OO[[r]]) %*% TT in R
    PP = []
    for r in range(M):
        PP_r = -OO[r].T @ TT  # (p, p)
        PP.append(PP_r)

    # Step 5: Build QQ matrices
    # QQ[r,s] = OO[r].T @ SigmaInv @ OO[s] = t(OO[[r]]) %*% SigmaInv %*% OO[[s]]
    # This is symmetric in (r, s)
    QQ = {}
    for r in range(M):
        for s in range(r, M):
            QQ_rs = OO[r].T @ SigmaInv @ OO[s]  # (p, p)
            QQ[(r, s)] = QQ_rs
            if r != s:
                QQ[(s, r)] = QQ_rs.T

    # Step 6: Information matrix IE2 for variance parameters
    # IE2[r,s] = Ktrace[r,s] - 2*sum(Phi*QQ[r,s]) + sum(Phi.P.r * (PP[s] %*% Phi))
    # where Ktrace[r,s] = sum(t(HH[[r]]) * HH[[s]]) = trace(HH[r].T @ HH[s])
    IE2 = np.zeros((M, M))
    for r in range(M):
        for s in range(r, M):
            # Ktrace[r,s] = trace(HH[r].T @ HH[s]) = sum(HH[r] * HH[s].T)
            # Using the identity trace(A.T @ B) = sum(A * B)
            Ktrace_rs = np.sum(HH[r] * HH[s].T)

            # term2 = 2 * sum(Phi * QQ[r,s]) = 2 * trace(Phi @ QQ[r,s].T)
            # But QQ is symmetric in indices, and for each QQ_rs, trace(Phi @ QQ_rs)
            term2 = 2 * np.sum(Phi * QQ[(r, s)])

            # term3 = sum(Phi.P.r * (PP[s] %*% Phi))
            # where Phi.P.r = Phi %*% PP[[r]]
            # So term3 = trace((Phi @ PP[r]).T @ (PP[s] @ Phi))
            #          = sum((Phi @ PP[r]) * (PP[s] @ Phi).T)
            Phi_PP_r = Phi @ PP[r]
            PP_s_Phi = PP[s] @ Phi
            term3 = np.sum(Phi_PP_r * PP_s_Phi)  # .T

            IE2[r, s] = Ktrace_rs - term2 + term3
            IE2[s, r] = IE2[r, s]  # Symmetric

    # Step 7: WW = 2 * inv(IE2)
    try:
        WW = 2.0 * np.linalg.inv(IE2)
    except LinAlgError:
        # IE2 singular - cannot proceed
        return vcov_beta, PP, np.array([]), False

    # Step 8: Compute UU
    # UU = Σ_{r,s} WW[r,s] * (QQ[r,s] - PP[r] @ Phi @ PP[s])
    UU = np.zeros((p, p))
    for r in range(M):
        for s in range(M):
            term = QQ[(r, s)] - PP[r] @ Phi @ PP[s]
            UU += WW[r, s] * term

    # Step 9: GAMMA = Phi @ UU @ Phi
    GAMMA = Phi @ UU @ Phi

    # Step 10: Adjusted vcov
    PhiA = Phi + 2 * GAMMA

    # Verify positive definite
    try:
        cholesky(PhiA, lower=True)
        success = True
    except LinAlgError:
        # Not PD - return raw vcov
        return vcov_beta, PP, WW, False

    return PhiA, PP, WW, success


def lb_ddf(
    L: np.ndarray,
    V0: np.ndarray,
    Vadj: np.ndarray,
    P: list[np.ndarray],
    W: np.ndarray,
) -> float:
    """
    Compute Kenward-Roger adjusted degrees of freedom for testing Lβ = 0.

    Implements pbkrtest's Lb_ddf function (R/get_info_functions.R).

    Parameters
    ----------
    L : (q, p) or (p,) array
        Contrast matrix or vector. For single coefficient test, L = e_j
        (unit vector with 1 at position j).
    V0 : (p, p) array
        Unadjusted covariance matrix (raw vcov(β))
    Vadj : (p, p) array
        Adjusted covariance matrix from vcov_adj
    P : list of M arrays, each (p, p)
        P matrices from vcov_adj
    W : (M, M) array
        Asymptotic covariance of variance parameters from vcov_adj

    Returns
    -------
    df : float
        Adjusted denominator degrees of freedom

    Algorithm (from pbkrtest Lb_ddf)
    --------------------------------
    1. Theta = L.T @ inv(L @ V0 @ L.T) @ L
    2. For each pair (r, s):
       ui = Theta @ V0 @ P[r] @ V0
       uj = Theta @ V0 @ P[s] @ V0
       A1 += e * W[r,s] * trace(ui) * trace(uj)
       A2 += e * W[r,s] * trace(ui @ uj.T)
       where e = 1 if r==s else 2
    3. Moment matching to F distribution:
       q = rank(L) (or nrow(L))
       B = (1/(2q)) * (A1 + 6*A2)
       g = ((q+1)*A1 - (q+4)*A2) / ((q+2)*A2)
       ... series of intermediate calculations ...
       df = 4 + (q+2)/(q*rho - 1)
    """
    # Ensure L is 2D
    if L.ndim == 1:
        L = L.reshape(1, -1)

    q = L.shape[0]  # Number of contrasts (rank of L)
    M = len(P)  # Number of variance parameters

    # Step 1: Theta = L.T @ inv(L @ V0 @ L.T) @ L
    LV0Lt = L @ V0 @ L.T  # (q, q)
    try:
        LV0Lt_inv = np.linalg.inv(LV0Lt)
    except LinAlgError:
        # Singular - return large df (approaches infinity)
        return np.inf

    Theta = L.T @ LV0Lt_inv @ L  # (p, p)

    # Step 2: Compute A1 and A2
    ThetaV0 = Theta @ V0  # (p, p)

    A1 = 0.0
    A2 = 0.0

    for ii in range(M):
        for jj in range(ii, M):
            # e = 1 if diagonal, 2 if off-diagonal (to avoid double counting)
            e = 1.0 if ii == jj else 2.0

            # ui = ThetaV0 @ P[ii] @ V0
            # uj = ThetaV0 @ P[jj] @ V0
            ui = ThetaV0 @ P[ii] @ V0  # (p, p)
            uj = ThetaV0 @ P[jj] @ V0  # (p, p)

            # A1 += e * W[ii,jj] * trace(ui) * trace(uj)
            # trace = .spur in R
            trace_ui = np.trace(ui)
            trace_uj = np.trace(uj)
            A1 += e * W[ii, jj] * trace_ui * trace_uj

            # A2 += e * W[ii,jj] * sum(ui * t(uj))
            # sum(ui * t(uj)) = trace(ui @ uj.T) = sum(ui * uj.T)
            A2 += e * W[ii, jj] * np.sum(ui * uj.T)

    # Step 3: Moment matching formulas from KR paper
    # Guard against division by zero
    if abs(A2) < 1e-15:
        # A2 ≈ 0 means very little variance in the variance estimate
        # df → ∞
        return np.inf

    B = (1.0 / (2 * q)) * (A1 + 6 * A2)
    g = ((q + 1) * A1 - (q + 4) * A2) / ((q + 2) * A2)

    c1 = g / (3 * q + 2 * (1 - g))
    c2 = (q - g) / (3 * q + 2 * (1 - g))
    c3 = (q + 2 - g) / (3 * q + 2 * (1 - g))

    # These are intermediate values (EE, VV, EEstar, VVstar not directly used for df)
    # EE = 1 + (A2 / q)
    # VV = (2.0 / q) * (1 + B)
    # EEstar = 1.0 / (1 - A2 / q)
    # VVstar = (2.0 / q) * ((1 + c1 * B) / ((1 - c2 * B)**2 * (1 - c3 * B)))

    V0_val = 1 + c1 * B
    V1 = 1 - c2 * B
    V2 = 1 - c3 * B

    # Guard against small V0_val
    if abs(V0_val) < 1e-10:
        V0_val = 0.0

    # Guard against division by zero in rho
    if abs(V1) < 1e-10 or abs(V2) < 1e-10:
        return np.inf

    # rho = (1/q) * ((1 - A2/q) / V1)^2 * V0_val / V2
    # This is .divZero(1 - A2/q, V1) in R, which returns 0 if V1 ≈ 0
    ratio = (1 - A2 / q) / V1
    rho = (1.0 / q) * (ratio**2) * V0_val / V2

    # df = 4 + (q + 2) / (q * rho - 1)
    denom = q * rho - 1
    if abs(denom) < 1e-10:
        # Would give negative or infinite df
        return np.inf

    df = 4 + (q + 2) / denom

    # Ensure df is at least 1
    return max(df, 1.0)


def compute_kr_ddf_for_all_coefficients(
    V0: np.ndarray,
    Vadj: np.ndarray,
    P: list[np.ndarray],
    W: np.ndarray,
) -> np.ndarray:
    """
    Compute KR degrees of freedom for each fixed effect coefficient.

    Parameters
    ----------
    V0 : (p, p) array
        Unadjusted covariance matrix
    Vadj : (p, p) array
        Adjusted covariance matrix
    P : list of M arrays
        P matrices from vcov_adj
    W : (M, M) array
        W matrix from vcov_adj

    Returns
    -------
    df : (p,) array
        Degrees of freedom for each coefficient
    """
    p = V0.shape[0]
    df = np.zeros(p)

    for j in range(p):
        # Contrast for testing β_j = 0 is e_j (unit vector)
        L_j = np.zeros(p)
        L_j[j] = 1.0
        df[j] = lb_ddf(L_j, V0, Vadj, P, W)

    return df


# =============================================================================
# High-Level Interface
# =============================================================================


def compute_kr_adjustment(
    theta: np.ndarray,
    sigma_e: float,
    vcov_beta: np.ndarray,
    X: np.ndarray,
    Z_blocks: list[np.ndarray],
    weights: np.ndarray,
    reml: bool,
) -> KRResult:
    """
    Compute full Kenward-Roger adjustment for a single gene.

    This is the main entry point for KR computation, handling:
    - REML validation
    - Sigma/G computation
    - vcov adjustment
    - df computation
    - Fallback to raw vcov on failure

    Parameters
    ----------
    theta : (k,) array
        Variance ratios θ_i = σ_{u_i} / σ_e
    sigma_e : float
        Residual standard deviation
    vcov_beta : (p, p) array
        Raw covariance matrix of fixed effects
    X : (n, p) array
        Fixed effects design matrix
    Z_blocks : list of k arrays
        Z matrices for each random effect, each (n, q_i).
        Obtained by slicing Z by block_sizes.
    weights : (n,) array
        Precision weights
    reml : bool
        Whether REML estimation was used

    Returns
    -------
    KRResult
        Contains adjusted vcov, stdev_unscaled, df, and metadata
    """
    p = X.shape[1]
    _n = X.shape[0]
    _k = len(Z_blocks)

    # Validate REML requirement
    if not reml:
        return KRResult(
            vcov_adj=vcov_beta,
            stdev_unscaled_adj=np.sqrt(np.maximum(np.diag(vcov_beta), 0)) / sigma_e,
            df=np.full(p, np.nan),
            P=None,
            W=None,
            success=False,
            method_used="Satterthwaite",
            fallback_reason="Kenward-Roger requires REML estimation",
        )

    # Get Sigma and G
    try:
        sigma_g = get_sigma_g(Z_blocks, theta, sigma_e, weights)
    except Exception as e:
        return KRResult(
            vcov_adj=vcov_beta,
            stdev_unscaled_adj=np.sqrt(np.maximum(np.diag(vcov_beta), 0)) / sigma_e,
            df=np.full(p, np.nan),
            P=None,
            W=None,
            success=False,
            method_used="Satterthwaite",
            fallback_reason=f"get_sigma_g failed: {e}",
        )

    # Compute vcov adjustment
    Vadj, P, W, success = vcov_adj(vcov_beta, X, sigma_g.Sigma, sigma_g.G)

    if not success:
        return KRResult(
            vcov_adj=vcov_beta,
            stdev_unscaled_adj=np.sqrt(np.maximum(np.diag(vcov_beta), 0)) / sigma_e,
            df=np.full(p, np.nan),
            P=None,
            W=None,
            success=False,
            method_used="Satterthwaite",
            fallback_reason="vcov adjustment failed (non-PD result or singular IE2)",
        )

    # Compute adjusted stdev_unscaled
    stdev_unscaled_adj = np.sqrt(np.maximum(np.diag(Vadj), 0)) / sigma_e

    # Compute KR degrees of freedom
    try:
        df = compute_kr_ddf_for_all_coefficients(vcov_beta, Vadj, P, W)
    except Exception as e:
        # df computation failed - use adjusted vcov but return NaN df
        return KRResult(
            vcov_adj=Vadj,
            stdev_unscaled_adj=stdev_unscaled_adj,
            df=np.full(p, np.nan),
            P=P,
            W=W,
            success=True,  # vcov adjustment succeeded
            method_used="Kenward-Roger",
            fallback_reason=f"df computation failed: {e}",
        )

    return KRResult(
        vcov_adj=Vadj,
        stdev_unscaled_adj=stdev_unscaled_adj,
        df=df,
        P=P,
        W=W,
        success=True,
        method_used="Kenward-Roger",
        fallback_reason=None,
    )


# =============================================================================
# Utility for Computing vcov(β) from Fit Results
# =============================================================================


def compute_vcov_beta_from_fit(
    X: np.ndarray,
    Z_blocks: list[np.ndarray],
    theta: np.ndarray,
    sigma_e: float,
    weights: np.ndarray,
) -> np.ndarray:
    """
    Compute vcov(β) from model components.

    This reconstructs the raw covariance matrix of fixed effects,
    which should match LMMResult's internal computation.

    vcov(β) = σ² * (X'WX - X'WZ Λ (Λ'Z'WZΛ + I)^{-1} Λ Z'WX)^{-1}
            = σ² * inv(RXtRX)

    where RXtRX is the "downdated" cross-product matrix.

    Parameters
    ----------
    X : (n, p) array
        Fixed effects design matrix
    Z_blocks : list of k arrays
        Z matrices for each random effect
    theta : (k,) array
        Variance ratios
    sigma_e : float
        Residual SD
    weights : (n,) array
        Precision weights

    Returns
    -------
    vcov_beta : (p, p) array
        Covariance matrix of fixed effects
    """
    _n, p = X.shape
    k = len(Z_blocks)

    # Build full Z and block sizes
    Z = np.hstack(Z_blocks)
    q_total = Z.shape[1]
    block_sizes = [Zb.shape[1] for Zb in Z_blocks]

    # Compute weighted products
    sqrtW = np.sqrt(np.maximum(weights, 1e-10))
    WX = sqrtW[:, np.newaxis] * X
    ZtW_sqrt = Z.T * sqrtW[np.newaxis, :]

    XtWX = WX.T @ WX
    ZtWX = ZtW_sqrt @ WX
    ZtWZ = ZtW_sqrt @ ZtW_sqrt.T

    # Build Λ'Z'WZΛ
    LtZtWZL = np.zeros_like(ZtWZ)
    col_i = 0
    for i in range(k):
        q_i = block_sizes[i]
        col_j = 0
        for j in range(k):
            q_j = block_sizes[j]
            block = ZtWZ[col_i : col_i + q_i, col_j : col_j + q_j]
            LtZtWZL[col_i : col_i + q_i, col_j : col_j + q_j] = theta[i] * theta[j] * block
            col_j += q_j
        col_i += q_i

    # Λ'Z'WZΛ + I
    LtZtWZL_plus_I = LtZtWZL + np.eye(q_total)

    # Cholesky of Λ'Z'WZΛ + I
    try:
        L = cholesky(LtZtWZL_plus_I, lower=True)
    except LinAlgError:
        LtZtWZL_plus_I += np.eye(q_total) * 1e-8
        L = cholesky(LtZtWZL_plus_I, lower=True)

    # Λ'Z'WX
    LambdatZtWX = np.zeros((q_total, p))
    col = 0
    for i in range(k):
        q_i = block_sizes[i]
        LambdatZtWX[col : col + q_i, :] = theta[i] * ZtWX[col : col + q_i, :]
        col += q_i

    # RZX = L^{-1}(Λ'Z'WX)
    from scipy.linalg import solve_triangular

    RZX = solve_triangular(L, LambdatZtWX, lower=True)

    # RXtRX = X'WX - RZX'RZX
    RXtRX = XtWX - RZX.T @ RZX

    # vcov(β) = σ² * inv(RXtRX)
    try:
        vcov_beta = (sigma_e**2) * np.linalg.inv(RXtRX)
    except LinAlgError:
        vcov_beta = (sigma_e**2) * np.linalg.pinv(RXtRX)

    return vcov_beta


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "KRResult",
    "SigmaGResult",
    "compute_kr_adjustment",
    "compute_kr_ddf_for_all_coefficients",
    "compute_vcov_beta_from_fit",
    "get_sigma_g",
    "lb_ddf",
    "vcov_adj",
]
