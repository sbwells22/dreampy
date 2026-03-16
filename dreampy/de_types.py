"""
Data structures for differential expression analysis in dreampy.

This module defines the core containers for:
- DEFit: Fitted model results for one assay (analogous to limma's MArrayLM)
- DEResult: Multi-assay container (ordered mapping from assay name → DEFit)
- FitStatus: Enum for gene-level fit outcomes
- DEControl: Configuration for fitting

These structures bridge the gap between single-gene LMM fitting (fitting.py)
and the downstream eBayes/testing pipeline.
"""

from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Literal

import numpy as np
import pandas as pd

# =============================================================================
# Fit Status Enum
# =============================================================================


class FitStatus(IntEnum):
    """
    Status codes for gene-level model fits.

    Using IntEnum allows direct comparison (status == FitStatus.SUCCESS)
    and array storage as integers.
    """

    SUCCESS = 0  # Converged normally
    FAILED = 1  # Optimization failed
    BOUNDARY = 2  # Converged to boundary (theta ≈ 0)
    SINGULAR = 3  # Design matrix singular after QR
    LOW_DF = 4  # Residual df < 1, unreliable inference
    NO_VARIANCE = 5  # Gene has zero variance

    @classmethod
    def from_lmm_result(cls, converged: bool, hit_boundary: bool = False) -> "FitStatus":
        """Convert LMMResult flags to FitStatus."""
        if not converged:
            return cls.FAILED
        if hit_boundary:
            return cls.BOUNDARY
        return cls.SUCCESS


# =============================================================================
# DEControl: Configuration
# =============================================================================


@dataclass
class DEControl:
    """
    Control parameters for differential expression fitting.

    Analogous to lme4::lmerControl, provides configuration for
    optimization, convergence checking, and input validation.

    Example
    -------
    >>> control = DEControl(optimizer='bobyqa', maxiter=200)
    >>> result = de_assay(counts, formula, metadata, control=control)
    """

    # === Optimizer settings ===
    optimizer: str = "auto"
    # Supported: "auto" (BOBYQA if available, else L-BFGS-B), "bobyqa", "lbfgsb"

    maxiter: int = 100
    # Maximum optimizer iterations

    tol: float = 1e-8
    # Convergence tolerance (rhoend for BOBYQA, ftol/gtol for L-BFGS-B)

    # === Theta bounds (natural space) ===
    theta_lower: float = 0.0
    theta_upper: float = 100.0
    # Box constraints for θ = σ_u / σ_e

    # === Convergence checks ===
    boundary_tol: float = 1e-5
    # Theta values below this are considered "at boundary"

    # === BOBYQA-specific ===
    rhobeg: float | None = None
    # Initial trust region radius. None = auto (0.5 * theta_init)

    rhobeg_scale: float = 0.5
    # Scale factor for automatic rhobeg computation

    # === Input validation ===
    check_rank: str = "warn+drop"
    # How to handle rank-deficient designs: "ignore", "warn+drop", "stop"

    check_scale: str = "warning"
    # How to handle poorly scaled data: "ignore", "warning", "stop"

    check_formula: str = "warn+drop"
    # How to handle problematic formula terms: "ignore", "warn+drop", "stop"

    # === Parallel fitting ===
    n_jobs: int = -1
    # Number of parallel jobs (-1 = all cores)

    backend: str = "loky"
    # Joblib backend: "loky", "threading", "multiprocessing"

    blas_threads: int = 1
    # BLAS threads per worker. MUST be 1 for parallel to avoid oversubscription.

    # === Degrees of freedom ===
    df_eb_method: str = "trace"
    # Method for df_resid_eb: "trace" (LMM), "classical" (n-p for LM)

    ddf: Literal["Satterthwaite", "Kenward-Roger", "adaptive"] | None = "adaptive"
    # Denominator degrees of freedom method for hypothesis tests:
    # - None: Classical df (n - p), no adjustment
    # - "Satterthwaite": df adjustment only
    # - "Kenward-Roger": vcov AND df adjustment (small-sample correction)
    # - "adaptive": KR if n_samples < 20, else Satterthwaite (matches dreamlet, default)

    z_df_threshold: float = 30.0
    # df threshold above which t ≈ z for z_std computation

    # === Diagnostics ===
    diagnostics: bool = False
    # Enable all diagnostics (convenience flag)

    diagnostics_surface: bool = False
    # Evaluate deviance surface (expensive, for debugging)

    diagnostics_path: bool = False
    # Track optimization path (cheap)

    # === Reproducibility ===
    random_seed: int | None = None
    # Seed for deterministic optimizer initialization

    def to_fit_kwargs(self) -> dict[str, Any]:
        """
        Convert to kwargs dict for fit_lmm().

        Returns
        -------
        kwargs : dict
            Parameters compatible with fit_lmm signature
        """
        return {
            "theta_bounds": (self.theta_lower, self.theta_upper),
            "maxiter": self.maxiter,
            "tol": self.tol,
            "optimizer": self.optimizer,
            "diagnostics_surface": self.diagnostics_surface or self.diagnostics,
            "diagnostics_path": self.diagnostics_path or self.diagnostics,
        }

    def to_parallel_kwargs(self) -> dict[str, Any]:
        """
        Convert to kwargs dict for fit_models().

        Returns
        -------
        kwargs : dict
            Parameters for parallel fitting
        """
        return {
            "n_jobs": self.n_jobs,
            "backend": self.backend,
            "diagnostics_surface": self.diagnostics_surface or self.diagnostics,
            "diagnostics_path": self.diagnostics_path or self.diagnostics,
            "ddf": self.ddf,
        }


# =============================================================================
# DEFit: Single-Assay Results Container
# =============================================================================


@dataclass
class DEFit:
    """
    Container for fitted DE models from a single assay/cell type.

    Analogous to limma's MArrayLM class. This object stores all information
    needed for:
    - Empirical Bayes moderation (ebayes)
    - Hypothesis testing
    - Results extraction (top_table)

    Attributes are organized by their role in the statistical pipeline.

    Notes
    -----
    **Two degrees of freedom axes:**
    - df_resid_eb: (n_genes,) scalar per gene, for eBayes variance shrinkage
    - df_resid_test: (n_genes, n_coef) per gene AND coefficient, for p-values

    **Test space:**
    After contrasts_fit(), the object switches from "coef" to "contrast" space.
    This affects the interpretation of coefficients, df, and test statistics.

    **Failed gene handling:**
    Genes that fail to fit are marked in fit_status, NOT dropped. Downstream
    functions (ebayes, top_table) propagate NaN for failed genes.

    Example
    -------
    >>> # After fitting
    >>> fit = de_assay(counts, formula, metadata)
    >>> fit.coefficients.shape  # (n_genes, n_coef)
    >>> fit.sigma.shape  # (n_genes,)
    >>>
    >>> # After eBayes
    >>> fit = ebayes(fit)
    >>> fit.t  # moderated t-statistics
    >>> fit.p_value  # p-values
    """

    # === Identity ===
    gene_names: np.ndarray
    # (n_genes,) gene identifiers (index into rows)

    sample_names: np.ndarray
    # (n_samples,) sample identifiers

    # === Model Specification ===
    formula: str
    # Original formula string

    design_matrix: np.ndarray
    # (n_samples, n_coef) fixed effects design matrix X

    coef_names: list[str]
    # Names of coefficients (columns of design_matrix)

    contrast_matrix: np.ndarray | None = None
    # (n_coef, n_contrasts) contrast matrix L, if contrasts applied

    contrast_names: list[str] | None = None
    # Names of contrasts

    # === Test Space ===
    test_space: str = "coef"
    # "coef" or "contrast" - affects interpretation of coefficients/df

    # === Point Estimates ===
    coefficients: np.ndarray = field(default_factory=lambda: np.array([]))
    # (n_genes, n_coef) or (n_genes, n_contrasts) β estimates

    stdev_unscaled: np.ndarray = field(default_factory=lambda: np.array([]))
    # (n_genes, n_coef) SE(β)/σ (unscaled standard errors)

    sigma: np.ndarray = field(default_factory=lambda: np.array([]))
    # (n_genes,) residual standard deviation σ_e

    # === Variance Components (for LMM) ===
    sigma_u: dict[str, np.ndarray] | None = None
    # Keys are RE names (e.g., "donor", "batch")
    # Values are (n_genes,) arrays of σ_u for that RE

    theta: dict[str, np.ndarray] | None = None
    # θ = σ_u / σ_e ratios, same structure as sigma_u

    # === Degrees of Freedom ===
    df_resid_eb: np.ndarray = field(default_factory=lambda: np.array([]))
    # (n_genes,) df for eBayes shrinkage (trace approximation for LMM)

    df_resid_test: np.ndarray = field(default_factory=lambda: np.array([]))
    # (n_genes, n_coef) df for hypothesis testing (Satterthwaite/KR for LMM)

    df_satterthwaite: dict[str, np.ndarray] | None = None
    # Satterthwaite/KR df per coefficient, keyed by coef name.
    # Each value is (n_genes,) array. When available, populates df_resid_test.

    # === Empirical Bayes Results (populated by ebayes()) ===
    s2_prior: float | None = None
    # Prior variance estimate from fitFDist

    df_prior: float | None = None
    # Prior degrees of freedom

    s2_post: np.ndarray | None = None
    # (n_genes,) posterior variance estimates

    df_total: np.ndarray | None = None
    # (n_genes,) df_prior + df_resid_eb

    var_prior: np.ndarray | None = None
    # (n_genes,) if trend=True, gene-specific prior variances

    # === Test Statistics (populated by ebayes()) ===
    t: np.ndarray | None = None
    # (n_genes, n_coef) moderated t-statistics

    p_value: np.ndarray | None = None
    # (n_genes, n_coef) raw p-values

    z_std: np.ndarray | None = None
    # (n_genes, n_coef) standardized z-scores (for cross-assay comparison)

    lods: np.ndarray | None = None
    # (n_genes, n_coef) log-odds of DE (B statistic)

    F: np.ndarray | None = None
    # (n_genes,) moderated F-statistic (if multiple contrasts)

    F_p_value: np.ndarray | None = None
    # (n_genes,) F-test p-values

    # === Quality Metrics ===
    ave_expr: np.ndarray = field(default_factory=lambda: np.array([]))
    # (n_genes,) average expression (Amean from voom)

    weights: np.ndarray | None = None
    # (n_genes, n_samples) precision weights, if weighted analysis

    # === Fit Status ===
    fit_status: np.ndarray = field(default_factory=lambda: np.array([]))
    # (n_genes,) FitStatus enum values

    fit_messages: dict[str, str] = field(default_factory=dict)
    # gene_name -> error message for failed fits

    # === Diagnostics ===
    diagnostics: dict[str, Any] | None = None
    # gene_name -> OptimizationDiagnostics for LMM genes with successful fits.
    # None for FE-only models. Minimal fields (theta_init_used, theta_optimal,
    # convergence_reason, n_feval, hit_lower_bound, restart_edge_triggered,
    # time_elapsed) are always populated. Expensive fields (surface, path)
    # require diagnostics_surface/diagnostics_path flags in DEControl.

    # === Metadata ===
    method: str = "lmm"
    # "lm" (fixed effects only) or "lmm" (mixed model)

    reml: bool = True
    # REML vs ML estimation

    eb_applied: bool = False
    # Has ebayes() been called?

    assay_name: str | None = None
    # Name of assay/cell type (for multi-assay pipelines)

    ddf_method: str | None = None
    # Denominator degrees of freedom method used:
    # - "Satterthwaite": df adjustment only
    # - "Kenward-Roger": vcov AND df adjustment
    # - "Satterthwaite (KR fallback: ...)": KR failed, fell back to Satterthwaite
    # - None: No df adjustment (classical or LM)

    # === Design Modifications (Transparency) ===
    design_warnings: dict[str, list[str]] = field(default_factory=dict)
    # gene_name -> list of warnings about design modifications
    # e.g., {"GENE1": ["Dropped constant term: batch"]}

    def __post_init__(self):
        """Validate dimensions after initialization."""
        # Allow empty initialization for incremental building
        if len(self.gene_names) == 0:
            return

        n_genes = len(self.gene_names)
        n_samples = len(self.sample_names)
        n_coef = len(self.coef_names)

        # Determine test dimension (coef or contrast space)
        if self.test_space == "contrast" and self.contrast_names is not None:
            n_test = len(self.contrast_names)
        else:
            n_test = n_coef

        # Validate shapes if arrays are populated
        if self.coefficients.size > 0:
            if self.coefficients.shape != (n_genes, n_test):
                raise ValueError(
                    f"coefficients shape {self.coefficients.shape} != ({n_genes}, {n_test})"
                )

        if self.stdev_unscaled.size > 0:
            if self.stdev_unscaled.shape != (n_genes, n_test):
                raise ValueError(
                    f"stdev_unscaled shape {self.stdev_unscaled.shape} != ({n_genes}, {n_test})"
                )

        if self.sigma.size > 0:
            if self.sigma.shape != (n_genes,):
                raise ValueError(f"sigma shape {self.sigma.shape} != ({n_genes},)")

        if self.df_resid_eb.size > 0:
            if self.df_resid_eb.shape != (n_genes,):
                raise ValueError(f"df_resid_eb shape {self.df_resid_eb.shape} != ({n_genes},)")

        if self.df_resid_test.size > 0:
            if self.df_resid_test.shape != (n_genes, n_test):
                raise ValueError(
                    f"df_resid_test shape {self.df_resid_test.shape} != ({n_genes}, {n_test})"
                )

        if self.design_matrix.size > 0:
            if self.design_matrix.shape != (n_samples, n_coef):
                raise ValueError(
                    f"design_matrix shape {self.design_matrix.shape} != ({n_samples}, {n_coef})"
                )

        if self.ave_expr.size > 0:
            if self.ave_expr.shape != (n_genes,):
                raise ValueError(f"ave_expr shape {self.ave_expr.shape} != ({n_genes},)")

        if self.fit_status.size > 0:
            if self.fit_status.shape != (n_genes,):
                raise ValueError(f"fit_status shape {self.fit_status.shape} != ({n_genes},)")

        if self.test_space not in ("coef", "contrast"):
            raise ValueError(f"test_space must be 'coef' or 'contrast', got '{self.test_space}'")

    @property
    def n_genes(self) -> int:
        """Number of genes."""
        return len(self.gene_names)

    @property
    def n_samples(self) -> int:
        """Number of samples."""
        return len(self.sample_names)

    @property
    def n_coef(self) -> int:
        """Number of coefficients in original model."""
        return len(self.coef_names)

    @property
    def n_test(self) -> int:
        """Number of tests (coefficients or contrasts)."""
        if self.test_space == "contrast" and self.contrast_names is not None:
            return len(self.contrast_names)
        return self.n_coef

    def failed_genes(self) -> np.ndarray:
        """Return gene names with failed fits."""
        mask = self.fit_status != FitStatus.SUCCESS
        return self.gene_names[mask]

    def successful_genes(self) -> np.ndarray:
        """Return gene names with successful fits."""
        mask = self.fit_status == FitStatus.SUCCESS
        return self.gene_names[mask]

    def summary(self) -> str:
        """Return summary string."""
        n_success = np.sum(self.fit_status == FitStatus.SUCCESS)
        n_failed = np.sum(self.fit_status == FitStatus.FAILED)
        n_boundary = np.sum(self.fit_status == FitStatus.BOUNDARY)

        lines = [
            f"DEFit: {self.n_genes} genes × {self.n_samples} samples",
            f"  Formula: {self.formula}",
            f"  Method: {self.method} ({'REML' if self.reml else 'ML'})",
            f"  Coefficients: {self.coef_names}",
            f"  Test space: {self.test_space}",
            f"  Fit status: {n_success} success, {n_failed} failed, {n_boundary} boundary",
            f"  eBayes applied: {self.eb_applied}",
        ]

        if self.ddf_method:
            lines.append(f"  DDF method: {self.ddf_method}")

        if self.eb_applied:
            lines.append(f"  Prior df: {self.df_prior:.2f}, Prior s²: {self.s2_prior:.4f}")

        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


# =============================================================================
# DEResult: Multi-Assay Container
# =============================================================================


@dataclass
class DEResult:
    """
    Container for DE results across multiple assays/cell types.

    Provides ordered, dict-like access to DEFit objects while preserving
    insertion order (typically alphabetical by assay name).

    Example
    -------
    >>> result = de_all_assays(pb_list, formula, control=control)
    >>> result['CD4_T']  # Access by name
    >>> for assay, fit in result:  # Iterate
    ...     print(assay, fit.n_genes)
    >>> result.summary()  # Per-assay stats
    """

    fits: dict[str, DEFit] = field(default_factory=dict)
    # Assay name -> DEFit mapping

    assay_order: list[str] = field(default_factory=list)
    # Ordered list of assay names (for reproducible iteration)

    errors_initial: dict[str, str] = field(default_factory=dict)
    # Assay name -> error message for assays that failed entirely

    def __getitem__(self, key: str) -> DEFit:
        """Access DEFit by assay name."""
        return self.fits[key]

    def __setitem__(self, key: str, value: DEFit) -> None:
        """Add/update DEFit for an assay."""
        if key not in self.fits:
            self.assay_order.append(key)
        self.fits[key] = value

    def __contains__(self, key: str) -> bool:
        """Check if assay exists."""
        return key in self.fits

    def __len__(self) -> int:
        """Number of successful assays."""
        return len(self.fits)

    def __iter__(self) -> Iterator[str]:
        """Iterate over assay names in order."""
        return iter(self.assay_order)

    def items(self) -> Iterator[tuple]:
        """Iterate over (assay_name, DEFit) pairs."""
        for assay in self.assay_order:
            yield assay, self.fits[assay]

    def keys(self) -> list[str]:
        """Return assay names in order."""
        return self.assay_order.copy()

    def values(self) -> list[DEFit]:
        """Return DEFit objects in order."""
        return [self.fits[a] for a in self.assay_order]

    def summary(self) -> pd.DataFrame:
        """
        Return summary DataFrame with per-assay statistics.

        Returns
        -------
        df : pd.DataFrame
            Columns: assay, n_genes, n_samples, n_success, n_failed, eb_applied
        """
        rows = []
        for assay in self.assay_order:
            fit = self.fits[assay]
            rows.append(
                {
                    "assay": assay,
                    "n_genes": fit.n_genes,
                    "n_samples": fit.n_samples,
                    "n_success": int(np.sum(fit.fit_status == FitStatus.SUCCESS)),
                    "n_failed": int(np.sum(fit.fit_status != FitStatus.SUCCESS)),
                    "eb_applied": fit.eb_applied,
                }
            )

        return pd.DataFrame(rows)

    def __repr__(self) -> str:
        n_assays = len(self.fits)
        n_errors = len(self.errors_initial)
        assays_str = ", ".join(self.assay_order[:5])
        if n_assays > 5:
            assays_str += f", ... ({n_assays - 5} more)"

        return (
            f"DEResult: {n_assays} assays\n"
            f"  Assays: {assays_str}\n"
            f"  Initial errors: {n_errors} assays failed"
        )


# =============================================================================
# Factory Functions
# =============================================================================


def create_de_fit_from_results(
    gene_results: list[dict],
    gene_names: np.ndarray,
    sample_names: np.ndarray,
    formula: str,
    design_matrix: np.ndarray,
    coef_names: list[str],
    *,
    method: str = "lmm",
    reml: bool = True,
    weights: np.ndarray | None = None,
    assay_name: str | None = None,
) -> DEFit:
    """
    Create DEFit from list of single-gene fit results.

    This factory function assembles the DEFit container from the
    dictionary outputs of fit_lmm() or parallel fitting.

    Parameters
    ----------
    gene_results : List[Dict]
        List of dicts from fit_lmm(), one per gene.
        Each dict has: sigma, Amean, fitted_values, converged, beta, ...
    gene_names : np.ndarray
        Gene identifiers (must match order of gene_results)
    sample_names : np.ndarray
        Sample identifiers
    formula : str
        Original formula string
    design_matrix : np.ndarray
        (n_samples, n_coef) design matrix X
    coef_names : List[str]
        Coefficient names
    method : str
        "lm" or "lmm"
    reml : bool
        REML or ML estimation
    weights : np.ndarray, optional
        (n_genes, n_samples) precision weights
    assay_name : str, optional
        Name of assay/cell type for multi-assay pipelines

    Returns
    -------
    fit : DEFit
        Assembled container ready for ebayes
    """
    n_genes = len(gene_results)
    n_coef = len(coef_names)

    # Allocate arrays
    coefficients = np.zeros((n_genes, n_coef))
    stdev_unscaled = np.zeros((n_genes, n_coef))
    sigma = np.zeros(n_genes)
    ave_expr = np.zeros(n_genes)
    fit_status = np.zeros(n_genes, dtype=np.int32)
    fit_messages = {}
    df_resid_eb = np.zeros(n_genes)

    # For LMM, collect variance components
    sigma_u_dict: dict[str, np.ndarray] = {}
    theta_dict: dict[str, np.ndarray] = {}
    diagnostics_dict: dict[str, Any] = {}

    for i, result in enumerate(gene_results):
        gene = gene_names[i]

        # Extract sigma (residual SD)
        sigma[i] = result.get("sigma", np.nan)
        ave_expr[i] = result.get("Amean", np.nan)

        # Extract coefficients
        beta = result.get("beta", None)
        if beta is not None:
            coefficients[i, : len(beta)] = beta

        # Extract stdev_unscaled (from fitting.py)
        su = result.get("stdev_unscaled", None)
        if su is not None and len(su) > 0:
            # Handle case where su might be shorter than n_coef (shouldn't happen normally)
            n_to_copy = min(len(su), n_coef)
            stdev_unscaled[i, :n_to_copy] = su[:n_to_copy]
            # Fill remaining with NaN if needed
            if n_to_copy < n_coef:
                stdev_unscaled[i, n_to_copy:] = np.nan
        else:
            # No stdev_unscaled from fit - use NaN
            stdev_unscaled[i, :] = np.nan

        # Extract df_resid (from fitting.py)
        df_resid_eb[i] = result.get("df_resid", np.nan)

        # Determine fit status
        converged = result.get("converged", False)
        hit_boundary = result.get("hit_lower_bound", False)

        if not converged or not np.isfinite(sigma[i]) or sigma[i] <= 0:
            fit_status[i] = FitStatus.FAILED
            fit_messages[gene] = result.get("error", "Fit did not converge")
        elif hit_boundary:
            fit_status[i] = FitStatus.BOUNDARY
        else:
            fit_status[i] = FitStatus.SUCCESS

        # Collect per-gene optimization diagnostics (populated for LMM genes,
        # None for FE genes and failed LMM genes)
        diag = result.get("diagnostics", None)
        if diag is not None:
            diagnostics_dict[gene] = diag

        # Collect variance components for LMM
        if method == "lmm":
            sigma_u_val = result.get("sigma_u", 0.0)
            if isinstance(sigma_u_val, dict):
                for re_name, val in sigma_u_val.items():
                    if re_name not in sigma_u_dict:
                        sigma_u_dict[re_name] = np.zeros(n_genes)
                    sigma_u_dict[re_name][i] = val
            elif sigma_u_val > 0:
                # Single RE case
                if "group" not in sigma_u_dict:
                    sigma_u_dict["group"] = np.zeros(n_genes)
                sigma_u_dict["group"][i] = sigma_u_val

            # Collect theta values (same structure as sigma_u)
            theta_val = result.get("theta", None)
            if theta_val is not None:
                if isinstance(theta_val, dict):
                    for re_name, val in theta_val.items():
                        if re_name not in theta_dict:
                            theta_dict[re_name] = np.zeros(n_genes)
                        theta_dict[re_name][i] = val
                elif isinstance(theta_val, (int, float)):
                    # Single RE scalar case
                    if "group" not in theta_dict:
                        theta_dict["group"] = np.zeros(n_genes)
                    theta_dict["group"][i] = theta_val

    # Collect Satterthwaite/KR df from gene results (if computed)
    df_satterthwaite_dict: dict[str, np.ndarray] | None = None
    has_satterthwaite = any(result.get("df_satterthwaite") is not None for result in gene_results)

    if has_satterthwaite:
        df_satterthwaite_dict = {coef_names[j]: np.full(n_genes, np.nan) for j in range(n_coef)}
        for i, result in enumerate(gene_results):
            df_satt = result.get("df_satterthwaite")
            if df_satt is not None:
                for j in range(min(len(df_satt), n_coef)):
                    df_satterthwaite_dict[coef_names[j]][i] = df_satt[j]

    # Collect ddf_method (should be consistent across genes)
    ddf_method = None
    for result in gene_results:
        method_used = result.get("ddf_method")
        if method_used is not None:
            ddf_method = method_used
            break

    # df_resid_test: use Satterthwaite/KR df if available, else replicate df_resid_eb
    if df_satterthwaite_dict is not None:
        # Build (n_genes, n_coef) array from dict
        df_resid_test = np.column_stack([df_satterthwaite_dict[coef] for coef in coef_names])
    else:
        # Fallback: replicate scalar df for all coefficients
        df_resid_test = np.tile(df_resid_eb[:, np.newaxis], (1, n_coef))

    return DEFit(
        gene_names=gene_names,
        sample_names=sample_names,
        formula=formula,
        design_matrix=design_matrix,
        coef_names=coef_names,
        coefficients=coefficients,
        stdev_unscaled=stdev_unscaled,
        sigma=sigma,
        sigma_u=sigma_u_dict if sigma_u_dict else None,
        theta=theta_dict if theta_dict else None,
        df_resid_eb=df_resid_eb,
        df_resid_test=df_resid_test,
        df_satterthwaite=df_satterthwaite_dict,
        ave_expr=ave_expr,
        weights=weights,
        fit_status=fit_status,
        fit_messages=fit_messages,
        diagnostics=diagnostics_dict if diagnostics_dict else None,
        method=method,
        reml=reml,
        assay_name=assay_name,
        ddf_method=ddf_method,
    )


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "DEControl",
    "DEFit",
    "DEResult",
    "FitStatus",
    "create_de_fit_from_results",
]