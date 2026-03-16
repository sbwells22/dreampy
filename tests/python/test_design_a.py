"""
test_design_a.py — End-to-end fixed-effects pipeline with intermediate validation.

Runs the full dreampy FE pipeline from raw counts through final eBayes
results for Design A (~ condition + age, 200 genes × 20 samples). At each
stage, compares dreampy's output against R reference CSVs.

Pipeline order (canonical):
  TMM → filter → log2CPM → voom weights → fit → eBayes

Each stage consumes real dreampy output from the previous stage.
The first failing stage is the diagnostic signal.

Design A specifics:
  - Fixed-effects only (no random effects)
  - 3 coefficients: (Intercept), conditioncontrol, age
  - Reference level: "case" (alphabetically first)
  - 200 genes → 190 after filtering (10 removed: 5 all-zero, 5 borderline)
  - df_resid = 17 (n=20, p=3), uniform across all genes
  - squeezeVar dispatches into legacy (equal-df) mode
"""

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from pathlib import Path
from numpy.testing import assert_allclose

import dreampy as dp

# =============================================================================
# Tolerances — from spec v0.8, per-quantity
# =============================================================================

TOL_TMM = dict(rtol=1e-5)
TOL_LOG2CPM = dict(rtol=1e-6)
TOL_VOOM_WEIGHTS = dict(rtol=1e-4)
TOL_COEF = dict(rtol=1e-5)
TOL_STDEV_UNSCALED = dict(rtol=1e-5)
TOL_SIGMA = dict(rtol=1e-5)
TOL_S2_PRIOR = dict(rtol=1e-4)
TOL_DF_PRIOR = dict(rtol=1e-4)
TOL_S2_POST = dict(rtol=1e-4)
TOL_DF_TOTAL = dict(rtol=1e-4)
TOL_T = dict(rtol=1e-4)
TOL_P = dict(rtol=1e-3, atol=1e-6)
TOL_LODS = dict(rtol=1e-3)

# =============================================================================
# Paths
# =============================================================================

DESIGN_A_DIR = Path(__file__).parent.parent / "data" / "synthetic" / "design_a"

# =============================================================================
# Helpers
# =============================================================================


def assert_allclose_finite(actual, expected, label="", **kwargs):
    """
    assert_allclose on shared-finite elements.
    Assert that NaN/Inf locations are identical in both arrays.

    Parameters
    ----------
    actual, expected : np.ndarray
    label : str
        Descriptive label for error messages.
    **kwargs : passed to np.testing.assert_allclose
    """
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)

    nan_actual = ~np.isfinite(actual)
    nan_expected = ~np.isfinite(expected)
    np.testing.assert_array_equal(
        nan_actual, nan_expected,
        err_msg=f"{label}: NaN/Inf locations differ between Python and R"
    )

    finite = np.isfinite(actual) & np.isfinite(expected)
    if finite.any():
        np.testing.assert_allclose(
            actual[finite], expected[finite],
            err_msg=f"{label}: values differ",
            **kwargs,
        )


def load_design_a_ref(filename, index_col=0):
    """Load an R reference CSV from the Design A directory."""
    return pd.read_csv(DESIGN_A_DIR / filename, index_col=index_col)


# =============================================================================
# Design A: Fixed-Effects Pipeline
# =============================================================================


class TestDesignA:
    """
    End-to-end fixed-effects pipeline:
      counts.csv + metadata.csv
        → TMM → filter → log2CPM → voom → lmFit → eBayes

    Each test method compares one pipeline stage's output against R reference.
    """

    FORMULA = "~ condition + age"
    ASSAY_NAME = "celltype_A"

    # dreampy coef name → R CSV coef name
    # dreampy uses underscore separation, R concatenates level to variable
    COEF_MAP = {
        "condition_control": "conditioncontrol",
    }

    # -----------------------------------------------------------------
    # Fixture: run full pipeline once, store all intermediates
    # -----------------------------------------------------------------

    @pytest.fixture(scope="class")
    def pipeline(self):
        """
        Run the complete Design A pipeline following the canonical workflow.
        Returns a dict of intermediate results for per-stage comparison.
        """
        # --- Load raw data ---
        counts_df = pd.read_csv(DESIGN_A_DIR / "counts.csv", index_col=0)
        metadata_df = pd.read_csv(DESIGN_A_DIR / "metadata.csv")
        metadata_df = metadata_df.set_index("sample_id")

        # Build AnnData: samples × genes
        pb = AnnData(
            X=counts_df.values.T.astype(np.float64),
            obs=metadata_df.loc[counts_df.columns].copy(),
            var=pd.DataFrame(index=counts_df.index),
        )
        # Required fields normally set by aggregate_pseudobulk
        pb.obs["assays"] = self.ASSAY_NAME
        pb.obs["n_cells"] = 100

        results = {}
        results["all_gene_names"] = list(counts_df.index)       # 200 genes
        results["sample_names"] = list(counts_df.columns)        # 20 samples

        # --- Step 2: TMM normalization (all 200 genes) ---
        pb = dp.compute_tmm_factors(pb, assay_col="assays")
        results["norm_factors"] = pb.obs["norm_factors"].values.copy()
        results["lib_size"] = pb.obs["lib_size"].values.copy()

        # --- Step 3: Gene filtering (200 → 190 genes) ---
        assays = dp.filter_by_expr(pb, assay_col="assays")
        pb = assays[self.ASSAY_NAME]
        results["kept_genes"] = list(pb.var_names)

        # --- Step 4: log2CPM ---
        pb = dp.log2cpm(pb, prior_count=0.5)
        results["log2cpm"] = pb.X.copy()

        # --- Step 5: Voom weights ---
        pb = dp.estimate_weights(
            pb, self.FORMULA, reml=False, n_jobs=-1, span=0.5
        )
        results["weights"] = pb.layers["weights"].copy()

        # --- Step 6: Model fitting (FE, explicit Satterthwaite) ---
        fit = dp.fit_models(
            pb, self.FORMULA,
            reml=True,
            n_jobs=-1,
            assay_name=self.ASSAY_NAME,
            ddf="Satterthwaite",
        )
        results["fit"] = fit

        # --- Step 7: eBayes ---
        fit_eb = dp.ebayes(fit)
        results["fit_eb"] = fit_eb

        # --- Metadata for downstream tests ---
        results["pb"] = pb
        results["gene_names"] = list(pb.var_names)               # 190 genes

        return results

    # -----------------------------------------------------------------
    # Shape guardrails (cheap, catch transpose bugs early)
    # -----------------------------------------------------------------

    def test_shape_invariants(self, pipeline):
        """Verify dimensional consistency across pipeline outputs."""
        pb = pipeline["pb"]
        fit = pipeline["fit"]
        fit_eb = pipeline["fit_eb"]
        n_genes = len(pipeline["gene_names"])
        n_samples = len(pipeline["sample_names"])

        # AnnData orientation
        assert pb.X.shape == (n_samples, n_genes)
        assert pb.layers["weights"].shape == pb.X.shape
        assert pb.layers["counts"].shape == pb.X.shape

        # DEFit shapes
        assert fit.coefficients.shape[0] == n_genes
        assert fit.stdev_unscaled.shape == fit.coefficients.shape
        assert fit.sigma.shape == (n_genes,)
        assert len(fit.gene_names) == n_genes
        assert len(fit.sample_names) == n_samples

        # eBayes preserves shapes
        assert fit_eb.t.shape == fit.coefficients.shape
        assert fit_eb.p_value.shape == fit.coefficients.shape
        assert fit_eb.lods.shape == fit.coefficients.shape
        assert fit_eb.s2_post.shape == (n_genes,)
        assert fit_eb.df_total.shape == fit.coefficients.shape

    # -----------------------------------------------------------------
    # Stage: TMM normalization
    # -----------------------------------------------------------------

    def test_tmm_norm_factors(self, pipeline):
        """Verify norm_factors match R's calcNormFactors output."""
        ref = pd.read_csv(DESIGN_A_DIR / "ref_tmm_factors.csv")
        ref = ref.set_index("sample_id")
        samples = pipeline["sample_names"]

        assert_allclose_finite(
            pipeline["norm_factors"],
            ref.loc[samples, "norm_factors"].values,
            label="norm_factors",
            **TOL_TMM,
        )

    def test_tmm_effective_lib_size(self, pipeline):
        """Verify lib_size × norm_factors ≈ R's effective_lib_size."""
        ref = pd.read_csv(DESIGN_A_DIR / "ref_tmm_factors.csv")
        ref = ref.set_index("sample_id")
        samples = pipeline["sample_names"]

        eff_python = pipeline["lib_size"] * pipeline["norm_factors"]
        assert_allclose_finite(
            eff_python,
            ref.loc[samples, "effective_lib_size"].values,
            label="effective_lib_size",
            **TOL_TMM,
        )

    def test_tmm_norm_factors_geometric_mean(self, pipeline):
        """TMM norm_factors should have geometric mean ≈ 1."""
        nf = pipeline["norm_factors"]
        geo_mean = np.exp(np.mean(np.log(nf)))
        assert abs(geo_mean - 1.0) < 1e-6, (
            f"TMM norm_factors geometric mean = {geo_mean}, expected ≈ 1.0"
        )

    # -----------------------------------------------------------------
    # Stage: Gene filtering
    # -----------------------------------------------------------------

    def test_filter_mask(self, pipeline):
        """Verify gene filter mask matches R's filterByExpr exactly."""
        ref = pd.read_csv(DESIGN_A_DIR / "ref_filter_mask.csv")

        kept = set(pipeline["kept_genes"])
        all_genes = pipeline["all_gene_names"]
        mask_python = np.array([g in kept for g in all_genes])

        mask_r = ref["keep"].map(
            {"TRUE": True, "FALSE": False, True: True, False: False}
        ).values.astype(bool)

        np.testing.assert_array_equal(mask_python, mask_r)

    def test_filter_counts(self, pipeline):
        """Verify expected filter counts: 10 removed, 190 kept."""
        assert len(pipeline["kept_genes"]) == 190
        n_filtered = len(pipeline["all_gene_names"]) - len(pipeline["kept_genes"])
        assert n_filtered == 10

    def test_filter_removes_expected_genes(self, pipeline):
        """Verify all-zero and borderline-low genes are filtered."""
        kept = set(pipeline["kept_genes"])
        for i in range(181, 191):
            gene = f"gene_{i:03d}"
            assert gene not in kept, f"{gene} should have been filtered"

    # -----------------------------------------------------------------
    # Stage: log2CPM
    # -----------------------------------------------------------------

    def test_log2cpm(self, pipeline):
        """Verify log2CPM values match R's voom log2CPM output."""
        ref = load_design_a_ref("ref_log2cpm.csv")
        genes = pipeline["gene_names"]
        samples = pipeline["sample_names"]

        # pipeline['log2cpm'] is samples × genes; ref is genes × samples
        python_vals = pipeline["log2cpm"].T
        ref_vals = ref.loc[genes, samples].values

        assert python_vals.shape == ref_vals.shape, (
            f"log2CPM shape mismatch: Python {python_vals.shape} vs R {ref_vals.shape}"
        )
        assert_allclose_finite(
            python_vals, ref_vals, label="log2CPM", **TOL_LOG2CPM
        )

    # -----------------------------------------------------------------
    # Stage: Voom weights
    # -----------------------------------------------------------------

    def test_voom_weights(self, pipeline):
        """Verify voom precision weights match R's voomWithDreamWeights."""
        ref = load_design_a_ref("ref_voom_weights.csv")
        genes = pipeline["gene_names"]
        samples = pipeline["sample_names"]

        # pipeline['weights'] is samples × genes; ref is genes × samples
        python_vals = pipeline["weights"].T
        ref_vals = ref.loc[genes, samples].values

        assert python_vals.shape == ref_vals.shape, (
            f"Voom weights shape mismatch: Python {python_vals.shape} vs R {ref_vals.shape}"
        )
        assert_allclose_finite(
            python_vals, ref_vals, label="voom_weights", **TOL_VOOM_WEIGHTS
        )

    def test_voom_weights_positive(self, pipeline):
        """All voom weights must be strictly positive."""
        assert (pipeline["weights"] > 0).all(), "Found non-positive voom weights"

    # -----------------------------------------------------------------
    # Stage: Model fitting (lmFit)
    # -----------------------------------------------------------------

    def test_lmfit_method(self, pipeline):
        """Design A should use fixed-effects (LM), not mixed model."""
        assert pipeline["fit"].method == "lm"

    def test_lmfit_coef_names(self, pipeline):
        """Verify coefficient names match expected dreampy convention."""
        expected = ["(Intercept)", "condition_control", "age"]
        assert list(pipeline["fit"].coef_names) == expected

    def test_lmfit_coefficients(self, pipeline):
        """Verify fitted coefficients match R's lmFit output."""
        ref = pd.read_csv(DESIGN_A_DIR / "ref_lmfit.csv").set_index("gene")
        fit = pipeline["fit"]
        genes = pipeline["gene_names"]

        for i, coef_name in enumerate(fit.coef_names):
            r_name = self.COEF_MAP.get(coef_name, coef_name)
            col = f"coef_{r_name}"
            assert_allclose_finite(
                fit.coefficients[:, i],
                ref.loc[genes, col].values,
                label=f"coef {coef_name}",
                **TOL_COEF,
            )

    def test_lmfit_stdev_unscaled(self, pipeline):
        """Verify unscaled standard errors match R's lmFit output."""
        ref = pd.read_csv(DESIGN_A_DIR / "ref_lmfit.csv").set_index("gene")
        fit = pipeline["fit"]
        genes = pipeline["gene_names"]

        for i, coef_name in enumerate(fit.coef_names):
            r_name = self.COEF_MAP.get(coef_name, coef_name)
            col = f"stdev_unscaled_{r_name}"
            assert_allclose_finite(
                fit.stdev_unscaled[:, i],
                ref.loc[genes, col].values,
                label=f"stdev_unscaled {coef_name}",
                **TOL_STDEV_UNSCALED,
            )

    def test_lmfit_sigma(self, pipeline):
        """Verify residual standard deviations match R's lmFit output."""
        ref = pd.read_csv(DESIGN_A_DIR / "ref_lmfit.csv").set_index("gene")
        fit = pipeline["fit"]
        genes = pipeline["gene_names"]

        assert_allclose_finite(
            fit.sigma,
            ref.loc[genes, "sigma"].values,
            label="sigma",
            **TOL_SIGMA,
        )

    def test_lmfit_no_failures(self, pipeline):
        """All 190 post-filter genes should fit successfully."""
        from dreampy import FitStatus

        fit = pipeline["fit"]
        failed = fit.fit_status != FitStatus.SUCCESS
        if failed.any():
            failed_genes = np.array(pipeline["gene_names"])[failed]
            pytest.fail(
                f"{failed.sum()} genes failed: {list(failed_genes[:10])}"
            )

    # -----------------------------------------------------------------
    # FE-specific df checks
    # -----------------------------------------------------------------

    def test_lmfit_df_resid_eb_uniform(self, pipeline):
        """
        FE model: df_resid_eb should be identical across all genes (n - p).
        This guarantees squeezeVar dispatches into legacy (equal-df) mode.
        """
        fit = pipeline["fit"]
        df = fit.df_resid_eb
        assert df.min() == df.max(), (
            f"FE df_resid_eb should be uniform: min={df.min()}, max={df.max()}"
        )
        # n=20 samples, p=3 coefficients → df = 17
        assert df[0] == 17.0, f"Expected df=17, got {df[0]}"

    def test_lmfit_df_resid_test_uniform(self, pipeline):
        """
        FE model: df_resid_test columns should all equal df_resid_eb.
        No Satterthwaite/KR adjustment for fixed-effects.
        """
        fit = pipeline["fit"]
        for j in range(fit.df_resid_test.shape[1]):
            np.testing.assert_array_equal(
                fit.df_resid_test[:, j],
                fit.df_resid_eb,
                err_msg=f"df_resid_test[:,{j}] != df_resid_eb",
            )

    # -----------------------------------------------------------------
    # Stage: eBayes
    # -----------------------------------------------------------------

    def test_ebayes_returns_new_fit(self, pipeline):
        """ebayes() must return a new DEFit, not modify the original."""
        fit = pipeline["fit"]
        fit_eb = pipeline["fit_eb"]
        assert fit is not fit_eb
        assert not fit.eb_applied
        assert fit_eb.eb_applied

    def test_ebayes_s2_prior(self, pipeline):
        """Verify prior variance scalar matches R."""
        ref = pd.read_csv(DESIGN_A_DIR / "ref_ebayes.csv")
        fit_eb = pipeline["fit_eb"]

        r_s2_prior = ref["s2_prior"].iloc[0]
        assert_allclose_finite(
            np.array([fit_eb.s2_prior]),
            np.array([r_s2_prior]),
            label="s2_prior",
            **TOL_S2_PRIOR,
        )

    def test_ebayes_df_prior(self, pipeline):
        """Verify prior degrees of freedom scalar matches R."""
        ref = pd.read_csv(DESIGN_A_DIR / "ref_ebayes.csv")
        fit_eb = pipeline["fit_eb"]

        r_df_prior = ref["df_prior"].iloc[0]
        assert_allclose_finite(
            np.array([fit_eb.df_prior]),
            np.array([r_df_prior]),
            label="df_prior",
            **TOL_DF_PRIOR,
        )

    def test_ebayes_s2_post(self, pipeline):
        """Verify posterior variance per gene matches R."""
        ref = pd.read_csv(DESIGN_A_DIR / "ref_ebayes.csv").set_index("gene")
        fit_eb = pipeline["fit_eb"]
        genes = pipeline["gene_names"]

        assert_allclose_finite(
            fit_eb.s2_post,
            ref.loc[genes, "s2_post"].values,
            label="s2_post",
            **TOL_S2_POST,
        )

    def test_ebayes_df_total(self, pipeline):
        """
        Verify df_total matches R.
        Design A: df_total is constant across genes and coefficients
        (df_prior + df_resid_eb, where df_resid_eb = n - p for all genes).
        """
        ref = pd.read_csv(DESIGN_A_DIR / "ref_ebayes.csv").set_index("gene")
        fit_eb = pipeline["fit_eb"]
        genes = pipeline["gene_names"]

        r_df_total = ref.loc[genes, "df_total"].values

        # df_total should be uniform for Design A
        assert ref["df_total"].nunique() == 1, (
            f"R df_total not uniform: {ref['df_total'].unique()}"
        )

        for j in range(fit_eb.df_total.shape[1]):
            assert_allclose_finite(
                fit_eb.df_total[:, j],
                r_df_total,
                label=f"df_total coef {j}",
                **TOL_DF_TOTAL,
            )

    def test_ebayes_t_statistics(self, pipeline):
        """Verify moderated t-statistics match R."""
        ref = pd.read_csv(DESIGN_A_DIR / "ref_ebayes.csv").set_index("gene")
        fit_eb = pipeline["fit_eb"]
        genes = pipeline["gene_names"]

        for i, coef_name in enumerate(fit_eb.coef_names):
            r_name = self.COEF_MAP.get(coef_name, coef_name)
            col = f"t_{r_name}"
            assert_allclose_finite(
                fit_eb.t[:, i],
                ref.loc[genes, col].values,
                label=f"t {coef_name}",
                **TOL_T,
            )

    def test_ebayes_p_values(self, pipeline):
        """Verify raw p-values match R."""
        ref = pd.read_csv(DESIGN_A_DIR / "ref_ebayes.csv").set_index("gene")
        fit_eb = pipeline["fit_eb"]
        genes = pipeline["gene_names"]

        for i, coef_name in enumerate(fit_eb.coef_names):
            r_name = self.COEF_MAP.get(coef_name, coef_name)
            col = f"p_{r_name}"
            assert_allclose_finite(
                fit_eb.p_value[:, i],
                ref.loc[genes, col].values,
                label=f"p {coef_name}",
                **TOL_P,
            )

    def test_ebayes_lods(self, pipeline):
        """Verify log-odds of DE (B statistic) match R."""
        ref = pd.read_csv(DESIGN_A_DIR / "ref_ebayes.csv").set_index("gene")
        fit_eb = pipeline["fit_eb"]
        genes = pipeline["gene_names"]

        for i, coef_name in enumerate(fit_eb.coef_names):
            r_name = self.COEF_MAP.get(coef_name, coef_name)
            col = f"b_{r_name}"
            assert_allclose_finite(
                fit_eb.lods[:, i],
                ref.loc[genes, col].values,
                label=f"lods {coef_name}",
                **TOL_LODS,
            )

    # -----------------------------------------------------------------
    # Statistical sanity checks (not tolerance-gated)
    # -----------------------------------------------------------------

    def test_sanity_effect_genes_larger_t(self, pipeline):
        """
        Effect genes should have larger |t| for condition_control
        than null genes on average.
        """
        manifest = pd.read_csv(DESIGN_A_DIR / "gene_manifest.csv")
        fit_eb = pipeline["fit_eb"]
        genes = pipeline["gene_names"]

        coef_idx = list(fit_eb.coef_names).index("condition_control")
        t_vals = pd.Series(
            np.abs(fit_eb.t[:, coef_idx]),
            index=genes,
            name="abs_t",
        )

        manifest_filt = manifest[manifest["gene"].isin(genes)].set_index("gene")
        effect_t = t_vals[manifest_filt["has_condition_effect"] == True].mean()
        null_t = t_vals[manifest_filt["has_condition_effect"] == False].mean()

        assert effect_t > null_t, (
            f"Effect genes mean |t| ({effect_t:.3f}) should exceed "
            f"null genes mean |t| ({null_t:.3f})"
        )

    def test_sanity_null_moderate_not_significant(self, pipeline):
        """null_moderate genes (191–195) should not be significant."""
        ref = pd.read_csv(DESIGN_A_DIR / "ref_ebayes.csv").set_index("gene")

        for i in range(191, 196):
            gene = f"gene_{i:03d}"
            if gene in ref.index:
                adj_p = ref.loc[gene, "adj_p_conditioncontrol"]
                assert adj_p > 0.05, (
                    f"{gene} (null_moderate) has adj_p={adj_p:.4f}, "
                    f"expected > 0.05"
                )