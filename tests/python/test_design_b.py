"""
test_design_b.py — End-to-end LMM pipeline with intermediate validation.

Runs the full dreampy mixed-effects pipeline from raw counts through final
eBayes results for Design B (~ condition + (1|collection_site), 250 genes × 24
samples). Two fitting passes: Satterthwaite and Kenward-Roger.

Pipeline order (canonical):
  TMM → filter → log2CPM → voom weights → fit (Satt) → eBayes (Satt)
                                          → fit (KR)   → eBayes (KR)

Design B specifics:
  - Single random effect: (1|collection_site), 4 sites
  - 2 coefficients: (Intercept), conditioncontrol
  - Reference level: "case" (alphabetically first)
  - 250 genes → 240 after filtering (5 allzero + 5 low-expression)
  - df_resid_eb varies per gene (trace-based rdf, range 19–22)
  - df_resid_test varies per coefficient (Satterthwaite denominator df)
  - KR: stdev_unscaled and df differ from Satterthwaite; coefficients and sigma identical

Edge-case gene categories:
  - boundary_theta (226–230): true site σ=0. Genes 226, 230 converge to θ=0 exactly.
  - high_site_var (231–235): true site σ=2. Large θ, rdf near 19.
  - site_condition_corr (236–240): site effect correlated with condition.
"""

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from pathlib import Path
from numpy.testing import assert_allclose

import dreampy as dp

# =============================================================================
# Tolerances — from spec v0.9, per-quantity
# =============================================================================

TOL_TMM = dict(rtol=1e-5)
TOL_LOG2CPM = dict(rtol=1e-6)
TOL_VOOM_WEIGHTS = dict(rtol=1e-4)
TOL_COEF = dict(rtol=1e-5)
TOL_STDEV_UNSCALED = dict(rtol=1e-5)
TOL_SIGMA = dict(rtol=1e-5)
TOL_THETA = dict(atol=1e-4)
TOL_RDF = dict(rtol=1e-4)
TOL_SATT_DF = dict(rtol=1e-3)
TOL_KR_DF = dict(rtol=1e-3)
TOL_S2_PRIOR = dict(rtol=1e-4)
TOL_DF_PRIOR = dict(rtol=1e-4)
TOL_S2_POST = dict(rtol=1e-4)
TOL_DF_TOTAL = dict(rtol=1e-4)
TOL_T = dict(rtol=1e-4)
TOL_P = dict(rtol=1e-3, atol=1e-6)
TOL_LODS = dict(rtol=1.5e-3)

# =============================================================================
# Paths
# =============================================================================

DESIGN_B_DIR = Path(__file__).parent.parent / "data" / "synthetic" / "design_b"

# =============================================================================
# Helpers
# =============================================================================


def assert_allclose_finite(actual, expected, label="", **kwargs):
    """
    assert_allclose on shared-finite elements.
    Assert that NaN/Inf locations are identical in both arrays.
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


def load_design_b_ref(filename, index_col=0):
    """Load an R reference CSV from the Design B directory."""
    return pd.read_csv(DESIGN_B_DIR / filename, index_col=index_col)


# =============================================================================
# Design B: Mixed-Effects Pipeline
# =============================================================================


class TestDesignB:
    """
    End-to-end LMM pipeline with Satterthwaite and Kenward-Roger passes.
    """

    FORMULA = "~ condition + (1|collection_site)"
    ASSAY_NAME = "celltype_A"

    # dreampy coef name → R CSV coef name
    COEF_MAP = {
        "condition_control": "conditioncontrol",
    }

    # RE key in dreampy's theta dict — may need adjustment if wrong
    RE_KEY = "collection_site"

    # -----------------------------------------------------------------
    # Fixture: run full pipeline once (both Satt and KR)
    # -----------------------------------------------------------------

    @pytest.fixture(scope="class")
    def pipeline(self):
        """
        Run the complete Design B pipeline: TMM → filter → log2CPM → voom →
        fit (Satterthwaite) → eBayes → fit (KR) → eBayes.

        Returns a dict of intermediate results for per-stage comparison.
        """
        # --- Load raw data ---
        counts_df = pd.read_csv(DESIGN_B_DIR / "counts.csv", index_col=0)
        metadata_df = pd.read_csv(DESIGN_B_DIR / "metadata.csv")
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
        results["all_gene_names"] = list(counts_df.index)       # 250 genes
        results["sample_names"] = list(counts_df.columns)        # 24 samples

        # --- Step 2: TMM normalization (all 250 genes) ---
        pb = dp.compute_tmm_factors(pb, assay_col="assays")
        results["norm_factors"] = pb.obs["norm_factors"].values.copy()
        results["lib_size"] = pb.obs["lib_size"].values.copy()

        # --- Step 3: Gene filtering (250 → 240 genes) ---
        assays = dp.filter_by_expr(pb, assay_col="assays")
        pb = assays[self.ASSAY_NAME]
        results["kept_genes"] = list(pb.var_names)

        # --- Step 4: log2CPM ---
        pb = dp.log2cpm(pb, prior_count=0.5)
        results["log2cpm"] = pb.X.copy()

        # --- Step 5: Voom weights ---
        pb = dp.estimate_weights(
            pb, self.FORMULA, reml=False, n_jobs=-1,
        )
        results["weights"] = pb.layers["weights"].copy()

        # --- Step 6a: Satterthwaite fit ---
        fit_satt = dp.fit_models(
            pb, self.FORMULA,
            reml=True,
            n_jobs=-1,
            assay_name=self.ASSAY_NAME,
            ddf="Satterthwaite",
        )
        results["fit_satt"] = fit_satt

        # --- Step 7a: eBayes (Satterthwaite) ---
        fit_satt_eb = dp.ebayes(fit_satt)
        results["fit_satt_eb"] = fit_satt_eb

        # --- Step 6b: Kenward-Roger fit ---
        fit_kr = dp.fit_models(
            pb, self.FORMULA,
            reml=True,
            n_jobs=-1,
            assay_name=self.ASSAY_NAME,
            ddf="Kenward-Roger",
        )
        results["fit_kr"] = fit_kr

        # --- Step 7b: eBayes (KR) ---
        fit_kr_eb = dp.ebayes(fit_kr)
        results["fit_kr_eb"] = fit_kr_eb

        # --- Metadata for downstream tests ---
        results["pb"] = pb
        results["gene_names"] = list(pb.var_names)               # 240 genes

        return results

    # =================================================================
    # Shape guardrails
    # =================================================================

    def test_shape_invariants(self, pipeline):
        """Verify dimensional consistency across pipeline outputs."""
        pb = pipeline["pb"]
        fit = pipeline["fit_satt"]
        fit_eb = pipeline["fit_satt_eb"]
        n_genes = len(pipeline["gene_names"])
        n_samples = len(pipeline["sample_names"])

        assert pb.X.shape == (n_samples, n_genes)
        assert pb.layers["weights"].shape == pb.X.shape
        assert pb.layers["counts"].shape == pb.X.shape

        assert fit.coefficients.shape[0] == n_genes
        assert fit.stdev_unscaled.shape == fit.coefficients.shape
        assert fit.sigma.shape == (n_genes,)
        assert len(fit.gene_names) == n_genes
        assert len(fit.sample_names) == n_samples

        assert fit_eb.t.shape == fit.coefficients.shape
        assert fit_eb.p_value.shape == fit.coefficients.shape
        assert fit_eb.lods.shape == fit.coefficients.shape
        assert fit_eb.s2_post.shape == (n_genes,)
        assert fit_eb.df_total.shape == fit.coefficients.shape

    # =================================================================
    # TMM normalization
    # =================================================================

    def test_tmm_norm_factors(self, pipeline):
        """Verify norm_factors match R."""
        ref = pd.read_csv(DESIGN_B_DIR / "ref_tmm_factors.csv")
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
        ref = pd.read_csv(DESIGN_B_DIR / "ref_tmm_factors.csv")
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

    # =================================================================
    # Gene filtering
    # =================================================================

    def test_filter_mask(self, pipeline):
        """Verify gene filter mask matches R's filterByExpr exactly."""
        ref = pd.read_csv(DESIGN_B_DIR / "ref_filter_mask.csv")

        kept = set(pipeline["kept_genes"])
        all_genes = pipeline["all_gene_names"]
        mask_python = np.array([g in kept for g in all_genes])

        mask_r = ref["keep"].map(
            {"TRUE": True, "FALSE": False, True: True, False: False}
        ).values.astype(bool)

        np.testing.assert_array_equal(mask_python, mask_r)

    def test_filter_counts(self, pipeline):
        """Verify expected filter counts: 10 removed, 240 kept."""
        assert len(pipeline["kept_genes"]) == 240
        n_filtered = len(pipeline["all_gene_names"]) - len(pipeline["kept_genes"])
        assert n_filtered == 10

    def test_filter_removes_allzero_genes(self, pipeline):
        """All-zero genes (221–225) must be filtered."""
        kept = set(pipeline["kept_genes"])
        for i in range(221, 226):
            gene = f"gene_{i:03d}"
            assert gene not in kept, f"{gene} (allzero) should have been filtered"

    def test_filter_preserves_edge_cases(self, pipeline):
        """boundary_theta, high_site_var, site_condition_corr genes must survive."""
        kept = set(pipeline["kept_genes"])
        for i in range(226, 251):  # 226–250 covers all three edge categories + padding
            gene = f"gene_{i:03d}"
            assert gene in kept, f"{gene} should have survived filtering"

    # =================================================================
    # log2CPM
    # =================================================================

    def test_log2cpm(self, pipeline):
        """Verify log2CPM values match R."""
        ref = load_design_b_ref("ref_log2cpm.csv")
        genes = pipeline["gene_names"]
        samples = pipeline["sample_names"]

        python_vals = pipeline["log2cpm"].T  # samples×genes → genes×samples
        ref_vals = ref.loc[genes, samples].values

        assert python_vals.shape == ref_vals.shape, (
            f"log2CPM shape mismatch: Python {python_vals.shape} vs R {ref_vals.shape}"
        )
        assert_allclose_finite(
            python_vals, ref_vals, label="log2CPM", **TOL_LOG2CPM
        )

    # =================================================================
    # Voom weights
    # =================================================================

    def test_voom_weights(self, pipeline):
        """Verify voom precision weights match R."""
        ref = load_design_b_ref("ref_voom_weights.csv")
        genes = pipeline["gene_names"]
        samples = pipeline["sample_names"]

        python_vals = pipeline["weights"].T  # samples×genes → genes×samples
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

    # =================================================================
    # Satterthwaite fit (lmer)
    # =================================================================

    def test_satt_method(self, pipeline):
        """Design B should use mixed model (LMM)."""
        assert pipeline["fit_satt"].method == "lmm"

    def test_satt_coef_names(self, pipeline):
        """Verify coefficient names match expected dreampy convention."""
        expected = ["(Intercept)", "condition_control"]
        assert list(pipeline["fit_satt"].coef_names) == expected

    def test_satt_coefficients(self, pipeline):
        """Verify fitted coefficients match R's dreamlet output."""
        ref = pd.read_csv(DESIGN_B_DIR / "ref_lmer.csv").set_index("gene")
        fit = pipeline["fit_satt"]
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

    def test_satt_stdev_unscaled(self, pipeline):
        """Verify unscaled standard errors match R."""
        ref = pd.read_csv(DESIGN_B_DIR / "ref_lmer.csv").set_index("gene")
        fit = pipeline["fit_satt"]
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

    def test_satt_sigma(self, pipeline):
        """Verify residual standard deviations match R."""
        ref = pd.read_csv(DESIGN_B_DIR / "ref_lmer.csv").set_index("gene")
        fit = pipeline["fit_satt"]
        genes = pipeline["gene_names"]

        assert_allclose_finite(
            fit.sigma,
            ref.loc[genes, "sigma"].values,
            label="sigma",
            **TOL_SIGMA,
        )

    def test_satt_no_failures(self, pipeline):
        """All 240 post-filter genes should fit successfully."""
        from dreampy import FitStatus

        fit = pipeline["fit_satt"]
        failed = fit.fit_status != FitStatus.SUCCESS
        # Allow BOUNDARY (θ=0) as success for boundary_theta genes
        boundary = fit.fit_status == FitStatus.BOUNDARY
        truly_failed = failed & ~boundary
        if truly_failed.any():
            failed_genes = np.array(pipeline["gene_names"])[truly_failed]
            pytest.fail(
                f"{truly_failed.sum()} genes failed: {list(failed_genes[:10])}"
            )

    # -----------------------------------------------------------------
    # Satterthwaite: theta (variance components)
    # -----------------------------------------------------------------

    def test_satt_theta(self, pipeline):
        """
        Verify θ = σ_u / σ_e matches R's varComp.
        R exports varComp = σ²_u. θ = sqrt(varComp) / sigma.
        """
        ref = pd.read_csv(DESIGN_B_DIR / "ref_lmer.csv").set_index("gene")
        fit = pipeline["fit_satt"]
        genes = pipeline["gene_names"]

        r_varcomp = ref.loc[genes, "varComp_collection_site"].values
        r_sigma = ref.loc[genes, "sigma"].values

        # Derive θ from R's varComp and sigma
        # Where varComp=0, theta=0. Where sigma=0, theta undefined — but no NaN in data.
        r_theta = np.sqrt(r_varcomp) / r_sigma

        assert_allclose_finite(
            fit.theta[self.RE_KEY],
            r_theta,
            label="theta_collection_site",
            **TOL_THETA,
        )

    # -----------------------------------------------------------------
    # Satterthwaite: degrees of freedom
    # -----------------------------------------------------------------

    def test_satt_df_resid_eb(self, pipeline):
        """
        Verify trace-based rdf matches R.
        LMM: rdf varies per gene (not uniform like FE).
        """
        ref = pd.read_csv(DESIGN_B_DIR / "ref_lmer.csv").set_index("gene")
        fit = pipeline["fit_satt"]
        genes = pipeline["gene_names"]

        assert_allclose_finite(
            fit.df_resid_eb,
            ref.loc[genes, "rdf"].values,
            label="df_resid_eb (rdf)",
            **TOL_RDF,
        )

    def test_satt_df_resid_eb_not_uniform(self, pipeline):
        """
        LMM df_resid_eb should vary across genes (unlike FE).
        Genes with θ=0 get rdf=22 (FE equivalent), others get less.
        """
        fit = pipeline["fit_satt"]
        assert fit.df_resid_eb.min() < fit.df_resid_eb.max(), (
            "LMM df_resid_eb should vary across genes"
        )

    def test_satt_df_resid_test(self, pipeline):
        """
        Verify per-coefficient Satterthwaite denominator df.
        These vary per gene AND per coefficient.
        """
        ref = pd.read_csv(DESIGN_B_DIR / "ref_satterthwaite.csv").set_index("gene")
        fit = pipeline["fit_satt"]
        genes = pipeline["gene_names"]

        for i, coef_name in enumerate(fit.coef_names):
            r_name = self.COEF_MAP.get(coef_name, coef_name)
            col = f"df_residual_{r_name}"
            assert_allclose_finite(
                fit.df_resid_test[:, i],
                ref.loc[genes, col].values,
                label=f"df_resid_test {coef_name}",
                **TOL_SATT_DF,
            )

    def test_satt_df_intercept_smaller_than_condition(self, pipeline):
        """
        Intercept Satterthwaite df should generally be smaller than
        condition df, because sites absorb intercept variation.
        """
        ref = pd.read_csv(DESIGN_B_DIR / "ref_satterthwaite.csv").set_index("gene")
        genes = pipeline["gene_names"]

        df_int = ref.loc[genes, "df_residual_(Intercept)"].values
        df_cond = ref.loc[genes, "df_residual_conditioncontrol"].values

        # On average, intercept df should be smaller
        assert df_int.mean() < df_cond.mean(), (
            f"Intercept mean df ({df_int.mean():.2f}) should be < "
            f"condition mean df ({df_cond.mean():.2f})"
        )

    # =================================================================
    # Satterthwaite: eBayes
    # =================================================================

    def test_satt_ebayes_returns_new_fit(self, pipeline):
        """ebayes() must return a new DEFit, not modify the original."""
        assert pipeline["fit_satt"] is not pipeline["fit_satt_eb"]
        assert not pipeline["fit_satt"].eb_applied
        assert pipeline["fit_satt_eb"].eb_applied

    def test_satt_ebayes_s2_prior(self, pipeline):
        """Verify prior variance scalar matches R."""
        ref = pd.read_csv(DESIGN_B_DIR / "ref_ebayes.csv")
        fit_eb = pipeline["fit_satt_eb"]

        assert_allclose_finite(
            np.array([fit_eb.s2_prior]),
            np.array([ref["s2_prior"].iloc[0]]),
            label="s2_prior",
            **TOL_S2_PRIOR,
        )

    def test_satt_ebayes_df_prior(self, pipeline):
        """Verify prior degrees of freedom scalar matches R."""
        ref = pd.read_csv(DESIGN_B_DIR / "ref_ebayes.csv")
        fit_eb = pipeline["fit_satt_eb"]

        assert_allclose_finite(
            np.array([fit_eb.df_prior]),
            np.array([ref["df_prior"].iloc[0]]),
            label="df_prior",
            **TOL_DF_PRIOR,
        )

    def test_satt_ebayes_s2_post(self, pipeline):
        """Verify posterior variance per gene matches R."""
        ref = pd.read_csv(DESIGN_B_DIR / "ref_ebayes.csv").set_index("gene")
        fit_eb = pipeline["fit_satt_eb"]
        genes = pipeline["gene_names"]

        assert_allclose_finite(
            fit_eb.s2_post,
            ref.loc[genes, "s2_post"].values,
            label="s2_post",
            **TOL_S2_POST,
        )

    def test_satt_ebayes_df_total(self, pipeline):
        """
        Verify df_total per coefficient.
        LMM: df_total varies per gene AND per coefficient.
        """
        ref = pd.read_csv(DESIGN_B_DIR / "ref_ebayes.csv").set_index("gene")
        fit_eb = pipeline["fit_satt_eb"]
        genes = pipeline["gene_names"]

        for i, coef_name in enumerate(fit_eb.coef_names):
            r_name = self.COEF_MAP.get(coef_name, coef_name)
            col = f"df_total_{r_name}"
            assert_allclose_finite(
                fit_eb.df_total[:, i],
                ref.loc[genes, col].values,
                label=f"df_total {coef_name}",
                **TOL_DF_TOTAL,
            )

    def test_satt_ebayes_t_statistics(self, pipeline):
        """Verify moderated t-statistics match R."""
        ref = pd.read_csv(DESIGN_B_DIR / "ref_ebayes.csv").set_index("gene")
        fit_eb = pipeline["fit_satt_eb"]
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

    def test_satt_ebayes_p_values(self, pipeline):
        """Verify raw p-values match R."""
        ref = pd.read_csv(DESIGN_B_DIR / "ref_ebayes.csv").set_index("gene")
        fit_eb = pipeline["fit_satt_eb"]
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

    def test_satt_ebayes_lods(self, pipeline):
        """Verify log-odds of DE (B statistic) match R."""
        ref = pd.read_csv(DESIGN_B_DIR / "ref_ebayes.csv").set_index("gene")
        fit_eb = pipeline["fit_satt_eb"]
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

    # =================================================================
    # Kenward-Roger: fit
    # =================================================================

    def test_kr_method(self, pipeline):
        """KR pass should also be LMM."""
        assert pipeline["fit_kr"].method == "lmm"

    def test_kr_coefficients_match_satt(self, pipeline):
        """KR does NOT change point estimates — coefficients must be identical."""
        fit_satt = pipeline["fit_satt"]
        fit_kr = pipeline["fit_kr"]

        np.testing.assert_array_equal(
            fit_satt.coefficients, fit_kr.coefficients,
            err_msg="KR coefficients should be identical to Satterthwaite"
        )

    def test_kr_sigma_matches_satt(self, pipeline):
        """KR does NOT change residual SD — sigma must be identical."""
        fit_satt = pipeline["fit_satt"]
        fit_kr = pipeline["fit_kr"]

        np.testing.assert_array_equal(
            fit_satt.sigma, fit_kr.sigma,
            err_msg="KR sigma should be identical to Satterthwaite"
        )

    def test_kr_stdev_unscaled_differs_from_satt(self, pipeline):
        """KR adjusts vcov → stdev_unscaled must differ from Satterthwaite."""
        fit_satt = pipeline["fit_satt"]
        fit_kr = pipeline["fit_kr"]

        # At least some genes should have different stdev_unscaled
        diff = np.abs(fit_satt.stdev_unscaled - fit_kr.stdev_unscaled)
        assert diff.max() > 1e-6, (
            "KR stdev_unscaled should differ from Satterthwaite — "
            "KR may not be engaged"
        )

    def test_kr_stdev_unscaled(self, pipeline):
        """Verify KR stdev_unscaled matches R."""
        ref = pd.read_csv(DESIGN_B_DIR / "ref_lmer_kr.csv").set_index("gene")
        fit_kr = pipeline["fit_kr"]
        genes = pipeline["gene_names"]

        for i, coef_name in enumerate(fit_kr.coef_names):
            r_name = self.COEF_MAP.get(coef_name, coef_name)
            col = f"stdev_unscaled_{r_name}"
            assert_allclose_finite(
                fit_kr.stdev_unscaled[:, i],
                ref.loc[genes, col].values,
                label=f"KR stdev_unscaled {coef_name}",
                **TOL_STDEV_UNSCALED,
            )

    def test_kr_df_resid_eb_matches_satt(self, pipeline):
        """KR does NOT change trace-based rdf (used by squeezeVar)."""
        fit_satt = pipeline["fit_satt"]
        fit_kr = pipeline["fit_kr"]

        np.testing.assert_array_equal(
            fit_satt.df_resid_eb, fit_kr.df_resid_eb,
            err_msg="KR rdf should be identical to Satterthwaite"
        )

    def test_kr_df_resid_test(self, pipeline):
        """Verify per-coefficient KR denominator df."""
        ref = pd.read_csv(DESIGN_B_DIR / "ref_kr_df.csv").set_index("gene")
        fit_kr = pipeline["fit_kr"]
        genes = pipeline["gene_names"]

        for i, coef_name in enumerate(fit_kr.coef_names):
            r_name = self.COEF_MAP.get(coef_name, coef_name)
            col = f"df_residual_{r_name}"
            assert_allclose_finite(
                fit_kr.df_resid_test[:, i],
                ref.loc[genes, col].values,
                label=f"KR df_resid_test {coef_name}",
                **TOL_KR_DF,
            )

    def test_kr_df_more_conservative(self, pipeline):
        """KR df should generally be ≤ Satterthwaite df (more conservative)."""
        fit_satt = pipeline["fit_satt"]
        fit_kr = pipeline["fit_kr"]

        # On average, KR should be more conservative
        assert fit_kr.df_resid_test.mean() <= fit_satt.df_resid_test.mean(), (
            f"KR mean df ({fit_kr.df_resid_test.mean():.2f}) should be ≤ "
            f"Satterthwaite mean df ({fit_satt.df_resid_test.mean():.2f})"
        )

    # =================================================================
    # Kenward-Roger: eBayes
    # =================================================================

    def test_kr_ebayes_s2_prior_matches_satt(self, pipeline):
        """s2_prior should be identical between Satt and KR (same rdf)."""
        fit_satt_eb = pipeline["fit_satt_eb"]
        fit_kr_eb = pipeline["fit_kr_eb"]

        assert fit_satt_eb.s2_prior == fit_kr_eb.s2_prior, (
            f"s2_prior: Satt={fit_satt_eb.s2_prior}, KR={fit_kr_eb.s2_prior}"
        )

    def test_kr_ebayes_df_prior_matches_satt(self, pipeline):
        """df_prior should be identical between Satt and KR (same rdf)."""
        fit_satt_eb = pipeline["fit_satt_eb"]
        fit_kr_eb = pipeline["fit_kr_eb"]

        assert fit_satt_eb.df_prior == fit_kr_eb.df_prior, (
            f"df_prior: Satt={fit_satt_eb.df_prior}, KR={fit_kr_eb.df_prior}"
        )

    def test_kr_ebayes_t_statistics(self, pipeline):
        """Verify KR moderated t-statistics match R."""
        ref = pd.read_csv(DESIGN_B_DIR / "ref_ebayes_kr.csv").set_index("gene")
        fit_kr_eb = pipeline["fit_kr_eb"]
        genes = pipeline["gene_names"]

        for i, coef_name in enumerate(fit_kr_eb.coef_names):
            r_name = self.COEF_MAP.get(coef_name, coef_name)
            col = f"t_{r_name}"
            assert_allclose_finite(
                fit_kr_eb.t[:, i],
                ref.loc[genes, col].values,
                label=f"KR t {coef_name}",
                **TOL_T,
            )

    def test_kr_ebayes_p_values(self, pipeline):
        """Verify KR raw p-values match R."""
        ref = pd.read_csv(DESIGN_B_DIR / "ref_ebayes_kr.csv").set_index("gene")
        fit_kr_eb = pipeline["fit_kr_eb"]
        genes = pipeline["gene_names"]

        for i, coef_name in enumerate(fit_kr_eb.coef_names):
            r_name = self.COEF_MAP.get(coef_name, coef_name)
            col = f"p_{r_name}"
            assert_allclose_finite(
                fit_kr_eb.p_value[:, i],
                ref.loc[genes, col].values,
                label=f"KR p {coef_name}",
                **TOL_P,
            )

    def test_kr_ebayes_lods(self, pipeline):
        """Verify KR log-odds of DE match R."""
        ref = pd.read_csv(DESIGN_B_DIR / "ref_ebayes_kr.csv").set_index("gene")
        fit_kr_eb = pipeline["fit_kr_eb"]
        genes = pipeline["gene_names"]

        for i, coef_name in enumerate(fit_kr_eb.coef_names):
            r_name = self.COEF_MAP.get(coef_name, coef_name)
            col = f"b_{r_name}"
            assert_allclose_finite(
                fit_kr_eb.lods[:, i],
                ref.loc[genes, col].values,
                label=f"KR lods {coef_name}",
                **TOL_LODS,
            )

    # =================================================================
    # Edge-case genes
    # =================================================================

    def test_boundary_theta_zero_genes(self, pipeline):
        """
        Genes 226 and 230 have true site σ=0 and converged to θ=0 in R.
        They should have rdf=22 (FE equivalent) and edf=2.
        """
        fit = pipeline["fit_satt"]
        genes = pipeline["gene_names"]
        gene_idx = {g: i for i, g in enumerate(genes)}

        for gene in ["gene_226", "gene_230"]:
            if gene not in gene_idx:
                pytest.skip(f"{gene} not in post-filter genes")
            idx = gene_idx[gene]
            theta = fit.theta[self.RE_KEY][idx]
            rdf = fit.df_resid_eb[idx]

            assert abs(theta) < 1e-6, (
                f"{gene}: θ={theta}, expected ≈ 0"
            )
            assert abs(rdf - 22.0) < 0.01, (
                f"{gene}: rdf={rdf}, expected ≈ 22 (FE equivalent)"
            )

    def test_high_site_var_genes(self, pipeline):
        """
        Genes 231–235 have true site σ=2 (large RE).
        They should have large θ and rdf near 19 (heavy RE consumption).
        """
        fit = pipeline["fit_satt"]
        genes = pipeline["gene_names"]
        gene_idx = {g: i for i, g in enumerate(genes)}

        thetas = []
        rdfs = []
        for i in range(231, 236):
            gene = f"gene_{i:03d}"
            if gene not in gene_idx:
                continue
            idx = gene_idx[gene]
            thetas.append(fit.theta[self.RE_KEY][idx])
            rdfs.append(fit.df_resid_eb[idx])

        assert len(thetas) > 0, "No high_site_var genes found post-filter"

        # θ should be substantially > 0
        mean_theta = np.mean(thetas)
        assert mean_theta > 0.5, (
            f"high_site_var genes mean θ={mean_theta:.4f}, expected > 0.5"
        )

        # rdf should be near 19 (much less than 22)
        mean_rdf = np.mean(rdfs)
        assert mean_rdf < 20.0, (
            f"high_site_var genes mean rdf={mean_rdf:.4f}, expected < 20"
        )

    # =================================================================
    # Statistical sanity
    # =================================================================

    def test_sanity_effect_genes_larger_t(self, pipeline):
        """Effect genes should have larger |t| for condition than null genes."""
        manifest = pd.read_csv(DESIGN_B_DIR / "gene_manifest.csv")
        fit_eb = pipeline["fit_satt_eb"]
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
