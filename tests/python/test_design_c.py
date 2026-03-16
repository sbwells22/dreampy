"""
test_design_c.py — End-to-end LMM pipeline with crossed random effects.

Runs the full dreampy mixed-effects pipeline from raw counts through final
eBayes results for Design C (~ condition + (1|collection_site) + (1|batch),
300 genes × 30 samples). Satterthwaite only — no Kenward-Roger pass.

Pipeline order (canonical):
  TMM → filter → log2CPM → voom weights → fit (Satt) → eBayes

Design C specifics:
  - Two crossed random effects: (1|collection_site) + (1|batch)
  - 5 sites × 3 batches = 15 cells, 2 samples per cell (balanced)
  - 2 coefficients: (Intercept), conditioncontrol
  - Reference level: "case" (alphabetically first)
  - 300 genes → 292 after filtering (5 allzero + 2 low-expression)
  - FE-equivalent rdf = 28 (30 samples - 2 FE parameters)

Edge-case gene categories:
  - both_boundary_theta (266–270): true site σ=0, batch σ=0. 4/5 converge
    to both θ=0 exactly. gene_270 picked up spurious site variance.
  - site_only (271–275): true site σ=1.5, batch σ=0. Site θ large, batch
    θ near zero (some pick up small noise).
  - batch_only (276–280): true site σ=0, batch σ=1.5. Unreliable on this
    sparse design — only 3 batches with 30 samples. 3/5 lost the batch
    signal. Tests are lenient on this category.
  - both_high_var (281–285): true site σ=1.5, batch σ=1.5. Both θ large.
"""

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from pathlib import Path
from numpy.testing import assert_allclose

import dreampy as dp

# =============================================================================
# Tolerances — from spec v0.10, per-quantity
# =============================================================================

TOL_TMM = dict(rtol=1e-5)
TOL_LOG2CPM = dict(rtol=1e-6)
TOL_VOOM_WEIGHTS = dict(rtol=7e-3)
TOL_COEF = dict(rtol=5e-5, atol=2e-4)
TOL_STDEV_UNSCALED = dict(rtol=1e-3)
TOL_SIGMA = dict(rtol=5e-4)
TOL_THETA = dict(atol=1e-3)
TOL_RDF = dict(rtol=1e-4)
TOL_SATT_DF = dict(rtol=5e-3)
TOL_S2_PRIOR = dict(rtol=1e-4)
TOL_DF_PRIOR = dict(rtol=1e-4)
TOL_S2_POST = dict(rtol=1e-3)
TOL_DF_TOTAL = dict(rtol=5e-3)
TOL_T = dict(rtol=5e-4, atol=1e-3)
TOL_P = dict(rtol=1e-3, atol=1e-6)
TOL_LODS = dict(rtol=1.5e-3)

# =============================================================================
# Paths
# =============================================================================

DESIGN_C_DIR = Path(__file__).parent.parent / "data" / "synthetic" / "design_c"

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


def load_design_c_ref(filename, index_col=0):
    """Load an R reference CSV from the Design C directory."""
    return pd.read_csv(DESIGN_C_DIR / filename, index_col=index_col)


# =============================================================================
# Design C: Crossed Random Effects Pipeline
# =============================================================================


class TestDesignC:
    """
    End-to-end LMM pipeline with two crossed random effects.
    Satterthwaite only — no Kenward-Roger pass.
    """

    FORMULA = "~ condition + (1|collection_site) + (1|batch)"
    ASSAY_NAME = "celltype_A"

    # dreampy coef name → R CSV coef name
    COEF_MAP = {
        "condition_control": "conditioncontrol",
    }

    # RE keys in dreampy's theta dict
    RE_KEY_SITE = "collection_site"
    RE_KEY_BATCH = "batch"

    # FE-equivalent rdf: 30 samples - 2 fixed effects = 28
    FE_EQUIV_RDF = 28.0

    # -----------------------------------------------------------------
    # Fixture: run full pipeline once
    # -----------------------------------------------------------------

    @pytest.fixture(scope="class")
    def pipeline(self):
        """
        Run the complete Design C pipeline: TMM → filter → log2CPM → voom →
        fit (Satterthwaite) → eBayes.

        Returns a dict of intermediate results for per-stage comparison.
        """
        # --- Load raw data ---
        counts_df = pd.read_csv(DESIGN_C_DIR / "counts.csv", index_col=0)
        metadata_df = pd.read_csv(DESIGN_C_DIR / "metadata.csv")
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
        results["all_gene_names"] = list(counts_df.index)       # 300 genes
        results["sample_names"] = list(counts_df.columns)        # 30 samples

        # --- Step 2: TMM normalization (all 300 genes) ---
        pb = dp.compute_tmm_factors(pb, assay_col="assays")
        results["norm_factors"] = pb.obs["norm_factors"].values.copy()
        results["lib_size"] = pb.obs["lib_size"].values.copy()

        # --- Step 3: Gene filtering (300 → 292 genes) ---
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

        # --- Step 6: Satterthwaite fit ---
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
        results["gene_names"] = list(pb.var_names)               # 292 genes

        return results

    # =================================================================
    # Shape guardrails
    # =================================================================

    def test_shape_invariants(self, pipeline):
        """Verify dimensional consistency across pipeline outputs."""
        pb = pipeline["pb"]
        fit = pipeline["fit"]
        fit_eb = pipeline["fit_eb"]
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
        ref = pd.read_csv(DESIGN_C_DIR / "ref_tmm_factors.csv")
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
        ref = pd.read_csv(DESIGN_C_DIR / "ref_tmm_factors.csv")
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
        ref = pd.read_csv(DESIGN_C_DIR / "ref_filter_mask.csv")

        kept = set(pipeline["kept_genes"])
        all_genes = pipeline["all_gene_names"]
        mask_python = np.array([g in kept for g in all_genes])

        mask_r = ref["keep"].map(
            {"TRUE": True, "FALSE": False, True: True, False: False}
        ).values.astype(bool)

        np.testing.assert_array_equal(mask_python, mask_r)

    def test_filter_counts(self, pipeline):
        """Verify expected filter counts: 8 removed, 292 kept."""
        assert len(pipeline["kept_genes"]) == 292
        n_filtered = len(pipeline["all_gene_names"]) - len(pipeline["kept_genes"])
        assert n_filtered == 8

    def test_filter_removes_allzero_genes(self, pipeline):
        """All-zero genes (261–265) must be filtered."""
        kept = set(pipeline["kept_genes"])
        for i in range(261, 266):
            gene = f"gene_{i:03d}"
            assert gene not in kept, f"{gene} (allzero) should have been filtered"

    def test_filter_preserves_edge_cases(self, pipeline):
        """All edge-case gene categories (266–300) must survive filtering."""
        kept = set(pipeline["kept_genes"])
        for i in range(266, 301):
            gene = f"gene_{i:03d}"
            assert gene in kept, f"{gene} should have survived filtering"

    # =================================================================
    # log2CPM
    # =================================================================

    def test_log2cpm(self, pipeline):
        """Verify log2CPM values match R."""
        ref = load_design_c_ref("ref_log2cpm.csv")
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
        ref = load_design_c_ref("ref_voom_weights.csv")
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

    def test_fit_method(self, pipeline):
        """Design C should use mixed model (LMM)."""
        assert pipeline["fit"].method == "lmm"

    def test_fit_coef_names(self, pipeline):
        """Verify coefficient names match expected dreampy convention."""
        expected = ["(Intercept)", "condition_control"]
        assert list(pipeline["fit"].coef_names) == expected

    def test_fit_coefficients(self, pipeline):
        """Verify fitted coefficients match R's dreamlet output."""
        ref = pd.read_csv(DESIGN_C_DIR / "ref_lmer.csv").set_index("gene")
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

    def test_fit_stdev_unscaled(self, pipeline):
        """Verify unscaled standard errors match R."""
        ref = pd.read_csv(DESIGN_C_DIR / "ref_lmer.csv").set_index("gene")
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

    def test_fit_sigma(self, pipeline):
        """Verify residual standard deviations match R."""
        ref = pd.read_csv(DESIGN_C_DIR / "ref_lmer.csv").set_index("gene")
        fit = pipeline["fit"]
        genes = pipeline["gene_names"]

        assert_allclose_finite(
            fit.sigma,
            ref.loc[genes, "sigma"].values,
            label="sigma",
            **TOL_SIGMA,
        )

    def test_fit_no_failures(self, pipeline):
        """All 292 post-filter genes should fit successfully."""
        from dreampy import FitStatus

        fit = pipeline["fit"]
        failed = fit.fit_status != FitStatus.SUCCESS
        boundary = fit.fit_status == FitStatus.BOUNDARY
        truly_failed = failed & ~boundary
        if truly_failed.any():
            failed_genes = np.array(pipeline["gene_names"])[truly_failed]
            pytest.fail(
                f"{truly_failed.sum()} genes failed: {list(failed_genes[:10])}"
            )

    # -----------------------------------------------------------------
    # Theta (variance components) — two RE keys
    # -----------------------------------------------------------------

    def test_theta_collection_site(self, pipeline):
        """Verify θ_site = sqrt(varComp_collection_site) / sigma matches R."""
        ref = pd.read_csv(DESIGN_C_DIR / "ref_lmer.csv").set_index("gene")
        fit = pipeline["fit"]
        genes = pipeline["gene_names"]

        r_varcomp = ref.loc[genes, "varComp_collection_site"].values
        r_sigma = ref.loc[genes, "sigma"].values
        r_theta = np.sqrt(r_varcomp) / r_sigma

        assert_allclose_finite(
            fit.theta[self.RE_KEY_SITE],
            r_theta,
            label="theta_collection_site",
            **TOL_THETA,
        )

    def test_theta_batch(self, pipeline):
        """Verify θ_batch = sqrt(varComp_batch) / sigma matches R."""
        ref = pd.read_csv(DESIGN_C_DIR / "ref_lmer.csv").set_index("gene")
        fit = pipeline["fit"]
        genes = pipeline["gene_names"]

        r_varcomp = ref.loc[genes, "varComp_batch"].values
        r_sigma = ref.loc[genes, "sigma"].values
        r_theta = np.sqrt(r_varcomp) / r_sigma

        assert_allclose_finite(
            fit.theta[self.RE_KEY_BATCH],
            r_theta,
            label="theta_batch",
            **TOL_THETA,
        )

    # -----------------------------------------------------------------
    # Degrees of freedom
    # -----------------------------------------------------------------

    def test_df_resid_eb(self, pipeline):
        """Verify trace-based rdf matches R."""
        ref = pd.read_csv(DESIGN_C_DIR / "ref_lmer.csv").set_index("gene")
        fit = pipeline["fit"]
        genes = pipeline["gene_names"]

        assert_allclose_finite(
            fit.df_resid_eb,
            ref.loc[genes, "rdf"].values,
            label="df_resid_eb (rdf)",
            **TOL_RDF,
        )

    def test_df_resid_eb_not_uniform(self, pipeline):
        """LMM df_resid_eb should vary across genes (unlike FE)."""
        fit = pipeline["fit"]
        assert fit.df_resid_eb.min() < fit.df_resid_eb.max(), (
            "LMM df_resid_eb should vary across genes"
        )

    def test_df_resid_test(self, pipeline):
        """Verify per-coefficient Satterthwaite denominator df."""
        ref = pd.read_csv(DESIGN_C_DIR / "ref_satterthwaite.csv").set_index("gene")
        fit = pipeline["fit"]
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

    def test_df_intercept_smaller_than_condition(self, pipeline):
        """
        Intercept Satterthwaite df should generally be smaller than
        condition df, because site and batch REs absorb intercept variation.
        """
        ref = pd.read_csv(DESIGN_C_DIR / "ref_satterthwaite.csv").set_index("gene")
        genes = pipeline["gene_names"]

        df_int = ref.loc[genes, "df_residual_(Intercept)"].values
        df_cond = ref.loc[genes, "df_residual_conditioncontrol"].values

        assert df_int.mean() < df_cond.mean(), (
            f"Intercept mean df ({df_int.mean():.2f}) should be < "
            f"condition mean df ({df_cond.mean():.2f})"
        )

    # =================================================================
    # eBayes
    # =================================================================

    def test_ebayes_returns_new_fit(self, pipeline):
        """ebayes() must return a new DEFit, not modify the original."""
        assert pipeline["fit"] is not pipeline["fit_eb"]
        assert not pipeline["fit"].eb_applied
        assert pipeline["fit_eb"].eb_applied

    def test_ebayes_s2_prior(self, pipeline):
        """Verify prior variance scalar matches R."""
        ref = pd.read_csv(DESIGN_C_DIR / "ref_ebayes.csv")
        fit_eb = pipeline["fit_eb"]

        assert_allclose_finite(
            np.array([fit_eb.s2_prior]),
            np.array([ref["s2_prior"].iloc[0]]),
            label="s2_prior",
            **TOL_S2_PRIOR,
        )

    def test_ebayes_df_prior(self, pipeline):
        """Verify prior degrees of freedom scalar matches R."""
        ref = pd.read_csv(DESIGN_C_DIR / "ref_ebayes.csv")
        fit_eb = pipeline["fit_eb"]

        assert_allclose_finite(
            np.array([fit_eb.df_prior]),
            np.array([ref["df_prior"].iloc[0]]),
            label="df_prior",
            **TOL_DF_PRIOR,
        )

    def test_ebayes_s2_post(self, pipeline):
        """Verify posterior variance per gene matches R."""
        ref = pd.read_csv(DESIGN_C_DIR / "ref_ebayes.csv").set_index("gene")
        fit_eb = pipeline["fit_eb"]
        genes = pipeline["gene_names"]

        assert_allclose_finite(
            fit_eb.s2_post,
            ref.loc[genes, "s2_post"].values,
            label="s2_post",
            **TOL_S2_POST,
        )

    def test_ebayes_df_total(self, pipeline):
        """Verify df_total per coefficient (varies per gene and per coef)."""
        ref = pd.read_csv(DESIGN_C_DIR / "ref_ebayes.csv").set_index("gene")
        fit_eb = pipeline["fit_eb"]
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

    def test_ebayes_t_statistics(self, pipeline):
        """Verify moderated t-statistics match R."""
        ref = pd.read_csv(DESIGN_C_DIR / "ref_ebayes.csv").set_index("gene")
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
        ref = pd.read_csv(DESIGN_C_DIR / "ref_ebayes.csv").set_index("gene")
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
        ref = pd.read_csv(DESIGN_C_DIR / "ref_ebayes.csv").set_index("gene")
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

    # =================================================================
    # Edge-case genes
    # =================================================================

    def test_both_boundary_theta_genes(self, pipeline):
        """
        Genes 266–269 have true site σ=0 and batch σ=0.
        4/5 converge to both θ=0 in R (gene_270 picked up spurious site var).
        Check at least 3 genes have both θ ≈ 0 and rdf ≈ 28.
        """
        fit = pipeline["fit"]
        genes = pipeline["gene_names"]
        gene_idx = {g: i for i, g in enumerate(genes)}

        n_both_zero = 0
        for i in range(266, 271):
            gene = f"gene_{i:03d}"
            if gene not in gene_idx:
                continue
            idx = gene_idx[gene]
            theta_site = fit.theta[self.RE_KEY_SITE][idx]
            theta_batch = fit.theta[self.RE_KEY_BATCH][idx]
            rdf = fit.df_resid_eb[idx]

            if abs(theta_site) < 1e-4 and abs(theta_batch) < 1e-4:
                n_both_zero += 1
                assert abs(rdf - self.FE_EQUIV_RDF) < 0.01, (
                    f"{gene}: both θ≈0 but rdf={rdf}, expected ≈ {self.FE_EQUIV_RDF}"
                )

        assert n_both_zero >= 3, (
            f"Expected ≥3 of 5 both_boundary_theta genes to have both θ≈0, "
            f"got {n_both_zero}"
        )

    def test_site_only_genes(self, pipeline):
        """
        Genes 271–275 have true site σ=1.5, batch σ=0.
        Site θ should be substantially > 0 on average.
        """
        fit = pipeline["fit"]
        genes = pipeline["gene_names"]
        gene_idx = {g: i for i, g in enumerate(genes)}

        site_thetas = []
        for i in range(271, 276):
            gene = f"gene_{i:03d}"
            if gene not in gene_idx:
                continue
            idx = gene_idx[gene]
            site_thetas.append(fit.theta[self.RE_KEY_SITE][idx])

        assert len(site_thetas) > 0, "No site_only genes found"
        assert np.mean(site_thetas) > 0.3, (
            f"site_only genes mean site θ={np.mean(site_thetas):.4f}, "
            f"expected > 0.3"
        )

    def test_both_high_var_genes(self, pipeline):
        """
        Genes 281–285 have true site σ=1.5 and batch σ=1.5.
        Both θ should be substantial. rdf should be well below FE-equivalent.
        """
        fit = pipeline["fit"]
        genes = pipeline["gene_names"]
        gene_idx = {g: i for i, g in enumerate(genes)}

        rdfs = []
        for i in range(281, 286):
            gene = f"gene_{i:03d}"
            if gene not in gene_idx:
                continue
            idx = gene_idx[gene]
            rdfs.append(fit.df_resid_eb[idx])

        assert len(rdfs) > 0, "No both_high_var genes found"
        mean_rdf = np.mean(rdfs)
        assert mean_rdf < self.FE_EQUIV_RDF - 2, (
            f"both_high_var genes mean rdf={mean_rdf:.4f}, "
            f"expected < {self.FE_EQUIV_RDF - 2} (heavy RE consumption)"
        )

    # =================================================================
    # Statistical sanity
    # =================================================================

    def test_sanity_effect_genes_larger_t(self, pipeline):
        """Effect genes should have larger |t| for condition than null genes."""
        manifest = pd.read_csv(DESIGN_C_DIR / "gene_manifest.csv")
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
