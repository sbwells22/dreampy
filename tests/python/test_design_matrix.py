"""Tests for design matrix construction from model formulas."""

import re

import numpy as np
import pandas as pd
import pytest

import dreampy as dp
from dreampy.design import build_model_matrices


@pytest.fixture
def metadata():
    """24 samples: a three-level factor, a two-level factor, a covariate, 12 donors."""
    rng = np.random.default_rng(11)
    n = 24
    return pd.DataFrame(
        {
            "disease": np.repeat(["ctrl", "ra", "sle"], 8),
            "sex": np.tile(["F", "M"], 12),
            "age": rng.normal(50, 10, n),
            "donor": np.tile([f"d{i:02d}" for i in range(12)], 2),
        },
        index=[f"s{i:02d}" for i in range(n)],
    )


def test_no_intercept_fit_matches_intercept_fit(metadata):
    """
    '~ 0 + disease' and '~ disease' describe the same model.

    Regression test for https://github.com/sbwells22/dreampy/issues/3. The
    no-intercept design used to drop the reference level, which forced that
    group's mean to zero and inflated the residual variance of every gene.
    """
    rng = np.random.default_rng(11)
    group_means = metadata["disease"].map({"ctrl": 5.0, "ra": 7.0, "sle": 6.0}).to_numpy()
    expression = group_means + rng.normal(0, 0.3, size=(50, len(metadata)))

    fit_reference = dp.fit_models(expression, "~ disease", metadata, show_progress=False)
    fit_means = dp.fit_models(expression, "~ 0 + disease", metadata, show_progress=False)

    assert fit_means.coef_names == ["disease_ctrl", "disease_ra", "disease_sle"]

    # Same model, so the same residual standard deviation
    np.testing.assert_allclose(fit_means.sigma, fit_reference.sigma, rtol=1e-8)

    # Each group mean equals the intercept plus that group's reference-coded effect
    b = fit_reference.coefficients
    expected_means = np.column_stack([b[:, 0], b[:, 0] + b[:, 1], b[:, 0] + b[:, 2]])
    np.testing.assert_allclose(fit_means.coefficients, expected_means, rtol=1e-8)


@pytest.mark.parametrize(
    ("formula", "expected_columns"),
    [
        # With an intercept: every factor is reference coded
        ("~ disease", ["(Intercept)", "disease_ra", "disease_sle"]),
        # Without an intercept: the first factor gets one column per level
        ("~ 0 + disease", ["disease_ctrl", "disease_ra", "disease_sle"]),
        ("~ 0 + sex", ["sex_F", "sex_M"]),
        ("~ 0 + disease + age", ["disease_ctrl", "disease_ra", "disease_sle", "age"]),
        ("~ 0 + age + disease", ["age", "disease_ctrl", "disease_ra", "disease_sle"]),
        # A second factor is reference coded
        ("~ 0 + disease + sex", ["disease_ctrl", "disease_ra", "disease_sle", "sex_M"]),
        ("~ 0 + sex + disease", ["sex_F", "sex_M", "disease_ra", "disease_sle"]),
        # Random effect grouping variables never contribute fixed-effect columns
        ("~ 0 + disease + (1|donor)", ["disease_ctrl", "disease_ra", "disease_sle"]),
        ("~ 0 + age + (1|donor)", ["age"]),
    ],
)
def test_design_matrix_columns(metadata, formula, expected_columns):
    """Formulas produce the expected columns, at full rank, with nothing dropped."""
    X, coef_names, _, _, _, _, cleaning_warnings = build_model_matrices(
        formula, metadata, verbose=False
    )

    assert coef_names == expected_columns
    assert X.shape == (len(metadata), len(expected_columns))
    assert np.linalg.matrix_rank(X) == len(expected_columns)
    assert cleaning_warnings == []


@pytest.mark.parametrize(
    ("formula", "expected_columns"),
    [
        ("~ stage", ["(Intercept)", "stage_ra", "stage_sle"]),
        ("~ 0 + stage", ["stage_ctrl", "stage_ra", "stage_sle"]),
    ],
)
def test_unused_category_levels_are_ignored(metadata, formula, expected_columns):
    """A categorical level with no samples does not become the reference level."""
    metadata["stage"] = pd.Categorical(
        metadata["disease"], categories=["healthy", "ctrl", "ra", "sle"]
    )

    _, coef_names, _, _, _, _, cleaning_warnings = build_model_matrices(
        formula, metadata, verbose=False
    )

    assert coef_names == expected_columns
    assert cleaning_warnings == []


def test_no_intercept_weights_match_intercept_weights(metadata):
    """
    Precision weights do not depend on how the same model is parameterised.

    estimate_weights() builds its design matrix through the same code as
    fit_models(), so it was affected by the same defect. See
    https://github.com/sbwells22/dreampy/issues/3.

    A fixed-effects formula is used so that the comparison does not depend
    on an optimiser, and a fixed span so that the smoother is identical in
    both calls.
    """
    rng = np.random.default_rng(11)
    n_genes, n_samples = 200, len(metadata)

    # Counts with a disease effect and a spread of expression levels
    disease_effect = metadata["disease"].map({"ctrl": 0.0, "ra": 0.8, "sle": -0.5}).to_numpy()
    baseline = rng.uniform(2.0, 8.0, size=(n_genes, 1))
    counts = rng.poisson(np.exp(baseline + disease_effect)).astype(float)

    lib_size = counts.sum(axis=0)
    log2_cpm = np.log2((counts + 0.5) / (lib_size + 1.0) * 1e6)

    weights_reference, _ = dp.estimate_weights(
        log2_cpm, "~ disease", metadata, lib_size, counts=counts, span=0.5, quiet=True
    )
    weights_means, _ = dp.estimate_weights(
        log2_cpm, "~ 0 + disease", metadata, lib_size, counts=counts, span=0.5, quiet=True
    )

    assert weights_means.shape == (n_genes, n_samples)
    assert np.all(np.isfinite(weights_means))
    np.testing.assert_allclose(weights_means, weights_reference, rtol=1e-8)


@pytest.mark.parametrize(
    ("formula", "message"),
    [
        ("~ disease + agee", "'agee' is not a column in the metadata"),
        ("~ Disease", "Did you mean 'disease'?"),
        ("~ disease:age", "interaction terms are not supported"),
        ("~ disease*age", "interaction terms are not supported"),
        ("~ age + I(age**2)", "transformations and function calls are not supported"),
        ("~ disease - age", "removing a term with '-' is not supported"),
        ("~ disease + 1|donor", "a random effect must be written in parentheses"),
    ],
)
def test_unusable_formula_terms_raise(metadata, formula, message):
    """A term that cannot be used raises an error; it is never dropped from the model."""
    with pytest.raises(ValueError, match=re.escape(message)):
        build_model_matrices(formula, metadata, verbose=False)


def test_all_unusable_terms_are_reported_together(metadata):
    """One error lists every problem in the formula, not only the first."""
    with pytest.raises(ValueError) as error:
        build_model_matrices("~ agee + disease:age", metadata, verbose=False)

    assert "'agee' is not a column in the metadata" in str(error.value)
    assert "interaction terms are not supported" in str(error.value)


@pytest.mark.parametrize(
    ("formula_minus_one", "formula_zero"),
    [
        ("~ disease - 1", "~ 0 + disease"),
        ("~ disease + age - 1", "~ 0 + disease + age"),
        ("~ disease + (1|donor) - 1", "~ 0 + disease + (1|donor)"),
    ],
)
def test_minus_one_suppresses_the_intercept(metadata, formula_minus_one, formula_zero):
    """'- 1' and '0 +' are two spellings of the same no-intercept model."""
    X_minus, names_minus, Z_minus, *_ = build_model_matrices(
        formula_minus_one, metadata, verbose=False
    )
    X_zero, names_zero, Z_zero, *_ = build_model_matrices(formula_zero, metadata, verbose=False)

    assert names_minus == names_zero
    np.testing.assert_array_equal(X_minus, X_zero)
    if Z_zero is None:
        assert Z_minus is None
    else:
        np.testing.assert_array_equal(Z_minus, Z_zero)


def test_column_name_containing_a_hyphen_is_usable(metadata):
    """A hyphen inside a column name is part of the name, not a request to remove a term."""
    metadata["cell-type"] = metadata["disease"]

    _, coef_names, *_ = build_model_matrices("~ cell-type", metadata, verbose=False)

    assert coef_names == ["(Intercept)", "cell-type_ra", "cell-type_sle"]


@pytest.mark.parametrize("variable", ["disease", "age", "donor"])
def test_missing_values_in_formula_variables_raise(metadata, variable):
    """A missing value in a categorical, continuous or grouping variable raises an error."""
    metadata[variable] = metadata[variable].astype(object)
    metadata.loc[metadata.index[[0, 5]], variable] = np.nan

    with pytest.raises(ValueError, match=re.escape(f"'{variable}' (2 of 24 samples)")):
        build_model_matrices("~ disease + age + (1|donor)", metadata, verbose=False)


def test_missing_values_in_unused_columns_are_ignored(metadata):
    """Only variables that appear in the formula are checked for missing values."""
    metadata.loc[metadata.index[0], "sex"] = np.nan

    _, coef_names, *_ = build_model_matrices("~ disease + age", metadata, verbose=False)

    assert coef_names == ["(Intercept)", "disease_ra", "disease_sle", "age"]


def test_fitting_functions_reject_unusable_formula(metadata):
    """fit_models() and estimate_weights() raise the same error as the design builder."""
    rng = np.random.default_rng(11)
    expression = rng.normal(5.0, 1.0, size=(5, len(metadata)))
    lib_size = np.full(len(metadata), 1e6)
    message = re.escape("'agee' is not a column in the metadata")

    with pytest.raises(ValueError, match=message):
        dp.fit_models(expression, "~ disease + agee", metadata, show_progress=False)

    with pytest.raises(ValueError, match=message):
        dp.estimate_weights(
            expression, "~ disease + agee", metadata, lib_size, span=0.5, quiet=True
        )