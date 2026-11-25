"""Tests for design matrix construction from model formulas."""

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
