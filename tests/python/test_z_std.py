"""Tests for the standardized z-score (z_std) computed by ebayes()."""

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.stats import t as t_dist

import dreampy as dp
from dreampy.ebayes import _compute_z_std

# Largest |z_std| that can be returned: the normal quantile for a two-sided
# p-value of 1e-300, the floor used by variancePartition::topTable().
Z_STD_CAP = norm.isf(1e-300 / 2)


def test_z_std_distinguishes_strongly_significant_genes():
    """
    z_std keeps increasing with the strength of evidence.

    It used to be computed from a p-value clipped at 1e-15, so every gene
    more significant than that received the same value, about 8.03.
    """
    rng = np.random.default_rng(11)
    metadata = pd.DataFrame({"group": np.repeat(["a", "b"], 12)})
    is_b = (metadata["group"] == "b").to_numpy()

    # 10 null genes, then 10 genes with effects far beyond p = 1e-15
    effects = np.concatenate([np.zeros(10), np.linspace(3.0, 12.0, 10)])
    expression = 5.0 + effects[:, None] * is_b + rng.normal(0, 0.05, size=(20, 24))
    strong = effects > 0

    fit = dp.ebayes(dp.fit_models(expression, "~ group", metadata, show_progress=False))
    column = fit.coef_names.index("group_b")
    t, p, z = fit.t[:, column], fit.p_value[:, column], fit.z_std[:, column]

    # The premise of the test: these genes are beyond the old clipping point
    assert np.all(p[strong] < 1e-15)

    # z_std is the normal quantile with the same two-sided p-value, signed like t
    np.testing.assert_allclose(2 * norm.sf(np.abs(z)), np.maximum(p, 1e-300), rtol=1e-9)
    np.testing.assert_array_equal(np.sign(z), np.sign(t))

    # Stronger evidence gives a larger z_std: no ties, same ordering as |t|
    assert len(np.unique(z[strong])) == strong.sum()
    np.testing.assert_array_equal(np.argsort(np.abs(z[strong])), np.argsort(np.abs(t[strong])))


def test_z_std_boundary_and_invalid_inputs():
    """Infinite, zero and missing inputs give the documented values."""
    t = np.array([[np.inf, -np.inf, 0.0, np.nan, 3.0, 3.0, 3.0, 3.0]])
    df = np.array([[50.0, 50.0, 50.0, 50.0, 0.0, -2.0, np.nan, np.inf]])

    z = _compute_z_std(t, df)[0]

    assert z[0] == Z_STD_CAP  # p underflows to 0 and is floored at 1e-300
    assert z[1] == -Z_STD_CAP
    assert z[2] == 0.0
    assert np.isnan(z[3])  # NaN t
    assert np.all(np.isnan(z[4:7]))  # df of zero, negative, NaN
    assert z[7] == 3.0  # infinite df: t is already a z-score


def test_z_std_accepts_integer_input():
    """Integer-valued input arrays do not truncate the result to whole numbers."""
    z = _compute_z_std(np.array([[5]]), np.array([[50]]))

    assert z.dtype == np.float64
    np.testing.assert_allclose(2 * norm.sf(z[0, 0]), 2 * t_dist.sf(5, 50))
