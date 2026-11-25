"""Tests for the public package interface."""

import dreampy


def test_all_names_resolve():
    """
    Every name in dreampy.__all__ exists on the package.

    Regression test for https://github.com/sbwells22/dreampy/issues/1,
    where __all__ listed functions that were never defined.
    """
    missing = [name for name in dreampy.__all__ if not hasattr(dreampy, name)]
    assert missing == [], f"Names in __all__ that do not exist: {missing}"