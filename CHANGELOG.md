# Changelog

All notable changes to dreampy are recorded in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
While the version is below 1.0.0, a minor release (0.x.0) may change behaviour;
a patch release (0.x.y) contains fixes only.

## [0.1.1] - 2026-11-25

A bug-fix release. Three of the fixes change results or turn a silent
problem into an error. Read "Behaviour changes" before upgrading.

### Behaviour changes

- **No-intercept formulas give different, correct results.** Models written
  as `~ 0 + factor` were fitted incorrectly in 0.1.0 (see "Fixed"). Results
  from such formulas in 0.1.0 are not valid and should be re-run.
- **Formulas that previously ran may now raise an error.** A formula term
  that cannot be used now raises a `ValueError`. In 0.1.0 such terms were
  dropped without warning and a different model was fitted. This affects
  misspelled column names, interactions (`a:b`, `a*b`), transformations
  (`I(x**2)`, `log(x)`), and removal of a term with `-`.
- **Missing values in formula variables raise an error.** If any metadata
  column used by the formula has missing values, fitting stops and reports
  the column and the number of samples affected. Samples are not dropped
  automatically: remove them from both the expression data and the metadata,
  or fill in the values.
- **`z_std` changes for very significant results.** Results with a p-value
  below 1e-15 all received a `z_std` of about 8.03 in 0.1.0. They now receive
  distinct values up to about 37.07. Results with a p-value of 1e-15 or
  larger are unchanged. The intercept column is the most visible case, since
  its p-values are almost always below 1e-15.
- **Installation.** The package source moved to `src/dreampy/`. After pulling
  this version, reinstall with `pip install .` (or `pip install -e .` for
  development).

### Fixed

- `import dreampy` returned an empty package with no functions when Python
  was started from the directory containing the cloned repository. The
  repository folder was being imported in place of the installed package.
  ([#1](https://github.com/sbwells22/dreampy/issues/1))
- `from dreampy import *` raised `AttributeError`, because the package listed
  five functions that do not exist.
  ([#1](https://github.com/sbwells22/dreampy/issues/1))
- `~ 0 + factor` dropped the first level of the factor while also omitting
  the intercept. That group's mean was forced to zero and the residual
  variance of every gene was inflated, which invalidated every statistic
  from the fit. The first categorical term of a no-intercept formula now
  gets one column per level, matching R's `model.matrix`.
  ([#3](https://github.com/sbwells22/dreampy/issues/3))
- A two-level factor could be removed from a no-intercept model by the
  collinearity check, which compared the factor's own indicator columns with
  each other. Only columns from different terms are compared now.
- A pandas `Categorical` with an unused first level produced the wrong
  reference level: the last observed level became the reference. Unused
  levels are now removed first, so the reference is the first observed level.
- Formula terms that could not be resolved were dropped without warning.
  For example, `~ disease + agee` (a misspelling of `age`) fitted
  `~ disease`, and `~ disease:age` fitted an intercept-only model.
- A missing value in a categorical variable was treated as belonging to the
  reference level. Missing values in continuous and grouping variables
  already stopped the fit, but with an unhelpful message.
- `z_std` saturated at about 8.03, because it was computed from a p-value
  clipped at 1e-15. The p-value is now floored at 1e-300, matching
  `variancePartition::topTable()`.
- The README linked to a tutorial file that does not exist and had an empty
  preprint link. `CITATION.cff` pointed to the repository's former name.

### Added

- `~ x - 1` is accepted as a no-intercept formula. It is equivalent to
  `~ 0 + x` and `~ -1 + x`.
- Formula errors list every problem in the formula at once, suggest close
  matches for misspelled column names, and show the available columns.

### Changed

- The cleaned formula stored on a fit is normalised: the no-intercept forms
  are all reported as `~ 0 + ...`, and random effects as `(1|group)` without
  inner spaces. The fitted model is unaffected.
- README installation instructions recommend `pip install .`. An editable
  install (`pip install -e .`) is documented for development.
- The `z_std` conversion is vectorised.

### Known limitations

- **Contrasts are not supported yet.** A fit does not store the covariance
  between coefficient estimates, so a contrast computed by hand from the
  fitted coefficients and their standard errors will not have a correct
  standard error or degrees of freedom. Support for contrasts is planned for
  0.2.0. ([#2](https://github.com/sbwells22/dreampy/issues/2),
  [#3](https://github.com/sbwells22/dreampy/issues/3))
- Until then, to compare two levels of a factor, make the desired baseline
  the first category and refit with an intercept:

  ```python
  metadata["disease"] = pd.Categorical(
      metadata["disease"], categories=["ra", "ctrl", "sle"]
  )
  fit = dp.fit_models(pb, "~ disease + (1|donor)")  # disease_sle is now sle vs ra
  ```

- In a no-intercept model each coefficient is a group mean, so its p-value
  tests whether that mean is zero. This is rarely the question of interest.
  No-intercept formulas become useful once contrasts are available.
- Interaction terms, transformations inside a formula, and random slopes are
  not supported. They now raise an error.

## [0.1.0] - 2026-03-21

Initial release.

[0.1.1]: https://github.com/sbwells22/dreampy/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/sbwells22/dreampy/releases/tag/v0.1.0
