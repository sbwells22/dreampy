"""
Formula parsing and validation for dreampy.

This module provides:
- Formula: Simple formula parser for model specification
- validate_formula(): Validate formula syntax and metadata compatibility

Supports formula syntax like: ~ x1 + x2 + (1|group)
"""

import re
from dataclasses import dataclass, field

import pandas as pd


@dataclass
class FormulaValidation:
    """
    Result of formula validation.

    Attributes
    ----------
    is_valid : bool
        True if formula is valid and supported
    fixed_terms : List[str]
        Fixed effect terms in the formula
    random_terms : List[Tuple[str, str]]
        Random effect terms as (form, group_var) tuples, e.g., ("1", "donor")
    warnings : List[str]
        Warning messages (non-fatal issues)
    errors : List[str]
        Error messages (fatal issues)
    metadata_issues : List[str], optional
        Issues with metadata columns (if metadata was provided)
    column_types : Dict[str, str], optional
        Detected column types: "categorical", "continuous", "mixed"
    """

    is_valid: bool
    fixed_terms: list[str]
    random_terms: list[tuple[str, str]]
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    metadata_issues: list[str] | None = None
    column_types: dict | None = None


class Formula:
    """
    Simple formula parser for model specification.

    Handles formulas like: ~ y ~ x1 + x2 + (1|group)

    Attributes
    ----------
    original : str
        Original formula string
    response : str or None
        Response variable (left of ~), if any
    fixed_terms : List[str]
        Fixed effect terms
    random_terms : List[Tuple[str, str]]
        Random effect terms as (form, group_var) tuples

    Example
    -------
    >>> f = Formula("~ treatment + age + (1|donor)")
    >>> f.fixed_terms
    ['treatment', 'age']
    >>> f.random_terms
    [('1', 'donor')]
    >>> f.has_random_effects()
    True
    """

    def __init__(self, formula_str: str):
        """Parse formula string into components."""
        self.original = formula_str.strip()
        self._parse()

    def _parse(self):
        """Parse the formula string."""
        s = self.original.strip()

        # Handle response ~ predictors format
        if "~" in s:
            lhs, rhs = s.split("~", 1)
            self.response = lhs.strip() if lhs.strip() else None
            s = rhs.strip()
        else:
            self.response = None

        # Find all parenthesized random effect terms
        random_pattern = r"\(\s*([^)]+)\s*\|\s*([^)]+)\s*\)"

        # Extract random terms
        random_matches = re.finditer(random_pattern, s)
        self.random_terms = [(m.group(1).strip(), m.group(2).strip()) for m in random_matches]

        # Remove random terms from string
        s_fixed = re.sub(random_pattern, "", s)

        # Extract fixed terms
        self.fixed_terms = [term.strip() for term in s_fixed.split("+") if term.strip()]

    def has_random_effects(self) -> bool:
        """Check if formula has random effects."""
        return len(self.random_terms) > 0

    def to_string(self) -> str:
        """Convert formula back to string."""
        parts = self.fixed_terms.copy()
        for effect, group in self.random_terms:
            parts.append(f"({effect}|{group})")
        return "~ " + " + ".join(parts) if parts else "~ 1"

    def remove_term(self, term: str) -> bool:
        """
        Remove a term from the formula.
        Returns True if term was found and removed.
        """
        term = term.strip()

        # Try to remove from fixed terms
        if term in self.fixed_terms:
            self.fixed_terms.remove(term)
            return True

        # Try to remove from random terms (match by group name)
        for effect, group in self.random_terms:
            if group == term or f"({effect}|{group})" == term:
                self.random_terms.remove((effect, group))
                return True

        return False

    def get_terms(self) -> set[str]:
        """Get all unique terms and variables in formula."""
        terms = set(self.fixed_terms)
        for effect, group in self.random_terms:
            terms.add(group)
            # Add variables in effect
            effect_vars = re.findall(r"\b\w+\b", effect)
            terms.update(effect_vars)
        return terms

    def __str__(self) -> str:
        return self.to_string()

    def __repr__(self) -> str:
        return f"Formula('{self.to_string()}')"


def validate_formula(
    formula: str,
    metadata: pd.DataFrame | None = None,
) -> FormulaValidation:
    """
    Validate formula syntax and optionally check metadata compatibility.

    Parameters
    ----------
    formula : str
        Formula string, e.g., "~ treatment + (1|donor)"
    metadata : pd.DataFrame, optional
        If provided, also validates that columns exist and types are compatible

    Returns
    -------
    FormulaValidation
        Dataclass with: is_valid, fixed_terms, random_terms, warnings, errors

    Raises
    ------
    ValueError
        If formula uses unsupported features (interactions, random slopes, etc.)

    Example
    -------
    >>> # Syntax-only validation
    >>> result = validate_formula("~ treatment + (1|donor)")
    >>> result.fixed_terms
    ['treatment']
    >>> result.random_terms
    [('1', 'donor')]

    >>> # With metadata validation
    >>> result = validate_formula("~ treatment + (1|donor)", metadata)
    >>> result.warnings
    ['Column "treatment" has only 2 levels']

    Notes
    -----
    **Supported formula syntax (v0.1):**
    - Fixed intercept: `~ 1 + ...`
    - Fixed effects: `~ treatment + age`
    - Random intercepts: `(1|donor)`
    - Multiple crossed RE: `(1|donor) + (1|batch)`
    - No intercept: `~ 0 + treatment`

    **NOT supported (v0.1):**
    - Interaction terms: `x:y`, `x*y`
    - Random slopes: `(treatment|donor)`
    - Nested random effects: `(1|site/donor)`
    """
    warnings_list = []
    errors_list = []

    # Parse formula
    try:
        f = Formula(formula)
    except Exception as e:
        return FormulaValidation(
            is_valid=False,
            fixed_terms=[],
            random_terms=[],
            errors=[f"Failed to parse formula: {e}"],
        )

    # Check for unsupported features

    # 1. Interaction terms
    for term in f.fixed_terms:
        if ":" in term:
            raise ValueError(
                f"Formula contains interaction term '{term}'. "
                "Interactions are not supported in v0.1. Use explicit dummy variables instead."
            )
        if "*" in term:
            raise ValueError(
                f"Formula contains interaction term '{term}'. "
                "Interactions are not supported in v0.1. Use explicit dummy variables instead."
            )

    # 2. Random slopes (anything other than "1" on LHS of |)
    for effect, group in f.random_terms:
        if effect.strip() != "1":
            raise ValueError(
                f"Formula contains random slope '({effect}|{group})'. "
                "Only random intercepts '(1|group)' are supported in v0.1."
            )

        # 3. Nested random effects (/ in group)
        if "/" in group:
            raise ValueError(
                f"Formula contains nested random effect '(1|{group})'. "
                "Nested random effects are not supported in v0.1. Use crossed effects instead."
            )

    # Metadata validation (if provided)
    metadata_issues = None
    column_types = None

    if metadata is not None:
        metadata_issues = []
        column_types = {}

        # Check all referenced variables exist in metadata
        all_vars = f.get_terms()
        # Remove '0' and '1' which are special formula terms
        all_vars = {v for v in all_vars if v not in ("0", "1")}

        for var in all_vars:
            if var not in metadata.columns:
                metadata_issues.append(f"Column '{var}' not found in metadata")
                errors_list.append(f"Column '{var}' not found in metadata")
            else:
                col = metadata[var]

                # Determine type
                if pd.api.types.is_numeric_dtype(col):
                    column_types[var] = "continuous"
                elif isinstance(col.dtype, pd.CategoricalDtype) or col.dtype == object:
                    column_types[var] = "categorical"
                    n_levels = col.nunique()
                    if n_levels == 1:
                        warnings_list.append(f"Column '{var}' has only 1 level (constant)")
                    elif n_levels == 2:
                        warnings_list.append(f"Column '{var}' has only 2 levels")
                else:
                    column_types[var] = "mixed"

        # Check random effect grouping variables have enough groups
        for _effect, group in f.random_terms:
            if group in metadata.columns:
                n_groups = metadata[group].nunique()
                if n_groups < 2:
                    errors_list.append(
                        f"Random effect grouping variable '{group}' has only {n_groups} level(s)"
                    )

    is_valid = len(errors_list) == 0

    return FormulaValidation(
        is_valid=is_valid,
        fixed_terms=f.fixed_terms,
        random_terms=f.random_terms,
        warnings=warnings_list,
        errors=errors_list,
        metadata_issues=metadata_issues,
        column_types=column_types,
    )


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "Formula",
    "FormulaValidation",
    "validate_formula",
]
