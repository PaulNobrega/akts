"""
Physical plausibility checks for kinetic parameters.

This module implements filters to reject models with unphysical parameter values
that produce poor predictions or unrealistic confidence intervals.
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from .datatypes import FitResult
from .utils import EA_BOUNDS


# Physical plausibility thresholds
MAX_REASONABLE_A = 1e20  # Pre-exponential factor > 1e20 s^-1 is unphysical
MIN_REASONABLE_A = 1e-10  # Pre-exponential factor < 1e-10 s^-1 is suspiciously slow
MIN_REASONABLE_EA, MAX_REASONABLE_EA = EA_BOUNDS


def check_parameter_plausibility(
    parameters: Dict[str, float],
    strict: bool = False
) -> Tuple[bool, List[str]]:
    """
    Check if kinetic parameters are physically plausible.

    Parameters
    ----------
    parameters : Dict[str, float]
        Dictionary of fitted parameters (must include 'Ea' and 'A' or 'A1', 'A2', etc.)
    strict : bool, default=False
        If True, apply stricter thresholds

    Returns
    -------
    is_plausible : bool
        True if all parameters pass plausibility checks
    issues : List[str]
        List of plausibility issues found (empty if is_plausible=True)

    Notes
    -----
    Common causes of unphysical parameters:
    - Model overfitting to noise
    - Incorrect initial guesses
    - Numerical instability in optimization
    - Model structure doesn't match the data
    """
    issues = []

    # Check activation energy
    if 'Ea' in parameters:
        Ea = parameters['Ea']
        Ea_kJ = Ea / 1000.0  # Convert to kJ/mol for reporting

        if strict:
            if Ea < MIN_REASONABLE_EA:
                issues.append(f"Ea = {Ea_kJ:.1f} kJ/mol is too low (< {MIN_REASONABLE_EA/1000:.1f} kJ/mol)")
            if Ea > MAX_REASONABLE_EA:
                issues.append(f"Ea = {Ea_kJ:.1f} kJ/mol is too high (> {MAX_REASONABLE_EA/1000:.1f} kJ/mol)")
        else:
            # More lenient checks for non-strict mode
            if Ea < 0:
                issues.append(f"Ea = {Ea_kJ:.1f} kJ/mol is negative (physically impossible)")
            if Ea > MAX_REASONABLE_EA:
                issues.append(f"Ea = {Ea_kJ:.1f} kJ/mol is extremely high (> {MAX_REASONABLE_EA/1000:.0f} kJ/mol)")

    # Check pre-exponential factor(s)
    a_keys = [k for k in parameters.keys() if k.startswith('A')]
    for key in a_keys:
        A = parameters[key]

        if A < 0:
            issues.append(f"{key} = {A:.2e} s^-1 is negative (physically impossible)")
        elif A > MAX_REASONABLE_A:
            issues.append(f"{key} = {A:.2e} s^-1 is extremely high (> {MAX_REASONABLE_A:.0e} s^-1)")
        elif strict and A < MIN_REASONABLE_A:
            issues.append(f"{key} = {A:.2e} s^-1 is extremely low (< {MIN_REASONABLE_A:.0e} s^-1)")

    # Check shape parameters (m, n for Sestak-Berggren)
    for param in ['m', 'n', 'm1', 'n1', 'm2', 'n2']:
        if param in parameters:
            value = parameters[param]
            # Shape parameters should be between -1 and 10 for most physical processes
            if value < -1 or value > 10:
                if strict:
                    issues.append(f"{param} = {value:.2f} is outside typical range [-1, 10]")

    is_plausible = len(issues) == 0
    return is_plausible, issues


def add_plausibility_to_fit(fit_result: FitResult) -> FitResult:
    """
    Add plausibility check to a FitResult object.

    Parameters
    ----------
    fit_result : FitResult
        Fitted model result

    Returns
    -------
    FitResult
        Same object with is_physically_plausible and plausibility_issues attributes set

    Notes
    -----
    This function modifies the FitResult object in-place and also returns it.
    """
    if hasattr(fit_result, 'parameters') and fit_result.parameters:
        is_plausible, issues = check_parameter_plausibility(
            fit_result.parameters,
            strict=False  # Use lenient checks by default
        )
        fit_result.is_physically_plausible = is_plausible
        fit_result.plausibility_issues = issues if issues else None
    else:
        fit_result.is_physically_plausible = None
        fit_result.plausibility_issues = None

    return fit_result


def filter_implausible_bootstrap_replicates(
    parameter_distributions: Dict[str, np.ndarray],
    strict: bool = True
) -> Tuple[Dict[str, np.ndarray], int, int]:
    """
    Filter bootstrap replicates with unphysical parameters.

    Parameters
    ----------
    parameter_distributions : Dict[str, np.ndarray]
        Bootstrap parameter distributions (each array has shape (n_replicates,))
    strict : bool, default=True
        If True, apply strict filtering (recommended for bootstrap)

    Returns
    -------
    filtered_distributions : Dict[str, np.ndarray]
        Filtered parameter distributions with only plausible replicates
    n_total : int
        Total number of replicates before filtering
    n_filtered : int
        Number of replicates filtered out

    Notes
    -----
    Bootstrap replicates with extreme parameters often produce unrealistic
    predictions and inflate confidence intervals. Filtering improves CI quality.
    """
    if not parameter_distributions:
        return parameter_distributions, 0, 0

    # Get number of replicates from first parameter
    first_key = next(iter(parameter_distributions.keys()))
    n_replicates = len(parameter_distributions[first_key])

    # Check plausibility for each replicate
    keep_mask = np.ones(n_replicates, dtype=bool)

    for i in range(n_replicates):
        # Build parameter dict for this replicate
        params = {key: values[i] for key, values in parameter_distributions.items()}

        # Check plausibility
        is_plausible, _ = check_parameter_plausibility(params, strict=strict)
        keep_mask[i] = is_plausible

    # Filter all distributions
    filtered_distributions = {
        key: values[keep_mask]
        for key, values in parameter_distributions.items()
    }

    n_filtered = n_replicates - np.sum(keep_mask)

    return filtered_distributions, n_replicates, n_filtered


def calculate_ci_quality_score(
    conversion_mean: np.ndarray,
    conversion_lower: Optional[np.ndarray],
    conversion_upper: Optional[np.ndarray]
) -> Optional[float]:
    """
    Calculate a quality score for confidence interval bands.

    Parameters
    ----------
    conversion_mean : np.ndarray
        Mean conversion over time
    conversion_lower : np.ndarray or None
        Lower CI bound
    conversion_upper : np.ndarray or None
        Upper CI bound

    Returns
    -------
    score : float or None
        CI quality score (higher is better), or None if CIs not available
        Score is based on:
        - CI width (narrower is better)
        - Monotonicity (smooth bands preferred)
        - Physical bounds (within [0, 1.2])

    Notes
    -----
    Score ranges approximately 0-100:
    - > 80: Excellent CI bands
    - 60-80: Good CI bands
    - 40-60: Acceptable CI bands
    - < 40: Poor CI bands (may indicate model/bootstrap issues)
    """
    if conversion_lower is None or conversion_upper is None:
        return None

    conversion_lower = np.asarray(conversion_lower)
    conversion_upper = np.asarray(conversion_upper)
    conversion_mean = np.asarray(conversion_mean)

    if len(conversion_lower) == 0 or len(conversion_upper) == 0:
        return None

    # 1. CI width penalty (narrower is better)
    ci_width = conversion_upper - conversion_lower
    mean_width = np.mean(ci_width)
    max_width = np.max(ci_width)

    # Penalize if mean width > 50% or max width > 80%
    width_score = 100.0 * np.exp(-2.0 * mean_width) * np.exp(-max_width)

    # 2. Monotonicity score (CI bands should widen monotonically)
    # Small decreases are OK (due to noise), but large decreases are bad
    width_changes = np.diff(ci_width)
    n_large_decreases = np.sum(width_changes < -0.05)  # Decreases > 5%
    monotonicity_score = 100.0 * np.exp(-0.5 * n_large_decreases)

    # 3. Bounds score (should stay within [0, 1.2] for conversion)
    out_of_bounds = (
        np.sum(conversion_lower < -0.1) +
        np.sum(conversion_upper > 1.2)
    )
    bounds_score = 100.0 * np.exp(-0.2 * out_of_bounds)

    # 4. Relative width score (CI width relative to mean)
    # For predictions, CI width should scale reasonably with mean
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        relative_width = ci_width / (conversion_mean + 0.01)
        mean_relative_width = np.nanmean(relative_width)
        relative_width_score = 100.0 * np.exp(-mean_relative_width)

    # Combine scores (weighted average)
    overall_score = (
        0.4 * width_score +
        0.3 * bounds_score +
        0.2 * monotonicity_score +
        0.1 * relative_width_score
    )

    return float(overall_score)
