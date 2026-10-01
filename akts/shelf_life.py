"""
ICH Q1E-compliant shelf-life determination using confidence-band crossing method.

This module implements regulatory-compliant shelf-life calculation per ICH Q1E guidance:
- One-sided 95% confidence bounds (not two-sided intervals)
- Confidence-band crossing method (not just percentiles of expiry times)
- Appropriate bound selection (lower for decreasing, upper for increasing attributes)

References
----------
ICH Q1E Guideline: https://database.ich.org/sites/default/files/Q1E_Guideline.pdf
"""
import numpy as np
from typing import Dict, Optional
import warnings


def calculate_shelf_life_ich_q1e(
    time: np.ndarray,
    conversion_mean: np.ndarray,
    conversion_lower: Optional[np.ndarray],
    conversion_upper: Optional[np.ndarray],
    specification_limit: float,
    attribute_direction: str = 'decreasing',
    time_units: str = 'seconds'
) -> Dict:
    """
    Calculate shelf-life per ICH Q1E using confidence-band crossing method.

    ICH Q1E requires that shelf-life be determined as the time when the
    appropriate one-sided 95% confidence bound crosses the specification limit:
    - Decreasing attribute (potency, monomer): lower bound crosses lower spec
    - Increasing attribute (aggregates, impurities): upper bound crosses upper spec

    This implements the "confidence-band crossing method" which is preferred over
    simply taking percentiles of expiry times, because different bootstrap curves
    may cross the specification differently, and some may not cross within the
    prediction horizon.

    Parameters
    ----------
    time : np.ndarray
        Time points (must be 1D array)
    conversion_mean : np.ndarray
        Mean predicted conversion trajectory
    conversion_lower : np.ndarray or None
        Lower confidence bound (mean, used for reference only)
        If None, only mean shelf-life is calculated
    conversion_upper : np.ndarray or None
        Upper confidence bound (95th percentile - conservative for both directions)
        This represents faster degradation/increase and gives conservative shelf-life
        If None, only mean shelf-life is calculated
    specification_limit : float
        Specification limit for the attribute
        - For decreasing (e.g., 90% potency = 0.10 degradation)
        - For increasing (e.g., 5% aggregates = 0.05 conversion)
    attribute_direction : str, default 'decreasing'
        'decreasing' - potency/monomer content (use lower bound)
        'increasing' - aggregates/impurities (use upper bound)
    time_units : str, default 'seconds'
        Time units for display in results

    Returns
    -------
    Dict with keys:
        'shelf_life_mean' : float
            Time when mean crosses specification
        'shelf_life_conservative' : float
            Time when one-sided CI bound crosses specification (ICH Q1E)
        'shelf_life_ich_q1e' : float
            Alias for shelf_life_conservative
        'method' : str
            'ich_q1e_one_sided_ci' or 'mean_only'
        'confidence_level' : float
            0.95
        'attribute_direction' : str
            'decreasing' or 'increasing'
        'has_confidence_interval' : bool
            Whether CI was available
        'time_units' : str
            Time units

    Examples
    --------
    >>> # For decreasing attribute (potency)
    >>> result = calculate_shelf_life_ich_q1e(
    ...     time=time_days,
    ...     conversion_mean=mean_conversion,
    ...     conversion_lower=lower_bound,  # 5th percentile
    ...     conversion_upper=None,
    ...     specification_limit=0.10,  # 10% degradation
    ...     attribute_direction='decreasing',
    ...     time_units='days'
    ... )
    >>> print(f"Shelf-life: {result['shelf_life_ich_q1e']:.1f} days")

    Notes
    -----
    ICH Q1E Compliance:
    - Uses one-sided confidence bounds (5th or 95th percentile)
    - Implements confidence-band crossing method
    - Conservative estimate: lower bound for decreasing, upper for increasing
    - Appropriate for regulatory submissions

    The conservative shelf-life will always be shorter than (or equal to) the
    mean shelf-life for properly configured attributes.
    """
    # Validate inputs
    time = np.asarray(time)
    conversion_mean = np.asarray(conversion_mean)

    if time.ndim != 1:
        raise ValueError(f"time must be 1D array, got shape {time.shape}")
    if conversion_mean.shape != time.shape:
        raise ValueError(f"conversion_mean shape {conversion_mean.shape} must match time shape {time.shape}")

    if attribute_direction not in ('decreasing', 'increasing'):
        raise ValueError(f"attribute_direction must be 'decreasing' or 'increasing', got '{attribute_direction}'")

    # Select the appropriate bound based on attribute direction
    # NOTE: "conversion" represents DEGRADATION or IMPURITY level (increases over time)
    # For decreasing-quality attributes (potency): use UPPER bound on degradation
    # (This is the "lower bound on potency" per ICH Q1E, expressed as degradation)
    if attribute_direction == 'decreasing':
        # Use upper bound (95th percentile of degradation = fastest potency loss)
        relevant_bound = conversion_upper
        comparison = lambda x: x >= specification_limit  # Degradation increases
    elif attribute_direction == 'increasing':
        # Use upper bound (95th percentile of impurity = fastest increase)
        relevant_bound = conversion_upper
        comparison = lambda x: x >= specification_limit  # Impurities increase
    else:
        raise ValueError(f"attribute_direction must be 'decreasing' or 'increasing'")

    # Find crossing time for mean trajectory
    crossing_mean_idx = np.where(comparison(conversion_mean))[0]
    if len(crossing_mean_idx) == 0:
        shelf_life_mean = np.inf
        warnings.warn(
            f"Mean trajectory does not cross specification limit {specification_limit:.4f} "
            f"within prediction horizon (max time: {time[-1]:.2f} {time_units})"
        )
    else:
        shelf_life_mean = _interpolate_crossing(
            time, conversion_mean, specification_limit, crossing_mean_idx[0]
        )

    # Find crossing time for one-sided confidence bound (ICH Q1E requirement)
    if relevant_bound is None:
        shelf_life_conservative = shelf_life_mean
        has_ci = False
        warnings.warn(
            "No confidence interval provided. Returning mean shelf-life only. "
            "For ICH Q1E compliance, provide bootstrap confidence intervals."
        )
    else:
        relevant_bound = np.asarray(relevant_bound)
        if relevant_bound.shape != time.shape:
            raise ValueError(
                f"Confidence bound shape {relevant_bound.shape} must match time shape {time.shape}"
            )

        crossing_ci_idx = np.where(comparison(relevant_bound))[0]
        if len(crossing_ci_idx) == 0:
            shelf_life_conservative = np.inf
            warnings.warn(
                f"Confidence bound does not cross specification limit {specification_limit:.4f} "
                f"within prediction horizon (max time: {time[-1]:.2f} {time_units})"
            )
        else:
            shelf_life_conservative = _interpolate_crossing(
                time, relevant_bound, specification_limit, crossing_ci_idx[0]
            )
        has_ci = True

    return {
        'shelf_life_mean': float(shelf_life_mean),
        'shelf_life_conservative': float(shelf_life_conservative),
        'shelf_life_ich_q1e': float(shelf_life_conservative),  # Alias for clarity
        'method': 'ich_q1e_one_sided_ci' if has_ci else 'mean_only',
        'confidence_level': 0.95,
        'attribute_direction': attribute_direction,
        'has_confidence_interval': has_ci,
        'time_units': time_units,
        'specification_limit': float(specification_limit),
    }


def _interpolate_crossing(
    time: np.ndarray,
    values: np.ndarray,
    threshold: float,
    idx: int
) -> float:
    """
    Interpolate the exact crossing time between grid points.

    Linear interpolation to find when values[idx-1] < threshold <= values[idx]
    or values[idx-1] > threshold >= values[idx].

    Parameters
    ----------
    time : np.ndarray
        Time points
    values : np.ndarray
        Conversion/degradation values
    threshold : float
        Threshold to cross
    idx : int
        Index where crossing first occurs

    Returns
    -------
    float
        Interpolated crossing time
    """
    if idx == 0:
        # Crossing happens at or before first time point
        return float(time[0])

    t1, t2 = time[idx - 1], time[idx]
    y1, y2 = values[idx - 1], values[idx]

    # Linear interpolation: t = t1 + (threshold - y1) * (t2 - t1) / (y2 - y1)
    if abs(y2 - y1) < 1e-12:
        # Values are essentially constant, return left endpoint
        return float(t1)

    t_cross = t1 + (threshold - y1) * (t2 - t1) / (y2 - y1)

    # Clamp to interval [t1, t2] (should be in range but avoid numerical issues)
    t_cross = max(t1, min(t2, t_cross))

    return float(t_cross)


def time_to_specification(
    prediction_result,
    specification_limit: float,
    attribute_direction: str = 'decreasing',
) -> Dict:
    """
    User-friendly wrapper for calculate_shelf_life_ich_q1e.

    Takes a PredictionResult and calculates ICH Q1E-compliant shelf-life.

    Parameters
    ----------
    prediction_result : PredictionResult
        Result from predict_conversion() with confidence intervals
    specification_limit : float
        Specification limit for the attribute
    attribute_direction : str, default 'decreasing'
        'decreasing' or 'increasing'

    Returns
    -------
    Dict
        Same as calculate_shelf_life_ich_q1e()

    Examples
    --------
    >>> from akts.prediction import predict_conversion
    >>> pred = predict_conversion(fit_result, temp_program, bootstrap_result=bootstrap)
    >>> shelf_life = time_to_specification(pred, specification_limit=0.10)
    >>> print(f"Shelf-life: {shelf_life['shelf_life_ich_q1e']/86400:.1f} days")
    """
    # Extract data from PredictionResult
    time = prediction_result.time
    conversion_mean = prediction_result.conversion

    # Extract confidence intervals if available
    if prediction_result.conversion_ci is not None:
        conversion_lower, conversion_upper = prediction_result.conversion_ci
    else:
        conversion_lower, conversion_upper = None, None

    # Infer time units from magnitude
    if len(time) > 0:
        max_time = np.max(time)
        if max_time < 3600:  # Less than 1 hour
            time_units = 'seconds'
        elif max_time < 86400 * 7:  # Less than 1 week
            time_units = 'hours'
        elif max_time < 86400 * 365:  # Less than 1 year
            time_units = 'days'
        else:
            time_units = 'years'
    else:
        time_units = 'seconds'

    return calculate_shelf_life_ich_q1e(
        time=time,
        conversion_mean=conversion_mean,
        conversion_lower=conversion_lower,
        conversion_upper=conversion_upper,
        specification_limit=specification_limit,
        attribute_direction=attribute_direction,
        time_units=time_units
    )
