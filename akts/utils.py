# utils.py
import numpy as np
from scipy.signal import savgol_filter
from bisect import bisect_right
import warnings
from typing import Callable, Dict, List, Tuple, Any

# Ideal gas constant (J/mol·K)
R_GAS = 8.31446261815324
EA_BOUNDS = (5e3, 1000e3)  # J/mol, fitting search range for every Ea parameter

# --- Canonical time-unit-to-seconds conversions ---
# Single source of truth for "month"/"year" as seconds -- other modules
# (helpers.py, reporting.py, plotting.py, regulatory_plots.py) previously each
# hardcoded their own approximation (some used a 30-day month / 365-day year,
# others 30.44 / 365.25), which drifted apart and gave slightly different
# shelf-life numbers depending on which module computed them.
#
# SECONDS_PER_YEAR uses the average Gregorian calendar year (365.2425 days --
# accounts for the actual leap-year rule: +1 day every 4 years, -1 every 100,
# +1 every 400), which is more accurate than the common Julian-year
# approximation (365.25 days) for any conversion meant to track calendar time.
# SECONDS_PER_MONTH is exactly SECONDS_PER_YEAR / 12, so "12 months" and
# "1 year" always agree exactly.
SECONDS_PER_MINUTE = 60.0
SECONDS_PER_HOUR = 3600.0
SECONDS_PER_DAY = 86400.0
SECONDS_PER_WEEK = 7.0 * SECONDS_PER_DAY
SECONDS_PER_YEAR = 365.2425 * SECONDS_PER_DAY
SECONDS_PER_MONTH = SECONDS_PER_YEAR / 12.0

# Every accepted spelling of each unit, mapped to its value in seconds.
TIME_UNITS_TO_SECONDS: Dict[str, float] = {
    'second': 1.0, 'seconds': 1.0, 's': 1.0,
    'minute': SECONDS_PER_MINUTE, 'minutes': SECONDS_PER_MINUTE, 'min': SECONDS_PER_MINUTE,
    'hour': SECONDS_PER_HOUR, 'hours': SECONDS_PER_HOUR, 'h': SECONDS_PER_HOUR, 'hr': SECONDS_PER_HOUR,
    'day': SECONDS_PER_DAY, 'days': SECONDS_PER_DAY, 'd': SECONDS_PER_DAY,
    'week': SECONDS_PER_WEEK, 'weeks': SECONDS_PER_WEEK,
    'month': SECONDS_PER_MONTH, 'months': SECONDS_PER_MONTH,
    'year': SECONDS_PER_YEAR, 'years': SECONDS_PER_YEAR, 'yr': SECONDS_PER_YEAR,
}


def seconds_per_time_unit(unit: str) -> float:
    """Looks up TIME_UNITS_TO_SECONDS case/whitespace-insensitively.

    Returns 1.0 (i.e. treats the value as already in seconds) for an unknown
    unit, matching the historical fallback behavior of reporting.py's plot
    helpers -- callers that must reject unknown units (e.g. user-facing
    "predict=(3, 'year')" parsing) should validate against
    TIME_UNITS_TO_SECONDS directly instead of relying on this fallback.
    """
    return TIME_UNITS_TO_SECONDS.get(unit.lower().strip(), 1.0)

def numerical_diff(x: np.ndarray, y: np.ndarray, *, window_length: int = 5, polyorder: int = 2) -> np.ndarray:
    """
    Calculates the derivative dy/dx using Savitzky-Golay filter.
    Handles potential issues with filter window size.
    """
    if len(x) < 3:
        warnings.warn("Not enough data points for differentiation, returning zeros.")
        return np.zeros_like(x)

    # Ensure window_length is odd and less than data length
    effective_window_length = min(window_length, len(x))
    if effective_window_length % 2 == 0:
        effective_window_length -= 1
    effective_window_length = max(3, effective_window_length) # Minimum required window

    # Ensure polyorder is less than window length
    effective_polyorder = min(polyorder, effective_window_length - 1)
    effective_polyorder = max(1, effective_polyorder) # Polyorder must be at least 1 for derivative

    # Check if differentiation is possible with the adjusted parameters
    if effective_window_length > len(x):
         warnings.warn(f"Effective window length ({effective_window_length}) > data length ({len(x)}). Cannot apply Savitzky-Golay filter. Returning simple gradient.")
         # Fallback to simple gradient on original data
         dydx = np.gradient(y, x, edge_order=2)
         dydx[~np.isfinite(dydx)] = 0.0
         return dydx

    # Calculate time steps, handle potential zero steps for gradient fallback
    dx = np.diff(x)
    if np.any(dx <= 0):
        warnings.warn("Non-positive steps found in x-data for differentiation. Results may be inaccurate if using gradient.")
        # SavGol might handle this better if spacing is somewhat regular

    try:
        # Use Savitzky-Golay filter for smoothing and differentiation
        # deriv=1 calculates the first derivative, delta_t=1 assumes unit spacing
        # We need to divide by the actual spacing dx/dt
        # A more robust approach for potentially uneven spacing: smooth y, then use gradient.
        y_smooth = savgol_filter(y, window_length=effective_window_length, polyorder=effective_polyorder, mode='interp')
        dydx = np.gradient(y_smooth, x, edge_order=2) # Use gradient on smoothed data

    except ValueError as e:
         warnings.warn(f"Savitzky-Golay filter failed: {e}. Returning simple gradient.")
         # Fallback to simple gradient on original data if Sav-Gol fails
         dydx = np.gradient(y, x, edge_order=2)


    # Handle potential NaNs or Infs resulting from differentiation
    dydx[~np.isfinite(dydx)] = 0.0
    return dydx


def calculate_aic(rss: float, n_params: int, n_datapoints: int) -> float:
    """Calculates Akaike Information Criterion corrected for small sample sizes (AICc)."""
    if n_datapoints <= 0: return np.inf
    if rss <= 0: rss = 1e-12 # Avoid log(0) or division by zero issues

    k = n_params
    n = n_datapoints

    # Basic AIC = n * ln(RSS/n) + 2k (information-theory based version)
    # Or AIC = 2k - 2*logLikelihood. Assuming normal errors, logLik = -n/2*log(2pi) - n/2*log(RSS/n) - n/2
    # AIC = 2k + n*log(2pi) + n*log(RSS/n) + n . Ignoring constants: 2k + n*log(RSS/n)
    aic = n * np.log(rss / n) + 2 * k

    # AICc (Corrected AIC) - recommended
    denominator = n - k - 1
    if denominator > 0:
        aicc = aic + (2 * k * (k + 1)) / denominator
    else:
        # If denominator is zero or negative, AICc is infinite or undefined
        # Return infinity as a penalty for overfitting severely
        aicc = np.inf
        warnings.warn(f"AICc correction term denominator is non-positive ({denominator}). Indicates severe overfitting (n <= k+1). Returning AICc=inf.")

    return aicc

def calculate_bic(rss: float, n_params: int, n_datapoints: int) -> float:
    """Calculates Bayesian Information Criterion."""
    k = n_params
    n = n_datapoints
    if n <= 0: return np.inf
    if rss <= 0: rss = 1e-12 # Avoid log(0)

    # BIC = n * ln(RSS/n) + k * ln(n)
    bic = n * np.log(rss / n) + k * np.log(n)
    return bic

def calculate_akaike_weights(aicc_values: List[float]) -> List[float]:
    """
    Converts a list of AICc values into Akaike weights: the estimated probability
    that each model is the best one in the candidate set, given the data.

    w_i = exp(-0.5 * (AICc_i - AICc_min)) / sum_j(exp(-0.5 * (AICc_j - AICc_min)))

    Models with a non-finite AICc get weight 0.0. If every value is non-finite,
    all weights are 0.0.
    """
    aicc = np.asarray(aicc_values, dtype=float)
    finite = np.isfinite(aicc)
    if not finite.any():
        return [0.0] * len(aicc)

    delta = np.full_like(aicc, np.inf)
    delta[finite] = aicc[finite] - np.min(aicc[finite])

    rel_likelihood = np.zeros_like(aicc)
    rel_likelihood[finite] = np.exp(-0.5 * delta[finite])

    total = rel_likelihood.sum()
    if total <= 0:
        return [0.0] * len(aicc)
    return (rel_likelihood / total).tolist()

def calculate_adjusted_r_squared(r_squared: float, n_params: int, n_datapoints: int) -> float:
    """
    Adjusted R^2 penalizes additional parameters, unlike raw R^2 which never decreases
    when a model gains a parameter. Returns NaN if there are too few points relative
    to the parameter count for the adjustment to be defined (n_datapoints <= n_params + 1).
    """
    n, k = n_datapoints, n_params
    if not np.isfinite(r_squared) or n <= k + 1:
        return np.nan
    return 1.0 - (1.0 - r_squared) * (n - 1) / (n - k - 1)

def get_temperature_interpolator(time: np.ndarray, temperature: np.ndarray) -> Callable:
    """Creates a fast piecewise-linear temperature function T(t) [K].

    Matches scipy's ``interp1d(kind='linear', fill_value='extrapolate')`` -- linear
    interpolation inside the data, linear extrapolation from the end segments
    outside it -- but is called once per ODE right-hand-side evaluation, so it is
    implemented without interp1d's per-call validation overhead (~100 us/call):

    - exactly constant temperature (the usual isothermal stability study) returns
      a constant function (~0.2 us/call);
    - otherwise it uses ``np.interp`` plus explicit end-slope extrapolation
      (~7 us/call).

    Scalar input returns a Python float; array input returns an ndarray.
    """
    time = np.asarray(time, dtype=float)
    temperature = np.asarray(temperature, dtype=float)
    if len(time) != len(temperature):
        raise ValueError("Time and temperature arrays must have the same length.")
    if len(time) < 2:
        if len(temperature) == 1:
            const_temp = float(temperature[0])
            warnings.warn("Only one data point provided for temperature interpolation. Returning constant temperature function.")
            return _constant_temperature_function(const_temp)
        raise ValueError("Cannot create temperature interpolator with less than 2 points.")

    if np.all(temperature == temperature[0]):
        return _constant_temperature_function(float(temperature[0]))

    order = np.argsort(time, kind='stable')
    t_sorted, T_sorted = time[order], temperature[order]
    t_lo, t_hi = float(t_sorted[0]), float(t_sorted[-1])
    T_lo, T_hi = float(T_sorted[0]), float(T_sorted[-1])
    dt_lo = float(t_sorted[1] - t_sorted[0])
    dt_hi = float(t_sorted[-1] - t_sorted[-2])
    slope_lo = float(T_sorted[1] - T_sorted[0]) / dt_lo if dt_lo != 0 else 0.0
    slope_hi = float(T_sorted[-1] - T_sorted[-2]) / dt_hi if dt_hi != 0 else 0.0

    t_list, T_list = t_sorted.tolist(), T_sorted.tolist()

    def temperature_at(t):
        if isinstance(t, (float, int, np.floating, np.integer)):
            # Scalar fast path (the ODE solver's call pattern): pure-Python
            # bisection avoids numpy's per-call array overhead.
            x = float(t)
            if x <= t_lo:
                return T_lo + slope_lo * (x - t_lo)
            if x >= t_hi:
                return T_hi + slope_hi * (x - t_hi)
            i = bisect_right(t_list, x) - 1
            t0, t1 = t_list[i], t_list[i + 1]
            T0 = T_list[i]
            return T0 + (T_list[i + 1] - T0) * (x - t0) / (t1 - t0) if t1 > t0 else T0
        t_arr = np.asarray(t, dtype=float)
        T = np.interp(t_arr, t_sorted, T_sorted)
        T = np.where(t_arr < t_lo, T_lo + slope_lo * (t_arr - t_lo), T)
        T = np.where(t_arr > t_hi, T_hi + slope_hi * (t_arr - t_hi), T)
        return float(T) if T.ndim == 0 else T

    return temperature_at


def _constant_temperature_function(value: float) -> Callable:
    def constant_temperature(t):
        if isinstance(t, (float, int, np.floating, np.integer)) or np.ndim(t) == 0:
            return value
        return np.full(np.shape(t), value)
    return constant_temperature

# --- NEW Helper Function ---
def construct_profile(segments: List[Dict[str, Any]], points_per_segment: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """
    Constructs a time-temperature profile from a list of segments.

    Args:
        segments: A list of dictionaries, where each dictionary defines a segment.
                  Required keys depend on the 'type':
                  - {'type': 'isothermal', 'duration': float, 'temperature': float}
                  - {'type': 'ramp', 'duration': float, 'start_temp': float, 'end_temp': float}
                  - {'type': 'custom', 'time_array': np.ndarray, 'temp_array': np.ndarray}
                    (Note: time_array for custom should be relative to segment start, i.e., start at 0)
        points_per_segment: Number of points to generate for isothermal/ramp segments.

    Returns:
        A tuple containing (combined_time_array_sec, combined_temp_array_K).
    """
    combined_time = [0.0]  # Start at time 0
    # Determine initial temperature from the first segment
    first_segment = segments[0]
    if first_segment['type'] == 'isothermal':
        current_temp = first_segment['temperature']
    elif first_segment['type'] == 'ramp':
        current_temp = first_segment['start_temp']
    elif first_segment['type'] == 'custom':
        if len(first_segment['temp_array']) == 0:
            raise ValueError("Custom segment temp_array cannot be empty.")
        current_temp = first_segment['temp_array'][0]
    else:
        raise ValueError(f"Unknown segment type: {first_segment.get('type')}")
    combined_temp = [current_temp]
    current_time = 0.0

    for i, segment in enumerate(segments):
        seg_type = segment.get('type')
        duration = segment.get('duration')  # Duration is required for isothermal/ramp

        if seg_type == 'isothermal':
            if duration is None or 'temperature' not in segment:
                raise ValueError("Isothermal segment requires 'duration' and 'temperature'.")
            end_time = current_time + duration
            temp = segment['temperature']
            # Add points within the segment (excluding start point if not first segment)
            num_points = max(2, points_per_segment)
            seg_times = np.linspace(current_time, end_time, num_points)
            seg_temps = np.full(num_points, temp)
            if i > 0:
                combined_time.extend(seg_times[1:])
                combined_temp.extend(seg_temps[1:])
            else:
                combined_time = list(seg_times)  # Overwrite initial [0.0]
                combined_temp = list(seg_temps)
            current_time = end_time
            current_temp = temp

        elif seg_type == 'ramp':
            if duration is None or 'start_temp' not in segment or 'end_temp' not in segment:
                raise ValueError("Ramp segment requires 'duration', 'start_temp', and 'end_temp'.")
            if i > 0 and not np.isclose(segment['start_temp'], current_temp):
                warnings.warn(f"Segment {i}: Ramp start_temp {segment['start_temp']} does not match previous end_temp {current_temp}.")
            end_time = current_time + duration
            start_temp = segment['start_temp']
            end_temp = segment['end_temp']
            num_points = max(2, points_per_segment)
            seg_times = np.linspace(current_time, end_time, num_points)
            seg_temps = np.linspace(start_temp, end_temp, num_points)
            if i > 0:
                combined_time.extend(seg_times[1:])
                combined_temp.extend(seg_temps[1:])
            else:
                combined_time = list(seg_times)
                combined_temp = list(seg_temps)
            current_time = end_time
            current_temp = end_temp

        elif seg_type == 'custom':
            if 'time_array' not in segment or 'temp_array' not in segment:
                raise ValueError("Custom segment requires 'time_array' and 'temp_array'.")
            seg_times_relative = segment['time_array']
            seg_temps = segment['temp_array']
            if len(seg_times_relative) != len(seg_temps) or len(seg_times_relative) == 0:
                raise ValueError("Custom segment time/temp arrays must be non-empty and equal length.")
            if i > 0 and not np.isclose(seg_temps[0], current_temp):
                warnings.warn(f"Segment {i}: Custom start_temp {seg_temps[0]} does not match previous end_temp {current_temp}.")
            seg_times_absolute = seg_times_relative + current_time
            end_time = seg_times_absolute[-1]
            if i > 0:
                start_index = 1 if np.isclose(seg_times_absolute[0], combined_time[-1]) else 0
                combined_time.extend(seg_times_absolute[start_index:])
                combined_temp.extend(seg_temps[start_index:])
            else:
                combined_time = list(seg_times_absolute)
                combined_temp = list(seg_temps)
            current_time = end_time
            current_temp = seg_temps[-1]

        else:
            raise ValueError(f"Unknown segment type '{seg_type}' in segment {i}.")

    return np.asarray(combined_time, dtype=float), np.asarray(combined_temp, dtype=float)


def is_isothermal(temperature: np.ndarray, tolerance_K: float = 0.5) -> bool:
    """
    Check if a temperature profile is constant within tolerance.

    Parameters
    ----------
    temperature : np.ndarray
        Temperature array in Kelvin
    tolerance_K : float, optional
        Maximum standard deviation in Kelvin to consider isothermal (default 0.5 K)

    Returns
    -------
    bool
        True if temperature profile is isothermal within tolerance

    Notes
    -----
    Uses standard deviation as the test metric. Single-point profiles
    are considered isothermal by definition.
    """
    if len(temperature) < 2:
        return True
    return np.std(temperature) < tolerance_K


def check_physical_plausibility(parameters: Dict[str, float],
                                strict: bool = False) -> Tuple[bool, List[str]]:
    """
    Check if fitted kinetic parameters are physically plausible.

    Flags parameters that fall outside typical ranges for solid-state/solution kinetics:
    - Ea (activation energy): 30-180 kJ/mol typical, 5-1000 kJ/mol absolute
    - A (pre-exponential): 10^6 to 10^18 s^-1 typical, 10^-2 to 10^25 absolute

    Parameters
    ----------
    parameters : Dict[str, float]
        Fitted parameters (Ea in J/mol, A in s^-1, etc.)
    strict : bool, optional
        If True, use tighter "typical" bounds. If False (default), use wider
        "absolute" bounds that catch only clearly unphysical values.

    Returns
    -------
    is_plausible : bool
        True if all parameters are within plausible ranges
    issues : List[str]
        List of issue descriptions (empty if plausible)

    Notes
    -----
    Typical ranges (strict=True):
    - Ea: 30-180 kJ/mol (most solid-state/protein degradation reactions)
    - A: 10^6 to 10^18 s^-1

    Absolute ranges (strict=False):
    - Ea: 5-1000 kJ/mol, same as EA_BOUNDS (catches clearly unphysical values only)
    - A: 10^-2 to 10^25 s^-1

    Very low Ea (<5 kJ/mol) suggests diffusion-limited or barrierless processes.
    Ea of 400-800 kJ/mol is typical of cooperative protein unfolding.
    Very low A (<0.1 s^-1) suggests highly ordered transition states.
    Very high A (>10^25 s^-1) exceeds molecular collision frequencies.

    References
    ----------
    - Brown et al. (2000). Handbook of Thermal Analysis and Calorimetry.
    - ICTAC Kinetics Committee recommendations (2011).
    """
    issues = []

    # Define bounds based on strictness
    if strict:
        Ea_min, Ea_max = 30e3, 180e3  # 30-180 kJ/mol typical
        A_min, A_max = 1e6, 1e18       # Typical pre-exponential range
    else:
        Ea_min, Ea_max = EA_BOUNDS
        A_min, A_max = 1e-2, 1e25      # Very permissive range

    for p_name, p_val in parameters.items():
        if p_name.startswith("Ea"):
            if p_val < Ea_min:
                issues.append(f"{p_name} = {p_val/1000:.1f} kJ/mol is below plausible minimum ({Ea_min/1000:.0f} kJ/mol)")
            elif p_val > Ea_max:
                issues.append(f"{p_name} = {p_val/1000:.1f} kJ/mol exceeds plausible maximum ({Ea_max/1000:.0f} kJ/mol)")

        elif p_name.startswith("A"):
            if p_val < A_min:
                issues.append(f"{p_name} = {p_val:.2e} s^-1 is below plausible minimum ({A_min:.0e} s^-1)")
            elif p_val > A_max:
                issues.append(f"{p_name} = {p_val:.2e} s^-1 exceeds plausible maximum ({A_max:.0e} s^-1)")

    is_plausible = len(issues) == 0
    return is_plausible, issues


def calculate_durbin_watson(residuals: np.ndarray) -> float:
    """
    Calculate the Durbin-Watson statistic for autocorrelation in residuals.

    The Durbin-Watson statistic ranges from 0 to 4:
    - DW ≈ 2: No autocorrelation (ideal)
    - DW < 2: Positive autocorrelation (systematic underestimation then overestimation)
    - DW > 2: Negative autocorrelation (oscillating pattern)
    - DW ≈ 0 or 4: Strong autocorrelation

    Parameters
    ----------
    residuals : np.ndarray
        Residuals from a model fit (observed - predicted)

    Returns
    -------
    float
        Durbin-Watson statistic. Returns NaN if insufficient data (<2 points)
        or if residuals contain NaN/inf values.

    Notes
    -----
    DW = Σ(residual[i] - residual[i-1])² / Σ(residual[i]²)

    For model validation:
    - 1.5 < DW < 2.5: Acceptable (no strong autocorrelation)
    - DW < 1.5 or DW > 2.5: Check for systematic model error

    References
    ----------
    Durbin, J., & Watson, G. S. (1950). Testing for serial correlation in
    least squares regression. Biometrika, 37(3/4), 409-428.
    """
    # Remove NaN/inf values
    valid_mask = np.isfinite(residuals)
    valid_residuals = residuals[valid_mask]

    if len(valid_residuals) < 2:
        warnings.warn("Durbin-Watson requires at least 2 finite residuals")
        return np.nan

    # Calculate DW statistic
    diff_squared = np.sum(np.diff(valid_residuals) ** 2)
    residual_squared = np.sum(valid_residuals ** 2)

    if residual_squared == 0:
        # Perfect fit (no residuals) - undefined DW
        return np.nan

    dw = diff_squared / residual_squared

    return float(dw)


def export_prediction_report(
    prediction,
    bootstrap_result=None,
    path_prefix: str = 'prediction',
    include_plot: bool = True,
    time_units: str = 'seconds',
    **csv_kwargs
):
    """
    One-call export of prediction results to CSV and optional plot.

    Convenience function for exporting prediction results with confidence intervals
    to CSV format and generating a standard plot. Ideal for quick exports without
    writing custom code.

    Parameters
    ----------
    prediction : PredictionResult
        Prediction result from predict_conversion()
    bootstrap_result : BootstrapResult, optional
        Bootstrap result for parameter uncertainty table
    path_prefix : str, default='prediction'
        Prefix for output files:
        - {path_prefix}.csv: prediction data
        - {path_prefix}_bootstrap.csv: bootstrap CI summary (if provided)
        - {path_prefix}_plot.png: plot (if include_plot=True)
    include_plot : bool, default=True
        Whether to generate and save a plot
    time_units : str, default='seconds'
        Time units for plot labels ('seconds', 'hours', 'days', 'months', 'years')
    **csv_kwargs
        Additional arguments passed to to_csv() (e.g., sep='\\t', float_format='%.6f')

    Returns
    -------
    dict
        Dictionary with paths to generated files:
        {'prediction_csv': str, 'bootstrap_csv': str (optional), 'plot': str (optional)}

    Examples
    --------
    >>> from akts import predict_conversion, run_bootstrap, export_prediction_report
    >>>
    >>> # Generate prediction with bootstrap CIs
    >>> prediction = predict_conversion(...)
    >>> bootstrap = run_bootstrap(...)
    >>>
    >>> # Export everything in one call
    >>> files = export_prediction_report(
    ...     prediction=prediction,
    ...     bootstrap_result=bootstrap,
    ...     path_prefix='stability_3year',
    ...     time_units='months'
    ... )
    >>> print(f"Exported: {files}")

    >>> # Just CSV, no plot
    >>> files = export_prediction_report(
    ...     prediction=prediction,
    ...     path_prefix='data',
    ...     include_plot=False
    ... )
    """
    from pathlib import Path

    output_files = {}

    # Export prediction CSV
    prediction_csv = f"{path_prefix}.csv"
    prediction.to_csv(prediction_csv, **csv_kwargs)
    output_files['prediction_csv'] = str(Path(prediction_csv).absolute())

    # Export bootstrap summary if provided
    if bootstrap_result is not None:
        bootstrap_csv = f"{path_prefix}_bootstrap.csv"
        summary_df = bootstrap_result.summary_dataframe()
        summary_df.to_csv(bootstrap_csv, index=False, **csv_kwargs)
        output_files['bootstrap_csv'] = str(Path(bootstrap_csv).absolute())

    # Generate plot if requested
    if include_plot:
        try:
            import matplotlib.pyplot as plt

            # Time unit conversion (seconds -> time_units, i.e. the inverse of
            # TIME_UNITS_TO_SECONDS)
            time_factor = 1.0 / seconds_per_time_unit(time_units)

            fig, ax = plt.subplots(figsize=(10, 6))

            # Convert time
            time_plot = prediction.time * time_factor
            conversion_pct = prediction.conversion * 100

            # Plot prediction
            ax.plot(time_plot, conversion_pct, 'b-', linewidth=2, label='Prediction')

            # Plot confidence interval if available
            if prediction.conversion_ci is not None:
                ci_lower_pct = prediction.conversion_ci[0] * 100
                ci_upper_pct = prediction.conversion_ci[1] * 100
                ax.fill_between(
                    time_plot,
                    ci_lower_pct,
                    ci_upper_pct,
                    alpha=0.3,
                    color='blue',
                    label='95% Confidence Interval'
                )

            ax.set_xlabel(f'Time ({time_units})', fontsize=12, fontweight='bold')
            ax.set_ylabel('Degradation (%)', fontsize=12, fontweight='bold')
            ax.set_title('Kinetic Prediction', fontsize=14, fontweight='bold')
            ax.legend(loc='best', fontsize=10)
            ax.grid(True, alpha=0.3)

            plot_path = f"{path_prefix}_plot.png"
            plt.tight_layout()
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close(fig)

            output_files['plot'] = str(Path(plot_path).absolute())

        except ImportError:
            warnings.warn("matplotlib not available. Skipping plot generation.")
        except Exception as e:
            warnings.warn(f"Failed to generate plot: {e}")

    return output_files