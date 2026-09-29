# utils.py
import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from scipy.stats import linregress
import warnings
from typing import Callable, Dict, List, Tuple, Union, Any  # Added List, Tuple, Union, Any

# Ideal gas constant (J/mol·K)
R_GAS = 8.31446261815324

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
    """Creates an interpolation function for temperature T(t)."""
    if len(time) != len(temperature):
        raise ValueError("Time and temperature arrays must have the same length.")
    if len(time) < 2:
        if len(temperature) == 1:
            const_temp = temperature[0]
            warnings.warn("Only one data point provided for temperature interpolation. Returning constant temperature function.")
            return lambda t: np.full_like(np.asarray(t), const_temp)  # Return array for vectorization
        else:
            raise ValueError("Cannot create temperature interpolator with less than 2 points.")
    # Use linear interpolation, handle edge cases by filling with endpoint values
    # Allow extrapolation for times outside the original range
    return interp1d(time, temperature, kind='linear', bounds_error=False, fill_value="extrapolate")

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
    - Ea (activation energy): 30-180 kJ/mol typical, 10-400 kJ/mol absolute
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
    - Ea: 10-400 kJ/mol (catches clearly unphysical values only)
    - A: 10^-2 to 10^25 s^-1

    Very low Ea (<10 kJ/mol) suggests diffusion-limited or barrierless processes.
    Very high Ea (>400 kJ/mol) is uncommon except for bond-breaking reactions.
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
        Ea_min, Ea_max = 10e3, 400e3  # 10-400 kJ/mol absolute
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

            # Time unit conversion
            time_conversion = {
                'seconds': 1.0,
                'hours': 1 / 3600,
                'days': 1 / (24 * 3600),
                'months': 1 / (30.44 * 24 * 3600),
                'years': 1 / (365.25 * 24 * 3600)
            }
            time_factor = time_conversion.get(time_units, 1.0)

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