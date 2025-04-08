# utils.py
import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
import random
from tqdm import tqdm # Optional: for progress bar
from scipy.stats import linregress
import warnings
from typing import Callable, Dict, List, Tuple, Union, Optional, Any  # Added List, Tuple, Union, Any
from .datatypes import KineticDataset # Import KineticDataset

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

# --- Add Ea(alpha) Interpolation ---

def create_ea_interpolator(alpha_values, ea_values):
    """ Creates interpolation function for Ea(alpha). More robust. """
    if alpha_values is None or ea_values is None or \
       len(alpha_values) < 2 or len(ea_values) != len(alpha_values):
        warnings.warn("Invalid input for Ea interpolator creation.")
        return None

    # Filter out non-finite values
    finite_mask = np.isfinite(alpha_values) & np.isfinite(ea_values)
    alpha_finite = alpha_values[finite_mask]
    ea_finite = ea_values[finite_mask]

    if len(alpha_finite) < 2:
        warnings.warn("Not enough finite Ea values to create interpolator.")
        return None

    # Ensure Ea is non-negative and above a minimum threshold
    ea_finite = np.maximum(ea_finite, 1e3)  # Minimum plausible Ea (e.g., 1 kJ/mol)

    # Sort values
    sort_indices = np.argsort(alpha_finite)
    alpha_sorted = alpha_finite[sort_indices]
    ea_sorted = ea_finite[sort_indices]

    # Remove duplicate alpha values
    unique_alpha, unique_idx = np.unique(alpha_sorted, return_index=True)
    if len(unique_alpha) < 2:
        warnings.warn("Not enough unique alpha values after filtering for interpolator.")
        return None
    unique_ea = ea_sorted[unique_idx]

    # Define fill values using the first and last valid Ea
    fill_val = (unique_ea[0], unique_ea[-1])

    try:
        # Use linear interpolation with stable extrapolation
        interpolator = interp1d(
            unique_alpha,
            unique_ea,
            kind='linear',
            bounds_error=False,
            fill_value=fill_val
        )
        return interpolator
    except ValueError as e:
        warnings.warn(f"Failed to create Ea interpolator: {e}")
        return None

# --- Add Bootstrap Logic ---

def bootstrap_prediction(
    experimental_runs,
    iso_analysis_func,
    fit_func,
    predict_func,
    n_bootstrap=500,
    confidence_level=95.0
    ):
    """
    Performs bootstrap resampling to estimate prediction confidence intervals.

    Returns:
        tuple: (median_prediction, confidence_interval, prediction_distribution)
               prediction_distribution: NumPy array of ALL prediction outcomes
                                        (float or np.inf). Returns empty array if none succeed.
    """
    n_runs = len(experimental_runs)
    bootstrap_predictions = []  # Store all prediction outcomes (float or np.inf)
    successful_iterations = 0

    print(f"Starting bootstrap analysis with {n_bootstrap} iterations...")
    for i in tqdm(range(n_bootstrap), desc="Bootstrap Progress"):
        # Resample runs with replacement
        resampled_indices = [random.randint(0, n_runs - 1) for _ in range(n_runs)]
        resampled_runs = [experimental_runs[idx] for idx in resampled_indices]

        try:
            # 1. Iso analysis
            alpha_values_boot, ea_values_boot = iso_analysis_func(resampled_runs)
            if alpha_values_boot is None or len(alpha_values_boot) < 2 or \
               ea_values_boot is None or len(ea_values_boot) != len(alpha_values_boot) or \
               np.all(~np.isfinite(ea_values_boot)):
                continue  # Skip iteration if iso analysis fails

            ea_interpolator_boot = create_ea_interpolator(alpha_values_boot, ea_values_boot)

            # 2. Fit model
            fit_params_boot = fit_func(resampled_runs, ea_interpolator_boot)
            if fit_params_boot is None:
                continue  # Skip iteration if fitting fails

            # 3. Make prediction
            prediction_boot = predict_func(fit_params_boot, ea_interpolator_boot)

            # Store result (float or np.inf), skip None/NaN
            if prediction_boot is not None and (np.isfinite(prediction_boot) or prediction_boot == np.inf):
                bootstrap_predictions.append(prediction_boot)
                successful_iterations += 1

        except Exception as e:
            print(f"Error during bootstrap iteration {i+1}: {e}. Skipping.")

    print(f"Bootstrap analysis complete. {successful_iterations} iterations yielded a prediction outcome (incl. inf).")

    # Convert list to numpy array for calculations
    bootstrap_predictions_arr = np.array(bootstrap_predictions, dtype=float)

    if successful_iterations == 0:  # No successful iterations at all
        print("Error: No valid predictions were generated during bootstrap.")
        return np.nan, (np.nan, np.nan), np.array([])

    # Separate finite values for median/CI calculation
    finite_predictions = bootstrap_predictions_arr[np.isfinite(bootstrap_predictions_arr)]

    median_pred = np.nan
    lower_bound = np.nan
    upper_bound = np.nan

    if len(finite_predictions) > 0:
        median_pred = np.median(finite_predictions)
        if len(finite_predictions) > 3:
            lower_percentile = (100.0 - confidence_level) / 2.0
            upper_percentile = 100.0 - lower_percentile
            lower_bound = np.percentile(finite_predictions, lower_percentile)
            upper_bound = np.percentile(finite_predictions, upper_percentile)
        else:
            lower_bound = upper_bound = median_pred
            warnings.warn(f"Only {len(finite_predictions)} finite predictions in bootstrap; CI may not be reliable.")

        if np.any(np.isinf(bootstrap_predictions_arr)):
            upper_bound = np.inf
    else:
        median_pred = np.inf
        lower_bound = np.inf
        upper_bound = np.inf

    return median_pred, (lower_bound, upper_bound), bootstrap_predictions_arr

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

    return np.array(combined_time), np.array(combined_temp)

def determine_auto_alpha_levels(
    datasets: List[KineticDataset],
    num_levels: int = 19,
    min_overlap: int = 2,
    alpha_bounds: Tuple[float, float] = (0.01, 0.99),
    resolution: int = 101
    ) -> Optional[np.ndarray]:
    """
    Determines suitable alpha levels for isoconversional analysis based on
    data overlap across multiple datasets.

    Args:
        datasets: List of KineticDataset objects.
        num_levels: The desired number of alpha levels to return.
        min_overlap: The minimum number of datasets that must contain data
                     at a given alpha level for it to be considered valid.
        alpha_bounds: Tuple (min_alpha, max_alpha) defining the search range.
        resolution: The number of points to check within the alpha_bounds.

    Returns:
        A NumPy array of suitable alpha levels, or None if no suitable range
        with the required overlap is found.
    """
    if not datasets:
        warnings.warn("Cannot determine auto alpha levels: No datasets provided.")
        return None
    if min_overlap < 2:
        warnings.warn("min_overlap should be at least 2 for isoconversional methods.")
        min_overlap = 2
    if len(datasets) < min_overlap:
        warnings.warn(f"Number of datasets ({len(datasets)}) is less than min_overlap ({min_overlap}). Cannot guarantee overlap.")

    # Create a grid of potential alpha values to check
    potential_alphas = np.linspace(alpha_bounds[0], alpha_bounds[1], resolution)
    valid_alphas_for_overlap = []

    print(f"Determining auto alpha levels (min_overlap={min_overlap})...")

    # Determine min/max alpha for each dataset
    run_alpha_ranges = []
    for i, ds in enumerate(datasets):
        if ds.conversion is not None and len(ds.conversion) > 1:
            finite_conv = ds.conversion[np.isfinite(ds.conversion)]
            if len(finite_conv) > 0:
                run_alpha_ranges.append((np.min(finite_conv), np.max(finite_conv)))
            else:
                run_alpha_ranges.append((np.nan, np.nan))
                warnings.warn(f"Dataset {i} contains no finite conversion data.")
        else:
            run_alpha_ranges.append((np.nan, np.nan))
            warnings.warn(f"Dataset {i} has insufficient conversion data.")

    # Check overlap for each potential alpha
    for alpha_check in potential_alphas:
        overlap_count = 0
        for min_a, max_a in run_alpha_ranges:
            if np.isfinite(min_a) and np.isfinite(max_a) and \
               (min_a - 1e-9) <= alpha_check <= (max_a + 1e-9):
                overlap_count += 1

        if overlap_count >= min_overlap:
            valid_alphas_for_overlap.append(alpha_check)

    if not valid_alphas_for_overlap:
        warnings.warn(f"Could not find any alpha range between {alpha_bounds[0]:.3f} and {alpha_bounds[1]:.3f} "
                      f"with at least {min_overlap} overlapping datasets.")
        return None

    # Select final levels from the valid range
    valid_alpha_min = min(valid_alphas_for_overlap)
    valid_alpha_max = max(valid_alphas_for_overlap)

    if valid_alpha_max - valid_alpha_min < 1e-6:
        warnings.warn(f"Valid overlapping alpha range is extremely narrow ({valid_alpha_min:.3f} - {valid_alpha_max:.3f}). "
                      f"Returning only the midpoint.")
        return np.array([np.mean([valid_alpha_min, valid_alpha_max])])

    # Generate evenly spaced levels within the valid range
    final_alpha_levels = np.linspace(valid_alpha_min, valid_alpha_max, num_levels)

    print(f"Auto alpha levels determined: {len(final_alpha_levels)} levels from {valid_alpha_min:.3f} to {valid_alpha_max:.3f}")
    return final_alpha_levels