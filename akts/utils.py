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

    return np.array(combined_time), np.array(combined_temp)