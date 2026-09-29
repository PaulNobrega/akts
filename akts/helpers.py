"""
High-level helper functions for non-expert users to perform automated kinetic analysis.

These functions provide sensible defaults and automated workflows for common use cases.
"""
import numpy as np
import time
import warnings
from typing import List, Dict, Optional, Callable, Union, Tuple
from pathlib import Path
from scipy.integrate import quad
from scipy.interpolate import interp1d

from .datatypes import KineticDataset, FitResult, BootstrapResult, IsoResult
from .loaders import load_data_file
from .core import fit_kinetic_model, run_bootstrap, predict_conversion, rank_models, predict_conversion_model_free
from .isoconversional import run_friedman, run_bootstrap_friedman
from .models import get_log_param_names
from .model_selector import models
from .utils import construct_profile, R_GAS, calculate_aic, calculate_bic
from .empirical import fit_empirical_global, predict_empirical
from .json_utils import (
    parse_json_data,
    serialize_results_to_json,
    fit_result_to_json,
    prediction_to_json,
    bootstrap_to_json,
    convert_numpy_to_python
)
from .reporting import generate_isothermal_report


# Time unit conversions to seconds
TIME_UNITS = {
    'second': 1,
    'seconds': 1,
    's': 1,
    'minute': 60,
    'minutes': 60,
    'min': 60,
    'hour': 3600,
    'hours': 3600,
    'h': 3600,
    'hr': 3600,
    'day': 86400,
    'days': 86400,
    'd': 86400,
    'week': 604800,
    'weeks': 604800,
    'month': 2592000,  # 30 days
    'months': 2592000,
    'year': 31536000,  # 365 days
    'years': 31536000,
    'yr': 31536000
}


# Keep these aliases for compatibility; model_selector owns the model lists.
DEFAULT_ISOTHERMAL_MODELS = models.default

ODE_MODELS = models.ode.all

EMPIRICAL_MODELS = models.empirical.all

# User-friendly display names for models
MODEL_DISPLAY_NAMES = {
    # f(alpha) models
    'F0': 'F0 (zero-order)',
    'F1': 'F1 (first-order)',
    'F2': 'F2 (second-order)',
    'F3': 'F3 (third-order)',
    'A2': 'A2 (Avrami-Erofeev, n=2)',
    'A3': 'A3 (Avrami-Erofeev, n=3)',
    'R2': 'R2 (contracting area)',
    'R3': 'R3 (contracting volume)',
    'D2': 'D2 (2D diffusion)',
    'D3': 'D3 (3D diffusion, Jander)',
    'D4': 'D4 (3D diffusion, Ginstling-Brounshtein)',
    'D1': 'D1 (1D diffusion)',
    'SB_mn': 'SB(m,n) (Sestak-Berggren, autocatalytic)',
    'Bna': 'Bna (Prout-Tompkins, autocatalytic)',
    # ODE models (multi-step)
    'A->B->C': 'A->B->C (consecutive reactions)',
    'A+B->C': 'A+B->C (bimolecular)',
    # Model-free (isoconversional)
    'Friedman': 'Friedman (model-free isoconversional)',
}


LOADER_TIME_UNITS_TO_SECONDS = {'s': 1, 'min': 60, 'h': 3600, 'days': 86400, 'weeks': 604800}


def _harmonize_loaded_datasets(datasets: List[KineticDataset], readout_type: str,
                               progress: Callable) -> None:
    """Put all file-loaded datasets on seconds and on one shared conversion scale (in place)."""
    for ds in datasets:
        unit = ds.metadata.get('time_units')
        if unit in LOADER_TIME_UNITS_TO_SECONDS and unit != 's':
            ds.time = ds.time * LOADER_TIME_UNITS_TO_SECONDS[unit]
            ds.metadata['time_units'] = 's'
        elif unit is not None and unit not in LOADER_TIME_UNITS_TO_SECONDS:
            warnings.warn(f"{ds.metadata.get('filename')}: could not tell time units from the "
                          "column name; assuming seconds. Name the column e.g. 'Time (days)'.")

    raw = [ds.metadata.get('readout_raw') for ds in datasets]
    if len(datasets) < 2 or any(r is None for r in raw):
        return
    # Per-file min-max scaling forces every temperature to end at conversion 1, which
    # hides the Arrhenius trend. Use one start/end for all files instead.
    start = float(np.mean([r[0] for r in raw]))
    end = float(max(r.max() for r in raw)) if readout_type == 'increasing' else float(min(r.min() for r in raw))
    if abs(end - start) < 1e-12:
        return
    for ds, r in zip(datasets, raw):
        ds.conversion = np.clip((r - start) / (end - start), 0.0, 1.0)
    progress(f"Normalized readouts on a shared scale ({start:.3g} -> {end:.3g} = conversion 0 -> 1)", {})


def _convert_time_to_seconds(value: float, unit: str) -> float:
    """Convert time value to seconds."""
    unit_lower = unit.lower().strip()
    if unit_lower not in TIME_UNITS:
        raise ValueError(f"Unknown time unit: {unit}. Supported: {list(TIME_UNITS.keys())}")
    return value * TIME_UNITS[unit_lower]


def _temperature_to_kelvin(temp: float, unit: str) -> float:
    """
    Convert temperature to Kelvin.

    Parameters
    ----------
    temp : float
        Temperature value
    unit : str
        Temperature unit ('K', 'C', or 'F')

    Returns
    -------
    float
        Temperature in Kelvin
    """
    unit_upper = unit.upper().strip()

    if unit_upper == 'K':
        return temp
    elif unit_upper == 'C':
        return temp + 273.15
    elif unit_upper == 'F':
        return (temp - 32) * 5/9 + 273.15
    else:
        raise ValueError(f"Unknown temperature unit: {unit}. Supported: 'K', 'C', 'F'")


def _temperature_from_kelvin(temp: float, unit: str) -> float:
    """
    Convert temperature from Kelvin to specified unit.

    Parameters
    ----------
    temp : float
        Temperature in Kelvin
    unit : str
        Target temperature unit ('K', 'C', or 'F')

    Returns
    -------
    float
        Temperature in target unit
    """
    unit_upper = unit.upper().strip()

    if unit_upper == 'K':
        return temp
    elif unit_upper == 'C':
        return temp - 273.15
    elif unit_upper == 'F':
        return (temp - 273.15) * 9/5 + 32
    else:
        raise ValueError(f"Unknown temperature unit: {unit}. Supported: 'K', 'C', 'F'")


def _create_progress_wrapper(user_callback: Optional[Callable[[str, Dict], None]]) -> Callable:
    """
    Create progress callback wrapper that adds timestamps and formatting.

    Parameters
    ----------
    user_callback : Callable or None
        User's callback function with signature (message: str, data: dict)

    Returns
    -------
    Callable
        Wrapped callback function
    """
    def wrapped_callback(message: str, data: Optional[Dict] = None):
        if user_callback is None:
            return

        if data is None:
            data = {}

        # Add timestamp
        data['timestamp'] = time.strftime('%H:%M:%S')

        # Call user's callback
        try:
            user_callback(message, data)
        except Exception as e:
            warnings.warn(f"Progress callback error: {e}")

    return wrapped_callback


def _fit_with_stability_check(
    model_info: Dict,
    datasets: List[KineticDataset],
    initial_guesses: Dict,
    bounds: Dict,
    use_multistart: bool,
    progress: Callable
) -> FitResult:
    """
    Fit a model with extra stability checks for problematic models like A+B->C.

    Parameters
    ----------
    model_info : Dict
        Model configuration
    datasets : List[KineticDataset]
        Experimental data
    initial_guesses : Dict
        Initial parameter guesses
    bounds : Dict
        Parameter bounds
    use_multistart : bool
        Whether to fit from several starting points (see _multistart_fit)
    progress : Callable
        Progress callback

    Returns
    -------
    FitResult
        Fit result (may be failed if unstable)
    """
    max_attempts = 3
    attempt = 0

    while attempt < max_attempts:
        attempt += 1

        try:
            if use_multistart:
                fit_res = _multistart_fit(
                    datasets=datasets,
                    model_name=model_info['type'],
                    model_definition_args=model_info['def_args'],
                    initial_guesses=initial_guesses,
                    parameter_bounds=bounds,
                    progress_callback=progress
                )
            else:
                fit_res = fit_kinetic_model(
                    datasets=datasets,
                    model_name=model_info['type'],
                    model_definition_args=model_info['def_args'],
                    initial_guesses=initial_guesses,
                    parameter_bounds=bounds
                )

            # Check for numerical stability
            if fit_res.success:
                # Check R² is reasonable
                if not np.isfinite(fit_res.r_squared) or fit_res.r_squared < -10:
                    if attempt < max_attempts:
                        progress(f"  Unstable fit (R²={fit_res.r_squared:.2f}), retrying with tighter bounds...", {})
                        # Tighten bounds for next attempt
                        for key in bounds:
                            if key.startswith('A'):
                                bounds[key] = (bounds[key][0] * 10, bounds[key][1] / 10)
                        continue
                    else:
                        fit_res.success = False
                        fit_res.message = f"Numerically unstable (R²={fit_res.r_squared:.2f})"

            return fit_res

        except Exception as e:
            if attempt < max_attempts:
                progress(f"  Fit failed: {e}. Retrying ({attempt}/{max_attempts})...", {})
                # Adjust initial guesses for next attempt
                for key in initial_guesses:
                    if key.startswith('Ea'):
                        initial_guesses[key] *= 1.2
                    elif key.startswith('A'):
                        initial_guesses[key] *= 0.5
            else:
                # Final attempt failed
                from .datatypes import FitResult
                return FitResult(
                    model_name=model_info['type'],
                    parameters={},
                    success=False,
                    message=f"Failed after {max_attempts} attempts: {e}",
                    rss=np.inf, n_datapoints=0, n_parameters=0
                )

    # Should not reach here
    from .datatypes import FitResult
    return FitResult(
        model_name=model_info['type'],
        parameters={},
        success=False,
        message="Unexpected error in stability check",
        rss=np.inf, n_datapoints=0, n_parameters=0
    )


def _get_smart_initial_guess(
    datasets: List[KineticDataset],
    progress_callback: Optional[Callable] = None
) -> Dict[str, float]:
    """
    Get smart initial guesses by fitting a simple F1 model first.

    This provides much better starting points for complex ODE models.

    Parameters
    ----------
    datasets : List[KineticDataset]
        Experimental datasets
    progress_callback : Callable, optional
        Progress callback function

    Returns
    -------
    Dict[str, float]
        Initial guess dictionary with 'Ea' and 'A' keys
    """
    try:
        # Quick F1 fit to get ballpark Ea and A
        quick_fit = fit_kinetic_model(
            datasets=datasets,
            model_name='single_step',
            model_definition_args={'f_alpha_model': 'F1'},
            initial_guesses={'Ea': 80000, 'A': 1e12},
            parameter_bounds={'Ea': (10000, 300000), 'A': (1e3, 1e25)}
        )

        if quick_fit.success:
            if progress_callback:
                progress_callback("Using F1 fit for better ODE initial guesses", {})
            return {
                'Ea': quick_fit.parameters.get('Ea', 80000),
                'A': quick_fit.parameters.get('A', 1e12)
            }
    except Exception as e:
        if progress_callback:
            progress_callback(f"F1 quick fit failed, using default guesses: {e}", {})

    # Fallback to defaults
    return {'Ea': 80000, 'A': 1e12}


def _multistart_fit(
    datasets: List[KineticDataset],
    model_name: str,
    model_definition_args: Dict,
    initial_guesses: Dict[str, float],
    parameter_bounds: Dict[str, Tuple[float, float]],
    n_starts: int = 4,
    progress_callback: Optional[Callable] = None
) -> FitResult:
    """
    Fit an ODE model from several starting points and keep the best result.

    Each start is a full fit_kinetic_model() call. fit_kinetic_model() already
    uses an Arrhenius reparameterization, a coarse rate-constant scan, and Powell
    to reliably find a good local optimum from one starting point (see core.py),
    so this only needs a handful of starts as cheap insurance against a bad local
    minimum -- it does not need to explore the space the way the Bayesian-search
    approach this replaced did.

    Parameters
    ----------
    datasets : List[KineticDataset]
        Experimental datasets.
    model_name : str
        Model type (e.g. 'A->B->C').
    model_definition_args : Dict
        Model-specific arguments.
    initial_guesses : Dict[str, float]
        Initial parameter guesses for the first (unperturbed) start.
    parameter_bounds : Dict[str, Tuple[float, float]]
        Parameter bounds (min, max) for each parameter.
    n_starts : int, default=4
        Number of fitting attempts: 1 unperturbed + (n_starts - 1) perturbed.
    progress_callback : Callable, optional
        Progress callback function.

    Returns
    -------
    FitResult
        Best result found across all starts (by R²), or a failed FitResult if
        every start failed.
    """
    try:
        log_param_names = get_log_param_names(model_name)
    except ValueError:
        log_param_names = frozenset()

    rng = np.random.default_rng()
    best_result: Optional[FitResult] = None
    last_message = "all starts failed"

    for start in range(n_starts):
        t0 = time.time()
        if progress_callback:
            progress_callback(
                f"  Multistart: starting fit {start + 1}/{n_starts}...",
                {'start': start + 1, 'n_starts': n_starts, 'phase': 'started'}
            )
        if start == 0:
            guesses = dict(initial_guesses)
        else:
            guesses = {}
            for name, value in initial_guesses.items():
                if name in log_param_names:
                    guesses[name] = value * (10.0 ** rng.uniform(-2.0, 2.0))
                else:
                    guesses[name] = value * rng.uniform(0.6, 1.4)

        try:
            fit_res = fit_kinetic_model(
                datasets=datasets,
                model_name=model_name,
                model_definition_args=model_definition_args,
                initial_guesses=guesses,
                parameter_bounds=parameter_bounds,
                verbose=(start == 0)
            )
            ok = fit_res.success and np.isfinite(fit_res.r_squared)
        except Exception as e:
            fit_res, ok = None, False
            last_message = f"error: {e}"

        elapsed = time.time() - t0
        is_improvement = ok and (best_result is None or fit_res.r_squared > best_result.r_squared)
        if is_improvement:
            best_result = fit_res
        elif not ok and fit_res is not None:
            last_message = fit_res.message

        if progress_callback:
            best_r2 = best_result.r_squared if best_result is not None else float('nan')
            if is_improvement:
                status = f"R²={fit_res.r_squared:.4f} * new best"
            elif ok:
                status = f"R²={fit_res.r_squared:.4f} (best {best_r2:.4f})"
            else:
                status = f"failed (best {best_r2:.4f})"
            progress_callback(
                f"  Multistart: {start + 1}/{n_starts} {status} [{elapsed:.1f}s]",
                {'start': start + 1, 'n_starts': n_starts, 'best_r2': best_r2, 'seconds': elapsed}
            )

    if best_result is not None:
        return best_result
    return FitResult(
        model_name=model_name, model_definition_args=model_definition_args,
        parameters={}, success=False,
        message=f"Multistart fit failed after {n_starts} attempts: {last_message}",
        rss=np.inf, n_datapoints=0, n_parameters=len(initial_guesses)
    )


def _setup_model_configs(
    models_to_try: List[str],
    smart_guess: Optional[Dict[str, float]] = None,
    initial_ratio_r: float = 1.0
) -> Tuple[List[Dict], Dict[str, Dict], Dict[str, Dict]]:
    """
    Setup model configurations with sensible default initial guesses and bounds.

    Parameters
    ----------
    models_to_try : List[str]
        List of model names (e.g., ['F1', 'F2', 'A2', 'A->B->C'])
        Can include both f(alpha) models and ODE models
    smart_guess : Dict[str, float], optional
        Smart initial guesses from a preliminary fit (Ea, A)
    initial_ratio_r : float, default=1.0
        Fixed [B]0/[A]0 ratio for the A+B->C bimolecular model. At r=1 with
        default m=n=1 the model is mathematically identical to F2, so this only
        matters if the true starting ratio of the two reactants is known and
        not 1:1.

    Returns
    -------
    Tuple[List[Dict], Dict, Dict]
        (models_config, initial_guesses_pool, bounds_pool)
    """
    models_config = []
    initial_guesses_pool = {}
    bounds_pool = {}

    # Default ranges for isothermal kinetics
    if smart_guess is not None:
        default_guess = smart_guess.copy()
    else:
        default_guess = {'Ea': 80000, 'A': 1e12}  # 80 kJ/mol, typical pre-exponential

    default_bounds = {
        'Ea': (10000, 300000),  # 10-300 kJ/mol
        'A': (1e3, 1e20)  # Reasonable range for pre-exponential (log(A) = 6.9 to 46)
    }

    for model_name in models_to_try:
        if model_name == 'Friedman':
            # Model-free isoconversional analysis -- no optimizer, no Ea/A
            # "parameters" in the usual sense, so no entry needed in
            # initial_guesses_pool/bounds_pool (the fitting loop dispatches on
            # model_info['type'] before ever indexing into those dicts for this type).
            models_config.append({'name': 'Friedman_model', 'type': 'Friedman', 'def_args': {}})
            continue

        # Check if it's an empirical model
        if model_name in EMPIRICAL_MODELS:
            model_key = f'Empirical_{model_name}_model'
            models_config.append({
                'name': model_key,
                'type': model_name,  # First_Order, Linear, Sqrt, Logistic, Exponential
                'def_args': {'empirical_type': model_name}
            })
            # Empirical model initial guesses
            ea_guess = default_guess.get('Ea', 80000)
            a_guess = default_guess.get('A', 1e10)

            if model_name == 'First_Order':
                initial_guesses_pool[model_key] = {'Ea': ea_guess, 'A': a_guess, 'A_scale': 1.0}
                bounds_pool[model_key] = {
                    'Ea': (1000, 500000), 'A': (1e-5, 1e20), 'A_scale': (0.0, 10.0)
                }
            elif model_name in ['Linear', 'Sqrt']:
                initial_guesses_pool[model_key] = {'Ea': ea_guess, 'A': a_guess, 'C': 0.0}
                bounds_pool[model_key] = {
                    'Ea': (1000, 500000), 'A': (1e-5, 1e20), 'C': (-1.0, 1.0)
                }
            elif model_name == 'Logistic':
                initial_guesses_pool[model_key] = {'Ea': ea_guess, 'A': a_guess, 'A_max': 1.0, 'B': 1.0}
                bounds_pool[model_key] = {
                    'Ea': (1000, 500000), 'A': (1e-5, 1e20),
                    'A_max': (0.0, 10.0), 'B': (0.01, 100.0)
                }
            elif model_name == 'Exponential':
                initial_guesses_pool[model_key] = {'Ea': ea_guess, 'A': a_guess, 'A_amp': 1.0, 'C': 0.0}
                bounds_pool[model_key] = {
                    'Ea': (1000, 500000), 'A': (1e-5, 1e20),
                    'A_amp': (0.0, 10.0), 'C': (-1.0, 1.0)
                }
            continue

        # Check if it's an ODE model (contains special characters)
        if '->' in model_name or '+' in model_name:
            # ODE model
            model_key = model_name  # Use model name directly
            if model_name == 'A->B->C':
                # Consecutive reactions: A→B→C
                # Requires f1_model and f2_model to define mechanism for each step
                # Default: both steps are first-order (F1)
                models_config.append({
                    'name': model_key,
                    'type': 'A->B->C',
                    'def_args': {
                        'f1_model': 'F1',  # First step mechanism
                        'f2_model': 'F1'   # Second step mechanism
                    }
                })
                # Use smart guesses if available, otherwise defaults
                ea_guess = default_guess.get('Ea', 80000)
                a_guess = default_guess.get('A', 1e12)
                initial_guesses_pool[model_key] = {
                    'Ea1': ea_guess, 'A1': a_guess,
                    'Ea2': ea_guess * 1.25, 'A2': a_guess * 10  # Second step slightly higher
                }
                bounds_pool[model_key] = {
                    'Ea1': (10000, 300000), 'A1': (1e3, 1e20),
                    'Ea2': (10000, 300000), 'A2': (1e3, 1e20)
                }
            elif model_name == 'A+B->C':
                # Bimolecular reaction (with tighter bounds for stability)
                models_config.append({
                    'name': model_key,
                    'type': 'A+B->C',
                    # initial_ratio_r is a fixed model input (not fitted), user-supplied
                    'def_args': {'bimol_params': {'initial_ratio_r': initial_ratio_r}}
                })
                ea_guess = default_guess.get('Ea', 80000)
                a_guess = default_guess.get('A', 1e12)
                initial_guesses_pool[model_key] = {'Ea': ea_guess, 'A': a_guess}
                bounds_pool[model_key] = {
                    'Ea': (10000, 250000),
                    'A': (1e6, 1e18),
                }
        else:
            # f(alpha) model
            model_key = f'{model_name}_model'
            models_config.append({
                'name': model_key,
                'type': 'single_step',
                'def_args': {'f_alpha_model': model_name}
            })

            # Start with default Ea/A guesses and bounds
            guesses = default_guess.copy()
            bounds = default_bounds.copy()

            # Add shape parameter guesses and bounds for fitted-parameter models
            if model_name == 'Fn':
                # Fitted-n model: n as a free parameter (start at first-order)
                guesses['n'] = 1.0
                bounds['n'] = (0.0, 5.0)  # n typically 0-3, allow up to 5 for flexibility
            elif model_name == 'SB':
                # Fitted SB(m,n) model: both m and n as free parameters
                guesses['m'] = 0.5  # Start at SB_mn default
                guesses['n'] = 1.0
                bounds['m'] = (0.0, 3.0)  # m typically 0-2
                bounds['n'] = (0.0, 3.0)  # n typically 0-3
            elif model_name == 'SB_mnp':
                # Extended SB(m,n,p) model: m, n, and p as free parameters
                guesses['m'] = 0.5
                guesses['n'] = 1.0
                guesses['p'] = 0.0  # Start with p=0 (reduces to SB_mn)
                bounds['m'] = (0.0, 3.0)
                bounds['n'] = (0.0, 3.0)
                bounds['p'] = (-2.0, 2.0)  # p can be positive or negative

            initial_guesses_pool[model_key] = guesses
            bounds_pool[model_key] = bounds

    return models_config, initial_guesses_pool, bounds_pool


def calculate_ich_q1e_ceiling(
    study_duration_months: float,
    is_long_term: bool = True
) -> float:
    """
    Calculate ICH Q1E extrapolation ceiling for stability studies.

    ICH Q1E guideline provides rules for how far beyond observed stability data
    a shelf-life claim can be extrapolated based on kinetic modeling:

    - **Long-term data**: Up to min(2 × duration, duration + 12 months)
    - **Accelerated data**: Up to 1.5 × duration

    This ensures that extrapolated shelf-life claims remain within acceptable
    confidence limits for regulatory submissions (pharmaceuticals, biologics).

    Parameters
    ----------
    study_duration_months : float
        Duration of the stability study in months (the longest time point
        in your dataset).
    is_long_term : bool, default=True
        If True, apply long-term storage rules (25°C/60% RH or similar).
        If False, apply accelerated storage rules (40°C/75% RH or similar).

    Returns
    -------
    float
        Maximum allowable shelf-life extrapolation in months according to
        ICH Q1E guidelines.

    Examples
    --------
    >>> calculate_ich_q1e_ceiling(12)  # 12-month study
    24.0  # Can extrapolate up to 24 months
    >>> calculate_ich_q1e_ceiling(18)  # 18-month study
    30.0  # min(2*18, 18+12) = 30 months
    >>> calculate_ich_q1e_ceiling(24)  # 24-month study
    36.0  # min(2*24, 24+12) = 36 months
    >>> calculate_ich_q1e_ceiling(6, is_long_term=False)  # 6-month accelerated
    9.0  # Accelerated: 1.5 × 6 = 9 months

    References
    ----------
    ICH Q1E: Evaluation of Stability Data (2003)
    https://database.ich.org/sites/default/files/Q1E%20Guideline.pdf
    """
    if study_duration_months <= 0:
        raise ValueError(f"study_duration_months must be positive, got {study_duration_months}")

    if is_long_term:
        # Long-term: min(2 × duration, duration + 12)
        return min(2 * study_duration_months, study_duration_months + 12)
    else:
        # Accelerated: 1.5 × duration
        return 1.5 * study_duration_months


def time_to_conversion(
    fit_result: FitResult,
    target_conversion: float,
    temperature_K: float,
    bootstrap_result: Optional[BootstrapResult] = None,
    max_search_time_sec: Optional[float] = None,
    n_eval_points: int = 500,
    one_sided_ci: bool = False,
) -> Dict[str, Optional[float]]:
    """
    Time to reach a target conversion (e.g. 5% degradation) at a fixed storage
    temperature, with an optional confidence interval from a bootstrap result.

    This is the inverse of predict_conversion(): instead of "what's the conversion
    at time t", it answers "at what time does conversion first reach alpha_target".
    Used for shelf-life-style questions (ICH Q1E "time to reach 5% degradation").

    Parameters
    ----------
    fit_result : FitResult
        A successful fit from fit_kinetic_model().
    target_conversion : float
        Target conversion fraction in (0, 1], e.g. 0.05 for 5% degradation.
    temperature_K : float
        Fixed storage temperature to evaluate at, in Kelvin.
    bootstrap_result : BootstrapResult, optional
        If provided (and matches fit_result.model_name), also returns a
        confidence interval on the time estimate, propagated from the
        conversion confidence band via the same crossing-time logic.
    max_search_time_sec : float, optional
        Upper bound of the search window, in seconds. Defaults to 10x the time
        a naive first-order estimate would take, extended automatically (up to
        a few more doublings) if the target isn't reached in that window.
    n_eval_points : int, default=500
        Number of points to simulate across the search window before
        interpolating for the crossing time. More points = more precise
        crossing time at the cost of more ODE evaluations.
    one_sided_ci : bool, default=False
        If True, compute one-sided 95% lower bound (5th percentile) instead of
        two-sided 95% CI (2.5th-97.5th percentiles). Use True for ICH Q1E
        regulatory shelf-life estimates (more conservative). When True,
        time_upper_sec will be None.

    Returns
    -------
    Dict[str, Optional[float]]
        'time_sec': time to reach target_conversion, or None if not reached
        within max_search_time_sec (or its auto-extended window).
        'time_lower_sec' / 'time_upper_sec': confidence interval bounds (from
        bootstrap_result), or None if no bootstrap_result was given or a bound
        was never reached. For one_sided_ci=True, time_upper_sec is always None.
    """
    if not (0.0 < target_conversion <= 1.0):
        raise ValueError(f"target_conversion must be in (0, 1], got {target_conversion}")
    if not fit_result.success:
        warnings.warn("Cannot compute time-to-conversion from an unsuccessful fit.")
        return {'time_sec': None, 'time_lower_sec': None, 'time_upper_sec': None}

    def _crossing_time(conversion: np.ndarray, time_sec: np.ndarray) -> Optional[float]:
        reached = np.where(conversion >= target_conversion)[0]
        if len(reached) == 0:
            return None
        idx = reached[0]
        if idx == 0:
            return float(time_sec[0])
        # Linear interpolation between the last point below target and the
        # first point at/above it, for a crossing time finer than the grid.
        t0, t1 = time_sec[idx - 1], time_sec[idx]
        c0, c1 = conversion[idx - 1], conversion[idx]
        if c1 == c0:
            return float(t1)
        frac = (target_conversion - c0) / (c1 - c0)
        return float(t0 + frac * (t1 - t0))

    # Rough first-order-style estimate to size the search window: t ~ target/k,
    # using whatever Ea/A-shaped parameters exist. Falls back to a fixed window
    # if the model has no simple rate constant to estimate from (e.g. multi-step).
    params = fit_result.parameters
    ea_keys = [k for k in params if k.startswith('Ea')]
    a_keys = [k for k in params if k.startswith('A') and not k.startswith('A_')]
    if max_search_time_sec is not None:
        window = max_search_time_sec
    elif ea_keys and a_keys:
        k_est = params[a_keys[0]] * np.exp(-params[ea_keys[0]] / (R_GAS * temperature_K))
        window = 10.0 * target_conversion / k_est if k_est > 0 else 365 * 86400.0
    else:
        window = 365 * 86400.0  # 1 year fallback

    window = float(np.clip(window, 3600.0, 100 * 365 * 86400.0))  # 1hr .. 100yr sanity bounds
    time_lower, time_upper = None, None
    max_window = 1000 * 365 * 86400.0  # hard ceiling regardless of how badly the estimate misses

    for _ in range(8):  # auto-extend the window if the naive k-based estimate undershot
        time_eval = np.linspace(0.0, window, n_eval_points)
        prediction = predict_conversion(
            kinetic_description=fit_result,
            temperature_program=lambda t: temperature_K,
            simulation_time_sec=time_eval,
            initial_alpha=0.0,
            bootstrap_result=bootstrap_result,
        )
        time_point = _crossing_time(prediction.conversion, prediction.time)
        if prediction.conversion_ci is not None:
            if one_sided_ci:
                # ICH Q1E: One-sided 95% lower bound = more conservative estimate
                # For shelf-life, we want the upper conversion CI bound (faster degradation)
                # which gives the shorter (more conservative) time estimate
                time_lower = _crossing_time(prediction.conversion_ci[1], prediction.time)  # upper conv bound -> earlier time (conservative)
                time_upper = None  # One-sided doesn't have upper bound
            else:
                # Two-sided CI (default behavior)
                time_lower = _crossing_time(prediction.conversion_ci[1], prediction.time)  # upper conv bound -> earlier time
                time_upper = _crossing_time(prediction.conversion_ci[0], prediction.time)  # lower conv bound -> later time

        if time_point is not None:
            return {'time_sec': time_point, 'time_lower_sec': time_lower, 'time_upper_sec': time_upper}
        if window >= max_window:
            break
        # The initial k-based guess can be badly off (e.g. a fitted A/Ea pair that
        # compensates for noisy data can look nothing like the "true" rate), so
        # widen aggressively rather than assuming the guess was only slightly small.
        window = min(window * 10.0, max_window)

    return {'time_sec': None, 'time_lower_sec': time_lower, 'time_upper_sec': time_upper}


def time_to_conversion_model_free(
    iso_result: IsoResult,
    target_conversion: float,
    temperature_K: float,
    initial_alpha: float = 0.0,
) -> Optional[float]:
    """
    Model-free counterpart to time_to_conversion(): time to reach a target
    conversion at a fixed storage temperature, without assuming a reaction
    model, using the Friedman isoconversional result directly.

    At constant temperature the model-free rate equation
    dalpha/dt = exp(ln_A_f_alpha(alpha)) * exp(-Ea(alpha)/(R*T)) has a closed-form
    time via direct quadrature -- no ODE solve needed:

        t(alpha_target, T) = integral from initial_alpha to alpha_target of
            dalpha' / [exp(ln_A_f_alpha(alpha')) * exp(-Ea(alpha')/(R*T))]

    Requires an IsoResult with both Ea and ln_A_f_alpha populated (currently
    only run_friedman() provides this -- see predict_conversion_model_free()'s
    docstring for why KAS/OFW don't qualify).

    Parameters
    ----------
    iso_result : IsoResult
        Result of run_friedman() with alpha, Ea, and ln_A_f_alpha all populated.
    target_conversion : float
        Target conversion fraction in (0, 1], e.g. 0.05 for 5% degradation.
    temperature_K : float
        Fixed storage temperature, in Kelvin.
    initial_alpha : float, default=0.0
        Starting conversion for the integration.

    Returns
    -------
    Optional[float]
        Time in seconds, or None if the integral diverges or fails (e.g.
        target_conversion falls in a region where the rate is estimated to be
        ~0, making the reciprocal blow up).
    """
    if not (0.0 < target_conversion <= 1.0):
        raise ValueError(f"target_conversion must be in (0, 1], got {target_conversion}")
    if iso_result.Ea is None or iso_result.ln_A_f_alpha is None:
        raise ValueError(
            "time_to_conversion_model_free requires an IsoResult with both Ea and "
            "ln_A_f_alpha populated. Only run_friedman() currently provides "
            "ln_A_f_alpha; KAS/OFW intercepts are not usable here."
        )
    if target_conversion <= initial_alpha:
        return 0.0

    valid = np.isfinite(iso_result.alpha) & np.isfinite(iso_result.Ea) & np.isfinite(iso_result.ln_A_f_alpha)
    if valid.sum() < 2:
        raise ValueError("Need at least 2 finite (alpha, Ea, ln_A_f_alpha) points to compute from.")

    alpha_fit = iso_result.alpha[valid]
    Ea_vals, lnAf_vals = iso_result.Ea[valid], iso_result.ln_A_f_alpha[valid]
    Ea_interp = interp1d(alpha_fit, Ea_vals, bounds_error=False, fill_value=(Ea_vals[0], Ea_vals[-1]))
    lnAf_interp = interp1d(alpha_fit, lnAf_vals, bounds_error=False, fill_value=(lnAf_vals[0], lnAf_vals[-1]))

    def integrand(alpha):
        rate = np.exp(lnAf_interp(alpha)) * np.exp(-Ea_interp(alpha) / (R_GAS * temperature_K))
        return 1.0 / rate if rate > 1e-300 else np.inf

    try:
        value, _ = quad(integrand, initial_alpha, target_conversion, limit=200)
    except Exception as e:
        warnings.warn(f"Model-free time-to-conversion quadrature failed: {e}")
        return None
    return float(value) if np.isfinite(value) else None


def _check_model_convergence(
    top_models: List[FitResult],
    datasets: List[KineticDataset],
    prediction_time_sec: float,
    threshold: float
) -> Tuple[bool, List[float]]:
    """
    Check if top models converge in their predictions.

    Parameters
    ----------
    top_models : List[FitResult]
        Top N fitted models
    datasets : List[KineticDataset]
        Datasets (for temperature info)
    prediction_time_sec : float
        Time point to check convergence
    threshold : float
        Relative difference threshold for convergence (e.g., 0.15 = 15%)

    Returns
    -------
    Tuple[bool, List[float]]
        (converged: bool, predictions: List[float])
    """
    if len(top_models) < 2:
        return True, []

    # Use first dataset's temperature for prediction
    temp_K = datasets[0].temperature.mean()

    # Generate predictions at the specified time for each model
    predictions = []
    for model_fit in top_models:
        try:
            # Simple isothermal prediction
            temp_program = lambda t: temp_K
            time_eval = np.array([0, prediction_time_sec])

            pred_result = predict_conversion(
                kinetic_description=model_fit,
                temperature_program=temp_program,
                simulation_time_sec=time_eval,
                initial_alpha=0.0
            )
            final_conversion = pred_result.conversion[-1]
            predictions.append(final_conversion)
        except Exception as e:
            warnings.warn(f"Prediction failed for {model_fit.model_name}: {e}")
            predictions.append(np.nan)

    # Check convergence
    valid_predictions = [p for p in predictions if not np.isnan(p)]
    if len(valid_predictions) < 2:
        return True, predictions

    mean_pred = np.mean(valid_predictions)
    max_pred = np.max(valid_predictions)
    min_pred = np.min(valid_predictions)

    if mean_pred == 0:
        relative_diff = 0
    else:
        relative_diff = (max_pred - min_pred) / mean_pred

    converged = relative_diff < threshold

    return converged, predictions


# Tie-break order among models with equal parameter counts: simplest mechanism first.
MODEL_SIMPLICITY_ORDER = ['F1', 'F2', 'F3', 'R2', 'R3', 'A2', 'A3', 'D2', 'D3', 'D4', 'A->B->C', 'A+B->C']

# BIC differences below this are "not worth more than a bare mention" (Kass & Raftery 1995).
INDISTINGUISHABLE_DELTA_BIC = 2.0


def _simplicity_key(model: Dict) -> Tuple[int, int]:
    base = model['model_name'].replace('_model', '')
    order = MODEL_SIMPLICITY_ORDER.index(base) if base in MODEL_SIMPLICITY_ORDER else len(MODEL_SIMPLICITY_ORDER)
    return (model['stats']['n_params'], order)


# Plausible range for drug degradation mechanisms (hydrolysis, oxidation, thermolysis).
# Narrower than core.py's generic overflow-prevention bounds (5-400 kJ/mol) -- this is
# a domain-specific plausibility check for stability studies, not a numerical-stability one.
TYPICAL_EA_RANGE_J_MOL = (30_000.0, 180_000.0)


def _physical_sanity_flags(parameters: Dict[str, float]) -> List[str]:
    """
    Check fitted parameters against typical ranges for drug degradation kinetics.
    Returns a list of human-readable warnings (empty if nothing looks unusual).
    A flagged fit is not necessarily wrong -- e.g. diffusion-limited or unusually
    stable formulations can fall outside this range -- but it is worth a second look.
    """
    flags = []
    lo, hi = TYPICAL_EA_RANGE_J_MOL
    range_str = f"{lo/1000:.0f}-{hi/1000:.0f} kJ/mol"
    for name, value in parameters.items():
        if not name.startswith('Ea'):
            continue
        if value < lo:
            flags.append(
                f"{name}={value/1000:.1f} kJ/mol is below the plausible {range_str} range "
                f"for drug degradation -- may indicate diffusion control or model misfit"
            )
        elif value > hi:
            flags.append(
                f"{name}={value/1000:.1f} kJ/mol is above the plausible {range_str} range "
                f"for drug degradation -- may indicate overparameterization or baseline instability"
            )
    return flags


def _wrap_friedman_as_fit_result(iso_result: IsoResult, datasets: List[KineticDataset]) -> FitResult:
    """
    Wraps a Friedman IsoResult as a FitResult so it flows through rank_models(),
    _select_simplest_equivalent(), the report, and JSON output exactly like any
    other model -- no special-casing needed in those consumers.

    parameters is deliberately {} (Friedman has no single Ea/A; the per-alpha
    Ea(alpha)/ln_A_f_alpha(alpha) live in the stashed IsoResult, not surfaced as
    top-level "parameters"). model_definition_args carries the IsoResult itself,
    the same role that field already plays for every other model (everything
    needed to predict again) -- predict_conversion() and the bootstrap dispatch
    both read it back out via model_name == "Friedman".
    """
    n_params = int(np.isfinite(iso_result.Ea).sum())
    if n_params < 2:
        return FitResult(
            model_name='Friedman', parameters={}, success=False,
            message="Friedman analysis resolved fewer than 2 alpha levels -- "
                    "need at least 2 datasets at different temperatures/rates.",
            rss=np.inf, n_datapoints=0, n_parameters=n_params,
        )

    total_rss, n_points = 0.0, 0
    for ds in datasets:
        try:
            pred = predict_conversion_model_free(
                iso_result, temperature_program=(ds.time, ds.temperature),
                simulation_time_sec=ds.time,
            )
        except Exception:
            continue
        residuals = ds.conversion - pred.conversion
        valid = np.isfinite(residuals)
        total_rss += float(np.sum(residuals[valid] ** 2))
        n_points += int(np.sum(valid))

    if n_points <= n_params:
        return FitResult(
            model_name='Friedman', parameters={}, success=False,
            message=f"Only {n_points} valid data points for {n_params} resolved alpha "
                    f"levels -- too few to compute meaningful fit statistics.",
            rss=np.inf, n_datapoints=n_points, n_parameters=n_params,
        )

    r_squared = np.nan
    all_conv = np.concatenate([ds.conversion for ds in datasets])
    ss_tot = np.sum((all_conv - np.mean(all_conv)) ** 2)
    if ss_tot > 0:
        r_squared = 1.0 - total_rss / ss_tot

    return FitResult(
        model_name='Friedman', parameters={}, success=True,
        message="Friedman model-free isoconversional analysis",
        rss=total_rss, n_datapoints=n_points, n_parameters=n_params,
        r_squared=r_squared,
        aic=calculate_aic(total_rss, n_params, n_points),
        bic=calculate_bic(total_rss, n_params, n_points),
        model_definition_args={'iso_result': iso_result},
    )


def _select_simplest_equivalent(ranked_models: List[Dict]) -> Tuple[Dict, str]:
    """Among models whose BIC is within INDISTINGUISHABLE_DELTA_BIC of the best, pick the simplest."""
    best_bic = min(m['stats']['bic'] for m in ranked_models)
    tied = [m for m in ranked_models if m['stats']['bic'] - best_bic < INDISTINGUISHABLE_DELTA_BIC]
    chosen = min(tied, key=_simplicity_key)
    if len(tied) == 1:
        return chosen, "Best BIC; no other model is statistically equivalent"
    names = ', '.join(m['model_name'].replace('_model', '') for m in tied)
    return chosen, (f"Simplest of {len(tied)} statistically indistinguishable models "
                    f"(ΔBIC < {INDISTINGUISHABLE_DELTA_BIC:g}): {names}")


def auto_model_isothermal_data(
    data_files: List[Union[str, Path, KineticDataset, Dict]],
    predict: Optional[Union[Tuple[float, str], Tuple[float, str, float]]] = None,
    predict_temperature_K: Optional[float] = None,
    simulate: Optional[List[Tuple[float, float]]] = None,
    simulate_time_unit: str = 'days',
    input_temperature_units: str = 'K',
    output_temperature_units: str = 'K',
    top_n: int = 3,
    convergence_threshold: float = 0.15,
    models_to_try: Optional[List[str]] = None,
    include_ode_models: bool = False,
    include_empirical_models: bool = False,
    initial_ratio_r: float = 1.0,
    bootstrap_iterations: int = 100,
    confidence_level: float = 0.95,
    filter_implausible: bool = False,
    report_format: str = "interactive",
    report_path: Optional[Union[str, Path]] = None,
    output_format: str = "dict",
    auto_open: bool = False,
    progress_callback: Optional[Callable[[str, Dict], None]] = None,
    n_jobs: int = -1,
    **loader_kwargs
) -> Union[Dict, str, Tuple[Dict, str]]:
    """
    Automated isothermal kinetic modeling with model selection and reporting.

    This function provides a high-level interface for non-expert users to:
    1. Load isothermal kinetic data from multiple formats
    2. Automatically fit multiple kinetic models
    3. Rank models by statistical criteria
    4. Select the best model(s) with convergence checking
    5. Generate predictions with confidence intervals
    6. Create professional HTML reports

    Parameters
    ----------
    data_files : List[Union[str, Path, KineticDataset, Dict]]
        List of data sources. Can be:
        - File paths (str or Path): Will be loaded using load_data_file()
        - KineticDataset objects: Used directly
        - Dicts: JSON data with 'time', 'temperature', 'conversion' keys
    predict : Union[Tuple[float, str], Tuple[float, str, float]], optional
        Prediction specification. Can be:
        - (time_value, time_unit): e.g., (3, 'year') - uses first dataset temperature
        - (time_value, time_unit, temp_K): e.g., (3, 'year', 298) - specify temperature
    predict_temperature_K : float, optional
        Temperature in Kelvin for predictions. Overrides temperature from predict tuple.
        If not specified, uses the average temperature from the first dataset.
    simulate : List[Tuple[float, float]], optional
        Simulate degradation under variable temperature conditions (e.g., shipping excursions).
        List of (time, temperature) tuples defining a temperature profile.
        Temperature values use input_temperature_units.
        Example: [(0, 298), (10, 313), (20, 298)] = 10 days at 25°C, heat to 40°C, cool back
        Useful for: shipping scenarios, storage with temperature fluctuations, accelerated aging
    simulate_time_unit : str, default='days'
        Time unit for simulate parameter. Options: 'seconds', 'minutes', 'hours', 'days', 'weeks', 'months', 'years'
    input_temperature_units : str, default='K'
        Temperature units for input data and parameters. Options: 'K' (Kelvin), 'C' (Celsius), 'F' (Fahrenheit)
        All input temperatures (data files, predict_temperature_K, simulate profile) will be converted from this unit.
    output_temperature_units : str, default='K'
        Temperature units for output results and reports. Options: 'K', 'C', 'F'
        All output temperatures (predictions, simulations, reports) will be displayed in this unit.
    top_n : int, default=3
        Number of top models to consider
    convergence_threshold : float, default=0.15
        Relative difference threshold for model convergence (0.15 = 15%)
    models_to_try : List[str], optional
        List of models to try. Use the model selector for IDE autocomplete:
        - models.kinetic.all (default mechanistic models)
        - models.empirical.all (algebraic fits)
        - models.ode.all (multi-step ODE models)
        - models.all (everything)
        ODE and empirical models are auto-detected from the list.
        Default: F0, F1, F2, F3, A2, A3, R2, R3, D2, D3, SB_mn, Bna
        Plus 'Friedman' if data has >=3 distinct temperatures (auto-added).
    include_ode_models : bool, default=False
        **DEPRECATED**: Use model selector instead (models.ode.all).
        Kept for backward compatibility. When models_to_try=None, adds ODE models
        to default list. ODE models (A->B->C, A+B->C) are slower (~5-10x) but
        more flexible. With model selector, ODE models are auto-enabled when present.
        Friedman model-free analysis ('Friedman') has no such flag -- it is added
        to the default list automatically whenever the loaded data spans >=3
        distinct temperatures (rounded to the nearest Kelvin), since that's the
        minimum needed for a meaningful per-alpha regression; with 1-2
        temperatures it's silently omitted rather than included and failing.
    include_empirical_models : bool, default=False
        **DEPRECATED**: Use model selector instead (models.empirical.all).
        Kept for backward compatibility. When models_to_try=None, adds empirical
        models to default list. Empirical models (First_Order, Linear, Sqrt, etc.)
        use global Arrhenius fitting and are much faster than mechanistic models.
        With model selector, empirical models are auto-enabled when present.
    initial_ratio_r : float, default=1.0
        Fixed [B]0/[A]0 ratio for the A+B->C bimolecular model, when included via
        include_ode_models or models_to_try. At the default r=1.0 (stoichiometric),
        A+B->C is mathematically identical to F2, so this only matters if the true
        starting ratio of the two reactants is known and not 1:1.
    bootstrap_iterations : int, default=100
        Number of bootstrap iterations for confidence intervals
    confidence_level : float, default=0.95
        Confidence level for intervals (0.95 = 95%)
    filter_implausible : bool, default=False
        If True, exclude models with physically implausible parameters from ranking.
        Uses permissive bounds (Ea: 10-400 kJ/mol, A: 10^-2 to 10^25 s^-1) to catch
        only clearly unphysical values. Implausible models are still fitted but won't
        appear in top_models or be selected. Useful for avoiding unrealistic extrapolations.
    report_format : str, default='interactive'
        Report visualization format: 'interactive', 'static', or 'both'
    report_path : str or Path, optional
        Path to save HTML report. If None, no report is generated.
    output_format : str, default='dict'
        Output format: 'dict', 'json', or 'both'
    auto_open : bool, default=False
        Automatically open HTML report in browser
    progress_callback : Callable[[str, Dict], None], optional
        Callback function for progress updates
    n_jobs : int, default=-1
        Number of worker processes for bootstrap fitting. The default uses all
        but one available CPU core; the worker count is capped at the number of
        bootstrap iterations. Set to 1 to use a single worker process.
    **loader_kwargs
        Additional keyword arguments passed to load_data_file()

    Returns
    -------
    Union[Dict, str, Tuple[Dict, str]]
        Results in requested format:
        - 'dict': Dictionary with results
        - 'json': JSON string
        - 'both': Tuple of (dict, json_string)

    Examples
    --------
    Basic usage with file paths:

    >>> def progress(msg, data):
    ...     print(f"{data['timestamp']} - {msg}")
    >>> results = auto_model_isothermal_data(
    ...     data_files=['data_313K.csv', 'data_323K.csv'],
    ...     predict=(1, 'year'),
    ...     report_path='report.html',
    ...     progress_callback=progress
    ... )
    >>> print(results['selected_model']['model_name'])
    'F1_model'

    JSON input for API:

    >>> json_data = {
    ...     'time': [0, 86400, 172800],
    ...     'temperature': [313, 313, 313],
    ...     'conversion': [0.0, 0.1, 0.2]
    ... }
    >>> results = auto_model_isothermal_data(
    ...     data_files=[json_data],
    ...     output_format='json',
    ...     report_path=None
    ... )
    >>> import json
    >>> data = json.loads(results)
    """
    # Initialize progress callback
    progress = _create_progress_wrapper(progress_callback)

    # Start timer
    start_time = time.time()

    # Step 1: Load data
    progress("Loading data files...", {'step': 1, 'total_steps': 7})
    datasets = []

    for i, data_source in enumerate(data_files):
        if isinstance(data_source, KineticDataset):
            datasets.append(data_source)
            progress(f"Loaded dataset {i+1}/{len(data_files)} (KineticDataset object)",
                    {'file_index': i+1, 'total_files': len(data_files)})

        elif isinstance(data_source, dict):
            # JSON data
            dataset = parse_json_data(data_source)
            datasets.append(dataset)
            progress(f"Loaded dataset {i+1}/{len(data_files)} (JSON data)",
                    {'file_index': i+1, 'total_files': len(data_files)})

        else:
            # File path
            filepath = Path(data_source)
            dataset = load_data_file(filepath, **loader_kwargs)
            datasets.append(dataset)
            progress(f"Loaded dataset {i+1}/{len(data_files)}: {filepath.name}",
                    {'file_index': i+1, 'total_files': len(data_files), 'filename': filepath.name})

    if not datasets:
        raise ValueError("No datasets were successfully loaded")

    _harmonize_loaded_datasets(datasets, loader_kwargs.get('readout_type', 'increasing'), progress)

    # Convert input temperatures to Kelvin if needed
    if input_temperature_units.upper() != 'K':
        progress(f"Converting input temperatures from {input_temperature_units} to Kelvin...", {})
        for dataset in datasets:
            dataset.temperature = np.array([
                _temperature_to_kelvin(t, input_temperature_units)
                for t in dataset.temperature
            ])

    # Step 2: Setup models
    progress("Setting up kinetic models...", {'step': 2})

    if models_to_try is None:
        models_to_try = models.default.copy()
        # Auto-add ODE models if requested via deprecated flag (backward compatibility)
        if include_ode_models:
            models_to_try.extend(ODE_MODELS)
            progress("Including ODE models (slower but more flexible)...", {})
        # Auto-add empirical models if requested via deprecated flag (backward compatibility)
        if include_empirical_models:
            models_to_try.extend(EMPIRICAL_MODELS)
            progress("Including empirical models (global Arrhenius fitting)...", {})
    else:
        # Auto-detect model types from models_to_try list
        # (When using model selector, include_ flags are redundant)
        has_ode = any(m in ODE_MODELS for m in models_to_try)
        has_empirical = any(m in EMPIRICAL_MODELS for m in models_to_try)

        if has_ode:
            progress("ODE models detected in selection (slower but more flexible)...", {})
        if has_empirical:
            progress("Empirical models detected in selection (global Arrhenius fitting)...", {})

    # Friedman auto-add when >=3 temperatures available
    # (Always auto-detect rather than requiring opt-in flag)
    n_distinct_temps = len({round(float(ds.temperature.mean())) for ds in datasets})
    if n_distinct_temps >= 3 and 'Friedman' not in models_to_try:
        models_to_try.append('Friedman')
        progress(f"Including Friedman model-free analysis ({n_distinct_temps} distinct temperatures detected)...", {})

    # Get smart initial guesses if ODE models are included
    smart_guess = None
    has_ode = any('->' in m or '+' in m for m in models_to_try)
    if has_ode and len(datasets) > 0:
        progress("Getting smart initial guesses for ODE models...", {})
        smart_guess = _get_smart_initial_guess(datasets, progress)

    models_config, initial_guesses, bounds = _setup_model_configs(models_to_try, smart_guess, initial_ratio_r)

    progress(f"Configured {len(models_config)} models to try: {', '.join(models_to_try)}",
            {'n_models': len(models_config), 'models': models_to_try})

    # Step 3: Fit models - we need to do this manually to keep FitResult objects
    progress("Fitting kinetic models (this may take a while)...", {'step': 3})

    all_fit_results = []
    fit_result_names = {}  # Map custom names to FitResult objects

    for model_info in models_config:
        custom_name = model_info['name']
        # Get display name (e.g., "F1 (first-order)" instead of "F1_model")
        base_model = custom_name.replace('_model', '')
        display_name = MODEL_DISPLAY_NAMES.get(base_model, base_model)

        progress(f"Fitting {display_name}...", {'model': custom_name})

        # Use multistart local fitting for ODE models (cheap insurance against a
        # bad local optimum now that a single fit_kinetic_model() call is fast).
        use_multistart = '->' in model_info['type'] or '+' in model_info['type']

        if model_info['type'] in EMPIRICAL_MODELS:
            # Empirical model - global fitting with Arrhenius
            progress(f"  Using global Arrhenius fitting (empirical model)...", {})
            try:
                fit_res = fit_empirical_global(
                    datasets=datasets,
                    model_type=model_info['type'],
                    initial_guess=initial_guesses.get(custom_name),
                    parameter_bounds=bounds.get(custom_name),
                    use_global_optimizer=False,  # Use local optimizer by default
                    verbose=False
                )
            except Exception as e:
                fit_res = FitResult(
                    model_name=f'Empirical_{model_info["type"]}',
                    parameters={}, success=False,
                    message=f"Empirical fit failed: {e}",
                    rss=np.inf, n_datapoints=0, n_parameters=0
                )
        elif model_info['type'] == 'Friedman':
            progress("  Running Friedman isoconversional analysis (model-free)...", {})
            try:
                # run_friedman()'s default alpha_levels (0.05-0.95) assumes every
                # dataset reaches near-complete conversion. Accelerated-aging /
                # short-duration studies often don't (e.g. mild conditions might
                # only reach a few % conversion) -- using the fixed default there
                # would ask for a regression point at every alpha but get none
                # with >=2 datasets overlapping, and the whole analysis fails.
                # Use the conversion range every dataset actually reaches instead
                # (10% margin below the tightest dataset's max, so the highest
                # requested alpha still has real data around it to regress on).
                max_reachable = min(ds.conversion.max() for ds in datasets if len(ds.conversion) > 0)
                alpha_hi = min(0.95, max_reachable * 0.9)
                alpha_lo = min(0.05, alpha_hi)  # keep the low end below alpha_hi even in a narrow-range case
                friedman_alpha_levels = np.linspace(alpha_lo, alpha_hi, 19)
                iso_result = run_friedman(datasets, alpha_levels=friedman_alpha_levels)
                fit_res = _wrap_friedman_as_fit_result(iso_result, datasets)
            except Exception as e:
                fit_res = FitResult(model_name='Friedman', parameters={}, success=False,
                                    message=f"Friedman analysis failed: {e}",
                                    rss=np.inf, n_datapoints=0, n_parameters=0)
        # Use stability checks for A+B->C (can be numerically unstable)
        elif model_info['type'] == 'A+B->C':
            if use_multistart:
                progress(f"  Using multistart local fitting with stability checks...", {})
            fit_res = _fit_with_stability_check(
                model_info=model_info,
                datasets=datasets,
                initial_guesses=initial_guesses[custom_name].copy(),
                bounds=bounds.get(custom_name).copy(),
                use_multistart=use_multistart,
                progress=progress
            )
        elif use_multistart:
            progress(f"  Using multistart local fitting (faster for ODE models)...", {})
            try:
                fit_res = _multistart_fit(
                    datasets=datasets,
                    model_name=model_info['type'],
                    model_definition_args=model_info['def_args'],
                    initial_guesses=initial_guesses[custom_name],
                    parameter_bounds=bounds.get(custom_name),
                    progress_callback=progress
                )
            except Exception as e:
                progress(f"  Multistart fit failed: {e}. Falling back to traditional...", {})
                fit_res = fit_kinetic_model(
                    datasets=datasets,
                    model_name=model_info['type'],
                    model_definition_args=model_info['def_args'],
                    initial_guesses=initial_guesses[custom_name],
                    parameter_bounds=bounds.get(custom_name)
                )
        else:
            # Traditional optimization
            fit_res = fit_kinetic_model(
                datasets=datasets,
                model_name=model_info['type'],
                model_definition_args=model_info['def_args'],
                initial_guesses=initial_guesses[custom_name],
                parameter_bounds=bounds.get(custom_name)
            )

        if fit_res.success:
            # Store the custom name in metadata but keep base model_name for compatibility
            fit_result_names[custom_name] = fit_res
            all_fit_results.append(fit_res)
            progress(f"{display_name} fit successful (R²={fit_res.r_squared:.4f})",
                    {'model': custom_name, 'r_squared': fit_res.r_squared})
        else:
            # Show error message to help debug
            error_msg = fit_res.message if hasattr(fit_res, 'message') else 'Unknown error'
            progress(f"{display_name} fit failed: {error_msg}", {'model': custom_name, 'error': error_msg})

    if not all_fit_results:
        raise RuntimeError("No models were successfully fitted. Check your data and try again.")

    # Filter out physically implausible models if requested
    fit_results_to_rank = all_fit_results
    if filter_implausible:
        plausible_fits = [fr for fr in all_fit_results
                          if fr.is_physically_plausible is not False]
        excluded_count = len(all_fit_results) - len(plausible_fits)
        if excluded_count > 0:
            excluded_names = [fr.model_name for fr in all_fit_results
                             if fr.is_physically_plausible is False]
            progress(f"Excluded {excluded_count} implausible model(s): {', '.join(excluded_names)}",
                    {'excluded_count': excluded_count})
        if plausible_fits:
            fit_results_to_rank = plausible_fits
        else:
            warnings.warn("All models have implausible parameters; disabling filter")
            fit_results_to_rank = all_fit_results

    # Rank the models
    ranked_models = rank_models(fit_results_to_rank)

    # Add custom names to ranked results
    for ranked_model in ranked_models:
        # Find the corresponding custom name
        for custom_name, fit_res in fit_result_names.items():
            if (ranked_model['parameters'] == fit_res.parameters and
                ranked_model['stats']['rss'] == fit_res.rss):
                ranked_model['model_name'] = custom_name
                break

    progress(f"Successfully fitted {len(ranked_models)} models",
            {'n_successful': len(ranked_models)})

    # Prefer the simplest model when the data can't distinguish mechanisms. Move it to the
    # front so bootstrap, predictions and simulation all use it.
    chosen, selection_reason = _select_simplest_equivalent(ranked_models)
    ranked_models.remove(chosen)
    ranked_models.insert(0, chosen)
    for rank, m in enumerate(ranked_models, start=1):
        m['rank'] = rank
    progress(f"Model choice: {chosen['model_name'].replace('_model', '')} - {selection_reason}", {})

    # Step 4: Select top N models - keep both dict and FitResult
    top_models_data = ranked_models[:top_n]
    top_fit_results = [fit_result_names[m['model_name']] for m in top_models_data
                       if m['model_name'] in fit_result_names]

    progress(f"Selected top {min(top_n, len(ranked_models))} models for detailed analysis",
            {'step': 4, 'n_selected': len(top_models_data)})

    # Step 5: Run bootstrap for top model only (to save time)
    progress("Running bootstrap analysis for top model...", {'step': 5})
    bootstrap_results = {}

    if bootstrap_iterations > 0 and top_fit_results:
        top_fit = top_fit_results[0]
        top_custom_name = next(n for n, r in fit_result_names.items() if r is top_fit)
        try:
            if top_fit.model_name == "Friedman":
                # No optimizer/ODE simulation to bootstrap here -- resample at the
                # per-alpha regression-input level instead (see run_bootstrap_friedman).
                iso_result = top_fit.model_definition_args['iso_result']
                bootstrap_result = run_bootstrap_friedman(
                    datasets=datasets, iso_result=iso_result,
                    n_iterations=bootstrap_iterations, confidence_level=confidence_level,
                )
            else:
                bootstrap_result = run_bootstrap(
                    datasets=datasets,
                    fit_result=top_fit,
                    optimizer_options={'method': 'Powell'},
                    parameter_bounds=bounds.get(top_custom_name),
                    n_iterations=bootstrap_iterations,
                    confidence_level=confidence_level,
                    n_jobs=n_jobs
                )
            if bootstrap_result:
                bootstrap_results[top_fit.model_name] = bootstrap_result
                progress(f"Bootstrap complete for {top_fit.model_name}",
                        {'iterations': bootstrap_result.n_iterations})
        except Exception as e:
            warnings.warn(f"Bootstrap failed for {top_fit.model_name}: {e}")

    # Step 6: Generate predictions
    predictions_dict = None
    prediction_result = None  # Store actual PredictionResult for regulatory plots
    prediction_time_sec = None

    if predict is not None and top_fit_results:
        progress("Generating predictions...", {'step': 6})

        # Parse predict parameter (time_value, time_unit) or (time_value, time_unit, temp)
        if len(predict) == 3:
            pred_value, pred_unit, temp_input = predict
            # Convert input temperature to Kelvin
            temp_K = _temperature_to_kelvin(temp_input, input_temperature_units)
        elif len(predict) == 2:
            pred_value, pred_unit = predict
            # Use predict_temperature_K parameter if provided, else first dataset temperature
            if predict_temperature_K is not None:
                # Convert input temperature to Kelvin
                temp_K = _temperature_to_kelvin(predict_temperature_K, input_temperature_units)
            else:
                temp_K = datasets[0].temperature.mean()  # Already in Kelvin after conversion
        else:
            raise ValueError("predict must be (time_value, time_unit) or (time_value, time_unit, temp)")

        prediction_time_sec = _convert_time_to_seconds(pred_value, pred_unit)
        time_eval = np.linspace(0, prediction_time_sec, 100)

        try:
            # Check if it's an empirical model
            if top_fit_results[0].model_name.startswith('Empirical_'):
                # Use empirical prediction (no bootstrap support yet)
                conversion_mean = predict_empirical(
                    fit_result=top_fit_results[0],
                    time_points=time_eval,
                    temperature_K=temp_K
                )
                # Create PredictionResult manually
                from akts.datatypes import PredictionResult
                pred_result = PredictionResult(
                    time=time_eval,
                    conversion=conversion_mean,
                    conversion_ci=None,  # No CI for empirical models yet
                    temperature=np.full_like(time_eval, temp_K)
                )
            else:
                # Use mechanistic prediction
                pred_result = predict_conversion(
                    kinetic_description=top_fit_results[0],
                    temperature_program=lambda t: temp_K,
                    simulation_time_sec=time_eval,
                    initial_alpha=0.0,
                    bootstrap_result=bootstrap_results.get(top_fit_results[0].model_name)
                )

            # Store the PredictionResult for regulatory plots
            prediction_result = pred_result

            # Convert temperature to output units
            temp_output = _temperature_from_kelvin(temp_K, output_temperature_units)

            predictions_dict = {
                'time': pred_result.time.tolist(),
                'conversion_mean': pred_result.conversion.tolist(),
                'time_unit': 'seconds',
                'requested_time': {'value': pred_value, 'unit': pred_unit},
                'temperature': temp_output,
                'temperature_units': output_temperature_units
            }

            if pred_result.conversion_ci is not None:
                predictions_dict['conversion_lower'] = pred_result.conversion_ci[0].tolist()
                predictions_dict['conversion_upper'] = pred_result.conversion_ci[1].tolist()
                # Also include Kelvin for backward compatibility
                predictions_dict['temperature_K'] = temp_K

            progress(f"Prediction for {pred_value} {pred_unit} at {temp_output:.1f} {output_temperature_units}: {pred_result.conversion[-1]:.2%} conversion",
                    {'time_value': pred_value, 'time_unit': pred_unit, 'temperature': temp_output, 'temperature_units': output_temperature_units})
        except Exception as e:
            warnings.warn(f"Prediction failed: {e}")

    # Step 6b: Calculate ICH Q1E regulatory outputs (always include when predictions are made)
    regulatory_results = None
    if predictions_dict is not None and top_fit_results:
        try:
            progress("Calculating ICH Q1E regulatory analysis...", {'step': '6b'})

            # Determine study duration from datasets (max time observed)
            max_time_sec = max(ds.time.max() for ds in datasets)
            study_duration_months = max_time_sec / (30.44 * 24 * 3600)  # 30.44 days/month average

            # ICH Q1E requires confidence intervals - run bootstrap if not already done
            bootstrap_result_selected = bootstrap_results.get(top_fit_results[0].model_name) if bootstrap_results else None

            if not bootstrap_result_selected:
                # Run bootstrap specifically for regulatory compliance
                progress("Running bootstrap for ICH Q1E regulatory compliance...", {'step': '6b-bootstrap'})
                top_fit = top_fit_results[0]
                top_custom_name = next(n for n, r in fit_result_names.items() if r is top_fit)

                try:
                    if top_fit.model_name == "Friedman":
                        iso_result = top_fit.model_definition_args['iso_result']
                        bootstrap_result_selected = run_bootstrap_friedman(
                            datasets=datasets, iso_result=iso_result,
                            n_iterations=50,  # Minimum for regulatory
                            confidence_level=confidence_level,
                        )
                    else:
                        bootstrap_result_selected = run_bootstrap(
                            datasets=datasets,
                            fit_result=top_fit,
                            optimizer_options={'method': 'Powell'},
                            parameter_bounds=bounds.get(top_custom_name),
                            n_iterations=50,  # Minimum for regulatory
                            confidence_level=confidence_level,
                            n_jobs=n_jobs
                        )
                    if bootstrap_result_selected:
                        bootstrap_results[top_fit.model_name] = bootstrap_result_selected
                        progress(f"Bootstrap complete for regulatory analysis",
                                {'iterations': bootstrap_result_selected.n_iterations})
                except Exception as e:
                    warnings.warn(f"Bootstrap for regulatory analysis failed: {e}. Regulatory section will be omitted.")
                    bootstrap_result_selected = None

            # Calculate shelf-life with one-sided CI at 5% degradation (standard threshold)
            # Use same temperature as predictions
            if bootstrap_result_selected:
                shelf_life_result = time_to_conversion(
                    fit_result=top_fit_results[0],
                    target_conversion=0.05,  # Standard pharmaceutical threshold (5% degradation)
                    temperature_K=temp_K,
                    bootstrap_result=bootstrap_result_selected,
                    one_sided_ci=True  # ICH Q1E mode (conservative estimate)
                )

                if shelf_life_result['time_sec'] is not None:
                    # Convert to months
                    shelf_life_months = shelf_life_result['time_sec'] / (30.44 * 24 * 3600)
                    shelf_life_lower_months = (
                        shelf_life_result['time_lower_sec'] / (30.44 * 24 * 3600)
                        if shelf_life_result['time_lower_sec'] is not None else None
                    )

                    # Calculate ICH Q1E extrapolation ceiling
                    ich_ceiling = calculate_ich_q1e_ceiling(study_duration_months, is_long_term=True)

                    regulatory_results = {
                        'shelf_life_months': shelf_life_months,
                        'shelf_life_lower_95': shelf_life_lower_months,
                        'target_conversion': 0.05,
                        'storage_temp_K': temp_K,
                        'study_duration_months': study_duration_months,
                        'ich_ceiling_months': ich_ceiling,
                        'exceeds_guideline': (
                            shelf_life_lower_months > ich_ceiling
                            if shelf_life_lower_months is not None else False
                        ),
                        'prediction': prediction_result  # PredictionResult object for regulatory plots
                    }

                    # Convert temperature for progress message
                    temp_progress = _temperature_from_kelvin(temp_K, output_temperature_units)
                    progress(
                        f"ICH Q1E shelf-life: {shelf_life_lower_months:.1f} months "
                        f"(95% lower bound at 5% degradation, {temp_progress:.1f} {output_temperature_units})",
                        {'shelf_life_months': shelf_life_lower_months, 'ich_ceiling_months': ich_ceiling}
                    )
        except Exception as e:
            warnings.warn(f"ICH Q1E regulatory calculation failed: {e}")

    # Step 7: Check convergence and select model(s)
    progress("Analyzing model convergence...", {'step': 7})

    selected_params = top_models_data[0].get('parameters', {})
    sanity_flags = _physical_sanity_flags(selected_params)
    selected_model_data = {
        'model_name': top_models_data[0]['model_name'],
        'rank': 1,
        'reason': selection_reason,
        'parameters': selected_params,
        'statistics': top_models_data[0].get('stats', {}),  # Note: rank_models uses 'stats' not 'statistics'
        'physical_sanity_flags': sanity_flags
    }

    progress(f"Selected model: {selected_model_data['model_name']}", {})
    for flag in sanity_flags:
        progress(f"  [!] {flag}", {'sanity_flag': flag})

    # Step 7: Run temperature excursion simulation
    simulation_dict = None
    if simulate is not None and top_fit_results:
        progress("Running temperature excursion simulation...", {'step': 7})
        try:
            # Convert simulate times to seconds and temperatures to Kelvin
            sim_times_sec = np.array([_convert_time_to_seconds(t, simulate_time_unit) for t, _ in simulate])
            sim_temps_K = np.array([_temperature_to_kelvin(temp, input_temperature_units) for _, temp in simulate])

            # Create temperature program from the profile
            def temp_program(t):
                return np.interp(t, sim_times_sec, sim_temps_K)

            # Generate fine time grid for smooth simulation
            time_eval = np.linspace(sim_times_sec[0], sim_times_sec[-1], 200)

            # Run simulation with selected model
            if top_fit_results[0].model_name.startswith('Empirical_'):
                # Empirical models don't support variable temperature simulation yet
                # Use average temperature as approximation
                avg_temp_K = np.mean(sim_temps_K)
                conversion_mean = predict_empirical(
                    fit_result=top_fit_results[0],
                    time_points=time_eval,
                    temperature_K=avg_temp_K
                )
                from akts.datatypes import PredictionResult
                sim_result = PredictionResult(
                    time=time_eval,
                    conversion=conversion_mean,
                    conversion_ci=None,
                    temperature=np.full_like(time_eval, avg_temp_K)
                )
                progress("  Note: Empirical model simulation uses average temperature (variable temp not supported)", {})
            else:
                sim_result = predict_conversion(
                    kinetic_description=top_fit_results[0],
                    temperature_program=temp_program,
                    simulation_time_sec=time_eval,
                    initial_alpha=0.0,
                    bootstrap_result=bootstrap_results.get(top_fit_results[0].model_name)
                )

            # Convert temperatures to output units
            temps_output = [_temperature_from_kelvin(temp_program(t), output_temperature_units)
                           for t in sim_result.time]
            profile_output = [(t, _temperature_from_kelvin(_temperature_to_kelvin(temp, input_temperature_units),
                                                            output_temperature_units))
                             for t, temp in simulate]

            simulation_dict = {
                'time': sim_result.time.tolist(),
                'conversion_mean': sim_result.conversion.tolist(),
                'temperature': temps_output,
                'temperature_units': output_temperature_units,
                'time_unit': simulate_time_unit,
                'input_profile': profile_output
            }

            if sim_result.conversion_ci is not None:
                simulation_dict['conversion_lower'] = sim_result.conversion_ci[0].tolist()
                simulation_dict['conversion_upper'] = sim_result.conversion_ci[1].tolist()

            final_conversion = sim_result.conversion[-1]
            max_temp_K = sim_temps_K.max()
            max_temp_output = _temperature_from_kelvin(max_temp_K, output_temperature_units)
            progress(f"Simulation complete: {final_conversion:.2%} conversion (max temp: {max_temp_output:.1f} {output_temperature_units})",
                    {'final_conversion': final_conversion, 'max_temp': max_temp_output, 'temperature_units': output_temperature_units})
        except Exception as e:
            warnings.warn(f"Temperature excursion simulation failed: {e}")

    # Step 7.5: Simulate conversion for top models for plotting
    for fit_res in top_fit_results:
        try:
            simulated_conversions = []
            for ds in datasets:
                temp_func = lambda t: np.interp(t, ds.time, ds.temperature)
                pred_result = predict_conversion(
                    kinetic_description=fit_res,
                    temperature_program=temp_func,
                    simulation_time_sec=ds.time,
                    initial_alpha=0.0
                )
                simulated_conversions.append(pred_result.conversion)
            # Add as attribute to FitResult object
            fit_res.conversion_simulated = simulated_conversions
        except Exception as e:
            warnings.warn(f"Failed to simulate conversion for plotting: {e}")
            fit_res.conversion_simulated = None

    # Step 8: Generate summary
    actual_bootstrap_iters = 0
    if bootstrap_results:
        for bs_result in bootstrap_results.values():
            actual_bootstrap_iters = max(actual_bootstrap_iters, bs_result.n_iterations)

    # Calculate temperature ranges in both Kelvin and output units
    temp_range_K = [
        float(min(ds.temperature.min() for ds in datasets)),
        float(max(ds.temperature.max() for ds in datasets))
    ]
    temp_range_output = [
        round(_temperature_from_kelvin(temp_range_K[0], output_temperature_units), 2),
        round(_temperature_from_kelvin(temp_range_K[1], output_temperature_units), 2)
    ]

    # Format temperature range as string with units
    temp_range_str = f"{temp_range_output[0]:.2f} - {temp_range_output[1]:.2f} °{output_temperature_units}"

    summary = {
        'datasets_count': len(datasets),
        'total_datapoints': sum(len(ds.time) for ds in datasets),
        'temperature_range': temp_range_str,  # Formatted string with units
        'models_tried': len(models_config),
        'models_successful': len(ranked_models),
        'top_n_selected': len(top_models_data),
        'bootstrap_iterations': actual_bootstrap_iters
    }

    # Step 9: Generate HTML report
    if report_path:
        progress("Generating HTML report...", {'step': 8})
        report_path_final = generate_isothermal_report(
            datasets=datasets,
            top_models=top_models_data,
            selected_model=selected_model_data,
            predictions=predictions_dict,
            simulation=simulation_dict,
            report_path=report_path,
            report_format=report_format,
            summary=summary,
            fit_results=top_fit_results,  # Pass the actual FitResult objects for plotting
            regulatory=regulatory_results  # ICH Q1E regulatory analysis (always included if available)
        )
        progress(f"Report saved to: {report_path_final}", {'report_path': report_path_final})

        if auto_open:
            import webbrowser
            webbrowser.open(f'file://{Path(report_path_final).absolute()}')

    # Step 10: Prepare output
    elapsed_time = time.time() - start_time
    progress(f"Analysis complete in {elapsed_time:.1f} seconds", {'elapsed_seconds': elapsed_time})

    results_dict = {
        'top_models': top_models_data,
        'selected_model': selected_model_data,
        'selection_reason': selection_reason,
        'predictions': predictions_dict,
        'simulation': simulation_dict,
        'report_path': str(report_path) if report_path else None,
        'bootstrap_results': bootstrap_results if bootstrap_results else None,
        'summary': summary,
        'regulatory': regulatory_results,  # ICH Q1E regulatory analysis
        # Additional objects for advanced users
        'datasets': datasets,  # Original datasets for custom plotting
        'fit_results': fit_result_names,  # FitResult objects by model name
        'prediction': predictions_dict.get('prediction') if predictions_dict else None,  # PredictionResult object
    }

    # Format output based on output_format
    if output_format == 'dict':
        return results_dict
    elif output_format == 'json':
        return serialize_results_to_json(
            top_models=top_models_data,
            selected_model=selected_model_data,
            predictions=predictions_dict,
            report_path=report_path,
            bootstrap_results=bootstrap_results,
            summary=summary,
            regulatory=regulatory_results
        )
    elif output_format == 'both':
        json_str = serialize_results_to_json(
            top_models=top_models_data,
            selected_model=selected_model_data,
            predictions=predictions_dict,
            report_path=report_path,
            bootstrap_results=bootstrap_results,
            summary=summary,
            regulatory=regulatory_results
        )
        return results_dict, json_str
    else:
        raise ValueError(f"Invalid output_format: {output_format}. Must be 'dict', 'json', or 'both'")
