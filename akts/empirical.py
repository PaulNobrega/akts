"""
Empirical degradation models for kinetic analysis.

These models fit conversion directly as a function of time, rather than using
differential equations. They support multi-temperature global fitting with
Arrhenius temperature dependence.

Models included:
- First Order: α = A·exp(-k(T)·t)
- Linear: α = k(T)·t + C
- Square Root: α = k(T)·√t + C
- Logistic: α = A/(1 + B·exp(-k(T)·t))
- Exponential: α = A·(1 - exp(-k(T)·t)) + C

All models use k(T) = A·exp(-Ea/RT) for temperature dependence.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from scipy.optimize import least_squares, differential_evolution
from scipy.stats import t as t_dist

from akts.datatypes import KineticDataset, FitResult, PredictionResult, BootstrapResult
from akts.utils import R_GAS, EA_BOUNDS


# ============================================================================
# Empirical Model Functions
# ============================================================================

def empirical_first_order(t: np.ndarray, k: float, A: float) -> np.ndarray:
    """
    First-order empirical: α = A·exp(-k·t)

    Parameters
    ----------
    t : array
        Time
    k : float
        Rate constant (temperature-dependent)
    A : float
        Amplitude/initial value

    Returns
    -------
    array
        Predicted conversion
    """
    return A * np.exp(-k * t)


def empirical_linear(t: np.ndarray, k: float, C: float) -> np.ndarray:
    """
    Linear empirical: α = k·t + C

    Parameters
    ----------
    t : array
        Time
    k : float
        Slope (temperature-dependent rate)
    C : float
        Intercept/baseline

    Returns
    -------
    array
        Predicted conversion
    """
    return k * t + C


def empirical_sqrt(t: np.ndarray, k: float, C: float) -> np.ndarray:
    """
    Square root empirical: α = k·√t + C

    Parameters
    ----------
    t : array
        Time
    k : float
        Rate coefficient (temperature-dependent)
    C : float
        Intercept/baseline

    Returns
    -------
    array
        Predicted conversion
    """
    return k * np.sqrt(t) + C


def empirical_logistic(t: np.ndarray, k: float, A: float, B: float) -> np.ndarray:
    """
    Logistic empirical: α = A / (1 + B·exp(-k·t))

    Parameters
    ----------
    t : array
        Time
    k : float
        Growth rate (temperature-dependent)
    A : float
        Maximum conversion (upper asymptote)
    B : float
        Shape parameter (determines inflection point)

    Returns
    -------
    array
        Predicted conversion
    """
    return A / (1.0 + B * np.exp(-k * t))


def empirical_exponential(t: np.ndarray, k: float, A: float, C: float) -> np.ndarray:
    """
    Exponential rise to maximum: α = A·(1 - exp(-k·t)) + C

    Parameters
    ----------
    t : array
        Time
    k : float
        Rate constant (temperature-dependent)
    A : float
        Amplitude (maximum change)
    C : float
        Baseline offset

    Returns
    -------
    array
        Predicted conversion
    """
    return A * (1.0 - np.exp(-k * t)) + C


# ============================================================================
# Arrhenius Temperature Dependence
# ============================================================================

def arrhenius_rate(T: float, Ea: float, A: float) -> float:
    """
    Calculate rate constant from Arrhenius equation.

    k(T) = A·exp(-Ea/RT)

    Parameters
    ----------
    T : float
        Temperature (K)
    Ea : float
        Activation energy (J/mol)
    A : float
        Pre-exponential factor (units depend on model)

    Returns
    -------
    float
        Rate constant k(T)
    """
    if T <= 0:
        return 0.0
    return A * np.exp(-Ea / (R_GAS * T))


# ============================================================================
# Global Fitting
# ============================================================================

def fit_empirical_global(
    datasets: List[KineticDataset],
    model_type: str,
    initial_guess: Optional[Dict[str, float]] = None,
    parameter_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    use_global_optimizer: bool = False,
    verbose: bool = False
) -> FitResult:
    """
    Global fit of empirical model across multiple temperatures.

    All datasets are fitted simultaneously with shared Arrhenius parameters
    (Ea, A) and model-specific shape parameters.

    Parameters
    ----------
    datasets : List[KineticDataset]
        Multi-temperature isothermal datasets
    model_type : str
        Empirical model type:
        - 'First_Order': α = A_scale·exp(-k(T)·t)
        - 'Linear': α = k(T)·t + C
        - 'Sqrt': α = k(T)·√t + C
        - 'Logistic': α = A_max/(1 + B·exp(-k(T)·t))
        - 'Exponential': α = A_amp·(1 - exp(-k(T)·t)) + C
    initial_guess : Dict, optional
        Initial parameter guesses
    parameter_bounds : Dict, optional
        Parameter bounds {param: (lower, upper)}
    use_global_optimizer : bool, default=False
        Use differential evolution (slower but more robust)
    verbose : bool, default=False
        Print progress

    Returns
    -------
    FitResult
        Fitted parameters, statistics, predictions
    """
    # Select model function and parameter names
    model_registry = {
        'First_Order': (empirical_first_order, ['Ea', 'A', 'A_scale']),
        'Linear': (empirical_linear, ['Ea', 'A', 'C']),
        'Sqrt': (empirical_sqrt, ['Ea', 'A', 'C']),
        'Logistic': (empirical_logistic, ['Ea', 'A', 'A_max', 'B']),
        'Exponential': (empirical_exponential, ['Ea', 'A', 'A_amp', 'C']),
    }

    if model_type not in model_registry:
        raise ValueError(f"Unknown empirical model: {model_type}. "
                        f"Choose from {list(model_registry.keys())}")

    model_func, param_names = model_registry[model_type]
    n_params = len(param_names)

    # Default initial guesses
    if initial_guess is None:
        initial_guess = {
            'Ea': 80000.0,  # 80 kJ/mol
            'A': 1e10,      # Pre-exponential factor
            'A_scale': 1.0, 'A_max': 1.0, 'A_amp': 1.0,  # Amplitudes
            'C': 0.0,       # Baselines
            'B': 1.0,       # Shape factor
        }

    # Default bounds
    if parameter_bounds is None:
        parameter_bounds = {
            'Ea': EA_BOUNDS,
            'A': (1e-5, 1e20),
            'A_scale': (0.0, 10.0),
            'A_max': (0.0, 10.0),
            'A_amp': (0.0, 10.0),
            'C': (-1.0, 1.0),
            'B': (0.01, 100.0),
        }

    # Extract initial values and bounds
    x0 = np.array([initial_guess.get(p, 1.0) for p in param_names])
    bounds_lower = np.array([parameter_bounds[p][0] for p in param_names])
    bounds_upper = np.array([parameter_bounds[p][1] for p in param_names])

    # Prepare data
    all_times = []
    all_temps = []
    all_conversions = []

    for ds in datasets:
        all_times.append(ds.time)
        all_temps.append(ds.temperature)
        all_conversions.append(ds.conversion)

    all_times = np.concatenate(all_times)
    all_temps = np.concatenate(all_temps)
    all_conversions = np.concatenate(all_conversions)
    n_points = len(all_times)

    # Residual function
    def residuals(params):
        param_dict = dict(zip(param_names, params))
        Ea = param_dict['Ea']
        A_pre = param_dict['A']

        predictions = np.zeros(n_points)
        idx = 0

        for ds in datasets:
            n_pts = len(ds.time)
            T_mean = np.mean(ds.temperature)
            k = arrhenius_rate(T_mean, Ea, A_pre)

            # Model-specific prediction
            if model_type == 'First_Order':
                A_scale = param_dict['A_scale']
                pred = empirical_first_order(ds.time, k, A_scale)
            elif model_type == 'Linear':
                C = param_dict['C']
                pred = empirical_linear(ds.time, k, C)
            elif model_type == 'Sqrt':
                C = param_dict['C']
                pred = empirical_sqrt(ds.time, k, C)
            elif model_type == 'Logistic':
                A_max = param_dict['A_max']
                B = param_dict['B']
                pred = empirical_logistic(ds.time, k, A_max, B)
            elif model_type == 'Exponential':
                A_amp = param_dict['A_amp']
                C = param_dict['C']
                pred = empirical_exponential(ds.time, k, A_amp, C)

            predictions[idx:idx+n_pts] = pred
            idx += n_pts

        return predictions - all_conversions

    # Optimize
    if use_global_optimizer:
        if verbose:
            print(f"[Empirical {model_type}] Using differential evolution (global optimizer)...")

        result = differential_evolution(
            lambda x: np.sum(residuals(x)**2),
            bounds=list(zip(bounds_lower, bounds_upper)),
            maxiter=500,
            seed=42,
            workers=1
        )
        params_opt = result.x
        success = result.success
    else:
        if verbose:
            print(f"[Empirical {model_type}] Using least squares (local optimizer)...")

        result = least_squares(
            residuals,
            x0=x0,
            bounds=(bounds_lower, bounds_upper),
            method='trf',
            max_nfev=5000
        )
        params_opt = result.x
        success = result.success

    # Calculate statistics
    final_residuals = residuals(params_opt)
    rss = np.sum(final_residuals**2)

    # R²
    ss_tot = np.sum((all_conversions - np.mean(all_conversions))**2)
    r_squared = 1.0 - (rss / ss_tot) if ss_tot > 0 else 0.0

    # AIC, BIC
    aic = n_points * np.log(rss / n_points) + 2 * n_params
    bic = n_points * np.log(rss / n_points) + n_params * np.log(n_points)

    # Build parameter dict
    fitted_params = dict(zip(param_names, params_opt))

    # Generate predictions for each dataset
    predictions = []
    for ds in datasets:
        T_mean = np.mean(ds.temperature)
        k = arrhenius_rate(T_mean, fitted_params['Ea'], fitted_params['A'])

        if model_type == 'First_Order':
            pred = empirical_first_order(ds.time, k, fitted_params['A_scale'])
        elif model_type == 'Linear':
            pred = empirical_linear(ds.time, k, fitted_params['C'])
        elif model_type == 'Sqrt':
            pred = empirical_sqrt(ds.time, k, fitted_params['C'])
        elif model_type == 'Logistic':
            pred = empirical_logistic(ds.time, k, fitted_params['A_max'], fitted_params['B'])
        elif model_type == 'Exponential':
            pred = empirical_exponential(ds.time, k, fitted_params['A_amp'], fitted_params['C'])

        predictions.append(pred)

    # Create FitResult
    fit_result = FitResult(
        model_name=f'Empirical_{model_type}',
        parameters=fitted_params,
        success=success,
        message=f"Global fit {'succeeded' if success else 'failed'}",
        r_squared=r_squared,
        rss=rss,
        aic=aic,
        bic=bic,
        n_parameters=n_params,
        n_datapoints=n_points,
        model_definition_args={
            'empirical_type': model_type,
            'datasets': datasets,  # Store for later use if needed
            'predictions': predictions
        }
    )

    return fit_result


# ============================================================================
# Prediction
# ============================================================================

def predict_empirical(
    fit_result: FitResult,
    time_points: np.ndarray,
    temperature_K: float
) -> np.ndarray:
    """
    Predict conversion using fitted empirical model.

    Parameters
    ----------
    fit_result : FitResult
        Fitted empirical model
    time_points : array
        Time points for prediction
    temperature_K : float
        Prediction temperature (K)

    Returns
    -------
    array
        Predicted conversion
    """
    model_type = fit_result.model_definition_args['empirical_type']
    params = fit_result.parameters

    # Calculate k(T)
    k = arrhenius_rate(temperature_K, params['Ea'], params['A'])

    # Model-specific prediction
    if model_type == 'First_Order':
        return empirical_first_order(time_points, k, params['A_scale'])
    elif model_type == 'Linear':
        return empirical_linear(time_points, k, params['C'])
    elif model_type == 'Sqrt':
        return empirical_sqrt(time_points, k, params['C'])
    elif model_type == 'Logistic':
        return empirical_logistic(time_points, k, params['A_max'], params['B'])
    elif model_type == 'Exponential':
        return empirical_exponential(time_points, k, params['A_amp'], params['C'])
    else:
        raise ValueError(f"Unknown model type: {model_type}")
