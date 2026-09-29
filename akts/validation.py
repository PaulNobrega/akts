# validation.py
"""
Model validation functions including cross-validation.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
import warnings

from .datatypes import KineticDataset, FitResult
from .core import fit_kinetic_model, get_log_param_names, _calculate_conversion_stats


def run_leave_one_out_cv(
    datasets: List[KineticDataset],
    model_name: str,
    model_definition_args: Dict,
    initial_guesses: Optional[Dict[str, float]] = None,
    parameter_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    solver_options: Optional[Dict] = None,
    optimizer_options: Optional[Dict] = None,
    verbose: bool = False
) -> Dict:
    """
    Leave-one-temperature-out cross-validation.

    Refits model N times, each time excluding one dataset (temperature),
    then scores the fit on the held-out dataset.

    Note on Model Selection with Fitted Parameters
    -----------------------------------------------
    For fitted-parameter models (Fn, SB, SB_mnp), simpler (lower exponent) models
    should be preferred when fit quality is similar. For example:
    - Fn with n≈1 should be preferred over n≈3 if R² is similar
    - SB with m,n≈(0.5,1) should be preferred over m,n≈(2,2) if fits are comparable

    Use CV to compare models, but apply Occam's Razor: when CV scores are within
    ~0.02-0.03, choose the simpler model. Fitted exponents near standard values
    (n=1,2; m=0.5, n=1.0) are more interpretable.

    Parameters
    ----------
    datasets : List[KineticDataset]
        All experimental datasets (typically one per temperature)
    model_name : str
        Model type ('single_step', 'A->B->C', etc.)
    model_definition_args : Dict
        Model definition arguments (e.g., {'f_alpha_model': 'F1'})
    initial_guesses : Dict[str, float], optional
        Initial parameter guesses. If None, uses full-data fit as warm start
    parameter_bounds : Dict[str, Tuple[float, float]], optional
        Parameter bounds for optimization
    solver_options : Dict, optional
        ODE solver options
    optimizer_options : Dict, optional
        Optimizer options (e.g., {'method': 'L-BFGS-B'})
    verbose : bool, default=False
        Print progress

    Returns
    -------
    Dict
        {
            'loo_results': List[Dict],  # Per-fold results
            'mean_held_out_r2': float,   # Average R² on held-out data
            'mean_held_out_rss': float,  # Average RSS on held-out data
            'cv_score': float,           # Overall CV score (higher is better)
            'n_folds': int,              # Number of folds
            'n_successful_folds': int    # Number of folds where fit succeeded
        }
    """
    n_datasets = len(datasets)

    if n_datasets < 3:
        warnings.warn("Cross-validation requires at least 3 datasets (temperatures). Returning None.")
        return {
            'loo_results': [],
            'mean_held_out_r2': np.nan,
            'mean_held_out_rss': np.nan,
            'cv_score': np.nan,
            'n_folds': n_datasets,
            'n_successful_folds': 0
        }

    solver_options = solver_options or {}
    optimizer_options = optimizer_options or {'method': 'L-BFGS-B'}

    # Get initial guesses from full-data fit (warm start)
    if initial_guesses is None:
        initial_guesses = _get_default_guesses(model_name, model_definition_args)
        if verbose:
            print("Running full-data fit for warm start...")
        try:
            full_fit = fit_kinetic_model(
                datasets=datasets,
                model_name=model_name,
                model_definition_args=model_definition_args,
                initial_guesses=initial_guesses,
                parameter_bounds=parameter_bounds,
                solver_options=solver_options,
                optimizer_options=optimizer_options,
                verbose=False
            )
            if full_fit.success:
                initial_guesses = full_fit.parameters
            else:
                warnings.warn("Full-data fit failed. Using default initial guesses.")
        except Exception as e:
            warnings.warn(f"Full-data fit error: {e}. Using default initial guesses.")

    loo_results = []
    n_successful = 0

    for i in range(n_datasets):
        # Exclude dataset i
        training_datasets = datasets[:i] + datasets[i+1:]
        held_out_dataset = datasets[i]
        held_out_temp = np.mean(held_out_dataset.temperature)

        if verbose:
            print(f"Fold {i+1}/{n_datasets}: Holding out T={held_out_temp-273.15:.1f}°C")

        # Refit on training data
        fit_res = fit_kinetic_model(
            datasets=training_datasets,
            model_name=model_name,
            model_definition_args=model_definition_args,
            initial_guesses=initial_guesses,  # Warm start
            parameter_bounds=parameter_bounds,
            solver_options=solver_options,
            optimizer_options=optimizer_options,
            verbose=False
        )

        # Score on held-out data
        if fit_res.success:
            n_successful += 1

            # Convert to logA for _calculate_conversion_stats
            params_logA = {}
            log_param_names = get_log_param_names(model_name)
            for k, v in fit_res.parameters.items():
                if k in log_param_names:
                    params_logA[f'log{k}'] = np.log(v)
                else:
                    params_logA[k] = v

            # Calculate held-out metrics
            held_out_rss, held_out_n_pts, held_out_r2, _, _ = \
                _calculate_conversion_stats(
                    datasets=[held_out_dataset],
                    params_logA=params_logA,
                    model_name=model_name,
                    model_definition_args=model_definition_args,
                    solver_options=solver_options
                )
        else:
            held_out_rss = np.inf
            held_out_r2 = np.nan
            held_out_n_pts = len(held_out_dataset.time)

        loo_results.append({
            'excluded_index': i,
            'excluded_temp_K': held_out_temp,
            'training_fit_success': fit_res.success,
            'held_out_rss': held_out_rss,
            'held_out_r2': held_out_r2,
            'held_out_n_points': held_out_n_pts
        })

    # Aggregate results
    valid_r2 = [r['held_out_r2'] for r in loo_results if np.isfinite(r['held_out_r2'])]
    mean_r2 = np.mean(valid_r2) if valid_r2 else np.nan

    valid_rss = [r['held_out_rss'] for r in loo_results if np.isfinite(r['held_out_rss'])]
    mean_rss = np.mean(valid_rss) if valid_rss else np.nan

    # CV score: mean R² (higher is better)
    cv_score = mean_r2

    return {
        'loo_results': loo_results,
        'mean_held_out_r2': mean_r2,
        'mean_held_out_rss': mean_rss,
        'cv_score': cv_score,
        'n_folds': n_datasets,
        'n_successful_folds': n_successful
    }


def _get_default_guesses(model_name: str, model_definition_args: Dict) -> Dict[str, float]:
    """Get default initial guesses for a model."""
    guesses = {}

    if model_name == "single_step":
        guesses['Ea'] = 100000  # 100 kJ/mol
        guesses['A'] = 1e11

        # Add shape parameters if needed
        f_alpha_model = model_definition_args.get('f_alpha_model', '')
        if f_alpha_model == 'Fn':
            guesses['n'] = 1.0
        elif f_alpha_model == 'SB':
            guesses['m'] = 0.5
            guesses['n'] = 1.0
        elif f_alpha_model == 'SB_mnp':
            guesses['m'] = 0.5
            guesses['n'] = 1.0
            guesses['p'] = 0.0

    elif model_name == "A->B->C":
        guesses['Ea1'] = 100000
        guesses['A1'] = 1e11
        guesses['Ea2'] = 120000
        guesses['A2'] = 1e12

    elif model_name == "A+B->C":
        guesses['Ea'] = 100000
        guesses['A'] = 1e11
        guesses['initial_ratio_r'] = 1.0

    elif model_name == "parallel_competing":
        guesses['Ea1'] = 100000
        guesses['A1'] = 1e11
        guesses['Ea2'] = 120000
        guesses['A2'] = 1e12

    return guesses
