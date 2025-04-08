# core.py
import numpy as np
from scipy.optimize import curve_fit # Used by new SB Sum fit
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.optimize import minimize, OptimizeResult # Used by existing fit_kinetic_model
from concurrent.futures import TimeoutError as FuturesTimeoutError
import warnings
import copy
import time
import traceback  # For more detailed error info
import functools  # For partial used in bootstrap callback
from typing import List, Dict, Tuple, Optional, Callable, Union, Any # Added Any
import concurrent.futures

# --- Import from sibling modules ---
from . import models # Use relative import if in a package
from . import utils
# --- Updated Datatype Imports ---
from .datatypes import (
    KineticDataset,
    IsoResult,
    FitResult,
    ParameterBootstrapResult, # Renamed
    PredictionCurveResult,    # Renamed
    BootstrapPredictionResult,# New
    SingleValuePredictionResult, # New
    FullAnalysisResult,        # New
    FAlphaCallable,
    OdeSystemCallable
)
# --- End Updated Datatype Imports ---
from .models import get_model_info, F_ALPHA_MODELS # Assuming F_ALPHA_MODELS exists
from .utils import (get_temperature_interpolator, calculate_aic, calculate_bic, R_GAS, numerical_diff,
                    create_ea_interpolator, bootstrap_prediction) # Added new utils imports
from .isoconversional import run_friedman, run_kas, run_ofw # Keep existing iso imports

# Define constants
R_GAS = 8.314462 # J/(mol*K) - Defined only once

# ==============================================================================
# <<< Existing Core Functionality (Fitting with minimize, Parameter Bootstrap) >>>
# ==============================================================================

# --- Helper to prepare the full ODE parameter dictionary ---
# (Keep existing _prepare_full_params_for_ode as it's used by fit_kinetic_model)
def _prepare_full_params_for_ode(
    params_template: Dict,
    current_params_logA: Dict
    ) -> Dict:
    """Merges template and current logA params, converting logA->A."""
    full_params_ode = copy.deepcopy(params_template)
    param_mapping = full_params_ode.pop('_param_mapping', {})

    for name_logA, value_logA in current_params_logA.items():
        original_name = name_logA
        value_to_use = value_logA
        # More robust check for A parameters (A, A1, A2, etc.)
        is_logA_param = name_logA.startswith("logA") and (name_logA[3:].startswith("A"))

        if is_logA_param:
            original_name = name_logA[3:] # Get the 'A' name (e.g., A1)
            # Safely calculate exp, handle potential overflow/invalid values
            if not np.isfinite(value_logA):
                warnings.warn(f"Invalid non-finite value for {name_logA}. Using default large A.")
                value_to_use = 1e30 # Assign a large value or handle as error
            else:
                try:
                    value_to_use = np.exp(value_logA)
                except OverflowError:
                    warnings.warn(f"Overflow converting logA to A for {original_name}. Using large value.")
                    value_to_use = 1e300 # Assign large value on overflow

        # Place the value (either original or converted A) into the dictionary
        if original_name in param_mapping:
            target_dict_name, target_key = param_mapping[original_name]
            if target_dict_name in full_params_ode and isinstance(full_params_ode[target_dict_name], dict):
                 full_params_ode[target_dict_name][target_key] = value_to_use
            else:
                 warnings.warn(f"Prepare ODE Params: Param mapping target '{target_dict_name}' error for {original_name}.")
        # Check if it's a key directly in the template structure (e.g., 'initial_ratio_r', 'f1_func')
        elif original_name in full_params_ode:
             full_params_ode[original_name] = value_to_use
        # Assume it's a base kinetic param (Ea) not otherwise mapped or in template structure
        else:
             full_params_ode[original_name] = value_to_use

    return full_params_ode

# --- Simulation Function with Fallback ---
# (Keep existing _simulate_single_dataset as it's used by fit_kinetic_model)
def _simulate_single_dataset(
    t_eval: np.ndarray,
    temp_func: Callable,
    ode_system: OdeSystemCallable,
    initial_state: np.ndarray,
    params_template: Dict, # Template containing functions, param dict structures
    current_params_logA: Dict, # Current kinetic params (Ea, logA, etc.)
    solver_options: Dict = {}
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulates conversion with solver fallback. Merges template and logA params.
    Returns (time_array, alpha_array)
    """
    primary_solver = solver_options.get('primary_solver', 'RK45')
    fallback_solver = solver_options.get('fallback_solver', 'LSODA')
    common_solver_kwargs = {
        'rtol': solver_options.get('rtol', 1e-6),
        'atol': solver_options.get('atol', 1e-9),
    }

    try:
        full_params_ode = _prepare_full_params_for_ode(params_template, current_params_logA)
    except Exception as e_prep:
        warnings.warn(f"Simulate: Error preparing ODE params: {e_prep}")
        alpha_nan = np.full((len(t_eval),), np.nan); return t_eval, alpha_nan

    sort_indices = np.argsort(t_eval); t_eval_sorted = t_eval[sort_indices]; t_start, t_end = t_eval_sorted[0], t_eval_sorted[-1]
    if t_start >= t_end: alpha_out = np.full_like(t_eval, initial_state[0]); unsort_indices = np.argsort(sort_indices); return t_eval, alpha_out[unsort_indices]

    sol = None; success = False
    for solver in [primary_solver, fallback_solver]:
        if success: break
        if solver is None: continue
        try:
            current_solver_kwargs = common_solver_kwargs.copy()
            sol = solve_ivp(
                fun=ode_system,
                t_span=(t_start, t_end),
                y0=initial_state,
                t_eval=t_eval_sorted,
                args=(temp_func, full_params_ode),
                method=solver,
                **current_solver_kwargs
            )
            success = sol.success
            if not success and solver == primary_solver: warnings.warn(f"Primary solver '{solver}' failed: {sol.message}. Trying fallback.")
            elif not success and solver == fallback_solver: warnings.warn(f"Fallback solver '{solver}' failed: {sol.message}")
            elif success and solver == fallback_solver: warnings.warn(f"Fallback solver '{solver}' succeeded.")
        except Exception as e_solve: warnings.warn(f"Solver '{solver}' failed execution: {e_solve}"); success = False
        if success: break

    if success and sol is not None:
        state_sim_sorted = sol.y; alpha_sim_sorted = np.zeros_like(sol.t); ode_func_name = ode_system.__name__
        if ode_func_name == 'ode_system_single_step': alpha_sim_sorted = state_sim_sorted[0, :]
        elif ode_func_name == 'ode_system_A_plus_B_C': alpha_sim_sorted = state_sim_sorted[0, :]
        elif ode_func_name == 'ode_system_A_B_C': alpha_sim_sorted = 1.0 - state_sim_sorted[0, :] - state_sim_sorted[1, :]
        else: alpha_sim_sorted = state_sim_sorted[0, :]
        alpha_sim_sorted = np.clip(alpha_sim_sorted, 0.0, 1.0); unsort_indices = np.argsort(sort_indices); alpha_sim_unsorted = alpha_sim_sorted[unsort_indices]
        return t_eval, alpha_sim_unsorted
    else:
        warnings.warn(f"Both solvers failed or simulation error occurred."); alpha_nan = np.full((len(t_eval),), np.nan); return t_eval, alpha_nan

# --- Objective Function for Fitting ---
# (Keep existing _objective_function as it's used by fit_kinetic_model)
def _objective_function(
    params_array_logA: np.ndarray,
    param_names_logA: List[str],
    datasets: List[KineticDataset],
    model_name: str,
    model_definition_args: Dict,
    solver_options: Dict,
    callback_func: Optional[Callable] = None,
    iteration_counter: List[int] = [0]
) -> float:
    total_weighted_rss = 0.0
    n_total_datapoints = 0
    current_params_logA = dict(zip(param_names_logA, params_array_logA))

    weight_transition = 10.0
    weight_baseline_plateau = 1.0
    alpha_lower = 0.05
    alpha_upper = 0.95

    if callback_func:
        try:
            current_params_A = {};
            for name_logA, val_logA in current_params_logA.items():
                 is_logA = name_logA.startswith("logA") and (name_logA[3:].startswith("A"))
                 original_name = name_logA[3:] if is_logA else name_logA
                 param_val_A = np.exp(val_logA) if is_logA else val_logA
                 current_params_A[original_name] = param_val_A
            callback_func(iteration_counter[0], current_params_A)
        except Exception as e_cb: warnings.warn(f"Objective callback failed: {e_cb}")
    iteration_counter[0] += 1

    try:
        ode_func, _, initial_state_dim, params_template = get_model_info(model_name, **model_definition_args)
    except ValueError as e:
        warnings.warn(f"Objective func: Model info error: {e}")
        return np.inf

    if model_name == "A->B->C":
        initial_state = np.array([1.0, 0.0])
    elif initial_state_dim == 1:
        initial_state = np.array([0.0])
    else:
        initial_state = np.zeros(initial_state_dim)

    for ds in datasets:
        if len(ds.time) < 2:
            continue
        try:
            temp_func = get_temperature_interpolator(ds.time, ds.temperature)
            t_sim_eval = ds.time
            t_sim_out, alpha_sim = _simulate_single_dataset(
                t_eval=t_sim_eval,
                temp_func=temp_func,
                ode_system=ode_func,
                initial_state=initial_state,
                params_template=params_template,
                current_params_logA=current_params_logA,
                solver_options=solver_options
            )
            if len(alpha_sim) != len(ds.conversion):
                warnings.warn(f"Sim length mismatch.")
                continue

            residuals = ds.conversion - alpha_sim
            valid_mask = np.isfinite(residuals) & np.isfinite(alpha_sim)

            weights = np.full_like(residuals, weight_baseline_plateau)
            transition_mask = (ds.conversion > alpha_lower) & (ds.conversion < alpha_upper)
            weights[transition_mask] = weight_transition
            weights = weights[valid_mask]
            valid_residuals = residuals[valid_mask]

            if len(valid_residuals) == 0 and len(residuals) > 0:
                rss = np.inf
            elif len(valid_residuals) > 0:
                weighted_sq_residuals = weights * (valid_residuals**2)
                rss = np.sum(weighted_sq_residuals)
            else:
                rss = 0.0

            if not np.isfinite(rss):
                rss = np.inf
            total_weighted_rss += rss
            n_total_datapoints += len(valid_residuals)

        except Exception as e:
            warnings.warn(f"Objective func: Sim error: {e}.")
            total_weighted_rss = np.inf
            break

    if not np.isfinite(total_weighted_rss):
        total_weighted_rss = 1e30

    return total_weighted_rss

# --- Objective Function for Fitting (Rate-Based) ---
# (Keep existing _objective_function_rate as it's used by fit_kinetic_model fallback)
def _objective_function_rate(
    params_array_logA: np.ndarray,
    param_names_logA: List[str],
    datasets: List[KineticDataset],
    exp_rates: List[np.ndarray],
    exp_times: List[np.ndarray],
    model_name: str,
    model_definition_args: Dict,
    solver_options: Dict,
    callback_func: Optional[Callable] = None,
    iteration_counter: List[int] = [0]
) -> float:
    total_rate_rss = 0.0
    current_params_logA = dict(zip(param_names_logA, params_array_logA))

    if callback_func:
        try:
            current_params_A = {
                name[3:] if name.startswith("logA") else name: np.exp(val) if name.startswith("logA") else val
                for name, val in current_params_logA.items()
            }
            callback_func(iteration_counter[0], current_params_A)
        except Exception as e_cb:
            warnings.warn(f"Objective callback failed: {e_cb}")
    iteration_counter[0] += 1

    try:
        ode_func, _, initial_state_dim, params_template = get_model_info(model_name, **model_definition_args)
    except ValueError as e:
        warnings.warn(f"Objective func (rate): Model info error: {e}")
        return np.inf

    if model_name == "A->B->C":
        initial_state = np.array([1.0, 0.0])
    elif initial_state_dim == 1:
        initial_state = np.array([0.0])
    else:
        initial_state = np.zeros(initial_state_dim)

    for i, ds in enumerate(datasets):
        if len(ds.time) < 2 or i >= len(exp_rates) or i >= len(exp_times):
            continue
        try:
            temp_func = get_temperature_interpolator(ds.time, ds.temperature)
            t_sim_eval = exp_times[i]
            t_sim_out, alpha_sim = _simulate_single_dataset(
                t_eval=t_sim_eval,
                temp_func=temp_func,
                ode_system=ode_func,
                initial_state=initial_state,
                params_template=params_template,
                current_params_logA=current_params_logA,
                solver_options=solver_options
            )
            if len(alpha_sim) != len(t_sim_eval):
                warnings.warn(f"Rate objective: Sim length mismatch.")
                continue

            rate_sim = numerical_diff(t_sim_out, alpha_sim)
            rate_exp = exp_rates[i]

            if len(rate_sim) != len(rate_exp):
                warnings.warn(f"Rate objective: Rate length mismatch after diff (Exp: {len(rate_exp)}, Sim: {len(rate_sim)}). Skipping dataset {i}.")
                continue

            rate_residuals = rate_exp - rate_sim
            valid_mask = np.isfinite(rate_residuals) & np.isfinite(rate_exp) & np.isfinite(rate_sim)

            rss = np.sum(rate_residuals[valid_mask]**2) if np.sum(valid_mask) > 0 else np.inf
            total_rate_rss += rss if np.isfinite(rss) else np.inf

        except Exception as e:
            warnings.warn(f"Objective func (rate): Sim/Diff error: {e}.")
            total_rate_rss = np.inf
            break

    return total_rate_rss if np.isfinite(total_rate_rss) else 1e30

# --- Helper Function to Calculate Stats ---
# (Keep existing _calculate_conversion_stats)
def _calculate_conversion_stats(
    datasets: List[KineticDataset],
    params_logA: Dict,
    model_name: str,
    model_definition_args: Dict,
    solver_options: Dict
) -> Tuple[float, int, float, float, float]:
    rss = np.inf
    n_pts = 0
    all_exp_conv = []
    all_sim_conv = []
    n_params = len(params_logA)

    try:
        ode_func, _, initial_state_dim, params_template = get_model_info(model_name, **model_definition_args)
        if model_name == "A->B->C":
            initial_state = np.array([1.0, 0.0])
        elif initial_state_dim == 1:
            initial_state = np.array([0.0])
        else:
            initial_state = np.zeros(initial_state_dim)
        current_rss_calc = 0.0
        for ds in datasets:
            if len(ds.time) < 2:
                continue
            temp_func = get_temperature_interpolator(ds.time, ds.temperature)
            t_sim_out, alpha_sim = _simulate_single_dataset(
                t_eval=ds.time,
                temp_func=temp_func,
                ode_system=ode_func,
                initial_state=initial_state,
                params_template=params_template,
                current_params_logA=params_logA,
                solver_options=solver_options
            )
            if len(alpha_sim) == len(ds.conversion):
                residuals = ds.conversion - alpha_sim
                valid_mask = np.isfinite(residuals) & np.isfinite(alpha_sim) & np.isfinite(ds.conversion)
                if np.sum(valid_mask) > 0:
                    current_rss_calc += np.sum(residuals[valid_mask]**2)
                    n_pts += np.sum(valid_mask)
                    all_exp_conv.extend(ds.conversion[valid_mask])
                    all_sim_conv.extend(alpha_sim[valid_mask])
        if n_pts > 0:
            rss = current_rss_calc
    except Exception as e_sim:
        warnings.warn(f"Failed to simulate conversion curve for stats calculation: {e_sim}")

    r_squared = np.nan
    aic = np.nan
    bic = np.nan
    if n_pts > n_params and np.isfinite(rss) and len(all_exp_conv) == n_pts:
        mean_y = np.mean(all_exp_conv)
        total_ss = np.sum((np.array(all_exp_conv) - mean_y)**2)
        r_squared = 1.0 - (rss / total_ss) if total_ss > 1e-12 else (1.0 if rss < 1e-12 else 0.0)
        aic = calculate_aic(rss, n_params, n_pts)
        bic = calculate_bic(rss, n_params, n_pts)

    return rss, n_pts, r_squared, aic, bic

# --- Main Fitting Function ---
# (Keep existing fit_kinetic_model)
def fit_kinetic_model(
    datasets: List[KineticDataset],
    model_name: str,
    model_definition_args: Dict,
    initial_guesses: Dict[str, float],
    parameter_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    solver_options: Dict = {},
    optimizer_options: Dict = {},
    callback: Optional[Callable[[int, Dict], None]] = None,
    ) -> FitResult:
    default_method = 'L-BFGS-B'; opt_method = optimizer_options.get('method', default_method); default_opt_options = {'disp': False, 'ftol': 1e-9, 'gtol': 1e-7}; opt_options = optimizer_options.get('options', default_opt_options)
    if not isinstance(opt_options, dict): opt_options = default_opt_options

    param_names_logA = []; original_param_names_map = {}
    try:
        _, original_param_names, _, _ = get_model_info(model_name, **model_definition_args)
        for name in original_param_names: is_A_param = name.startswith("A") and (name[1:].isdigit() or len(name)==1); logA_name = "log" + name if is_A_param else name; param_names_logA.append(logA_name); original_param_names_map[logA_name] = name
    except ValueError as e: return FitResult(model_name=model_name, parameters={}, success=False, message=f"Model setup error: {e}", rss=np.inf, n_datapoints=0, n_parameters=0, r_squared=np.nan, model_definition_args=model_definition_args)

    initial_guesses_logA = {}; parameter_bounds_logA = {} if parameter_bounds is not None else None; final_bounds_logA = {}
    if not all(p in initial_guesses for p in original_param_names): missing = [p for p in original_param_names if p not in initial_guesses]; return FitResult(model_name=model_name, model_definition_args=model_definition_args, parameters={}, success=False, message=f"Missing initial guesses for: {missing}", rss=np.inf, n_datapoints=0, n_parameters=len(param_names_logA), r_squared=np.nan)
    try:
        for p_logA_name in param_names_logA: original_name = original_param_names_map[p_logA_name]; is_logA_param = p_logA_name.startswith("logA"); guess_val = initial_guesses[original_name]; initial_guesses_logA[p_logA_name] = np.log(guess_val) if is_logA_param else guess_val;
        if is_logA_param and guess_val <= 0: raise ValueError(f"Initial guess for {original_name} must be positive.")
    except (ValueError, TypeError, KeyError) as e_conv: return FitResult(model_name=model_name, model_definition_args=model_definition_args, parameters={}, success=False, message=f"Error converting initial guess: {e_conv}", rss=np.inf, n_datapoints=0, n_parameters=len(param_names_logA), r_squared=np.nan)

    if parameter_bounds_logA is not None:
        for p_logA_name in param_names_logA:
            original_name = original_param_names_map[p_logA_name]
            is_logA_param = p_logA_name.startswith("logA")
            user_bound = parameter_bounds.get(original_name)
            if user_bound is not None:
                min_val, max_val = user_bound
                if is_logA_param:
                    min_log = np.log(min_val) if min_val is not None and min_val > 0 else -np.inf
                    max_log = np.log(max_val) if max_val is not None and max_val > 0 else np.inf
                    parameter_bounds_logA[p_logA_name] = (min_log, max_log)
                else:
                    parameter_bounds_logA[p_logA_name] = (min_val, max_val)

    default_bounds_logA = {};
    for p_name in param_names_logA:
        if p_name.startswith("Ea"): default_bounds_logA[p_name] = (1e3, 600e3)
        elif p_name.startswith("logA"): default_bounds_logA[p_name] = (np.log(1e-2), np.log(1e25))
        elif p_name.endswith("n") or p_name.endswith("m") or p_name.startswith("p1_") or p_name.startswith("p2_"): default_bounds_logA[p_name] = (0, 8)
        elif p_name == "initial_ratio_r": default_bounds_logA[p_name] = (1e-3, 1e3)
        else: default_bounds_logA[p_name] = (-np.inf, np.inf)
    final_bounds_logA = default_bounds_logA.copy();
    if parameter_bounds_logA: final_bounds_logA.update(parameter_bounds_logA)
    initial_params_array_logA = np.array([initial_guesses_logA[p] for p in param_names_logA]); bounds_list_logA = None
    methods_supporting_bounds = ['L-BFGS-B', 'TNC', 'SLSQP', 'trust-constr']
    if opt_method in methods_supporting_bounds and final_bounds_logA:
        try: bounds_list_logA = [(final_bounds_logA.get(p, (-np.inf, np.inf))[0] if np.isfinite(final_bounds_logA.get(p, (-np.inf, np.inf))[0]) else None, final_bounds_logA.get(p, (-np.inf, np.inf))[1] if np.isfinite(final_bounds_logA.get(p, (-np.inf, np.inf))[1]) else None) for p in param_names_logA]
        except Exception as e: return FitResult(model_name=model_name, model_definition_args=model_definition_args, parameters={}, success=False, message=f"Invalid bounds format: {e}", rss=np.inf, n_datapoints=0, n_parameters=len(param_names_logA), r_squared=np.nan)
    elif parameter_bounds and opt_method not in methods_supporting_bounds: warnings.warn(f"Optimizer {opt_method} ignores bounds.")

    print(f"--- Fitting '{model_name}': Attempting optimization on CONVERSION residuals (weighted) ---")
    iteration_counter_conv = [0]
    opt_result_conv = None
    success_conv = False
    try:
        minimize_args_conv = (param_names_logA, datasets, model_name, model_definition_args, solver_options, callback, iteration_counter_conv)
        opt_result_conv = minimize(fun=_objective_function, x0=initial_params_array_logA, args=minimize_args_conv, method=opt_method, bounds=bounds_list_logA, options=opt_options)
        success_conv = opt_result_conv.success
    except Exception as e_conv:
        warnings.warn(f"Conversion-based optimization failed with exception: {e_conv}")
        success_conv = False

    opt_result = opt_result_conv
    used_rate_fallback = False
    if not success_conv:
        warnings.warn("Conversion-based fit failed. Falling back to RATE-based optimization.")
        used_rate_fallback = True
        iteration_counter_rate = [0]
        exp_rates = []; exp_times_sec = []; valid_datasets_indices = []
        diff_options = {'window_length': 5, 'polyorder': 2}
        for i, ds in enumerate(datasets):
            if len(ds.time) < diff_options['window_length']: warnings.warn(f"Dataset {i} too short for rate calc."); continue
            time_sec = ds.time
            rate = numerical_diff(time_sec, ds.conversion, **diff_options);
            if rate is not None and len(rate) == len(time_sec):
                exp_rates.append(rate); exp_times_sec.append(time_sec); valid_datasets_indices.append(i)
            else:
                 warnings.warn(f"Rate calculation failed or length mismatch for dataset {i}.")

        datasets_for_rate_fit = [datasets[i] for i in valid_datasets_indices]

        if not datasets_for_rate_fit:
             return FitResult(model_name=model_name, model_definition_args=model_definition_args, parameters={}, success=False, message="Conversion fit failed & no datasets valid for rate fit fallback.", rss=np.inf, n_datapoints=0, n_parameters=len(param_names_logA), r_squared=np.nan, used_rate_fallback=True)

        print(f"--- Fitting '{model_name}': Optimizing on RATE residuals ---")
        opt_result_rate = None
        try:
            minimize_args_rate = (param_names_logA, datasets_for_rate_fit, exp_rates, exp_times_sec, model_name, model_definition_args, solver_options, callback, iteration_counter_rate)
            opt_result_rate = minimize(fun=_objective_function_rate, x0=initial_params_array_logA, args=minimize_args_rate, method=opt_method, bounds=bounds_list_logA, options=opt_options)
            opt_result = opt_result_rate
        except Exception as e_rate:
            fail_message = f"Conversion fit failed ({opt_result_conv.message if opt_result_conv else 'Exception'}). Rate fit fallback also failed ({e_rate})."
            return FitResult(model_name=model_name, model_definition_args=model_definition_args, parameters={}, success=False, message=fail_message, rss=np.inf, n_datapoints=0, n_parameters=len(param_names_logA), r_squared=np.nan, used_rate_fallback=True)

    if opt_result is None or not opt_result.success:
        fail_message = f"Optimization failed. Initial attempt: {opt_result_conv.message if opt_result_conv else 'Exception'}. "
        if used_rate_fallback: fail_message += f"Rate fallback attempt: {opt_result.message if opt_result else 'Exception'}."
        else: fail_message += "Rate fallback not attempted."
        return FitResult(model_name=model_name, model_definition_args=model_definition_args, parameters={}, success=False, message=fail_message, rss=np.inf, n_datapoints=0, n_parameters=len(param_names_logA), r_squared=np.nan, used_rate_fallback=used_rate_fallback)

    fitted_params_logA = dict(zip(param_names_logA, opt_result.x));
    n_params = len(param_names_logA)
    fitted_params_final = {};
    for name_logA, value_logA in fitted_params_logA.items(): is_logA_param = name_logA.startswith("logA") and (name_logA[3:].startswith("A")); fitted_params_final[name_logA[3:] if is_logA_param else name_logA] = np.exp(value_logA) if is_logA_param else value_logA

    final_conversion_rss, n_total_datapoints_final, r_squared, aic, bic = _calculate_conversion_stats(
        datasets, fitted_params_logA, model_name, model_definition_args, solver_options
    )

    param_std_err_final = None;
    covariance_matrix = None
    if opt_result.success and hasattr(opt_result, 'hess_inv'):
        hess_inv = None;
        if isinstance(opt_result.hess_inv, np.ndarray): hess_inv = opt_result.hess_inv; covariance_matrix = hess_inv
        elif hasattr(opt_result.hess_inv, 'todense'):
            try: hess_inv = opt_result.hess_inv.todense(); covariance_matrix = hess_inv
            except Exception: pass
        if hess_inv is not None:
            try:
                diag_hess_inv = np.diag(hess_inv)
                if np.all(diag_hess_inv > 0) and n_total_datapoints_final > n_params and np.isfinite(final_conversion_rss):
                    sigma_sq_est = final_conversion_rss / (n_total_datapoints_final - n_params);
                    sigma_sq_est = max(sigma_sq_est, 0)
                    param_variances_logA = diag_hess_inv * sigma_sq_est
                    param_variances_logA = np.maximum(param_variances_logA, 0)
                    param_std_err_logA_arr = np.sqrt(param_variances_logA);
                    param_std_err_logA = dict(zip(param_names_logA, param_std_err_logA_arr))
                    param_std_err_final = {}
                    for name_logA, std_err_logA in param_std_err_logA.items():
                        is_logA_param = name_logA.startswith("logA") and (name_logA[3:].startswith("A"))
                        if is_logA_param: original_A_key = name_logA[3:]; A_value = fitted_params_final[original_A_key]; param_std_err_final[original_A_key] = std_err_logA * A_value
                        else: param_std_err_final[name_logA] = std_err_logA
                else:
                     warnings.warn("Could not estimate std errors: Hessian diagonal not positive or insufficient data.")
            except Exception as e: warnings.warn(f"Could not estimate/propagate std errors: {e}")

    for p_name, p_val in fitted_params_final.items():
        if p_name.startswith("Ea"):
            if p_val < 5e3: warnings.warn(f"Fitted {p_name} ({p_val/1000:.1f} kJ/mol) is very low.")
            if p_val > 400e3: warnings.warn(f"Fitted {p_name} ({p_val/1000:.1f} kJ/mol) is very high.")
        elif p_name.startswith("A"):
             if p_val < 1e-1: warnings.warn(f"Fitted {p_name} ({p_val:.1e} 1/s) is very low.")
             if p_val > 1e20: warnings.warn(f"Fitted {p_name} ({p_val:.1e} 1/s) is very high.")
    initial_r = None;
    if model_name == "A+B->C":
        fixed_r = model_definition_args.get('bimol_params', {}).get('initial_ratio_r')
        if fixed_r is not None and 'initial_ratio_r' not in param_names_logA: initial_r = fixed_r
        elif 'initial_ratio_r' in fitted_params_logA: initial_r = fitted_params_logA['initial_ratio_r']

    return FitResult(
        model_name=model_name, model_definition_args=model_definition_args,
        parameters=fitted_params_final, success=opt_result.success, message=opt_result.message,
        rss=final_conversion_rss, n_datapoints=n_total_datapoints_final, n_parameters=n_params,
        param_std_err=param_std_err_final,
        covariance_matrix=covariance_matrix,
        aic=aic, bic=bic, r_squared=r_squared,
        initial_ratio_r=initial_r,
        used_rate_fallback=used_rate_fallback
    )

# --- Bootstrapping Worker Function ---
# (Keep existing _fit_on_resampled_data)
def _fit_on_resampled_data(
    datasets: List[KineticDataset], model_name: str, model_definition_args: Dict,
    best_fit_params_logA: Dict[str, float], parameter_bounds_logA: Optional[Dict[str, Tuple[float, float]]],
    solver_options: Dict, optimizer_options: Dict, iteration_index: int,
    end_callback_func: Optional[Callable[[int, str, Optional[Dict]], None]]
) -> Optional[Dict]:
    if end_callback_func:
        try:
            end_callback_func(iteration_index, "started", None)
        except Exception as e_cb:
            warnings.warn(f"Bootstrap start callback failed: {e_cb}")

    resampled_datasets = []
    param_names_logA = list(best_fit_params_logA.keys())
    try:
        ode_func, _, initial_state_dim, params_template = get_model_info(model_name, **model_definition_args)
    except ValueError as e:
        warnings.warn(f"Bootstrap {iteration_index}: Model info error: {e}")
        if end_callback_func: end_callback_func(iteration_index, "failed", {'error': f"Model info error: {e}"})
        return None

    if model_name == "A->B->C":
        initial_state = np.array([1.0, 0.0])
    elif initial_state_dim == 1:
        initial_state = np.array([0.0])
    else:
        initial_state = np.zeros(initial_state_dim)

    all_residuals = []
    original_indices_map = []
    original_alphas = []

    for i, ds in enumerate(datasets):
        if len(ds.time) < 2:
            continue
        try:
            temp_func = get_temperature_interpolator(ds.time, ds.temperature)
            t_sim, alpha_sim = _simulate_single_dataset(
                t_eval=ds.time, temp_func=temp_func, ode_system=ode_func,
                initial_state=initial_state, params_template=params_template,
                current_params_logA=best_fit_params_logA, solver_options=solver_options
            )
            if len(alpha_sim) == len(ds.conversion):
                residuals = ds.conversion - alpha_sim
                valid_mask = np.isfinite(residuals) & np.isfinite(alpha_sim) & np.isfinite(ds.conversion)
                all_residuals.extend(residuals[valid_mask])
                original_indices_map.extend([(i, k) for k, valid in enumerate(valid_mask) if valid])
                original_alphas.extend(ds.conversion[valid_mask])
            else:
                warnings.warn(f"Bootstrap {iteration_index}: Res calc sim length mismatch ds {i}.")
        except Exception as e_sim:
            warnings.warn(f"Bootstrap {iteration_index}: Res calc sim failed ds {i}: {e_sim}.")

    if not all_residuals:
        warnings.warn(f"Bootstrap {iteration_index}: No valid residuals.")
        if end_callback_func: end_callback_func(iteration_index, "failed", {'error': "No valid residuals"})
        return None

    all_residuals = np.array(all_residuals)
    centered_residuals = all_residuals - np.mean(all_residuals)
    original_alphas = np.array(original_alphas)
    n_residuals = len(centered_residuals)

    weights = np.ones(n_residuals)
    transition_weight = 10.0
    alpha_lower_bound = 0.05
    alpha_upper_bound = 0.95
    transition_indices = np.where((original_alphas > alpha_lower_bound) & (original_alphas < alpha_upper_bound))[0]
    weights[transition_indices] = transition_weight
    sum_weights = np.sum(weights)
    if sum_weights <= 0:
         weights = np.ones(n_residuals) / n_residuals
    else:
         weights /= sum_weights

    try:
        resampled_residual_indices = np.random.choice(n_residuals, size=n_residuals, replace=True, p=weights)
    except ValueError as e_choice:
        warnings.warn(f"Bootstrap {iteration_index}: Weighted sampling failed ({e_choice}). Using uniform.")
        resampled_residual_indices = np.random.choice(n_residuals, size=n_residuals, replace=True)

    for i, ds in enumerate(datasets):
        if len(ds.time) < 2:
            continue
        try:
            temp_func = get_temperature_interpolator(ds.time, ds.temperature)
            t_sim_orig, alpha_sim_orig = _simulate_single_dataset(
                t_eval=ds.time, temp_func=temp_func, ode_system=ode_func,
                initial_state=initial_state, params_template=params_template,
                current_params_logA=best_fit_params_logA, solver_options=solver_options
            )
            if len(alpha_sim_orig) == len(ds.conversion):
                synthetic_alpha = np.copy(alpha_sim_orig)
                current_ds_point_idx = 0
                for k in range(len(ds.time)):
                    original_map_idx = -1
                    for map_i, (ds_idx, pt_idx) in enumerate(original_indices_map):
                        if ds_idx == i and pt_idx == k:
                            original_map_idx = map_i
                            break

                    if original_map_idx != -1:
                        chosen_residual_idx = resampled_residual_indices[original_map_idx]
                        synthetic_alpha[k] += centered_residuals[chosen_residual_idx]

                synthetic_alpha = np.clip(synthetic_alpha, 0.0, 1.0)
                resampled_datasets.append(KineticDataset(
                    time=ds.time, temperature=ds.temperature,
                    conversion=synthetic_alpha, heating_rate=ds.heating_rate,
                    metadata={'original_run_index': i}
                ))
            else:
                 warnings.warn(f"Bootstrap {iteration_index}: Synth data sim length mismatch ds {i}.")
        except Exception as e_resample:
            warnings.warn(f"Bootstrap {iteration_index}: Error creating resampled ds {i}: {e_resample}")

    if not resampled_datasets:
        warnings.warn(f"Bootstrap {iteration_index}: Failed to create resampled datasets.")
        if end_callback_func: end_callback_func(iteration_index, "failed", {'error': "Failed to create resampled datasets"})
        return None

    perturbed_guesses_logA = {}
    noise_factor = 0.05
    abs_noise = {'Ea': 500, 'logA': 0.2}
    for name_logA, val_logA in best_fit_params_logA.items():
        noise_level = abs(val_logA) * noise_factor
        if name_logA.startswith("Ea"):
            noise_level = max(noise_level, abs_noise['Ea'])
        elif name_logA.startswith("logA"):
            noise_level = max(noise_level, abs_noise['logA'])
        perturbed_guesses_logA[name_logA] = val_logA + np.random.normal(0, noise_level)

    output_dict = None
    opt_result_boot = None
    try:
        bounds_list_logA_boot = None
        methods_supporting_bounds = ['L-BFGS-B', 'TNC', 'SLSQP', 'trust-constr']
        opt_method_boot = optimizer_options.get('method', 'L-BFGS-B')
        if opt_method_boot in methods_supporting_bounds and parameter_bounds_logA:
            try: bounds_list_logA_boot = [(parameter_bounds_logA.get(p, (-np.inf, np.inf))[0] if np.isfinite(parameter_bounds_logA.get(p, (-np.inf, np.inf))[0]) else None,
                                           parameter_bounds_logA.get(p, (-np.inf, np.inf))[1] if np.isfinite(parameter_bounds_logA.get(p, (-np.inf, np.inf))[1]) else None)
                                          for p in param_names_logA]
            except Exception as e_bounds: warnings.warn(f"Bootstrap {iteration_index}: Error formatting bounds: {e_bounds}"); bounds_list_logA_boot = None

        opt_result_boot = minimize(
            fun=_objective_function,
            x0=np.array(list(perturbed_guesses_logA.values())),
            args=(param_names_logA, resampled_datasets, model_name, model_definition_args, solver_options, None, [0]),
            method=opt_method_boot,
            bounds=bounds_list_logA_boot,
            options=optimizer_options.get('options', {})
        )
        if opt_result_boot.success:
            fitted_params_logA_replicate = dict(zip(param_names_logA, opt_result_boot.x))
            stats_dict = calculate_stats_for_replicate(
                datasets=resampled_datasets,
                params_logA=fitted_params_logA_replicate,
                model_name=model_name,
                model_definition_args=model_definition_args,
                solver_options=solver_options
            )
            output_dict = {'params_logA': fitted_params_logA_replicate, 'stats': stats_dict}
            if end_callback_func: end_callback_func(iteration_index, "success", output_dict)
        else:
             warnings.warn(f"Bootstrap replicate {iteration_index} fit failed: {opt_result_boot.message}")
             if end_callback_func: end_callback_func(iteration_index, "failed", {'error': f"Fit failed: {opt_result_boot.message}"})

    except Exception as e:
        warnings.warn(f"Bootstrap replicate {iteration_index} fit exception: {e}")
        if end_callback_func: end_callback_func(iteration_index, "failed", {'error': f"Exception: {e}"})

    return output_dict

# --- Parameter Bootstrap Main Function ---
# (Keep existing run_bootstrap, but change return type)
def run_bootstrap(
    datasets: List[KineticDataset],
    fit_result: FitResult,
    optimizer_options: Dict,
    parameter_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    n_iterations: int = 100,
    confidence_level: float = 0.95,
    n_jobs: int = -1,
    solver_options: Dict = {},
    end_callback: Optional[Callable[[int, str, Optional[Dict]], None]] = None,
    timeout_per_replicate: Optional[float] = None,
    return_replicate_params: bool = False
    ) -> Optional[ParameterBootstrapResult]:
    if not fit_result.success: warnings.warn("Initial fit failed."); return None
    model_definition_args = getattr(fit_result, 'model_definition_args', None)
    if model_definition_args is None: warnings.warn("FitResult missing 'model_definition_args'. Cannot run bootstrap reliably."); return None
    model_name = fit_result.model_name; best_fit_params_A = fit_result.parameters
    best_fit_params_logA = {}; param_names_logA = []
    try:
        for name, val in best_fit_params_A.items(): is_A_param = name.startswith("A") and (name[1:].isdigit() or len(name)==1); logA_name = "log" + name if is_A_param else name; best_fit_params_logA[logA_name] = np.log(val) if is_A_param else val; param_names_logA.append(logA_name)
    except (ValueError, TypeError) as e_conv: warnings.warn(f"Cannot take log of parameter for bootstrap: {e_conv}."); return None

    parameter_bounds_logA = None
    if parameter_bounds:
        parameter_bounds_logA = {}
        for name_logA in param_names_logA:
            is_logA_param = name_logA.startswith("logA") and (name_logA[3:].startswith("A"))
            original_key = name_logA[3:] if is_logA_param else name_logA
            if original_key in parameter_bounds:
                 min_A, max_A = parameter_bounds[original_key]
                 if is_logA_param:
                      min_logA = np.log(min_A) if min_A is not None and min_A > 0 else -np.inf
                      max_logA = np.log(max_A) if max_A is not None and max_A > 0 else np.inf
                      parameter_bounds_logA[name_logA] = (min_logA, max_logA)
                 else:
                      parameter_bounds_logA[name_logA] = (min_A, max_A)
            else:
                 parameter_bounds_logA[name_logA] = (-np.inf, np.inf)

    max_workers = n_jobs if n_jobs is not None and n_jobs > 0 else None
    print(f"Starting {n_iterations} bootstrap fits (logA scale) using concurrent.futures (max_workers={max_workers})...")
    results_data = [None] * n_iterations; futures_list = []; n_submitted = 0
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            for i in range(n_iterations):
                future = executor.submit(_fit_on_resampled_data, datasets, model_name, model_definition_args, best_fit_params_logA, parameter_bounds_logA, solver_options, optimizer_options, i, end_callback)
                futures_list.append(future); n_submitted += 1
            print(f"\nSubmitted {n_submitted} tasks. Collecting results...")
            for i, future in enumerate(futures_list):
                try: result = future.result(timeout=timeout_per_replicate); results_data[i] = result
                except FuturesTimeoutError: warnings.warn(f"Bootstrap replicate {i+1} timed out after {timeout_per_replicate}s."); results_data[i] = None
                except Exception as e: warnings.warn(f"Bootstrap replicate {i+1} failed with exception: {type(e).__name__}: {e}"); results_data[i] = None
    except Exception as e_pool: warnings.warn(f"Error during bootstrap pool execution: {type(e_pool).__name__}: {e_pool}"); results_data = [results_data[i] if i < len(results_data) else None for i in range(n_iterations)]

    successful_params_logA = []; successful_replicate_stats = []; n_failed_or_timed_out = 0
    print("\nProcessing received results...")
    for i, p_data in enumerate(results_data):
        if p_data is None or not isinstance(p_data, dict) or 'params_logA' not in p_data or 'stats' not in p_data: n_failed_or_timed_out += 1
        else:
            if all(key in p_data['params_logA'] for key in param_names_logA):
                successful_params_logA.append(p_data['params_logA']);
                successful_replicate_stats.append(p_data['stats'])
            else: warnings.warn(f"Replicate {i+1} returned unexpected dict keys."); n_failed_or_timed_out += 1
    n_success = len(successful_params_logA)
    print(f"\nBootstrap finished processing. {n_success}/{n_iterations} replicates successful ({n_failed_or_timed_out} failed/timed out).")
    if n_success == 0: return None
    elif n_success < n_iterations * 0.75: warnings.warn(f"Low success rate ({n_success}/{n_iterations}). Results may be less reliable.")

    median_params_logA = {}; median_params_A = {}; median_stats = None
    if n_success > 0:
        param_distributions_logA_temp = {p_name: np.array([params[p_name] for params in successful_params_logA]) for p_name in param_names_logA}
        for p_name in param_names_logA: median_params_logA[p_name] = np.median(param_distributions_logA_temp[p_name])
        try:
            med_rss, med_n_pts, med_r2, med_aic, med_bic = _calculate_conversion_stats(datasets, median_params_logA, model_name, model_definition_args, solver_options)
            if np.isfinite(med_bic): median_stats = {'rss': med_rss, 'r_squared': med_r2, 'aic': med_aic, 'bic': med_bic}
        except Exception as e_med_sim: warnings.warn(f"Failed to calculate stats for median parameters: {e_med_sim}")
        for name_logA, val_logA in median_params_logA.items(): is_logA = name_logA.startswith("logA") and (name_logA[3:].startswith("A")); median_params_A[name_logA[3:] if is_logA else name_logA] = np.exp(val_logA) if is_logA else val_logA

    ranked_replicates = rank_replicates(successful_params_logA, successful_replicate_stats)

    param_distributions_logA = {p_name: np.array([params[p_name] for params in successful_params_logA]) for p_name in param_names_logA}
    param_ci_logA = {}
    alpha_level = (1.0 - confidence_level) / 2.0

    for p in param_names_logA:
        dist = param_distributions_logA[p]
        lower, upper = (np.nan, np.nan)
        if len(dist) > 3 and np.std(dist) > 1e-9 * abs(np.mean(dist)) + 1e-12:
            lower, upper = np.percentile(dist, [alpha_level * 100.0, (1.0 - alpha_level) * 100.0])
        elif len(dist) > 0: lower = upper = np.mean(dist)
        param_ci_logA[p] = (lower, upper)

    param_distributions_final = {}; param_ci_final = {}
    raw_parameter_list_A = None
    if return_replicate_params: raw_parameter_list_A = []
    for params_logA in successful_params_logA:
        params_A = {}
        for name_logA, val_logA in params_logA.items(): is_logA_param = name_logA.startswith("logA") and (name_logA[3:].startswith("A")); params_A[name_logA[3:] if is_logA_param else name_logA] = np.exp(val_logA) if is_logA_param else val_logA
        if return_replicate_params: raw_parameter_list_A.append(params_A)

    for name_logA, dist_logA in param_distributions_logA.items():
        is_logA_param = name_logA.startswith("logA") and (name_logA[3:].startswith("A")); original_key = name_logA[3:] if is_logA_param else name_logA; param_distributions_final[original_key] = np.exp(dist_logA) if is_logA_param else dist_logA; ci_logA = param_ci_logA[name_logA];
        if np.isfinite(ci_logA[0]) and np.isfinite(ci_logA[1]): param_ci_final[original_key] = (np.exp(ci_logA[0]), np.exp(ci_logA[1])) if is_logA_param else ci_logA
        else: param_ci_final[original_key] = (np.nan, np.nan)

    return ParameterBootstrapResult(
        model_name=model_name,
        parameter_distributions=param_distributions_final,
        parameter_ci=param_ci_final,
        n_iterations=n_success,
        confidence_level=confidence_level,
        raw_parameter_list=raw_parameter_list_A,
        ranked_replicates=ranked_replicates,
        median_parameters=median_params_A,
        median_stats=median_stats
    )

# --- Simulation Function ---
# (Keep existing simulate_kinetics)
def simulate_kinetics(
    model_name: str,
    model_definition_args: Dict,
    kinetic_params: Dict,
    initial_alpha: float,
    temperature_program: Union[Callable, Tuple[np.ndarray, np.ndarray]],
    simulation_time_sec: Optional[np.ndarray] = None,
    solver_options: Dict = {}
    ) -> PredictionCurveResult:
    params_A = kinetic_params; params_logA = {}; param_names_A = list(params_A.keys()); param_names_logA = []
    try:
        for name, val in params_A.items():
            is_A_param = name.startswith("A") and (name[1:].isdigit() or len(name)==1)
            logA_name = "log" + name if is_A_param else name
            params_logA[logA_name] = np.log(val) if is_A_param else val
            param_names_logA.append(logA_name)
    except (ValueError, TypeError) as e_conv: raise ValueError(f"Cannot take log of parameter {name}={val}: {e_conv}")

    try: ode_func, _, initial_state_dim, params_template = get_model_info(model_name, **model_definition_args)
    except Exception as e: raise ValueError(f"Model setup error during simulation: {e}")

    if callable(temperature_program):
        temp_func = temperature_program;
        if simulation_time_sec is None: raise ValueError("simulation_time_sec required if temp program is callable.")
        t_eval_sec = simulation_time_sec;
    elif isinstance(temperature_program, tuple) and len(temperature_program) == 2:
        t_prog_sec, temp_prog_K = temperature_program;
        temp_func = get_temperature_interpolator(t_prog_sec, temp_prog_K);
        t_eval_sec = simulation_time_sec if simulation_time_sec is not None else t_prog_sec
    else: raise TypeError("temperature_program must be a callable or a tuple of (time_sec, temp_K) arrays.")

    temp_eval_K = temp_func(t_eval_sec)

    if model_name == "A->B->C":
        initial_state = np.array([1.0 - initial_alpha, 0.0])
    elif initial_state_dim == 1:
        initial_state = np.array([initial_alpha])
    else:
        warnings.warn(f"Initial state setting for multi-state model '{model_name}' needs verification. Assuming state[0] relates to 1-alpha.")
        initial_state = np.zeros(initial_state_dim)
        initial_state[0] = 1.0 - initial_alpha

    t_pred_sec, alpha_pred = _simulate_single_dataset(
        t_eval=t_eval_sec,
        temp_func=temp_func,
        ode_system=ode_func,
        initial_state=initial_state,
        params_template=params_template,
        current_params_logA=params_logA,
        solver_options=solver_options
    )

    return PredictionCurveResult(
        time=t_pred_sec,
        temperature=temp_eval_K,
        conversion=alpha_pred,
        conversion_ci=None
    )

# --- Prediction Function ---
# (Keep existing predict_conversion, change return type)
def predict_conversion(
    kinetic_description: Union[FitResult, IsoResult],
    temperature_program: Union[Callable, Tuple[np.ndarray, np.ndarray]],
    simulation_time_sec: Optional[np.ndarray] = None,
    initial_alpha: float = 0.0,
    solver_options: Dict = {},
    bootstrap_result: Optional[ParameterBootstrapResult] = None
    ) -> PredictionCurveResult:
    if isinstance(kinetic_description, IsoResult):
        warnings.warn("Prediction directly from IsoResult not implemented yet.");
        return PredictionCurveResult(time=np.array([]), temperature=np.array([]), conversion=np.array([]))
    elif isinstance(kinetic_description, FitResult):
        fit_result = kinetic_description
        if not fit_result.success: warnings.warn("Cannot predict from unsuccessful fit."); return PredictionCurveResult(time=np.array([]), temperature=np.array([]), conversion=np.array([]))

        model_definition_args = getattr(fit_result, 'model_definition_args', None)
        if model_definition_args is None: warnings.warn("FitResult missing 'model_definition_args'. Prediction may fail or be incorrect."); model_definition_args = {}
        model_name = fit_result.model_name; params_A = fit_result.parameters

        base_prediction = simulate_kinetics(
            model_name=model_name,
            model_definition_args=model_definition_args,
            kinetic_params=params_A,
            initial_alpha=initial_alpha,
            temperature_program=temperature_program,
            simulation_time_sec=simulation_time_sec,
            solver_options=solver_options
        )

        alpha_lower_ci, alpha_upper_ci = None, None

        if bootstrap_result is not None and bootstrap_result.model_name == model_name and bootstrap_result.n_iterations > 0:
            print(f"Calculating prediction CI from {bootstrap_result.n_iterations} bootstrap parameter sets...")
            n_boot_iter = bootstrap_result.n_iterations;
            t_eval_sec = base_prediction.time
            all_boot_alphas = np.full((n_boot_iter, len(t_eval_sec)), np.nan);
            param_dist_A = bootstrap_result.parameter_distributions
            param_names_A = list(params_A.keys())

            if not all(p in param_dist_A for p in param_names_A):
                 warnings.warn("Bootstrap parameter distribution missing keys. Skipping CI calculation.")
            else:
                first_param_dist_len = len(param_dist_A.get(param_names_A[0], []))
                if first_param_dist_len < n_boot_iter:
                     warnings.warn(f"Bootstrap parameter distribution length ({first_param_dist_len}) mismatch with n_iterations ({n_boot_iter}). Using available data.")
                     n_boot_iter = first_param_dist_len

                for i in range(n_boot_iter):
                    boot_params_A = {p: param_dist_A[p][i] for p in param_names_A}
                    try:
                        boot_pred = simulate_kinetics(
                            model_name=model_name,
                            model_definition_args=model_definition_args,
                            kinetic_params=boot_params_A,
                            initial_alpha=initial_alpha,
                            temperature_program=temperature_program,
                            simulation_time_sec=t_eval_sec,
                            solver_options=solver_options
                        )
                        if len(boot_pred.conversion) == len(t_eval_sec):
                            all_boot_alphas[i, :] = boot_pred.conversion
                        else:
                             warnings.warn(f"Simulation length mismatch for bootstrap replicate {i}. Skipping.")
                    except Exception as e_boot_sim: warnings.warn(f"Simulation failed for bootstrap replicate {i}: {e_boot_sim}")

                if n_boot_iter > 0 and np.any(np.isfinite(all_boot_alphas)):
                    alpha_ci_level = (1.0 - bootstrap_result.confidence_level) / 2.0
                    lower_perc = alpha_ci_level * 100.0
                    upper_perc = (1.0 - alpha_ci_level) * 100.0
                    with warnings.catch_warnings():
                         warnings.simplefilter("ignore", category=RuntimeWarning)
                         alpha_lower_ci = np.nanpercentile(all_boot_alphas[:n_boot_iter, :], lower_perc, axis=0)
                         alpha_upper_ci = np.nanpercentile(all_boot_alphas[:n_boot_iter, :], upper_perc, axis=0)
                    print("CI calculation complete.")
                else:
                     warnings.warn("Could not calculate CIs; no successful bootstrap simulations or all results were NaN.")

        base_prediction.conversion_ci = (alpha_lower_ci, alpha_upper_ci) if alpha_lower_ci is not None and alpha_upper_ci is not None else None
        return base_prediction
    else: raise TypeError("kinetic_description must be FitResult or IsoResult.")


# --- Helper Functions for Ranking ---
# (Keep existing calculate_stats_for_replicate, calculate_median_params_and_stats, rank_replicates)
def calculate_stats_for_replicate(datasets, params_logA, model_name, model_definition_args, solver_options):
    rss, n_pts, r_squared, aic, bic = _calculate_conversion_stats(
        datasets, params_logA, model_name, model_definition_args, solver_options
    )
    return {
        'rss': rss,
        'r_squared': r_squared,
        'aic': aic,
        'bic': bic,
        'n_points': n_pts
    }

def calculate_median_params_and_stats(datasets, successful_params_logA, model_name, model_definition_args, solver_options):
    if not successful_params_logA:
        return {}, None
    median_params_logA = {
        p_name: np.median([params[p_name] for params in successful_params_logA])
        for p_name in successful_params_logA[0].keys()
    }
    rss, n_pts, r_squared, aic, bic = _calculate_conversion_stats(
        datasets, median_params_logA, model_name, model_definition_args, solver_options
    )
    median_stats = None
    if np.isfinite(bic):
         median_stats = {
            'rss': rss,
            'r_squared': r_squared,
            'aic': aic,
            'bic': bic,
            'n_points': n_pts
         }
    return median_params_logA, median_stats

def rank_replicates(successful_params_logA, successful_replicate_stats):
    if len(successful_params_logA) != len(successful_replicate_stats):
        warnings.warn("Mismatch between number of parameters and stats for ranking replicates.")
        return []

    ranked_replicates = []
    for i, (params_logA, stats) in enumerate(zip(successful_params_logA, successful_replicate_stats)):
        params_A = {
            name_logA[3:] if name_logA.startswith("logA") else name_logA: np.exp(val_logA) if name_logA.startswith("logA") else val_logA
            for name_logA, val_logA in params_logA.items()
        }
        sort_metric = stats.get('rss', np.inf)
        ranked_replicates.append({
            'rank': 0,
            'source': f'replicate_{i+1}',
            'parameters': params_A,
            'stats': stats,
            'sort_metric': sort_metric
        })

    ranked_replicates.sort(key=lambda x: (x['sort_metric']))
    for rank, item in enumerate(ranked_replicates):
        item['rank'] = rank + 1
    return ranked_replicates

# --- Model Discovery Functions ---
# (Keep existing rank_models and discover_kinetic_models)
def rank_models(
    fit_results: List[FitResult],
    score_weights: Optional[Dict[str, float]] = None
) -> List[Dict]:
    if not fit_results:
        return []

    default_weights = {'bic': 0.4, 'r_squared': 0.4, 'rss': 0.1, 'n_params': 0.1}
    weights = score_weights if score_weights and np.isclose(sum(score_weights.values()), 1.0) else default_weights
    print(f"Using ranking weights: {weights}")

    valid_fits_data = []
    stat_values = {'rss': [], 'r_squared': [], 'aic': [], 'bic': [], 'n_params': []}

    for res in fit_results:
        rss = getattr(res, 'rss', np.inf)
        r2 = getattr(res, 'r_squared', -np.inf)
        aic = getattr(res, 'aic', np.inf)
        bic = getattr(res, 'bic', np.inf)
        n_params = getattr(res, 'n_parameters', np.inf)
        n_points = getattr(res, 'n_datapoints', 0)

        if not all(np.isfinite([rss, aic, bic, n_params])) or n_points <= n_params:
            warnings.warn(f"Model '{res.model_name}' has invalid stats (RSS={rss}, AIC={aic}, BIC={bic}, Np={n_params}, Npts={n_points}) for ranking. Skipping.")
            continue

        r2 = r2 if np.isfinite(r2) else -np.inf
        valid_fits_data.append({
            'model_name': res.model_name,
            'parameters': res.parameters,
            'stats': {'rss': rss, 'r_squared': r2, 'aic': aic, 'bic': bic, 'n_params': n_params, 'n_points': n_points},
            'n_params': n_params
        })
        stat_values['rss'].append(rss)
        stat_values['r_squared'].append(r2)
        stat_values['aic'].append(aic)
        stat_values['bic'].append(bic)
        stat_values['n_params'].append(n_params)

    if not valid_fits_data:
        print("No models with valid stats found for ranking.")
        return []

    ranges = {}
    for key in ['rss', 'aic', 'bic', 'n_params']:
        vals = np.array(stat_values[key])
        min_val = np.min(vals) if len(vals) > 0 else 0
        range_width = np.ptp(vals) if len(vals) > 0 else 0
        ranges[key] = (min_val, range_width)

    finite_r2 = [r for r in stat_values['r_squared'] if np.isfinite(r)]
    if finite_r2:
        min_r2_norm = np.min(finite_r2)
        max_r2_norm = np.max(finite_r2)
        range_r2_norm = max_r2_norm - min_r2_norm
        ranges['r_squared'] = (min_r2_norm, max_r2_norm, range_r2_norm)
    else:
        ranges['r_squared'] = (0, 0, 0)

    for item in valid_fits_data:
        stats = item['stats']
        score = 0.0
        for key, weight in weights.items():
            if weight == 0: continue

            if key == 'r_squared':
                min_r2_norm, max_r2_norm, range_r2_norm = ranges['r_squared']
                val = stats.get('r_squared', -np.inf)
                if not np.isfinite(val):
                    norm_val = 1.0
                elif range_r2_norm > 1e-9:
                    norm_val = (max_r2_norm - val) / range_r2_norm
                else:
                    norm_val = 0.0
                score += weight * norm_val
            elif key in ['rss', 'aic', 'bic', 'n_params']:
                min_val, range_width = ranges[key]
                val = stats.get(key, np.inf)
                if not np.isfinite(val):
                    norm_val = 1.0
                elif range_width > 1e-9:
                    norm_val = (val - min_val) / range_width
                else:
                    norm_val = 0.0
                score += weight * norm_val
        item['score'] = score

    valid_fits_data.sort(key=lambda x: (x['score'], x['n_params']))
    for rank, item in enumerate(valid_fits_data):
        item['rank'] = rank + 1

    return valid_fits_data


def discover_kinetic_models(
    datasets: List[KineticDataset],
    models_to_try: List[Dict],
    initial_guesses_pool: Dict[str, Dict],
    parameter_bounds_pool: Optional[Dict[str, Dict]] = None,
    solver_options: Dict = {},
    optimizer_options: Dict = {},
    score_weights: Optional[Dict[str, float]] = None
) -> List[Dict]:
    all_fit_results = []
    print(f"--- Starting Kinetic Model Discovery ({len(models_to_try)} models) ---")

    for model_info in models_to_try:
        user_model_name = model_info.get('name')
        internal_model_type = model_info.get('type')
        def_args = model_info.get('def_args')
        if not user_model_name or not internal_model_type or def_args is None:
            warnings.warn(f"Skipping invalid model definition: {model_info}")
            continue
        print(f"\n--- Fitting Model: {user_model_name} (Type: {internal_model_type}) ---")
        guesses = initial_guesses_pool.get(user_model_name)
        if guesses is None:
            warnings.warn(f"No initial guesses found for model '{user_model_name}'. Skipping.")
            continue
        bounds = parameter_bounds_pool.get(user_model_name) if parameter_bounds_pool else None

        fit_res = fit_kinetic_model(
            datasets=datasets,
            model_name=internal_model_type,
            model_definition_args=def_args,
            initial_guesses=guesses,
            parameter_bounds=bounds,
            solver_options=solver_options,
            optimizer_options=optimizer_options,
            callback=None
        )
        if fit_res.success:
            print(f"Model '{user_model_name}' fit successful.")
            fit_res.model_name = user_model_name
            all_fit_results.append(fit_res)
        else:
            print(f"Model '{user_model_name}' fit failed: {fit_res.message}")

    print(f"\n--- Model Discovery Finished ---")
    if not all_fit_results:
        print("No models fitted successfully.")
        return []

    ranked_list = rank_models(all_fit_results, score_weights=score_weights)
    print(f"Ranking {len(ranked_list)} successful models by combined score...")

    return ranked_list

# ==============================================================================
# <<< NEW SB-Sum + Ea(variable) Workflow >>>
# ==============================================================================

# --- ODE Function for Variable Ea ---
def _kinetic_ode_ea_variable(t, alpha_state, A, model_func, model_params, temp_interpolator, ea_interpolator):
    """
    Defines the ODE system d(alpha)/dt = k(T, alpha) * f(alpha)
    where Ea is a function of alpha.
    Assumes alpha_state is a list/array where alpha_state[0] is the conversion alpha.
    """
    alpha = alpha_state[0]
    alpha_clipped = np.clip(alpha, 0.0, 1.0)

    Ea = ea_interpolator(alpha_clipped)
    # --- Add print for debugging ---
    # if alpha < 0.01:  # Print only near the start
    #     print(f"Debug ODE: t={t:.2f}, alpha={alpha:.4f}, Ea={Ea:.2f}")
    # --- End print ---
    if Ea <= 0:
        Ea = 1e-6

    T = temp_interpolator(t)
    if T <= 0:
        return [0.0]

    k = A * np.exp(-Ea / (R_GAS * T))

    f_alpha = model_func(alpha_clipped, *model_params)

    dadt = k * f_alpha

    dadt = np.nan_to_num(dadt, nan=0.0, posinf=1e6, neginf=-1e6)

    return [dadt]

# --- Fitting Function for SB Sum Model using Ea(alpha) ---
def fit_sb_sum_model_ea_variable(
    experimental_runs: List[KineticDataset],
    ea_interpolator: Callable,
    initial_params: Optional[Dict[str, float]] = None, # <<< ENSURE THIS ARGUMENT IS PRESENT
    param_bounds: Optional[Dict[str, Tuple[float, float]]] = None
    ) -> Tuple[Optional[Dict[str, float]], Optional[np.ndarray]]:
    """
    Fits the Sestak-Berggren sum model using NLLS (curve_fit), incorporating Ea(alpha).
    """
    # ... (setup code: all_times, all_alphas, etc. remains the same) ...
    all_times = []
    all_alphas = []
    all_temp_interpolators = []
    all_initial_alphas = []

    for run in experimental_runs:
        time_data = run.time
        temp_data = run.temperature
        alpha_data = run.conversion
        if len(time_data) < 2: continue
        sort_idx = np.argsort(time_data)
        time_data = time_data[sort_idx]
        temp_data = temp_data[sort_idx]
        alpha_data = alpha_data[sort_idx]
        temp_interp = interp1d(time_data, temp_data, kind='linear', fill_value="extrapolate")
        all_times.append(time_data)
        all_alphas.append(alpha_data)
        all_temp_interpolators.append(temp_interp)
        all_initial_alphas.append(alpha_data[0])

    if not all_times:
         print("Error: No valid experimental runs provided for fitting.")
         return None, None

    # --- Define the objective function (remains the same) ---
    def _objective_multi_run_curvefit(t_combined_dummy, A, c1, m1, n1, m2, n2):
        # ... (objective function code remains the same) ...
        simulated_alphas_combined = []
        model_params = (c1, m1, n1, m2, n2)
        # --- Add Debug Print Here (Optional) ---
        # print(f"Debug Fit Obj: A={A:.2e}, c1={c1:.2f}, m1={m1:.2f}, n1={n1:.2f}, m2={m2:.2f}, n2={n2:.2f}")
        # ---
        for i, time_data in enumerate(all_times):
            temp_interp = all_temp_interpolators[i]
            alpha_initial = all_initial_alphas[i]
            t_span = (time_data[0], time_data[-1])
            try: # Add try-except around solve_ivp for better debugging
                sol = solve_ivp(
                    _kinetic_ode_ea_variable, t_span, [alpha_initial], t_eval=time_data,
                    args=(A, models.f_sb_sum, model_params, temp_interp, ea_interpolator),
                    method='Radau', rtol=1e-6, atol=1e-9
                )
                if sol.status != 0:
                    # print(f"Warning: ODE integration failed for run {i} during fitting (Status: {sol.status}). Params: A={A:.2e}, c1={c1:.2f}, m1={m1:.2f}, n1={n1:.2f}, m2={m2:.2f}, n2={n2:.2f}")
                    return np.full(sum(len(t) for t in all_times), np.inf) # Return inf on failure
                if len(sol.y[0]) != len(time_data):
                    # print(f"Warning: Simulation length mismatch for run {i}. Params: A={A:.2e}, c1={c1:.2f}, m1={m1:.2f}, n1={n1:.2f}, m2={m2:.2f}, n2={n2:.2f}")
                    return np.full(sum(len(t) for t in all_times), np.inf) # Return inf on mismatch
                simulated_alphas_combined.extend(sol.y[0])
            except Exception as ode_err:
                 # print(f"Error during solve_ivp for run {i}: {ode_err}. Params: A={A:.2e}, c1={c1:.2f}, m1={m1:.2f}, n1={n1:.2f}, m2={m2:.2f}, n2={n2:.2f}")
                 return np.full(sum(len(t) for t in all_times), np.inf) # Return inf on exception
        return np.array(simulated_alphas_combined)
    # --- End objective function ---

    # Prepare parameters for curve_fit
    param_order = ['A', 'c1', 'm1', 'n1', 'm2', 'n2']
    default_initial = {'A': 1e10, 'c1': 0.5, 'm1': 0.5, 'n1': 1.0, 'm2': 1.5, 'n2': 0.5}

    # **** USE initial_params argument ****
    p0_dict = default_initial.copy()
    if initial_params:
        # Validate provided initial_params keys
        valid_keys = set(param_order)
        provided_keys = set(initial_params.keys())
        if not provided_keys.issubset(valid_keys):
             unknown_keys = provided_keys - valid_keys
             warnings.warn(f"Ignoring unknown keys in initial_params: {unknown_keys}")
             initial_params = {k: v for k, v in initial_params.items() if k in valid_keys}
        p0_dict.update(initial_params) # Update defaults with validated provided guesses
    p0 = [p0_dict[k] for k in param_order] # Create list in correct order
    # **** END USE initial_params ****

    # Adjust default bounds for exponents
    default_bounds = {
        'A': (1e1, 1e25),
        'c1': (0, 1),
        'm1': (0, 5),  # Lower bound changed from -1 to 0
        'n1': (0, 5),  # Lower bound changed from -1 to 0
        'm2': (0, 5),  # Lower bound changed from -1 to 0
        'n2': (0, 5)   # Lower bound changed from -1 to 0
    }

    if param_bounds:
        bounds_lower = [param_bounds.get(k, default_bounds[k])[0] for k in param_order]
        bounds_upper = [param_bounds.get(k, default_bounds[k])[1] for k in param_order]
    else:
        bounds_lower = [default_bounds[k][0] for k in param_order]
        bounds_upper = [default_bounds[k][1] for k in param_order]
    bounds = (bounds_lower, bounds_upper)

    # ... (data flattening remains the same) ...
    total_points = sum(len(t) for t in all_times)
    t_combined_dummy = np.arange(total_points)
    alpha_combined_exp = np.concatenate(all_alphas)
    if len(alpha_combined_exp) != total_points:
         print("Error: Length mismatch between combined experimental alpha and expected total points.")
         return None, None

    # --- curve_fit call (remains the same) ---
    try:
        print(f"Starting NLLS fitting for SB Sum model using curve_fit with initial guesses: {p0_dict}") # Print guesses being used
        popt, pcov = curve_fit(
            _objective_multi_run_curvefit, t_combined_dummy, alpha_combined_exp,
            p0=p0, bounds=bounds, method='trf', ftol=1e-6, xtol=1e-6, gtol=1e-6, max_nfev=3000
        )
        print("Fitting successful.")
        fit_params_dict = dict(zip(param_order, popt))
        return fit_params_dict, pcov
    except ValueError as ve: # Catch the specific error
         print(f"Error during curve_fit: {ve}")
         print("This often indicates the ODE solver failed with the initial parameters.")
         print(f"Initial parameters used: {p0_dict}")
         # Optionally add more debugging here if needed
         return None, None
    except Exception as e:
        print(f"Unexpected error during curve_fit for SB Sum model: {e}")
        traceback.print_exc()
        return None, None

# --- Prediction Function for SB Sum Model (Time to reach alpha_target isothermally) ---
def predict_time_to_alpha_iso(
    fit_params: Dict[str, float],
    ea_interpolator: Callable,
    target_temp_K: float,
    target_alpha: float,
    alpha_initial: float = 0.0,
    t_max: float = 1e12,
    rate_threshold: float = 1e-25  # Threshold to consider rate negligible
    ) -> Optional[float]:
    """
    Predicts the time to reach a target alpha under isothermal conditions
    using the SB-Sum model and variable Ea(alpha).
    Checks for negligible initial rate before integration.

    Args:
        fit_params: Dictionary of fitted parameters.
        ea_interpolator: Callable to interpolate activation energy as a function of alpha.
        target_temp_K: Target temperature in Kelvin.
        target_alpha: Target conversion alpha.
        alpha_initial: Initial conversion alpha.
        t_max: Maximum time for integration.
        rate_threshold: If initial d(alpha)/dt is below this, assume target will not be reached.

    Returns:
        float: Predicted time (in seconds) to reach target_alpha, or None if not reached
               due to negligible rate or integration failure/timeout.
    """
    if target_alpha <= alpha_initial:
        return 0.0

    required_params = ['A', 'c1', 'm1', 'n1', 'm2', 'n2']
    if not all(p in fit_params for p in required_params):
        missing = [p for p in required_params if p not in fit_params]
        print(f"Error: Missing parameters for prediction: {missing}")
        return None

    A = fit_params['A']
    model_params = (fit_params['c1'], fit_params['m1'], fit_params['n1'], fit_params['m2'], fit_params['n2'])
    temp_interp_const = lambda t: target_temp_K

    # --- Check Initial Rate ---
    try:
        alpha_check = max(alpha_initial, 1e-9)  # Use small positive alpha if starting at 0
        Ea_initial = ea_interpolator(np.clip(alpha_check, 0.0, 1.0))
        if Ea_initial <= 0:
            Ea_initial = 1e-6
        k_initial = A * np.exp(-Ea_initial / (R_GAS * target_temp_K))
        f_alpha_initial = models.f_sb_sum(alpha_check, *model_params)
        initial_rate = k_initial * f_alpha_initial

        # Debug information
        print(f"\n--- Debug Info for predict_time_to_alpha_iso ---")
        print(f"Target Temp: {target_temp_K:.2f} K ({target_temp_K - 273.15:.1f} C)")
        print(f"Target Alpha: {target_alpha:.3f}")
        print(f"Initial Alpha: {alpha_initial:.4f}")
        print(f"Fitted Params: A={A:.3e}, c1={model_params[0]:.3f}, m1={model_params[1]:.3g}, n1={model_params[2]:.3f}, m2={model_params[3]:.3f}, n2={model_params[4]:.3f}")
        print(f"Ea at alpha~{alpha_check:.3g}: {Ea_initial:.2f} J/mol ({Ea_initial/1000:.1f} kJ/mol)")
        print(f"Rate constant k at T={target_temp_K:.1f}K, alpha~{alpha_check:.3g}: {k_initial:.3e} 1/s")
        print(f"f(alpha) at alpha={alpha_check:.3g}: {f_alpha_initial:.3e}")
        print(f"Estimated initial d(alpha)/dt: {initial_rate:.3e} 1/s")
        print(f"--- End Debug Info ---")

        if not np.isfinite(initial_rate) or abs(initial_rate) < rate_threshold:
            print(f"Warning: Initial predicted rate ({initial_rate:.2e} 1/s) is negligible or non-finite. Target alpha ({target_alpha}) assumed not reachable.")
            return None  # Return None immediately

    except Exception as e_debug:
        print(f"Error during initial rate check: {e_debug}")
        return None  # Cannot proceed if initial check fails
    # --- End Check Initial Rate ---

    # Define event function (accepts *args)
    def reach_target_alpha(t, alpha_state, *args):
        return alpha_state[0] - target_alpha
    reach_target_alpha.terminal = True
    reach_target_alpha.direction = 1

    print(f"Starting prediction integration: Time to alpha={target_alpha} at T={target_temp_K:.2f} K...")
    try:
        ode_args = (A, models.f_sb_sum, model_params, temp_interp_const, ea_interpolator)
        sol = solve_ivp(
            _kinetic_ode_ea_variable, (0, t_max), [alpha_initial],
            args=ode_args, method='Radau', events=reach_target_alpha,
            dense_output=True, rtol=1e-7, atol=1e-10
        )

        if sol.status == 0 or sol.status == 1:
            if sol.t_events and sol.t_events[0].size > 0:
                time_at_target = sol.t_events[0][0]
                print(f"Prediction successful: Time = {time_at_target:.4g} seconds")
                return time_at_target
            else:
                final_alpha_reached = sol.y[0][-1]
                print(f"Warning: Target alpha ({target_alpha}) not reached within t_max ({t_max:.2g} s). Final alpha reached: {final_alpha_reached:.4f}")
                return None  # Return None if target not reached by solver
        else:
            print(f"Warning: ODE integration failed during prediction (Status: {sol.status}, Message: {sol.message}).")
            return None

    except Exception as e:
        print(f"Error during prediction integration: {e}")
        traceback.print_exc()
        return None

# --- Orchestration Function for SB Sum + Ea(variable) Workflow ---
def run_full_analysis_sb_sum(
    experimental_runs: List[KineticDataset],
    iso_analysis_func: Callable,
    initial_params_sb: Optional[Dict[str, float]] = None,  # Initial guess for the *first* fit
    target_temp_C: float = 25.0,
    target_alpha: float = 0.05,
    n_bootstrap: int = 500,
    confidence_level: float = 95.0,
    prediction_time_unit: str = "days",
    perturb_bootstrap_guess: float = 0.01  # Add small relative perturbation factor
    ) -> Optional[FullAnalysisResult]:
    """
    Orchestrates the full analysis using Ea(variable) and SB-Sum model:
    Iso -> Fit -> Predict -> Bootstrap Prediction CI.

    Args:
        experimental_runs: List of KineticDataset objects.
        iso_analysis_func: The isoconversional function to use (e.g., iso.run_vyazovkin).
        initial_params_sb: Optional dictionary of initial guesses for the *first* SB-Sum fit.
        target_temp_C: Target prediction temperature (Celsius).
        target_alpha: Target prediction alpha.
        n_bootstrap: Number of bootstrap iterations for prediction CI.
        confidence_level: Confidence level for prediction interval (e.g., 95.0).
        prediction_time_unit: Unit for reporting ('seconds', 'minutes', 'hours', 'days', 'years').
        perturb_bootstrap_guess: Relative factor (e.g., 0.01 for 1%) to perturb the
                                 original fit parameters as initial guesses for bootstrap fits.
                                 Set to 0 for no perturbation.

    Returns:
        FullAnalysisResult: Dataclass containing all results, or None if a critical step fails.
    """
    print("--- Starting Full Kinetic Analysis (SB-Sum + Ea Variable) ---")
    start_time_analysis = time.time()
    final_result = FullAnalysisResult(input_datasets=experimental_runs)

    # --- Step 1: Isoconversional Analysis ---
    print("Step 1: Running Isoconversional Analysis...")
    try:
        iso_result: Optional[IsoResult] = iso_analysis_func(experimental_runs)
        if (iso_result is None or iso_result.alpha is None or len(iso_result.alpha) < 2 or
            iso_result.Ea is None or len(iso_result.Ea) != len(iso_result.alpha) or
            np.all(~np.isfinite(iso_result.Ea))):
            print("Error: Isoconversional analysis failed or returned invalid/insufficient data.")
            return None
        final_result.isoconversional_result = iso_result
        print(f"Isoconversional analysis complete using {iso_result.method}. Found Ea for {len(iso_result.alpha)} alpha values.")
        ea_interpolator = create_ea_interpolator(iso_result.alpha, iso_result.Ea)
    except Exception as e_iso:
        print(f"Error during Isoconversional Analysis: {e_iso}")
        traceback.print_exc()
        return None

    # --- Step 2: Model Fitting ---
    print("\nStep 2: Fitting SB Sum model using Ea(alpha)...")
    fit_params_dict = None
    try:
        fit_params_dict, fit_cov = fit_sb_sum_model_ea_variable(
            experimental_runs,
            ea_interpolator,
            initial_params=initial_params_sb
        )
        if fit_params_dict is None:
            print("Error: SB-Sum model fitting failed on original data.")
            if iso_result and iso_result.Ea is not None:
                valid_ea = iso_result.Ea[np.isfinite(iso_result.Ea)]
                if len(valid_ea) > 0:
                    print(f"Hint: Calculated Ea range: {np.min(valid_ea)/1000:.1f} - {np.max(valid_ea)/1000:.1f} kJ/mol. Adjust initial 'A' guess accordingly.")
            return None

        fit_result = FitResult(
            model_name="SestakBerggrenSum_EaVar",
            parameters=fit_params_dict,
            success=True,
            message="Fit successful (curve_fit)",
            rss=np.nan,
            n_datapoints=sum(len(run.time) for run in experimental_runs if len(run.time) > 0),
            n_parameters=len(fit_params_dict),
            covariance_matrix=fit_cov,
            aic=np.nan,
            bic=np.nan,
            r_squared=np.nan
        )
        final_result.fit_result = fit_result
        print("SB-Sum model fitting complete.")
        print("Fitted Parameters:", fit_params_dict)

    except Exception as e_fit:
        print(f"Error during SB-Sum Model Fitting: {e_fit}")
        traceback.print_exc()
        return None

    # --- Step 3: Prediction ---
    print("\nStep 3: Making prediction using parameters from original data fit...")
    if fit_params_dict is None:
        print("Error: Cannot proceed to prediction, fitting failed.")
        return final_result

    target_temp_K = target_temp_C + 273.15
    prediction_orig_sec = predict_time_to_alpha_iso(
        fit_params_dict, ea_interpolator, target_temp_K, target_alpha, t_max=1e12
    )
    time_conversion_factors = {"seconds": 1.0, "minutes": 60.0, "hours": 3600.0, "days": 86400.0, "years": 31536000.0}
    time_factor = time_conversion_factors.get(prediction_time_unit.lower(), 1.0)
    prediction_orig_unit = prediction_orig_sec / time_factor if prediction_orig_sec is not None else None
    single_pred_result = None
    pred_desc = f"Time to {target_alpha:.3f} alpha at {target_temp_C}°C"
    if prediction_orig_unit is not None:
        print(f"Prediction (Original Fit): {pred_desc} = {prediction_orig_unit:.4g} {prediction_time_unit}")
        single_pred_result = SingleValuePredictionResult(
            target_description=pred_desc,
            predicted_value=prediction_orig_unit,
            unit=prediction_time_unit
        )
    else:
        print(f"Prediction (Original Fit): Target alpha ({target_alpha}) not reached at {target_temp_C}°C within t_max.")
        single_pred_result = SingleValuePredictionResult(
            target_description=pred_desc,
            predicted_value=np.inf,
            unit=prediction_time_unit
        )
    final_result.prediction_from_original_fit = single_pred_result

    # --- Step 4: Bootstrap Analysis ---
    print(f"\nStep 4: Running Bootstrap Analysis ({n_bootstrap} iterations) for Prediction Confidence Interval...")
    bootstrap_pred_result = None
    if n_bootstrap > 0:
        def _bootstrap_fit_wrapper(runs, ea_interp_boot):
            perturbed_guess = {}
            if perturb_bootstrap_guess > 0:
                for key, val in fit_params_dict.items():
                    # Use a combination of relative and absolute noise floor
                    noise_std_dev = abs(val * perturb_bootstrap_guess) + 1e-6  # Add small absolute floor
                    perturbed_guess[key] = val + np.random.normal(0, noise_std_dev)
                # Ensure c1 stays within reasonable bounds after perturbation
                perturbed_guess['c1'] = np.clip(perturbed_guess.get('c1', 0.5), 0.01, 0.99)
                # Optional: Clip exponents if they go too far out of bounds
                perturbed_guess['m1'] = np.clip(perturbed_guess.get('m1', 0), 0, 5)
                perturbed_guess['n1'] = np.clip(perturbed_guess.get('n1', 0), 0, 5)
                perturbed_guess['m2'] = np.clip(perturbed_guess.get('m2', 0), 0, 5)
                perturbed_guess['n2'] = np.clip(perturbed_guess.get('n2', 0), 0, 5)
            else:
                perturbed_guess = fit_params_dict.copy()  # Use original fit directly if no perturbation

            params_boot, _ = fit_sb_sum_model_ea_variable(
                runs,
                ea_interp_boot,
                initial_params=perturbed_guess
            )
            return params_boot

        def _bootstrap_iso_wrapper(runs):
            iso_res_boot = iso_analysis_func(runs)
            # Check for valid result object and finite Ea values
            if iso_res_boot and iso_res_boot.alpha is not None and iso_res_boot.Ea is not None and np.any(np.isfinite(iso_res_boot.Ea)):
                return (iso_res_boot.alpha, iso_res_boot.Ea)
            else:
                return (None, None)  # Indicate failure

        def _bootstrap_predict_wrapper(params_boot, ea_interp_boot):
            # Add check for valid interpolator
            if params_boot is None or ea_interp_boot is None:
                return None  # Cannot predict if fit or iso failed
            pred_sec = predict_time_to_alpha_iso(params_boot, ea_interp_boot, target_temp_K, target_alpha, t_max=1e12)
            return pred_sec / time_factor if pred_sec is not None else np.inf

        try:
            median_pred_boot_unit, ci_boot_unit, boot_distribution = utils.bootstrap_prediction(
                experimental_runs=experimental_runs,
                iso_analysis_func=_bootstrap_iso_wrapper,
                fit_func=_bootstrap_fit_wrapper,
                predict_func=_bootstrap_predict_wrapper,
                n_bootstrap=n_bootstrap,
                confidence_level=confidence_level
            )
            num_success_bootstrap = 0
            valid_predictions = []
            if boot_distribution is not None:
                valid_predictions = boot_distribution[np.isfinite(boot_distribution)]
                num_success_bootstrap = len(valid_predictions)

            if num_success_bootstrap > 0:
                median_pred_finite = np.median(valid_predictions)
                lower_perc = (100.0 - confidence_level) / 2.0
                upper_perc = 100.0 - lower_perc
                ci_finite = (np.percentile(valid_predictions, lower_perc),
                             np.percentile(valid_predictions, upper_perc))
                bootstrap_pred_result = BootstrapPredictionResult(
                    target_description=pred_desc,
                    predicted_value_distribution=valid_predictions,
                    predicted_value_median=median_pred_finite,
                    predicted_value_ci=ci_finite,
                    unit=prediction_time_unit,
                    n_iterations=num_success_bootstrap,
                    confidence_level=confidence_level
                )
                print(f"\nBootstrap Prediction ({confidence_level}% CI from {num_success_bootstrap}/{n_bootstrap} successful iterations reaching target):")
                print(f"  Median: {median_pred_finite:.4g} {prediction_time_unit}")
                print(f"  CI: ({ci_finite[0]:.4g} - {ci_finite[1]:.4g}) {prediction_time_unit}")
            else:
                print(f"\nBootstrap Prediction: Target alpha ({target_alpha}) not reached in any of the {n_bootstrap} bootstrap iterations.")
                bootstrap_pred_result = BootstrapPredictionResult(
                    target_description=pred_desc,
                    predicted_value_distribution=np.array([]),
                    predicted_value_median=np.inf,
                    predicted_value_ci=(np.inf, np.inf),
                    unit=prediction_time_unit,
                    n_iterations=0,
                    confidence_level=confidence_level
                )
            final_result.prediction_bootstrap_result = bootstrap_pred_result

        except Exception as e_boot:
            print(f"Error during Bootstrap Analysis execution: {e_boot}")
            traceback.print_exc()
    else:
        print("\nStep 4: Skipping Bootstrap Analysis (n_bootstrap = 0).")

    analysis_end_time = time.time()
    print(f"\n--- Analysis Complete (Total Time: {analysis_end_time - start_time_analysis:.2f} seconds) ---")
    return final_result