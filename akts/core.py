import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize, OptimizeResult
from concurrent.futures import TimeoutError as FuturesTimeoutError
import warnings
import copy
import time
import traceback  # For more detailed error info
import functools  # For partial used in bootstrap callback
from typing import List, Dict, Tuple, Optional, Callable, Union
import concurrent.futures

# --- Import from sibling modules ---
from .datatypes import (KineticDataset, FitResult, BootstrapResult,
                       PredictionResult, IsoResult, FAlphaCallable, OdeSystemCallable)
from .models import get_model_info, F_ALPHA_MODELS
from .utils import (get_temperature_interpolator, calculate_aic, calculate_bic, R_GAS, numerical_diff)
from .isoconversional import run_friedman, run_kas, run_ofw

# --- Helper to prepare the full ODE parameter dictionary ---
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

    # --- Prepare the full parameter dict for the ODE solver ---
    try:
        full_params_ode = _prepare_full_params_for_ode(params_template, current_params_logA)
    except Exception as e_prep:
        warnings.warn(f"Simulate: Error preparing ODE params: {e_prep}")
        alpha_nan = np.full((len(t_eval),), np.nan); return t_eval, alpha_nan

    # --- Prepare time evaluation ---
    sort_indices = np.argsort(t_eval); t_eval_sorted = t_eval[sort_indices]; t_start, t_end = t_eval_sorted[0], t_eval_sorted[-1]
    if t_start >= t_end: alpha_out = np.full_like(t_eval, initial_state[0]); unsort_indices = np.argsort(sort_indices); return t_eval, alpha_out[unsort_indices]

    # --- Attempt Solvers ---
    sol = None; success = False
    for solver in [primary_solver, fallback_solver]:
        if success: break # Stop if primary succeeded
        if solver is None: continue # Skip if no fallback defined
        try:
            current_solver_kwargs = common_solver_kwargs.copy()
            sol = solve_ivp(
                fun=ode_system,
                t_span=(t_start, t_end),
                y0=initial_state,
                t_eval=t_eval_sorted,
                args=(temp_func, full_params_ode),  # Added full_params_ode here
                method=solver,
                **current_solver_kwargs
            )
            success = sol.success
            if not success and solver == primary_solver: warnings.warn(f"Primary solver '{solver}' failed: {sol.message}. Trying fallback.")
            elif not success and solver == fallback_solver: warnings.warn(f"Fallback solver '{solver}' failed: {sol.message}")
            elif success and solver == fallback_solver: warnings.warn(f"Fallback solver '{solver}' succeeded.")
        except Exception as e_solve: warnings.warn(f"Solver '{solver}' failed execution: {e_solve}"); success = False
        if success: break # Exit loop if successful

    # --- Process Result or Handle Failure ---
    if success and sol is not None:
        state_sim_sorted = sol.y; alpha_sim_sorted = np.zeros_like(sol.t); ode_func_name = ode_system.__name__
        # Determine alpha based on model
        if ode_func_name == 'ode_system_single_step': alpha_sim_sorted = state_sim_sorted[0, :]
        elif ode_func_name == 'ode_system_A_plus_B_C': alpha_sim_sorted = state_sim_sorted[0, :]
        elif ode_func_name == 'ode_system_A_B_C': alpha_sim_sorted = 1.0 - state_sim_sorted[0, :] - state_sim_sorted[1, :]
        else: alpha_sim_sorted = state_sim_sorted[0, :] # Default assumption
        alpha_sim_sorted = np.clip(alpha_sim_sorted, 0.0, 1.0); unsort_indices = np.argsort(sort_indices); alpha_sim_unsorted = alpha_sim_sorted[unsort_indices]
        return t_eval, alpha_sim_unsorted
    else:
        warnings.warn(f"Both solvers failed or simulation error occurred."); alpha_nan = np.full((len(t_eval),), np.nan); return t_eval, alpha_nan

# --- Objective Function for Fitting ---
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
    total_weighted_rss = 0.0  # Changed variable name
    n_total_datapoints = 0
    current_params_logA = dict(zip(param_names_logA, params_array_logA))

    # --- Weighting Parameters ---
    weight_transition = 10.0  # Give 10x more weight to transition points
    weight_baseline_plateau = 1.0
    alpha_lower = 0.05
    alpha_upper = 0.95
    # --------------------------
    current_params_logA = dict(zip(param_names_logA, params_array_logA))
    if callback_func: # Callback logic
        try:
            current_params_A = {};
            # **** Correctly handle non-logA params ****
            for name_logA, val_logA in current_params_logA.items():
                 # Check if it's a logA parameter based on the internal name
                 is_logA = name_logA.startswith("logA") and (name_logA[3:].startswith("A"))
                 # Get the original name (A1 or Ea1 or p1_n etc.)
                 original_name = name_logA[3:] if is_logA else name_logA
                 # Get the value in A-scale (or original scale if not logA)
                 param_val_A = np.exp(val_logA) if is_logA else val_logA
                 current_params_A[original_name] = param_val_A # Store with original name
            callback_func(iteration_counter[0], current_params_A) # Pass dict with A-scale params
        except Exception as e_cb: warnings.warn(f"Objective callback failed: {e_cb}")
    iteration_counter[0] += 1

    try:
        ode_func, _, initial_state_dim, params_template = get_model_info(model_name, **model_definition_args)  # Get model info
    except ValueError as e:
        warnings.warn(f"Objective func: Model info error: {e}")
        return np.inf

    if model_name == "A->B->C":
        initial_state = np.array([1.0, 0.0])  # Initial state
    else:
        initial_state = np.zeros(initial_state_dim)

    for ds in datasets:  # Simulation loop
        if len(ds.time) < 2:
            continue
        try:
            temp_func = get_temperature_interpolator(ds.time, ds.temperature)
            t_sim_eval = ds.time
            # Call simulation - Pass template and current logA params
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
            valid_mask = np.isfinite(residuals) & np.isfinite(alpha_sim)  # Ensure both are finite

            # --- Apply Weights ---
            weights = np.full_like(residuals, weight_baseline_plateau)
            transition_mask = (ds.conversion > alpha_lower) & (ds.conversion < alpha_upper)
            weights[transition_mask] = weight_transition
            # Only consider valid points for weighting and RSS
            weights = weights[valid_mask]
            valid_residuals = residuals[valid_mask]
            # ---------------------

            if len(valid_residuals) == 0 and len(residuals) > 0:
                rss = np.inf
            elif len(valid_residuals) > 0:
                # --- Calculate Weighted RSS ---
                weighted_sq_residuals = weights * (valid_residuals**2)
                rss = np.sum(weighted_sq_residuals)
                # ----------------------------
            else:
                rss = 0.0

            if not np.isfinite(rss):
                rss = np.inf
            total_weighted_rss += rss  # Sum weighted RSS
            n_total_datapoints += len(valid_residuals)  # Keep track of total points for AIC/BIC

        except Exception as e:
            warnings.warn(f"Objective func: Sim error: {e}.")
            total_weighted_rss = np.inf
            break

    if not np.isfinite(total_weighted_rss):
        total_weighted_rss = 1e30

    # NOTE: We return the WEIGHTED RSS to the optimizer.
    # AIC/BIC calculations in fit_kinetic_model might be less meaningful
    # if they use this weighted RSS directly without accounting for weights.
    # For now, we prioritize getting better parameters.
    return total_weighted_rss

# --- Objective Function for Fitting (Rate-Based) ---
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
    """Calculates total RSS between simulated and experimental RATES."""
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

    initial_state = np.array([1.0, 0.0]) if model_name == "A->B->C" else np.zeros(initial_state_dim)

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
                warnings.warn(f"Rate objective: Rate length mismatch after diff.")
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

# --- New Helper Functions ---
def _calculate_conversion_stats(
    datasets: List[KineticDataset],
    params_logA: Dict,
    model_name: str,
    model_definition_args: Dict,
    solver_options: Dict
) -> Tuple[float, int, float, float, float]:
    """Calculates RSS, N, R², AICc, and BIC for a given parameter set."""
    rss = np.inf
    n_pts = 0
    all_exp_conv = []
    n_params = len(params_logA)

    try:
        ode_func, _, initial_state_dim, params_template = get_model_info(model_name, **model_definition_args)
        initial_state = np.array([1.0, 0.0]) if model_name == "A->B->C" else np.zeros(initial_state_dim)
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
                valid_mask = np.isfinite(residuals) & np.isfinite(alpha_sim)
                if np.sum(valid_mask) > 0:
                    current_rss_calc += np.sum(residuals[valid_mask]**2)
                    n_pts += np.sum(valid_mask)
                    all_exp_conv.extend(ds.conversion[valid_mask])
        if n_pts > 0:
            rss = current_rss_calc
    except Exception as e_sim:
        warnings.warn(f"Failed to simulate conversion curve for stats calculation: {e_sim}")

    r_squared = np.nan
    aic = np.nan
    bic = np.nan
    if n_pts > n_params and np.isfinite(rss):
        mean_y = np.mean(all_exp_conv)
        total_ss = np.sum((np.array(all_exp_conv) - mean_y)**2)
        r_squared = 1.0 - (rss / total_ss) if total_ss > 1e-12 else (1.0 if rss < 1e-12 else 0.0)
        aic = calculate_aic(rss, n_params, n_pts)
        bic = calculate_bic(rss, n_params, n_pts)

    return rss, n_pts, r_squared, aic, bic

# --- Main Fitting Function ---
def fit_kinetic_model(
    datasets: List[KineticDataset],
    model_name: str,
    model_definition_args: Dict,
    initial_guesses: Dict[str, float], # Expects guesses for A (not logA)
    parameter_bounds: Optional[Dict[str, Tuple[float, float]]] = None, # Expects bounds for A
    solver_options: Dict = {},
    optimizer_options: Dict = {},
    callback: Optional[Callable[[int, Dict], None]] = None,
    optimize_on_rate: bool = True
    ) -> FitResult:
    """
    Fits model using logA scaling internally. User provides guesses/bounds for A.
    Attempts optimization on CONVERSION residuals first. If that fails,
    falls back to optimizing on RATE residuals.
    Reports final stats based on UNWEIGHTED CONVERSION fit quality.
    Includes defaults, warnings, callback. Stores model_definition_args in result.
    """
    default_method = 'L-BFGS-B'; opt_method = optimizer_options.get('method', default_method); default_opt_options = {'disp': False, 'ftol': 1e-9, 'gtol': 1e-7}; opt_options = optimizer_options.get('options', default_opt_options)
    if not isinstance(opt_options, dict): opt_options = default_opt_options

    param_names_logA = []; original_param_names_map = {}
    try: # Determine logA parameter names and map
        _, original_param_names, _, _ = get_model_info(model_name, **model_definition_args)
        for name in original_param_names: is_A_param = name.startswith("A") and (name[1:].isdigit() or len(name)==1); logA_name = "log" + name if is_A_param else name; param_names_logA.append(logA_name); original_param_names_map[logA_name] = name
    except ValueError as e: return FitResult(model_name=model_name, parameters={}, success=False, message=f"Model setup error: {e}", rss=np.inf, n_datapoints=0, n_parameters=0, r_squared=np.nan, model_definition_args=model_definition_args)

    # Convert User Guesses/Bounds (A) to logA scale
    initial_guesses_logA = {}; parameter_bounds_logA = {} if parameter_bounds is not None else None; final_bounds_logA = {}
    if not all(p in initial_guesses for p in original_param_names): missing = [p for p in original_param_names if p not in initial_guesses]; return FitResult(model_name=model_name, model_definition_args=model_definition_args, parameters={}, success=False, message=f"Missing initial guesses for: {missing}", rss=np.inf, n_datapoints=0, n_parameters=len(param_names_logA), r_squared=np.nan)
    try: # Convert guesses
        for p_logA_name in param_names_logA: original_name = original_param_names_map[p_logA_name]; is_logA_param = p_logA_name.startswith("logA"); guess_val = initial_guesses[original_name]; initial_guesses_logA[p_logA_name] = np.log(guess_val) if is_logA_param else guess_val;
        if is_logA_param and guess_val <= 0: raise ValueError(f"Initial guess for {original_name} must be positive.")
    except (ValueError, TypeError, KeyError) as e_conv: return FitResult(model_name=model_name, model_definition_args=model_definition_args, parameters={}, success=False, message=f"Error converting initial guess: {e_conv}", rss=np.inf, n_datapoints=0, n_parameters=len(param_names_logA), r_squared=np.nan)
    if parameter_bounds_logA is not None:
            user_bound = parameter_bounds.get(original_name)
            if user_bound is not None:
                min_val, max_val = user_bound
                if is_logA_param:
                    min_log = np.log(min_val) if min_val is not None and min_val > 0 else -np.inf
                    max_log = np.log(max_val) if max_val is not None and max_val > 0 else np.inf
                    parameter_bounds_logA[p_logA_name] = (min_log, max_log)
                else:
                    parameter_bounds_logA[p_logA_name] = (min_val, max_val)

    # Define default bounds and merge
    default_bounds_logA = {}; # Define default bounds
    for p_name in param_names_logA:
        if p_name.startswith("Ea"): default_bounds_logA[p_name] = (1e3, 600e3)
        elif p_name.startswith("logA"): default_bounds_logA[p_name] = (np.log(1e-2), np.log(1e25))
        elif p_name.endswith("n") or p_name.endswith("m") or p_name.startswith("p1_") or p_name.startswith("p2_"): default_bounds_logA[p_name] = (0, 8)
        elif p_name == "initial_ratio_r": default_bounds_logA[p_name] = (1e-3, 1e3)
        else: default_bounds_logA[p_name] = (-np.inf, np.inf)
    final_bounds_logA = default_bounds_logA.copy(); # Merge user bounds
    if parameter_bounds_logA: final_bounds_logA.update(parameter_bounds_logA)
    initial_params_array_logA = np.array([initial_guesses_logA[p] for p in param_names_logA]); bounds_list_logA = None # Prepare bounds list
    methods_supporting_bounds = ['L-BFGS-B', 'TNC', 'SLSQP', 'trust-constr']
    if opt_method in methods_supporting_bounds and final_bounds_logA:
        try: bounds_list_logA = [(final_bounds_logA.get(p, (-np.inf, np.inf))[0] if np.isfinite(final_bounds_logA.get(p, (-np.inf, np.inf))[0]) else None, final_bounds_logA.get(p, (-np.inf, np.inf))[1] if np.isfinite(final_bounds_logA.get(p, (-np.inf, np.inf))[1]) else None) for p in param_names_logA]
        except Exception as e: return FitResult(model_name=model_name, model_definition_args=model_definition_args, parameters={}, success=False, message=f"Invalid bounds format: {e}", rss=np.inf, n_datapoints=0, n_parameters=len(param_names_logA), r_squared=np.nan)
    elif parameter_bounds and opt_method not in methods_supporting_bounds: warnings.warn(f"Optimizer {opt_method} ignores bounds.")

    # --- Attempt Optimization 1: Conversion Residuals ---
    print("--- Attempting optimization on CONVERSION residuals (weighted) ---")
    iteration_counter_conv = [0]
    opt_result = None # Initialize
    success_conv = False
    try:
        minimize_args_conv = (param_names_logA, datasets, model_name, model_definition_args, solver_options, callback, iteration_counter_conv)
        opt_result_conv = minimize(fun=_objective_function, x0=initial_params_array_logA, args=minimize_args_conv, method=opt_method, bounds=bounds_list_logA, options=opt_options)
        success_conv = opt_result_conv.success
        opt_result = opt_result_conv # Store result
    except Exception as e_conv:
        warnings.warn(f"Conversion-based optimization failed with exception: {e_conv}")
        success_conv = False

    used_rate_fallback = False
    # --- Attempt Optimization 2: Rate Residuals (Fallback) ---
    if not success_conv:
        warnings.warn("Conversion-based fit failed. Falling back to RATE-based optimization.")
        used_rate_fallback = True
        iteration_counter_rate = [0] # Reset counter for rate fit
        # Pre-calculate experimental rates
        exp_rates = []; exp_times_sec = []; valid_datasets_indices = []
        diff_options = {'window_length': 5, 'polyorder': 2}
        for i, ds in enumerate(datasets):
            if len(ds.time) < diff_options['window_length']: warnings.warn(f"Dataset {i} too short for rate calc."); continue
            time_sec = ds.time * 60.0; rate = numerical_diff(time_sec, ds.conversion, **diff_options); exp_rates.append(rate); exp_times_sec.append(time_sec); valid_datasets_indices.append(i)
        datasets_for_rate_fit = [datasets[i] for i in valid_datasets_indices]

        if not datasets_for_rate_fit: # Check if rate fit is possible
             return FitResult(model_name=model_name, model_definition_args=model_definition_args, parameters={}, success=False, message="Conversion fit failed & no datasets long enough for rate fit fallback.", rss=np.inf, n_datapoints=0, n_parameters=len(param_names_logA), r_squared=np.nan, used_rate_fallback=True)

        print("--- Optimizing on RATE residuals ---")
        try:
            minimize_args_rate = (param_names_logA, datasets_for_rate_fit, exp_rates, exp_times_sec, model_name, model_definition_args, solver_options, callback, iteration_counter_rate)
            # Use the same initial guess for the rate fit
            opt_result_rate = minimize(fun=_objective_function_rate, x0=initial_params_array_logA, args=minimize_args_rate, method=opt_method, bounds=bounds_list_logA, options=opt_options)
            opt_result = opt_result_rate # Use rate result
        except Exception as e_rate:
            # If rate fit also fails with exception, return the original conversion failure message
            fail_message = f"Conversion fit failed ({opt_result_conv.message if opt_result else 'Exception'}). Rate fit fallback also failed ({e_rate})."
            return FitResult(model_name=model_name, model_definition_args=model_definition_args, parameters={}, success=False, message=fail_message, rss=np.inf, n_datapoints=0, n_parameters=len(param_names_logA), r_squared=np.nan, used_rate_fallback=True)

    # --- Check final success ---
    if opt_result is None or not opt_result.success:
        fail_message = f"Optimization failed. Initial attempt: {opt_result_conv.message if opt_result_conv else 'Exception'}. "
        if used_rate_fallback: fail_message += f"Rate fallback attempt: {opt_result.message if opt_result else 'Exception'}."
        else: fail_message += "Rate fallback not attempted."
        return FitResult(model_name=model_name, model_definition_args=model_definition_args, parameters={}, success=False, message=fail_message, rss=np.inf, n_datapoints=0, n_parameters=len(param_names_logA), r_squared=np.nan, used_rate_fallback=used_rate_fallback)

    # --- Process successful results ---
    fitted_params_logA = dict(zip(param_names_logA, opt_result.x));
    n_params = len(param_names_logA)
    fitted_params_final = {}; # Convert logA to A
    for name_logA, value_logA in fitted_params_logA.items(): is_logA_param = name_logA.startswith("logA") and (name_logA[3:].startswith("A")); fitted_params_final[name_logA[3:] if is_logA_param else name_logA] = np.exp(value_logA) if is_logA_param else value_logA

    # Calculate FINAL stats based on UNWEIGHTED CONVERSION fit (Robustly)
    final_conversion_rss, n_total_datapoints_final, r_squared, aic, bic = _calculate_conversion_stats(
        datasets, fitted_params_logA, model_name, model_definition_args, solver_options
    )

    param_std_err_final = None; # Estimate/propagate std errors
    if opt_result.success and hasattr(opt_result, 'hess_inv'): # Error propagation logic
        hess_inv = None;
        if isinstance(opt_result.hess_inv, np.ndarray): hess_inv = opt_result.hess_inv
        elif hasattr(opt_result.hess_inv, 'todense'):
            try: hess_inv = opt_result.hess_inv.todense()
            except Exception: pass
        if hess_inv is not None:
            try:
                diag_hess_inv = np.diag(hess_inv)
                if np.all(diag_hess_inv > 0) and n_total_datapoints_final > n_params and np.isfinite(final_conversion_rss):
                    sigma_sq_est = final_conversion_rss / (n_total_datapoints_final - n_params); param_variances_logA = diag_hess_inv * sigma_sq_est
                    param_std_err_logA_arr = np.sqrt(param_variances_logA); param_std_err_logA = dict(zip(param_names_logA, param_std_err_logA_arr))
                    param_std_err_final = {}
                    for name_logA, std_err_logA in param_std_err_logA.items():
                        is_logA_param = name_logA.startswith("logA") and (name_logA[3:].startswith("A"))
                        if is_logA_param: original_A_key = name_logA[3:]; A_value = fitted_params_final[original_A_key]; param_std_err_final[original_A_key] = std_err_logA * A_value
                        else: param_std_err_final[name_logA] = std_err_logA
            except Exception as e: warnings.warn(f"Could not estimate/propagate std errors: {e}")
    for p_name, p_val in fitted_params_final.items(): # Parameter warnings
        if p_name.startswith("Ea"):
            if p_val < 5e3: warnings.warn(f"Fitted {p_name} ({p_val/1000:.1f} kJ/mol) is very low.")
            if p_val > 400e3: warnings.warn(f"Fitted {p_name} ({p_val/1000:.1f} kJ/mol) is very high.")
        elif p_name.startswith("A"):
             if p_val < 1e-1: warnings.warn(f"Fitted {p_name} ({p_val:.1e} 1/s) is very low.")
             if p_val > 1e20: warnings.warn(f"Fitted {p_name} ({p_val:.1e} 1/s) is very high.")
    initial_r = None; # Store initial_r
    if model_name == "A+B->C":
        fixed_r = model_definition_args.get('bimol_params', {}).get('initial_ratio_r')
        if fixed_r is not None and 'initial_ratio_r' not in param_names_logA: initial_r = fixed_r
        elif 'initial_ratio_r' in fitted_params_logA: initial_r = fitted_params_logA['initial_ratio_r']

    return FitResult(
        model_name=model_name, model_definition_args=model_definition_args,
        parameters=fitted_params_final, success=opt_result.success, message=opt_result.message,
        rss=final_conversion_rss, n_datapoints=n_total_datapoints_final, n_parameters=n_params,
        param_std_err=param_std_err_final, aic=aic, bic=bic, r_squared=r_squared,
        initial_ratio_r=initial_r,
        used_rate_fallback=used_rate_fallback # Store flag
    )

# --- Bootstrapping Worker Function ---
def _fit_on_resampled_data(
    datasets: List[KineticDataset], model_name: str, model_definition_args: Dict,
    best_fit_params_logA: Dict[str, float], parameter_bounds_logA: Optional[Dict[str, Tuple[float, float]]],
    solver_options: Dict, optimizer_options: Dict, iteration_index: int,
    end_callback_func: Optional[Callable[[int, str, Optional[Dict]], None]]
) -> Optional[Dict]:
    """Fits model (using logA) to one bootstrap sample using CONVERSION residual resampling. Returns dict {'params_logA': ..., 'stats': ...} or None."""
    callback_data = {}
    if end_callback_func:
        try:
            end_callback_func(iteration_index, "started", None)
        except Exception as e_cb:
            warnings.warn(f"Bootstrap start callback failed: {e_cb}")

    # --- Resampling logic (Weighted Conversion Residuals) ---
    resampled_datasets = []
    param_names_logA = list(best_fit_params_logA.keys())
    try:
        ode_func, _, initial_state_dim, params_template = get_model_info(model_name, **model_definition_args)
    except ValueError as e:
        warnings.warn(f"Bootstrap {iteration_index}: Model info error: {e}")
        return None

    initial_state = np.array([1.0, 0.0]) if model_name == "A->B->C" else np.zeros(initial_state_dim)
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
                valid_mask = np.isfinite(residuals) & np.isfinite(alpha_sim)
                all_residuals.extend(residuals[valid_mask])
                original_indices_map.extend([(i, k) for k, valid in enumerate(valid_mask) if valid])
                original_alphas.extend(ds.conversion[valid_mask])
            else:
                warnings.warn(f"Bootstrap {iteration_index}: Res calc sim length mismatch ds {i}.")
        except Exception as e_sim:
            warnings.warn(f"Bootstrap {iteration_index}: Res calc sim failed ds {i}: {e_sim}.")

    if not all_residuals:
        warnings.warn(f"Bootstrap {iteration_index}: No valid residuals.")
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
    weights /= np.sum(weights)

    try:
        resampled_indices_indices = np.random.choice(n_residuals, size=n_residuals, replace=True, p=weights)
    except ValueError as e_choice:
        warnings.warn(f"Bootstrap {iteration_index}: Weighted sampling failed ({e_choice}). Using uniform.")
        resampled_indices_indices = np.random.choice(n_residuals, size=n_residuals, replace=True)

    resampled_residuals_map = {orig_idx: [] for orig_idx in original_indices_map}
    for i, chosen_idx in enumerate(resampled_indices_indices):
        resampled_residuals_map[original_indices_map[chosen_idx]].append(centered_residuals[i])

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
                for k in range(len(ds.time)):
                    orig_idx_tuple = (i, k)
                    if orig_idx_tuple in resampled_residuals_map:
                        residuals_for_point = resampled_residuals_map[orig_idx_tuple]
                        if residuals_for_point:
                            synthetic_alpha[k] += np.mean(residuals_for_point)
                synthetic_alpha = np.clip(synthetic_alpha, 0.0, 1.0)
                resampled_datasets.append(KineticDataset(
                    time=ds.time, temperature=ds.temperature,
                    conversion=synthetic_alpha, heating_rate=ds.heating_rate
                ))
        except Exception as e_resample:
            warnings.warn(f"Bootstrap {iteration_index}: Error creating resampled ds {i}: {e_resample}")

    if not resampled_datasets:
        warnings.warn(f"Bootstrap {iteration_index}: Failed to create resampled datasets.")
        return None

    # --- Perturb Initial Guess (logA) ---
    perturbed_guesses_logA = {}
    noise_factor = 0.05
    abs_noise = {'Ea': 500, 'logA': 0.2}
    for name_logA, val_logA in best_fit_params_logA.items():
        noise_level = abs(val_logA) * noise_factor
        if name_logA.startswith("Ea"):
            noise_level += abs_noise['Ea']
        elif name_logA.startswith("logA"):
            noise_level += abs_noise['logA']
        perturbed_guesses_logA[name_logA] = val_logA + np.random.normal(0, noise_level)

    # --- Fit model (logA) to synthetic data ---
    output_dict = None
    try:
        opt_result_boot = minimize(
            fun=_objective_function,
            x0=np.array(list(perturbed_guesses_logA.values())),
            args=(param_names_logA, resampled_datasets, model_name, model_definition_args, solver_options, None, [0]),
            method=optimizer_options.get('method', 'L-BFGS-B'),
            bounds=parameter_bounds_logA,
            options=optimizer_options.get('options', {})
        )
        if opt_result_boot.success:
            fitted_params_logA_replicate = dict(zip(param_names_logA, opt_result_boot.x))
            # Calculate stats for this replicate
            stats_dict = calculate_stats_for_replicate(
                datasets=resampled_datasets,
                params_logA=fitted_params_logA_replicate,
                model_name=model_name,
                model_definition_args=model_definition_args,
                solver_options=solver_options
            )
            output_dict = {'params_logA': fitted_params_logA_replicate, 'stats': stats_dict}
    except Exception as e:
        warnings.warn(f"Bootstrap replicate {iteration_index} fit exception: {e}")
    return output_dict


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
) -> Optional[BootstrapResult]:
    """Performs bootstrap using logA scaling internally. Uses concurrent.futures."""
    if not fit_result.success:
        warnings.warn("Initial fit failed.")
        return None

    model_definition_args = getattr(fit_result, 'model_definition_args', None)
    if model_definition_args is None:
        warnings.warn("FitResult missing 'model_definition_args'. Cannot run bootstrap reliably.")
        return None

    model_name = fit_result.model_name
    best_fit_params_A = fit_result.parameters

    # Convert A to logA
    best_fit_params_logA = {}
    param_names_logA = []
    for name, val in best_fit_params_A.items():
        is_A_param = name.startswith("A") and (name[1:].isdigit() or len(name) == 1)
        logA_name = "log" + name if is_A_param else name
        best_fit_params_logA[logA_name] = np.log(val) if is_A_param else val
        param_names_logA.append(logA_name)

    # Convert A bounds to logA bounds
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

    max_workers = n_jobs if n_jobs > 0 else None
    print(f"Starting {n_iterations} bootstrap fits (logA scale) using concurrent.futures (max_workers={max_workers})...")

    results_data = [None] * n_iterations
    futures_list = []

    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            for i in range(n_iterations):
                future = executor.submit(
                    _fit_on_resampled_data,
                    datasets, model_name, model_definition_args, best_fit_params_logA,
                    parameter_bounds_logA, solver_options, optimizer_options,
                    i, end_callback
                )
                futures_list.append(future)

            print(f"\nSubmitted {len(futures_list)} tasks. Collecting results...")
            for i, future in enumerate(futures_list):
                try:
                    result = future.result(timeout=timeout_per_replicate)
                    results_data[i] = result
                except FuturesTimeoutError:
                    warnings.warn(f"Bootstrap replicate {i+1} timed out after {timeout_per_replicate}s.")
                except Exception as e:
                    warnings.warn(f"Bootstrap replicate {i+1} failed with exception: {type(e).__name__}: {e}")

    except Exception as e_pool:
        warnings.warn(f"Error during bootstrap pool execution: {type(e_pool).__name__}: {e_pool}")

    # Process collected results
    successful_params_logA = [res['params_logA'] for res in results_data if res]
    successful_replicate_stats = [res['stats'] for res in results_data if res]
    n_success = len(successful_params_logA)
    n_failed_or_timed_out = n_iterations - n_success

    print(f"\nBootstrap finished processing. {n_success}/{n_iterations} replicates successful ({n_failed_or_timed_out} failed/timed out).")
    if n_success == 0:
        return None

    param_distributions_logA = {
        p_name: np.array([params[p_name] for params in successful_params_logA])
        for p_name in param_names_logA
    }
    param_ci_logA = {}
    alpha_level = (1.0 - confidence_level) / 2.0
    for p in param_names_logA:
        dist = param_distributions_logA[p]
        lower, upper = (np.nan, np.nan)
        if len(dist) > 3 and np.std(dist) > 1e-9 * abs(np.mean(dist)) + 1e-12:
            lower, upper = np.percentile(dist, [alpha_level * 100.0, (1.0 - alpha_level) * 100.0])
        elif len(dist) > 0:
            lower = upper = np.mean(dist)
        param_ci_logA[p] = (lower, upper)

    param_distributions_final = {}
    param_ci_final = {}
    raw_parameter_list_A = None
    if return_replicate_params:
        raw_parameter_list_A = []
    for params_logA in successful_params_logA:
        params_A = {}
        for name_logA, val_logA in params_logA.items():
            is_logA = name_logA.startswith("logA") and (name_logA[3:].startswith("A"))
            params_A[name_logA[3:] if is_logA else name_logA] = np.exp(val_logA) if is_logA else val_logA
        if return_replicate_params:
            raw_parameter_list_A.append(params_A)
    for name_logA, dist_logA in param_distributions_logA.items():
        is_logA_param = name_logA.startswith("logA") and (name_logA[3:].startswith("A"))
        original_key = name_logA[3:] if is_logA_param else name_logA
        param_distributions_final[original_key] = np.exp(dist_logA) if is_logA_param else dist_logA
        ci_logA = param_ci_logA[name_logA]
        if np.isfinite(ci_logA[0]) and np.isfinite(ci_logA[1]):
            param_ci_final[original_key] = (np.exp(ci_logA[0]), np.exp(ci_logA[1])) if is_logA_param else ci_logA
        else:
            param_ci_final[original_key] = (np.nan, np.nan)

    # Calculate median parameters and stats
    median_params_logA = {}
    median_params_A = {}
    median_stats = None
    if n_success > 0:
        param_distributions_logA_temp = {
            p_name: np.array([params[p_name] for params in successful_params_logA])
            for p_name in param_names_logA
        }
        for p_name in param_names_logA:
            median_params_logA[p_name] = np.median(param_distributions_logA_temp[p_name])
        try:
            med_rss, med_n_pts, med_r2, med_aic, med_bic = _calculate_conversion_stats(
                datasets, median_params_logA, model_name, model_definition_args, solver_options
            )
            if np.isfinite(med_bic):
                median_stats = {'rss': med_rss, 'r_squared': med_r2, 'aic': med_aic, 'bic': med_bic}
        except Exception as e_med_sim:
            warnings.warn(f"Failed to calculate stats for median parameters: {e_med_sim}")
        for name_logA, val_logA in median_params_logA.items():
            is_logA = name_logA.startswith("logA") and (name_logA[3:].startswith("A"))
            median_params_A[name_logA[3:] if is_logA else name_logA] = np.exp(val_logA) if is_logA else val_logA

    # Rank replicates
    ranked_replicates = rank_replicates(successful_params_logA, successful_replicate_stats)

    return BootstrapResult(
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

# --- NEW Simulation Function ---
def simulate_kinetics( model_name: str, model_definition_args: Dict, kinetic_params: Dict, initial_alpha: float, temperature_program: Union[Callable, Tuple[np.ndarray, np.ndarray]], simulation_time_sec: Optional[np.ndarray] = None, solver_options: Dict = {}) -> PredictionResult:
    """ Simulates kinetic process given model, parameters (A), and temperature program. """
    params_A = kinetic_params; params_logA = {}; param_names_A = list(params_A.keys()); param_names_logA = [] # Convert A to logA
    for name, val in params_A.items():
        is_A_param = name.startswith("A") and (name[1:].isdigit() or len(name)==1)
        logA_name = "log" + name if is_A_param else name
        try:
            params_logA[logA_name] = np.log(val) if is_A_param else val
            param_names_logA.append(logA_name)
        except (ValueError, TypeError): raise ValueError(f"Cannot take log of parameter {name}={val}")
    try: ode_func, _, initial_state_dim, params_template = get_model_info(model_name, **model_definition_args) # Get model info
    except Exception as e: raise ValueError(f"Model setup error during simulation: {e}")
    # Prepare temperature program
    if callable(temperature_program): temp_func = temperature_program; t_eval_sec = simulation_time_sec;
    elif isinstance(temperature_program, tuple) and len(temperature_program) == 2: t_prog_sec, temp_prog_K = temperature_program; temp_func = get_temperature_interpolator(t_prog_sec, temp_prog_K); t_eval_sec = simulation_time_sec if simulation_time_sec is not None else t_prog_sec
    else: raise TypeError("temp program invalid type.")
    if t_eval_sec is None: raise ValueError("simulation_time_sec required if temp program is callable, or defaults to program times.")
    temp_eval_K = temp_func(t_eval_sec)
    # Define initial state
    if model_name == "A->B->C": initial_state = np.array([1.0 - initial_alpha, 0.0])
    else: initial_state = np.array([initial_alpha] * initial_state_dim)
    # Run simulation
    t_pred_sec, alpha_pred = _simulate_single_dataset(t_eval=t_eval_sec, temp_func=temp_func, ode_system=ode_func, initial_state=initial_state, params_template=params_template, current_params_logA=params_logA, solver_options=solver_options)
    return PredictionResult(time=t_pred_sec, temperature=temp_eval_K, conversion=alpha_pred, conversion_ci=None)

# --- Prediction Function ---
def predict_conversion( kinetic_description: Union[FitResult, IsoResult], temperature_program: Union[Callable, Tuple[np.ndarray, np.ndarray]], simulation_time_sec: Optional[np.ndarray] = None, initial_alpha: float = 0.0, solver_options: Dict = {}, bootstrap_result: Optional[BootstrapResult] = None) -> PredictionResult:
    """ Predicts conversion using fitted parameters. Calculates CI if bootstrap results provided. """
    if isinstance(kinetic_description, IsoResult): warnings.warn("Prediction from IsoResult not implemented."); return PredictionResult(time=np.array([]), temperature=np.array([]), conversion=np.array([]))
    elif isinstance(kinetic_description, FitResult):
        fit_result = kinetic_description
        if not fit_result.success: warnings.warn("Cannot predict from unsuccessful fit."); return PredictionResult(time=np.array([]), temperature=np.array([]), conversion=np.array([]))
        # **** Retrieve model_definition_args from fit_result ****
        model_definition_args = getattr(fit_result, 'model_definition_args', None)
        if model_definition_args is None: warnings.warn("FitResult missing 'model_definition_args'. Prediction may fail or be incorrect."); model_definition_args = {} # Or raise error
        model_name = fit_result.model_name; params_A = fit_result.parameters

        # Simulate base prediction
        base_prediction = simulate_kinetics(model_name=model_name, model_definition_args=model_definition_args, kinetic_params=params_A, initial_alpha=initial_alpha, temperature_program=temperature_program, simulation_time_sec=simulation_time_sec, solver_options=solver_options)
        alpha_lower_ci, alpha_upper_ci = None, None # Calculate CIs using bootstrap_result and simulate_kinetics
        if bootstrap_result is not None and bootstrap_result.model_name == model_name and bootstrap_result.n_iterations > 0:
            n_boot_iter = bootstrap_result.n_iterations; t_eval_sec = base_prediction.time
            all_boot_alphas = np.full((n_boot_iter, len(t_eval_sec)), np.nan); param_dist_A = bootstrap_result.parameter_distributions
            param_names_A = list(params_A.keys()) # Get names from original fit
            # TODO: Parallelize this loop if needed
            for i in range(n_boot_iter):
                # Ensure parameter distribution dictionary has all required keys
                if not all(p in param_dist_A for p in param_names_A):
                    warnings.warn(f"Bootstrap parameter distribution missing keys for replicate {i}. Skipping CI calculation.")
                    all_boot_alphas = np.array([]) # Ensure percentile calculation fails gracefully
                    break
                # Ensure the distribution for this iteration has the correct index
                if i >= len(param_dist_A[param_names_A[0]]): # Check length using first param name
                    warnings.warn(f"Bootstrap parameter distribution index {i} out of bounds. Skipping CI calculation.")
                    all_boot_alphas = np.array([])
                    break

                boot_params_A = {p: param_dist_A[p][i] for p in param_names_A}
                try:
                    boot_pred = simulate_kinetics(model_name=model_name, model_definition_args=model_definition_args, kinetic_params=boot_params_A, initial_alpha=initial_alpha, temperature_program=temperature_program, simulation_time_sec=t_eval_sec, solver_options=solver_options)
                    if len(boot_pred.conversion) == len(t_eval_sec): all_boot_alphas[i, :] = boot_pred.conversion
                except Exception as e_boot_sim: warnings.warn(f"Sim failed for bootstrap replicate {i}: {e_boot_sim}")

            # Only calculate percentiles if simulations were successful
            if all_boot_alphas.size > 0 and np.any(np.isfinite(all_boot_alphas)):
                alpha_ci_level = (1.0 - bootstrap_result.confidence_level) / 2.0
                with warnings.catch_warnings():
                     warnings.simplefilter("ignore", category=RuntimeWarning) # Ignore warnings from nan slices
                     alpha_lower_ci = np.nanpercentile(all_boot_alphas, alpha_ci_level * 100.0, axis=0)
                     alpha_upper_ci = np.nanpercentile(all_boot_alphas, (1.0 - alpha_ci_level) * 100.0, axis=0)
            else:
                 warnings.warn("Could not calculate CIs; no successful bootstrap simulations or all results were NaN.")


        base_prediction.conversion_ci = (alpha_lower_ci, alpha_upper_ci) if alpha_lower_ci is not None else None
        return base_prediction
    else: raise TypeError("kinetic_description must be FitResult or IsoResult.")

# --- New Helper Functions ---
def calculate_stats_for_replicate(datasets, params_logA, model_name, model_definition_args, solver_options):
    """Calculates RSS, R-squared, AIC, and BIC for a replicate."""
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
    """Calculates median parameters and stats for the original datasets."""
    median_params_logA = {
        p_name: np.median([params[p_name] for params in successful_params_logA])
        for p_name in successful_params_logA[0].keys()
    }
    rss, n_pts, r_squared, aic, bic = _calculate_conversion_stats(
        datasets, median_params_logA, model_name, model_definition_args, solver_options
    )
    return median_params_logA, {
        'rss': rss,
        'r_squared': r_squared,
        'aic': aic,
        'bic': bic,
        'n_points': n_pts
    }

def rank_replicates(successful_params_logA, successful_replicate_stats):
    """Ranks replicates by RSS and returns a sorted list."""
    ranked_replicates = []
    for i, (params_logA, stats) in enumerate(zip(successful_params_logA, successful_replicate_stats)):
        params_A = {
            name_logA[3:] if name_logA.startswith("logA") else name_logA: np.exp(val_logA) if name_logA.startswith("logA") else val_logA
            for name_logA, val_logA in params_logA.items()
        }
        ranked_replicates.append({
            'rank': 0,  # Placeholder
            'source': f'replicate_{i+1}',
            'parameters': params_A,
            'stats': stats,
            'sort_metric': stats.get('rss', np.inf)
        })
    ranked_replicates.sort(key=lambda x: x['sort_metric'])
    for rank, item in enumerate(ranked_replicates):
        item['rank'] = rank + 1
    return ranked_replicates

def rank_models(fit_results: List[FitResult], rank_by: str = 'bic') -> List[Dict]:
    """Ranks a list of FitResult objects based on BIC (default) or AICc."""
    valid_fits = []
    for res in fit_results:
        if not res.success:
            continue
        sort_key_val = getattr(res, rank_by.lower(), np.inf)
        if not np.isfinite(sort_key_val):
            warnings.warn(f"Model '{res.model_name}' missing valid '{rank_by}' value for ranking.")
            continue
        valid_fits.append({
            'rank': 0,
            'model_name': res.model_name,
            'parameters': res.parameters,
            'stats': {
                'rss': res.rss,
                'r_squared': res.r_squared,
                'aic': res.aic,
                'bic': res.bic,
                'n_params': res.n_parameters,
                'n_points': res.n_datapoints
            },
            'sort_key': sort_key_val,
            'n_params': res.n_parameters
        })
    valid_fits.sort(key=lambda x: (x['sort_key'], x['n_params']))
    for rank, item in enumerate(valid_fits):
        item['rank'] = rank + 1
    return valid_fits

def discover_kinetic_models(
    datasets: List[KineticDataset],
    models_to_try: List[Dict],
    initial_guesses_pool: Dict[str, Dict],
    parameter_bounds_pool: Optional[Dict[str, Dict]] = None,
    solver_options: Dict = {},
    optimizer_options: Dict = {},
    score_weights: Optional[Dict[str, float]] = None  # Keep score_weights
) -> List[Dict]:
    """
    Fits multiple kinetic models to the data and ranks them using a combined score.
    Uses conversion objective first, falls back to rate objective if needed.
    """
    all_fit_results = []
    print(f"--- Starting Kinetic Model Discovery ({len(models_to_try)} models) ---")

    for model_info in models_to_try:
        name = model_info.get('name')
        m_type = model_info.get('type')
        def_args = model_info.get('def_args')
        if not name or not m_type or def_args is None:
            warnings.warn(f"Skipping invalid model definition: {model_info}")
            continue
        print(f"\n--- Fitting Model: {name} ---")
        guesses = initial_guesses_pool.get(name)
        if guesses is None:
            warnings.warn(f"No initial guesses for model '{name}'. Skipping.")
            continue
        bounds = parameter_bounds_pool.get(name) if parameter_bounds_pool else None

        # Removed optimize_on_rate from this call
        fit_res = fit_kinetic_model(
            datasets=datasets, model_name=m_type, model_definition_args=def_args,
            initial_guesses=guesses, parameter_bounds=bounds,
            solver_options=solver_options, optimizer_options=optimizer_options,
            callback=None  # No detailed callback during discovery loop
        )
        if fit_res.success:
            print(f"Model '{name}' fit successful.")
            fit_res.model_name = name  # Overwrite internal type name with user name
            all_fit_results.append(fit_res)
        else:
            print(f"Model '{name}' fit failed: {fit_res.message}")

    print(f"\n--- Model Discovery Finished ---")
    if not all_fit_results:
        print("No models fitted successfully.")
        return []

    # Rank using the new function
    ranked_list = rank_models(all_fit_results, score_weights=score_weights)
    print(f"Ranking {len(ranked_list)} successful models by combined score...")

    return ranked_list

# --- NEW Model Discovery Functions ---
def rank_models(
    fit_results: List[FitResult],
    score_weights: Optional[Dict[str, float]] = None
) -> List[Dict]:
    """
    Ranks a list of FitResult objects based on a combined score considering
    BIC/AICc, R-squared, RSS, and number of parameters.
    """
    if not fit_results:
        return []

    # --- Define Default Weights ---
    default_weights = {'bic': 0.4, 'r_squared': 0.4, 'rss': 0.1, 'n_params': 0.1}
    weights = score_weights if score_weights and np.isclose(sum(score_weights.values()), 1.0) else default_weights

    # --- Prepare data for ranking ---
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
            warnings.warn(f"Model '{res.model_name}' has invalid stats for ranking. Skipping.")
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

    # --- Calculate Normalization Ranges ---
    ranges = {key: (np.min(stat_values[key]), np.ptp(stat_values[key])) for key in ['rss', 'aic', 'bic', 'n_params']}
    finite_r2 = [r for r in stat_values['r_squared'] if np.isfinite(r)]
    ranges['r_squared'] = (np.min(finite_r2), np.max(finite_r2), np.ptp(finite_r2)) if finite_r2 else (0, 0, 0)

    # --- Calculate Score for each model ---
    for item in valid_fits_data:
        stats = item['stats']
        score = 0.0
        for key, weight in weights.items():
            if key == 'r_squared':
                min_r2_norm, max_r2_norm, range_r2_norm = ranges['r_squared']
                val = stats.get('r_squared', -np.inf)
                if not np.isfinite(val):
                    norm_val = 1.0  # Penalize invalid R2 maximally
                elif range_r2_norm > 1e-9:
                    norm_val = (max_r2_norm - val) / range_r2_norm  # Higher R2 -> lower score component
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

    # --- Sort by Score (ascending) ---
    valid_fits_data.sort(key=lambda x: x['score'])
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
    """
    Fits multiple kinetic models to the data and ranks them using a combined score.
    """
    all_fit_results = []
    print(f"--- Starting Kinetic Model Discovery ({len(models_to_try)} models) ---")

    for model_info in models_to_try:
        name = model_info.get('name')
        m_type = model_info.get('type')
        def_args = model_info.get('def_args')
        if not name or not m_type or def_args is None:
            warnings.warn(f"Skipping invalid model definition: {model_info}")
            continue
        print(f"\n--- Fitting Model: {name} ---")
        guesses = initial_guesses_pool.get(name)
        if guesses is None:
            warnings.warn(f"No initial guesses for model '{name}'. Skipping.")
            continue
        bounds = parameter_bounds_pool.get(name) if parameter_bounds_pool else None

        fit_res = fit_kinetic_model(
            datasets=datasets, model_name=m_type, model_definition_args=def_args,
            initial_guesses=guesses, parameter_bounds=bounds,
            solver_options=solver_options, optimizer_options=optimizer_options,
            callback=None  # No detailed callback during discovery loop
        )
        if fit_res.success:
            print(f"Model '{name}' fit successful.")
            fit_res.model_name = name
            all_fit_results.append(fit_res)
        else:
            print(f"Model '{name}' fit failed: {fit_res.message}")

    print(f"\n--- Model Discovery Finished ---")
    if not all_fit_results:
        print("No models fitted successfully.")
        return []

    ranked_list = rank_models(all_fit_results, score_weights=score_weights)
    print(f"Ranking {len(ranked_list)} successful models by combined score...")

    return ranked_list


# **** MODIFIED run_bootstrap: Uses rank_replicates ****
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
    ) -> Optional[BootstrapResult]:
    """ Performs bootstrap using logA scaling internally. Uses concurrent.futures. Returns replicate stats, ranked replicates, median params/stats, and optionally raw parameters. """
    # ... (Setup code: check fit_result, get model_def_args, convert params/bounds to logA - remains the same) ...
    if not fit_result.success: warnings.warn("Initial fit failed."); return None
    model_definition_args = getattr(fit_result, 'model_definition_args', None)
    if model_definition_args is None: warnings.warn("FitResult missing 'model_definition_args'. Cannot run bootstrap reliably."); return None
    model_name = fit_result.model_name; best_fit_params_A = fit_result.parameters
    best_fit_params_logA = {}; param_names_logA = []
    try:
        for name, val in best_fit_params_A.items(): is_A_param = name.startswith("A") and (name[1:].isdigit() or len(name)==1); logA_name = "log" + name if is_A_param else name; best_fit_params_logA[logA_name] = np.log(val) if is_A_param else val; param_names_logA.append(logA_name)
    except (ValueError, TypeError) as e_conv: warnings.warn(f"Cannot take log of parameter for bootstrap: {e_conv}."); return None
    parameter_bounds_logA = None # Convert A bounds to logA bounds
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
                 else: # **** ADDED ELSE BLOCK ****
                      # Keep original bounds for non-logA params (like Ea)
                      parameter_bounds_logA[name_logA] = (min_A, max_A)
            else: # If original key not in user bounds, use default infinite bounds
                 parameter_bounds_logA[name_logA] = (-np.inf, np.inf)
    max_workers = n_jobs if n_jobs is not None and n_jobs > 0 else None
    print(f"Starting {n_iterations} bootstrap fits (logA scale) using concurrent.futures (max_workers={max_workers})...")
    results_data = [None] * n_iterations; futures_list = []; n_submitted = 0
    try: # Run parallel fits
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

    # --- Process collected results ---
    successful_params_logA = []; successful_replicate_stats = []; n_failed_or_timed_out = 0
    print("\nProcessing received results...")
    for i, p_data in enumerate(results_data):
        if p_data is None or not isinstance(p_data, dict) or 'params_logA' not in p_data or 'stats' not in p_data: n_failed_or_timed_out += 1
        else:
            if all(key in p_data['params_logA'] for key in param_names_logA): successful_params_logA.append(p_data['params_logA']); successful_replicate_stats.append(p_data['stats'])
            else: warnings.warn(f"Replicate {i+1} returned unexpected dict keys."); n_failed_or_timed_out += 1
    n_success = len(successful_params_logA)
    print(f"\nBootstrap finished processing. {n_success}/{n_iterations} replicates successful ({n_failed_or_timed_out} failed/timed out).")
    if n_success == 0: return None
    elif n_success < n_iterations * 0.75: warnings.warn(f"Low success rate ({n_success}/{n_iterations}). Results may be less reliable.")

    # --- Calculate Median Parameters and Stats ---
    median_params_logA = {}; median_params_A = {}; median_stats = None
    if n_success > 0:
        param_distributions_logA_temp = {p_name: np.array([params[p_name] for params in successful_params_logA]) for p_name in param_names_logA}
        for p_name in param_names_logA: median_params_logA[p_name] = np.median(param_distributions_logA_temp[p_name])
        try: # Simulate with median parameters on ORIGINAL data
            med_rss, med_n_pts, med_r2, med_aic, med_bic = _calculate_conversion_stats(datasets, median_params_logA, model_name, model_definition_args, solver_options)
            if np.isfinite(med_bic): median_stats = {'rss': med_rss, 'r_squared': med_r2, 'aic': med_aic, 'bic': med_bic}
        except Exception as e_med_sim: warnings.warn(f"Failed to calculate stats for median parameters: {e_med_sim}")
        for name_logA, val_logA in median_params_logA.items(): is_logA = name_logA.startswith("logA") and (name_logA[3:].startswith("A")); median_params_A[name_logA[3:] if is_logA else name_logA] = np.exp(val_logA) if is_logA else val_logA

    # --- Rank Replicates ---
    ranked_replicates = rank_replicates(successful_params_logA, successful_replicate_stats)

    # --- Calculate CIs and convert final distributions ---
    param_distributions_logA = {p_name: np.array([params[p_name] for params in successful_params_logA]) for p_name in param_names_logA}
    param_ci_logA = {}
    alpha_level = (1.0 - confidence_level) / 2.0 # Removed semicolon

    # **** Corrected loop structure ****
    for p in param_names_logA:
        dist = param_distributions_logA[p]
        lower, upper = (np.nan, np.nan) # Default CIs
        if len(dist) > 3 and np.std(dist) > 1e-9 * abs(np.mean(dist)) + 1e-12:
            # Calculate percentiles if enough points and variance
            lower, upper = np.percentile(dist, [alpha_level * 100.0, (1.0 - alpha_level) * 100.0])
        elif len(dist) > 0:
            # Use mean if too few points or zero variance
            lower = upper = np.mean(dist)
        # else: keep as nan if dist is empty (shouldn't happen if n_success > 0)
        param_ci_logA[p] = (lower, upper)

    param_distributions_final = {}; param_ci_final = {}
    raw_parameter_list_A = None
    if return_replicate_params: raw_parameter_list_A = [] # This will be unsorted unless we re-sort based on rank
    for params_logA in successful_params_logA: # Convert successful logA params back to A
        params_A = {}
        for name_logA, val_logA in params_logA.items(): is_logA = name_logA.startswith("logA") and (name_logA[3:].startswith("A")); params_A[name_logA[3:] if is_logA else name_logA] = np.exp(val_logA) if is_logA else val_logA
        if return_replicate_params: raw_parameter_list_A.append(params_A) # Currently unsorted list
    for name_logA, dist_logA in param_distributions_logA.items(): # Convert distributions and CIs
        is_logA_param = name_logA.startswith("logA") and (name_logA[3:].startswith("A")); original_key = name_logA[3:] if is_logA_param else name_logA; param_distributions_final[original_key] = np.exp(dist_logA) if is_logA_param else dist_logA; ci_logA = param_ci_logA[name_logA];
        if np.isfinite(ci_logA[0]) and np.isfinite(ci_logA[1]): param_ci_final[original_key] = (np.exp(ci_logA[0]), np.exp(ci_logA[1])) if is_logA_param else ci_logA
        else: param_ci_final[original_key] = (np.nan, np.nan)

    return BootstrapResult(
        model_name=model_name,
        parameter_distributions=param_distributions_final,
        parameter_ci=param_ci_final,
        n_iterations=n_success,
        confidence_level=confidence_level,
        # replicate_stats=successful_replicate_stats, # Store list of UNRANKED stats dicts
        raw_parameter_list=raw_parameter_list_A, # Store optional list of UNRANKED param dicts
        ranked_replicates=ranked_replicates, # Store RANKED list
        median_parameters=median_params_A, # Store median params
        median_stats=median_stats  # Store stats for median params on original data
    )

