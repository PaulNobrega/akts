import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
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
from .models import get_model_info, get_log_param_names, F_ALPHA_MODELS
from .utils import (get_temperature_interpolator, calculate_aic, calculate_bic, R_GAS, numerical_diff,
                    calculate_akaike_weights, calculate_adjusted_r_squared)
from .isoconversional import run_friedman, run_kas, run_ofw

# --- Helper: decode a possibly-"log"-prefixed internal parameter name ---
# Encoding is done exclusively by _to_logA_name below (name -> "log"+name for
# params in a model's get_log_param_names() set). Decoding here only needs to
# reverse that fixed, single-source encoding — it does not re-derive which
# parameters are log-scaled from the string shape itself.
def _split_logA_name(name_logA: str) -> Tuple[bool, str]:
    """Returns (is_logA_param, original_name) for an internal parameter name."""
    if name_logA.startswith("logA") and name_logA[3:].startswith("A"):
        return True, name_logA[3:]
    return False, name_logA

def _to_logA_name(name: str, log_param_names) -> str:
    """Encodes an original (A-scale) parameter name for a model, given the set
    of parameter names that model fits on a log scale (see
    models.get_log_param_names)."""
    return "log" + name if name in log_param_names else name

# --- Helper to prepare the full ODE parameter dictionary ---
def _prepare_full_params_for_ode(
    params_template: Dict,
    current_params_logA: Dict
    ) -> Dict:
    """Merges template and current logA params, converting logA->A."""
    full_params_ode = copy.deepcopy(params_template)
    param_mapping = full_params_ode.pop('_param_mapping', {})

    for name_logA, value_logA in current_params_logA.items():
        is_logA_param, original_name = _split_logA_name(name_logA)
        value_to_use = value_logA

        if is_logA_param:
            # Safely calculate exp, handle potential overflow/invalid values
            if not np.isfinite(value_logA):
                warnings.warn(f"Invalid non-finite value for {name_logA}. Using default large A.")
                value_to_use = 1e30 # Assign a large value or handle as error
            else:
                # Only the upper end can overflow; optimizers probe extremes, so clamp silently.
                value_logA = min(value_logA, 700.0)

                try:
                    value_to_use = np.exp(value_logA)
                except (OverflowError, RuntimeWarning):
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


# f(alpha) is 0 (Avrami) or infinite (diffusion) at alpha=0, so a simulation started at
# exactly 0 never moves. Seed single-step models with a negligible conversion instead.
ALPHA_SEED = 1e-6


def _initial_state(model_name: str, state_dim: int, initial_alpha: float = 0.0) -> np.ndarray:
    if model_name == "A->B->C":
        return np.array([1.0 - initial_alpha, 0.0])
    if model_name == "single_step":
        return np.full(state_dim, max(initial_alpha, ALPHA_SEED))
    return np.full(state_dim, initial_alpha)


EA_SCALE = 1e4


class _ArrheniusReparam:
    """Maps (Ea, lnA) pairs to (Ea/EA_SCALE, ln k(T_ref)) for the optimizer.

    Ea and lnA are almost perfectly correlated and differ in magnitude by ~1e4, which makes
    gradient optimizers stall at the initial guess. ln k at the data's mean temperature is
    nearly independent of Ea. The map is linear: x_external = J @ y_internal.
    """

    def __init__(self, param_names_logA: List[str], datasets: List[KineticDataset]):
        inv_T = np.concatenate([1.0 / np.asarray(ds.temperature, float) for ds in datasets if len(ds.temperature)])
        self.c = 1.0 / (R_GAS * (1.0 / np.mean(inv_T)))  # 1/(R*T_ref)
        self.pairs = []
        for i, name in enumerate(param_names_logA):
            if name.startswith("Ea"):
                partner = "logA" + name[2:]
                if partner in param_names_logA:
                    self.pairs.append((i, param_names_logA.index(partner)))
        n = len(param_names_logA)
        self.J = np.eye(n)
        for i, j in self.pairs:
            self.J[i, i] = EA_SCALE
            self.J[j, i] = EA_SCALE * self.c

    def to_external(self, y: np.ndarray) -> np.ndarray:
        return self.J @ np.asarray(y, float)

    def to_internal(self, x: np.ndarray) -> np.ndarray:
        return np.linalg.solve(self.J, np.asarray(x, float))

    def bounds_to_internal(self, bounds: Optional[List[Tuple]]) -> Optional[List[Tuple]]:
        if bounds is None:
            return None
        out = list(bounds)
        for i, j in self.pairs:
            ea_lo, ea_hi = bounds[i]
            la_lo, la_hi = bounds[j]
            out[i] = (None if ea_lo is None else ea_lo / EA_SCALE, None if ea_hi is None else ea_hi / EA_SCALE)
            lo = None if (la_lo is None or ea_hi is None) else la_lo - ea_hi * self.c
            hi = None if (la_hi is None or ea_lo is None) else la_hi - ea_lo * self.c
            out[j] = (lo, hi)
        return out

    def wrap(self, objective: Callable) -> Callable:
        return lambda y, *args: objective(self.to_external(y), *args)

    def wrap_residual(self, residual_func: Callable) -> Callable:
        """Wrap residual vector function for least_squares.

        Similar to wrap() but for residual vectors instead of scalar objectives.
        Applies the Arrhenius reparameterization transform.

        Parameters
        ----------
        residual_func : Callable
            Function that takes params_array_logA (external) and returns residual vector

        Returns
        -------
        Callable
            Wrapped function that takes y_internal and returns residual vector
        """
        return lambda y, *args: residual_func(self.to_external(y), *args)

    def coarse_start(self, y0: np.ndarray, objective: Callable, args: tuple,
                     bounds: Optional[List[Tuple]], deadline: Optional[float] = None) -> np.ndarray:
        """Grid-scan each ln k(T_ref) so the optimizer starts inside the basin.

        Far from the answer the model is fully reacted (or not at all) everywhere, the
        objective is flat, and gradient methods stop at the initial guess. Runs
        25 evaluations per Arrhenius pair, so multi-step models (e.g. A->B->C, two
        pairs) cost proportionally more -- `deadline` (a time.time()-style
        timestamp) lets a caller cut the scan short if it is eating into a shared
        wall-clock budget, using whatever the scan already found.
        """
        y = np.array(y0, float)
        f = self.wrap(objective)
        for _, j in self.pairs:
            if deadline is not None and time.time() > deadline:
                break
            lo, hi = y[j] - 12.0, y[j] + 12.0
            if bounds is not None:
                blo, bhi = bounds[j]
                lo = lo if blo is None else max(lo, blo)
                hi = hi if bhi is None else min(hi, bhi)
            if hi <= lo:
                continue
            best_val, best_lk = f(y, *args), y[j]
            for lk in np.linspace(lo, hi, 25):
                if deadline is not None and time.time() > deadline:
                    break
                trial = y.copy(); trial[j] = lk
                val = f(trial, *args)
                if val < best_val:
                    best_val, best_lk = val, lk
            y[j] = best_lk
        return y


class _RhsBudgetExceeded(Exception):
    pass


def _with_eval_budget(fun: Callable, max_evals: int) -> Callable:
    count = [0]
    def wrapped(t, y, *args):
        count[0] += 1
        if count[0] > max_evals:
            raise _RhsBudgetExceeded(f"ODE solve exceeded {max_evals} RHS evaluations")
        return fun(t, y, *args)
    return wrapped


# --- Fast Closed-Form Simulation for Isothermal Conditions ---
def _simulate_single_dataset_closed_form(
    t_eval: np.ndarray,
    T_const: float,
    alpha_of_kt_func: Callable,
    Ea: float,
    A: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fast closed-form evaluation for isothermal datasets.

    For constant temperature T, many kinetic models have analytic solutions
    α(t) = g⁻¹(kt) where k = A·exp(-Ea/RT).

    Parameters
    ----------
    t_eval : np.ndarray
        Time points for evaluation
    T_const : float
        Constant temperature in Kelvin
    alpha_of_kt_func : Callable
        Function α(kt) that maps dimensionless time kt → α
    Ea : float
        Activation energy in J/mol
    A : float
        Pre-exponential factor in s⁻¹

    Returns
    -------
    t_eval : np.ndarray
        Time array (unchanged)
    alpha : np.ndarray
        Conversion array, clipped to [0, 1]

    Notes
    -----
    This is ~1000× faster than ODE integration for isothermal conditions.
    Numerical overflow is prevented by clipping exp(-Ea/RT) argument.
    """
    from .utils import R_GAS

    # Calculate rate constant k = A·exp(-Ea/RT)
    # Prevent overflow: if Ea/(RT) > 700, k ≈ 0
    exp_arg = -Ea / (R_GAS * T_const)
    if exp_arg < -700:
        k = 0.0
    else:
        k = A * np.exp(exp_arg)

    # Dimensionless time kt
    kt = k * t_eval

    # Evaluate closed-form solution
    alpha = alpha_of_kt_func(kt)

    return t_eval, np.clip(alpha, 0.0, 1.0)


# --- Simulation Function with Fallback ---
def _simulate_single_dataset(
    t_eval: np.ndarray,
    temp_func: Callable,
    ode_system: OdeSystemCallable,
    initial_state: np.ndarray,
    params_template: Dict, # Template containing functions, param dict structures
    current_params_logA: Dict, # Current kinetic params (Ea, logA, etc.)
    solver_options: Dict = {},
    return_final_state: bool = False
    ) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Simulates conversion with solver fallback. Merges template and logA params.
    Returns (time_array, alpha_array) or (time_array, alpha_array, final_state_array)
    if return_final_state is True.
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
            # Cap work per solve so pathological parameter sets fail fast instead of hanging.
            sol = solve_ivp(
                fun=_with_eval_budget(ode_system, solver_options.get('max_rhs_evals', 20000)),
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
        if return_final_state:
            final_state_sorted = state_sim_sorted[:, -1]
            return t_eval, alpha_sim_unsorted, final_state_sorted
        else:
            return t_eval, alpha_sim_unsorted
    else:
        warnings.warn(f"Both solvers failed or simulation error occurred."); alpha_nan = np.full((len(t_eval),), np.nan)
        if return_final_state:
            return t_eval, alpha_nan, initial_state
        else:
            return t_eval, alpha_nan

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
    total_weighted_rss = 0.0
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
            for name_logA, val_logA in current_params_logA.items():
                 is_logA, original_name = _split_logA_name(name_logA)
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

    initial_state = _initial_state(model_name, initial_state_dim)

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

    # Return weighted RSS for optimizer.
    # AIC/BIC calculations in fit_kinetic_model might be less meaningful
    # if they use this weighted RSS directly without accounting for weights.
    # For now, we prioritize getting better parameters.
    return total_weighted_rss


# --- Residual Vector Function for least_squares ---
def _residual_vector_function(
    params_array_logA: np.ndarray,
    param_names_logA: List[str],
    datasets: List[KineticDataset],
    model_name: str,
    model_definition_args: Dict,
    solver_options: Dict,
    use_closed_form: bool = False,
    alpha_of_kt_func: Optional[Callable] = None,
    callback_func: Optional[Callable] = None,
    iteration_counter: List[int] = [0]
) -> np.ndarray:
    """
    Returns flattened residual vector for scipy.optimize.least_squares.

    This function is structurally similar to _objective_function but returns
    per-point residuals instead of scalar RSS. Used for TRF optimization with
    closed-form models or ODE models.

    Parameters
    ----------
    params_array_logA : np.ndarray
        Parameter array in logA scale
    param_names_logA : List[str]
        Parameter names corresponding to params_array_logA
    datasets : List[KineticDataset]
        Experimental datasets
    model_name : str
        Model identifier ('single_step', 'A_B_C', etc.)
    model_definition_args : Dict
        Model-specific arguments (e.g., {'f_alpha_model': 'F1'})
    solver_options : Dict
        ODE solver options (ignored if use_closed_form=True)
    use_closed_form : bool, optional
        If True, use closed-form simulation for isothermal datasets
    alpha_of_kt_func : Callable, optional
        Closed-form α(kt) function (required if use_closed_form=True)
    callback_func : Callable, optional
        Callback function for progress tracking
    iteration_counter : List[int], optional
        Mutable counter for iteration tracking

    Returns
    -------
    np.ndarray
        Flattened residual vector with transition weighting applied via sqrt(weight)

    Notes
    -----
    Transition weighting (10× for 0.05 < α < 0.95) is applied as sqrt(weight) on
    residuals rather than weight on RSS, so least_squares minimizes weighted sum
    of squared residuals correctly.
    """
    from .utils import is_isothermal

    residuals_list = []
    current_params_logA = dict(zip(param_names_logA, params_array_logA))

    # --- Weighting Parameters (same as _objective_function) ---
    weight_transition = 10.0
    weight_baseline_plateau = 1.0
    alpha_lower = 0.05
    alpha_upper = 0.95
    # --------------------------

    # Callback
    if callback_func:
        try:
            current_params_A = {}
            for name_logA, val_logA in current_params_logA.items():
                is_logA, original_name = _split_logA_name(name_logA)
                param_val_A = np.exp(val_logA) if is_logA else val_logA
                current_params_A[original_name] = param_val_A
            callback_func(iteration_counter[0], current_params_A)
        except Exception as e_cb:
            warnings.warn(f"Residual callback failed: {e_cb}")
    iteration_counter[0] += 1

    # Extract kinetic parameters
    Ea = current_params_logA.get('Ea')
    logA = current_params_logA.get('logA')
    A = np.exp(logA) if logA is not None else None

    # Get model info (only needed for ODE path)
    if not use_closed_form:
        try:
            ode_func, _, initial_state_dim, params_template = get_model_info(model_name, **model_definition_args)
            initial_state = _initial_state(model_name, initial_state_dim)
        except ValueError as e:
            warnings.warn(f"Residual func: Model info error: {e}")
            # Return large residuals to signal failure
            n_total = sum(len(ds.conversion) for ds in datasets)
            return np.full(n_total, 1e6)

    # Simulation loop
    for ds in datasets:
        if len(ds.time) < 2:
            continue

        try:
            # Dispatch: closed-form for isothermal, ODE otherwise
            if use_closed_form and is_isothermal(ds.temperature):
                T_const = np.mean(ds.temperature)
                t_sim_out, alpha_sim = _simulate_single_dataset_closed_form(
                    t_eval=ds.time,
                    T_const=T_const,
                    alpha_of_kt_func=alpha_of_kt_func,
                    Ea=Ea,
                    A=A
                )
            else:
                # Fall back to ODE path
                temp_func = get_temperature_interpolator(ds.time, ds.temperature)
                t_sim_out, alpha_sim = _simulate_single_dataset(
                    t_eval=ds.time,
                    temp_func=temp_func,
                    ode_system=ode_func,
                    initial_state=initial_state,
                    params_template=params_template,
                    current_params_logA=current_params_logA,
                    solver_options=solver_options
                )

            if len(alpha_sim) != len(ds.conversion):
                warnings.warn(f"Residual: Sim length mismatch.")
                # Return large residuals for this dataset
                residuals_list.append(np.full(len(ds.conversion), 1e6))
                continue

            residuals = ds.conversion - alpha_sim
            valid_mask = np.isfinite(residuals) & np.isfinite(alpha_sim)

            # --- Apply Weights via sqrt(weight) on residuals ---
            # least_squares minimizes sum(residuals²), so we scale residuals by sqrt(weight)
            # to achieve weighted RSS minimization
            sqrt_weights = np.full_like(residuals, np.sqrt(weight_baseline_plateau))
            transition_mask = (ds.conversion > alpha_lower) & (ds.conversion < alpha_upper)
            sqrt_weights[transition_mask] = np.sqrt(weight_transition)

            # Apply weights and mask
            weighted_residuals = residuals[valid_mask] * sqrt_weights[valid_mask]
            residuals_list.append(weighted_residuals)

        except Exception as e:
            warnings.warn(f"Residual func: Sim error: {e}.")
            # Return large residuals for this dataset
            residuals_list.append(np.full(len(ds.conversion), 1e6))

    # Concatenate all residuals
    if len(residuals_list) == 0:
        return np.array([1e6])  # Return single large residual if no valid datasets

    return np.concatenate(residuals_list)


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
            current_params_A = {}
            for name, val in current_params_logA.items():
                is_logA, original_name = _split_logA_name(name)
                current_params_A[original_name] = np.exp(val) if is_logA else val
            callback_func(iteration_counter[0], current_params_A)
        except Exception as e_cb:
            warnings.warn(f"Objective callback failed: {e_cb}")
    iteration_counter[0] += 1

    try:
        ode_func, _, initial_state_dim, params_template = get_model_info(model_name, **model_definition_args)
    except ValueError as e:
        warnings.warn(f"Objective func (rate): Model info error: {e}")
        return np.inf

    initial_state = _initial_state(model_name, initial_state_dim)

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
    solver_options: Dict,
    return_residuals: bool = False
) -> Union[Tuple[float, int, float, float, float], Tuple[float, int, float, float, float, np.ndarray]]:
    """Calculates RSS, N, R², AICc, and BIC for a given parameter set.

    Parameters
    ----------
    datasets : List[KineticDataset]
        Datasets to evaluate
    params_logA : Dict
        Parameters in logA scale
    model_name : str
        Model name
    model_definition_args : Dict
        Model definition arguments
    solver_options : Dict
        Solver options
    return_residuals : bool, optional
        If True, return residuals as 6th element of tuple

    Returns
    -------
    Tuple[float, int, float, float, float] or Tuple[float, int, float, float, float, np.ndarray]
        (rss, n_pts, r_squared, aic, bic) or (rss, n_pts, r_squared, aic, bic, residuals)
    """
    rss = np.inf
    n_pts = 0
    all_exp_conv = []
    all_residuals = []
    n_params = len(params_logA)

    try:
        ode_func, _, initial_state_dim, params_template = get_model_info(model_name, **model_definition_args)
        initial_state = _initial_state(model_name, initial_state_dim)
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
                    valid_residuals = residuals[valid_mask]
                    current_rss_calc += np.sum(valid_residuals**2)
                    n_pts += np.sum(valid_mask)
                    all_exp_conv.extend(ds.conversion[valid_mask])
                    if return_residuals:
                        all_residuals.extend(valid_residuals)
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

    if return_residuals:
        return rss, n_pts, r_squared, aic, bic, np.array(all_residuals)
    else:
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
    optimize_on_rate: bool = True,
    verbose: bool = True
    ) -> FitResult:
    """
    Fits model using logA scaling internally. User provides guesses/bounds for A.
    Attempts optimization on CONVERSION residuals first. If that fails,
    falls back to optimizing on RATE residuals.
    Reports final stats based on UNWEIGHTED CONVERSION fit quality.
    Includes defaults, warnings, callback. Stores model_definition_args in result.
    """
    # Powell is derivative-free: ODE solver tolerance makes the objective noisy at small
    # scales, so finite-difference gradients (L-BFGS-B) are unreliable and stall early.
    default_method = 'Powell'; opt_method = optimizer_options.get('method', default_method)
    default_opt_options = {'xtol': 1e-4, 'ftol': 1e-8, 'maxfev': 4000} if opt_method == 'Powell' else {'ftol': 1e-9, 'gtol': 1e-7}
    opt_options = optimizer_options.get('options', default_opt_options)
    if not isinstance(opt_options, dict): opt_options = default_opt_options

    # Wall-clock deadline, not just an eval-count cap: ODE objective calls vary
    # widely in cost (a stiff parameter region can be >50x slower than a benign
    # one -- measured 0.03s to 1.8s per call for A->B->C), so a fixed maxfev that
    # is safe for cheap single_step models can still run for a very long time on
    # multi-parameter ODE models. Enforced via scipy's per-iteration callback,
    # which is set below once param_names_logA (and so the parameter count) is known.
    user_max_seconds = optimizer_options.get('max_seconds', None)
    _opt_deadline = [None]

    def _deadline_callback(*_args, **_kwargs):
        if _opt_deadline[0] is not None and time.time() > _opt_deadline[0]:
            raise StopIteration("optimization exceeded its wall-clock budget")

    param_names_logA = []; original_param_names_map = {}
    try: # Determine logA parameter names and map
        _, original_param_names, _, _ = get_model_info(model_name, **model_definition_args)
        log_param_names = get_log_param_names(model_name)
        for name in original_param_names:
            is_A_param = name in log_param_names
            logA_name = "log" + name if is_A_param else name
            param_names_logA.append(logA_name)
            original_param_names_map[logA_name] = name
    except ValueError as e: return FitResult(model_name=model_name, parameters={}, success=False, message=f"Model setup error: {e}", rss=np.inf, n_datapoints=0, n_parameters=0, r_squared=np.nan, model_definition_args=model_definition_args)

    # 60s/parameter as a default budget: enough for Powell to make real progress on
    # a single_step (2-param) model even in a slow region, without letting a
    # multi-parameter ODE model (4+ params, costlier per-evaluation) run unbounded.
    default_max_seconds = 60.0 * max(1, len(param_names_logA))
    _opt_deadline[0] = time.time() + (user_max_seconds if user_max_seconds is not None else default_max_seconds)

    # Convert User Guesses/Bounds (A) to logA scale
    initial_guesses_logA = {}; parameter_bounds_logA = {} if parameter_bounds is not None else None; final_bounds_logA = {}
    if not all(p in initial_guesses for p in original_param_names): missing = [p for p in original_param_names if p not in initial_guesses]; return FitResult(model_name=model_name, model_definition_args=model_definition_args, parameters={}, success=False, message=f"Missing initial guesses for: {missing}", rss=np.inf, n_datapoints=0, n_parameters=len(param_names_logA), r_squared=np.nan)
    try: # Convert guesses
        for p_logA_name in param_names_logA:
            original_name = original_param_names_map[p_logA_name]
            is_logA_param = p_logA_name.startswith("logA")
            guess_val = initial_guesses[original_name]
            if is_logA_param and guess_val <= 0:
                raise ValueError(f"Initial guess for {original_name} must be positive.")
            initial_guesses_logA[p_logA_name] = np.log(guess_val) if is_logA_param else guess_val
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
    methods_supporting_bounds = ['L-BFGS-B', 'TNC', 'SLSQP', 'trust-constr', 'Powell', 'Nelder-Mead']
    if opt_method in methods_supporting_bounds and final_bounds_logA:
        try: bounds_list_logA = [(final_bounds_logA.get(p, (-np.inf, np.inf))[0] if np.isfinite(final_bounds_logA.get(p, (-np.inf, np.inf))[0]) else None, final_bounds_logA.get(p, (-np.inf, np.inf))[1] if np.isfinite(final_bounds_logA.get(p, (-np.inf, np.inf))[1]) else None) for p in param_names_logA]
        except Exception as e: return FitResult(model_name=model_name, model_definition_args=model_definition_args, parameters={}, success=False, message=f"Invalid bounds format: {e}", rss=np.inf, n_datapoints=0, n_parameters=len(param_names_logA), r_squared=np.nan)
    elif parameter_bounds and opt_method not in methods_supporting_bounds: warnings.warn(f"Optimizer {opt_method} ignores bounds.")

    reparam = _ArrheniusReparam(param_names_logA, datasets)
    x0_internal = reparam.to_internal(initial_params_array_logA)
    bounds_internal = reparam.bounds_to_internal(bounds_list_logA)

    # --- Detect Closed-Form Eligibility ---
    use_closed_form = False
    alpha_of_kt_func = None

    if model_name == "single_step":
        from .models import has_closed_form, CLOSED_FORM_REGISTRY
        from .utils import is_isothermal

        f_alpha_model = model_definition_args.get('f_alpha_model')
        if f_alpha_model and has_closed_form(f_alpha_model):
            # Check if ALL datasets are isothermal
            all_isothermal = all(is_isothermal(ds.temperature) for ds in datasets)
            if all_isothermal:
                use_closed_form = True
                alpha_of_kt_func = CLOSED_FORM_REGISTRY[f_alpha_model]
                if verbose:
                    print(f"Using closed-form solution for {f_alpha_model} (all datasets isothermal)")

    # --- Attempt Optimization with Closed-Form or ODE ---
    if use_closed_form:
        # --- Fast Path: least_squares with Closed-Form ---
        if verbose:
            print("--- Optimizing with scipy.optimize.least_squares (TRF method) ---")

        iteration_counter_conv = [0]

        try:
            from scipy.optimize import least_squares

            # Prepare residual function
            residual_args = (
                param_names_logA, datasets, model_name, model_definition_args,
                solver_options, True, alpha_of_kt_func, callback, iteration_counter_conv
            )
            wrapped_residual = reparam.wrap_residual(_residual_vector_function)

            # Convert bounds to format least_squares expects
            if bounds_internal:
                lower_bounds = np.array([b[0] if b[0] is not None else -np.inf for b in bounds_internal])
                upper_bounds = np.array([b[1] if b[1] is not None else np.inf for b in bounds_internal])
                bounds_lsq = (lower_bounds, upper_bounds)
            else:
                bounds_lsq = (-np.inf, np.inf)

            # Run least_squares
            opt_result = least_squares(
                fun=wrapped_residual,
                x0=x0_internal,
                args=residual_args,
                bounds=bounds_lsq,
                method='trf',
                ftol=1e-8,
                xtol=1e-8,
                max_nfev=2000,
                jac='2-point',  # Finite-difference Jacobian (analytic Jac deferred to Phase 1b)
                verbose=0
            )

            # Reconstruct hess_inv from Jacobian for param_std_err propagation
            if hasattr(opt_result, 'jac') and opt_result.jac is not None:
                try:
                    J = opt_result.jac
                    # hess_inv = inv(J^T J)
                    JTJ = J.T @ J
                    # Add small regularization for numerical stability
                    hess_inv_internal = np.linalg.inv(JTJ + 1e-12 * np.eye(len(param_names_logA)))
                    opt_result.hess_inv = hess_inv_internal
                except Exception as e_hess:
                    warnings.warn(f"Could not reconstruct hess_inv from Jacobian: {e_hess}")
                    opt_result.hess_inv = None

            success_conv = opt_result.success
            used_rate_fallback = False

        except Exception as e_lsq:
            warnings.warn(f"least_squares optimization failed: {e_lsq}. Falling back to Powell.")
            use_closed_form = False  # Fall through to Powell path
            success_conv = False
            opt_result = None

    if not use_closed_form:
        # --- Standard Path: Powell with ODE ---
        # --- Attempt Optimization 1: Conversion Residuals ---
        if verbose:
            print("--- Attempting optimization on CONVERSION residuals (weighted) ---")
        iteration_counter_conv = [0]
        opt_result = None # Initialize
        success_conv = False
        try:
            minimize_args_conv = (param_names_logA, datasets, model_name, model_definition_args, solver_options, callback, iteration_counter_conv)
            x0_internal = reparam.coarse_start(x0_internal, _objective_function, minimize_args_conv, bounds_internal, deadline=_opt_deadline[0])
            opt_result_conv = minimize(fun=reparam.wrap(_objective_function), x0=x0_internal, args=minimize_args_conv, method=opt_method, bounds=bounds_internal, options=opt_options, callback=_deadline_callback)
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
                time_sec = ds.time; rate = numerical_diff(time_sec, ds.conversion, **diff_options); exp_rates.append(rate); exp_times_sec.append(time_sec); valid_datasets_indices.append(i)
            datasets_for_rate_fit = [datasets[i] for i in valid_datasets_indices]

            if not datasets_for_rate_fit: # Check if rate fit is possible
                 return FitResult(model_name=model_name, model_definition_args=model_definition_args, parameters={}, success=False, message="Conversion fit failed & no datasets long enough for rate fit fallback.", rss=np.inf, n_datapoints=0, n_parameters=len(param_names_logA), r_squared=np.nan, used_rate_fallback=True)

            if verbose:
                print("--- Optimizing on RATE residuals ---")
            try:
                minimize_args_rate = (param_names_logA, datasets_for_rate_fit, exp_rates, exp_times_sec, model_name, model_definition_args, solver_options, callback, iteration_counter_rate)
                # Use the same initial guess for the rate fit
                opt_result_rate = minimize(fun=reparam.wrap(_objective_function_rate), x0=x0_internal, args=minimize_args_rate, method=opt_method, bounds=bounds_internal, options=opt_options, callback=_deadline_callback)
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
    opt_result.x = reparam.to_external(opt_result.x)
    if hasattr(opt_result, 'hess_inv'):
        try:
            h = opt_result.hess_inv.todense() if hasattr(opt_result.hess_inv, 'todense') else np.asarray(opt_result.hess_inv)
            opt_result.hess_inv = reparam.J @ h @ reparam.J.T
        except Exception:
            del opt_result['hess_inv']
    fitted_params_logA = dict(zip(param_names_logA, opt_result.x));
    n_params = len(param_names_logA)
    fitted_params_final = {}; # Convert logA to A
    for name_logA, value_logA in fitted_params_logA.items():
        is_logA_param, original_key = _split_logA_name(name_logA)
        fitted_params_final[original_key] = np.exp(value_logA) if is_logA_param else value_logA

    # Calculate FINAL stats based on UNWEIGHTED CONVERSION fit (Robustly)
    stats_result = _calculate_conversion_stats(
        datasets, fitted_params_logA, model_name, model_definition_args, solver_options,
        return_residuals=True
    )
    final_conversion_rss, n_total_datapoints_final, r_squared, aic, bic, residuals = stats_result

    # Calculate Durbin-Watson statistic for residual autocorrelation
    from .utils import calculate_durbin_watson, check_physical_plausibility
    durbin_watson = calculate_durbin_watson(residuals)

    # Check physical plausibility of parameters (permissive bounds by default)
    is_plausible, plausibility_issues = check_physical_plausibility(
        fitted_params_final, strict=False
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
                        is_logA_param, original_key = _split_logA_name(name_logA)
                        if is_logA_param: A_value = fitted_params_final[original_key]; param_std_err_final[original_key] = std_err_logA * A_value
                        else: param_std_err_final[original_key] = std_err_logA
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
        durbin_watson=durbin_watson,
        is_physically_plausible=is_plausible,
        plausibility_issues=plausibility_issues if not is_plausible else None,
        initial_ratio_r=initial_r,
        used_rate_fallback=used_rate_fallback # Store flag
    )

# --- Bootstrapping Worker Functions ---
_BOOTSTRAP_WORKER_FUNCTION = None
_BOOTSTRAP_WORKER_ARGS = None


def _initialize_bootstrap_worker(worker_function, worker_args):
    """Set invariant bootstrap inputs once in each worker process."""
    global _BOOTSTRAP_WORKER_FUNCTION, _BOOTSTRAP_WORKER_ARGS
    _BOOTSTRAP_WORKER_FUNCTION = worker_function
    _BOOTSTRAP_WORKER_ARGS = worker_args


def _execute_bootstrap_worker(iteration_index: int):
    """Run one replicate using the context installed by the pool initializer."""
    if _BOOTSTRAP_WORKER_FUNCTION is None or _BOOTSTRAP_WORKER_ARGS is None:
        raise RuntimeError("Bootstrap worker context was not initialized")
    return _BOOTSTRAP_WORKER_FUNCTION(
        *_BOOTSTRAP_WORKER_ARGS[:-1], iteration_index, _BOOTSTRAP_WORKER_ARGS[-1]
    )


def _get_bootstrap_max_workers(n_jobs: Optional[int], n_iterations: int) -> int:
    """Resolve the worker count, leaving one core free by default."""
    import os

    cpu_count = os.cpu_count() or 1
    if n_jobs is not None and n_jobs > 0:
        requested_workers = n_jobs
    else:
        requested_workers = max(1, cpu_count - 1)
    return min(requested_workers, max(1, n_iterations))


def _fit_empirical_on_resampled_data(
    datasets: List[KineticDataset],
    model_type: str,
    best_fit_params: Dict[str, float],
    parameter_bounds: Optional[Dict[str, Tuple[float, float]]],
    iteration_index: int,
    end_callback_func: Optional[Callable[[int, str, Optional[Dict]], None]]
) -> Optional[Dict]:
    """
    Fits empirical model to one bootstrap sample using weighted residual resampling.
    Returns dict {'params': ..., 'stats': ...} or None.
    """
    from akts.empirical import fit_empirical_global, predict_empirical, FitResult

    if end_callback_func:
        try:
            end_callback_func(iteration_index, "started", None)
        except Exception as e_cb:
            warnings.warn(f"Bootstrap start callback failed: {e_cb}")

    # Calculate residuals from best-fit predictions
    all_residuals = []
    original_indices_map = []
    original_alphas = []

    # Create minimal FitResult for prediction
    temp_fit_result = FitResult(
        model_name=f'Empirical_{model_type}',
        parameters=best_fit_params,
        success=True,
        message="Bootstrap prediction",
        rss=0.0,
        n_datapoints=0,
        n_parameters=len(best_fit_params),
        model_definition_args={'empirical_type': model_type}
    )

    for i, ds in enumerate(datasets):
        if len(ds.time) < 2:
            continue
        try:
            T_mean = np.mean(ds.temperature)
            # Predict using empirical model
            alpha_pred = predict_empirical(temp_fit_result, ds.time, T_mean)

            if len(alpha_pred) == len(ds.conversion):
                residuals = ds.conversion - alpha_pred
                valid_mask = np.isfinite(residuals) & np.isfinite(alpha_pred)
                all_residuals.extend(residuals[valid_mask])
                original_indices_map.extend([(i, k) for k, valid in enumerate(valid_mask) if valid])
                original_alphas.extend(ds.conversion[valid_mask])
            else:
                warnings.warn(f"Bootstrap {iteration_index}: Prediction length mismatch for dataset {i}")
        except Exception as e_pred:
            warnings.warn(f"Bootstrap {iteration_index}: Prediction failed for dataset {i}: {e_pred}")

    if not all_residuals:
        warnings.warn(f"Bootstrap {iteration_index}: No valid residuals.")
        return None

    # Weighted resampling (same as ODE bootstrap)
    all_residuals = np.array(all_residuals)
    centered_residuals = all_residuals - np.mean(all_residuals)
    original_alphas = np.array(original_alphas)
    n_residuals = len(centered_residuals)

    # Weight transition region more heavily
    weights = np.ones(n_residuals)
    transition_weight = 10.0
    alpha_lower_bound = 0.05
    alpha_upper_bound = 0.95

    for i, alpha in enumerate(original_alphas):
        if alpha_lower_bound <= alpha <= alpha_upper_bound:
            weights[i] = transition_weight

    weights /= weights.sum()

    # Resample residuals with replacement
    resampled_indices = np.random.choice(n_residuals, size=n_residuals, replace=True, p=weights)
    resampled_residuals = centered_residuals[resampled_indices]

    # Create synthetic datasets
    resampled_datasets = []
    residual_idx = 0

    for i, ds in enumerate(datasets):
        dataset_indices = [idx for idx, (ds_i, _) in enumerate(original_indices_map) if ds_i == i]
        n_pts_in_ds = len(dataset_indices)

        if n_pts_in_ds == 0:
            continue

        synthetic_alpha = ds.conversion.copy()
        for local_j, global_idx in enumerate(dataset_indices):
            data_point_idx = original_indices_map[global_idx][1]
            if residual_idx < len(resampled_residuals):
                synthetic_alpha[data_point_idx] = ds.conversion[data_point_idx] + resampled_residuals[residual_idx]
                residual_idx += 1

        synthetic_alpha = np.clip(synthetic_alpha, 0.0, 1.0)
        resampled_datasets.append(KineticDataset(
            time=ds.time,
            temperature=ds.temperature,
            conversion=synthetic_alpha
        ))

    if not resampled_datasets:
        warnings.warn(f"Bootstrap {iteration_index}: No synthetic datasets created.")
        return None

    # Refit empirical model to synthetic data
    try:
        replicate_fit = fit_empirical_global(
            datasets=resampled_datasets,
            model_type=model_type,
            initial_guess=best_fit_params,
            parameter_bounds=parameter_bounds,
            use_global_optimizer=False,
            verbose=False
        )

        if not replicate_fit.success:
            return None

        result_dict = {
            'params': replicate_fit.parameters,
            'stats': {
                'rss': replicate_fit.rss,
                'r_squared': replicate_fit.r_squared,
                'aic': replicate_fit.aic,
                'bic': replicate_fit.bic
            }
        }

        if end_callback_func:
            try:
                end_callback_func(iteration_index, "completed", result_dict)
            except Exception:
                pass

        return result_dict

    except Exception as e_fit:
        warnings.warn(f"Bootstrap {iteration_index}: Refit failed: {e_fit}")
        return None


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

    initial_state = _initial_state(model_name, initial_state_dim)
    all_residuals = []
    original_indices_map = []
    original_alphas = []
    original_predictions = [None] * len(datasets)

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
                original_predictions[i] = alpha_sim
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
        alpha_sim_orig = original_predictions[i]
        if alpha_sim_orig is None:
            continue
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
    # scipy.optimize.minimize requires an ordered list of (min, max) tuples
    # aligned with x0, not the {param_name: (min, max)} dict we carry around.
    bounds_list_logA = None
    if parameter_bounds_logA:
        bounds_list_logA = []
        for p in param_names_logA:
            lo, hi = parameter_bounds_logA.get(p, (-np.inf, np.inf))
            bounds_list_logA.append((lo if np.isfinite(lo) else None, hi if np.isfinite(hi) else None))

    output_dict = None
    try:
        reparam = _ArrheniusReparam(param_names_logA, resampled_datasets)
        method = optimizer_options.get('method', 'Powell')
        default_opts = {'xtol': 1e-4, 'ftol': 1e-8, 'maxfev': 2000} if method == 'Powell' else {}
        opt_result_boot = minimize(
            fun=reparam.wrap(_objective_function),
            x0=reparam.to_internal(np.array([perturbed_guesses_logA[p] for p in param_names_logA])),
            args=(param_names_logA, resampled_datasets, model_name, model_definition_args, solver_options, None, [0]),
            method=method,
            bounds=reparam.bounds_to_internal(bounds_list_logA),
            options=optimizer_options.get('options', default_opts)
        )
        if opt_result_boot.success:
            fitted_params_logA_replicate = dict(zip(param_names_logA, reparam.to_external(opt_result_boot.x)))
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

# --- NEW Simulation Function ---
def simulate_kinetics( model_name: str, model_definition_args: Dict, kinetic_params: Dict, initial_alpha: float, temperature_program: Union[Callable, Tuple[np.ndarray, np.ndarray]], simulation_time_sec: Optional[np.ndarray] = None, solver_options: Dict = {}) -> PredictionResult:
    """ Simulates kinetic process given model, parameters (A), and temperature program.

    For long predictions (>2000 points), automatically chunks the simulation to avoid
    memory issues and enable better progress tracking. Each chunk uses the final state
    from the previous chunk as its initial condition.
    """
    # Empirical models use direct prediction, not ODE simulation
    if model_name.startswith('Empirical_'):
        from akts.empirical import predict_empirical, FitResult

        # Prepare time points
        if callable(temperature_program):
            t_eval_sec = simulation_time_sec
            if t_eval_sec is None:
                raise ValueError("simulation_time_sec required for empirical model with callable temperature program")
            temp_K = temperature_program(t_eval_sec)
        elif isinstance(temperature_program, tuple) and len(temperature_program) == 2:
            t_prog_sec, temp_prog_K = temperature_program
            t_eval_sec = simulation_time_sec if simulation_time_sec is not None else t_prog_sec
            temp_interp = get_temperature_interpolator(t_prog_sec, temp_prog_K)
            temp_K = temp_interp(t_eval_sec)
        else:
            raise TypeError("temp program invalid type.")

        # Use mean temperature for empirical model (isothermal assumption)
        T_mean = np.mean(temp_K)

        # Create minimal FitResult for prediction function
        fit_res = FitResult(
            model_name=model_name,
            parameters=kinetic_params,
            success=True,
            message="Empirical model prediction",
            rss=0.0,
            n_datapoints=len(t_eval_sec),
            n_parameters=len(kinetic_params),
            model_definition_args=model_definition_args
        )

        alpha_pred = predict_empirical(fit_res, t_eval_sec, T_mean)

        return PredictionResult(
            time=t_eval_sec,
            conversion=alpha_pred,
            temperature=temp_K
        )

    params_A = kinetic_params; params_logA = {}; param_names_A = list(params_A.keys()); param_names_logA = [] # Convert A to logA
    try: log_param_names = get_log_param_names(model_name)
    except ValueError as e: raise ValueError(f"Model setup error during simulation: {e}")
    for name, val in params_A.items():
        is_A_param = name in log_param_names
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
    initial_state = _initial_state(model_name, initial_state_dim, initial_alpha)

    # Vectorized chunked simulation for long predictions
    chunk_size = solver_options.get('chunk_size', 2000)
    n_points = len(t_eval_sec)

    if n_points <= chunk_size:
        # Short prediction - use standard path
        t_pred_sec, alpha_pred = _simulate_single_dataset(
            t_eval=t_eval_sec, temp_func=temp_func, ode_system=ode_func,
            initial_state=initial_state, params_template=params_template,
            current_params_logA=params_logA, solver_options=solver_options
        )
    else:
        # Long prediction - chunk it, preserving full state continuity
        t_chunks = []
        alpha_chunks = []
        current_state = initial_state.copy()

        for i in range(0, n_points, chunk_size):
            # Extract chunk
            chunk_end = min(i + chunk_size, n_points)
            t_chunk = t_eval_sec[i:chunk_end]

            # Simulate chunk starting from current state, requesting final state
            result = _simulate_single_dataset(
                t_eval=t_chunk,
                temp_func=temp_func,
                ode_system=ode_func,
                initial_state=current_state,
                params_template=params_template,
                current_params_logA=params_logA,
                solver_options=solver_options,
                return_final_state=True
            )
            t_sim, alpha_sim, final_state = result

            t_chunks.append(t_sim)
            alpha_chunks.append(alpha_sim)

            # Update initial state for next chunk from final state of this chunk
            current_state = final_state.copy()

        t_pred_sec = np.concatenate(t_chunks)
        alpha_pred = np.concatenate(alpha_chunks)

    return PredictionResult(time=t_pred_sec, temperature=temp_eval_K, conversion=alpha_pred, conversion_ci=None)

def predict_conversion_model_free(
    iso_result: IsoResult,
    temperature_program: Union[Callable, Tuple[np.ndarray, np.ndarray]],
    simulation_time_sec: Optional[np.ndarray] = None,
    initial_alpha: float = 0.0,
    solver_options: Dict = {}
) -> PredictionResult:
    """
    Predicts conversion by integrating the Friedman isoconversional result directly,
    without assuming any reaction model f(alpha). Requires an IsoResult with both
    Ea and ln_A_f_alpha populated (currently only run_friedman() provides this --
    KAS/OFW are integral methods whose regression intercepts don't have this
    interpretation).

    Integrates: dalpha/dt = exp(ln_A_f_alpha(alpha)) * exp(-Ea(alpha) / (R*T(t)))
    with Ea(alpha) and ln_A_f_alpha(alpha) built as 1D interpolators over the
    alpha range the isoconversional analysis actually covered. Outside that
    range the nearest edge value is held constant (extrapolating a fitted-model
    Ea(alpha) trend past the data it was measured on is exactly the kind of
    silent error this whole method exists to avoid) -- predictions that need
    conversion levels beyond what run_friedman()'s alpha_levels covered should
    rerun it with a wider alpha_levels rather than rely on this extrapolation.

    Parameters
    ----------
    iso_result : IsoResult
        Result of run_friedman() with alpha, Ea, and ln_A_f_alpha all populated.
    temperature_program : Callable or (time_sec, temp_K) tuple
        Same shape as simulate_kinetics()'s temperature_program.
    simulation_time_sec : np.ndarray, optional
        Times to evaluate at. Required if temperature_program is callable.
    initial_alpha : float, default=0.0
        Starting conversion for the integration.
    solver_options : Dict, optional
        Passed to scipy.integrate.solve_ivp (method, rtol, atol).

    Returns
    -------
    PredictionResult
        Same shape as the model-based prediction path, so callers don't need
        to special-case a model-free result.
    """
    if iso_result.Ea is None or iso_result.ln_A_f_alpha is None:
        raise ValueError(
            "predict_conversion_model_free requires an IsoResult with both Ea and "
            "ln_A_f_alpha populated. Only run_friedman() currently provides "
            "ln_A_f_alpha; KAS/OFW intercepts are not usable here."
        )

    valid = np.isfinite(iso_result.alpha) & np.isfinite(iso_result.Ea) & np.isfinite(iso_result.ln_A_f_alpha)
    if valid.sum() < 2:
        raise ValueError("Need at least 2 finite (alpha, Ea, ln_A_f_alpha) points to predict from.")

    alpha_fit = iso_result.alpha[valid]
    Ea_interp = interp1d(alpha_fit, iso_result.Ea[valid], bounds_error=False,
                         fill_value=(iso_result.Ea[valid][0], iso_result.Ea[valid][-1]))
    lnAf_interp = interp1d(alpha_fit, iso_result.ln_A_f_alpha[valid], bounds_error=False,
                           fill_value=(iso_result.ln_A_f_alpha[valid][0], iso_result.ln_A_f_alpha[valid][-1]))
    alpha_max = float(alpha_fit.max())

    if callable(temperature_program):
        temp_func = temperature_program; t_eval_sec = simulation_time_sec
    elif isinstance(temperature_program, tuple) and len(temperature_program) == 2:
        t_prog_sec, temp_prog_K = temperature_program
        temp_func = get_temperature_interpolator(t_prog_sec, temp_prog_K)
        t_eval_sec = simulation_time_sec if simulation_time_sec is not None else t_prog_sec
    else:
        raise TypeError("temperature_program invalid type.")
    if t_eval_sec is None:
        raise ValueError("simulation_time_sec required if temperature_program is callable.")

    def rhs(t, y):
        alpha = min(max(y[0], 0.0), alpha_max)  # interpolators are undefined/edge-held past this
        T = float(temp_func(t))
        if T <= 0:
            return [0.0]
        rate = np.exp(float(lnAf_interp(alpha))) * np.exp(-float(Ea_interp(alpha)) / (R_GAS * T))
        return [max(0.0, rate) if np.isfinite(rate) else 0.0]

    t_eval_sorted = np.sort(np.asarray(t_eval_sec, dtype=float))
    t_start, t_end = t_eval_sorted[0], t_eval_sorted[-1]
    method = solver_options.get('primary_solver', solver_options.get('method', 'RK45'))
    sol = solve_ivp(
        fun=_with_eval_budget(rhs, solver_options.get('max_rhs_evals', 20000)),
        t_span=(t_start, t_end), y0=[initial_alpha], t_eval=t_eval_sorted, method=method,
        rtol=solver_options.get('rtol', 1e-6), atol=solver_options.get('atol', 1e-9),
    )
    if not sol.success:
        warnings.warn(f"Model-free prediction ODE solve failed: {sol.message}")
        alpha_out = np.full_like(t_eval_sorted, np.nan)
    else:
        alpha_out = np.clip(sol.y[0], 0.0, 1.0)

    temp_eval_K = temp_func(t_eval_sorted)
    return PredictionResult(time=t_eval_sorted, temperature=temp_eval_K, conversion=alpha_out, conversion_ci=None)


# --- Prediction Function ---
def predict_conversion( kinetic_description: Union[FitResult, IsoResult], temperature_program: Union[Callable, Tuple[np.ndarray, np.ndarray]], simulation_time_sec: Optional[np.ndarray] = None, initial_alpha: float = 0.0, solver_options: Dict = {}, bootstrap_result: Optional[BootstrapResult] = None) -> PredictionResult:
    """ Predicts conversion using fitted parameters. Calculates CI if bootstrap results provided. """
    if isinstance(kinetic_description, IsoResult):
        if kinetic_description.Ea is not None and kinetic_description.ln_A_f_alpha is not None:
            return predict_conversion_model_free(
                iso_result=kinetic_description, temperature_program=temperature_program,
                simulation_time_sec=simulation_time_sec, initial_alpha=initial_alpha,
                solver_options=solver_options
            )
        warnings.warn("Prediction from this IsoResult not implemented (method lacks ln_A_f_alpha; only run_friedman() provides it)."); return PredictionResult(time=np.array([]), temperature=np.array([]), conversion=np.array([]))
    elif isinstance(kinetic_description, FitResult):
        fit_result = kinetic_description
        if not fit_result.success: warnings.warn("Cannot predict from unsuccessful fit."); return PredictionResult(time=np.array([]), temperature=np.array([]), conversion=np.array([]))

        if fit_result.model_name == "Friedman":
            # Friedman is wrapped as a FitResult (helpers._wrap_friedman_as_fit_result)
            # so it flows through ranking/selection/reporting like any other model,
            # but it has no Ea/A "parameters" -- the real IsoResult is stashed in
            # model_definition_args, the same "everything needed to predict again"
            # role that field plays for every other model.
            iso_result = (fit_result.model_definition_args or {}).get('iso_result')
            if iso_result is None:
                warnings.warn("Friedman FitResult missing stashed iso_result; cannot predict.")
                return PredictionResult(time=np.array([]), temperature=np.array([]), conversion=np.array([]))

            base_prediction = predict_conversion_model_free(
                iso_result, temperature_program, simulation_time_sec, initial_alpha, solver_options
            )
            if (bootstrap_result is not None and bootstrap_result.model_name == "Friedman"
                    and bootstrap_result.raw_parameter_list):
                t_eval_sec = base_prediction.time
                replicate_alphas = []
                for rep in bootstrap_result.raw_parameter_list:
                    valid = np.isfinite(rep['alpha']) & np.isfinite(rep['Ea']) & np.isfinite(rep['ln_A_f_alpha'])
                    if valid.sum() < 2:
                        continue
                    rep_iso = IsoResult(method="Friedman", alpha=rep['alpha'][valid],
                                        Ea=rep['Ea'][valid], ln_A_f_alpha=rep['ln_A_f_alpha'][valid])
                    try:
                        rep_pred = predict_conversion_model_free(
                            rep_iso, temperature_program, t_eval_sec, initial_alpha, solver_options
                        )
                        if len(rep_pred.conversion) == len(t_eval_sec):
                            replicate_alphas.append(rep_pred.conversion)
                    except Exception as e_boot:
                        warnings.warn(f"Friedman bootstrap replicate prediction failed: {e_boot}")
                if replicate_alphas:
                    all_boot_alphas = np.array(replicate_alphas)
                    ci_level = (1.0 - bootstrap_result.confidence_level) / 2.0

                    # Filter degenerate bootstrap samples (Friedman model)
                    final_conversions = all_boot_alphas[:, -1]
                    valid_mask = np.isfinite(final_conversions) & (final_conversions >= 0.01)
                    n_valid = np.sum(valid_mask)
                    n_total = all_boot_alphas.shape[0]

                    if n_valid < n_total:
                        warnings.warn(f"Friedman: Filtered {n_total - n_valid}/{n_total} degenerate bootstrap samples.")

                    if n_valid >= 10:
                        filtered_alphas = all_boot_alphas[valid_mask, :]
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", category=RuntimeWarning)
                            lower = np.nanpercentile(filtered_alphas, ci_level * 100.0, axis=0)
                            upper = np.nanpercentile(filtered_alphas, (1.0 - ci_level) * 100.0, axis=0)
                    else:
                        warnings.warn(f"Friedman: Only {n_valid} valid bootstrap samples.")
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", category=RuntimeWarning)
                            lower = np.nanpercentile(all_boot_alphas, ci_level * 100.0, axis=0)
                            upper = np.nanpercentile(all_boot_alphas, (1.0 - ci_level) * 100.0, axis=0)

                    base_prediction.conversion_ci = (lower, upper)
            return base_prediction

        # Retrieve model_definition_args from fit_result
        model_definition_args = getattr(fit_result, 'model_definition_args', None)
        if model_definition_args is None: warnings.warn("FitResult missing 'model_definition_args'. Prediction may fail or be incorrect."); model_definition_args = {} # Or raise error
        model_name = fit_result.model_name; params_A = fit_result.parameters

        # Simulate base prediction
        base_prediction = simulate_kinetics(model_name=model_name, model_definition_args=model_definition_args, kinetic_params=params_A, initial_alpha=initial_alpha, temperature_program=temperature_program, simulation_time_sec=simulation_time_sec, solver_options=solver_options)
        alpha_lower_ci, alpha_upper_ci = None, None # Calculate CIs using bootstrap_result and simulate_kinetics
        if bootstrap_result is not None and bootstrap_result.model_name == model_name and bootstrap_result.n_iterations > 0:
            n_boot_iter = bootstrap_result.n_iterations; t_eval_sec = base_prediction.time
            all_boot_alphas = np.full((n_boot_iter, len(t_eval_sec)), np.nan); param_dist_A = bootstrap_result.parameter_distributions
            param_names_A = list(params_A.keys())  # Get names from original fit
            # Generate bootstrap predictions sequentially
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
            if all_boot_alphas.size > 0:
                # Count successful simulations
                n_successful = np.sum(np.isfinite(all_boot_alphas[:, 0]))  # Check first time point
                n_total = all_boot_alphas.shape[0]

                if n_successful < 0.5 * n_total:
                    warnings.warn(f"Only {n_successful}/{n_total} bootstrap simulations succeeded. CI may be unreliable.")

                if np.any(np.isfinite(all_boot_alphas)):
                    alpha_ci_level = (1.0 - bootstrap_result.confidence_level) / 2.0

                    # Filter out degenerate bootstrap samples (those predicting <1% conversion at end)
                    # These cause the lower CI to collapse to zero
                    final_conversions = all_boot_alphas[:, -1]
                    valid_mask = np.isfinite(final_conversions) & (final_conversions >= 0.01)
                    n_valid = np.sum(valid_mask)
                    n_total = all_boot_alphas.shape[0]

                    if n_valid < n_total:
                        warnings.warn(f"Filtered {n_total - n_valid}/{n_total} degenerate bootstrap samples "
                                     f"(final conversion < 1%). CI calculation uses {n_valid} valid samples.")

                    if n_valid >= 10:  # Need at least 10 samples for reasonable CI
                        # Use only valid samples for CI calculation
                        filtered_alphas = all_boot_alphas[valid_mask, :]

                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", category=RuntimeWarning)
                            alpha_lower_ci = np.nanpercentile(filtered_alphas, alpha_ci_level * 100.0, axis=0)
                            alpha_upper_ci = np.nanpercentile(filtered_alphas, (1.0 - alpha_ci_level) * 100.0, axis=0)
                    else:
                        warnings.warn(f"Only {n_valid} valid bootstrap samples - CI may be unreliable.")
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", category=RuntimeWarning)
                            alpha_lower_ci = np.nanpercentile(all_boot_alphas, alpha_ci_level * 100.0, axis=0)
                            alpha_upper_ci = np.nanpercentile(all_boot_alphas, (1.0 - alpha_ci_level) * 100.0, axis=0)

                    # Check if CI is degenerate (all zeros)
                    if np.all(alpha_lower_ci == 0) and np.max(alpha_upper_ci) > 0:
                        warnings.warn("Lower CI bound is all zeros - bootstrap may have insufficient variation or failures.")
                else:
                    warnings.warn("Could not calculate CIs; no successful bootstrap simulations or all results were NaN.")
            else:
                warnings.warn("Could not calculate CIs; bootstrap result array is empty.")


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
        params_A = {}
        for name_logA, val_logA in params_logA.items():
            is_logA, original_key = _split_logA_name(name_logA)
            params_A[original_key] = np.exp(val_logA) if is_logA else val_logA
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

def rank_models(
    fit_results: List[FitResult],
    score_weights: Optional[Dict[str, float]] = None,
    ranking_method: str = 'combined'
) -> List[Dict]:
    """
    Ranks a list of FitResult objects based on various ranking methods.

    Parameters
    ----------
    fit_results : List[FitResult]
        List of fit results to rank
    score_weights : Dict[str, float], optional
        Custom weights for combined scoring (only used when ranking_method='combined')
        Default: {'bic': 0.4, 'r_squared': 0.4, 'rss': 0.1, 'n_params': 0.1}
    ranking_method : str, default='combined'
        Ranking approach:
        - 'combined': Weighted combination of BIC, R², RSS, n_params (default)
        - 'bic': Rank by BIC alone (lower is better)
        - 'aic': Rank by AICc alone (lower is better)
        - 'akaike_weight': Rank by Akaike weight (higher is better)
        - 'r_squared': Rank by R² alone (higher is better)

    Returns
    -------
    List[Dict]
        Ranked models with stats, scores, and Akaike weights

    Notes
    -----
    - BIC/AIC ranking methods provide direct interpretation: ΔBIC > 10 is "very strong"
      evidence against the higher-BIC model
    - Akaike weights give the probability each model is the best in the candidate set
    - Combined scoring (default) balances multiple criteria but is less interpretable
    - Simplicity penalty (for fitted shape parameters) is applied to all methods
    """
    if not fit_results:
        return []

    # Validate ranking method
    valid_methods = ['combined', 'bic', 'aic', 'akaike_weight', 'r_squared']
    if ranking_method not in valid_methods:
        raise ValueError(f"ranking_method must be one of {valid_methods}, got '{ranking_method}'")

    # --- Define Default Weights (only used for 'combined') ---
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
        durbin_watson = getattr(res, 'durbin_watson', np.nan)
        is_plausible = getattr(res, 'is_physically_plausible', None)
        plausibility_issues = getattr(res, 'plausibility_issues', None)
        valid_fits_data.append({
            'model_name': res.model_name,
            'parameters': res.parameters,
            'stats': {
                'rss': rss, 'r_squared': r2, 'aic': aic, 'bic': bic,
                'n_params': n_params, 'n_points': n_points,
                'r_squared_adj': calculate_adjusted_r_squared(r2, n_params, n_points),
                'rmse': np.sqrt(rss / n_points) if n_points > 0 else np.nan,
                'durbin_watson': durbin_watson,
                'is_physically_plausible': is_plausible,
                'plausibility_issues': plausibility_issues,
            },
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

    # --- Calculate Score for each model based on ranking method ---
    if ranking_method == 'combined':
        # Original combined scoring with normalization
        ranges = {key: (np.min(stat_values[key]), np.ptp(stat_values[key])) for key in ['rss', 'aic', 'bic', 'n_params']}
        finite_r2 = [r for r in stat_values['r_squared'] if np.isfinite(r)]
        ranges['r_squared'] = (np.min(finite_r2), np.max(finite_r2), np.ptp(finite_r2)) if finite_r2 else (0, 0, 0)

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

    elif ranking_method == 'bic':
        # Rank by BIC alone (lower is better)
        for item in valid_fits_data:
            item['score'] = item['stats']['bic']

    elif ranking_method == 'aic':
        # Rank by AICc alone (lower is better)
        for item in valid_fits_data:
            item['score'] = item['stats']['aic']

    elif ranking_method == 'akaike_weight':
        # Rank by Akaike weight (higher is better, so negate for ascending sort)
        # Compute Akaike weights first
        temp_akaike_weights = calculate_akaike_weights([item['stats']['aic'] for item in valid_fits_data])
        for item, w in zip(valid_fits_data, temp_akaike_weights):
            item['score'] = -w  # Negate so higher weight = lower score = better rank

    elif ranking_method == 'r_squared':
        # Rank by R² alone (higher is better, so negate for ascending sort)
        for item in valid_fits_data:
            r2 = item['stats']['r_squared']
            item['score'] = -r2 if np.isfinite(r2) else np.inf  # Negate for ascending sort

    # --- Apply Simplicity Penalty for Fitted Shape Parameters ---
    # Prefer lower values of fitted parameters (n, m, p) when fit quality is similar
    # Penalty is small (~0.01 per unit) so it only affects ranking when scores are close
    for item in valid_fits_data:
        params = item['parameters']
        simplicity_penalty = 0.0

        # Fn model: prefer lower n (e.g., n=1 over n=3)
        if 'n' in params and item['model_name'] == 'single_step':
            # Check if this is a fitted n (not fixed F1/F2/F3)
            # Penalty scales with distance from n=1 (most common reaction order)
            n_val = params['n']
            if n_val > 1.0:
                simplicity_penalty += 0.01 * (n_val - 1.0)  # Penalize n > 1

        # SB/SB_mnp models: prefer lower m,n values
        if 'm' in params and 'n' in params:
            m_val = params['m']
            n_val = params['n']
            # Penalty scales with distance from standard SB_mn values (m=0.5, n=1.0)
            simplicity_penalty += 0.01 * abs(m_val - 0.5)
            simplicity_penalty += 0.01 * abs(n_val - 1.0)

            # Additional penalty for p in SB_mnp (prefer p=0, i.e., reduces to SB)
            if 'p' in params:
                p_val = params['p']
                simplicity_penalty += 0.01 * abs(p_val)

        item['score'] += simplicity_penalty
        item['simplicity_penalty'] = simplicity_penalty  # Store for transparency

    # --- Sort by Score (ascending) ---
    valid_fits_data.sort(key=lambda x: x['score'])
    for rank, item in enumerate(valid_fits_data):
        item['rank'] = rank + 1

    # Akaike weights: probability each model is the best in this candidate set,
    # given AICc. Computed over the same finite-AICc models used for scoring/ranking
    # above (ties in 'score' do not affect this -- it is a separate, additive stat).
    akaike_weights = calculate_akaike_weights([item['stats']['aic'] for item in valid_fits_data])
    for item, w in zip(valid_fits_data, akaike_weights):
        item['stats']['akaike_weight'] = w

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
    """Bootstrap fitted parameters using residual resampling.

    Replicates run in a process pool. ``n_jobs=-1`` uses all but one available
    CPU core by default, and the worker count is capped at ``n_iterations``.
    """
    # ... (Setup code: check fit_result, get model_def_args, convert params/bounds to logA - remains the same) ...
    if not fit_result.success: warnings.warn("Initial fit failed."); return None
    model_definition_args = getattr(fit_result, 'model_definition_args', None)
    if model_definition_args is None: warnings.warn("FitResult missing 'model_definition_args'. Cannot run bootstrap reliably."); return None
    model_name = fit_result.model_name; best_fit_params_A = fit_result.parameters

    # Route empirical models to their own bootstrap implementation
    if model_name.startswith('Empirical_'):
        return run_bootstrap_empirical(
            datasets=datasets,
            fit_result=fit_result,
            parameter_bounds=parameter_bounds,
            n_iterations=n_iterations,
            confidence_level=confidence_level,
            n_jobs=n_jobs,
            end_callback=end_callback,
            timeout_per_replicate=timeout_per_replicate,
            return_replicate_params=return_replicate_params
        )
    best_fit_params_logA = {}; param_names_logA = []
    try:
        log_param_names = get_log_param_names(model_name)
        for name, val in best_fit_params_A.items():
            is_A_param = name in log_param_names
            logA_name = "log" + name if is_A_param else name
            best_fit_params_logA[logA_name] = np.log(val) if is_A_param else val
            param_names_logA.append(logA_name)
    except (ValueError, TypeError) as e_conv: warnings.warn(f"Cannot take log of parameter for bootstrap: {e_conv}."); return None
    parameter_bounds_logA = None # Convert A bounds to logA bounds
    if parameter_bounds:
        parameter_bounds_logA = {}
        for name_logA in param_names_logA:
            is_logA_param, original_key = _split_logA_name(name_logA)

            if original_key in parameter_bounds:
                 min_A, max_A = parameter_bounds[original_key]
                 if is_logA_param:
                      min_logA = np.log(min_A) if min_A is not None and min_A > 0 else -np.inf
                      max_logA = np.log(max_A) if max_A is not None and max_A > 0 else np.inf
                      parameter_bounds_logA[name_logA] = (min_logA, max_logA)
                 else:
                      # Keep original bounds for non-logA params (like Ea)
                      parameter_bounds_logA[name_logA] = (min_A, max_A)
            else: # If original key not in user bounds, use default infinite bounds
                 parameter_bounds_logA[name_logA] = (-np.inf, np.inf)
    max_workers = _get_bootstrap_max_workers(n_jobs, n_iterations)
    print(f"Starting {n_iterations} bootstrap fits (logA scale) using concurrent.futures (max_workers={max_workers})...")
    results_data = [None] * n_iterations; futures_list = []; n_submitted = 0
    try: # Run parallel fits
        worker_args = (
            datasets, model_name, model_definition_args, best_fit_params_logA,
            parameter_bounds_logA, solver_options, optimizer_options, end_callback
        )
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=_initialize_bootstrap_worker,
            initargs=(_fit_on_resampled_data, worker_args)
        ) as executor:
            for i in range(n_iterations):
                future = executor.submit(_execute_bootstrap_worker, i)
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
        for name_logA, val_logA in median_params_logA.items():
            is_logA, original_key = _split_logA_name(name_logA)
            median_params_A[original_key] = np.exp(val_logA) if is_logA else val_logA

    # --- Rank Replicates ---
    ranked_replicates = rank_replicates(successful_params_logA, successful_replicate_stats)

    # --- Calculate CIs and convert final distributions ---
    param_distributions_logA = {p_name: np.array([params[p_name] for params in successful_params_logA]) for p_name in param_names_logA}
    param_ci_logA = {}
    alpha_level = (1.0 - confidence_level) / 2.0 # Removed semicolon

    # Calculate weighted scores for model ranking
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
        for name_logA, val_logA in params_logA.items():
            is_logA, original_key = _split_logA_name(name_logA)
            params_A[original_key] = np.exp(val_logA) if is_logA else val_logA
        if return_replicate_params: raw_parameter_list_A.append(params_A) # Currently unsorted list
    for name_logA, dist_logA in param_distributions_logA.items(): # Convert distributions and CIs
        is_logA_param, original_key = _split_logA_name(name_logA)
        param_distributions_final[original_key] = np.exp(dist_logA) if is_logA_param else dist_logA
        ci_logA = param_ci_logA[name_logA]
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


def run_bootstrap_empirical(
    datasets: List[KineticDataset],
    fit_result: FitResult,
    parameter_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    n_iterations: int = 100,
    confidence_level: float = 0.95,
    n_jobs: int = -1,
    end_callback: Optional[Callable[[int, str, Optional[Dict]], None]] = None,
    timeout_per_replicate: Optional[float] = None,
    return_replicate_params: bool = False
) -> Optional[BootstrapResult]:
    """
    Performs bootstrap for empirical models using weighted residual resampling.

    Uses the same methodology as ODE model bootstrap but calls empirical fitting functions.

    Parameters
    ----------
    datasets : List[KineticDataset]
        Original experimental datasets
    fit_result : FitResult
        Fitted empirical model result
    parameter_bounds : Dict, optional
        Parameter bounds for refitting
    n_iterations : int, default=100
        Number of bootstrap iterations
    confidence_level : float, default=0.95
        Confidence level (e.g., 0.95 for 95% CI)
    n_jobs : int, default=-1
        Number of parallel workers (-1 = all CPUs)
    end_callback : Callable, optional
        Callback function for progress tracking
    timeout_per_replicate : float, optional
        Timeout in seconds per replicate
    return_replicate_params : bool, default=False
        If True, return all replicate parameters (can be large)

    Returns
    -------
    BootstrapResult or None
        Bootstrap results with parameter distributions and CI information
    """
    if not fit_result.success:
        warnings.warn("Initial fit failed.")
        return None

    model_definition_args = getattr(fit_result, 'model_definition_args', None)
    if model_definition_args is None:
        warnings.warn("FitResult missing 'model_definition_args'.")
        return None

    model_type = model_definition_args.get('empirical_type')
    if not model_type:
        warnings.warn("Empirical model type not found in model_definition_args.")
        return None

    best_fit_params = fit_result.parameters

    # Run parallel bootstrap fits, leaving one core free by default.
    max_workers = _get_bootstrap_max_workers(n_jobs, n_iterations)
    print(f"Starting {n_iterations} bootstrap fits for Empirical_{model_type} using concurrent.futures (max_workers={max_workers})...")

    results_data = [None] * n_iterations
    futures_list = []
    n_submitted = 0

    try:
        worker_args = (
            datasets, model_type, best_fit_params, parameter_bounds, end_callback
        )
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=_initialize_bootstrap_worker,
            initargs=(_fit_empirical_on_resampled_data, worker_args)
        ) as executor:
            for i in range(n_iterations):
                future = executor.submit(_execute_bootstrap_worker, i)
                futures_list.append(future)
                n_submitted += 1

            print(f"\nSubmitted {n_submitted} tasks. Collecting results...")

            for i, future in enumerate(futures_list):
                try:
                    result = future.result(timeout=timeout_per_replicate)
                    results_data[i] = result
                except concurrent.futures.TimeoutError:
                    warnings.warn(f"Bootstrap replicate {i+1} timed out after {timeout_per_replicate}s.")
                    results_data[i] = None
                except Exception as e:
                    warnings.warn(f"Bootstrap replicate {i+1} failed with exception: {type(e).__name__}: {e}")
                    results_data[i] = None

    except Exception as e_pool:
        warnings.warn(f"Error during bootstrap pool execution: {type(e_pool).__name__}: {e_pool}")

    # Process collected results
    successful_params = []
    successful_stats = []
    n_failed = 0

    print("\nProcessing received results...")
    for i, p_data in enumerate(results_data):
        if p_data is None or not isinstance(p_data, dict) or 'params' not in p_data or 'stats' not in p_data:
            n_failed += 1
        else:
            successful_params.append(p_data['params'])
            successful_stats.append(p_data['stats'])

    n_successful = len(successful_params)
    print(f"\nBootstrap finished processing. {n_successful}/{n_iterations} replicates successful ({n_failed} failed/timed out).")

    if n_successful < 10:
        warnings.warn(f"Too few successful bootstrap replicates ({n_successful}). Cannot compute reliable CI.")
        return None

    # Calculate parameter statistics
    param_names = list(best_fit_params.keys())
    param_distributions = {name: [] for name in param_names}

    for params in successful_params:
        for name in param_names:
            if name in params:
                param_distributions[name].append(params[name])

    # Calculate percentiles for CI
    ci_lower_pct = (1.0 - confidence_level) / 2.0 * 100.0
    ci_upper_pct = (1.0 + confidence_level) / 2.0 * 100.0

    param_ci = {}
    param_std = {}
    for name in param_names:
        values = np.array(param_distributions[name])
        if len(values) > 0:
            param_ci[name] = (
                float(np.percentile(values, ci_lower_pct)),
                float(np.percentile(values, ci_upper_pct))
            )
            param_std[name] = float(np.std(values))
        else:
            param_ci[name] = (np.nan, np.nan)
            param_std[name] = np.nan

    # Rank replicates by R²
    ranked_replicates = []
    for i, (params, stats) in enumerate(zip(successful_params, successful_stats)):
        ranked_replicates.append({
            'replicate_index': i,
            'parameters': params,
            'r_squared': stats.get('r_squared', np.nan),
            'rss': stats.get('rss', np.inf),
            'aic': stats.get('aic', np.inf),
            'bic': stats.get('bic', np.inf)
        })

    ranked_replicates.sort(key=lambda x: x['r_squared'], reverse=True)

    # Calculate median parameters
    median_params = {}
    for name in param_names:
        values = np.array(param_distributions[name])
        if len(values) > 0:
            median_params[name] = float(np.median(values))
        else:
            median_params[name] = best_fit_params[name]

    # Calculate median stats (refit with median params)
    median_stats = {'r_squared': np.nan, 'rss': np.inf}
    try:
        from akts.empirical import fit_empirical_global
        median_fit = fit_empirical_global(
            datasets=datasets,
            model_type=model_type,
            initial_guess=median_params,
            parameter_bounds=parameter_bounds,
            verbose=False
        )
        if median_fit.success:
            median_stats = {
                'r_squared': median_fit.r_squared,
                'rss': median_fit.rss,
                'aic': median_fit.aic,
                'bic': median_fit.bic
            }
    except Exception:
        pass

    # Prepare raw parameter list if requested
    raw_parameter_list = successful_params if return_replicate_params else None

    # Convert parameter lists to distributions (numpy arrays) for BootstrapResult
    param_distributions_arrays = {}
    for name in param_names:
        values = np.array(param_distributions[name])
        param_distributions_arrays[name] = values

    return BootstrapResult(
        model_name=fit_result.model_name,
        parameter_distributions=param_distributions_arrays,
        parameter_ci=param_ci,
        n_iterations=n_successful,  # Store successful count
        confidence_level=confidence_level,
        raw_parameter_list=raw_parameter_list,
        ranked_replicates=ranked_replicates,
        median_parameters=median_params,
        median_stats=median_stats
    )

