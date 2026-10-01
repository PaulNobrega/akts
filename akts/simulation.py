"""
ODE / closed-form simulation primitives shared by fitting, bootstrap and prediction.

Also owns the internal log-scale parameter encoding (A <-> logA) and the Arrhenius
reparameterization used by the optimizers.
"""
import numpy as np
import warnings
import time
from typing import List, Dict, Tuple, Optional, Callable, Union
from scipy.integrate import solve_ivp

from .datatypes import KineticDataset, PredictionResult, OdeSystemCallable
from .models import get_model_info, get_log_param_names
from .utils import get_temperature_interpolator, R_GAS


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


def params_A_to_logA(params_A: Dict[str, float], model_name: str) -> Dict[str, float]:
    """Converts user-facing (A-scale) parameters to the internal logA encoding.

    Pre-exponential factors listed in models.get_log_param_names(model_name) become
    "log"+name with value ln(A); every other parameter passes through unchanged.
    Raises ValueError for a non-positive pre-exponential factor.
    """
    log_param_names = get_log_param_names(model_name)
    params_logA: Dict[str, float] = {}
    for name, value in params_A.items():
        if name in log_param_names:
            if value is None or not value > 0:
                raise ValueError(f"Cannot take log of parameter {name}={value}")
            params_logA["log" + name] = float(np.log(value))
        else:
            params_logA[name] = float(value)
    return params_logA


def params_logA_to_A(params_logA: Dict[str, float]) -> Dict[str, float]:
    """Inverse of params_A_to_logA: decodes internal logA names back to A-scale floats."""
    params_A: Dict[str, float] = {}
    for name_logA, value in params_logA.items():
        is_logA, original_name = _split_logA_name(name_logA)
        params_A[original_name] = float(np.exp(value) if is_logA else value)
    return params_A


# Default ODE solver configuration. LSODA switches automatically between a
# non-stiff (Adams) and stiff (BDF) method, so it handles both the benign and
# the stiff parameter regions an optimizer visits; RK45 is the fallback.
# Override per call with solver_options={'primary_solver': ..., 'fallback_solver': ...}
# (fallback_solver=None disables the fallback). See docs/advanced_usage.md.
DEFAULT_SOLVER_OPTIONS: Dict = {
    'primary_solver': 'LSODA',
    'fallback_solver': 'RK45',
    'rtol': 1e-6,
    'atol': 1e-9,
    'max_rhs_evals': 20000,
    'chunk_size': 2000,
}


def resolve_solver_options(solver_options: Optional[Dict] = None) -> Dict:
    """Returns DEFAULT_SOLVER_OPTIONS overlaid with any user-supplied keys."""
    resolved = dict(DEFAULT_SOLVER_OPTIONS)
    if solver_options:
        resolved.update(solver_options)
    return resolved


def _resolve_temperature_program(
    temperature_program: Union[Callable, Tuple[np.ndarray, np.ndarray]],
    simulation_time_sec: Optional[np.ndarray] = None,
) -> Tuple[Callable, np.ndarray, np.ndarray]:
    """Normalizes a temperature program to (temp_func, t_eval_sec, temp_eval_K).

    temperature_program is either a callable T(t) [K] -- in which case
    simulation_time_sec is required -- or a (time_sec, temp_K) tuple, whose own
    time grid is used when simulation_time_sec is None.
    """
    if callable(temperature_program):
        temp_func = temperature_program
        t_eval_sec = simulation_time_sec
    elif isinstance(temperature_program, tuple) and len(temperature_program) == 2:
        t_prog_sec, temp_prog_K = temperature_program
        temp_func = get_temperature_interpolator(np.asarray(t_prog_sec, dtype=float),
                                                 np.asarray(temp_prog_K, dtype=float))
        t_eval_sec = simulation_time_sec if simulation_time_sec is not None else t_prog_sec
    else:
        raise TypeError("temperature_program must be a callable T(t) or a (time_sec, temp_K) tuple.")
    if t_eval_sec is None:
        raise ValueError("simulation_time_sec is required when temperature_program is callable.")
    t_eval_sec = np.asarray(t_eval_sec, dtype=float)
    temp_eval_K = np.broadcast_to(np.asarray(temp_func(t_eval_sec), dtype=float), t_eval_sec.shape).copy()
    return temp_func, t_eval_sec, temp_eval_K


def _constant_temperature(temp_eval_K: np.ndarray, tolerance_K: float = 0.5) -> Optional[float]:
    """Returns the temperature if the evaluated program is isothermal, else None."""
    if temp_eval_K.size and np.all(np.isfinite(temp_eval_K)) and np.ptp(temp_eval_K) < 2 * tolerance_K:
        return float(np.mean(temp_eval_K))
    return None

# --- Helper to prepare the full ODE parameter dictionary ---
def _prepare_full_params_for_ode(
    params_template: Dict,
    current_params_logA: Dict
    ) -> Dict:
    """Merges template and current logA params, converting logA->A."""
    # Shallow copies suffice: only the top level and the nested parameter dicts are
    # mutated below (the template also holds functions, which deepcopy would walk).
    full_params_ode = {k: (dict(v) if isinstance(v, dict) else v) for k, v in params_template.items()}
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

    # Fitted shape parameters (Fn: n; SB: m, n; SB_mnp: m, n, p) are optimizer
    # outputs that live at the top level; fold them into f_alpha_params once per
    # solve so the ODE right-hand side doesn't rebuild that dict on every call.
    fitted_shape = full_params_ode.pop('fitted_shape_params', None)
    if fitted_shape:
        f_params = dict(full_params_ode.get('f_alpha_params') or {})
        for name in fitted_shape:
            if name in full_params_ode:
                f_params[name] = full_params_ode[name]
        full_params_ode['f_alpha_params'] = f_params

    return full_params_ode


# f(alpha) is 0 (Avrami) or infinite (diffusion) at alpha=0, so a simulation started at
# exactly 0 never moves. Seed single-step models with a negligible conversion instead.
ALPHA_SEED = 1e-6


def _initial_state(model_name: str, state_dim: int, initial_alpha: float = 0.0) -> np.ndarray:
    if model_name == "A->B->C":
        return np.array([1.0 - initial_alpha, 0.0])
    if model_name in ("single_step", "SB2"):
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
    solver_options: Optional[Dict] = None,
    return_final_state: bool = False
    ) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Simulates conversion with solver fallback. Merges template and logA params.
    Returns (time_array, alpha_array) or (time_array, alpha_array, final_state_array)
    if return_final_state is True.

    Solver selection comes from resolve_solver_options(solver_options); see
    DEFAULT_SOLVER_OPTIONS.
    """
    solver_options = resolve_solver_options(solver_options)
    primary_solver = solver_options['primary_solver']
    fallback_solver = solver_options['fallback_solver']
    common_solver_kwargs = {
        'rtol': solver_options['rtol'],
        'atol': solver_options['atol'],
    }

    # --- Prepare the full parameter dict for the ODE solver ---
    try:
        full_params_ode = _prepare_full_params_for_ode(params_template, current_params_logA)
    except Exception as e_prep:
        warnings.warn(f"Simulate: Error preparing ODE params: {e_prep}")
        alpha_nan = np.full((len(t_eval),), np.nan); return t_eval, alpha_nan

    # --- Prepare time evaluation ---
    sort_indices = np.argsort(t_eval, kind='stable')
    t_eval_sorted = t_eval[sort_indices]
    t_eval_unique, unique_inverse = np.unique(t_eval_sorted, return_inverse=True)
    t_start, t_end = t_eval_unique[0], t_eval_unique[-1]
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
                fun=_with_eval_budget(ode_system, solver_options['max_rhs_evals']),
                t_span=(t_start, t_end),
                y0=initial_state,
                t_eval=t_eval_unique,
                args=(temp_func, full_params_ode),
                method=solver,
                **current_solver_kwargs
            )
            success = sol.success
            if not success and solver == primary_solver: warnings.warn(f"Primary solver '{solver}' failed: {sol.message}. Trying fallback.")
            elif not success and solver == fallback_solver: warnings.warn(f"Fallback solver '{solver}' failed: {sol.message}")
            # Informational only; the default "once per location" warning filter keeps
            # this from repeating on every one of an optimizer's many solves.
            elif success and solver == fallback_solver and solver != primary_solver:
                warnings.warn(f"Primary solver '{primary_solver}' failed; fallback solver '{solver}' succeeded.")
        except Exception as e_solve: warnings.warn(f"Solver '{solver}' failed execution: {e_solve}"); success = False
        if success: break # Exit loop if successful

    # --- Process Result or Handle Failure ---
    if success and sol is not None:
        state_sim_unique = sol.y; alpha_sim_unique = np.zeros_like(sol.t); ode_func_name = ode_system.__name__
        # Determine alpha based on model
        if ode_func_name == 'ode_system_single_step': alpha_sim_unique = state_sim_unique[0, :]
        elif ode_func_name == 'ode_system_A_plus_B_C': alpha_sim_unique = state_sim_unique[0, :]
        elif ode_func_name == 'ode_system_A_B_C': alpha_sim_unique = 1.0 - state_sim_unique[0, :] - state_sim_unique[1, :]
        else: alpha_sim_unique = state_sim_unique[0, :] # Default assumption
        alpha_sim_sorted = np.clip(alpha_sim_unique, 0.0, 1.0)[unique_inverse]
        unsort_indices = np.argsort(sort_indices, kind='stable')
        alpha_sim_unsorted = alpha_sim_sorted[unsort_indices]
        if return_final_state:
            final_state_sorted = state_sim_unique[:, -1]
            return t_eval, alpha_sim_unsorted, final_state_sorted
        else:
            return t_eval, alpha_sim_unsorted
    else:
        warnings.warn("Both solvers failed or simulation error occurred."); alpha_nan = np.full((len(t_eval),), np.nan)
        if return_final_state:
            return t_eval, alpha_nan, initial_state
        else:
            return t_eval, alpha_nan

def simulate_kinetics(
    model_name: str,
    model_definition_args: Dict,
    kinetic_params: Dict[str, float],
    initial_alpha: float,
    temperature_program: Union[Callable, Tuple[np.ndarray, np.ndarray]],
    simulation_time_sec: Optional[np.ndarray] = None,
    solver_options: Optional[Dict] = None,
) -> PredictionResult:
    """Simulates conversion for a model, A-scale parameters, and temperature program.

    Isothermal programs for single-step models with an analytic solution
    (models.CLOSED_FORM_REGISTRY) are evaluated in closed form; everything else is
    integrated with solve_ivp. Long ODE predictions (> solver_options['chunk_size']
    points) are integrated in chunks, carrying the full state between chunks.
    """
    solver_options = resolve_solver_options(solver_options)
    temp_func, t_eval_sec, temp_eval_K = _resolve_temperature_program(temperature_program, simulation_time_sec)

    # Empirical models use direct prediction, not ODE simulation
    if model_name.startswith('Empirical_'):
        from akts.empirical import predict_empirical
        from .datatypes import FitResult
        fit_res = FitResult(
            model_name=model_name, parameters=kinetic_params, success=True,
            message="Empirical model prediction", rss=0.0,
            n_datapoints=len(t_eval_sec), n_parameters=len(kinetic_params),
            model_definition_args=model_definition_args,
        )
        # Empirical models are isothermal fits; evaluate at the mean temperature.
        alpha_pred = predict_empirical(fit_res, t_eval_sec, float(np.mean(temp_eval_K)))
        return PredictionResult(time=t_eval_sec, conversion=alpha_pred, temperature=temp_eval_K)

    try:
        params_logA = params_A_to_logA(kinetic_params, model_name)
    except ValueError as e:
        raise ValueError(f"Model setup error during simulation: {e}")

    # Closed-form fast path (exact, ~1000x faster than ODE integration).
    if model_name == "single_step" and initial_alpha == 0.0:
        from .models import CLOSED_FORM_REGISTRY
        alpha_of_kt = CLOSED_FORM_REGISTRY.get(model_definition_args.get('f_alpha_model'))
        T_const = _constant_temperature(temp_eval_K)
        if alpha_of_kt is not None and T_const is not None and 'Ea' in kinetic_params and 'A' in kinetic_params:
            t0 = float(np.min(t_eval_sec))
            _, alpha_pred = _simulate_single_dataset_closed_form(
                t_eval_sec - t0, T_const, alpha_of_kt, float(kinetic_params['Ea']), float(kinetic_params['A']))
            return PredictionResult(time=t_eval_sec, temperature=temp_eval_K,
                                    conversion=np.asarray(alpha_pred, dtype=float), conversion_ci=None)

    try:
        ode_func, _, initial_state_dim, params_template = get_model_info(model_name, **model_definition_args)
    except Exception as e:
        raise ValueError(f"Model setup error during simulation: {e}")
    initial_state = _initial_state(model_name, initial_state_dim, initial_alpha)

    chunk_size = solver_options['chunk_size']
    n_points = len(t_eval_sec)

    if n_points <= chunk_size:
        t_pred_sec, alpha_pred = _simulate_single_dataset(
            t_eval=t_eval_sec, temp_func=temp_func, ode_system=ode_func,
            initial_state=initial_state, params_template=params_template,
            current_params_logA=params_logA, solver_options=solver_options
        )
    else:
        # Long prediction - chunk it, preserving full state continuity
        t_chunks, alpha_chunks = [], []
        current_state = initial_state.copy()
        for i in range(0, n_points, chunk_size):
            t_chunk = t_eval_sec[i:min(i + chunk_size, n_points)]
            t_sim, alpha_sim, final_state = _simulate_single_dataset(
                t_eval=t_chunk, temp_func=temp_func, ode_system=ode_func,
                initial_state=current_state, params_template=params_template,
                current_params_logA=params_logA, solver_options=solver_options,
                return_final_state=True
            )
            t_chunks.append(t_sim)
            alpha_chunks.append(alpha_sim)
            current_state = final_state.copy()
        t_pred_sec = np.concatenate(t_chunks)
        alpha_pred = np.concatenate(alpha_chunks)

    return PredictionResult(time=t_pred_sec, temperature=temp_eval_K, conversion=alpha_pred, conversion_ci=None)
