"""
Model fitting: objective functions, fit statistics, fit_kinetic_model and
discover_kinetic_models.

All optimizers work on the internal logA parameter encoding (see
simulation.params_A_to_logA) through the Arrhenius reparameterization
(simulation._ArrheniusReparam). Two optimization paths exist:

- closed form: single-step models with an analytic isothermal solution
  (models.CLOSED_FORM_REGISTRY) fitted to all-isothermal data use
  scipy.optimize.least_squares on the analytic alpha(kt) -- no ODE solves;
- ODE: everything else uses a coarse ln k(T_ref) scan followed by Powell on the
  weighted conversion RSS, falling back to a rate-residual objective if that
  fails.

Both paths are implemented once in _optimize(), which fit_kinetic_model() and the
bootstrap replicate refits (bootstrap._fit_on_resampled_data) share.
"""
import numpy as np
import warnings
import time
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Callable, Union
from scipy.optimize import minimize, least_squares, OptimizeResult

from .datatypes import KineticDataset, FitResult
from .models import get_model_info, get_log_param_names, has_closed_form, CLOSED_FORM_REGISTRY
from .utils import (get_temperature_interpolator, calculate_aic, calculate_bic, numerical_diff,
                    is_isothermal, calculate_durbin_watson, check_physical_plausibility)
from .simulation import (_split_logA_name, _ArrheniusReparam, _initial_state,
                         _simulate_single_dataset, _simulate_single_dataset_closed_form,
                         params_logA_to_A)
from .ranking import rank_models


# Conversion points in the kinetically informative transition region get this much
# more weight than baseline/plateau points, both in the fit objective and in the
# bootstrap's residual resampling. Plateau points carry little information about
# the rate, and without the weighting a fit can "win" by matching the plateau.
TRANSITION_WEIGHT = 10.0
BASELINE_WEIGHT = 1.0
TRANSITION_ALPHA_RANGE = (0.05, 0.95)  # exclusive bounds on observed conversion

# Returned by objectives in place of inf/NaN so derivative-free optimizers keep going.
OBJECTIVE_FAILURE_VALUE = 1e30
RESIDUAL_FAILURE_VALUE = 1e6


def transition_weights(conversion: np.ndarray) -> np.ndarray:
    """Per-point weights: TRANSITION_WEIGHT inside TRANSITION_ALPHA_RANGE, else BASELINE_WEIGHT."""
    lo, hi = TRANSITION_ALPHA_RANGE
    conversion = np.asarray(conversion, dtype=float)
    return np.where((conversion > lo) & (conversion < hi), TRANSITION_WEIGHT, BASELINE_WEIGHT)


# --- Shared simulation context ---
@dataclass
class _SimulationContext:
    """Everything an objective needs that does not change between evaluations.

    Built once per optimization instead of on every objective call (the previous
    objectives re-ran get_model_info() and rebuilt a temperature interpolator per
    dataset on every single evaluation).
    """
    model_name: str
    ode_func: Callable
    initial_state: np.ndarray
    params_template: Dict
    temp_funcs: List[Optional[Callable]]
    solver_options: Optional[Dict]
    alpha_of_kt_func: Optional[Callable] = None  # set => closed-form evaluation
    isothermal_T: Optional[List[Optional[float]]] = None


def _build_context(
    datasets: List[KineticDataset],
    model_name: str,
    model_definition_args: Dict,
    solver_options: Optional[Dict],
    use_closed_form: bool = False,
    alpha_of_kt_func: Optional[Callable] = None,
) -> _SimulationContext:
    ode_func, _, initial_state_dim, params_template = get_model_info(model_name, **model_definition_args)
    temp_funcs = [get_temperature_interpolator(ds.time, ds.temperature) if len(ds.time) >= 2 else None
                  for ds in datasets]
    isothermal_T = None
    if use_closed_form:
        isothermal_T = [float(np.mean(ds.temperature)) if is_isothermal(ds.temperature) else None
                        for ds in datasets]
    return _SimulationContext(
        model_name=model_name, ode_func=ode_func,
        initial_state=_initial_state(model_name, initial_state_dim),
        params_template=params_template, temp_funcs=temp_funcs,
        solver_options=solver_options,
        alpha_of_kt_func=alpha_of_kt_func if use_closed_form else None,
        isothermal_T=isothermal_T,
    )


def _simulate_dataset(ctx: _SimulationContext, i: int, ds: KineticDataset,
                      params_logA: Dict[str, float], t_eval: Optional[np.ndarray] = None) -> np.ndarray:
    """Simulated conversion for dataset i at t_eval (default: its own time points)."""
    t_eval = ds.time if t_eval is None else t_eval
    if ctx.alpha_of_kt_func is not None and ctx.isothermal_T is not None and ctx.isothermal_T[i] is not None:
        logA = params_logA.get('logA')
        _, alpha = _simulate_single_dataset_closed_form(
            t_eval=t_eval, T_const=ctx.isothermal_T[i], alpha_of_kt_func=ctx.alpha_of_kt_func,
            Ea=params_logA.get('Ea'), A=np.exp(logA) if logA is not None else None)
        return alpha
    _, alpha = _simulate_single_dataset(
        t_eval=t_eval, temp_func=ctx.temp_funcs[i], ode_system=ctx.ode_func,
        initial_state=ctx.initial_state, params_template=ctx.params_template,
        current_params_logA=params_logA, solver_options=ctx.solver_options)
    return alpha


def _notify_callback(callback_func: Optional[Callable], iteration_counter: List[int],
                     params_logA: Dict[str, float]) -> None:
    if callback_func:
        try:
            callback_func(iteration_counter[0], params_logA_to_A(params_logA))
        except Exception as e_cb:
            warnings.warn(f"Objective callback failed: {e_cb}")
    iteration_counter[0] += 1


def _new_counter(iteration_counter: Optional[List[int]]) -> List[int]:
    return iteration_counter if iteration_counter is not None else [0]


# --- Objective Function for Fitting ---
def _objective_function(
    params_array_logA: np.ndarray,
    param_names_logA: List[str],
    datasets: List[KineticDataset],
    model_name: str,
    model_definition_args: Dict,
    solver_options: Optional[Dict],
    callback_func: Optional[Callable] = None,
    iteration_counter: Optional[List[int]] = None,
    ctx: Optional[_SimulationContext] = None,
) -> float:
    """Weighted conversion RSS (see TRANSITION_WEIGHT) for the scalar optimizers.

    Pass a prebuilt ``ctx`` (from _build_context) to avoid re-resolving the model
    and temperature interpolators on every evaluation.
    """
    iteration_counter = _new_counter(iteration_counter)
    current_params_logA = dict(zip(param_names_logA, params_array_logA))
    _notify_callback(callback_func, iteration_counter, current_params_logA)

    if ctx is None:
        try:
            ctx = _build_context(datasets, model_name, model_definition_args, solver_options)
        except ValueError as e:
            warnings.warn(f"Objective func: Model info error: {e}")
            return np.inf

    total_weighted_rss = 0.0
    for i, ds in enumerate(datasets):
        if len(ds.time) < 2:
            continue
        try:
            alpha_sim = _simulate_dataset(ctx, i, ds, current_params_logA)
        except Exception as e:
            warnings.warn(f"Objective func: Sim error: {e}.")
            total_weighted_rss = np.inf
            break
        if len(alpha_sim) != len(ds.conversion):
            warnings.warn("Sim length mismatch.")
            continue
        residuals = ds.conversion - alpha_sim
        valid_mask = np.isfinite(residuals) & np.isfinite(alpha_sim)
        if residuals.size and not valid_mask.any():
            total_weighted_rss = np.inf
            continue
        weights = transition_weights(ds.conversion)[valid_mask]
        rss = float(np.sum(weights * residuals[valid_mask] ** 2))
        total_weighted_rss += rss if np.isfinite(rss) else np.inf

    return total_weighted_rss if np.isfinite(total_weighted_rss) else OBJECTIVE_FAILURE_VALUE


# --- Residual Vector Function for least_squares ---
def _residual_vector_function(
    params_array_logA: np.ndarray,
    param_names_logA: List[str],
    datasets: List[KineticDataset],
    model_name: str,
    model_definition_args: Dict,
    solver_options: Optional[Dict],
    use_closed_form: bool = False,
    alpha_of_kt_func: Optional[Callable] = None,
    callback_func: Optional[Callable] = None,
    iteration_counter: Optional[List[int]] = None,
    ctx: Optional[_SimulationContext] = None,
) -> np.ndarray:
    """
    Flattened residual vector for scipy.optimize.least_squares.

    Same weighting as _objective_function, applied as sqrt(weight) on each
    residual so that least_squares' sum of squares equals the weighted RSS.
    Isothermal datasets use the closed-form alpha(kt) when use_closed_form=True.
    """
    iteration_counter = _new_counter(iteration_counter)
    current_params_logA = dict(zip(param_names_logA, params_array_logA))
    _notify_callback(callback_func, iteration_counter, current_params_logA)

    if ctx is None:
        try:
            ctx = _build_context(datasets, model_name, model_definition_args, solver_options,
                                 use_closed_form=use_closed_form, alpha_of_kt_func=alpha_of_kt_func)
        except ValueError as e:
            warnings.warn(f"Residual func: Model info error: {e}")
            return np.full(sum(len(ds.conversion) for ds in datasets), RESIDUAL_FAILURE_VALUE)

    residuals_list = []
    for i, ds in enumerate(datasets):
        if len(ds.time) < 2:
            continue
        try:
            alpha_sim = _simulate_dataset(ctx, i, ds, current_params_logA)
        except Exception as e:
            warnings.warn(f"Residual func: Sim error: {e}.")
            residuals_list.append(np.full(len(ds.conversion), RESIDUAL_FAILURE_VALUE))
            continue
        if len(alpha_sim) != len(ds.conversion):
            warnings.warn("Residual: Sim length mismatch.")
            residuals_list.append(np.full(len(ds.conversion), RESIDUAL_FAILURE_VALUE))
            continue
        residuals = ds.conversion - alpha_sim
        valid_mask = np.isfinite(residuals) & np.isfinite(alpha_sim)
        sqrt_weights = np.sqrt(transition_weights(ds.conversion))
        residuals_list.append(residuals[valid_mask] * sqrt_weights[valid_mask])

    if not residuals_list:
        return np.array([RESIDUAL_FAILURE_VALUE])
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
    solver_options: Optional[Dict],
    callback_func: Optional[Callable] = None,
    iteration_counter: Optional[List[int]] = None,
    ctx: Optional[_SimulationContext] = None,
) -> float:
    """Total RSS between simulated and experimental RATES (fallback objective)."""
    iteration_counter = _new_counter(iteration_counter)
    current_params_logA = dict(zip(param_names_logA, params_array_logA))
    _notify_callback(callback_func, iteration_counter, current_params_logA)

    if ctx is None:
        try:
            ctx = _build_context(datasets, model_name, model_definition_args, solver_options)
        except ValueError as e:
            warnings.warn(f"Objective func (rate): Model info error: {e}")
            return np.inf

    total_rate_rss = 0.0
    for i, ds in enumerate(datasets):
        if len(ds.time) < 2 or i >= len(exp_rates) or i >= len(exp_times):
            continue
        try:
            t_sim_eval = exp_times[i]
            alpha_sim = _simulate_dataset(ctx, i, ds, current_params_logA, t_eval=t_sim_eval)
            if len(alpha_sim) != len(t_sim_eval):
                warnings.warn("Rate objective: Sim length mismatch.")
                continue
            rate_sim = numerical_diff(t_sim_eval, alpha_sim)
            rate_exp = exp_rates[i]
            if len(rate_sim) != len(rate_exp):
                warnings.warn("Rate objective: Rate length mismatch after diff.")
                continue
            rate_residuals = rate_exp - rate_sim
            valid_mask = np.isfinite(rate_residuals) & np.isfinite(rate_exp) & np.isfinite(rate_sim)
            rss = np.sum(rate_residuals[valid_mask] ** 2) if np.sum(valid_mask) > 0 else np.inf
            total_rate_rss += rss if np.isfinite(rss) else np.inf
        except Exception as e:
            warnings.warn(f"Objective func (rate): Sim/Diff error: {e}.")
            total_rate_rss = np.inf
            break

    return total_rate_rss if np.isfinite(total_rate_rss) else OBJECTIVE_FAILURE_VALUE


# --- Fit statistics ---
def _calculate_conversion_stats(
    datasets: List[KineticDataset],
    params_logA: Dict,
    model_name: str,
    model_definition_args: Dict,
    solver_options: Optional[Dict],
    return_residuals: bool = False,
    use_closed_form: bool = False,
) -> Union[Tuple[float, int, float, float, float], Tuple[float, int, float, float, float, np.ndarray]]:
    """Calculates unweighted conversion RSS, N, R², AICc, and BIC for a parameter set.

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
    use_closed_form : bool, optional
        Evaluate isothermal datasets with the model's analytic solution when it
        has one (see _closed_form_function).

    Returns
    -------
    Tuple[float, int, float, float, float] or Tuple[float, int, float, float, float, np.ndarray]
        (rss, n_pts, r_squared, aic, bic) or (rss, n_pts, r_squared, aic, bic, residuals)
    """
    rss = float(np.inf)
    n_pts = 0  # Plain Python int throughout -- np.sum(bool_array) returns np.int64,
               # which would otherwise leak into FitResult.n_datapoints.
    all_exp_conv = []
    all_residuals = []
    n_params = len(params_logA)

    try:
        alpha_of_kt = _closed_form_function(model_name, model_definition_args) if use_closed_form else None
        ctx = _build_context(datasets, model_name, model_definition_args, solver_options,
                             use_closed_form=alpha_of_kt is not None, alpha_of_kt_func=alpha_of_kt)
        current_rss_calc = 0.0
        for i, ds in enumerate(datasets):
            if len(ds.time) < 2:
                continue
            alpha_sim = _simulate_dataset(ctx, i, ds, params_logA)
            if len(alpha_sim) == len(ds.conversion):
                residuals = ds.conversion - alpha_sim
                valid_mask = np.isfinite(residuals) & np.isfinite(alpha_sim)
                n_valid = int(np.sum(valid_mask))
                if n_valid > 0:
                    valid_residuals = residuals[valid_mask]
                    current_rss_calc += float(np.sum(valid_residuals ** 2))
                    n_pts += n_valid
                    all_exp_conv.extend(ds.conversion[valid_mask])
                    if return_residuals:
                        all_residuals.extend(valid_residuals)
        if n_pts > 0:
            rss = current_rss_calc
    except Exception as e_sim:
        warnings.warn(f"Failed to simulate conversion curve for stats calculation: {e_sim}")

    r_squared = float(np.nan)
    aic = float(np.nan)
    bic = float(np.nan)
    if n_pts > n_params and np.isfinite(rss):
        mean_y = np.mean(all_exp_conv)
        total_ss = float(np.sum((np.array(all_exp_conv) - mean_y) ** 2))
        r_squared = 1.0 - (rss / total_ss) if total_ss > 1e-12 else (1.0 if rss < 1e-12 else 0.0)
        aic = float(calculate_aic(rss, n_params, n_pts))
        bic = float(calculate_bic(rss, n_params, n_pts))

    if return_residuals:
        return rss, n_pts, r_squared, aic, bic, np.array(all_residuals)
    return rss, n_pts, r_squared, aic, bic


# --- Optimization core shared by fit_kinetic_model and bootstrap refits ---
def _sum_sq_residuals(params_array_logA: np.ndarray, *residual_args) -> float:
    """Scalar objective equal to least_squares' cost x2, for the coarse start scan."""
    r = _residual_vector_function(params_array_logA, *residual_args)
    value = float(np.sum(r ** 2))
    return value if np.isfinite(value) else OBJECTIVE_FAILURE_VALUE


def _closed_form_function(model_name: str, model_definition_args: Dict) -> Optional[Callable]:
    """The analytic alpha(kt) for single-step models that have one, else None."""
    if model_name != "single_step":
        return None
    f_alpha_model = model_definition_args.get('f_alpha_model')
    if f_alpha_model and has_closed_form(f_alpha_model):
        return CLOSED_FORM_REGISTRY[f_alpha_model]
    return None


def _closed_form_eligible(datasets: List[KineticDataset], model_name: str,
                          model_definition_args: Dict) -> Optional[Callable]:
    """alpha(kt) if every dataset is isothermal and the model has a closed form, else None."""
    alpha_of_kt = _closed_form_function(model_name, model_definition_args)
    if alpha_of_kt is not None and all(is_isothermal(ds.temperature) for ds in datasets):
        return alpha_of_kt
    return None


def _log_param_names_for(model_name: str, model_definition_args: Dict) -> Tuple[List[str], Dict[str, str]]:
    """(internal logA parameter names in fit order, {logA name: A-scale name})."""
    _, original_param_names, _, _ = get_model_info(model_name, **model_definition_args)
    log_param_names = get_log_param_names(model_name)
    names_logA, name_map = [], {}
    for name in original_param_names:
        logA_name = "log" + name if name in log_param_names else name
        names_logA.append(logA_name)
        name_map[logA_name] = name
    return names_logA, name_map


def _default_bounds_logA(param_names_logA: List[str]) -> Dict[str, Tuple[float, float]]:
    bounds = {}
    for p_name in param_names_logA:
        if p_name.startswith("Ea"):
            bounds[p_name] = (1e3, 600e3)
        elif p_name.startswith("logA"):
            bounds[p_name] = (np.log(1e-2), np.log(1e25))
        elif p_name.endswith("n") or p_name.endswith("m") or p_name.startswith("p1_") or p_name.startswith("p2_"):
            bounds[p_name] = (0, 8)
        elif p_name == "initial_ratio_r":
            bounds[p_name] = (1e-3, 1e3)
        else:
            bounds[p_name] = (-np.inf, np.inf)
    return bounds


def _bounds_A_to_logA(parameter_bounds: Optional[Dict[str, Tuple[float, float]]],
                      param_names_logA: List[str], name_map: Dict[str, str]) -> Dict[str, Tuple[float, float]]:
    """User A-scale bounds -> logA-scale bounds for the parameters that have them."""
    out: Dict[str, Tuple[float, float]] = {}
    if not parameter_bounds:
        return out
    for p_logA in param_names_logA:
        user_bound = parameter_bounds.get(name_map[p_logA])
        if user_bound is None:
            continue
        lo, hi = user_bound
        if p_logA.startswith("logA"):
            out[p_logA] = (np.log(lo) if lo is not None and lo > 0 else -np.inf,
                           np.log(hi) if hi is not None and hi > 0 else np.inf)
        else:
            out[p_logA] = (lo, hi)
    return out


def _bounds_list(bounds_logA: Dict[str, Tuple[float, float]], param_names_logA: List[str]) -> List[Tuple]:
    """Ordered [(lo|None, hi|None)] aligned with param_names_logA (scipy's format)."""
    out = []
    for p in param_names_logA:
        lo, hi = bounds_logA.get(p, (-np.inf, np.inf))
        out.append((lo if lo is not None and np.isfinite(lo) else None,
                    hi if hi is not None and np.isfinite(hi) else None))
    return out


_METHODS_SUPPORTING_BOUNDS = ('L-BFGS-B', 'TNC', 'SLSQP', 'trust-constr', 'Powell', 'Nelder-Mead')


@dataclass
class _OptimizeOutcome:
    result: Optional[OptimizeResult]  # .x in EXTERNAL (logA) coordinates when successful
    used_rate_fallback: bool
    message: str
    reparam: Optional[_ArrheniusReparam] = None
    early_failure: bool = False  # failed before any optimizer produced a result


def _optimize(
    datasets: List[KineticDataset],
    model_name: str,
    model_definition_args: Dict,
    param_names_logA: List[str],
    x0_logA: np.ndarray,
    bounds_list_logA: Optional[List[Tuple]],
    solver_options: Optional[Dict],
    optimizer_options: Optional[Dict],
    deadline: Optional[float],
    callback: Optional[Callable] = None,
    coarse_start: bool = True,
    allow_rate_fallback: bool = True,
    powell_maxfev: int = 4000,
    verbose: bool = False,
) -> _OptimizeOutcome:
    """Runs the closed-form least_squares path or the Powell/ODE path (with rate fallback).

    On success, ``outcome.result.x`` is in external (logA) coordinates and, when
    available, ``outcome.result.hess_inv`` has been transformed to match.
    """
    optimizer_options = optimizer_options or {}
    # Powell is derivative-free: ODE solver tolerance makes the objective noisy at small
    # scales, so finite-difference gradients (L-BFGS-B) are unreliable and stall early.
    opt_method = optimizer_options.get('method', 'Powell')
    default_opt_options = ({'xtol': 1e-4, 'ftol': 1e-8, 'maxfev': powell_maxfev} if opt_method == 'Powell'
                           else {'ftol': 1e-9, 'gtol': 1e-7})
    opt_options = optimizer_options.get('options', default_opt_options)
    if not isinstance(opt_options, dict):
        opt_options = default_opt_options
    if bounds_list_logA is not None and opt_method not in _METHODS_SUPPORTING_BOUNDS:
        warnings.warn(f"Optimizer {opt_method} ignores bounds.")
        bounds_list_logA = None

    def _deadline_callback(*_args, **_kwargs):
        if deadline is not None and time.time() > deadline:
            raise StopIteration("optimization exceeded its wall-clock budget")

    reparam = _ArrheniusReparam(param_names_logA, datasets)
    x0_internal = reparam.to_internal(x0_logA)
    bounds_internal = reparam.bounds_to_internal(bounds_list_logA)

    opt_result = None
    opt_result_conv = None
    used_rate_fallback = False
    alpha_of_kt = _closed_form_eligible(datasets, model_name, model_definition_args)

    # --- Fast path: least_squares on the closed-form solution ---
    if alpha_of_kt is not None:
        if verbose:
            print(f"Using closed-form solution for {model_definition_args.get('f_alpha_model')} (all datasets isothermal)")
            print("--- Optimizing with scipy.optimize.least_squares (TRF method) ---")
        try:
            ctx = _build_context(datasets, model_name, model_definition_args, solver_options,
                                 use_closed_form=True, alpha_of_kt_func=alpha_of_kt)
            residual_args = (param_names_logA, datasets, model_name, model_definition_args,
                             solver_options, True, alpha_of_kt, None, [0], ctx)
            if coarse_start:
                # Same ln k(T_ref) scan as the ODE path: far from the answer every
                # point is fully reacted (or not at all), the residuals are flat, and
                # least_squares would stop at the initial guess.
                x0_internal = reparam.coarse_start(x0_internal, _sum_sq_residuals, residual_args,
                                                   bounds_internal, deadline=deadline)
                residual_args = residual_args[:7] + (callback,) + residual_args[8:]
            if bounds_internal:
                bounds_lsq = (np.array([b[0] if b[0] is not None else -np.inf for b in bounds_internal]),
                              np.array([b[1] if b[1] is not None else np.inf for b in bounds_internal]))
                # least_squares requires a strictly feasible start
                x0_internal = np.clip(x0_internal, bounds_lsq[0], bounds_lsq[1])
            else:
                bounds_lsq = (-np.inf, np.inf)
            opt_result = least_squares(
                fun=reparam.wrap_residual(_residual_vector_function), x0=x0_internal,
                args=residual_args, bounds=bounds_lsq, method='trf',
                ftol=1e-8, xtol=1e-8, max_nfev=2000, jac='2-point', verbose=0,
            )
            # Reconstruct hess_inv = inv(J^T J) for param_std_err propagation
            if getattr(opt_result, 'jac', None) is not None:
                try:
                    J = opt_result.jac
                    opt_result.hess_inv = np.linalg.inv(J.T @ J + 1e-12 * np.eye(len(param_names_logA)))
                except Exception as e_hess:
                    warnings.warn(f"Could not reconstruct hess_inv from Jacobian: {e_hess}")
                    opt_result.hess_inv = None
            opt_result_conv = opt_result
        except Exception as e_lsq:
            warnings.warn(f"least_squares optimization failed: {e_lsq}. Falling back to Powell.")
            alpha_of_kt = None
            opt_result = None

    # --- Standard path: Powell on the weighted conversion objective (ODE) ---
    if alpha_of_kt is None:
        if verbose:
            print("--- Attempting optimization on CONVERSION residuals (weighted) ---")
        success_conv = False
        try:
            ctx = _build_context(datasets, model_name, model_definition_args, solver_options)
            minimize_args_conv = (param_names_logA, datasets, model_name, model_definition_args,
                                  solver_options, callback, [0], ctx)
            if coarse_start:
                x0_internal = reparam.coarse_start(x0_internal, _objective_function, minimize_args_conv,
                                                   bounds_internal, deadline=deadline)
            opt_result_conv = minimize(fun=reparam.wrap(_objective_function), x0=x0_internal,
                                       args=minimize_args_conv, method=opt_method, bounds=bounds_internal,
                                       options=opt_options, callback=_deadline_callback)
            success_conv = opt_result_conv.success
            opt_result = opt_result_conv
        except Exception as e_conv:
            warnings.warn(f"Conversion-based optimization failed with exception: {e_conv}")

        # --- Fallback: rate residuals ---
        if not success_conv and allow_rate_fallback:
            warnings.warn("Conversion-based fit failed. Falling back to RATE-based optimization.")
            used_rate_fallback = True
            exp_rates, exp_times_sec, rate_datasets = [], [], []
            diff_options = {'window_length': 5, 'polyorder': 2}
            for i, ds in enumerate(datasets):
                if len(ds.time) < diff_options['window_length']:
                    warnings.warn(f"Dataset {i} too short for rate calc.")
                    continue
                exp_rates.append(numerical_diff(ds.time, ds.conversion, **diff_options))
                exp_times_sec.append(ds.time)
                rate_datasets.append(ds)
            if not rate_datasets:
                return _OptimizeOutcome(None, True, "Conversion fit failed & no datasets long enough for rate fit fallback.",
                                        reparam, early_failure=True)
            if verbose:
                print("--- Optimizing on RATE residuals ---")
            try:
                rate_ctx = _build_context(rate_datasets, model_name, model_definition_args, solver_options)
                minimize_args_rate = (param_names_logA, rate_datasets, exp_rates, exp_times_sec, model_name,
                                      model_definition_args, solver_options, callback, [0], rate_ctx)
                opt_result = minimize(fun=reparam.wrap(_objective_function_rate), x0=x0_internal,
                                      args=minimize_args_rate, method=opt_method, bounds=bounds_internal,
                                      options=opt_options, callback=_deadline_callback)
            except Exception as e_rate:
                conv_msg = opt_result_conv.message if opt_result_conv is not None else 'Exception'
                return _OptimizeOutcome(None, True,
                                        f"Conversion fit failed ({conv_msg}). Rate fit fallback also failed ({e_rate}).",
                                        reparam, early_failure=True)

    if opt_result is None or not opt_result.success:
        conv_msg = opt_result_conv.message if opt_result_conv is not None else 'Exception'
        msg = f"Optimization failed. Initial attempt: {conv_msg}. "
        if used_rate_fallback:
            msg += f"Rate fallback attempt: {opt_result.message if opt_result is not None else 'Exception'}."
        else:
            msg += "Rate fallback not attempted."
        return _OptimizeOutcome(None, used_rate_fallback, msg, reparam)

    # --- Map back to external (logA) coordinates ---
    opt_result.x = reparam.to_external(opt_result.x)
    if getattr(opt_result, 'hess_inv', None) is not None:
        try:
            h = opt_result.hess_inv.todense() if hasattr(opt_result.hess_inv, 'todense') else np.asarray(opt_result.hess_inv)
            opt_result.hess_inv = reparam.J @ h @ reparam.J.T
        except Exception:
            opt_result.hess_inv = None
    return _OptimizeOutcome(opt_result, used_rate_fallback, str(opt_result.message), reparam)


def _failed_fit(model_name: str, model_definition_args: Dict, message: str, n_parameters: int,
                used_rate_fallback: bool = False) -> FitResult:
    return FitResult(model_name=model_name, model_definition_args=model_definition_args, parameters={},
                     success=False, message=message, rss=np.inf, n_datapoints=0,
                     n_parameters=n_parameters, r_squared=np.nan, used_rate_fallback=used_rate_fallback)


# --- Main Fitting Function ---
def fit_kinetic_model(
    datasets: List[KineticDataset],
    model_name: str,
    model_definition_args: Dict,
    initial_guesses: Dict[str, float],  # Expects guesses for A (not logA)
    parameter_bounds: Optional[Dict[str, Tuple[float, float]]] = None,  # Expects bounds for A
    solver_options: Optional[Dict] = None,
    optimizer_options: Optional[Dict] = None,
    callback: Optional[Callable[[int, Dict], None]] = None,
    optimize_on_rate: bool = True,
    verbose: bool = True
) -> FitResult:
    """
    Fits a kinetic model. User provides guesses/bounds on the A scale; the
    optimizer works on ln(A) internally.

    Single-step models with an analytic isothermal solution, fitted to
    all-isothermal data, use least_squares on the closed form. Otherwise the
    weighted CONVERSION residuals are optimized with Powell (after a coarse
    ln k scan); if that fails and ``optimize_on_rate`` is True, the RATE residuals
    are optimized instead. Final statistics are always the UNWEIGHTED conversion
    fit quality.

    optimizer_options keys: 'method' (default 'Powell'), 'options' (passed to
    scipy), 'max_seconds' (wall-clock budget; default 60 s per parameter).
    solver_options: see simulation.DEFAULT_SOLVER_OPTIONS.
    """
    optimizer_options = optimizer_options or {}

    try:
        param_names_logA, name_map = _log_param_names_for(model_name, model_definition_args)
    except ValueError as e:
        return FitResult(model_name=model_name, parameters={}, success=False, message=f"Model setup error: {e}",
                         rss=np.inf, n_datapoints=0, n_parameters=0, r_squared=np.nan,
                         model_definition_args=model_definition_args)
    n_params = len(param_names_logA)

    # Wall-clock deadline, not just an eval-count cap: ODE objective calls vary
    # widely in cost (a stiff parameter region can be >50x slower than a benign
    # one), so a fixed maxfev that is safe for cheap single_step models can still
    # run for a very long time on multi-parameter ODE models. 60s/parameter by default.
    user_max_seconds = optimizer_options.get('max_seconds', None)
    max_seconds = user_max_seconds if user_max_seconds is not None else 60.0 * max(1, n_params)
    deadline = time.time() + max_seconds

    missing = [name_map[p] for p in param_names_logA if name_map[p] not in initial_guesses]
    if missing:
        return _failed_fit(model_name, model_definition_args, f"Missing initial guesses for: {missing}", n_params)
    try:
        x0_logA = []
        for p_logA in param_names_logA:
            guess = initial_guesses[name_map[p_logA]]
            if p_logA.startswith("logA"):
                if guess <= 0:
                    raise ValueError(f"Initial guess for {name_map[p_logA]} must be positive.")
                x0_logA.append(np.log(guess))
            else:
                x0_logA.append(guess)
        x0_logA = np.array(x0_logA, dtype=float)
    except (ValueError, TypeError, KeyError) as e_conv:
        return _failed_fit(model_name, model_definition_args, f"Error converting initial guess: {e_conv}", n_params)

    try:
        bounds_logA = _default_bounds_logA(param_names_logA)
        bounds_logA.update(_bounds_A_to_logA(parameter_bounds, param_names_logA, name_map))
        bounds_list_logA = _bounds_list(bounds_logA, param_names_logA)
    except Exception as e:
        return _failed_fit(model_name, model_definition_args, f"Invalid bounds format: {e}", n_params)

    outcome = _optimize(
        datasets, model_name, model_definition_args, param_names_logA, x0_logA, bounds_list_logA,
        solver_options, optimizer_options, deadline, callback=callback,
        allow_rate_fallback=optimize_on_rate, verbose=verbose,
    )
    if outcome.result is None:
        return _failed_fit(model_name, model_definition_args, outcome.message, n_params, outcome.used_rate_fallback)
    opt_result = outcome.result

    fitted_params_logA = dict(zip(param_names_logA, opt_result.x))
    # params_logA_to_A returns plain Python floats, so FitResult.parameters
    # (Dict[str, float]) never holds numpy scalars.
    fitted_params_final = params_logA_to_A(fitted_params_logA)

    # FINAL stats: UNWEIGHTED conversion fit
    final_conversion_rss, n_total_datapoints_final, r_squared, aic, bic, residuals = _calculate_conversion_stats(
        datasets, fitted_params_logA, model_name, model_definition_args, solver_options,
        return_residuals=True, use_closed_form=True,
    )
    durbin_watson = calculate_durbin_watson(residuals)
    is_plausible, plausibility_issues = check_physical_plausibility(fitted_params_final, strict=False)

    # Standard errors from the (reparam-corrected) inverse Hessian
    param_std_err_final = None
    hess_inv = getattr(opt_result, 'hess_inv', None)
    if hess_inv is not None:
        try:
            diag_hess_inv = np.diag(np.asarray(hess_inv))
            if np.all(diag_hess_inv > 0) and n_total_datapoints_final > n_params and np.isfinite(final_conversion_rss):
                sigma_sq_est = final_conversion_rss / (n_total_datapoints_final - n_params)
                std_err_logA = np.sqrt(diag_hess_inv * sigma_sq_est)
                param_std_err_final = {}
                for name_logA, se in zip(param_names_logA, std_err_logA):
                    is_logA_param, original_key = _split_logA_name(name_logA)
                    # d(A) = A * d(lnA)
                    param_std_err_final[original_key] = float(se * fitted_params_final[original_key]) if is_logA_param else float(se)
        except Exception as e:
            warnings.warn(f"Could not estimate/propagate std errors: {e}")

    for p_name, p_val in fitted_params_final.items():
        if p_name.startswith("Ea"):
            if p_val < 5e3:
                warnings.warn(f"Fitted {p_name} ({p_val/1000:.1f} kJ/mol) is very low.")
            if p_val > 400e3:
                warnings.warn(f"Fitted {p_name} ({p_val/1000:.1f} kJ/mol) is very high.")
        elif p_name.startswith("A"):
            if p_val < 1e-1:
                warnings.warn(f"Fitted {p_name} ({p_val:.1e} 1/s) is very low.")
            if p_val > 1e20:
                warnings.warn(f"Fitted {p_name} ({p_val:.1e} 1/s) is very high.")

    initial_r = None
    if model_name == "A+B->C":
        fixed_r = model_definition_args.get('bimol_params', {}).get('initial_ratio_r')
        if fixed_r is not None and 'initial_ratio_r' not in param_names_logA:
            initial_r = float(fixed_r)
        elif 'initial_ratio_r' in fitted_params_logA:
            initial_r = float(fitted_params_logA['initial_ratio_r'])

    return FitResult(
        model_name=model_name, model_definition_args=model_definition_args,
        parameters=fitted_params_final, success=bool(opt_result.success), message=str(opt_result.message),
        rss=float(final_conversion_rss), n_datapoints=int(n_total_datapoints_final), n_parameters=n_params,
        param_std_err=param_std_err_final, aic=float(aic), bic=float(bic), r_squared=float(r_squared),
        durbin_watson=durbin_watson,
        is_physically_plausible=is_plausible,
        plausibility_issues=plausibility_issues if not is_plausible else None,
        initial_ratio_r=initial_r,
        used_rate_fallback=outcome.used_rate_fallback,
    )


def discover_kinetic_models(
    datasets: List[KineticDataset],
    models_to_try: List[Dict],
    initial_guesses_pool: Dict[str, Dict],
    parameter_bounds_pool: Optional[Dict[str, Dict]] = None,
    solver_options: Optional[Dict] = None,
    optimizer_options: Optional[Dict] = None,
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

    print("\n--- Model Discovery Finished ---")
    if not all_fit_results:
        print("No models fitted successfully.")
        return []

    ranked_list = rank_models(all_fit_results, score_weights=score_weights)
    print(f"Ranking {len(ranked_list)} successful models by combined score...")

    return ranked_list
