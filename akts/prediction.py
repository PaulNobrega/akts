"""
Forward prediction (with optional bootstrap confidence bands) from a FitResult
or a Friedman IsoResult.
"""
import numpy as np
import warnings
from typing import Dict, List, Tuple, Optional, Callable, Union
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

from .datatypes import FitResult, BootstrapResult, PredictionResult, IsoResult
from .utils import R_GAS
from .simulation import (simulate_kinetics, _with_eval_budget, _resolve_temperature_program,
                         resolve_solver_options)


def predict_conversion_model_free(
    iso_result: IsoResult,
    temperature_program: Union[Callable, Tuple[np.ndarray, np.ndarray]],
    simulation_time_sec: Optional[np.ndarray] = None,
    initial_alpha: float = 0.0,
    solver_options: Optional[Dict] = None
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
        'primary_solver' (default LSODA), 'rtol', 'atol', 'max_rhs_evals' -- see
        simulation.DEFAULT_SOLVER_OPTIONS.

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

    solver_options = resolve_solver_options(solver_options)
    alpha_fit = iso_result.alpha[valid]
    Ea_interp = interp1d(alpha_fit, iso_result.Ea[valid], bounds_error=False,
                         fill_value=(iso_result.Ea[valid][0], iso_result.Ea[valid][-1]))
    lnAf_interp = interp1d(alpha_fit, iso_result.ln_A_f_alpha[valid], bounds_error=False,
                           fill_value=(iso_result.ln_A_f_alpha[valid][0], iso_result.ln_A_f_alpha[valid][-1]))
    alpha_max = float(alpha_fit.max())

    temp_func, t_eval_sec, _ = _resolve_temperature_program(temperature_program, simulation_time_sec)

    def rhs(t, y):
        alpha = min(max(y[0], 0.0), alpha_max)  # interpolators are undefined/edge-held past this
        T = float(temp_func(t))
        if T <= 0:
            return [0.0]
        rate = np.exp(float(lnAf_interp(alpha))) * np.exp(-float(Ea_interp(alpha)) / (R_GAS * T))
        return [max(0.0, rate) if np.isfinite(rate) else 0.0]

    t_eval_sorted = np.sort(t_eval_sec)
    t_start, t_end = t_eval_sorted[0], t_eval_sorted[-1]
    # 'method' is accepted as a legacy alias for 'primary_solver'.
    method = solver_options.get('method', solver_options['primary_solver'])
    sol = solve_ivp(
        fun=_with_eval_budget(rhs, solver_options['max_rhs_evals']),
        t_span=(t_start, t_end), y0=[initial_alpha], t_eval=t_eval_sorted, method=method,
        rtol=solver_options['rtol'], atol=solver_options['atol'],
    )
    if not sol.success:
        warnings.warn(f"Model-free prediction ODE solve failed: {sol.message}")
        alpha_out = np.full_like(t_eval_sorted, np.nan)
    else:
        alpha_out = np.clip(sol.y[0], 0.0, 1.0)

    temp_eval_K = np.broadcast_to(np.asarray(temp_func(t_eval_sorted), dtype=float), t_eval_sorted.shape).copy()
    return PredictionResult(time=t_eval_sorted, temperature=temp_eval_K, conversion=alpha_out, conversion_ci=None)


# Bootstrap replicates whose predicted final conversion is below this are treated as
# degenerate (e.g. a replicate that drifted to a near-zero rate constant) and are
# excluded from the CI band -- otherwise they drag the lower bound to zero.
DEGENERATE_FINAL_CONVERSION = 0.01
MIN_VALID_REPLICATES_FOR_CI = 10


def _ci_from_replicate_curves(
    curves: np.ndarray,
    confidence_level: float,
    label: str = "",
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Percentile CI band from an (n_replicates, n_times) array of conversion curves.

    Replicates with a non-finite or < DEGENERATE_FINAL_CONVERSION final value are
    dropped when at least MIN_VALID_REPLICATES_FOR_CI replicates remain; otherwise
    all replicates are used and a warning is issued. Returns None if no replicate
    produced a finite curve.
    """
    prefix = f"{label}: " if label else ""
    curves = np.asarray(curves, dtype=float)
    if curves.ndim != 2 or curves.size == 0 or not np.any(np.isfinite(curves)):
        warnings.warn(f"{prefix}Could not calculate CIs; no successful bootstrap simulations.")
        return None

    n_total = curves.shape[0]
    n_successful = int(np.sum(np.isfinite(curves[:, 0])))
    if n_successful < 0.5 * n_total:
        warnings.warn(f"{prefix}Only {n_successful}/{n_total} bootstrap simulations succeeded. CI may be unreliable.")

    final_conversions = curves[:, -1]
    valid_mask = np.isfinite(final_conversions) & (final_conversions >= DEGENERATE_FINAL_CONVERSION)
    n_valid = int(np.sum(valid_mask))
    if n_valid < n_total:
        warnings.warn(f"{prefix}Filtered {n_total - n_valid}/{n_total} degenerate bootstrap samples "
                      f"(final conversion < {DEGENERATE_FINAL_CONVERSION:.0%}). "
                      f"CI calculation uses {n_valid} valid samples.")
    if n_valid >= MIN_VALID_REPLICATES_FOR_CI:
        used = curves[valid_mask, :]
    else:
        warnings.warn(f"{prefix}Only {n_valid} valid bootstrap samples - CI may be unreliable.")
        used = curves

    tail = (1.0 - confidence_level) / 2.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        lower = np.nanpercentile(used, tail * 100.0, axis=0)
        upper = np.nanpercentile(used, (1.0 - tail) * 100.0, axis=0)
    if np.all(lower == 0) and np.max(upper) > 0:
        warnings.warn(f"{prefix}Lower CI bound is all zeros - bootstrap may have insufficient variation or failures.")
    return lower, upper


def _empty_prediction() -> PredictionResult:
    return PredictionResult(time=np.array([]), temperature=np.array([]), conversion=np.array([]))


def predict_conversion(
    kinetic_description: Union[FitResult, IsoResult],
    temperature_program: Union[Callable, Tuple[np.ndarray, np.ndarray]],
    simulation_time_sec: Optional[np.ndarray] = None,
    initial_alpha: float = 0.0,
    solver_options: Optional[Dict] = None,
    bootstrap_result: Optional[BootstrapResult] = None,
) -> PredictionResult:
    """Predicts conversion using fitted parameters. Calculates CI if bootstrap results provided."""
    if isinstance(kinetic_description, IsoResult):
        if kinetic_description.Ea is not None and kinetic_description.ln_A_f_alpha is not None:
            return predict_conversion_model_free(
                iso_result=kinetic_description, temperature_program=temperature_program,
                simulation_time_sec=simulation_time_sec, initial_alpha=initial_alpha,
                solver_options=solver_options
            )
        warnings.warn("Prediction from this IsoResult not implemented "
                      "(method lacks ln_A_f_alpha; only run_friedman() provides it).")
        return _empty_prediction()
    if not isinstance(kinetic_description, FitResult):
        raise TypeError("kinetic_description must be FitResult or IsoResult.")

    fit_result = kinetic_description
    if not fit_result.success:
        warnings.warn("Cannot predict from unsuccessful fit.")
        return _empty_prediction()

    if fit_result.model_name == "Friedman":
        # Friedman is wrapped as a FitResult (helpers._wrap_friedman_as_fit_result)
        # so it flows through ranking/selection/reporting like any other model,
        # but it has no Ea/A "parameters" -- the real IsoResult is stashed in
        # model_definition_args, the same "everything needed to predict again"
        # role that field plays for every other model.
        iso_result = (fit_result.model_definition_args or {}).get('iso_result')
        if iso_result is None:
            warnings.warn("Friedman FitResult missing stashed iso_result; cannot predict.")
            return _empty_prediction()

        base_prediction = predict_conversion_model_free(
            iso_result, temperature_program, simulation_time_sec, initial_alpha, solver_options
        )
        if (bootstrap_result is not None and bootstrap_result.model_name == "Friedman"
                and bootstrap_result.raw_parameter_list):
            t_eval_sec = base_prediction.time
            replicate_alphas: List[np.ndarray] = []
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
                base_prediction.conversion_ci = _ci_from_replicate_curves(
                    np.array(replicate_alphas), bootstrap_result.confidence_level, label="Friedman")
        return base_prediction

    model_definition_args = getattr(fit_result, 'model_definition_args', None)
    if model_definition_args is None:
        warnings.warn("FitResult missing 'model_definition_args'. Prediction may fail or be incorrect.")
        model_definition_args = {}
    model_name = fit_result.model_name
    params_A = fit_result.parameters

    base_prediction = simulate_kinetics(
        model_name=model_name, model_definition_args=model_definition_args, kinetic_params=params_A,
        initial_alpha=initial_alpha, temperature_program=temperature_program,
        simulation_time_sec=simulation_time_sec, solver_options=solver_options)

    if (bootstrap_result is not None and bootstrap_result.model_name == model_name
            and bootstrap_result.n_iterations > 0):
        t_eval_sec = base_prediction.time
        param_dist_A = bootstrap_result.parameter_distributions
        param_names_A = list(params_A.keys())
        missing = [p for p in param_names_A if p not in param_dist_A]
        if missing:
            warnings.warn(f"Bootstrap parameter distributions missing {missing}. Skipping CI calculation.")
        else:
            n_boot = min(bootstrap_result.n_iterations, min(len(param_dist_A[p]) for p in param_names_A))
            curves = np.full((n_boot, len(t_eval_sec)), np.nan)
            for i in range(n_boot):
                boot_params_A = {p: float(param_dist_A[p][i]) for p in param_names_A}
                try:
                    boot_pred = simulate_kinetics(
                        model_name=model_name, model_definition_args=model_definition_args,
                        kinetic_params=boot_params_A, initial_alpha=initial_alpha,
                        temperature_program=temperature_program, simulation_time_sec=t_eval_sec,
                        solver_options=solver_options)
                    if len(boot_pred.conversion) == len(t_eval_sec):
                        curves[i, :] = boot_pred.conversion
                except Exception as e_boot_sim:
                    warnings.warn(f"Sim failed for bootstrap replicate {i}: {e_boot_sim}")
            base_prediction.conversion_ci = _ci_from_replicate_curves(curves, bootstrap_result.confidence_level)

    return base_prediction
