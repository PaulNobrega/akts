"""
Residual-resampling bootstrap for fitted kinetic and empirical models.

Each replicate:

1. evaluates the best fit on every dataset,
2. draws centered residuals with replacement (weighted towards the kinetically
   informative transition region, see fitting.TRANSITION_WEIGHT) and adds them
   to the FITTED values to form a synthetic dataset,
3. refits the model to the synthetic data from a slightly perturbed start.

Replicates run in a process pool. Worker entry points are module-level so they
pickle under the Windows "spawn" start method. Each replicate gets its own
independent random stream derived from ``random_state`` via
``np.random.SeedSequence.spawn``, so results are reproducible for a fixed seed
regardless of worker count or completion order.
"""
import os
import time
import warnings
import concurrent.futures
from concurrent.futures import TimeoutError as FuturesTimeoutError
from typing import List, Dict, Tuple, Optional, Callable, Union

import numpy as np

from .datatypes import KineticDataset, FitResult, BootstrapResult
from .simulation import params_A_to_logA, params_logA_to_A
from .utils import R_GAS
from .fitting import (_calculate_conversion_stats, _build_context, _simulate_dataset,
                      _closed_form_eligible, _optimize, _bounds_A_to_logA, _bounds_list, _default_bounds_logA,
                      transition_weights)

RandomState = Optional[Union[int, np.random.SeedSequence, np.random.Generator]]
EndCallback = Optional[Callable[[int, str, Optional[Dict]], None]]
BOOTSTRAP_METHODS = ('monte_carlo', 'parametric', 'residual')


def _normalize_bootstrap_method(bootstrap_method: str) -> str:
    method = bootstrap_method.strip().lower() if isinstance(bootstrap_method, str) else ''
    if method not in BOOTSTRAP_METHODS:
        raise ValueError(f"bootstrap_method must be one of {BOOTSTRAP_METHODS}; got {bootstrap_method!r}.")
    return method


# --- Worker plumbing -------------------------------------------------------
_BOOTSTRAP_WORKER_FUNCTION = None
_BOOTSTRAP_WORKER_ARGS = None


def _initialize_bootstrap_worker(worker_function: Callable, worker_args: Tuple) -> None:
    """Set invariant bootstrap inputs once in each worker process."""
    global _BOOTSTRAP_WORKER_FUNCTION, _BOOTSTRAP_WORKER_ARGS
    _BOOTSTRAP_WORKER_FUNCTION = worker_function
    _BOOTSTRAP_WORKER_ARGS = worker_args


def _execute_bootstrap_worker(iteration_index: int, seed: Optional[np.random.SeedSequence] = None) -> Optional[Dict]:
    """Run one replicate using the context installed by the pool initializer."""
    if _BOOTSTRAP_WORKER_FUNCTION is None or _BOOTSTRAP_WORKER_ARGS is None:
        raise RuntimeError("Bootstrap worker context was not initialized")
    return _BOOTSTRAP_WORKER_FUNCTION(*_BOOTSTRAP_WORKER_ARGS, iteration_index=iteration_index, seed=seed)


def _get_bootstrap_max_workers(n_jobs: Optional[int], n_iterations: int) -> int:
    """Resolve the worker count, leaving one core free by default."""
    cpu_count = os.cpu_count() or 1
    if n_jobs is not None and n_jobs > 0:
        requested_workers = n_jobs
    else:
        requested_workers = max(1, cpu_count - 1)
    return min(requested_workers, max(1, n_iterations))


def _spawn_seeds(random_state: RandomState, n: int) -> List[np.random.SeedSequence]:
    """n independent child seeds. None => fresh OS entropy (non-reproducible)."""
    if isinstance(random_state, np.random.Generator):
        random_state = np.random.SeedSequence(int(random_state.integers(0, 2**63 - 1)))
    elif not isinstance(random_state, np.random.SeedSequence):
        random_state = np.random.SeedSequence(random_state)
    return random_state.spawn(n)


def _run_replicates(
    worker_function: Callable,
    worker_args: Tuple,
    n_iterations: int,
    n_jobs: Optional[int],
    timeout_per_replicate: Optional[float],
    random_state: RandomState,
    label: str,
) -> List[Optional[Dict]]:
    """Runs worker_function(*worker_args, iteration_index=i, seed=s_i) for each replicate.

    Uses a process pool; n_jobs=1 still uses one worker process (matching the
    previous behavior) so that a crashing replicate cannot take down the caller.
    """
    seeds = _spawn_seeds(random_state, n_iterations)
    max_workers = _get_bootstrap_max_workers(n_jobs, n_iterations)
    print(f"Starting {n_iterations} bootstrap fits for {label} using concurrent.futures (max_workers={max_workers})...")

    results: List[Optional[Dict]] = [None] * n_iterations
    try:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=_initialize_bootstrap_worker,
            initargs=(worker_function, worker_args),
        ) as executor:
            futures = [executor.submit(_execute_bootstrap_worker, i, seeds[i]) for i in range(n_iterations)]
            print(f"\nSubmitted {len(futures)} tasks. Collecting results...")
            for i, future in enumerate(futures):
                try:
                    results[i] = future.result(timeout=timeout_per_replicate)
                except FuturesTimeoutError:
                    warnings.warn(f"Bootstrap replicate {i+1} timed out after {timeout_per_replicate}s.")
                except Exception as e:
                    warnings.warn(f"Bootstrap replicate {i+1} failed with exception: {type(e).__name__}: {e}")
    except Exception as e_pool:
        warnings.warn(f"Error during bootstrap pool execution: {type(e_pool).__name__}: {e_pool}")
    return results


def _call_end_callback(end_callback_func: EndCallback, iteration_index: int, status: str,
                       data: Optional[Dict]) -> None:
    if end_callback_func:
        try:
            end_callback_func(iteration_index, status, data)
        except Exception as e_cb:
            warnings.warn(f"Bootstrap {status} callback failed: {e_cb}")


# --- Resampling ------------------------------------------------------------
def _resample_residuals(
    observed: List[np.ndarray],
    fitted: List[Optional[np.ndarray]],
    rng: np.random.Generator,
) -> List[Optional[np.ndarray]]:
    """Weighted residual bootstrap: synthetic_i = fitted_i + resampled centered residuals.

    Residuals r = observed - fitted are pooled across datasets, centered, and drawn
    with replacement -- one draw per data point -- with probability proportional to
    transition_weights(observed). Returns one synthetic conversion array per
    dataset (clipped to [0, 1]); None where the fit could not be evaluated.
    Points whose residual is non-finite keep their fitted value.
    """
    pooled, pooled_alpha, index_map = [], [], []
    for i, (obs, fit) in enumerate(zip(observed, fitted)):
        if fit is None or len(fit) != len(obs):
            continue
        r = obs - fit
        valid = np.isfinite(r)
        pooled.append(r[valid])
        pooled_alpha.append(obs[valid])
        index_map.extend((i, k) for k in np.flatnonzero(valid))
    if not index_map:
        return [None] * len(observed)

    residuals = np.concatenate(pooled)
    centered = residuals - residuals.mean()
    weights = transition_weights(np.concatenate(pooled_alpha))
    weights = weights / weights.sum()
    draws = rng.choice(len(centered), size=len(centered), replace=True, p=weights)

    synthetic: List[Optional[np.ndarray]] = [None if f is None or len(f) != len(o) else np.array(f, dtype=float)
                                             for o, f in zip(observed, fitted)]
    for slot, drawn in enumerate(draws):
        i, k = index_map[slot]
        synthetic[i][k] += centered[drawn]
    return [None if s is None else np.clip(s, 0.0, 1.0) for s in synthetic]


def _resample_cases(datasets: List[KineticDataset], rng: np.random.Generator) -> List[KineticDataset]:
    """Case bootstrap: sample complete observed rows with replacement per dataset."""
    resampled = []
    for dataset in datasets:
        n_points = len(dataset.time)
        if n_points == 0:
            continue
        indices = rng.integers(0, n_points, size=n_points)
        indices = indices[np.argsort(dataset.time[indices], kind='stable')]
        humidity = dataset.relative_humidity
        if humidity is not None and len(humidity) == n_points:
            humidity = humidity[indices]
        resampled.append(KineticDataset(
            time=dataset.time[indices],
            temperature=dataset.temperature[indices],
            conversion=dataset.conversion[indices],
            heating_rate=dataset.heating_rate,
            relative_humidity=humidity,
            metadata=dict(dataset.metadata),
        ))
    return resampled


def _resample_parametric_errors(
    observed: List[np.ndarray],
    fitted: List[Optional[np.ndarray]],
    rng: np.random.Generator,
) -> List[Optional[np.ndarray]]:
    """Parametric bootstrap: add per-dataset Gaussian residual noise to fitted curves."""
    residual_pools = []
    for obs, fit in zip(observed, fitted):
        if fit is not None and len(fit) == len(obs):
            residual = np.asarray(obs, dtype=float) - np.asarray(fit, dtype=float)
            residual_pools.append(residual[np.isfinite(residual)])
    finite_residuals = np.concatenate([pool for pool in residual_pools if pool.size]) if any(
        pool.size for pool in residual_pools
    ) else np.array([], dtype=float)
    pooled_sigma = float(np.std(finite_residuals, ddof=1)) if finite_residuals.size > 1 else 0.0

    synthetic = []
    for obs, fit in zip(observed, fitted):
        if fit is None or len(fit) != len(obs):
            synthetic.append(None)
            continue
        obs = np.asarray(obs, dtype=float)
        fit = np.asarray(fit, dtype=float)
        residual = obs - fit
        valid = np.isfinite(obs) & np.isfinite(fit)
        local_residuals = residual[np.isfinite(residual)]
        sigma = float(np.std(local_residuals, ddof=1)) if local_residuals.size > 1 else pooled_sigma
        if not np.isfinite(sigma):
            sigma = pooled_sigma
        curve = fit.copy()
        curve[valid] += rng.normal(0.0, sigma, size=int(np.sum(valid)))
        synthetic.append(np.clip(curve, 0.0, 1.0))
    return synthetic


def _make_resampled_datasets(
    datasets: List[KineticDataset],
    fitted: List[Optional[np.ndarray]],
    rng: np.random.Generator,
    bootstrap_method: str,
) -> List[KineticDataset]:
    if bootstrap_method == 'monte_carlo':
        return _resample_cases(datasets, rng)
    observed = [dataset.conversion for dataset in datasets]
    if bootstrap_method == 'parametric':
        synthetic = _resample_parametric_errors(observed, fitted, rng)
    else:
        synthetic = _resample_residuals(observed, fitted, rng)
    return [
        KineticDataset(
            time=dataset.time, temperature=dataset.temperature, conversion=values,
            heating_rate=dataset.heating_rate,
            relative_humidity=dataset.relative_humidity,
            metadata=dict(dataset.metadata),
        )
        for dataset, values in zip(datasets, synthetic) if values is not None
    ]


def _perturb_start(params_logA: Dict[str, float], rng: np.random.Generator,
                   inv_RT_ref: float) -> Dict[str, float]:
    """Jitter the refit's starting point so replicates don't all start exactly at the best fit.

    Each Ea/logA pair is jittered in (Ea, ln k(T_ref)) coordinates -- the same
    coordinates the optimizer uses (simulation._ArrheniusReparam) -- so the start
    keeps the best fit's rate at the data's reference temperature. Jittering Ea
    and ln A independently moves ln k by several units, which on data that
    saturates quickly puts the start on a flat part of the objective where the
    optimizer never moves.
    """
    out = dict(params_logA)
    for name, value in params_logA.items():
        if name.startswith("Ea"):
            partner = "logA" + name[2:]
            dEa = rng.normal(0.0, 0.05 * abs(value) + 500.0)
            out[name] = float(value + dEa)
            if partner in params_logA:
                # ln k(T_ref) = lnA - Ea/(R T_ref): shift lnA with Ea, plus a small ln k jitter
                out[partner] = float(params_logA[partner] + dEa * inv_RT_ref + rng.normal(0.0, 0.05))
        elif not name.startswith("logA"):
            out[name] = float(value + rng.normal(0.0, 0.05 * abs(value) + 1e-3))
    return out


# --- Replicate workers -----------------------------------------------------
def _fit_on_resampled_data(
    datasets: List[KineticDataset],
    model_name: str,
    model_definition_args: Dict,
    best_fit_params_logA: Dict[str, float],
    parameter_bounds_logA: Optional[Dict[str, Tuple[float, float]]],
    solver_options: Optional[Dict],
    optimizer_options: Optional[Dict],
    end_callback_func: EndCallback,
    replicate_max_seconds: Optional[float] = None,
    bootstrap_method: str = 'residual',
    *,
    iteration_index: int,
    seed: Optional[np.random.SeedSequence] = None,
) -> Optional[Dict]:
    """One kinetic-model bootstrap replicate. Returns {'params_logA', 'stats'} or None."""
    _call_end_callback(end_callback_func, iteration_index, "started", None)
    rng = np.random.default_rng(seed)
    param_names_logA = list(best_fit_params_logA.keys())

    # Case bootstrap resamples observed rows directly; other methods need fitted curves.
    fitted: List[Optional[np.ndarray]] = [None] * len(datasets)
    if bootstrap_method != 'monte_carlo':
        try:
            alpha_of_kt = _closed_form_eligible(datasets, model_name, model_definition_args)
            ctx = _build_context(datasets, model_name, model_definition_args, solver_options,
                                 use_closed_form=alpha_of_kt is not None, alpha_of_kt_func=alpha_of_kt)
        except ValueError as e:
            warnings.warn(f"Bootstrap {iteration_index}: Model info error: {e}")
            return None
        for i, ds in enumerate(datasets):
            if len(ds.time) < 2:
                continue
            try:
                fitted[i] = _simulate_dataset(ctx, i, ds, best_fit_params_logA)
            except Exception as e_sim:
                warnings.warn(f"Bootstrap {iteration_index}: Res calc sim failed ds {i}: {e_sim}.")

    resampled_datasets = _make_resampled_datasets(datasets, fitted, rng, bootstrap_method)
    if not resampled_datasets:
        warnings.warn(f"Bootstrap {iteration_index}: No valid residuals.")
        return None

    # 3. Refit with the same optimizer core as fit_kinetic_model. The replicate
    # starts next to the best fit, so the coarse ln k scan is skipped.
    inv_T = np.concatenate([1.0 / np.asarray(ds.temperature, float) for ds in resampled_datasets])
    x0 = _perturb_start(best_fit_params_logA, rng, inv_RT_ref=float(np.mean(inv_T)) / R_GAS)
    bounds_list = None
    if parameter_bounds_logA:
        bounds_list = _bounds_list(parameter_bounds_logA, param_names_logA)
    deadline = time.time() + replicate_max_seconds if replicate_max_seconds else None
    optimizer_options = dict(optimizer_options or {})
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            outcome = _optimize(
                resampled_datasets, model_name, model_definition_args, param_names_logA,
                np.array([x0[p] for p in param_names_logA]), bounds_list,
                solver_options, optimizer_options, deadline,
                coarse_start=False, allow_rate_fallback=False, powell_maxfev=2000,
            )
    except Exception as e:
        warnings.warn(f"Bootstrap replicate {iteration_index} fit exception: {e}")
        return None
    if outcome.result is None:
        return None

    fitted_params_logA = {p: float(v) for p, v in zip(param_names_logA, outcome.result.x)}
    stats = calculate_stats_for_replicate(resampled_datasets, fitted_params_logA, model_name,
                                          model_definition_args, solver_options)
    output = {'params_logA': fitted_params_logA, 'stats': stats}
    _call_end_callback(end_callback_func, iteration_index, "completed", output)
    return output


def _fit_empirical_on_resampled_data(
    datasets: List[KineticDataset],
    model_type: str,
    best_fit_params: Dict[str, float],
    parameter_bounds: Optional[Dict[str, Tuple[float, float]]],
    end_callback_func: EndCallback,
    bootstrap_method: str = 'residual',
    *,
    iteration_index: int,
    seed: Optional[np.random.SeedSequence] = None,
) -> Optional[Dict]:
    """One empirical-model bootstrap replicate. Returns {'params', 'stats'} or None."""
    from .empirical import fit_empirical_global, predict_empirical

    _call_end_callback(end_callback_func, iteration_index, "started", None)
    rng = np.random.default_rng(seed)

    best_fit = FitResult(
        model_name=f'Empirical_{model_type}', parameters=best_fit_params, success=True,
        message="Bootstrap prediction", rss=0.0, n_datapoints=0,
        n_parameters=len(best_fit_params), model_definition_args={'empirical_type': model_type},
    )
    fitted: List[Optional[np.ndarray]] = [None] * len(datasets)
    if bootstrap_method != 'monte_carlo':
        for i, ds in enumerate(datasets):
            if len(ds.time) < 2:
                continue
            try:
                fitted[i] = np.asarray(
                    predict_empirical(best_fit, ds.time, float(np.mean(ds.temperature))), dtype=float
                )
            except Exception as e_pred:
                warnings.warn(f"Bootstrap {iteration_index}: Prediction failed for dataset {i}: {e_pred}")

    resampled_datasets = _make_resampled_datasets(datasets, fitted, rng, bootstrap_method)
    if not resampled_datasets:
        warnings.warn(f"Bootstrap {iteration_index}: No valid residuals.")
        return None

    try:
        replicate_fit = fit_empirical_global(
            datasets=resampled_datasets, model_type=model_type, initial_guess=best_fit_params,
            parameter_bounds=parameter_bounds, use_global_optimizer=False, verbose=False,
        )
    except Exception as e_fit:
        warnings.warn(f"Bootstrap {iteration_index}: Refit failed: {e_fit}")
        return None
    if not replicate_fit.success:
        return None

    output = {
        'params': {k: float(v) for k, v in replicate_fit.parameters.items()},
        'stats': {
            'rss': float(replicate_fit.rss),
            'r_squared': float(replicate_fit.r_squared),
            'aic': float(replicate_fit.aic),
            'bic': float(replicate_fit.bic),
        },
    }
    _call_end_callback(end_callback_func, iteration_index, "completed", output)
    return output


# --- Replicate statistics ---------------------------------------------------
def calculate_stats_for_replicate(
    datasets: List[KineticDataset],
    params_logA: Dict[str, float],
    model_name: str,
    model_definition_args: Dict,
    solver_options: Optional[Dict]
) -> Dict[str, float]:
    """Unweighted RSS, R², AIC and BIC of a replicate's fit to its own synthetic data."""
    rss, n_pts, r_squared, aic, bic = _calculate_conversion_stats(
        datasets, params_logA, model_name, model_definition_args, solver_options, use_closed_form=True
    )
    return {'rss': rss, 'r_squared': r_squared, 'aic': aic, 'bic': bic, 'n_points': n_pts}


def rank_replicates(
    successful_params_logA: List[Dict[str, float]],
    successful_replicate_stats: List[Dict[str, float]]
) -> List[Dict]:
    """Ranks replicates by RSS (ascending) and returns a sorted list."""
    ranked = []
    for i, (params_logA, stats) in enumerate(zip(successful_params_logA, successful_replicate_stats)):
        ranked.append({
            'rank': 0,  # assigned after sorting
            'source': f'replicate_{i+1}',
            'parameters': params_logA_to_A(params_logA),
            'stats': stats,
            'sort_metric': stats.get('rss', np.inf),
        })
    ranked.sort(key=lambda x: x['sort_metric'])
    for rank, item in enumerate(ranked):
        item['rank'] = rank + 1
    return ranked


# A replicate is refit to its own resampled data, so a converged refit has RSS
# comparable to its peers. Refits whose RSS is this many times the median
# diverged (optimizer failure, not sampling variation) and are excluded.
FAILED_REFIT_RSS_FACTOR = 10.0


def _converged_replicate_mask(stats_list: List[Dict]) -> np.ndarray:
    rss = np.array([s.get('rss', np.nan) for s in stats_list], dtype=float)
    finite = np.isfinite(rss)
    if finite.sum() < 3:
        return finite
    return finite & (rss <= FAILED_REFIT_RSS_FACTOR * np.median(rss[finite]))


def _percentile_ci(values: np.ndarray, confidence_level: float) -> Tuple[float, float]:
    """
    Calculate two-sided confidence interval (for parameter reporting).

    Note: For ICH Q1E-compliant shelf-life determination, use
    _percentile_ci_one_sided() instead.
    """
    tail = (1.0 - confidence_level) / 2.0
    lo, hi = np.percentile(values, [tail * 100.0, (1.0 - tail) * 100.0])
    return float(lo), float(hi)


def _percentile_ci_one_sided(
    values: np.ndarray,
    confidence_level: float,
    side: str = 'lower'
) -> float:
    """
    Calculate one-sided confidence limit per ICH Q1E requirements.

    ICH Q1E mandates one-sided confidence bounds for shelf-life determination:
    - For decreasing attributes (potency, monomer): use lower bound
    - For increasing attributes (aggregates, impurities): use upper bound

    Parameters
    ----------
    values : np.ndarray
        Bootstrap distribution
    confidence_level : float
        Confidence level (typically 0.95)
    side : str
        'lower' for decreasing attributes (potency, monomer)
        'upper' for increasing attributes (aggregates, impurities)

    Returns
    -------
    float
        One-sided confidence limit

    Notes
    -----
    For 95% confidence:
    - Lower one-sided bound: 5th percentile (NOT 2.5th)
    - Upper one-sided bound: 95th percentile (NOT 97.5th)

    This is NOT a two-sided interval [2.5%, 97.5%].

    References
    ----------
    ICH Q1E: "Evaluation of Stability Data"
    - Section on confidence intervals for shelf-life determination
    - Requires one-sided 95% confidence limits

    Examples
    --------
    >>> values = np.random.normal(100, 10, size=1000)
    >>> lower_bound = _percentile_ci_one_sided(values, 0.95, side='lower')
    >>> # For 95% CI, lower_bound ≈ 5th percentile
    """
    if side == 'lower':
        # Lower 95% one-sided limit: 5th percentile
        percentile = (1.0 - confidence_level) * 100.0
    elif side == 'upper':
        # Upper 95% one-sided limit: 95th percentile
        percentile = confidence_level * 100.0
    else:
        raise ValueError(f"side must be 'lower' or 'upper', got '{side}'")

    return float(np.percentile(values, percentile))


# --- Public entry points ----------------------------------------------------
def run_bootstrap(
    datasets: List[KineticDataset],
    fit_result: FitResult,
    optimizer_options: Optional[Dict] = None,
    parameter_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    n_iterations: int = 100,
    confidence_level: float = 0.95,
    n_jobs: int = -1,
    solver_options: Optional[Dict] = None,
    end_callback: EndCallback = None,
    timeout_per_replicate: Optional[float] = None,
    return_replicate_params: bool = False,
    random_state: RandomState = None,
    bootstrap_method: str = 'monte_carlo',
) -> Optional[BootstrapResult]:
    """Bootstrap fitted parameters using case, parametric, or residual resampling.

    Replicates run in a process pool. ``n_jobs=-1`` uses all but one available
    CPU core, capped at ``n_iterations``. Pass ``random_state`` (an int seed,
    SeedSequence or Generator) for reproducible results.

    Replicate refits use the same optimizer as fit_kinetic_model: closed-form
    least_squares for isothermal data with an analytic model, Powell on the ODE
    objective otherwise. ``timeout_per_replicate`` also bounds each refit's
    wall-clock optimizer budget.
    """
    bootstrap_method = _normalize_bootstrap_method(bootstrap_method)
    if not fit_result.success:
        warnings.warn("Initial fit failed.")
        return None
    model_definition_args = getattr(fit_result, 'model_definition_args', None)
    if model_definition_args is None:
        warnings.warn("FitResult missing 'model_definition_args'. Cannot run bootstrap reliably.")
        return None
    model_name = fit_result.model_name

    if model_name.startswith('Empirical_'):
        return run_bootstrap_empirical(
            datasets=datasets, fit_result=fit_result, parameter_bounds=parameter_bounds,
            n_iterations=n_iterations, confidence_level=confidence_level, n_jobs=n_jobs,
            end_callback=end_callback, timeout_per_replicate=timeout_per_replicate,
            return_replicate_params=return_replicate_params, random_state=random_state,
            bootstrap_method=bootstrap_method,
        )

    try:
        best_fit_params_logA = params_A_to_logA(fit_result.parameters, model_name)
    except (ValueError, TypeError) as e_conv:
        warnings.warn(f"Cannot take log of parameter for bootstrap: {e_conv}.")
        return None
    param_names_logA = list(best_fit_params_logA.keys())
    name_map = {p: (p[3:] if p.startswith("logA") else p) for p in param_names_logA}
    # Same bounds fit_kinetic_model applies: its physical defaults, overlaid with
    # the user's. Without them a replicate refit on data that can't pin Ea down
    # (e.g. a single temperature) can wander to unphysical values like negative Ea.
    parameter_bounds_logA = _default_bounds_logA(param_names_logA)
    parameter_bounds_logA.update(_bounds_A_to_logA(parameter_bounds, param_names_logA, name_map))

    worker_args = (datasets, model_name, model_definition_args, best_fit_params_logA,
                   parameter_bounds_logA, solver_options, optimizer_options, end_callback,
                   timeout_per_replicate, bootstrap_method)
    results = _run_replicates(_fit_on_resampled_data, worker_args, n_iterations, n_jobs,
                              timeout_per_replicate, random_state, label=f"{model_name} (logA scale)")

    print("\nProcessing received results...")
    successful_params_logA, successful_stats = [], []
    for i, data in enumerate(results):
        if not isinstance(data, dict) or 'params_logA' not in data or 'stats' not in data:
            continue
        if all(k in data['params_logA'] for k in param_names_logA):
            successful_params_logA.append(data['params_logA'])
            successful_stats.append(data['stats'])
        else:
            warnings.warn(f"Replicate {i+1} returned unexpected dict keys.")
    converged = _converged_replicate_mask(successful_stats)
    n_diverged = int(len(converged) - converged.sum())
    if n_diverged:
        warnings.warn(f"Excluded {n_diverged} bootstrap refits that did not converge "
                      f"(RSS > {FAILED_REFIT_RSS_FACTOR:g}x the median replicate RSS).")
        successful_params_logA = [p for p, ok in zip(successful_params_logA, converged) if ok]
        successful_stats = [s for s, ok in zip(successful_stats, converged) if ok]
    n_success = len(successful_params_logA)
    print(f"\nBootstrap finished processing. {n_success}/{n_iterations} replicates successful "
          f"({n_iterations - n_success} failed/timed out).")
    if n_success == 0:
        return None
    if n_success < n_iterations * 0.75:
        warnings.warn(f"Low success rate ({n_success}/{n_iterations}). Results may be less reliable.")

    # Distributions / medians / CIs are computed on the (log) fit scale and mapped back.
    dist_logA = {p: np.array([params[p] for params in successful_params_logA]) for p in param_names_logA}
    median_params_logA = {p: float(np.median(dist_logA[p])) for p in param_names_logA}
    median_stats = None
    try:
        med_rss, _, med_r2, med_aic, med_bic = _calculate_conversion_stats(
            datasets, median_params_logA, model_name, model_definition_args, solver_options, use_closed_form=True)
        if np.isfinite(med_bic):
            median_stats = {'rss': med_rss, 'r_squared': med_r2, 'aic': med_aic, 'bic': med_bic}
    except Exception as e_med_sim:
        warnings.warn(f"Failed to calculate stats for median parameters: {e_med_sim}")

    param_distributions, param_ci = {}, {}
    for p_logA, dist in dist_logA.items():
        is_log = p_logA.startswith("logA")
        key = name_map[p_logA]
        param_distributions[key] = np.exp(dist) if is_log else dist
        if len(dist) > 3 and np.std(dist) > 1e-9 * abs(np.mean(dist)) + 1e-12:
            lo, hi = _percentile_ci(dist, confidence_level)
        elif len(dist) > 0:
            lo = hi = float(np.mean(dist))  # too few points / zero variance
        else:
            lo = hi = float('nan')
        param_ci[key] = ((float(np.exp(lo)), float(np.exp(hi))) if is_log and np.isfinite(lo) and np.isfinite(hi)
                         else (float(lo), float(hi)))

    return BootstrapResult(
        model_name=model_name,
        parameter_distributions=param_distributions,
        parameter_ci=param_ci,
        n_iterations=n_success,
        confidence_level=confidence_level,
        bootstrap_method=bootstrap_method,
        raw_parameter_list=[params_logA_to_A(p) for p in successful_params_logA] if return_replicate_params else None,
        ranked_replicates=rank_replicates(successful_params_logA, successful_stats),
        median_parameters=params_logA_to_A(median_params_logA),
        median_stats=median_stats,
    )


def run_bootstrap_empirical(
    datasets: List[KineticDataset],
    fit_result: FitResult,
    parameter_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    n_iterations: int = 100,
    confidence_level: float = 0.95,
    n_jobs: int = -1,
    end_callback: EndCallback = None,
    timeout_per_replicate: Optional[float] = None,
    return_replicate_params: bool = False,
    random_state: RandomState = None,
    bootstrap_method: str = 'monte_carlo',
) -> Optional[BootstrapResult]:
    """
    Bootstrap an empirical model using case, parametric, or residual resampling.

    Returns None if fewer than 10 replicates succeed.
    """
    bootstrap_method = _normalize_bootstrap_method(bootstrap_method)
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

    best_fit_params = {k: float(v) for k, v in fit_result.parameters.items()}
    worker_args = (datasets, model_type, best_fit_params, parameter_bounds, end_callback, bootstrap_method)
    results = _run_replicates(_fit_empirical_on_resampled_data, worker_args, n_iterations, n_jobs,
                              timeout_per_replicate, random_state, label=f"Empirical_{model_type}")

    print("\nProcessing received results...")
    successful = [r for r in results if isinstance(r, dict) and 'params' in r and 'stats' in r]
    converged = _converged_replicate_mask([r['stats'] for r in successful])
    n_diverged = int(len(converged) - converged.sum())
    if n_diverged:
        warnings.warn(f"Excluded {n_diverged} bootstrap refits that did not converge "
                      f"(RSS > {FAILED_REFIT_RSS_FACTOR:g}x the median replicate RSS).")
        successful = [r for r, ok in zip(successful, converged) if ok]
    n_successful = len(successful)
    print(f"\nBootstrap finished processing. {n_successful}/{n_iterations} replicates successful "
          f"({n_iterations - n_successful} failed/timed out).")
    if n_successful < 10:
        warnings.warn(f"Too few successful bootstrap replicates ({n_successful}). Cannot compute reliable CI.")
        return None

    param_names = list(best_fit_params.keys())
    param_distributions = {name: np.array([r['params'][name] for r in successful if name in r['params']])
                           for name in param_names}
    param_ci, median_params = {}, {}
    for name, values in param_distributions.items():
        if len(values) > 0:
            param_ci[name] = _percentile_ci(values, confidence_level)
            median_params[name] = float(np.median(values))
        else:
            param_ci[name] = (float('nan'), float('nan'))
            median_params[name] = best_fit_params[name]

    ranked_replicates = sorted(
        ({'replicate_index': i, 'parameters': r['params'],
          'r_squared': r['stats'].get('r_squared', np.nan), 'rss': r['stats'].get('rss', np.inf),
          'aic': r['stats'].get('aic', np.inf), 'bic': r['stats'].get('bic', np.inf)}
         for i, r in enumerate(successful)),
        key=lambda x: x['r_squared'], reverse=True)

    median_stats = {'r_squared': float('nan'), 'rss': float('inf')}
    try:
        from .empirical import fit_empirical_global
        median_fit = fit_empirical_global(datasets=datasets, model_type=model_type, initial_guess=median_params,
                                          parameter_bounds=parameter_bounds, verbose=False)
        if median_fit.success:
            median_stats = {'r_squared': float(median_fit.r_squared), 'rss': float(median_fit.rss),
                            'aic': float(median_fit.aic), 'bic': float(median_fit.bic)}
    except Exception:
        pass

    return BootstrapResult(
        model_name=fit_result.model_name,
        parameter_distributions=param_distributions,
        parameter_ci=param_ci,
        n_iterations=n_successful,
        confidence_level=confidence_level,
        bootstrap_method=bootstrap_method,
        raw_parameter_list=[r['params'] for r in successful] if return_replicate_params else None,
        ranked_replicates=ranked_replicates,
        median_parameters=median_params,
        median_stats=median_stats,
    )
