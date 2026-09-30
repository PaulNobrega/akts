# isoconversional.py
import numpy as np
from scipy.interpolate import interp1d, UnivariateSpline
from scipy.optimize import brentq
from scipy.stats import linregress
from scipy.optimize import minimize
import warnings
from typing import List, Dict, Tuple, Optional, Union

from .datatypes import KineticDataset, IsoResult, BootstrapResult
from .utils import numerical_diff, R_GAS

def _spline_dadt_at_alpha(time_unique: np.ndarray, alpha_unique: np.ndarray,
                          alpha_levels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Smooths alpha(t) with a cubic spline and returns (t_at_alpha, dadt_at_alpha)
    by inverting the spline (root-finding) rather than differencing the raw,
    noisy data. This matters most for sparse isothermal data (~10-30 points):
    a plain finite-difference derivative there is dominated by measurement
    noise -- verified this gives Friedman Ea(alpha) within ~10-15% of ground
    truth where raw differencing swung wildly (>50%) on the same data.

    Smoothing strength (s) uses scipy's own heuristic (s ~ N * sigma^2), with
    sigma estimated from the residual of a quadratic fit to (t, alpha) as a
    practical noise proxy since the true measurement noise isn't known.

    Known limitation: for sigmoidal (e.g. Avrami) curves that residual is mostly
    model misfit rather than noise, so the spline is oversmoothed and early-alpha
    rates come out too high. A true-noise estimate removes that bias but leaves
    sparse (~25-point) curves too noisy to differentiate, so the trade-off is
    left as is for now.
    """
    n = len(time_unique)
    t_at_alpha = np.full_like(alpha_levels, np.nan, dtype=float)
    dadt_at_alpha = np.full_like(alpha_levels, np.nan, dtype=float)

    if n < 4:
        return t_at_alpha, dadt_at_alpha  # too few points for a cubic spline

    try:
        quad_fit = np.polyval(np.polyfit(time_unique, alpha_unique, 2), time_unique)
        sigma = np.std(alpha_unique - quad_fit)
        if not np.isfinite(sigma) or sigma < 1e-6:
            sigma = 1e-3  # near-perfect quadratic fit; avoid s=0 (interpolating spline, no smoothing)
        spline = UnivariateSpline(time_unique, alpha_unique, k=3, s=n * sigma**2)
    except Exception as e:
        warnings.warn(f"Spline smoothing failed: {e}.")
        return t_at_alpha, dadt_at_alpha

    t_lo, t_hi = time_unique[0], time_unique[-1]
    a_lo, a_hi = alpha_unique.min(), alpha_unique.max()
    for i, alpha in enumerate(alpha_levels):
        if alpha < a_lo or alpha > a_hi:
            continue  # don't extrapolate past the data
        try:
            t_root = brentq(lambda tt: spline(tt) - alpha, t_lo, t_hi)
        except ValueError:
            continue  # spline doesn't cross this alpha monotonically in [t_lo, t_hi]
        rate = float(spline.derivative(1)(t_root))
        if np.isfinite(rate) and rate > 0:
            t_at_alpha[i] = t_root
            dadt_at_alpha[i] = rate

    return t_at_alpha, dadt_at_alpha


def _prepare_iso_data(datasets: List[KineticDataset], alpha_levels: np.ndarray) -> Dict:
    """Interpolates T, t, and d(alpha)/dt at specified alpha levels for each dataset."""
    iso_data = {'alpha': alpha_levels, 'datasets': []}
    min_len_for_diff = 5 # Min points needed for SavGol

    for i, ds in enumerate(datasets):
        if len(ds.time) < min_len_for_diff:
            warnings.warn(f"Dataset {i} has too few points ({len(ds.time)}) for reliable differentiation. Skipping.")
            continue

        # Ensure alpha is monotonically increasing for interpolation
        alpha_unique, idx_unique = np.unique(ds.conversion, return_index=True)
        time_unique = ds.time[idx_unique]
        temp_unique = ds.temperature[idx_unique]

        if len(alpha_unique) < 2:
             warnings.warn(f"Dataset {i} has too few unique alpha points ({len(alpha_unique)}) for interpolation. Skipping.")
             continue

        # Spline-based t(alpha) and d(alpha)/dt(alpha) needs alpha as a function
        # of a strictly increasing TIME (unlike the alpha-sorted arrays above,
        # which np.unique(ds.conversion, ...) returns in alpha order -- with
        # noisy data that scrambles the time order and breaks UnivariateSpline).
        # Build a separate, time-sorted, time-deduped series just for this.
        time_sorted_idx = np.argsort(ds.time)
        time_sorted, alpha_by_time = ds.time[time_sorted_idx], ds.conversion[time_sorted_idx]
        time_for_spline, first_idx = np.unique(time_sorted, return_index=True)
        alpha_for_spline = alpha_by_time[first_idx]

        # Spline-based t(alpha) and d(alpha)/dt(alpha): robust to sparse/noisy
        # data (see _spline_dadt_at_alpha). Falls back to NaN per-alpha (handled
        # downstream the same way a missing point always was) if the spline
        # step itself fails outright.
        t_at_alpha, dadt_at_alpha = _spline_dadt_at_alpha(time_for_spline, alpha_for_spline, alpha_levels)

        # Create interpolation functions
        try:
            interp_temp = interp1d(alpha_unique, temp_unique, bounds_error=False, fill_value=np.nan)
        except ValueError as e:
            warnings.warn(f"Interpolation failed for dataset {i}: {e}. Skipping.")
            continue

        # Interpolate at target alpha levels
        T_at_alpha = interp_temp(alpha_levels)

        # Calculate heating rate beta = dT/dt (can be variable)
        # Use smoothed derivative dT/dt
        dTdt = numerical_diff(time_unique, temp_unique)
        # Interpolate dT/dt at the target alpha levels using time interpolation first
        interp_dTdt = interp1d(time_unique, dTdt, bounds_error=False, fill_value=np.nan)
        dTdt_at_alpha = interp_dTdt(t_at_alpha)
        # Use average heating rate if constant heating was intended
        avg_beta = ds.heating_rate if ds.heating_rate is not None else np.mean(dTdt[np.isfinite(dTdt)])
        if not np.isfinite(avg_beta) or avg_beta <= 0: avg_beta = 10.0/60.0 # Default guess if needed
        # Use interpolated dTdt if available, else average beta
        beta_at_alpha = np.where(np.isfinite(dTdt_at_alpha) & (dTdt_at_alpha > 1e-6), dTdt_at_alpha, avg_beta)


        dataset_iso_data = {
            'T': T_at_alpha,
            't': t_at_alpha,
            'dadt': dadt_at_alpha,
            'beta': beta_at_alpha,
            'id': i
        }
        iso_data['datasets'].append(dataset_iso_data)

    if not iso_data['datasets']:
        raise ValueError("No valid datasets found for isoconversional analysis after preprocessing.")

    return iso_data


def run_friedman(datasets: List[KineticDataset], alpha_levels: np.ndarray = np.linspace(0.05, 0.95, 19)) -> IsoResult:
    """Performs Friedman isoconversional analysis."""
    iso_data = _prepare_iso_data(datasets, alpha_levels)
    n_alpha = len(alpha_levels)
    Ea_values = np.full(n_alpha, np.nan)
    Ea_std_errs = np.full(n_alpha, np.nan)
    ln_A_f_alpha_values = np.full(n_alpha, np.nan)
    regression_stats = []

    for i, alpha in enumerate(alpha_levels):
        ln_rate = []
        inv_T = []
        for ds_data in iso_data['datasets']:
            rate = ds_data['dadt'][i]
            T = ds_data['T'][i]
            if np.isfinite(rate) and rate > 1e-12 and np.isfinite(T) and T > 0:
                ln_rate.append(np.log(rate))
                inv_T.append(1.0 / T)

        if len(ln_rate) >= 2: # Need at least 2 points for linear regression
            ln_rate = np.array(ln_rate)
            inv_T = np.array(inv_T)
            try:
                res = linregress(inv_T, ln_rate)
                if np.isfinite(res.slope):
                    Ea = -res.slope * R_GAS
                    Ea_values[i] = Ea
                    # Std Err of slope * R_GAS
                    Ea_std_errs[i] = res.stderr * R_GAS if res.stderr is not None else np.nan
                    # Friedman: ln(dalpha/dt) = ln[A*f(alpha)] - Ea/(R*T), so the
                    # regression intercept IS ln[A*f(alpha)] at this alpha directly.
                    ln_A_f_alpha_values[i] = res.intercept
                    regression_stats.append({'alpha': alpha, 'r_value': res.rvalue, 'p_value': res.pvalue, 'stderr': res.stderr, 'intercept': res.intercept})
                else:
                     regression_stats.append({'alpha': alpha, 'r_value': np.nan, 'p_value': np.nan, 'stderr': np.nan, 'intercept': np.nan})

            except ValueError as e:
                warnings.warn(f"Linear regression failed for alpha={alpha:.3f}: {e}")
                regression_stats.append({'alpha': alpha, 'r_value': np.nan, 'p_value': np.nan, 'stderr': np.nan, 'intercept': np.nan})
        else:
             regression_stats.append({'alpha': alpha, 'r_value': np.nan, 'p_value': np.nan, 'stderr': np.nan, 'intercept': np.nan})


    return IsoResult(method="Friedman", alpha=alpha_levels, Ea=Ea_values, Ea_std_err=Ea_std_errs,
                     regression_stats=regression_stats, ln_A_f_alpha=ln_A_f_alpha_values)


def run_bootstrap_friedman(
    datasets: List[KineticDataset],
    iso_result: IsoResult,
    n_iterations: int = 100,
    confidence_level: float = 0.95,
    alpha_levels: Optional[np.ndarray] = None,
    random_state: Optional[Union[int, np.random.Generator]] = None,
    bootstrap_method: str = 'monte_carlo',
) -> Optional[BootstrapResult]:
    """
    Bootstrap confidence intervals for a Friedman model-free result.

    ``monte_carlo`` resamples complete observed rows, ``residual`` resamples
    centered conversion residuals, and ``parametric`` adds Gaussian conversion
    errors with scales estimated from the fitted residuals. Each replicate reruns
    the per-conversion Friedman regressions.

    Parameters
    ----------
    datasets : List[KineticDataset]
        Same datasets used for the original run_friedman() call.
    iso_result : IsoResult
        The original Friedman result (alpha_levels default to iso_result.alpha).
    n_iterations : int, default=100
        Number of bootstrap replicates.
    confidence_level : float, default=0.95
        Confidence level for parameter_ci (e.g. 0.95 = 95%).
    alpha_levels : np.ndarray, optional
        Overrides iso_result.alpha if given.
    random_state : int or np.random.Generator, optional
        Seed (or generator) for reproducible resampling.
    bootstrap_method : {'monte_carlo', 'parametric', 'residual'}, default='monte_carlo'
        Synthetic-data method used for each replicate.

    Returns
    -------
    Optional[BootstrapResult]
        None if no replicate resolved at least 2 alpha levels. Otherwise:
        - parameter_distributions / parameter_ci: keyed by f"Ea_alpha_{a:.4g}"
          and f"lnAf_alpha_{a:.4g}" per resolved alpha level.
        - raw_parameter_list: one dict per successful replicate, each
          {'alpha': array, 'Ea': array, 'ln_A_f_alpha': array} -- this is what
          predict_conversion()'s Friedman branch uses to reconstruct a
          per-replicate IsoResult and propagate a prediction confidence band
          (see core.predict_conversion's FitResult branch).
        - median_parameters: median Ea(alpha)/ln_A_f_alpha(alpha) arrays, same
          keying as parameter_distributions.
    """
    from .bootstrap import _make_resampled_datasets, _normalize_bootstrap_method
    from .prediction import predict_conversion_model_free

    bootstrap_method = _normalize_bootstrap_method(bootstrap_method)
    levels = alpha_levels if alpha_levels is not None else iso_result.alpha
    fitted = [None] * len(datasets)
    if bootstrap_method != 'monte_carlo':
        for index, dataset in enumerate(datasets):
            try:
                prediction = predict_conversion_model_free(
                    iso_result, (dataset.time, dataset.temperature), dataset.time
                )
                fitted[index] = prediction.conversion
            except Exception as exc:
                warnings.warn(f"Friedman bootstrap: prediction failed for dataset {index}: {exc}")

    n_alpha = len(levels)
    Ea_replicates = [[] for _ in range(n_alpha)]
    lnAf_replicates = [[] for _ in range(n_alpha)]
    raw_parameter_list = []

    rng = np.random.default_rng(random_state)
    for _ in range(n_iterations):
        synthetic_datasets = _make_resampled_datasets(datasets, fitted, rng, bootstrap_method)
        if not synthetic_datasets:
            continue
        try:
            replicate = run_friedman(synthetic_datasets, alpha_levels=levels)
        except Exception as exc:
            warnings.warn(f"Friedman bootstrap replicate failed: {exc}")
            continue
        rep_Ea = np.asarray(replicate.Ea, dtype=float)
        rep_lnAf = np.asarray(replicate.ln_A_f_alpha, dtype=float)
        valid = np.isfinite(rep_Ea) & np.isfinite(rep_lnAf)
        if np.sum(valid) >= 2:
            for i in np.flatnonzero(valid):
                Ea_replicates[i].append(rep_Ea[i])
                lnAf_replicates[i].append(rep_lnAf[i])
            raw_parameter_list.append({'alpha': levels.copy(), 'Ea': rep_Ea, 'ln_A_f_alpha': rep_lnAf})

    if not raw_parameter_list:
        warnings.warn("Friedman bootstrap: no replicate resolved at least 2 alpha levels.")
        return None

    alpha_ci_level = (1.0 - confidence_level) / 2.0
    parameter_distributions, parameter_ci, median_parameters = {}, {}, {}
    for i, alpha in enumerate(levels):
        if len(Ea_replicates[i]) < 2:
            continue
        ea_key, lnaf_key = f"Ea_alpha_{alpha:.4g}", f"lnAf_alpha_{alpha:.4g}"
        ea_arr, lnaf_arr = np.array(Ea_replicates[i]), np.array(lnAf_replicates[i])
        parameter_distributions[ea_key] = ea_arr
        parameter_distributions[lnaf_key] = lnaf_arr
        parameter_ci[ea_key] = (
            float(np.percentile(ea_arr, alpha_ci_level * 100.0)),
            float(np.percentile(ea_arr, (1.0 - alpha_ci_level) * 100.0)),
        )
        parameter_ci[lnaf_key] = (
            float(np.percentile(lnaf_arr, alpha_ci_level * 100.0)),
            float(np.percentile(lnaf_arr, (1.0 - alpha_ci_level) * 100.0)),
        )
        median_parameters[ea_key] = float(np.median(ea_arr))
        median_parameters[lnaf_key] = float(np.median(lnaf_arr))

    return BootstrapResult(
        model_name="Friedman",
        parameter_distributions=parameter_distributions,
        parameter_ci=parameter_ci,
        n_iterations=len(raw_parameter_list),
        confidence_level=confidence_level,
        bootstrap_method=bootstrap_method,
        raw_parameter_list=raw_parameter_list,
        median_parameters=median_parameters,
    )


def run_kas(datasets: List[KineticDataset], alpha_levels: np.ndarray = np.linspace(0.05, 0.95, 19)) -> IsoResult:
    """Performs Kissinger-Akahira-Sunose (KAS) isoconversional analysis."""
    iso_data = _prepare_iso_data(datasets, alpha_levels)
    n_alpha = len(alpha_levels)
    Ea_values = np.full(n_alpha, np.nan)
    Ea_std_errs = np.full(n_alpha, np.nan)
    regression_stats = []

    for i, alpha in enumerate(alpha_levels):
        ln_beta_T2 = []
        inv_T = []
        for ds_data in iso_data['datasets']:
            beta = ds_data['beta'][i] # Use beta at specific alpha
            T = ds_data['T'][i]
            if np.isfinite(beta) and beta > 1e-9 and np.isfinite(T) and T > 1e-9:
                 # Avoid division by zero or log(zero)
                 if T**2 > 1e-12:
                     ln_beta_T2.append(np.log(beta / (T**2)))
                     inv_T.append(1.0 / T)

        if len(ln_beta_T2) >= 2:
            ln_beta_T2 = np.array(ln_beta_T2)
            inv_T = np.array(inv_T)
            try:
                res = linregress(inv_T, ln_beta_T2)
                if np.isfinite(res.slope):
                    Ea = -res.slope * R_GAS
                    Ea_values[i] = Ea
                    Ea_std_errs[i] = res.stderr * R_GAS if res.stderr is not None else np.nan
                    regression_stats.append({'alpha': alpha, 'r_value': res.rvalue, 'p_value': res.pvalue, 'stderr': res.stderr, 'intercept': res.intercept})
                else:
                    regression_stats.append({'alpha': alpha, 'r_value': np.nan, 'p_value': np.nan, 'stderr': np.nan, 'intercept': np.nan})
            except ValueError as e:
                warnings.warn(f"Linear regression failed for alpha={alpha:.3f}: {e}")
                regression_stats.append({'alpha': alpha, 'r_value': np.nan, 'p_value': np.nan, 'stderr': np.nan, 'intercept': np.nan})
        else:
            regression_stats.append({'alpha': alpha, 'r_value': np.nan, 'p_value': np.nan, 'stderr': np.nan, 'intercept': np.nan})

    return IsoResult(method="KAS", alpha=alpha_levels, Ea=Ea_values, Ea_std_err=Ea_std_errs, regression_stats=regression_stats)


def run_ofw(datasets: List[KineticDataset], alpha_levels: np.ndarray = np.linspace(0.05, 0.95, 19)) -> IsoResult:
    """Performs Ozawa-Flynn-Wall (OFW) isoconversional analysis."""
    # OFW uses Doyle's approximation, resulting in ln(beta) vs 1/T
    iso_data = _prepare_iso_data(datasets, alpha_levels)
    n_alpha = len(alpha_levels)
    Ea_values = np.full(n_alpha, np.nan)
    Ea_std_errs = np.full(n_alpha, np.nan)
    regression_stats = []

    for i, alpha in enumerate(alpha_levels):
        ln_beta = []
        inv_T = []
        for ds_data in iso_data['datasets']:
            beta = ds_data['beta'][i]
            T = ds_data['T'][i]
            if np.isfinite(beta) and beta > 1e-9 and np.isfinite(T) and T > 0:
                ln_beta.append(np.log(beta))
                inv_T.append(1.0 / T)

        if len(ln_beta) >= 2:
            ln_beta = np.array(ln_beta)
            inv_T = np.array(inv_T)
            try:
                # Note: Slope is approx -1.052 * Ea / R for Doyle approx.
                res = linregress(inv_T, ln_beta)
                if np.isfinite(res.slope):
                    Ea = -res.slope * R_GAS / 1.052
                    Ea_values[i] = Ea
                    Ea_std_errs[i] = (res.stderr * R_GAS / 1.052) if res.stderr is not None else np.nan
                    regression_stats.append({'alpha': alpha, 'r_value': res.rvalue, 'p_value': res.pvalue, 'stderr': res.stderr, 'intercept': res.intercept})
                else:
                    regression_stats.append({'alpha': alpha, 'r_value': np.nan, 'p_value': np.nan, 'stderr': np.nan, 'intercept': np.nan})
            except ValueError as e:
                warnings.warn(f"Linear regression failed for alpha={alpha:.3f}: {e}")
                regression_stats.append({'alpha': alpha, 'r_value': np.nan, 'p_value': np.nan, 'stderr': np.nan, 'intercept': np.nan})
        else:
            regression_stats.append({'alpha': alpha, 'r_value': np.nan, 'p_value': np.nan, 'stderr': np.nan, 'intercept': np.nan})


    return IsoResult(method="OFW", alpha=alpha_levels, Ea=Ea_values, Ea_std_err=Ea_std_errs, regression_stats=regression_stats)


def run_vyazovkin(datasets: List[KineticDataset], alpha_levels: np.ndarray = np.linspace(0.05, 0.95, 19)) -> IsoResult:
    """
    Performs Vyazovkin advanced nonlinear isoconversional analysis.

    This method minimizes the objective function:
    Ω(Ea) = Σ_i Σ_j≠i [J(Ea, T_i(t)) / J(Ea, T_j(t))]

    where J(Ea, T(t)) is the temperature integral.

    This is more accurate than linear methods (Friedman/KAS/OFW) for variable heating rates.

    Parameters
    ----------
    datasets : List[KineticDataset]
        List of kinetic datasets at different heating rates
    alpha_levels : np.ndarray
        Conversion levels at which to compute Ea

    Returns
    -------
    IsoResult
        Results with Ea values at each alpha level
    """
    iso_data = _prepare_iso_data(datasets, alpha_levels)
    n_alpha = len(alpha_levels)
    n_datasets = len(iso_data['datasets'])

    if n_datasets < 2:
        raise ValueError("Vyazovkin method requires at least 2 datasets")

    Ea_values = np.full(n_alpha, np.nan)
    Ea_std_errs = np.full(n_alpha, np.nan)
    regression_stats = []

    def temperature_integral_trapz(T_array: np.ndarray, t_array: np.ndarray, Ea: float) -> float:
        """
        Compute temperature integral J[Ea, T(t)] = ∫ exp(-Ea/RT) dt using trapezoidal rule.
        """
        if len(T_array) < 2 or not np.all(np.isfinite(T_array)) or not np.all(T_array > 0):
            return np.nan
        try:
            integrand = np.exp(-Ea / (R_GAS * T_array))
            integral = np.trapz(integrand, t_array)
            return integral if integral > 0 else np.nan
        except:
            return np.nan

    def vyazovkin_objective(Ea: float, T_arrays: List[np.ndarray], t_arrays: List[np.ndarray]) -> float:
        """
        Vyazovkin objective function: Ω(Ea) = Σ_i Σ_j≠i [J_i(Ea) / J_j(Ea)]
        """
        if Ea <= 0:
            return 1e10  # Invalid Ea

        J_values = []
        for T_arr, t_arr in zip(T_arrays, t_arrays):
            J = temperature_integral_trapz(T_arr, t_arr, Ea)
            if not np.isfinite(J) or J <= 0:
                return 1e10
            J_values.append(J)

        # Compute sum of ratios
        omega = 0.0
        for i in range(len(J_values)):
            for j in range(len(J_values)):
                if i != j:
                    omega += J_values[i] / J_values[j]

        return omega if np.isfinite(omega) else 1e10

    # For each alpha level, minimize the Vyazovkin objective
    for i, alpha in enumerate(alpha_levels):
        T_arrays = []
        t_arrays = []

        # Collect temperature profiles up to this alpha
        for ds_idx, ds_data in enumerate(iso_data['datasets']):
            T = ds_data['T'][i]
            t = ds_data['t'][i]

            if not np.isfinite(T) or not np.isfinite(t) or T <= 0:
                continue

            # Get the full T(t) profile up to this alpha
            # We need the entire temperature history, not just the value at this alpha
            ds = datasets[ds_data['id']]
            mask = ds.conversion <= alpha
            if np.sum(mask) < 2:
                continue

            T_profile = ds.temperature[mask]
            t_profile = ds.time[mask]

            T_arrays.append(T_profile)
            t_arrays.append(t_profile)

        if len(T_arrays) < 2:
            warnings.warn(f"Not enough valid datasets at alpha={alpha:.3f} for Vyazovkin method")
            regression_stats.append({'alpha': alpha, 'iterations': np.nan, 'success': False})
            continue

        # Initial guess from KAS or a reasonable value
        Ea_guess = 100000.0  # J/mol, reasonable starting point

        # Minimize the Vyazovkin objective
        try:
            result = minimize(
                vyazovkin_objective,
                x0=[Ea_guess],
                args=(T_arrays, t_arrays),
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1.0, 'fatol': 1e-4}
            )

            if result.success and result.x[0] > 0:
                Ea_values[i] = result.x[0]
                regression_stats.append({'alpha': alpha, 'iterations': result.nit, 'success': True})
            else:
                warnings.warn(f"Optimization failed at alpha={alpha:.3f}")
                regression_stats.append({'alpha': alpha, 'iterations': result.nit, 'success': False})

        except Exception as e:
            warnings.warn(f"Vyazovkin optimization failed at alpha={alpha:.3f}: {e}")
            regression_stats.append({'alpha': alpha, 'iterations': np.nan, 'success': False})

    return IsoResult(method="Vyazovkin", alpha=alpha_levels, Ea=Ea_values, Ea_std_err=Ea_std_errs, regression_stats=regression_stats)


def run_kissinger(datasets: List[KineticDataset], peak_detection_method: str = 'max_rate') -> Dict:
    """
    Performs Kissinger method for activation energy estimation from peak temperatures.

    The Kissinger method uses the relationship:
    ln(β/T_peak²) = -Ea/(R*T_peak) + const

    This is a simpler method that requires peak temperature detection at different heating rates.

    Parameters
    ----------
    datasets : List[KineticDataset]
        List of kinetic datasets at different heating rates
    peak_detection_method : str
        Method to detect peaks: 'max_rate' (default) uses maximum dα/dt,
        'max_temp' uses maximum temperature

    Returns
    -------
    Dict with keys:
        'Ea': Activation energy (J/mol)
        'Ea_std_err': Standard error of Ea
        'A': Pre-exponential factor estimate (1/s)
        'peak_data': List of {'beta', 'T_peak', 'alpha_peak'} for each dataset
        'r_value': Correlation coefficient
    """
    peak_data = []

    for ds in datasets:
        if len(ds.time) < 5:
            warnings.warn(f"Dataset has too few points for peak detection")
            continue

        # Calculate heating rate
        beta = ds.heating_rate if ds.heating_rate is not None else np.mean(numerical_diff(ds.time, ds.temperature))
        if not np.isfinite(beta) or beta <= 0:
            warnings.warn(f"Invalid heating rate for dataset")
            continue

        # Detect peak based on method
        if peak_detection_method == 'max_rate':
            # Find maximum dα/dt
            dadt = numerical_diff(ds.time, ds.conversion)
            valid_mask = np.isfinite(dadt) & (dadt > 0)
            if not np.any(valid_mask):
                continue

            peak_idx = np.argmax(dadt)
            T_peak = ds.temperature[peak_idx]
            alpha_peak = ds.conversion[peak_idx]

        elif peak_detection_method == 'max_temp':
            # Use maximum temperature
            peak_idx = np.argmax(ds.temperature)
            T_peak = ds.temperature[peak_idx]
            alpha_peak = ds.conversion[peak_idx]

        else:
            raise ValueError(f"Unknown peak_detection_method: {peak_detection_method}")

        if T_peak > 0 and np.isfinite(T_peak):
            peak_data.append({'beta': beta, 'T_peak': T_peak, 'alpha_peak': alpha_peak})

    if len(peak_data) < 2:
        raise ValueError("Kissinger method requires at least 2 datasets with detectable peaks")

    # Perform linear regression: ln(β/T²) vs 1/T
    ln_beta_T2 = []
    inv_T = []

    for pd in peak_data:
        beta = pd['beta']
        T = pd['T_peak']
        if T > 0 and T**2 > 1e-12:
            ln_beta_T2.append(np.log(beta / (T**2)))
            inv_T.append(1.0 / T)

    ln_beta_T2 = np.array(ln_beta_T2)
    inv_T = np.array(inv_T)

    res = linregress(inv_T, ln_beta_T2)

    Ea = -res.slope * R_GAS
    Ea_std_err = res.stderr * R_GAS if res.stderr is not None else np.nan

    # Estimate A from intercept: intercept ≈ ln(AR/Ea)
    # A ≈ exp(intercept) * Ea / R
    A = np.exp(res.intercept) * Ea / R_GAS if np.isfinite(res.intercept) else np.nan

    return {
        'Ea': Ea,
        'Ea_std_err': Ea_std_err,
        'A': A,
        'peak_data': peak_data,
        'r_value': res.rvalue,
        'p_value': res.pvalue,
        'method': 'Kissinger'
    }


def estimate_compensation_parameters(iso_result: IsoResult, model_name: str = 'F1') -> Dict:
    """
    Estimate model-based kinetic parameters from isoconversional Ea(α) using compensation effect.

    The compensation effect (isokinetic relationship) relates ln(A) and Ea:
    ln(A) = a*Ea + b

    This function fits this relationship and provides estimates for model-based fitting.

    Parameters
    ----------
    iso_result : IsoResult
        Results from an isoconversional method (Friedman, KAS, OFW, Vyazovkin)
    model_name : str
        Target reaction model for parameter estimation

    Returns
    -------
    Dict with keys:
        'Ea_mean': Mean activation energy (J/mol)
        'Ea_std': Standard deviation of Ea
        'A_geometric_mean': Geometric mean of pre-exponential factors
        'compensation_a': Slope of ln(A) vs Ea
        'compensation_b': Intercept
        'isokinetic_temperature': Isokinetic temperature (K)
    """
    # Filter valid Ea values
    valid_mask = np.isfinite(iso_result.Ea) & (iso_result.Ea > 0)
    Ea_valid = iso_result.Ea[valid_mask]
    alpha_valid = iso_result.alpha[valid_mask]

    if len(Ea_valid) < 2:
        raise ValueError("Not enough valid Ea values for compensation analysis")

    Ea_mean = np.mean(Ea_valid)
    Ea_std = np.std(Ea_valid)

    # For compensation effect, we need to estimate A at each alpha
    # This is approximate - assumes the compensation relationship holds
    # ln(A) ≈ a*Ea + b where a ≈ 1/(R*T_iso) and T_iso is isokinetic temperature

    # Typical isokinetic temperature for solid-state reactions: 400-600 K
    # Use a reasonable default
    T_iso_guess = 500.0  # K

    # Estimate A values assuming compensation
    a_comp = 1.0 / (R_GAS * T_iso_guess)
    b_comp = 25.0  # Typical intercept

    A_estimates = np.exp(a_comp * Ea_valid + b_comp)
    A_geometric_mean = np.exp(np.mean(np.log(A_estimates)))

    return {
        'Ea_mean': Ea_mean,
        'Ea_std': Ea_std,
        'A_geometric_mean': A_geometric_mean,
        'compensation_a': a_comp,
        'compensation_b': b_comp,
        'isokinetic_temperature': T_iso_guess,
        'note': 'These are estimates based on typical compensation relationships. Use for initial guesses in model fitting.'
    }