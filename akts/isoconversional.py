# isoconversional.py
import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import linregress
from scipy.optimize import minimize_scalar # Needed for Vyazovkin
import warnings
from typing import List, Dict, Tuple, Callable, Optional

from .datatypes import KineticDataset, IsoResult
from .utils import numerical_diff, R_GAS, determine_auto_alpha_levels

# --- Helper Function (Keep as is) ---
def _prepare_iso_data(datasets: List[KineticDataset], alpha_levels: np.ndarray) -> Dict:
    """Interpolates T, t, and d(alpha)/dt at specified alpha levels for each dataset."""
    iso_data = {'alpha': alpha_levels, 'datasets': []}
    min_len_for_diff = 5 # Min points needed for SavGol

    for i, ds in enumerate(datasets):
        if len(ds.time) < min_len_for_diff:
            warnings.warn(f"Dataset {i} has too few points ({len(ds.time)}) for reliable differentiation. Skipping.")
            continue

        # Ensure alpha is monotonically increasing for interpolation
        # Handle potential duplicate alpha values by taking the first occurrence
        alpha_unique, idx_unique = np.unique(ds.conversion, return_index=True)
        # Ensure we sort by the unique alpha values to maintain order
        sort_idx_unique = np.argsort(alpha_unique)
        alpha_unique = alpha_unique[sort_idx_unique]
        idx_unique = idx_unique[sort_idx_unique]

        time_unique = ds.time[idx_unique]
        temp_unique = ds.temperature[idx_unique] # Assumed Kelvin

        if len(alpha_unique) < 2:
             warnings.warn(f"Dataset {i} has too few unique alpha points ({len(alpha_unique)}) for interpolation. Skipping.")
             continue

        # Calculate d(alpha)/dt using unique points
        # Ensure time is also sorted according to unique alpha indices
        dadt = numerical_diff(time_unique, alpha_unique) # Use time-based derivative

        # Create interpolation functions based on unique alpha
        try:
            # Use linear interpolation, allow extrapolation (fill with NaN outside range)
            interp_temp = interp1d(alpha_unique, temp_unique, kind='linear', bounds_error=False, fill_value=np.nan)
            interp_time = interp1d(alpha_unique, time_unique, kind='linear', bounds_error=False, fill_value=np.nan)
            # Interpolate dadt based on alpha
            interp_dadt = interp1d(alpha_unique, dadt, kind='linear', bounds_error=False, fill_value=np.nan)
        except ValueError as e:
            warnings.warn(f"Interpolation failed for dataset {i}: {e}. Skipping.")
            continue

        # Interpolate at target alpha levels
        T_at_alpha = interp_temp(alpha_levels)
        t_at_alpha = interp_time(alpha_levels)
        dadt_at_alpha = interp_dadt(alpha_levels)

        # Calculate heating rate beta = dT/dt (can be variable)
        # Use smoothed derivative dT/dt on the unique points
        dTdt = numerical_diff(time_unique, temp_unique)
        # Interpolate dT/dt at the target alpha levels using time interpolation first
        # Need to handle NaNs in t_at_alpha
        dTdt_at_alpha = np.full_like(t_at_alpha, np.nan)
        valid_t_mask = np.isfinite(t_at_alpha)
        if np.any(valid_t_mask):
             try:
                 interp_dTdt = interp1d(time_unique, dTdt, kind='linear', bounds_error=False, fill_value=np.nan)
                 dTdt_at_alpha[valid_t_mask] = interp_dTdt(t_at_alpha[valid_t_mask])
             except ValueError as e_dtdt:
                  warnings.warn(f"Interpolation of dT/dt failed for dataset {i}: {e_dtdt}")


        # Use average heating rate if constant heating was intended or dTdt fails
        avg_beta_K_s = np.nan # Default to NaN
        if ds.heating_rate is not None:
             # Assume heating_rate is in K/min, convert to K/s
             avg_beta_K_s = ds.heating_rate / 60.0
        else:
             finite_dtdt = dTdt[np.isfinite(dTdt)]
             if len(finite_dtdt) > 0:
                  avg_beta_K_s = np.mean(finite_dtdt)

        # Use interpolated dTdt if available and positive, else average beta
        # Ensure avg_beta_K_s is valid before using it as fallback
        fallback_beta = avg_beta_K_s if np.isfinite(avg_beta_K_s) and avg_beta_K_s > 1e-9 else 10.0/60.0 # Default guess if needed

        beta_at_alpha = np.where(np.isfinite(dTdt_at_alpha) & (dTdt_at_alpha > 1e-9),
                                 dTdt_at_alpha,
                                 fallback_beta)

        dataset_iso_data = {
            'T': T_at_alpha, # Kelvin
            't': t_at_alpha, # Seconds (assuming input time is seconds)
            'dadt': dadt_at_alpha, # 1/s
            'beta': beta_at_alpha, # K/s
            'id': i
        }
        iso_data['datasets'].append(dataset_iso_data)

    if not iso_data['datasets']:
        raise ValueError("No valid datasets found for isoconversional analysis after preprocessing.")

    return iso_data

# --- Friedman Method (Keep as is) ---
def run_friedman(datasets: List[KineticDataset], alpha_levels: np.ndarray = np.linspace(0.05, 0.95, 19)) -> IsoResult:
    """Performs Friedman isoconversional analysis."""
    iso_data = _prepare_iso_data(datasets, alpha_levels)
    n_alpha = len(alpha_levels)
    Ea_values = np.full(n_alpha, np.nan)
    Ea_std_errs = np.full(n_alpha, np.nan)
    regression_stats = []

    for i, alpha in enumerate(alpha_levels):
        ln_rate = []
        inv_T = []
        valid_points_count = 0 # Count valid points for this alpha
        for ds_data in iso_data['datasets']:
            rate = ds_data['dadt'][i]
            T = ds_data['T'][i]
            # Check for valid, positive rate and temperature
            if np.isfinite(rate) and rate > 1e-12 and np.isfinite(T) and T > 0:
                ln_rate.append(np.log(rate))
                inv_T.append(1.0 / T)
                valid_points_count += 1

        # **** ADD DEBUG PRINT ****
        print(f"Debug Friedman: alpha={alpha:.3f}, N_valid={valid_points_count}, inv_T={inv_T}, ln_rate={ln_rate}")
        # **** END DEBUG PRINT ****

        if valid_points_count >= 2: # Need at least 2 points for linear regression
            ln_rate = np.array(ln_rate)
            inv_T = np.array(inv_T)
            try:
                # Perform linear regression: ln(rate) vs 1/T
                res = linregress(inv_T, ln_rate)
                if np.isfinite(res.slope):
                    Ea = -res.slope * R_GAS # Ea = -Slope * R
                    Ea_values[i] = Ea
                    # Std Err of slope * R_GAS
                    Ea_std_errs[i] = res.stderr * R_GAS if res.stderr is not None and np.isfinite(res.stderr) else np.nan
                    regression_stats.append({'alpha': alpha, 'r_value': res.rvalue, 'p_value': res.pvalue, 'stderr': res.stderr, 'intercept': res.intercept, 'n_points': valid_points_count})
                else:
                     regression_stats.append({'alpha': alpha, 'r_value': np.nan, 'p_value': np.nan, 'stderr': np.nan, 'intercept': np.nan, 'n_points': valid_points_count})

            except ValueError as e:
                warnings.warn(f"Friedman: Linear regression failed for alpha={alpha:.3f}: {e}")
                regression_stats.append({'alpha': alpha, 'r_value': np.nan, 'p_value': np.nan, 'stderr': np.nan, 'intercept': np.nan, 'n_points': valid_points_count})
        else:
             # Not enough valid points at this alpha level
             regression_stats.append({'alpha': alpha, 'r_value': np.nan, 'p_value': np.nan, 'stderr': np.nan, 'intercept': np.nan, 'n_points': valid_points_count})

    return IsoResult(method="Friedman", alpha=alpha_levels, Ea=Ea_values, Ea_std_err=Ea_std_errs, regression_stats=regression_stats)

# --- KAS Method (Keep as is) ---
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
        valid_points_count = 0
        for ds_data in iso_data['datasets']:
            beta = ds_data['beta'][i] # Use beta at specific alpha (K/s)
            T = ds_data['T'][i] # Kelvin
            # Check for valid, positive beta and temperature
            if np.isfinite(beta) and beta > 1e-9 and np.isfinite(T) and T > 1e-9:
                 T_squared = T**2
                 if T_squared > 1e-12: # Avoid division by zero
                     ln_beta_T2.append(np.log(beta / T_squared))
                     inv_T.append(1.0 / T)
                     valid_points_count += 1

        if valid_points_count >= 2:
            ln_beta_T2 = np.array(ln_beta_T2)
            inv_T = np.array(inv_T)
            try:
                # Perform linear regression: ln(beta/T^2) vs 1/T
                res = linregress(inv_T, ln_beta_T2)
                if np.isfinite(res.slope):
                    Ea = -res.slope * R_GAS # Ea = -Slope * R
                    Ea_values[i] = Ea
                    Ea_std_errs[i] = res.stderr * R_GAS if res.stderr is not None and np.isfinite(res.stderr) else np.nan
                    regression_stats.append({'alpha': alpha, 'r_value': res.rvalue, 'p_value': res.pvalue, 'stderr': res.stderr, 'intercept': res.intercept, 'n_points': valid_points_count})
                else:
                    regression_stats.append({'alpha': alpha, 'r_value': np.nan, 'p_value': np.nan, 'stderr': np.nan, 'intercept': np.nan, 'n_points': valid_points_count})
            except ValueError as e:
                warnings.warn(f"KAS: Linear regression failed for alpha={alpha:.3f}: {e}")
                regression_stats.append({'alpha': alpha, 'r_value': np.nan, 'p_value': np.nan, 'stderr': np.nan, 'intercept': np.nan, 'n_points': valid_points_count})
        else:
            regression_stats.append({'alpha': alpha, 'r_value': np.nan, 'p_value': np.nan, 'stderr': np.nan, 'intercept': np.nan, 'n_points': valid_points_count})

    return IsoResult(method="KAS", alpha=alpha_levels, Ea=Ea_values, Ea_std_err=Ea_std_errs, regression_stats=regression_stats)

# --- OFW Method (Keep as is) ---
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
        valid_points_count = 0
        for ds_data in iso_data['datasets']:
            beta = ds_data['beta'][i] # K/s
            T = ds_data['T'][i] # Kelvin
            # Check for valid, positive beta and temperature
            if np.isfinite(beta) and beta > 1e-9 and np.isfinite(T) and T > 0:
                ln_beta.append(np.log(beta))
                inv_T.append(1.0 / T)
                valid_points_count += 1

        if valid_points_count >= 2:
            ln_beta = np.array(ln_beta)
            inv_T = np.array(inv_T)
            try:
                # Perform linear regression: ln(beta) vs 1/T
                res = linregress(inv_T, ln_beta)
                if np.isfinite(res.slope):
                    # Note: Slope is approx -1.052 * Ea / R for Doyle approx.
                    Ea = -res.slope * R_GAS / 1.052
                    Ea_values[i] = Ea
                    Ea_std_errs[i] = (res.stderr * R_GAS / 1.052) if res.stderr is not None and np.isfinite(res.stderr) else np.nan
                    regression_stats.append({'alpha': alpha, 'r_value': res.rvalue, 'p_value': res.pvalue, 'stderr': res.stderr, 'intercept': res.intercept, 'n_points': valid_points_count})
                else:
                    regression_stats.append({'alpha': alpha, 'r_value': np.nan, 'p_value': np.nan, 'stderr': np.nan, 'intercept': np.nan, 'n_points': valid_points_count})
            except ValueError as e:
                warnings.warn(f"OFW: Linear regression failed for alpha={alpha:.3f}: {e}")
                regression_stats.append({'alpha': alpha, 'r_value': np.nan, 'p_value': np.nan, 'stderr': np.nan, 'intercept': np.nan, 'n_points': valid_points_count})
        else:
            regression_stats.append({'alpha': alpha, 'r_value': np.nan, 'p_value': np.nan, 'stderr': np.nan, 'intercept': np.nan, 'n_points': valid_points_count})

    return IsoResult(method="OFW", alpha=alpha_levels, Ea=Ea_values, Ea_std_err=Ea_std_errs, regression_stats=regression_stats)


# --- Vyazovkin Method Implementation ---

def _vyazovkin_temp_integral_approx(Ea: float, T: float) -> float:
    """
    Calculates the temperature integral I(Ea, T) using the Senum-Yang approximation.
    I(Ea, T) = integral from 0 to T of exp(-Ea / RT) dT
    Approximation: (R * T^2 / Ea) * exp(-x) * P(x) / Q(x) where x = Ea / RT
    P(x) = x^2 + 1.8*x + 0.8
    Q(x) = x^2 + 2.8*x + 1.8
    Args:
        Ea (float): Activation energy in J/mol.
        T (float): Temperature in Kelvin.

    Returns:
        float: Approximate value of the temperature integral. Returns 0 if inputs invalid.
    """
    if not np.isfinite(Ea) or Ea <= 0 or not np.isfinite(T) or T <= 0:
        return 0.0
    x = Ea / (R_GAS * T)
    # Avoid potential overflow in exp(-x) for very large x
    if x > 700: # exp(-700) is extremely small
        return 0.0

    exp_neg_x = np.exp(-x)
    numerator = x**2 + 1.8 * x + 0.8
    denominator = x**2 + 2.8 * x + 1.8

    if denominator < 1e-100: # Avoid division by zero
        return 0.0

    integral_val = (R_GAS * T**2 / Ea) * exp_neg_x * (numerator / denominator)
    return integral_val

def _vyazovkin_objective(Ea: float, T_alpha_list: np.ndarray, beta_list: np.ndarray) -> float:
    """
    Objective function Psi(Ea) for the Vyazovkin method for a single alpha level.
    Minimizes sum_{i=1..n} sum_{j!=i} [ I(Ea, T_i) * beta_j ] / [ I(Ea, T_j) * beta_i ]

    Args:
        Ea (float): Activation energy guess (J/mol).
        T_alpha_list (np.ndarray): Array of temperatures (K) at this alpha for different runs.
        beta_list (np.ndarray): Array of heating rates (K/s) at this alpha for different runs.

    Returns:
        float: Value of the objective function Psi(Ea). Returns infinity if calculation fails.
    """
    if Ea <= 0: # Ea must be positive
        return np.inf

    n = len(T_alpha_list)
    if n < 2:
        return 0.0 # Cannot calculate objective with less than 2 runs

    # Calculate temperature integrals for all runs at this Ea
    integrals = np.array([_vyazovkin_temp_integral_approx(Ea, T) for T in T_alpha_list])

    # Check for invalid integral values (e.g., zero or NaN)
    if np.any(integrals <= 0) or np.any(~np.isfinite(integrals)):
        # Penalize Ea values leading to invalid integrals
        # This can happen if Ea is extremely high or low for the given T range
        return np.inf

    # Calculate the double summation
    total_sum = 0.0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # Avoid division by zero if beta_i or integral_j is zero (already checked integrals > 0)
            if abs(beta_list[i]) < 1e-12:
                 return np.inf # Should not happen if beta > 0 check done before

            term = (integrals[i] * beta_list[j]) / (integrals[j] * beta_list[i])
            total_sum += term

    # Check for NaN or Inf in the final sum
    if not np.isfinite(total_sum):
        return np.inf

    return total_sum

def run_vyazovkin(datasets: List[KineticDataset],
                  alpha_levels: np.ndarray = np.linspace(0.05, 0.95, 19),
                  ea_bounds_kJmol: Tuple[float, float] = (1.0, 500.0),
                  opt_tolerance: float = 1e-6
                  ) -> IsoResult:
    """
    Performs non-linear Vyazovkin isoconversional analysis.

    Args:
        datasets: List of KineticDataset objects.
        alpha_levels: Array of conversion levels to analyze.
        ea_bounds_kJmol: Tuple (min_Ea, max_Ea) in kJ/mol for the optimizer search.
        opt_tolerance: Tolerance for the optimization routine.

    Returns:
        IsoResult object containing Ea values at each alpha level.
        Standard errors are not typically calculated directly by this method.
    """
    print("Running Vyazovkin analysis...")
    iso_data = _prepare_iso_data(datasets, alpha_levels)
    n_alpha = len(alpha_levels)
    Ea_values = np.full(n_alpha, np.nan)
    # Standard errors are generally not derived directly from the Vyazovkin minimization
    Ea_std_errs = np.full(n_alpha, np.nan)
    regression_stats = [] # Store optimization results if needed

    # Convert Ea bounds to J/mol
    ea_bounds_Jmol = (ea_bounds_kJmol[0] * 1000.0, ea_bounds_kJmol[1] * 1000.0)

    for i, alpha in enumerate(alpha_levels):
        # Collect valid (T, beta) pairs for this alpha level
        T_list = []
        beta_list = []
        valid_points_count = 0
        for ds_data in iso_data['datasets']:
            T = ds_data['T'][i]
            beta = ds_data['beta'][i]
            # Ensure T and beta are valid positive numbers
            if np.isfinite(T) and T > 0 and np.isfinite(beta) and beta > 1e-9:
                T_list.append(T)
                beta_list.append(beta)
                valid_points_count += 1

        if valid_points_count < 2:
            warnings.warn(f"Vyazovkin: Less than 2 valid data points for alpha={alpha:.3f}. Skipping.")
            regression_stats.append({'alpha': alpha, 'min_Ea_Jmol': np.nan, 'fun_value': np.nan, 'n_points': valid_points_count})
            continue

        T_array = np.array(T_list)
        beta_array = np.array(beta_list)

        # Define the objective function specific to this alpha's data
        objective_func = lambda Ea: _vyazovkin_objective(Ea, T_array, beta_array)

        # Minimize the objective function to find Ea
        try:
            # Use minimize_scalar for bounded 1D optimization
            result = minimize_scalar(
                objective_func,
                bounds=ea_bounds_Jmol,
                method='bounded',
                options={'xatol': opt_tolerance, 'maxiter': 500} # Add maxiter
            )

            if result.success and np.isfinite(result.x):
                Ea_values[i] = result.x # Optimized Ea in J/mol
                regression_stats.append({'alpha': alpha, 'min_Ea_Jmol': result.x, 'fun_value': result.fun, 'n_points': valid_points_count})
                # print(f"  Alpha={alpha:.3f}, Ea={result.x/1000:.1f} kJ/mol (Obj={result.fun:.3g})") # Optional progress print
            else:
                warnings.warn(f"Vyazovkin: Optimization failed for alpha={alpha:.3f}. Message: {result.message}")
                regression_stats.append({'alpha': alpha, 'min_Ea_Jmol': np.nan, 'fun_value': np.nan, 'n_points': valid_points_count})

        except Exception as e:
            warnings.warn(f"Vyazovkin: Optimization encountered an error for alpha={alpha:.3f}: {e}")
            regression_stats.append({'alpha': alpha, 'min_Ea_Jmol': np.nan, 'fun_value': np.nan, 'n_points': valid_points_count})

    print("Vyazovkin analysis complete.")
    return IsoResult(method="Vyazovkin", alpha=alpha_levels, Ea=Ea_values, Ea_std_err=Ea_std_errs, regression_stats=regression_stats)

# --- Add other isoconversional methods if needed ---

def run_iso_isothermal(
    datasets: List[KineticDataset],
    alpha_levels: Optional[np.ndarray] = None,
    num_levels: int = 10,
    min_overlap: int = 2
    ) -> IsoResult:
    """
    Performs isoconversional analysis using time-to-alpha from multiple isothermal runs.
    Automatically determines alpha levels if not provided.

    Args:
        datasets: List of KineticDataset objects (MUST be isothermal).
        alpha_levels: Array of conversion levels to analyze. If None, auto-determined.
        num_levels: Number of alpha levels to auto-generate if alpha_levels is None.
        min_overlap: Minimum number of datasets that must overlap at each alpha level.

    Returns:
        IsoResult object.
    """
    print("Running Isothermal Isoconversional (ln(t) vs 1/T) analysis...")

    # --- Auto-determine alpha levels if not provided ---
    if alpha_levels is None:
        print("alpha_levels not provided, attempting auto-determination...")
        alpha_levels = determine_auto_alpha_levels(
            datasets,
            num_levels=num_levels,
            min_overlap=3, #min_overlap,
            alpha_bounds=(0.01, 0.95)  # Adjust bounds as needed
        )
        if alpha_levels is None:
            # Auto-determination failed, return an empty/failure result
            return IsoResult(
                method="Isothermal_ln(t)_vs_1/T (Failed Auto Alpha)",
                alpha=np.array([]),
                Ea=np.array([])
            )
    # --- End auto-determination ---

    # Use _prepare_iso_data just to get interpolated times and temperatures
    try:
        iso_data = _prepare_iso_data(datasets, alpha_levels)
    except ValueError as e_prep:
        print(f"Error preparing iso data: {e_prep}")
        return IsoResult(
            method="Isothermal_ln(t)_vs_1/T (Prep Failed)",
            alpha=alpha_levels,
            Ea=np.full_like(alpha_levels, np.nan)
        )

    n_alpha = len(alpha_levels)
    Ea_values = np.full(n_alpha, np.nan)
    Ea_std_errs = np.full(n_alpha, np.nan)
    regression_stats = []

    # Check if data looks isothermal
    for ds_data in iso_data['datasets']:
        if len(ds_data['T']) > 0 and np.std(ds_data['T'][np.isfinite(ds_data['T'])]) > 1.0:
            warnings.warn(f"Dataset {ds_data['id']} does not appear isothermal. Results may be inaccurate.")

    for i, alpha in enumerate(alpha_levels):
        ln_t = []
        inv_T = []
        valid_points_count = 0
        for ds_data in iso_data['datasets']:
            t_alpha = ds_data['t'][i]
            T_iso = ds_data['T'][i]
            if np.isfinite(t_alpha) and t_alpha > 1e-9 and np.isfinite(T_iso) and T_iso > 0:
                ln_t.append(np.log(t_alpha))
                inv_T.append(1.0 / T_iso)
                valid_points_count += 1

        if valid_points_count >= min_overlap:
            ln_t = np.array(ln_t)
            inv_T = np.array(inv_T)
            try:
                res = linregress(inv_T, ln_t)
                if np.isfinite(res.slope):
                    Ea = res.slope * R_GAS
                    Ea_values[i] = Ea
                    Ea_std_errs[i] = res.stderr * R_GAS if res.stderr is not None and np.isfinite(res.stderr) else np.nan
                    regression_stats.append({
                        'alpha': alpha,
                        'r_value': res.rvalue,
                        'p_value': res.pvalue,
                        'stderr': res.stderr,
                        'intercept': res.intercept,
                        'n_points': valid_points_count
                    })
                else:
                    regression_stats.append({
                        'alpha': alpha,
                        'r_value': np.nan,
                        'p_value': np.nan,
                        'stderr': np.nan,
                        'intercept': np.nan,
                        'n_points': valid_points_count
                    })
            except ValueError as e:
                warnings.warn(f"Isothermal Iso: Linear regression failed for alpha={alpha:.3f}: {e}")
                regression_stats.append({
                    'alpha': alpha,
                    'r_value': np.nan,
                    'p_value': np.nan,
                    'stderr': np.nan,
                    'intercept': np.nan,
                    'n_points': valid_points_count
                })
        else:
            regression_stats.append({
                'alpha': alpha,
                'r_value': np.nan,
                'p_value': np.nan,
                'stderr': np.nan,
                'intercept': np.nan,
                'n_points': valid_points_count
            })

    print("Isothermal Isoconversional analysis complete.")
    return IsoResult(
        method="Isothermal_ln(t)_vs_1/T",
        alpha=alpha_levels,
        Ea=Ea_values,
        Ea_std_err=Ea_std_errs,
        regression_stats=regression_stats
    )