# isoconversional.py
import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import linregress
import warnings
from typing import List, Dict, Tuple

from .datatypes import KineticDataset, IsoResult
from .utils import numerical_diff, R_GAS

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


        # Calculate d(alpha)/dt
        dadt = numerical_diff(time_unique, alpha_unique) # Use time-based derivative

        # Create interpolation functions
        try:
            interp_temp = interp1d(alpha_unique, temp_unique, bounds_error=False, fill_value=np.nan)
            interp_time = interp1d(alpha_unique, time_unique, bounds_error=False, fill_value=np.nan)
            interp_dadt = interp1d(alpha_unique, dadt, bounds_error=False, fill_value=np.nan)
        except ValueError as e:
            warnings.warn(f"Interpolation failed for dataset {i}: {e}. Skipping.")
            continue


        # Interpolate at target alpha levels
        T_at_alpha = interp_temp(alpha_levels)
        t_at_alpha = interp_time(alpha_levels)
        dadt_at_alpha = interp_dadt(alpha_levels)

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
                    regression_stats.append({'alpha': alpha, 'r_value': res.rvalue, 'p_value': res.pvalue, 'stderr': res.stderr, 'intercept': res.intercept})
                else:
                     regression_stats.append({'alpha': alpha, 'r_value': np.nan, 'p_value': np.nan, 'stderr': np.nan, 'intercept': np.nan})

            except ValueError as e:
                warnings.warn(f"Linear regression failed for alpha={alpha:.3f}: {e}")
                regression_stats.append({'alpha': alpha, 'r_value': np.nan, 'p_value': np.nan, 'stderr': np.nan, 'intercept': np.nan})
        else:
             regression_stats.append({'alpha': alpha, 'r_value': np.nan, 'p_value': np.nan, 'stderr': np.nan, 'intercept': np.nan})


    return IsoResult(method="Friedman", alpha=alpha_levels, Ea=Ea_values, Ea_std_err=Ea_std_errs, regression_stats=regression_stats)


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