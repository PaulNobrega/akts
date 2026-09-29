"""
Convenience plotting functions for kinetic analysis.

Thin wrappers around matplotlib for common plots: Ea(α), fit overlays,
bootstrap CI bands, Arrhenius plots, etc.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use headless backend for compatibility
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import List, Optional, Tuple, Dict, Union
import warnings

from .datatypes import IsoResult, FitResult, KineticDataset, PredictionResult, BootstrapResult


def plot_ea_vs_alpha(
    iso_result: IsoResult,
    figsize: Tuple[float, float] = (10, 6),
    show_error_bars: bool = True,
    ax: Optional[plt.Axes] = None
) -> Figure:
    """
    Plot activation energy Ea vs. conversion α (isoconversional plot).

    Parameters
    ----------
    iso_result : IsoResult
        Isoconversional analysis result (from run_friedman, run_kas, etc.)
    figsize : Tuple[float, float], default=(10, 6)
        Figure size in inches (only used if ax is None)
    show_error_bars : bool, default=True
        Show error bars if Ea_std_err is available
    ax : plt.Axes, optional
        Existing axes to plot on. If None, creates new figure.

    Returns
    -------
    Figure
        Matplotlib figure object

    Examples
    --------
    >>> from akts import run_friedman, plot_ea_vs_alpha
    >>> friedman = run_friedman(datasets)
    >>> fig = plot_ea_vs_alpha(friedman)
    >>> fig.savefig('ea_vs_alpha.png')
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    alpha_pct = iso_result.alpha * 100
    Ea_kJ = iso_result.Ea / 1000  # Convert to kJ/mol

    # Plot Ea vs alpha
    if show_error_bars and iso_result.Ea_std_err is not None:
        Ea_std_err_kJ = iso_result.Ea_std_err / 1000
        ax.errorbar(
            alpha_pct,
            Ea_kJ,
            yerr=Ea_std_err_kJ,
            fmt='o-',
            capsize=4,
            capthick=1.5,
            linewidth=2,
            markersize=6,
            label=iso_result.method
        )
    else:
        ax.plot(
            alpha_pct,
            Ea_kJ,
            'o-',
            linewidth=2,
            markersize=6,
            label=iso_result.method
        )

    ax.set_xlabel('Conversion α (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Activation Energy Ea (kJ/mol)', fontsize=12, fontweight='bold')
    ax.set_title(f'Isoconversional Analysis: {iso_result.method}', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_fit_overlay(
    datasets: List[KineticDataset],
    fit_result: FitResult,
    prediction: Optional[Union[PredictionResult, List[PredictionResult]]] = None,
    figsize: Tuple[float, float] = (10, 6),
    time_units: str = 'seconds',
    ax: Optional[plt.Axes] = None,
    show_ci: bool = True
) -> Figure:
    """
    Plot experimental data with model fit overlay and optional confidence intervals.

    Parameters
    ----------
    datasets : List[KineticDataset]
        Experimental data
    fit_result : FitResult
        Model fit result
    prediction : PredictionResult or list of PredictionResult, optional
        Model prediction(s) to overlay. A list is matched by index to datasets,
        allowing each temperature series to have its own fit and confidence band.
    figsize : Tuple[float, float], default=(10, 6)
        Figure size in inches
    time_units : str, default='seconds'
        Time units for x-axis ('seconds', 'hours', 'days', 'months', 'years')
    ax : plt.Axes, optional
        Existing axes to plot on
    show_ci : bool, default=True
        If True and prediction has CI bands, display them as shaded region

    Returns
    -------
    Figure
        Matplotlib figure object

    Examples
    --------
    >>> fig = plot_fit_overlay(datasets, fit_result)
    >>> fig.savefig('fit_overlay.png')

    >>> # With bootstrap confidence intervals
    >>> prediction_with_ci = predict_conversion(..., bootstrap_result=bootstrap)
    >>> fig = plot_fit_overlay(datasets, fit_result, prediction_with_ci, show_ci=True)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Time conversion factors
    time_conversion = {
        'seconds': 1.0,
        'hours': 1 / 3600,
        'days': 1 / (24 * 3600),
        'months': 1 / (30.44 * 24 * 3600),
        'years': 1 / (365.25 * 24 * 3600)
    }
    time_factor = time_conversion.get(time_units, 1.0)

    # Normalize predictions to one optional prediction per dataset.
    if isinstance(prediction, (list, tuple)):
        predictions = list(prediction)
        if len(predictions) != len(datasets):
            raise ValueError("A prediction list must contain one prediction per dataset")
    elif prediction is None:
        predictions = []
    else:
        predictions = [prediction]

    colors = plt.get_cmap('tab10')(np.linspace(0, 1, max(len(datasets), 1)))
    has_any_ci = False

    # Plot each experimental series and its corresponding fit and CI.
    for index, ds in enumerate(datasets):
        temp_K = np.mean(ds.temperature)
        temp_C = temp_K - 273.15
        time_plot = ds.time * time_factor
        conversion_pct = ds.conversion * 100
        color = colors[index]

        label = f'Data at {temp_C:.0f}°C'
        ax.scatter(
            time_plot,
            conversion_pct,
            label=label,
            s=60,
            alpha=0.7,
            color=color,
            edgecolors='black',
            linewidths=0.5
        )

        if index >= len(predictions):
            continue
        series_prediction = predictions[index]
        prediction_time_plot = series_prediction.time * time_factor
        prediction_conversion_pct = series_prediction.conversion * 100

        # Plot CI bands if available and requested
        has_ci = (series_prediction.conversion_ci is not None and
                  show_ci and
                  len(series_prediction.conversion_ci) == 2)

        if has_ci:
            ci_lower_pct = series_prediction.conversion_ci[0] * 100
            ci_upper_pct = series_prediction.conversion_ci[1] * 100
            ax.fill_between(
                prediction_time_plot,
                ci_lower_pct,
                ci_upper_pct,
                alpha=0.25,
                color=color,
                label=f'95% CI at {temp_C:.0f}°C'
            )
            has_any_ci = True

        fit_label = f'{fit_result.model_name} fit at {temp_C:.0f}°C' if len(predictions) > 1 else f'{fit_result.model_name} fit'
        ax.plot(
            prediction_time_plot,
            prediction_conversion_pct,
            color=color,
            linestyle='-',
            linewidth=2,
            label=fit_label
        )

    ax.set_xlabel(f'Time ({time_units})', fontsize=12, fontweight='bold')
    ax.set_ylabel('Degradation (%)', fontsize=12, fontweight='bold')
    title = 'Experimental Data vs Model Fit'
    if has_any_ci:
        title += ' with 95% CI'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_bootstrap_ci_bands(
    prediction: PredictionResult,
    figsize: Tuple[float, float] = (10, 6),
    time_units: str = 'seconds',
    ax: Optional[plt.Axes] = None
) -> Figure:
    """
    Plot prediction with bootstrap confidence interval bands.

    Parameters
    ----------
    prediction : PredictionResult
        Prediction with CI bands (conversion_ci must not be None)
    figsize : Tuple[float, float], default=(10, 6)
        Figure size in inches
    time_units : str, default='seconds'
        Time units for x-axis
    ax : plt.Axes, optional
        Existing axes to plot on

    Returns
    -------
    Figure
        Matplotlib figure object

    Examples
    --------
    >>> prediction = predict_conversion(fit_result, ..., bootstrap_result=bootstrap)
    >>> fig = plot_bootstrap_ci_bands(prediction, time_units='months')
    """
    if prediction.conversion_ci is None:
        raise ValueError("prediction.conversion_ci is None - bootstrap CI not available")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Time conversion
    time_conversion = {
        'seconds': 1.0,
        'hours': 1 / 3600,
        'days': 1 / (24 * 3600),
        'months': 1 / (30.44 * 24 * 3600),
        'years': 1 / (365.25 * 24 * 3600)
    }
    time_factor = time_conversion.get(time_units, 1.0)

    time_plot = prediction.time * time_factor
    conversion_pct = prediction.conversion * 100
    ci_lower_pct = prediction.conversion_ci[0] * 100
    ci_upper_pct = prediction.conversion_ci[1] * 100

    # Plot mean prediction
    ax.plot(time_plot, conversion_pct, 'b-', linewidth=2, label='Mean Prediction')

    # Plot CI bands
    ax.fill_between(
        time_plot,
        ci_lower_pct,
        ci_upper_pct,
        alpha=0.3,
        color='blue',
        label='95% Confidence Interval'
    )

    ax.set_xlabel(f'Time ({time_units})', fontsize=12, fontweight='bold')
    ax.set_ylabel('Degradation (%)', fontsize=12, fontweight='bold')
    ax.set_title('Prediction with Bootstrap Confidence Intervals', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_arrhenius(
    fit_result: FitResult,
    figsize: Tuple[float, float] = (10, 6),
    ax: Optional[plt.Axes] = None,
    datasets: Optional[List[KineticDataset]] = None,
    bootstrap_result: Optional['BootstrapResult'] = None
) -> Figure:
    """
    Plot Arrhenius plot: ln(k) vs 1/T with optional data points and CI.

    Only works for single-step models with Ea and A parameters.

    Parameters
    ----------
    fit_result : FitResult
        Model fit result with Ea and A parameters
    figsize : Tuple[float, float], default=(10, 6)
        Figure size in inches
    ax : plt.Axes, optional
        Existing axes to plot on
    datasets : List[KineticDataset], optional
        Original datasets to extract rate constants from (shown as points)
    bootstrap_result : BootstrapResult, optional
        Bootstrap results to show CI bands

    Returns
    -------
    Figure
        Matplotlib figure object

    Examples
    --------
    >>> fig = plot_arrhenius(fit_result)
    >>> fig.savefig('arrhenius.png')

    >>> # With data points and CI
    >>> fig = plot_arrhenius(fit_result, datasets=datasets, bootstrap_result=bootstrap)
    """
    if 'Ea' not in fit_result.parameters or 'A' not in fit_result.parameters:
        raise ValueError("fit_result must have 'Ea' and 'A' parameters for Arrhenius plot")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    Ea = fit_result.parameters['Ea']
    A = fit_result.parameters['A']
    R = 8.314  # Gas constant

    # Determine temperature range from data if available, otherwise use default
    if datasets and len(datasets) > 0:
        all_temps = np.concatenate([ds.temperature for ds in datasets])
        T_min = max(all_temps.min() - 10, 273)
        T_max = min(all_temps.max() + 10, 450)
    else:
        T_min, T_max = 273, 400

    # Plot bootstrap CI bands if available
    if bootstrap_result is not None and hasattr(bootstrap_result, 'replicate_params'):
        # Extract Ea and A from bootstrap replicates
        replicate_params = bootstrap_result.replicate_params
        if replicate_params and len(replicate_params) > 0:
            # Generate predictions for each replicate
            T_range = np.linspace(T_min, T_max, 100)
            inv_T = 1000 / T_range
            ln_k_replicates = []

            for rep_params in replicate_params[:100]:  # Limit to 100 for performance
                if 'Ea' in rep_params and 'A' in rep_params:
                    Ea_rep = rep_params['Ea']
                    A_rep = rep_params['A']
                    ln_k_rep = np.log(A_rep) - Ea_rep / (R * T_range)
                    ln_k_replicates.append(ln_k_rep)

            if ln_k_replicates:
                ln_k_array = np.array(ln_k_replicates)
                ln_k_lower = np.nanpercentile(ln_k_array, 2.5, axis=0)
                ln_k_upper = np.nanpercentile(ln_k_array, 97.5, axis=0)

                # Plot CI band
                ax.fill_between(
                    inv_T, ln_k_lower, ln_k_upper,
                    alpha=0.25, color='blue', label='95% CI'
                )

    # Create temperature range for fitted line
    T_range = np.linspace(T_min, T_max, 100)
    inv_T = 1000 / T_range  # 1000/T for better scaling
    ln_k = np.log(A) - Ea / (R * T_range)

    ax.plot(inv_T, ln_k, 'b-', linewidth=2, label='Fitted line')

    # Plot data points if datasets provided
    if datasets and len(datasets) > 0:
        # For F1 model, k can be estimated from slope of -ln(1-α) vs t
        # For other models, this is approximate
        for ds in datasets:
            T_mean = np.mean(ds.temperature)
            # Simple rate estimation: k ≈ α_final / t_final for small conversions
            # More accurate: fit -ln(1-α) = kt for F1
            if np.max(ds.conversion) < 0.95 and np.max(ds.conversion) > 0.1:
                # Use linear fit of -ln(1-α) vs t
                mask = (ds.conversion > 0.01) & (ds.conversion < 0.99)
                if np.sum(mask) > 2:
                    t_fit = ds.time[mask]
                    alpha_fit = ds.conversion[mask]
                    ln_term = -np.log(1 - alpha_fit)
                    # Linear regression
                    k_est = np.polyfit(t_fit, ln_term, 1)[0]
                    if k_est > 0:
                        ln_k_data = np.log(k_est)
                        inv_T_data = 1000 / T_mean
                        ax.scatter(inv_T_data, ln_k_data, s=80, c='red',
                                 edgecolors='black', linewidths=1.5, zorder=5,
                                 label='Data points' if ds == datasets[0] else '')

    # Add fit parameters as text
    Ea_kJ = Ea / 1000
    text = f'Ea = {Ea_kJ:.1f} kJ/mol\nln(A) = {np.log(A):.2f}\nA = {A:.2e} s⁻¹'
    ax.text(
        0.05, 0.95,
        text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )

    ax.set_xlabel('1000/T (K⁻¹)', fontsize=12, fontweight='bold')
    ax.set_ylabel('ln(k)', fontsize=12, fontweight='bold')
    ax.set_title('Arrhenius Plot', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_parameter_distributions(
    bootstrap_result: BootstrapResult,
    parameters: Optional[List[str]] = None,
    figsize: Tuple[float, float] = (12, 4),
    bins: int = 30
) -> Figure:
    """
    Plot histograms of bootstrap parameter distributions.

    Parameters
    ----------
    bootstrap_result : BootstrapResult
        Bootstrap analysis result
    parameters : List[str], optional
        List of parameters to plot. If None, plots all parameters.
    figsize : Tuple[float, float], default=(12, 4)
        Figure size in inches
    bins : int, default=30
        Number of histogram bins

    Returns
    -------
    Figure
        Matplotlib figure object

    Examples
    --------
    >>> fig = plot_parameter_distributions(bootstrap_result, parameters=['Ea', 'A'])
    >>> fig.savefig('parameter_distributions.png')
    """
    if parameters is None:
        parameters = list(bootstrap_result.parameter_distributions.keys())

    n_params = len(parameters)
    fig, axes = plt.subplots(1, n_params, figsize=figsize)

    if n_params == 1:
        axes = [axes]

    for ax, param_name in zip(axes, parameters):
        if param_name not in bootstrap_result.parameter_distributions:
            warnings.warn(f"Parameter '{param_name}' not found in bootstrap result")
            continue

        values = bootstrap_result.parameter_distributions[param_name]
        ci_lower, ci_upper = bootstrap_result.parameter_ci[param_name]

        # Histogram
        ax.hist(values, bins=bins, alpha=0.7, edgecolor='black', linewidth=0.5)

        # CI lines
        ax.axvline(ci_lower, color='red', linestyle='--', linewidth=2, label='95% CI')
        ax.axvline(ci_upper, color='red', linestyle='--', linewidth=2)

        # Median line
        median_val = np.median(values)
        ax.axvline(median_val, color='green', linestyle='-', linewidth=2, label='Median')

        # Format parameter name for display
        if param_name == 'Ea':
            ax.set_xlabel('Ea (J/mol)', fontsize=10, fontweight='bold')
        elif param_name == 'A':
            ax.set_xlabel('A (s⁻¹)', fontsize=10, fontweight='bold')
        else:
            ax.set_xlabel(param_name, fontsize=10, fontweight='bold')

        ax.set_ylabel('Frequency', fontsize=10, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle(f'Bootstrap Parameter Distributions (n={bootstrap_result.n_iterations})',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    return fig


def plot_multi_temperature_data(
    datasets: List[KineticDataset],
    figsize: Tuple[float, float] = (10, 6),
    time_units: str = 'seconds',
    ax: Optional[plt.Axes] = None
) -> Figure:
    """
    Plot experimental data at multiple temperatures.

    Parameters
    ----------
    datasets : List[KineticDataset]
        Experimental datasets
    figsize : Tuple[float, float], default=(10, 6)
        Figure size in inches
    time_units : str, default='seconds'
        Time units for x-axis
    ax : plt.Axes, optional
        Existing axes to plot on

    Returns
    -------
    Figure
        Matplotlib figure object

    Examples
    --------
    >>> fig = plot_multi_temperature_data(datasets, time_units='hours')
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    time_conversion = {
        'seconds': 1.0,
        'hours': 1 / 3600,
        'days': 1 / (24 * 3600),
        'months': 1 / (30.44 * 24 * 3600),
        'years': 1 / (365.25 * 24 * 3600)
    }
    time_factor = time_conversion.get(time_units, 1.0)

    temps_seen = set()
    for ds in datasets:
        temp_K = np.mean(ds.temperature)
        temp_C = temp_K - 273.15

        if temp_K not in temps_seen:
            time_plot = ds.time * time_factor
            conversion_pct = ds.conversion * 100

            ax.plot(
                time_plot,
                conversion_pct,
                'o-',
                label=f'{temp_C:.0f}°C',
                markersize=5,
                alpha=0.7
            )
            temps_seen.add(temp_K)

    ax.set_xlabel(f'Time ({time_units})', fontsize=12, fontweight='bold')
    ax.set_ylabel('Degradation (%)', fontsize=12, fontweight='bold')
    ax.set_title('Multi-Temperature Kinetic Data', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig
