"""Plot regression-based shelf-life estimates and configured reference limits."""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional, Tuple
from matplotlib.figure import Figure

from .datatypes import PredictionResult
from .utils import SECONDS_PER_MONTH


def create_regulatory_shelf_life_plot(
    prediction: PredictionResult,
    target_conversion: float,
    shelf_life_mean_sec: float,
    shelf_life_lower_sec: Optional[float] = None,
    study_duration_sec: Optional[float] = None,
    ich_ceiling_sec: Optional[float] = None,
    storage_temp_K: float = 298.15,
    figsize: Tuple[float, float] = (10, 6),
    confidence_band_level: float = 0.95,
) -> Figure:
    """
    Plot a regression-based shelf-life estimate and its confidence band.

    Shows model predictions with confidence intervals, target threshold,
    mean and conservative shelf-life estimates, and ICH Q1E extrapolation ceiling.

    Parameters
    ----------
    prediction : PredictionResult
        Prediction result with time, conversion, and optional CI
    target_conversion : float
        Target degradation threshold (e.g., 0.05 for 5%)
    shelf_life_mean_sec : float
        Mean shelf-life estimate in seconds
    shelf_life_lower_sec : float, optional
        Conservative shelf-life (one-sided 95% lower bound) in seconds
    study_duration_sec : float, optional
        Duration of experimental study in seconds
    ich_ceiling_sec : float, optional
        Configured shelf-life extrapolation ceiling in seconds
    storage_temp_K : float, default=298.15
        Storage temperature in Kelvin
    figsize : Tuple[float, float], default=(10, 6)
        Figure size in inches
    confidence_band_level : float, default=0.95
        Confidence level represented by prediction.conversion_ci.

    Returns
    -------
    Figure
        Matplotlib figure object

    Notes
    -----
    The plot visualizes the selected regression and configured limits. It does not
    establish regulatory compliance; validate the study design and model separately.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Convert time to months for display
    time_months = prediction.time / SECONDS_PER_MONTH
    conversion_pct = prediction.conversion * 100

    # Draw the confidence band and its boundaries before the mean so narrow bands remain legible.
    if prediction.conversion_ci is not None:
        ci_lower_pct = prediction.conversion_ci[0] * 100
        ci_upper_pct = prediction.conversion_ci[1] * 100
        ax.fill_between(
            time_months,
            ci_lower_pct,
            ci_upper_pct,
            alpha=0.3,
            color='#2c7fb8',
            label=f'{confidence_band_level*100:.0f}% Confidence Interval'
        )
        ax.plot(time_months, ci_lower_pct, color='#075985', linestyle='--', linewidth=1.0,
                alpha=0.9, label='_nolegend_')
        ax.plot(time_months, ci_upper_pct, color='#075985', linestyle='--', linewidth=1.0,
                alpha=0.9, label='_nolegend_')

    ax.plot(time_months, conversion_pct, color='#173f5f', linewidth=2.2, label='Regression Mean', zorder=4)

    # Target threshold line
    target_pct = target_conversion * 100
    ax.axhline(
        y=target_pct,
        color='red',
        linestyle='--',
        linewidth=2,
        label=f'Specification Limit ({target_pct:.0f}%)'
    )

    # Mean shelf-life estimate
    shelf_life_mean_months = shelf_life_mean_sec / SECONDS_PER_MONTH
    ax.axvline(
        x=shelf_life_mean_months,
        color='green',
        linestyle='-',
        linewidth=2,
        alpha=0.7,
        label=f'Mean Shelf-Life ({shelf_life_mean_months:.1f} months)'
    )

    # Conservative shelf-life (one-sided CI)
    if shelf_life_lower_sec is not None:
        shelf_life_lower_months = shelf_life_lower_sec / SECONDS_PER_MONTH
        ax.axvline(
            x=shelf_life_lower_months,
            color='darkgreen',
            linestyle='--',
            linewidth=2,
            alpha=0.9,
            label=f'Conservative Shelf-Life (95% Lower Bound: {shelf_life_lower_months:.1f} months)'
        )

    # Study duration marker
    if study_duration_sec is not None:
        study_duration_months = study_duration_sec / SECONDS_PER_MONTH
        ax.axvline(
            x=study_duration_months,
            color='gray',
            linestyle=':',
            linewidth=2,
            alpha=0.6,
            label=f'Study Duration ({study_duration_months:.0f} months)'
        )

    # ICH Q1E extrapolation ceiling
    if ich_ceiling_sec is not None:
        ich_ceiling_months = ich_ceiling_sec / SECONDS_PER_MONTH
        ax.axvline(
            x=ich_ceiling_months,
            color='orange',
            linestyle=':',
            linewidth=2,
            alpha=0.8,
            label=f'Extrapolation Ceiling ({ich_ceiling_months:.0f} months)'
        )

        # Add warning zone if shelf-life exceeds ceiling
        if shelf_life_lower_sec and shelf_life_lower_sec > ich_ceiling_sec:
            ax.axvspan(
                ich_ceiling_months,
                max(time_months),
                alpha=0.1,
                color='orange',
                label='Exceeds configured ceiling'
            )

    # Labels and formatting
    storage_temp_C = storage_temp_K - 273.15
    ax.set_xlabel('Time (months)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Degradation (%)', fontsize=12, fontweight='bold')
    ax.set_title(
        f'Shelf-Life Trend Analysis\nStorage Temperature: {storage_temp_C:.0f}°C',
        fontsize=14,
        fontweight='bold'
    )

    # Legend
    ax.legend(loc='best', fontsize=9, framealpha=0.9)

    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')

    # Set reasonable y-axis limits based on actual data
    max_conversion = conversion_pct.max()
    if prediction.conversion_ci is not None:
        max_conversion = max(max_conversion, ci_upper_pct.max())

    # Upper limit: larger of (data max + 10%, target × 3, 15%)
    y_upper = max(max_conversion * 1.1, target_pct * 3, 15)
    ax.set_ylim(0, y_upper)

    # Set x-axis limits to show full range
    ax.set_xlim(0, max(time_months))

    plt.tight_layout()

    return fig


def create_regulatory_comparison_plot(
    datasets: list,
    prediction: PredictionResult,
    storage_temp_K: float = 298.15,
    figsize: Tuple[float, float] = (10, 6)
) -> Figure:
    """
    Create regulatory plot comparing observed data to model predictions.

    Shows experimental data points at different temperatures alongside
    the extrapolated prediction at storage temperature.

    Parameters
    ----------
    datasets : list
        List of KineticDataset objects (experimental data)
    prediction : PredictionResult
        Model prediction at storage temperature
    storage_temp_K : float, default=298.15
        Storage temperature in Kelvin
    figsize : Tuple[float, float], default=(10, 6)
        Figure size in inches

    Returns
    -------
    Figure
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Plot experimental data by temperature
    temps_seen = set()
    for ds in datasets:
        temp_K = np.mean(ds.temperature)
        temp_C = temp_K - 273.15

        if temp_K not in temps_seen:
            time_months = ds.time / SECONDS_PER_MONTH
            conversion_pct = ds.conversion * 100

            ax.scatter(
                time_months,
                conversion_pct,
                label=f'Data at {temp_C:.0f}°C',
                s=60,
                alpha=0.7,
                edgecolors='black',
                linewidths=0.5
            )
            temps_seen.add(temp_K)

    # Plot prediction at storage temperature
    time_months = prediction.time / SECONDS_PER_MONTH
    conversion_pct = prediction.conversion * 100

    storage_temp_C = storage_temp_K - 273.15
    ax.plot(
        time_months,
        conversion_pct,
        'b-',
        linewidth=2,
        label=f'Prediction at {storage_temp_C:.0f}°C'
    )

    # Plot confidence interval if available
    if prediction.conversion_ci is not None:
        ci_lower_pct = prediction.conversion_ci[0] * 100
        ci_upper_pct = prediction.conversion_ci[1] * 100
        ax.fill_between(
            time_months,
            ci_lower_pct,
            ci_upper_pct,
            alpha=0.2,
            color='blue',
            label='95% CI'
        )

    # Labels
    ax.set_xlabel('Time (months)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Degradation (%)', fontsize=12, fontweight='bold')
    ax.set_title(
        'Model Verification: Observed Data vs. Predictions',
        fontsize=14,
        fontweight='bold'
    )

    # Legend
    ax.legend(loc='best', fontsize=9, framealpha=0.9)

    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()

    return fig
