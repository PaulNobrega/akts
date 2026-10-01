"""
HTML report generation for AKTS kinetic analysis results.

Supports both static (matplotlib) and interactive (Plotly) visualizations.
"""
import numpy as np
import base64
from io import BytesIO
from typing import List, Dict, Optional, Tuple, Union
from pathlib import Path
import warnings
import matplotlib.pyplot as plt

from .datatypes import KineticDataset, FitResult, BootstrapResult, PredictionResult
from .json_utils import convert_numpy_to_python
from .regulatory_plots import create_regulatory_shelf_life_plot, create_regulatory_comparison_plot
from .utils import seconds_per_time_unit as _seconds_per_time_unit, SECONDS_PER_MONTH
from .models import model_display_name, parse_sb2_model


def _embed_base64_image(fig) -> str:
    """
    Convert matplotlib figure to base64-encoded PNG string.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Matplotlib figure to convert

    Returns
    -------
    str
        Base64-encoded PNG image string
    """
    buffer = BytesIO()
    fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode()
    buffer.close()
    return f"data:image/png;base64,{image_base64}"


def _create_comparison_table_html(ranked_models: List[Dict]) -> str:
    """
    Generate HTML table comparing fitted models.

    Parameters
    ----------
    ranked_models : List[Dict]
        List of ranked model dictionaries with statistics

    Returns
    -------
    str
        HTML table string
    """
    html = '<table class="comparison-table">\n'
    html += '  <thead>\n'
    html += '    <tr>\n'
    html += '      <th>Rank</th><th>Model</th><th>R²</th><th>R²(adj)</th><th>AIC</th><th>ΔAIC</th><th>BIC</th>'
    html += '<th>RSS</th><th>RMSE</th><th>Params</th><th>Score</th><th>Akaike weight</th>\n'
    html += '    </tr>\n'
    html += '  </thead>\n'
    html += '  <tbody>\n'

    def format_number(value, spec, fallback='—'):
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            return fallback
        return format(number, spec) if np.isfinite(number) else fallback

    finite_aics = []
    for model in ranked_models:
        try:
            aic_value = float(model.get('stats', model.get('statistics', {})).get('aic'))
        except (TypeError, ValueError, OverflowError):
            continue
        if np.isfinite(aic_value):
            finite_aics.append(aic_value)
    min_aic = min(finite_aics) if finite_aics else 0.0

    for model in ranked_models:
        rank = model.get('rank', '?')
        name = model.get('model_name', 'Unknown')
        display_name = model_display_name(name)  # "F1_model" -> "F1 (first-order)"

        stats = model.get('stats', model.get('statistics', {}))  # Support both 'stats' and 'statistics'
        score = model.get('score')
        n_params = model.get('n_parameters', stats.get('n_params', '?'))

        aic = stats.get('aic')
        aic_number = None
        try:
            aic_number = float(aic)
        except (TypeError, ValueError, OverflowError):
            pass
        delta_aic = (aic_number - min_aic) if aic_number is not None and np.isfinite(aic_number) else None
        r2_adj = stats.get('r_squared_adj')
        r2_adj_str = format_number(r2_adj, '.4f')
        rmse = stats.get('rmse')
        rmse_str = format_number(rmse, '.4e')
        akaike_weight = stats.get('akaike_weight')

        # Highlight top 3 models
        row_class = 'top-model' if rank <= 3 else ''

        html += f'    <tr class="{row_class}">\n'
        html += f'      <td>{rank}</td>\n'
        html += f'      <td><strong>{display_name}</strong></td>\n'
        html += f'      <td>{format_number(stats.get("r_squared"), ".4f")}</td>\n'
        html += f'      <td>{r2_adj_str}</td>\n'
        html += f'      <td>{format_number(aic, ".2f")}</td>\n'
        html += f'      <td>{format_number(delta_aic, ".2f")}</td>\n'
        html += f'      <td>{format_number(stats.get("bic"), ".2f")}</td>\n'
        html += f'      <td>{format_number(stats.get("rss"), ".4e")}</td>\n'
        html += f'      <td>{rmse_str}</td>\n'
        html += f'      <td>{n_params}</td>\n'
        html += f'      <td>{format_number(score, ".4f")}</td>\n'
        html += f'      <td>{format_number(akaike_weight, ".1%")}</td>\n'
        html += '    </tr>\n'

    html += '  </tbody>\n'
    html += '</table>\n'
    return html


def _fit_ci_for_dataset(fit_result: FitResult, i: int,
                        attr: str = 'conversion_simulated_ci') -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """(lower, upper) band for dataset i: bootstrap CI by default, or attr='conversion_simulated_pi'."""
    bands = getattr(fit_result, attr, None)
    if not bands or i >= len(bands) or bands[i] is None:
        return None
    lower, upper = bands[i]
    if lower is None or upper is None:
        return None
    return np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)


TYPICAL_EA_RANGES = (
    ('Thermal denaturation &amp; unfolding (upper limit)', '400 – 800',
     'Cooperatively breaking a large network of weak non-covalent interactions (hydrogen bonds, '
     'hydrophobic effects) at once requires an exceptionally high barrier.'),
    ('Enzymatic / proteolytic cleavage', '20 – 100',
     'Proteases (e.g. proteasome, lysosomal enzymes) catalyze peptide-bond hydrolysis, lowering Ea '
     'to biological ranges.'),
    ('Spontaneous / pyrolytic hydrolysis', '90 – 140',
     'Uncatalyzed chemical or thermal degradation of amino acids or stable protein backbones.'),
)


def _typical_ea_note_html() -> str:
    rows = ''.join(f'<tr><td>{m}</td><td>{r}</td><td>{d}</td></tr>' for m, r, d in TYPICAL_EA_RANGES)
    return (
        '<p style="font-size: 0.9em; color: #666; margin-top: 10px;"><strong>Note: typical activation '
        'energy ranges.</strong> Use these to judge whether a fitted Ea is physically reasonable for the '
        'expected degradation mechanism. Fits are bounded to 5 – 1000 kJ/mol.</p>'
        '<table class="param-table"><thead><tr><th>Mechanism</th><th>Typical Ea (kJ/mol)</th>'
        f'<th>Description</th></tr></thead><tbody>{rows}</tbody></table>'
    )


def _rgba(color: str, alpha: float) -> str:
    """'rgb(r, g, b)' -> 'rgba(r, g, b, alpha)' for Plotly fill colors."""
    return color.replace('rgb(', 'rgba(').replace(')', f', {alpha})')


def _create_fit_plot_matplotlib(
    datasets: List[KineticDataset],
    fit_result: FitResult,
    title: str = "Model Fit"
) -> str:
    """Static fit plot: one column per temperature (own y-scale) with 95% CI and PI; residuals below."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        warnings.warn("matplotlib not available. Skipping static plots.")
        return ""

    n_cols = max(len(datasets), 1)
    fig, axes = plt.subplots(2, n_cols, figsize=(4.2 * n_cols, 6.5), height_ratios=[3, 1], squeeze=False)
    viridis_colors = ['#440154', '#3b528b', '#21918c', '#5ec962', '#fde725']
    colors = [viridis_colors[int(i * (len(viridis_colors) - 1) / max(len(datasets) - 1, 1))]
              for i in range(len(datasets))]
    simulated = getattr(fit_result, 'conversion_simulated', None)

    for i, dataset in enumerate(datasets):
        ax1, ax2 = axes[0][i], axes[1][i]
        temp_C = dataset.temperature.mean() - 273.15
        if isinstance(simulated, list) and i < len(simulated):
            pband = _fit_ci_for_dataset(fit_result, i, 'conversion_simulated_pi')
            if pband is not None:
                ax1.fill_between(dataset.time / 86400.0, pband[0], pband[1], color=colors[i], alpha=0.12,
                                 linewidth=0, label='95% PI')
            band = _fit_ci_for_dataset(fit_result, i)
            if band is not None:
                ax1.fill_between(dataset.time / 86400.0, band[0], band[1], color=colors[i], alpha=0.35,
                                 linewidth=0, label='95% CI')
            ax1.plot(dataset.time / 86400.0, simulated[i], color=colors[i], linewidth=2, label='Fit')
            ax2.scatter(dataset.time / 86400.0, dataset.conversion - np.asarray(simulated[i]),
                        color=colors[i], alpha=0.7, s=20)
        ax1.scatter(dataset.time / 86400.0, dataset.conversion, color=colors[i], alpha=0.8, s=40,
                    edgecolors='black', linewidths=0.5, label='Data', zorder=5)
        ax1.set_title(f'{temp_C:.0f}°C', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left', fontsize=8)
        ax2.axhline(y=0, color='k', linestyle='--', linewidth=1)
        ax2.set_xlabel('Time (days)', fontsize=11)
        ax2.grid(True, alpha=0.3)
    axes[0][0].set_ylabel('Conversion', fontsize=12)
    axes[1][0].set_ylabel('Residuals', fontsize=12)
    fig.suptitle(title, fontsize=14, fontweight='bold')
    fig.tight_layout()
    image_str = _embed_base64_image(fig)
    plt.close(fig)
    return f'<img src="{image_str}" alt="{title}" style="max-width: 100%;" />'


def _create_fit_plot_interactive(
    datasets: List[KineticDataset],
    fit_result: FitResult,
    title: str = "Model Fit"
) -> str:
    """
    Interactive fit plot: one column per temperature (own y-scale, so low-conversion
    series and their narrow bands stay visible), fit + 95% CI + 95% PI on top,
    residuals below.
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        warnings.warn("plotly not available. Falling back to static plots.")
        return _create_fit_plot_matplotlib(datasets, fit_result, title)

    n_cols = max(len(datasets), 1)
    fig = make_subplots(
        rows=2, cols=n_cols,
        row_heights=[0.7, 0.3],
        subplot_titles=[f'{ds.temperature.mean() - 273.15:.0f}°C' for ds in datasets] + [''] * n_cols,
        vertical_spacing=0.12, horizontal_spacing=0.06,
    )

    viridis_colors = ['rgb(68, 1, 84)', 'rgb(59, 82, 139)', 'rgb(33, 145, 140)',
                      'rgb(94, 201, 98)', 'rgb(253, 231, 37)']
    colors = [viridis_colors[int(i * (len(viridis_colors)-1) / max(len(datasets)-1, 1))]
              for i in range(len(datasets))]
    simulated = getattr(fit_result, 'conversion_simulated', None)

    for i, dataset in enumerate(datasets):
        col = i + 1
        temp_C = dataset.temperature.mean() - 273.15
        t_list = (dataset.time / 86400.0).tolist()

        if isinstance(simulated, list) and i < len(simulated):
            for attr, label, opacity in (('conversion_simulated_pi', 'PI', 0.12),
                                         ('conversion_simulated_ci', 'CI', 0.35)):
                band = _fit_ci_for_dataset(fit_result, i, attr)
                if band is None:
                    continue
                fig.add_trace(go.Scatter(
                    x=t_list + t_list[::-1], y=band[1].tolist() + band[0].tolist()[::-1],
                    fill='toself', fillcolor=_rgba(colors[i], opacity),
                    line=dict(color='rgba(255,255,255,0)'),
                    name=f'95% {label} @ {temp_C:.0f}°C', hoverinfo='skip',
                ), row=1, col=col)
            fig.add_trace(go.Scatter(
                x=t_list, y=np.asarray(simulated[i]).tolist(), mode='lines',
                name=f'Fit @ {temp_C:.0f}°C', line=dict(color=colors[i], width=2),
                hovertemplate='Time: %{x:.1f} d<br>Conversion: %{y:.4f}<extra></extra>',
            ), row=1, col=col)
            residuals = dataset.conversion - np.asarray(simulated[i])
            fig.add_trace(go.Scatter(
                x=t_list, y=residuals.tolist(), mode='markers',
                name=f'Residuals @ {temp_C:.0f}°C', marker=dict(color=colors[i], size=6, opacity=0.7),
                showlegend=False, hovertemplate='Time: %{x:.1f} d<br>Residual: %{y:.4f}<extra></extra>',
            ), row=2, col=col)
            fig.add_hline(y=0, line=dict(color='black', dash='dash', width=1), row=2, col=col)

        fig.add_trace(go.Scatter(
            x=t_list, y=dataset.conversion.tolist(), mode='markers',
            name=f'Data @ {temp_C:.0f}°C',
            marker=dict(color=colors[i], size=8, opacity=0.8, line=dict(color='black', width=0.5)),
            hovertemplate='Time: %{x:.1f} d<br>Conversion: %{y:.4f}<extra></extra>',
        ), row=1, col=col)
        fig.update_xaxes(title_text="Time (days)", row=2, col=col)

    fig.update_yaxes(title_text="Conversion", row=1, col=1)
    fig.update_yaxes(title_text="Residuals", row=2, col=1)
    fig.update_layout(
        title=dict(text=title), height=650, hovermode='closest', template='plotly_white',
        showlegend=True, legend=dict(orientation='h', x=0, y=-0.12, xanchor='left', yanchor='top'),
    )
    return fig.to_html(include_plotlyjs='cdn', div_id=f'plot_{title.replace(" ", "_")}')


def _create_prediction_plot_interactive(
    predictions: Dict,
    title: str = "Prediction/Extrapolation"
) -> str:
    """
    Create interactive prediction plot using Plotly.

    Parameters
    ----------
    predictions : Dict
        Dictionary with 'time', 'conversion_mean', and optionally 'conversion_lower', 'conversion_upper'
    title : str
        Plot title

    Returns
    -------
    str
        HTML div with Plotly plot
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        warnings.warn("Plotly is unavailable; using a static Matplotlib prediction plot.")
        return _create_prediction_plot_matplotlib(predictions, title)

    fig = go.Figure()

    time = predictions.get('time', [])
    requested_time = predictions.get('requested_time')
    time_unit = (requested_time.get('unit') if requested_time else
                 predictions.get('time_unit', 'seconds'))
    if requested_time:
        time = np.asarray(time, dtype=float) / _seconds_per_time_unit(time_unit)
    conversion = predictions.get('conversion_mean', [])
    lower = predictions.get('conversion_lower', None)
    upper = predictions.get('conversion_upper', None)

    # Confidence interval
    if lower is not None and upper is not None:
        fig.add_trace(go.Scatter(
            # Closed polygon: forward along the upper bound, back along the lower.
            # Build as lists -- `time` may be an ndarray, where `+` would add
            # element-wise instead of concatenating.
            x=list(time) + list(time)[::-1],
            y=list(upper) + list(lower)[::-1],
            fill='toself',
            fillcolor='rgba(0, 100, 200, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='95% CI',
            hoverinfo='skip'
        ))

    # Mean prediction
    fig.add_trace(go.Scatter(
        x=time,
        y=conversion,
        mode='lines',
        name='Predicted Conversion',
        line=dict(color='rgb(0, 100, 200)', width=3),
        hovertemplate='Time: %{x}<br>Conversion: %{y:.4f}<extra></extra>'
    ))

    # Add temperature to title if available
    temp = predictions.get('temperature')
    temp_units = predictions.get('temperature_units', 'K')
    if temp is not None:
        title_with_temp = f"{title} (at {temp:.1f}°{temp_units})"
    else:
        # Fallback to temperature_K for backward compatibility
        temp_K = predictions.get('temperature_K')
        if temp_K is not None:
            temp_C = temp_K - 273.15
            title_with_temp = f"{title} (at {temp_C:.1f}°C / {temp_K:.1f} K)"
        else:
            title_with_temp = title

    fig.update_xaxes(title_text=f"Time ({time_unit})")
    fig.update_yaxes(title_text="Conversion", range=[-0.05, 1.05])
    fig.update_layout(
        title=title_with_temp,
        height=500,
        hovermode='x unified',
        template='plotly_white'
    )

    return fig.to_html(include_plotlyjs='cdn', div_id='prediction_plot')


def _create_prediction_plot_matplotlib(
    predictions: Dict,
    title: str = "Prediction/Extrapolation"
) -> str:
    """Create a static prediction plot when Plotly is unavailable."""
    time = np.asarray(predictions.get('time', []), dtype=float)
    conversion = np.asarray(predictions.get('conversion_mean', []), dtype=float)
    if time.size == 0 or conversion.size == 0:
        return "<p>Prediction plot unavailable: no prediction data.</p>"

    requested_time = predictions.get('requested_time')
    time_unit = (requested_time.get('unit') if requested_time else
                 predictions.get('time_unit', 'seconds'))
    if requested_time:
        time = time / _seconds_per_time_unit(time_unit)
    fig, ax = plt.subplots(figsize=(10, 5))
    lower = predictions.get('conversion_lower')
    upper = predictions.get('conversion_upper')
    if lower is not None and upper is not None:
        ax.fill_between(time, lower, upper, color='#0064c8', alpha=0.2, label='95% CI')
    ax.plot(time, conversion, color='#0064c8', linewidth=3, label='Predicted Conversion')
    temp = predictions.get('temperature')
    temp_units = predictions.get('temperature_units', 'K')
    if temp is not None:
        title = f"{title} (at {temp:.1f}°{temp_units})"
    else:
        # Fallback to temperature_K for backward compatibility
        temp_K = predictions.get('temperature_K')
        if temp_K is not None:
            title = f"{title} (at {temp_K - 273.15:.1f}°C / {temp_K:.1f} K)"
    ax.set(title=title, xlabel=f"Time ({time_unit})", ylabel="Conversion", ylim=(-0.05, 1.05))
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    fig.tight_layout()
    image = _embed_base64_image(fig)
    plt.close(fig)
    return f'<img src="{image}" alt="{title}" style="max-width: 100%;" />'


def _create_simulation_plot_interactive(
    simulation: Dict,
    title: str = "Temperature Excursion Simulation"
) -> str:
    """
    Create interactive simulation plot showing conversion with temperature excursions.
    Uses vertical drop lines and annotations to indicate temperature changes.

    Parameters
    ----------
    simulation : Dict
        Dictionary with 'time', 'conversion_mean', 'temperature', and optionally
        'conversion_lower', 'conversion_upper', 'input_profile'
    title : str
        Plot title

    Returns
    -------
    str
        HTML div with Plotly plot (single y-axis: conversion)
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        warnings.warn("Plotly is unavailable; using a static Matplotlib simulation plot.")
        return _create_simulation_plot_matplotlib(simulation, title)

    # Create figure with single y-axis
    fig = go.Figure()

    time = simulation.get('time', [])
    conversion = simulation.get('conversion_mean', [])
    temperature = simulation.get('temperature', [])
    lower = simulation.get('conversion_lower', None)
    upper = simulation.get('conversion_upper', None)
    time_unit = simulation.get('time_unit', 'days')
    time = np.asarray(time, dtype=float) / _seconds_per_time_unit(time_unit)
    temp_units = simulation.get('temperature_units', 'C')  # Get temperature units
    input_profile = simulation.get('input_profile', [])

    # Identify temperature changes from input profile
    temp_changes = []
    annotations = []  # Collect annotations for layout

    if input_profile and len(input_profile) > 1:
        for i in range(len(input_profile)):
            t, temp = input_profile[i]
            temp_changes.append({'time': t, 'temp': temp})

    # Add vertical drop lines for temperature changes
    if temp_changes:
        for i, change in enumerate(temp_changes):
            # Vertical drop line
            fig.add_shape(
                type="line",
                x0=change['time'], x1=change['time'],
                y0=0, y1=1.05,
                line=dict(color="rgba(150, 150, 150, 0.6)", width=2, dash="dot"),
                layer='below'
            )

            # Annotate temperature in the region between drop lines
            if i < len(temp_changes) - 1:
                # Mid-point between this and next change
                mid_time = (change['time'] + temp_changes[i+1]['time']) / 2
                # Format temperature with correct units
                temp_value = change['temp']
                if temp_units.upper() == 'K':
                    temp_text = f"{temp_value:.0f}K ({temp_value-273.15:.0f}°C)"
                elif temp_units.upper() == 'C':
                    temp_text = f"{temp_value:.0f}°C"
                elif temp_units.upper() == 'F':
                    temp_text = f"{temp_value:.0f}°F"
                else:
                    temp_text = f"{temp_value:.1f}°{temp_units}"

                # Prepare annotation for layout
                annotations.append(dict(
                    x=mid_time,
                    y=1.02,
                    text=temp_text,
                    showarrow=False,
                    font=dict(size=11, color='rgb(100,100,100)'),
                    bgcolor='rgba(255,255,255,0.9)',
                    bordercolor='rgb(150,150,150)',
                    borderwidth=1,
                    borderpad=4,
                    xref='x',
                    yref='y'
                ))
            else:
                # Last segment - from this change to end of time
                mid_time = (change['time'] + time[-1]) / 2
                # Format temperature with correct units
                temp_value = change['temp']
                if temp_units.upper() == 'K':
                    temp_text = f"{temp_value:.0f}K ({temp_value-273.15:.0f}°C)"
                elif temp_units.upper() == 'C':
                    temp_text = f"{temp_value:.0f}°C"
                elif temp_units.upper() == 'F':
                    temp_text = f"{temp_value:.0f}°F"
                else:
                    temp_text = f"{temp_value:.1f}°{temp_units}"
                annotations.append(dict(
                    x=mid_time,
                    y=1.02,
                    text=temp_text,
                    showarrow=False,
                    font=dict(size=11, color='rgb(100,100,100)'),
                    bgcolor='rgba(255,255,255,0.9)',
                    bordercolor='rgb(150,150,150)',
                    borderwidth=1,
                    borderpad=4,
                    xref='x',
                    yref='y'
                ))

    # Confidence interval for conversion
    if lower is not None and upper is not None:
        fig.add_trace(go.Scatter(
            # Closed polygon: forward along the upper bound, back along the lower.
            # Build as lists -- `time` may be an ndarray, where `+` would add
            # element-wise instead of concatenating.
            x=list(time) + list(time)[::-1],
            y=list(upper) + list(lower)[::-1],
            fill='toself',
            fillcolor='rgba(31, 119, 180, 0.4)',  # High opacity for visibility
            line=dict(color='rgba(255,255,255,0)'),
            name='95% CI',
            hoverinfo='skip',
            showlegend=True
        ))

    # Conversion trace
    fig.add_trace(go.Scatter(
        x=time,
        y=conversion,
        mode='lines',
        name='Conversion (mean)',
        line=dict(color='rgb(31, 119, 180)', width=3),
        hovertemplate='Time: %{x:.2f} ' + time_unit + '<br>Conversion: %{y:.4f}<extra></extra>'
    ))

    # Set axis titles and layout
    fig.update_xaxes(title_text=f"Time ({time_unit})")
    fig.update_yaxes(title_text="Conversion", range=[-0.05, 1.1])  # Slightly higher to fit annotations

    # Add footer annotation about temperature markers
    annotations.append(dict(
        text="Temperature indicated by annotations between vertical markers",
        xref="paper", yref="paper",
        x=0.5, y=-0.15,
        showarrow=False,
        font=dict(size=10, color='gray')
    ))

    fig.update_layout(
        title=title,
        height=600,
        hovermode='x unified',
        template='plotly_white',
        legend=dict(x=0.01, y=0.99, xanchor='left', yanchor='top'),
        annotations=annotations  # Add all collected annotations
    )

    return fig.to_html(include_plotlyjs='cdn', div_id='simulation_plot')


def _create_simulation_plot_matplotlib(
    simulation: Dict,
    title: str = "Temperature Excursion Simulation"
) -> str:
    """Create a static conversion plot with temperature-excursion markers."""
    time_seconds = np.asarray(simulation.get('time', []), dtype=float)
    conversion = np.asarray(simulation.get('conversion_mean', []), dtype=float)
    if time_seconds.size == 0 or conversion.size == 0:
        return "<p>Simulation plot unavailable: no simulation data.</p>"

    time_unit = simulation.get('time_unit', 'days')
    time = time_seconds / _seconds_per_time_unit(time_unit)

    fig, conversion_ax = plt.subplots(figsize=(10, 5))
    lower = simulation.get('conversion_lower')
    upper = simulation.get('conversion_upper')
    if lower is not None and upper is not None:
        conversion_ax.fill_between(time, lower, upper, color='#1f77b4', alpha=0.4, label='95% CI')
    conversion_ax.plot(time, conversion, color='#1f77b4', linewidth=3, label='Conversion (mean)')
    conversion_ax.set_xlabel(f"Time ({time_unit})")
    conversion_ax.set_ylabel('Conversion')
    conversion_ax.set_ylim(-0.05, 1.1)
    conversion_ax.set_title(title)
    conversion_ax.grid(True, alpha=0.3)

    input_profile = simulation.get('input_profile', [])
    temp_units = simulation.get('temperature_units', 'C')
    if len(input_profile) > 1:
        conversion_ax.vlines(
            [profile_time for profile_time, _ in input_profile], 0, 1.05,
            color='0.6', linestyle=':', linewidth=2, alpha=0.6, zorder=0
        )
        for index, (profile_time, temp_value) in enumerate(input_profile):
            next_time = input_profile[index + 1][0] if index + 1 < len(input_profile) else time[-1]
            midpoint = (profile_time + next_time) / 2
            if temp_units.upper() == 'K':
                temp_text = f"{temp_value:.0f}K ({temp_value - 273.15:.0f}°C)"
            elif temp_units.upper() == 'C':
                temp_text = f"{temp_value:.0f}°C"
            elif temp_units.upper() == 'F':
                temp_text = f"{temp_value:.0f}°F"
            else:
                temp_text = f"{temp_value:.1f}°{temp_units}"
            conversion_ax.annotate(
                temp_text, xy=(midpoint, 1.02), xycoords='data',
                ha='center', va='bottom', fontsize=9, color='0.4',
                bbox={'facecolor': 'white', 'edgecolor': '0.6', 'alpha': 0.9, 'pad': 3}
            )

    conversion_ax.legend(loc='upper left')
    conversion_ax.text(
        0.5, -0.18, 'Temperature indicated by annotations between vertical markers',
        transform=conversion_ax.transAxes, ha='center', va='top', fontsize=9, color='gray'
    )
    fig.tight_layout()
    image = _embed_base64_image(fig)
    plt.close(fig)
    return f'<img src="{image}" alt="{title}" style="max-width: 100%;" />'


def _generate_html_template(
    title: str,
    summary_html: str,
    comparison_table_html: str,
    fit_plots_html: str,
    prediction_plot_html: str = "",
    simulation_plot_html: str = "",
    regulatory_html: str = "",
    methods_html: str = "",
    details_html: str = ""
) -> str:
    """
    Generate complete HTML document with embedded CSS.

    Parameters
    ----------
    title : str
        Report title
    summary_html : str
        Executive summary HTML
    comparison_table_html : str
        Model comparison table HTML
    fit_plots_html : str
        Fit plots HTML
    prediction_plot_html : str, optional
        Prediction plot HTML
    details_html : str, optional
        Statistical details HTML

    Returns
    -------
    str
        Complete HTML document
    """
    css = """
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .header h1 {
            margin: 0 0 10px 0;
            font-size: 2.5em;
        }
        .header p {
            margin: 0;
            opacity: 0.9;
        }
        .section {
            background: white;
            padding: 25px;
            margin-bottom: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        .section h2 {
            color: #667eea;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
            margin-top: 0;
        }
        .comparison-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }
        .comparison-table th {
            background-color: #667eea;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: 600;
        }
        .comparison-table td {
            padding: 10px 12px;
            border-bottom: 1px solid #e0e0e0;
        }
        .comparison-table tr:hover {
            background-color: #f9f9f9;
        }
        .comparison-table .top-model {
            background-color: #fff3e0;
        }
        .comparison-table .top-model:hover {
            background-color: #ffe0b2;
        }
        .summary-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }
        .summary-card {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 6px;
            border-left: 4px solid #667eea;
        }
        .summary-card h3 {
            margin: 0 0 5px 0;
            font-size: 0.9em;
            color: #666;
            text-transform: uppercase;
        }
        .summary-card p {
            margin: 0;
            font-size: 1.5em;
            font-weight: 600;
            color: #333;
        }
        .plot-container {
            margin: 20px 0;
        }
        .param-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }
        .param-table th, .param-table td {
            padding: 8px 12px;
            text-align: left;
            border-bottom: 1px solid #e0e0e0;
        }
        .param-table th {
            background-color: #f8f9fa;
            font-weight: 600;
        }
        .info-btn {
            background-color: #4CAF50;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.9em;
            margin-left: 10px;
        }
        .info-btn:hover {
            background-color: #45a049;
        }
        .regulatory-summary {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .shelf-life-card, .extrapolation-card {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #4CAF50;
        }
        .extrapolation-card.warning {
            border-left-color: #ff9800;
            background-color: #fff3e0;
        }
        .extrapolation-card.compliant {
            border-left-color: #4CAF50;
            background-color: #e8f5e9;
        }
        .shelf-life-value {
            font-size: 2.5em;
            font-weight: 700;
            color: #4CAF50;
            margin: 10px 0;
        }
        .shelf-life-detail {
            font-size: 0.9em;
            color: #666;
            margin: 5px 0;
        }
        .guideline-note {
            font-weight: 600;
            margin-top: 10px;
        }
        .modal {
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            overflow: auto;
            background-color: rgba(0,0,0,0.4);
        }
        .modal-content {
            background-color: #fefefe;
            margin: 10% auto;
            padding: 30px;
            border: 1px solid #888;
            border-radius: 8px;
            width: 80%;
            max-width: 600px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .modal-content h3 {
            color: #667eea;
            margin-top: 0;
        }
        .close {
            color: #aaa;
            float: right;
            font-size: 28px;
            font-weight: bold;
            cursor: pointer;
        }
        .close:hover,
        .close:focus {
            color: #000;
        }
        @media (max-width: 768px) {
            body {
                padding: 10px;
            }
            .header h1 {
                font-size: 1.8em;
            }
            .summary-grid {
                grid-template-columns: 1fr;
            }
            .regulatory-summary {
                grid-template-columns: 1fr;
            }
            .modal-content {
                width: 95%;
                margin: 20% auto;
            }
        }
    </style>
    """

    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    {css}
</head>
<body>
    <div class="header">
        <h1>{title}</h1>
        <p>Automated Kinetic Analysis Report</p>
    </div>

    <div class="section">
        <h2>Executive Summary</h2>
        {summary_html}
    </div>

    <div class="section">
        <h2>Model Comparison</h2>
        <p>All fitted models ranked by combined statistical score (BIC, R², RSS, parameter count):</p>
        {comparison_table_html}
    </div>

    <div class="section">
        <h2>Model Fit Visualization</h2>
        {fit_plots_html}
        <p style="font-size: 0.9em; color: #666;">
            <strong>Bands:</strong> the darker <strong>95% CI</strong> is the uncertainty of the fitted curve itself
            (bootstrap parameter uncertainty). The lighter <strong>95% PI</strong> (prediction interval) adds the residual
            scatter of individual measurements; about 95% of data points should fall inside it if the model is adequate.
            These bands are for visualization; the regulatory shelf-life uses its own one-sided bound.
        </p>
    </div>

    {f'<div class="section"><h2>Prediction/Extrapolation</h2>{prediction_plot_html}</div>' if prediction_plot_html else ''}

    {f'<div class="section"><h2>Temperature Excursion Simulation</h2><p>Simulated degradation under variable temperature conditions (e.g., shipping, storage with fluctuations)</p>{simulation_plot_html}</div>' if simulation_plot_html else ''}

    {regulatory_html if regulatory_html else ''}

    {f'<div class="section"><h2>Statistical Details</h2>{details_html}</div>' if details_html else ''}

    {methods_html if methods_html else ''}

    <div class="section" style="background-color: #f8f9fa; text-align: center;">
        <p style="margin: 0; color: #666;">
            Generated with AKTS Python Library
        </p>
    </div>
</body>
</html>
    """

    return html


def _create_methods_section_html(selected_model: Dict, datasets: List[KineticDataset],
                                 bootstrap_iterations: int = 0, confidence_level: float = 0.95,
                                 bootstrap_method: str = 'monte_carlo') -> str:
    """
    Generate Methods section for publication-ready documentation.

    Parameters
    ----------
    selected_model : Dict
        Selected model dictionary with name and parameters
    datasets : List[KineticDataset]
        Experimental datasets
    bootstrap_iterations : int
        Number of bootstrap iterations performed
    confidence_level : float
        Confidence level used (e.g., 0.95 for 95%)

    Returns
    -------
    str
        HTML section with detailed methods description
    """
    model_name = selected_model.get('model_name', 'Unknown')
    params = selected_model.get('parameters', {})

    def finite_number(value):
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        return number if np.isfinite(number) else None

    # Determine model type and equation
    model_equations = {
        'F1_model': 'f(α) = 1 - α (first-order)',
        'F2_model': 'f(α) = (1 - α)² (second-order)',
        'F3_model': 'f(α) = (1 - α)³ (third-order)',
        'A2_model': 'f(α) = 2(1 - α)[-ln(1 - α)]^(1/2) (Avrami-Erofeev, n=2)',
        'A3_model': 'f(α) = 3(1 - α)[-ln(1 - α)]^(2/3) (Avrami-Erofeev, n=3)',
        'R2_model': 'f(α) = 2(1 - α)^(1/2) (contracting area)',
        'R3_model': 'f(α) = 3(1 - α)^(2/3) (contracting volume)',
        'D2_model': 'f(α) = [-ln(1 - α)]^(-1) (2D diffusion)',
        'D3_model': 'f(α) = (3/2)(1 - α)^(2/3) / [1 - (1 - α)^(1/3)] (3D diffusion, Jander)',
        'SB_mn_model': 'f(α) = α^m · (1 - α)^n (Sestak-Berggren)',
    }

    base_model_name = model_name.replace('_model', '')
    is_sb2 = base_model_name == 'SB2' or parse_sb2_model(base_model_name) is not None or (
        {'Ea1', 'Ea2', 'm1', 'n1'} <= set(params))
    if base_model_name == 'Friedman':
        model_equation = 'No reaction model assumed; activation energy is estimated as a function of conversion, Ea(α)'
    elif base_model_name == 'A->B->C':
        model_equation = 'Sequential reactions A → B → C with separate kinetics for each step'
    elif is_sb2:
        model_equation = 'dα/dt = k₁(T)·α^m₁·(1 - α)^n₁ + k₂(T)·α^m₂·(1 - α)^n₂ (two parallel Sestak-Berggren steps)'
    else:
        model_equation = model_equations.get(model_name, 'Model-specific reaction model')

    # Build Arrhenius equation - ensure numeric types
    Ea = finite_number(params.get('Ea'))
    A = finite_number(params.get('A'))

    step_parameters = None
    if base_model_name == 'A->B->C' or is_sb2:
        step_parameters = {
            'Ea1': finite_number(params.get('Ea1')),
            'A1': finite_number(params.get('A1')),
            'Ea2': finite_number(params.get('Ea2')),
            'A2': finite_number(params.get('A2')),
        }

    if is_sb2 and step_parameters and all(value is not None for value in step_parameters.values()):
        shape = {k: finite_number(params.get(k)) for k in ('m1', 'n1', 'm2', 'n2')}
        fixed = parse_sb2_model(base_model_name)
        if fixed:
            shape = dict(zip(('m1', 'n1', 'm2', 'n2'), (float(v) for v in fixed)))
        def fmt(v):
            return 'n/a' if v is None else f'{v:.3f}'
        arrhenius_html = f'''
        <h3>Arrhenius Temperature Dependence by Reaction Step</h3>
        <p>Two parallel steps act on the same conversion; each has its own rate constant:</p>
        <ul>
            <li><strong>Step 1:</strong> k₁(T) = A₁ · exp(-Ea₁ / RT), Ea₁ = {step_parameters['Ea1']/1000:.1f} kJ/mol,
                ln(A₁·s) = {np.log(step_parameters['A1']):.3f}, m₁ = {fmt(shape['m1'])}, n₁ = {fmt(shape['n1'])}</li>
            <li><strong>Step 2:</strong> k₂(T) = A₂ · exp(-Ea₂ / RT), Ea₂ = {step_parameters['Ea2']/1000:.1f} kJ/mol,
                ln(A₂·s) = {np.log(step_parameters['A2']):.3f}, m₂ = {fmt(shape['m2'])}, n₂ = {fmt(shape['n2'])}</li>
        </ul>
        <p>R is the gas constant (8.314 J/(mol·K)); T is absolute temperature (K).</p>
        '''
    elif step_parameters and all(value is not None for value in step_parameters.values()):
        arrhenius_html = f'''
        <h3>Arrhenius Temperature Dependence by Reaction Step</h3>
        <p>Each step has its own temperature-dependent rate constant:</p>
        <ul>
            <li><strong>A → B:</strong> k₁(T) = A₁ · exp(-Ea₁ / RT), with Ea₁ = {step_parameters['Ea1']/1000:.1f} kJ/mol and A₁ = {step_parameters['A1']:.2e} s⁻¹</li>
            <li><strong>B → C:</strong> k₂(T) = A₂ · exp(-Ea₂ / RT), with Ea₂ = {step_parameters['Ea2']/1000:.1f} kJ/mol and A₂ = {step_parameters['A2']:.2e} s⁻¹</li>
        </ul>
        <p>R is the gas constant (8.314 J/(mol·K)); T is absolute temperature (K).</p>
        '''
    elif Ea is not None and A is not None:
        arrhenius_html = f'''
        <h3>Arrhenius Temperature Dependence</h3>
        <p>The rate constant follows the Arrhenius equation:</p>
        <div style="background-color: #f8f9fa; padding: 15px; border-left: 4px solid #667eea; margin: 15px 0; font-family: monospace;">
            k(T) = A · exp(-Ea / RT)
        </div>
        <p>where:</p>
        <ul>
            <li><strong>A</strong> = pre-exponential factor ({A:.2e} s⁻¹)</li>
            <li><strong>Ea</strong> = activation energy ({Ea/1000:.1f} kJ/mol)</li>
            <li><strong>R</strong> = gas constant (8.314 J/(mol·K))</li>
            <li><strong>T</strong> = absolute temperature (K)</li>
        </ul>
        '''
    elif base_model_name == 'A->B->C':
        arrhenius_html = '''
        <h3>Arrhenius Temperature Dependence by Reaction Step</h3>
        <p>Separate Arrhenius parameters are fitted for A → B and B → C.</p>
        '''
    else:
        arrhenius_html = '''
        <h3>Temperature Dependence</h3>
        <p>
            This model does not provide a single global Arrhenius parameter pair
            (Ea, A). Its temperature dependence is represented by model-specific
            estimates rather than one pre-exponential factor.
        </p>
        '''
    if (step_parameters and all(v is not None for v in step_parameters.values())) or (Ea is not None and A is not None):
        arrhenius_html += _typical_ea_note_html()

    if base_model_name == 'Friedman':
        kinetic_model_html = '''
        <p>
            At fixed conversion α, the differential rate has the Arrhenius form:
        </p>
        <div style="background-color: #f8f9fa; padding: 15px; border-left: 4px solid #667eea; margin: 15px 0; font-family: monospace;">
            (dα/dt)<sub>α,T</sub> = A f(α) · exp[−Ea(α)/(R T)]
        </div>
        <p>
            Friedman estimates Ea(α) separately at each conversion and does not
            specify f(α), so it does not impose a particular reaction mechanism.
        </p>
        '''
        fitting_method_html = '''
        <p>
            For each selected conversion α, the rate measured at each temperature
            is regressed against inverse absolute temperature using the linearized equation:
        </p>
        <div style="background-color: #f8f9fa; padding: 15px; border-left: 4px solid #667eea; margin: 15px 0; font-family: monospace;">
            ln[(dα/dt)<sub>α</sub>] = ln[A f(α)] − Ea(α)/(R T)
        </div>
        <div style="background-color: #f8f9fa; padding: 15px; border-left: 4px solid #667eea; margin: 15px 0; font-family: monospace;">
            m<sub>α</sub> = d ln[(dα/dt)<sub>α</sub>] / d(1/T) = −Ea(α)/R;
            Ea(α) = −R m<sub>α</sub>
        </div>
        <p>
            R is the gas constant (8.314 J/(mol·K)) and T is absolute temperature (K).
            The intercept ln[A f(α)] is not a standalone pre-exponential factor because
            f(α) is unspecified.
        </p>
        '''
    elif base_model_name == 'A->B->C':
        kinetic_model_html = '''
        <p>
            The consecutive-reaction model represents A → B followed by B → C.
            Each step has its own rate constant and reaction mechanism.
        </p>
        '''
        fitting_method_html = '''
        <p>
            The coupled reaction system was fitted across the available datasets.
            Candidate models were compared using the statistical criteria below.
        </p>
        '''
    else:
        kinetic_model_html = '''
        <p>
            The degradation kinetics were described using the selected model
            with the general differential rate equation:
        </p>
        <div style="background-color: #f8f9fa; padding: 15px; border-left: 4px solid #667eea; margin: 15px 0; font-family: monospace;">
            dα/dt = k(T) · f(α)
        </div>
        <p>where:</p>
        <ul>
            <li><strong>α</strong> = conversion (extent of degradation, 0 to 1)</li>
            <li><strong>t</strong> = time</li>
            <li><strong>k(T)</strong> = temperature-dependent rate constant</li>
            <li><strong>f(α)</strong> = reaction model: {model_equation}</li>
        </ul>
        '''
        fitting_method_html = '''
        <p>
            Kinetic parameters were estimated by fitting the selected model across
            the available datasets. Candidate models were compared using the
            statistical criteria below.
        </p>
        '''

    html = f'''
    <div class="section">
        <h2>Methods</h2>

        <h3>Kinetic Model</h3>
        <p>Selected method: <strong>{model_name.replace("_model", "")}</strong>.</p>
        {kinetic_model_html}

        {arrhenius_html}
        <h3>Experimental Data</h3>
        <p>
            Multi-temperature isothermal degradation data were collected at <strong>{len(datasets)}</strong> temperature(s):
        </p>
        <ul>
    '''

    for ds in datasets:
        temp_K = np.mean(ds.temperature)
        temp_C = temp_K - 273.15
        n_points = len(ds.time)
        duration = ds.time[-1] - ds.time[0]
        time_unit = 'seconds'
        if duration > 86400:
            duration /= 86400
            time_unit = 'days'
        elif duration > 3600:
            duration /= 3600
            time_unit = 'hours'
        html += f'            <li>{temp_C:.1f}°C ({temp_K:.1f} K): {n_points} data points over {duration:.1f} {time_unit}</li>\n'

    html += f'''
        </ul>

        <h3>Model Fitting and Selection</h3>
        {fitting_method_html}
        <p>Multiple candidate models were evaluated and ranked using:</p>
        <ul>
            <li><strong>Bayesian Information Criterion (BIC)</strong> to penalize model complexity</li>
            <li><strong>Akaike Information Criterion (AIC)</strong> for model comparison</li>
            <li><strong>R² and adjusted R²</strong> for goodness of fit</li>
            <li><strong>Akaike weights</strong> to quantify relative model probability</li>
        </ul>
        <p>
            The final model was selected based on a combined score balancing statistical fit quality
            and model simplicity (Occam's Razor principle).
        </p>
    '''

    if bootstrap_iterations > 0:
        base_model_name = model_name.replace('_model', '')
        if bootstrap_method == 'monte_carlo':
            method_label = 'Monte Carlo case bootstrap'
            sampling_description = '''
                Complete observed rows (time, temperature, and conversion) were sampled with replacement within each dataset, retaining its original row count.
                This can omit some observed time points and repeat others.
            '''
        elif bootstrap_method == 'parametric':
            method_label = 'parametric Gaussian bootstrap'
            sampling_description = '''
                Independent Gaussian conversion errors were simulated around the fitted curves using residual standard deviations estimated per dataset
                (with a pooled estimate when a dataset had too few finite residuals).
            '''
        else:
            method_label = 'weighted residual bootstrap'
            sampling_description = '''
                Centered conversion residuals were pooled across datasets and resampled with replacement using conversion-transition weights,
                then added to fitted curves; synthetic conversions were constrained to [0, 1].
            '''

        if base_model_name == 'Friedman':
            refit_description = 'The Friedman isoconversional regressions were rerun for each synthetic dataset.'
        else:
            refit_description = 'The selected model was re-fitted to each synthetic dataset.'

        html += f'''
        <h3>Uncertainty Quantification</h3>
        <p>
            Uncertainty was estimated for the selected model using <strong>{method_label}</strong>
            with {bootstrap_iterations} successful replicates:
        </p>
        <ol>
            <li>{sampling_description}</li>
            <li>{refit_description}</li>
        </ol>
        <p>
            The {confidence_level*100:.0f}% parameter confidence intervals use percentile bounds from
            successful bootstrap estimates. Pointwise conversion bands use percentiles of bootstrap
            prediction curves when those predictions are requested and computable.
        </p>
        '''

    html += '''
        <h3>Software</h3>
        <p>
            All kinetic analyses were performed using the AKTS Python library (open-source kinetic
            analysis toolkit). Numerical integration of ordinary differential equations (ODEs) was
            performed using SciPy's <code>solve_ivp</code> with adaptive step-size control (RK45 method
            with LSODA fallback for stiff systems).
        </p>

    </div>
    '''

    return html


def _create_regulatory_section_html(regulatory: Dict) -> str:
    """
    Generate a regression-based shelf-life and extrapolation summary.

    Parameters
    ----------
    regulatory : Dict
        Regulatory analysis results containing:
        - shelf_life_months: Mean shelf-life estimate
        - shelf_life_lower_95: One-sided 95% lower bound (ICH Q1E)
        - target_conversion: Failure criterion (e.g., 0.05 for 5% degradation)
        - storage_temp_K: Storage temperature in Kelvin
        - study_duration_months: Observed stability study duration
        - ich_ceiling_months: ICH Q1E extrapolation limit
        - exceeds_guideline: bool, whether estimate exceeds ceiling
        - prediction: PredictionResult (optional, for plotting)

    Returns
    -------
    str
        HTML section with regulatory analysis, plot, and info modal
    """
    # Calculate display values
    temp_c = regulatory['storage_temp_K'] - 273.15
    target_pct = regulatory['target_conversion'] * 100
    shelf_life_confidence = regulatory.get('shelf_life_confidence_level', 0.95)
    shelf_life_lower = regulatory.get('shelf_life_lower_confidence', regulatory['shelf_life_lower_95'])
    trend_type = regulatory.get('trend_type', 'linear')
    warning_class = 'warning' if regulatory['exceeds_guideline'] else 'compliant'
    if regulatory.get('shelf_life_is_long_term', True):
        ceiling_formula = 'min(2 × study duration, study duration + 12 months)'
    else:
        ceiling_formula = '1.5 × study duration'

    if regulatory['exceeds_guideline']:
        guideline_note = 'Estimate exceeds the configured extrapolation ceiling; further review is needed.'
    else:
        guideline_note = 'Estimate is within the configured extrapolation ceiling; this alone does not establish compliance.'

    # Generate regulatory plot if prediction data available
    regulatory_plot_html = ''
    if 'prediction' in regulatory and regulatory['prediction'] is not None:
        try:
            # Convert months back to seconds for plotting
            shelf_life_mean_sec = regulatory['shelf_life_months'] * SECONDS_PER_MONTH
            shelf_life_lower_sec = regulatory.get('shelf_life_lower_95', 0) * SECONDS_PER_MONTH if regulatory.get('shelf_life_lower_95') else None
            study_duration_sec = regulatory['study_duration_months'] * SECONDS_PER_MONTH
            ich_ceiling_sec = regulatory['ich_ceiling_months'] * SECONDS_PER_MONTH

            fig = create_regulatory_shelf_life_plot(
                prediction=regulatory['prediction'],
                target_conversion=regulatory['target_conversion'],
                shelf_life_mean_sec=shelf_life_mean_sec,
                shelf_life_lower_sec=shelf_life_lower_sec,
                study_duration_sec=study_duration_sec,
                ich_ceiling_sec=ich_ceiling_sec,
                storage_temp_K=regulatory['storage_temp_K'],
                confidence_band_level=regulatory.get('prediction_band_level', 0.95),
            )

            plot_base64 = _embed_base64_image(fig)
            plt.close(fig)

            regulatory_plot_html = f'''
            <div style="margin-top: 30px;">
                <h3>Shelf-Life Visualization</h3>
                <p style="font-size: 0.9em; color: #666;">
                    This plot shows the selected same-temperature {trend_type} trend, confidence band, and regulatory thresholds.
                    The shelf-life bound uses the one-sided {shelf_life_confidence:.0%} confidence limit in the adverse direction; no bootstrap is used for this regression calculation.
                </p>
                <img src="{plot_base64}" alt="Regulatory Shelf-Life Plot" style="max-width: 100%; height: auto;">
            </div>
            '''
        except Exception as e:
            warnings.warn(f"Failed to generate regulatory plot: {e}")
            regulatory_plot_html = ''
    else:
        regulatory_plot_html = ''

    html = f'''
    <div class="section">
        <h2>
            Shelf-Life Regression Analysis
            <button class="info-btn" onclick="showICHInfo()">ℹ️ Method Details</button>
        </h2>

        <div class="regulatory-summary">
            <div class="shelf-life-card">
                <h3>Shelf-Life Estimate</h3>
                <p class="shelf-life-value">{regulatory['shelf_life_months']:.1f} months</p>
                <p class="shelf-life-detail">
                    <strong>{shelf_life_confidence:.0%} Lower Shelf-Life Bound:</strong> {shelf_life_lower:.1f} months<br>
                    At {target_pct:.0f}% degradation threshold<br>
                    Storage temperature: {temp_c:.0f}°C<br>
                    Trend: {trend_type}<br>
                    Method: {regulatory.get('regression_method', 'linear regression')}
                </p>
                <p style="font-size: 0.85em; color: #666; margin-top: 10px;">
                    <em>{trend_type.title()} regression/ANCOVA at the storage condition; bootstrap resampling is not used for this estimate.</em>
                </p>
            </div>

            <div class="extrapolation-card {warning_class}">
                <h3>ICH Q1E Extrapolation Ceiling</h3>
                <p><strong>Study Duration:</strong> {regulatory['study_duration_months']:.0f} months</p>
                <p><strong>Maximum Allowed Extrapolation:</strong> {regulatory['ich_ceiling_months']:.0f} months</p>
                <p style="font-size: 0.85em; color: #666; margin: 5px 0;">
                    Formula: {ceiling_formula}
                </p>
                <p class="guideline-note" style="margin-top: 15px;">{guideline_note}</p>
            </div>
        </div>

        <p style="font-size: 0.9em; color: #666; margin-top: 20px;">
            <strong>Interpretation:</strong> Shelf life is the time when the one-sided {shelf_life_confidence:.0%} confidence limit for the mean
            in the adverse direction reaches the specification. The selected limit is upper for increasing degradation
            conversion and lower for decreasing attributes. If the estimate exceeds the ICH Q1E ceiling, additional
            long-term stability data should be collected to support the shelf-life claim. This automated estimate
            does not by itself establish regulatory compliance.
        </p>

        {regulatory_plot_html}
    </div>

    <div id="ich-info-modal" class="modal">
        <div class="modal-content">
            <span class="close" onclick="closeICHInfo()">&times;</span>
            <h3>Shelf-Life Regression Method</h3>

            <h4>Shelf-Life Determination</h4>
            <ul>
                <li>Use the <strong>one-sided {shelf_life_confidence:.0%} confidence limit in the adverse direction</strong> for shelf-life estimates</li>
                <li>Shelf-life is when that limit crosses the specification (e.g., an upper limit for increasing degradation)</li>
                <li>This approach is more conservative than using the mean prediction</li>
                <li>Provides adequate assurance that the product will remain within specifications</li>
            </ul>

            <h4>Extrapolation Limits</h4>
            <ul>
                <li><strong>Long-term data:</strong> Up to min(2 × study duration, study duration + 12 months)</li>
                <li>Example: 12-month study → maximum 24-month shelf-life claim</li>
                <li>Example: 18-month study → maximum 30-month shelf-life claim (min(36, 30))</li>
                <li>Example: 24-month study → maximum 36-month shelf-life claim</li>
                <li><strong>Accelerated data:</strong> Up to 1.5 × study duration (not shown here)</li>
            </ul>

            <h4>Scope</h4>
            <p>
                This automated regression and extrapolation summary is not a determination of ICH Q1E
                compliance. Confirm the selected trend, specification, study design, and applicable
                regulatory requirements before using a shelf-life estimate in a submission.
            </p>

            <p style="margin-top: 15px;">
                <strong>Reference:</strong> ICH Q1E: Evaluation of Stability Data (2003)<br>
                <a href="https://database.ich.org/sites/default/files/Q1E%20Guideline.pdf" target="_blank"
                   style="color: #667eea;">View ICH Q1E Guideline</a>
            </p>
        </div>
    </div>

    <script>
    function showICHInfo() {{
        document.getElementById('ich-info-modal').style.display = 'block';
    }}

    function closeICHInfo() {{
        document.getElementById('ich-info-modal').style.display = 'none';
    }}

    // Close modal when clicking outside of it
    window.onclick = function(event) {{
        var modal = document.getElementById('ich-info-modal');
        if (event.target == modal) {{
            modal.style.display = 'none';
        }}
    }}
    </script>
    '''

    return html


def generate_isothermal_report(
    datasets: List[KineticDataset],
    top_models: List[Dict],
    selected_model: Dict,
    predictions: Optional[Dict] = None,
    simulation: Optional[Dict] = None,
    report_path: Union[str, Path] = None,
    report_format: str = "interactive",
    summary: Optional[Dict] = None,
    fit_results: Optional[List[FitResult]] = None,
    regulatory: Optional[Dict] = None
) -> str:
    """
    Generate comprehensive HTML report for isothermal kinetic analysis.

    Parameters
    ----------
    datasets : List[KineticDataset]
        Experimental datasets
    top_models : List[Dict]
        Top-ranked models with statistics
    selected_model : Dict
        Selected model(s) after convergence check
    predictions : Dict, optional
        Prediction results
    report_path : str or Path, optional
        Path to save HTML file (if None, returns HTML string)
    report_format : str
        "interactive", "static", or "both"
    summary : Dict, optional
        Summary statistics
    regulatory : Dict, optional
        Regression-based shelf-life estimates, selected trend, confidence level,
        and extrapolation ceiling information

    Returns
    -------
    str
        Path to generated report or HTML string
    """
    # Generate summary HTML
    if summary:
        summary_cards = []
        for key, value in summary.items():
            label = key.replace('_', ' ').title()
            summary_cards.append(f'<div class="summary-card"><h3>{label}</h3><p>{value}</p></div>')
        summary_html = '<div class="summary-grid">' + ''.join(summary_cards) + '</div>'
    else:
        summary_html = "<p>No summary available</p>"

    # Generate comparison table
    comparison_table_html = _create_comparison_table_html(top_models)

    # Generate fit plots
    fit_plots_html = ""
    if top_models and datasets and fit_results:
        # Plot the top model using the FitResult object
        top_fit = fit_results[0]
        # Use the display name from top_models
        model_display = top_models[0].get('model_name', 'Top Model')
        friendly_name = model_display_name(model_display)

        fit_plots_html += f"<h3>{friendly_name}</h3>"

        try:
            if report_format == "interactive":
                fit_plots_html += _create_fit_plot_interactive(datasets, top_fit, f"{friendly_name} Fit")
            elif report_format == "static":
                fit_plots_html += _create_fit_plot_matplotlib(datasets, top_fit, f"{friendly_name} Fit")
            else:  # both
                fit_plots_html += "<h4>Interactive Plot:</h4>"
                fit_plots_html += _create_fit_plot_interactive(datasets, top_fit, f"{friendly_name} Fit")
                fit_plots_html += "<h4>Static Plot:</h4>"
                fit_plots_html += _create_fit_plot_matplotlib(datasets, top_fit, f"{friendly_name} Fit")
        except Exception as e:
            fit_plots_html += f"<p>Error generating plot: {e}</p>"
            warnings.warn(f"Plot generation failed: {e}")

    # Generate prediction plot
    prediction_plot_html = ""
    if predictions and report_format in ["interactive", "both"]:
        prediction_plot_html = _create_prediction_plot_interactive(predictions)

    # Generate simulation plot
    simulation_plot_html = ""
    if simulation and report_format in ["interactive", "both"]:
        simulation_plot_html = _create_simulation_plot_interactive(simulation)

    # Generate regulatory section (ICH Q1E)
    regulatory_html = ""
    if regulatory:
        regulatory_html = _create_regulatory_section_html(regulatory)

    # Generate methods section for publication
    methods_html = ""
    if selected_model and datasets:
        # Try to extract bootstrap info from summary or use defaults
        bootstrap_iters = 0
        if summary and 'bootstrap_iterations' in summary:
            bootstrap_iters = summary['bootstrap_iterations']
        bootstrap_confidence_level = (summary or {}).get('confidence_level', 0.95)
        methods_html = _create_methods_section_html(
            selected_model=selected_model,
            datasets=datasets,
            bootstrap_iterations=bootstrap_iters,
            confidence_level=bootstrap_confidence_level,
            bootstrap_method=(summary or {}).get('bootstrap_method', 'monte_carlo'),
        )

    # Generate details HTML
    details_html = ""
    if selected_model:
        details_html += "<h3>Selected Model Parameters</h3>"
        selected_model_name = selected_model.get('model_name', '').replace('_model', '')
        params = selected_model.get('parameters', {})
        stats = selected_model.get('statistics', {})

        def format_number(value, spec):
            try:
                number = float(value)
            except (TypeError, ValueError, OverflowError):
                return 'N/A'
            return format(number, spec) if np.isfinite(number) else 'N/A'

        if params:
            details_html += '<table class="param-table"><thead><tr><th>Parameter</th><th>Value</th><th>Units</th></tr></thead><tbody>'

            # Ea with better formatting
            if 'Ea' in params:
                Ea = format_number(params['Ea'], '.2f')
                if Ea != 'N/A':
                    Ea = format(float(Ea) / 1000, '.2f')
                details_html += f'<tr><td><strong>Ea</strong> (Activation Energy)</td><td>{Ea}</td><td>kJ/mol</td></tr>'

            # A (pre-exponential factor)
            if 'A' in params:
                A = format_number(params['A'], '.4e')
                details_html += f'<tr><td><strong>A</strong> (Pre-exponential Factor)</td><td>{A}</td><td>s⁻¹</td></tr>'

            step_labels = None
            if selected_model_name == 'A->B->C':
                step_labels = (('1', 'A → B'), ('2', 'B → C'))
            elif {'Ea1', 'Ea2'} <= set(params):
                step_labels = (('1', 'Step 1'), ('2', 'Step 2'))
            if step_labels:
                for step_name, description in step_labels:
                    ea_name, a_name = f'Ea{step_name}', f'A{step_name}'
                    if ea_name in params:
                        ea_value = format_number(params[ea_name], '.2f')
                        if ea_value != 'N/A':
                            ea_value = format(float(ea_value) / 1000, '.2f')
                        details_html += f'<tr><td><strong>{ea_name}</strong> ({description} activation energy)</td><td>{ea_value}</td><td>kJ/mol</td></tr>'
                    if a_name in params:
                        a_value = format_number(params[a_name], '.4e')
                        details_html += f'<tr><td><strong>{a_name}</strong> ({description} pre-exponential factor)</td><td>{a_value}</td><td>s⁻¹</td></tr>'

            # Other parameters
            for param, value in params.items():
                if param not in ['Ea', 'A', 'Ea1', 'A1', 'Ea2', 'A2']:
                    formatted_value = format_number(value, '.4e')
                    details_html += f'<tr><td>{param}</td><td>{formatted_value}</td><td>—</td></tr>'

            details_html += '</tbody></table>'
            if any(p.startswith('Ea') for p in params):
                details_html += _typical_ea_note_html()

        # Add goodness of fit statistics
        details_html += "<h3>Goodness of Fit</h3>"
        details_html += '<table class="param-table"><thead><tr><th>Statistic</th><th>Value</th></tr></thead><tbody>'

        for key, label, spec in (
            ('r_squared', 'R²', '.4f'),
            ('r_squared_adj', 'Adjusted R²', '.4f'),
            ('aic', 'AIC', '.2f'),
            ('bic', 'BIC', '.2f'),
            ('rss', 'RSS', '.4e'),
            ('rmse', 'RMSE', '.4e'),
        ):
            if key in stats:
                details_html += f'<tr><td>{label}</td><td>{format_number(stats[key], spec)}</td></tr>'
        if 'n_params' in stats:
            details_html += f'<tr><td>Parameters</td><td>{format_number(stats["n_params"], '.0f')}</td></tr>'
        if 'n_datapoints' in stats or 'n_points' in stats:
            n_pts = stats.get('n_datapoints', stats.get('n_points', '?'))
            details_html += f'<tr><td>Data Points</td><td>{n_pts}</td></tr>'
            if 'n_params' in stats:
                try:
                    dof = float(n_pts) - float(stats['n_params'])
                except (TypeError, ValueError, OverflowError):
                    dof = None
                if dof is not None and np.isfinite(dof):
                    details_html += f'<tr><td>Degrees of Freedom</td><td>{dof:g}</td></tr>'

        details_html += '</tbody></table>'

        # Include requested settings even when every requested replicate failed.
        bootstrap_requested = (summary or {}).get(
            'bootstrap_iterations_requested', (summary or {}).get('bootstrap_iterations', 0)
        )
        bootstrap_completed = (summary or {}).get('bootstrap_iterations', 0)
        if summary and bootstrap_requested > 0:
            method_names = {
                'monte_carlo': 'Monte Carlo case resampling',
                'parametric': 'Parametric Gaussian resampling',
                'residual': 'Weighted residual resampling',
            }
            method = summary.get('bootstrap_method')
            method_label = method_names.get(method, method or 'Not recorded')
            confidence_level = summary.get('confidence_level', 0.95)
            details_html += "<h3>Bootstrap Analysis</h3>"
            details_html += '<table class="param-table"><thead><tr><th>Item</th><th>Value</th></tr></thead><tbody>'
            details_html += f'<tr><td>Bootstrap Method</td><td>{method_label}</td></tr>'
            details_html += f'<tr><td>Iterations Requested</td><td>{bootstrap_requested}</td></tr>'
            details_html += f'<tr><td>Successful Iterations</td><td>{bootstrap_completed}</td></tr>'
            details_html += f'<tr><td>Confidence Level</td><td>{confidence_level:.0%}</td></tr>'

            # Note about degenerate sample filtering (our new feature!)
            details_html += '<tr><td>Quality Control</td><td>Bootstrap replicates with unrealistic predictions (&lt;1% final conversion) automatically excluded from CI calculation</td></tr>'

            details_html += '</tbody></table>'
            details_html += '<p style="font-size: 0.9em; color: #666; margin-top: 10px;">'
            details_html += '<em><strong>Note:</strong> Confidence intervals are calculated from bootstrap replicates. '
            details_html += 'Replicates that predict negligible degradation (typically 2-3 out of 100) are excluded as they represent '
            details_html += 'numerical artifacts rather than realistic parameter uncertainty.</em></p>'

    # Generate complete HTML
    html_content = _generate_html_template(
        title="AKTS Isothermal Analysis Report",
        summary_html=summary_html,
        comparison_table_html=comparison_table_html,
        fit_plots_html=fit_plots_html,
        prediction_plot_html=prediction_plot_html,
        simulation_plot_html=simulation_plot_html,
        regulatory_html=regulatory_html,
        methods_html=methods_html,
        details_html=details_html
    )

    # Save or return
    if report_path:
        report_path = Path(report_path)
        # Create parent directory if it doesn't exist
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        return str(report_path)
    else:
        return html_content
