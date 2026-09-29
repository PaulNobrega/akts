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


_SECONDS_PER_TIME_UNIT = {
    'second': 1, 'seconds': 1, 's': 1,
    'minute': 60, 'minutes': 60, 'min': 60,
    'hour': 3600, 'hours': 3600, 'h': 3600, 'hr': 3600,
    'day': 86400, 'days': 86400, 'd': 86400,
    'week': 604800, 'weeks': 604800,
    'month': 2592000, 'months': 2592000,
    'year': 31536000, 'years': 31536000, 'yr': 31536000,
}


def _seconds_per_time_unit(unit: str) -> float:
    return _SECONDS_PER_TIME_UNIT.get(unit.lower().strip(), 1)


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

    # Model display names for prettier output
    MODEL_NAMES = {
        'F1': 'F1 (first-order)', 'F2': 'F2 (second-order)', 'F3': 'F3 (third-order)',
        'A2': 'A2 (Avrami-Erofeev, n=2)', 'A3': 'A3 (Avrami-Erofeev, n=3)',
        'R2': 'R2 (contracting area)', 'R3': 'R3 (contracting volume)',
        'D2': 'D2 (2D diffusion)', 'D3': 'D3 (3D diffusion, Jander)',
        'D4': 'D4 (3D diffusion, G-B)', 'D1': 'D1 (1D diffusion)'
    }

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
        # Convert "F1_model" to "F1 (first-order)"
        base_name = name.replace('_model', '')
        display_name = MODEL_NAMES.get(base_name, name)

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


def _create_fit_plot_matplotlib(
    datasets: List[KineticDataset],
    fit_result: FitResult,
    title: str = "Model Fit"
) -> str:
    """
    Create static fit plot using matplotlib, return as base64.

    Parameters
    ----------
    datasets : List[KineticDataset]
        Experimental datasets
    fit_result : FitResult
        Fitted model result
    title : str
        Plot title

    Returns
    -------
    str
        Base64-encoded PNG image string
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        warnings.warn("matplotlib not available. Skipping static plots.")
        return ""

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), height_ratios=[3, 1])

    # Plot experimental data and fits
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(datasets)))

    for i, dataset in enumerate(datasets):
        temp_K = dataset.temperature.mean()
        temp_C = temp_K - 273.15

        # Experimental data
        ax1.scatter(dataset.time, dataset.conversion,
                   label=f'Data @ {temp_C:.0f}°C',
                   color=colors[i], alpha=0.7, s=40)

        # Fitted curve (if available)
        if hasattr(fit_result, 'conversion_simulated') and fit_result.conversion_simulated is not None:
            if isinstance(fit_result.conversion_simulated, list):
                if i < len(fit_result.conversion_simulated):
                    ax1.plot(dataset.time, fit_result.conversion_simulated[i],
                            label=f'Fit @ {temp_C:.0f}°C',
                            color=colors[i], linewidth=2)

                    # Residuals
                    residuals = dataset.conversion - fit_result.conversion_simulated[i]
                    ax2.scatter(dataset.time, residuals, color=colors[i], alpha=0.7, s=20)
            else:
                # Single dataset case
                ax1.plot(dataset.time, fit_result.conversion_simulated,
                        label='Fit', color=colors[i], linewidth=2)
                residuals = dataset.conversion - fit_result.conversion_simulated
                ax2.scatter(dataset.time, residuals, color=colors[i], alpha=0.7, s=20)

    ax1.set_xlabel('Time', fontsize=12)
    ax1.set_ylabel('Conversion', fontsize=12)
    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.05, 1.05)

    # Residuals plot
    ax2.axhline(y=0, color='k', linestyle='--', linewidth=1)
    ax2.set_xlabel('Time', fontsize=12)
    ax2.set_ylabel('Residuals', fontsize=12)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    image_str = _embed_base64_image(fig)
    plt.close(fig)

    return f'<img src="{image_str}" alt="{title}" style="max-width: 100%;" />'


def _create_fit_plot_interactive(
    datasets: List[KineticDataset],
    fit_result: FitResult,
    title: str = "Model Fit"
) -> str:
    """
    Create interactive fit plot using Plotly, return as HTML div.

    Parameters
    ----------
    datasets : List[KineticDataset]
        Experimental datasets
    fit_result : FitResult
        Fitted model result
    title : str
        Plot title

    Returns
    -------
    str
        HTML div with Plotly plot
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        warnings.warn("plotly not available. Falling back to static plots.")
        return _create_fit_plot_matplotlib(datasets, fit_result, title)

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        subplot_titles=(title, "Residuals"),
        vertical_spacing=0.12
    )

    # Generate colors without matplotlib dependency
    # Simple viridis-like color scheme
    viridis_colors = ['rgb(68, 1, 84)', 'rgb(59, 82, 139)', 'rgb(33, 145, 140)',
                      'rgb(94, 201, 98)', 'rgb(253, 231, 37)']
    n_datasets = len(datasets)
    colors = [viridis_colors[int(i * (len(viridis_colors)-1) / max(n_datasets-1, 1))]
              for i in range(n_datasets)]

    for i, dataset in enumerate(datasets):
        temp_K = dataset.temperature.mean()
        temp_C = temp_K - 273.15

        # Experimental data
        fig.add_trace(
            go.Scatter(
                x=dataset.time.tolist(),
                y=dataset.conversion.tolist(),
                mode='markers',
                name=f'Data @ {temp_C:.0f}°C',
                marker=dict(color=colors[i], size=8, opacity=0.7),
                hovertemplate='Time: %{x}<br>Conversion: %{y:.4f}<extra></extra>'
            ),
            row=1, col=1
        )

        # Fitted curve
        if hasattr(fit_result, 'conversion_simulated') and fit_result.conversion_simulated is not None:
            if isinstance(fit_result.conversion_simulated, list):
                if i < len(fit_result.conversion_simulated):
                    fig.add_trace(
                        go.Scatter(
                            x=dataset.time.tolist(),
                            y=fit_result.conversion_simulated[i].tolist(),
                            mode='lines',
                            name=f'Fit @ {temp_C:.0f}°C',
                            line=dict(color=colors[i], width=2),
                            hovertemplate='Time: %{x}<br>Conversion: %{y:.4f}<extra></extra>'
                        ),
                        row=1, col=1
                    )

                    # Residuals
                    residuals = dataset.conversion - fit_result.conversion_simulated[i]
                    fig.add_trace(
                        go.Scatter(
                            x=dataset.time.tolist(),
                            y=residuals.tolist(),
                            mode='markers',
                            name=f'Residuals @ {temp_C:.0f}°C',
                            marker=dict(color=colors[i], size=6, opacity=0.7),
                            showlegend=False,
                            hovertemplate='Time: %{x}<br>Residual: %{y:.4f}<extra></extra>'
                        ),
                        row=2, col=1
                    )

    # Add zero line to residuals spanning full x-axis
    if datasets:
        # Get full time range across ALL datasets
        all_times = np.concatenate([ds.time for ds in datasets])
        t_min, t_max = all_times.min(), all_times.max()
        fig.add_trace(
            go.Scatter(
                x=[t_min, t_max],
                y=[0, 0],
                mode='lines',
                line=dict(color='black', dash='dash', width=1),
                showlegend=False,
                hoverinfo='skip'
            ),
            row=2, col=1
        )

    fig.update_xaxes(title_text="Time", row=1, col=1)
    fig.update_xaxes(title_text="Time", row=2, col=1)
    fig.update_yaxes(title_text="Conversion", row=1, col=1, range=[-0.05, 1.05])
    fig.update_yaxes(title_text="Residuals", row=2, col=1)

    fig.update_layout(
        height=700,
        hovermode='closest',
        template='plotly_white',
        showlegend=True,
        legend=dict(x=1.05, y=1, xanchor='left', yanchor='top')
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
            x=time + time[::-1],
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
        ax.fill_between(time, lower, upper, color='#1f77b4', alpha=0.2, label='Confidence interval')
    ax.plot(time, conversion, color='#1f77b4', linewidth=2, label='Predicted conversion')
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
            x=time + time[::-1],
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
    """Create a static conversion and temperature plot without Plotly."""
    time_seconds = np.asarray(simulation.get('time', []), dtype=float)
    conversion = np.asarray(simulation.get('conversion_mean', []), dtype=float)
    temperature = np.asarray(simulation.get('temperature', []), dtype=float)
    if time_seconds.size == 0 or conversion.size == 0:
        return "<p>Simulation plot unavailable: no simulation data.</p>"

    time_unit = simulation.get('time_unit', 'seconds')
    time = time_seconds / _seconds_per_time_unit(time_unit)

    fig, conversion_ax = plt.subplots(figsize=(10, 5))
    conversion_ax.plot(time, conversion, color='#1f77b4', linewidth=2, label='Conversion')
    lower = simulation.get('conversion_lower')
    upper = simulation.get('conversion_upper')
    if lower is not None and upper is not None:
        conversion_ax.fill_between(time, lower, upper, color='#1f77b4', alpha=0.2, label='Confidence interval')
    conversion_ax.set_xlabel(f"Time ({time_unit})")
    conversion_ax.set_ylabel('Conversion')
    conversion_ax.set_ylim(-0.05, 1.05)
    conversion_ax.set_title(title)
    conversion_ax.grid(True, alpha=0.3)

    temperature_ax = conversion_ax.twinx()
    if temperature.size == time.size:
        temperature_ax.plot(time, temperature, color='#d97706', linestyle='--', label='Temperature')
    temperature_ax.set_ylabel(f"Temperature ({simulation.get('temperature_units', 'K')})")

    input_profile = simulation.get('input_profile', [])
    if input_profile:
        profile_times, profile_temps = zip(*input_profile)
        temperature_ax.scatter(profile_times, profile_temps, color='#d97706', marker='o', zorder=3, label='Profile points')

    handles_1, labels_1 = conversion_ax.get_legend_handles_labels()
    handles_2, labels_2 = temperature_ax.get_legend_handles_labels()
    conversion_ax.legend(handles_1 + handles_2, labels_1 + labels_2, loc='best')
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
                                 bootstrap_iterations: int = 0, confidence_level: float = 0.95) -> str:
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
    if base_model_name == 'Friedman':
        model_equation = 'No reaction model assumed; activation energy is estimated as a function of conversion, Ea(α)'
    elif base_model_name == 'A->B->C':
        model_equation = 'Sequential reactions A → B → C with separate kinetics for each step'
    else:
        model_equation = model_equations.get(model_name, 'Model-specific reaction model')

    # Build Arrhenius equation - ensure numeric types
    Ea = finite_number(params.get('Ea'))
    A = finite_number(params.get('A'))

    step_parameters = None
    if base_model_name == 'A->B->C':
        step_parameters = {
            'Ea1': finite_number(params.get('Ea1')),
            'A1': finite_number(params.get('A1')),
            'Ea2': finite_number(params.get('Ea2')),
            'A2': finite_number(params.get('A2')),
        }

    if step_parameters and all(value is not None for value in step_parameters.values()):
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

    if base_model_name == 'Friedman':
        kinetic_model_html = f'''
        <p>
            Friedman isoconversional analysis estimates activation energy at
            conversion levels without assuming a specific reaction model.
        </p>
        '''
        fitting_method_html = '''
        <p>
            Activation energy was estimated at conversion levels using linear
            regression of log reaction rate against inverse absolute temperature.
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
        html += f'''
        <h3>Uncertainty Quantification</h3>
        <p>
            Parameter uncertainties and prediction confidence intervals were estimated using
            <strong>parametric bootstrap resampling</strong> with {bootstrap_iterations} iterations.
            For each bootstrap replicate:
        </p>
        <ol>
            <li>Residuals from the best-fit model were calculated</li>
            <li>Synthetic datasets were generated by resampling residuals with replacement</li>
            <li>The model was re-fitted to the synthetic data</li>
            <li>Parameter estimates and predictions were recorded</li>
        </ol>
        <p>
            The {confidence_level*100:.0f}% confidence intervals were computed from the distribution
            of bootstrap parameter estimates using the percentile method.
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

        <h3>Data Availability</h3>
        <p>
            The kinetic parameters and fitted model are provided in the Statistical Details section.
            Raw experimental data and complete analysis scripts are available upon request.
        </p>
    </div>
    '''

    return html


def _create_regulatory_section_html(regulatory: Dict) -> str:
    """
    Generate ICH Q1E regulatory compliance section HTML.

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
    warning_class = 'warning' if regulatory['exceeds_guideline'] else 'compliant'

    if regulatory['exceeds_guideline']:
        guideline_note = '⚠️ Estimate exceeds ICH Q1E ceiling - additional stability data needed for regulatory submission'
    else:
        guideline_note = '✓ Within ICH Q1E guidelines - suitable for regulatory shelf-life claim'

    # Generate regulatory plot if prediction data available
    regulatory_plot_html = ''
    if 'prediction' in regulatory and regulatory['prediction'] is not None:
        try:
            # Convert months back to seconds for plotting
            shelf_life_mean_sec = regulatory['shelf_life_months'] * 30.44 * 24 * 3600
            shelf_life_lower_sec = regulatory.get('shelf_life_lower_95', 0) * 30.44 * 24 * 3600 if regulatory.get('shelf_life_lower_95') else None
            study_duration_sec = regulatory['study_duration_months'] * 30.44 * 24 * 3600
            ich_ceiling_sec = regulatory['ich_ceiling_months'] * 30.44 * 24 * 3600

            fig = create_regulatory_shelf_life_plot(
                prediction=regulatory['prediction'],
                target_conversion=regulatory['target_conversion'],
                shelf_life_mean_sec=shelf_life_mean_sec,
                shelf_life_lower_sec=shelf_life_lower_sec,
                study_duration_sec=study_duration_sec,
                ich_ceiling_sec=ich_ceiling_sec,
                storage_temp_K=regulatory['storage_temp_K']
            )

            plot_base64 = _embed_base64_image(fig)
            plt.close(fig)

            regulatory_plot_html = f'''
            <div style="margin-top: 30px;">
                <h3>Shelf-Life Visualization</h3>
                <p style="font-size: 0.9em; color: #666;">
                    This plot shows the model prediction, confidence intervals, and regulatory thresholds.
                    The conservative shelf-life (one-sided 95% lower bound) is the recommended value for regulatory claims.
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
            ICH Q1E Regulatory Analysis
            <button class="info-btn" onclick="showICHInfo()">ℹ️ ICH Q1E Guidelines</button>
        </h2>

        <div class="regulatory-summary">
            <div class="shelf-life-card">
                <h3>Shelf-Life Estimate</h3>
                <p class="shelf-life-value">{regulatory['shelf_life_months']:.1f} months</p>
                <p class="shelf-life-detail">
                    <strong>95% Lower Bound (ICH Q1E):</strong> {regulatory['shelf_life_lower_95']:.1f} months<br>
                    At {target_pct:.0f}% degradation threshold<br>
                    Storage temperature: {temp_c:.0f}°C
                </p>
                <p style="font-size: 0.85em; color: #666; margin-top: 10px;">
                    <em>Note: ICH Q1E requires one-sided 95% lower confidence bound for conservative shelf-life claims.</em>
                </p>
            </div>

            <div class="extrapolation-card {warning_class}">
                <h3>ICH Q1E Extrapolation Ceiling</h3>
                <p><strong>Study Duration:</strong> {regulatory['study_duration_months']:.0f} months</p>
                <p><strong>Maximum Allowed Extrapolation:</strong> {regulatory['ich_ceiling_months']:.0f} months</p>
                <p style="font-size: 0.85em; color: #666; margin: 5px 0;">
                    Formula: min(2 × study duration, study duration + 12 months)
                </p>
                <p class="guideline-note" style="margin-top: 15px;">{guideline_note}</p>
            </div>
        </div>

        <p style="font-size: 0.9em; color: #666; margin-top: 20px;">
            <strong>Interpretation:</strong> The one-sided 95% lower bound provides a conservative shelf-life estimate
            suitable for regulatory submissions. If the estimate exceeds the ICH Q1E ceiling, additional long-term
            stability data should be collected to support the shelf-life claim.
        </p>

        {regulatory_plot_html}
    </div>

    <div id="ich-info-modal" class="modal">
        <div class="modal-content">
            <span class="close" onclick="closeICHInfo()">&times;</span>
            <h3>ICH Q1E Stability Testing Guidelines</h3>

            <h4>Shelf-Life Determination</h4>
            <ul>
                <li>Use <strong>one-sided 95% confidence interval</strong> (lower bound) for shelf-life estimates</li>
                <li>Shelf-life is when the 95% lower bound crosses the specification limit (e.g., 5% degradation)</li>
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

            <h4>Regulatory Context</h4>
            <p>
                The ICH Q1E guideline provides a framework for evaluating and extrapolating stability data
                for drug substances and products. The extrapolation limits ensure that shelf-life claims
                are supported by adequate stability data and modeling, reducing the risk of product failure
                in the field.
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
        ICH Q1E regulatory compliance analysis containing shelf-life estimates
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
        # Convert to friendly name
        base_name = model_display.replace('_model', '')
        MODEL_NAMES = {
            'F1': 'F1 (first-order)', 'F2': 'F2 (second-order)', 'F3': 'F3 (third-order)',
            'A2': 'A2 (Avrami-Erofeev, n=2)', 'A3': 'A3 (Avrami-Erofeev, n=3)',
            'R2': 'R2 (contracting area)', 'R3': 'R3 (contracting volume)',
            'D2': 'D2 (2D diffusion)', 'D3': 'D3 (3D diffusion, Jander)',
        }
        friendly_name = MODEL_NAMES.get(base_name, model_display)

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
        methods_html = _create_methods_section_html(
            selected_model=selected_model,
            datasets=datasets,
            bootstrap_iterations=bootstrap_iters,
            confidence_level=0.95
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

            if selected_model_name == 'A->B->C':
                for step_name, description in (('1', 'A → B'), ('2', 'B → C')):
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

        # Add bootstrap information if available
        if summary and summary.get('bootstrap_iterations', 0) > 0:
            details_html += "<h3>Bootstrap Analysis</h3>"
            details_html += '<table class="param-table"><thead><tr><th>Item</th><th>Value</th></tr></thead><tbody>'
            details_html += f'<tr><td>Bootstrap Iterations</td><td>{summary["bootstrap_iterations"]}</td></tr>'
            details_html += f'<tr><td>Confidence Level</td><td>95%</td></tr>'

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
