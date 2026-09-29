import numpy as np
import builtins

from akts.datatypes import KineticDataset
from akts.reporting import (
    _create_prediction_plot_interactive,
    _create_simulation_plot_interactive,
    generate_isothermal_report,
)


def test_report_handles_missing_arrhenius_parameters_and_statistics():
    dataset = KineticDataset(
        time=np.array([0.0, 1.0, 2.0]),
        temperature=np.array([298.15, 298.15, 298.15]),
        conversion=np.array([0.0, 0.1, 0.2]),
    )
    selected_model = {
        'model_name': 'Friedman',
        'parameters': {'Ea': None, 'A': None, 'Ea_alpha_0.1': 85000.0},
        'statistics': {'r_squared': None, 'aic': np.nan, 'n_params': None},
    }

    html = generate_isothermal_report(
        datasets=[dataset],
        top_models=[{
            'rank': 1,
            'model_name': 'Friedman',
            'statistics': {
                'r_squared': None,
                'aic': np.nan,
                'bic': None,
                'rss': None,
                'rmse': None,
                'r_squared_adj': None,
                'akaike_weight': None,
            },
            'score': None,
            'n_parameters': None,
        }],
        selected_model=selected_model,
        report_path=None,
        report_format='static',
    )

    assert 'Temperature Dependence' in html
    assert 'does not provide a single global Arrhenius parameter pair' in html
    assert 'Activation energy was estimated at conversion levels using linear' in html
    assert '{fitting_method_html}' not in html
    assert '<td>N/A</td>' in html
    assert 'nan' not in html.lower()


def test_report_handles_empty_parameter_dictionary():
    dataset = KineticDataset(
        time=np.array([0.0, 1.0, 2.0]),
        temperature=np.array([298.15, 298.15, 298.15]),
        conversion=np.array([0.0, 0.1, 0.2]),
    )

    html = generate_isothermal_report(
        datasets=[dataset],
        top_models=[],
        selected_model={'model_name': 'Unknown', 'parameters': {}, 'statistics': {}},
        report_path=None,
        report_format='static',
    )

    assert 'Methods' in html
    assert 'Goodness of Fit' in html


def test_report_formats_separate_arrhenius_pairs_for_consecutive_reactions():
    dataset = KineticDataset(
        time=np.array([0.0, 1.0, 2.0]),
        temperature=np.array([298.15, 298.15, 298.15]),
        conversion=np.array([0.0, 0.1, 0.2]),
    )

    html = generate_isothermal_report(
        datasets=[dataset],
        top_models=[],
        selected_model={
            'model_name': 'A->B->C',
            'parameters': {
                'Ea1': 85000.0,
                'A1': 1e11,
                'Ea2': 95000.0,
                'A2': 1e12,
            },
            'statistics': {},
        },
        report_path=None,
        report_format='static',
    )

    assert 'Arrhenius Temperature Dependence by Reaction Step' in html
    assert 'A → B' in html
    assert 'B → C' in html
    assert 'Ea1' in html and 'A1' in html
    assert 'Ea2' in html and 'A2' in html


def test_prediction_and_simulation_plots_fall_back_when_plotly_is_unavailable(monkeypatch):
    original_import = builtins.__import__

    def import_without_plotly(name, *args, **kwargs):
        if name == 'plotly' or name.startswith('plotly.'):
            raise ImportError('Plotly is unavailable for this test')
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', import_without_plotly)

    prediction_html = _create_prediction_plot_interactive({
        'time': [0, 1, 2],
        'time_unit': 'days',
        'conversion_mean': [0.0, 0.1, 0.25],
        'conversion_lower': [0.0, 0.08, 0.2],
        'conversion_upper': [0.0, 0.12, 0.3],
    })
    simulation_html = _create_simulation_plot_interactive({
        'time': [0, 86400, 172800],
        'time_unit': 'days',
        'conversion_mean': [0.0, 0.1, 0.25],
        'conversion_lower': [0.0, 0.08, 0.2],
        'conversion_upper': [0.0, 0.12, 0.3],
        'temperature': [298.15, 313.15, 298.15],
        'temperature_units': 'K',
        'input_profile': [(0, 298.15), (1, 313.15), (2, 298.15)],
    })

    assert 'data:image/png;base64,' in prediction_html
    assert 'data:image/png;base64,' in simulation_html
    assert 'Plotly not available' not in prediction_html + simulation_html


def test_interactive_prediction_and_simulation_axes_use_display_units(monkeypatch):
    import plotly.graph_objects as go

    from akts.reporting import (
        _create_prediction_plot_interactive,
        _create_simulation_plot_interactive,
    )

    figures = []

    def capture_figure(self, **kwargs):
        figures.append(self)
        return 'captured'

    monkeypatch.setattr(go.Figure, 'to_html', capture_figure)

    _create_prediction_plot_interactive({
        'time': [0, 31_536_000, 63_072_000],
        'time_unit': 'seconds',
        'requested_time': {'value': 2, 'unit': 'years'},
        'conversion_mean': [0.0, 0.1, 0.2],
    })
    _create_simulation_plot_interactive({
        'time': [0, 86_400, 172_800],
        'time_unit': 'days',
        'conversion_mean': [0.0, 0.1, 0.2],
        'temperature': [298.15, 313.15, 298.15],
        'temperature_units': 'K',
        'input_profile': [(0, 298.15), (1, 313.15), (2, 298.15)],
    })

    assert list(figures[0].data[0].x) == [0.0, 1.0, 2.0]
    assert figures[0].layout.xaxis.title.text == 'Time (years)'
    assert list(figures[1].data[0].x) == [0.0, 1.0, 2.0]
    assert figures[1].layout.xaxis.title.text == 'Time (days)'
    assert [(shape.x0, shape.x1) for shape in figures[1].layout.shapes] == [
        (0, 0), (1, 1), (2, 2),
    ]
