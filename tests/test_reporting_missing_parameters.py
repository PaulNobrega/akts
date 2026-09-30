import numpy as np
import builtins
import akts.reporting as reporting
from types import SimpleNamespace

from akts.datatypes import BootstrapResult, FitResult, KineticDataset
from akts.helpers import _predict_empirical_with_ci
from akts.reporting import (
    _create_methods_section_html,
    _create_prediction_plot_interactive,
    _create_simulation_plot_interactive,
    generate_isothermal_report,
)
from akts.utils import SECONDS_PER_YEAR, SECONDS_PER_DAY


def test_methods_describe_bootstrap_by_model_family_and_only_when_used():
    dataset = KineticDataset(
        time=np.array([0.0, 1.0, 2.0]),
        temperature=np.array([313.15, 313.15, 313.15]),
        conversion=np.array([0.0, 0.1, 0.2]),
    )

    friedman_html = _create_methods_section_html(
        {'model_name': 'Friedman', 'parameters': {}}, [dataset], bootstrap_iterations=37
    )
    assert 'Monte Carlo case bootstrap' in friedman_html
    assert 'Complete observed rows' in friedman_html
    assert 'Friedman isoconversional regressions were rerun' in friedman_html
    assert 'parametric bootstrap' not in friedman_html
    assert '37 successful replicates' in friedman_html

    empirical_html = _create_methods_section_html(
        {'model_name': 'Empirical_Linear', 'parameters': {}}, [dataset],
        bootstrap_iterations=24, bootstrap_method='parametric'
    )
    assert 'parametric Gaussian bootstrap' in empirical_html
    assert 'Gaussian conversion errors' in empirical_html
    assert 'selected model was re-fitted' in empirical_html

    mechanistic_html = _create_methods_section_html(
        {'model_name': 'A3_model', 'parameters': {}}, [dataset],
        bootstrap_iterations=18, bootstrap_method='residual'
    )
    assert 'weighted residual bootstrap' in mechanistic_html
    assert 'selected model was re-fitted' in mechanistic_html

    no_bootstrap_html = _create_methods_section_html(
        {'model_name': 'A3_model', 'parameters': {}}, [dataset], bootstrap_iterations=0
    )
    assert 'Uncertainty Quantification' not in no_bootstrap_html


def test_empirical_bootstrap_prediction_produces_confidence_band():
    fit_result = FitResult(
        model_name='Empirical_Linear',
        parameters={'Ea': 0.0, 'A': 0.1, 'C': 0.1},
        success=True,
        message='Synthetic fit',
        rss=0.0,
        n_datapoints=3,
        n_parameters=3,
        model_definition_args={'empirical_type': 'Linear'},
    )
    bootstrap_result = BootstrapResult(
        model_name='Empirical_Linear',
        parameter_distributions={
            'Ea': np.array([0.0, 0.0, 0.0]),
            'A': np.array([0.1, 0.2, 0.3]),
            'C': np.array([0.05, 0.1, 0.15]),
        },
        parameter_ci={},
        n_iterations=3,
        confidence_level=0.95,
    )

    prediction = _predict_empirical_with_ci(
        fit_result, np.array([0.0, 1.0, 2.0]), 298.15, bootstrap_result
    )

    assert prediction.conversion_ci is not None
    lower, upper = prediction.conversion_ci
    assert lower.shape == prediction.conversion.shape
    assert upper.shape == prediction.conversion.shape
    assert np.all(lower < upper)


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
    assert '(dα/dt)<sub>α,T</sub> = A f(α)' in html
    assert 'ln[(dα/dt)<sub>α</sub>] = ln[A f(α)]' in html
    assert 'Ea(α) = −R m<sub>α</sub>' in html
    assert 'not a standalone pre-exponential factor' in html
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


def test_report_bootstrap_details_reflect_auto_model_settings():
    dataset = KineticDataset(
        time=np.array([0.0, 1.0, 2.0]),
        temperature=np.array([313.15, 313.15, 313.15]),
        conversion=np.array([0.0, 0.1, 0.2]),
    )
    html = generate_isothermal_report(
        datasets=[dataset],
        top_models=[],
        selected_model={'model_name': 'A3_model', 'parameters': {}, 'statistics': {}},
        summary={
            'bootstrap_method': 'parametric',
            'bootstrap_iterations_requested': 100,
            'bootstrap_iterations': 87,
            'confidence_level': 0.90,
        },
        report_path=None,
        report_format='static',
    )

    assert '<td>Bootstrap Method</td><td>Parametric Gaussian resampling</td>' in html
    assert '<td>Iterations Requested</td><td>100</td>' in html
    assert '<td>Successful Iterations</td><td>87</td>' in html
    assert '<td>Confidence Level</td><td>90%</td>' in html
    assert 'The 90% parameter confidence intervals' in html


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


def test_static_prediction_and_simulation_plots_match_interactive_styling(monkeypatch):
    figures = []
    monkeypatch.setattr(
        reporting, '_embed_base64_image', lambda fig: figures.append(fig) or 'image'
    )

    reporting._create_prediction_plot_matplotlib({
        'time': [0, 1, 2],
        'time_unit': 'days',
        'conversion_mean': [0.0, 0.1, 0.25],
        'conversion_lower': [0.0, 0.08, 0.2],
        'conversion_upper': [0.0, 0.12, 0.3],
    })
    reporting._create_simulation_plot_matplotlib({
        'time': [0, SECONDS_PER_DAY, 2 * SECONDS_PER_DAY],
        'time_unit': 'days',
        'conversion_mean': [0.0, 0.1, 0.25],
        'conversion_lower': [0.0, 0.08, 0.2],
        'conversion_upper': [0.0, 0.12, 0.3],
        'temperature': [298.15, 313.15, 298.15],
        'temperature_units': 'K',
        'input_profile': [(0, 298.15), (1, 313.15), (2, 298.15)],
    })

    prediction_ax = figures[0].axes[0]
    assert prediction_ax.lines[0].get_color() == '#0064c8'
    assert prediction_ax.lines[0].get_linewidth() == 3
    assert prediction_ax.get_legend_handles_labels()[1] == ['95% CI', 'Predicted Conversion']

    simulation_fig = figures[1]
    simulation_ax = simulation_fig.axes[0]
    assert len(simulation_fig.axes) == 1
    assert simulation_ax.get_ylabel() == 'Conversion'
    assert simulation_ax.get_ylim() == (-0.05, 1.1)
    assert [segment[0][0] for segment in simulation_ax.collections[1].get_segments()] == [0, 1, 2]
    assert [annotation.get_text() for annotation in simulation_ax.texts[:3]] == [
        '298K (25°C)', '313K (40°C)', '298K (25°C)',
    ]

    dataset = KineticDataset(
        time=np.array([0.0, 1.0, 2.0]),
        temperature=np.array([298.15, 298.15, 298.15]),
        conversion=np.array([0.0, 0.1, 0.2]),
    )
    reporting._create_fit_plot_matplotlib(
        [dataset], SimpleNamespace(conversion_simulated=[np.array([0.0, 0.1, 0.2])])
    )
    assert figures[2].axes[0].lines[0].get_color() == '#440154'


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

    # Use the library's own SECONDS_PER_YEAR/SECONDS_PER_DAY rather than a
    # hardcoded 365-day-year/86400-day literal, so this test doesn't pin one
    # particular year/day convention independently of akts.utils.
    _create_prediction_plot_interactive({
        'time': [0, SECONDS_PER_YEAR, 2 * SECONDS_PER_YEAR],
        'time_unit': 'seconds',
        'requested_time': {'value': 2, 'unit': 'years'},
        'conversion_mean': [0.0, 0.1, 0.2],
    })
    _create_simulation_plot_interactive({
        'time': [0, SECONDS_PER_DAY, 2 * SECONDS_PER_DAY],
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
