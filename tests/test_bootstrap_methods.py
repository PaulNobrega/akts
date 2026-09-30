import inspect

import numpy as np
import pytest
from types import SimpleNamespace

from akts import (
    BOOTSTRAP_METHODS,
    run_bootstrap,
    run_bootstrap_empirical,
    run_bootstrap_friedman,
    simulate_kinetics,
)
import akts.bootstrap as bootstrap
import akts.empirical as empirical
from akts.bootstrap import _make_resampled_datasets, _normalize_bootstrap_method
from akts.datatypes import BootstrapResult, FitResult, KineticDataset
from akts.helpers import auto_model_isothermal_data
from akts.isoconversional import run_friedman
from akts.json_utils import bootstrap_to_json


def _dataset(offset=0.0):
    time = np.arange(8, dtype=float)
    return KineticDataset(
        time=time,
        temperature=np.full_like(time, 298.15),
        conversion=np.linspace(0.02, 0.72, len(time)) + offset,
        relative_humidity=np.linspace(0.3, 0.7, len(time)),
    )


def test_case_bootstrap_resamples_complete_rows_and_preserves_dataset_size():
    datasets = [_dataset(), _dataset(0.01)]
    result = _make_resampled_datasets(
        datasets, [None, None], np.random.default_rng(8), 'monte_carlo'
    )

    assert [len(dataset.time) for dataset in result] == [len(dataset.time) for dataset in datasets]
    for original, sampled in zip(datasets, result):
        assert np.all(np.diff(sampled.time) >= 0)
        original_rows = set(zip(original.time, original.temperature, original.conversion,
                                original.relative_humidity))
        sampled_rows = zip(sampled.time, sampled.temperature, sampled.conversion,
                           sampled.relative_humidity)
        assert all(tuple(row) in original_rows for row in sampled_rows)


def test_case_bootstrap_ode_simulation_handles_duplicate_times():
    original = _dataset()
    sampled = _make_resampled_datasets(
        [original], [None], np.random.default_rng(8), 'monte_carlo'
    )[0]
    assert len(np.unique(sampled.time)) < len(sampled.time)

    prediction = simulate_kinetics(
        model_name='A->B->C',
        model_definition_args={'f1_model': 'F1', 'f2_model': 'F1'},
        kinetic_params={'Ea1': 50000.0, 'A1': 1e5, 'Ea2': 60000.0, 'A2': 1e6},
        initial_alpha=0.0,
        temperature_program=(sampled.time, sampled.temperature),
        simulation_time_sec=sampled.time,
    )

    assert len(prediction.conversion) == len(sampled.time)
    assert np.all(np.isfinite(prediction.conversion))
    unique_times = np.unique(sampled.time)
    for time in unique_times:
        values = prediction.conversion[sampled.time == time]
        assert np.all(values == values[0])


@pytest.mark.parametrize('method', ['parametric', 'residual'])
def test_model_based_bootstraps_keep_observation_coordinates(method):
    datasets = [_dataset()]
    fitted = [datasets[0].conversion - np.array([-.01, .02, -.015, .01, -.02, .015, -.01, .005])]

    first = _make_resampled_datasets(datasets, fitted, np.random.default_rng(17), method)
    second = _make_resampled_datasets(datasets, fitted, np.random.default_rng(17), method)

    assert len(first[0].time) == len(datasets[0].time)
    assert np.array_equal(first[0].time, datasets[0].time)
    assert np.array_equal(first[0].temperature, datasets[0].temperature)
    assert np.allclose(first[0].conversion, second[0].conversion)
    assert np.all((first[0].conversion >= 0.0) & (first[0].conversion <= 1.0))


def test_bootstrap_method_defaults_and_json_metadata():
    assert BOOTSTRAP_METHODS == ('monte_carlo', 'parametric', 'residual')
    for function in (run_bootstrap, run_bootstrap_empirical, run_bootstrap_friedman,
                     auto_model_isothermal_data):
        assert inspect.signature(function).parameters['bootstrap_method'].default == 'monte_carlo'

    result = BootstrapResult(
        model_name='F1', parameter_distributions={}, parameter_ci={}, n_iterations=4,
        confidence_level=0.95, bootstrap_method='parametric',
    )
    assert bootstrap_to_json(result)['bootstrap_method'] == 'parametric'


@pytest.mark.parametrize('method', BOOTSTRAP_METHODS)
def test_empirical_bootstrap_dispatches_selected_method(monkeypatch, method):
    fit_result = FitResult(
        model_name='Empirical_Linear',
        parameters={'Ea': 50000.0, 'A': 1e5, 'C': 0.01},
        success=True,
        message='fit',
        rss=0.01,
        n_datapoints=8,
        n_parameters=3,
        model_definition_args={'empirical_type': 'Linear'},
    )
    received_methods = []

    def fake_run_replicates(worker, worker_args, n_iterations, *args, **kwargs):
        received_methods.append(worker_args[-1])
        return [
            {'params': dict(worker_args[2]),
             'stats': {'r_squared': 0.9, 'rss': 0.01, 'aic': -10.0, 'bic': -8.0}}
            for _ in range(n_iterations)
        ]

    monkeypatch.setattr(bootstrap, '_run_replicates', fake_run_replicates)
    monkeypatch.setattr(
        empirical,
        'fit_empirical_global',
        lambda **kwargs: SimpleNamespace(success=True, r_squared=0.9, rss=0.01, aic=-10.0, bic=-8.0),
    )
    result = run_bootstrap_empirical(
        [_dataset()], fit_result, n_iterations=10, bootstrap_method=method
    )

    assert result is not None
    assert received_methods == [method]
    assert result.bootstrap_method == method


def test_bootstrap_method_validation():
    assert _normalize_bootstrap_method(' RESIDUAL ') == 'residual'
    with pytest.raises(ValueError, match='bootstrap_method'):
        _normalize_bootstrap_method('unknown')


@pytest.mark.parametrize('method', BOOTSTRAP_METHODS)
def test_friedman_bootstrap_supports_each_method(method):
    datasets = []
    time = np.linspace(0.0, 100.0, 30)
    for temperature in (298.15, 308.15, 318.15, 328.15):
        rate = 1e12 * np.exp(-80000.0 / (8.314 * temperature))
        datasets.append(KineticDataset(
            time=time,
            temperature=np.full_like(time, temperature),
            conversion=1.0 - np.exp(-rate * time),
        ))
    alpha_levels = np.linspace(0.05, 0.5, 8)
    iso_result = run_friedman(datasets, alpha_levels=alpha_levels)
    result = run_bootstrap_friedman(
        datasets, iso_result, n_iterations=5, alpha_levels=alpha_levels,
        random_state=19, bootstrap_method=method,
    )

    assert result is not None
    assert result.bootstrap_method == method
    assert result.n_iterations > 0