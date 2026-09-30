"""
Tests for wiring Friedman model-free prediction into auto_model_isothermal_data()
(TODO.md §6b, "Model-free (isoconversional) prediction"). Per the project
directive, Friedman is exposed via the existing `models_to_try` model
designation (e.g. models_to_try=['F1', 'Friedman']) rather than a separate
method='model_free' parameter, and participates in the full fit -> rank ->
select -> predict -> bootstrap pipeline like any other model.

Covers:
1. _setup_model_configs(['Friedman']) config shape.
2. _wrap_friedman_as_fit_result(): Friedman IsoResult -> FitResult with real
   RSS/R^2/AIC/BIC, indistinguishable from any other model to rank_models().
3. run_bootstrap_friedman(): per-alpha regression-input resampling (not the
   generic ODE-residual bootstrap, which doesn't apply -- Friedman has no
   optimizer/ODE step).
4. predict_conversion() dispatch for a Friedman FitResult, with and without a
   bootstrap_result attached (confidence band propagation).
5. Full auto_model_isothermal_data() runs with Friedman in models_to_try:
   normal multi-temperature case, Friedman-only case (forces it to be
   selected, exercising its own bootstrap+prediction path), and a
   single-dataset case (Friedman must fail gracefully, not crash the run).
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from akts import KineticDataset, simulate_kinetics, predict_conversion, auto_model_isothermal_data, models
from akts.isoconversional import run_friedman, run_bootstrap_friedman
from akts.helpers import _setup_model_configs, _wrap_friedman_as_fit_result


EA_TRUE, A_TRUE = 90000.0, 1e6
TEMPS = (313.0, 323.0, 333.0, 343.0)
ALPHA_LEVELS = np.linspace(0.01, 0.10, 10)


def _make_isothermal_datasets(Ea=EA_TRUE, A=A_TRUE, temps_K=TEMPS, t_end_days=182,
                              n_points=27, noise_std=0.005, seed_offset=0):
    """Same fixture pattern as tests/test_model_free_prediction.py."""
    datasets = []
    for i, T in enumerate(temps_K):
        t_sec = np.linspace(0, t_end_days * 86400, n_points)
        sim = simulate_kinetics(
            model_name='single_step', model_definition_args={'f_alpha_model': 'F1'},
            kinetic_params={'Ea': Ea, 'A': A}, initial_alpha=1e-6,
            temperature_program=(t_sec, np.full_like(t_sec, T)),
        )
        rng = np.random.default_rng(seed_offset + i)
        conv = np.clip(sim.conversion + rng.normal(0, noise_std, size=sim.conversion.shape), 0, 1)
        datasets.append(KineticDataset(time=t_sec, temperature=np.full_like(t_sec, T), conversion=conv))
    return datasets


class TestSetupModelConfigs:
    def test_friedman_config_shape(self):
        cfg, guesses, bounds = _setup_model_configs(['Friedman'])
        assert cfg == [{'name': 'Friedman_model', 'type': 'Friedman', 'def_args': {}}]
        # No optimizer, so no entries expected in the guess/bounds pools.
        assert 'Friedman_model' not in guesses
        assert 'Friedman_model' not in bounds

    def test_friedman_mixes_with_other_models(self):
        cfg, guesses, bounds = _setup_model_configs(['F1', 'Friedman', 'A2'])
        types = [c['type'] for c in cfg]
        assert types == ['single_step', 'Friedman', 'single_step']


class TestWrapFriedmanAsFitResult:
    def _fit_friedman(self):
        datasets = _make_isothermal_datasets()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            iso = run_friedman(datasets, alpha_levels=ALPHA_LEVELS)
        return iso, datasets

    def test_produces_valid_fit_result(self):
        iso, datasets = self._fit_friedman()
        fit_res = _wrap_friedman_as_fit_result(iso, datasets)
        assert fit_res.model_name == 'Friedman'
        assert fit_res.success
        assert fit_res.parameters == {}
        assert np.isfinite(fit_res.rss)
        assert np.isfinite(fit_res.aic)
        assert np.isfinite(fit_res.bic)
        assert 0.0 < fit_res.r_squared <= 1.0
        assert fit_res.n_parameters == int(np.isfinite(iso.Ea).sum())
        assert fit_res.model_definition_args['iso_result'] is iso

    def test_fails_gracefully_with_too_few_resolved_alphas(self):
        # A single dataset can't produce >=2 points for any per-alpha regression.
        datasets = _make_isothermal_datasets(temps_K=(313.0,))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            iso = run_friedman(datasets, alpha_levels=ALPHA_LEVELS)
        fit_res = _wrap_friedman_as_fit_result(iso, datasets)
        assert not fit_res.success
        assert fit_res.n_parameters < 2


class TestRunBootstrapFriedman:
    def _fit_friedman(self):
        datasets = _make_isothermal_datasets()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            iso = run_friedman(datasets, alpha_levels=ALPHA_LEVELS)
        return iso, datasets

    def test_produces_bootstrap_result(self):
        iso, datasets = self._fit_friedman()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            boot = run_bootstrap_friedman(datasets, iso, n_iterations=50)
        assert boot is not None
        assert boot.model_name == 'Friedman'
        assert boot.n_iterations > 0
        assert len(boot.parameter_ci) > 0
        assert all(k.startswith('Ea_alpha_') or k.startswith('lnAf_alpha_') for k in boot.parameter_ci)

    def test_raw_parameter_list_reconstructs_iso_result_shape(self):
        iso, datasets = self._fit_friedman()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            boot = run_bootstrap_friedman(datasets, iso, n_iterations=30)
        assert boot.raw_parameter_list
        for rep in boot.raw_parameter_list:
            assert set(rep.keys()) == {'alpha', 'Ea', 'ln_A_f_alpha'}
            assert len(rep['alpha']) == len(rep['Ea']) == len(rep['ln_A_f_alpha'])


class TestPredictConversionDispatch:
    def _fit_and_wrap(self):
        datasets = _make_isothermal_datasets()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            iso = run_friedman(datasets, alpha_levels=ALPHA_LEVELS)
        return _wrap_friedman_as_fit_result(iso, datasets), datasets

    def test_predicts_without_bootstrap(self):
        fit_res, _ = self._fit_and_wrap()
        t_eval = np.linspace(0, 365 * 86400, 30)
        pred = predict_conversion(fit_res, temperature_program=lambda t: 298.15, simulation_time_sec=t_eval)
        assert np.all(np.isfinite(pred.conversion))
        assert pred.conversion_ci is None

    def test_predicts_with_bootstrap_ci_brackets_point(self):
        fit_res, datasets = self._fit_and_wrap()
        iso_result = fit_res.model_definition_args['iso_result']
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            boot = run_bootstrap_friedman(datasets, iso_result, n_iterations=40)

        t_eval = np.linspace(0, 2 * 365 * 86400, 30)
        pred = predict_conversion(fit_res, temperature_program=lambda t: 298.15,
                                  simulation_time_sec=t_eval, bootstrap_result=boot)
        assert pred.conversion_ci is not None
        lower, upper = pred.conversion_ci
        assert np.all(lower <= pred.conversion + 1e-9)
        assert np.all(pred.conversion <= upper + 1e-9)


class TestAutoModelIsothermalDataWithFriedman:
    def test_friedman_participates_in_ranking(self):
        datasets = _make_isothermal_datasets()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            results = auto_model_isothermal_data(
                data_files=datasets, models_to_try=['F1', 'F2', 'A2', 'Friedman'],
                bootstrap_iterations=0, report_path=None,
            )
        model_names = [m['model_name'] for m in results['top_models']]
        assert any('Friedman' in name for name in model_names)

    def test_friedman_only_gets_selected_and_predicts_with_ci(self):
        datasets = _make_isothermal_datasets()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            results = auto_model_isothermal_data(
                data_files=datasets, models_to_try=['Friedman'],
                bootstrap_iterations=30, predict=(1, 'year'), report_path=None,
            )
        assert 'Friedman' in results['selected_model']['model_name']
        pred = results['predictions']
        assert pred is not None
        assert np.isfinite(pred['conversion_mean'][-1])
        assert 'conversion_lower' in pred
        assert 'conversion_upper' in pred

    def test_single_dataset_fails_gracefully_not_crash(self):
        datasets = _make_isothermal_datasets(temps_K=(313.0,))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            results = auto_model_isothermal_data(
                data_files=datasets, models_to_try=['F1', 'Friedman'],
                bootstrap_iterations=0, report_path=None,
            )
        # Friedman can't resolve any alpha level from one temperature -- the run
        # must still succeed overall, just without Friedman in the results.
        model_names = [m['model_name'] for m in results['top_models']]
        assert not any('Friedman' in name for name in model_names)
        assert 'F1' in results['selected_model']['model_name']


class TestFriedmanModelSelection:
    """Friedman is opt-in and requires at least three temperatures for a fit."""

    def test_not_included_by_default_with_three_temperatures(self):
        datasets = _make_isothermal_datasets(temps_K=(313.0, 328.0, 343.0))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            results = auto_model_isothermal_data(
                data_files=datasets, top_n=20, bootstrap_iterations=0, report_path=None,
            )
        model_names = [m['model_name'] for m in results['top_models']]
        assert not any('Friedman' in name for name in model_names)

    def test_not_included_by_default_with_four_temperatures(self):
        datasets = _make_isothermal_datasets()  # TEMPS has 4 distinct values
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            results = auto_model_isothermal_data(
                data_files=datasets, top_n=20, bootstrap_iterations=0, report_path=None,
            )
        model_names = [m['model_name'] for m in results['top_models']]
        assert not any('Friedman' in name for name in model_names)

    def test_explicit_empirical_selection_does_not_add_friedman(self):
        datasets = _make_isothermal_datasets()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            results = auto_model_isothermal_data(
                data_files=datasets, models_to_try=models.empirical.all,
                top_n=20, bootstrap_iterations=0, report_path=None,
            )
        model_names = [m['model_name'] for m in results['top_models']]
        assert not any('Friedman' in name for name in model_names)

    def test_excluded_by_default_with_two_temperatures(self):
        datasets = _make_isothermal_datasets(temps_K=(313.0, 343.0))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            results = auto_model_isothermal_data(
                data_files=datasets, top_n=20, bootstrap_iterations=0, report_path=None,
            )
        model_names = [m['model_name'] for m in results['top_models']]
        assert not any('Friedman' in name for name in model_names)

    def test_excluded_by_default_with_one_temperature(self):
        datasets = _make_isothermal_datasets(temps_K=(313.0,))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            results = auto_model_isothermal_data(
                data_files=datasets, top_n=20, bootstrap_iterations=0, report_path=None,
            )
        model_names = [m['model_name'] for m in results['top_models']]
        assert not any('Friedman' in name for name in model_names)

    def test_explicit_friedman_with_two_temperatures_fails_gracefully(self):
        # Friedman is selected explicitly, but two temperatures cannot provide
        # enough information for its per-conversion regression.
        datasets = _make_isothermal_datasets(temps_K=(313.0, 343.0))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            results = auto_model_isothermal_data(
                data_files=datasets, models_to_try=['F1', 'Friedman'],
                bootstrap_iterations=0, report_path=None,
            )
        # 2 temperatures resolves at most 1 alpha level (0 DoF) -- expect the
        # same graceful-failure outcome as the single-dataset case, not a crash.
        model_names = [m['model_name'] for m in results['top_models']]
        assert not any('Friedman' in name for name in model_names)
