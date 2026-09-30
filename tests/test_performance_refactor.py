"""
Tests for the fitting/bootstrap speed and consolidation work:

- fast temperature interpolation matches scipy's interp1d(fill_value='extrapolate')
- solver defaults (LSODA primary, RK45 fallback) and overrides
- closed-form simulate_kinetics matches ODE integration
- residual bootstrap adds resampled residuals to FITTED values, and is
  reproducible with random_state
- every name historically importable from akts.core still is
"""
import warnings

import numpy as np
import pytest
from scipy.interpolate import interp1d

from akts import KineticDataset, fit_kinetic_model, run_bootstrap, simulate_kinetics
from akts.bootstrap import _resample_residuals
from akts.simulation import (DEFAULT_SOLVER_OPTIONS, resolve_solver_options,
                             params_A_to_logA, params_logA_to_A)
from akts.utils import get_temperature_interpolator


def _isothermal_f1_datasets(seed=0, Ea=100e3, A=1e11, noise=0.005):
    rng = np.random.default_rng(seed)
    datasets = []
    for T in (313.15, 323.15, 333.15):
        t = np.linspace(0, 60 * 86400, 20)
        k = A * np.exp(-Ea / (8.314462618 * T))
        alpha = np.clip(1 - np.exp(-k * t) + rng.normal(0, noise, t.size), 0, 1)
        datasets.append(KineticDataset(time=t, temperature=np.full_like(t, T), conversion=alpha))
    return datasets


class TestTemperatureInterpolator:
    @pytest.mark.parametrize("t, T", [
        (np.linspace(0, 3600, 30), 300 + 0.1 * np.linspace(0, 3600, 30)),   # ramp
        (np.array([0, 10, 20, 50.]), np.array([300, 310, 305, 320.])),      # piecewise
        (np.array([50, 0, 20, 10.]), np.array([320, 300, 305, 310.])),      # unsorted input
    ])
    def test_matches_interp1d_inside_and_outside_range(self, t, T):
        ref = interp1d(t, T, kind='linear', bounds_error=False, fill_value='extrapolate')
        f = get_temperature_interpolator(t, T)
        q = np.linspace(t.min() - 100, t.max() + 100, 777)
        np.testing.assert_allclose(f(q), ref(q), atol=1e-9)
        # scalar path (used by the ODE solver) agrees with the array path
        np.testing.assert_allclose([f(float(x)) for x in q], f(q), atol=1e-9)
        assert isinstance(f(float(q[5])), float)

    def test_isothermal_is_constant(self):
        t = np.linspace(0, 1e6, 20)
        f = get_temperature_interpolator(t, np.full_like(t, 313.15))
        assert f(-1e9) == 313.15 and f(12345.0) == 313.15
        assert f(np.array([0.0, 1.0, 2.0])).tolist() == [313.15] * 3


class TestSolverOptions:
    def test_defaults(self):
        assert DEFAULT_SOLVER_OPTIONS['primary_solver'] == 'LSODA'
        assert DEFAULT_SOLVER_OPTIONS['fallback_solver'] == 'RK45'

    def test_override_and_disable_fallback(self):
        opts = resolve_solver_options({'primary_solver': 'BDF', 'fallback_solver': None, 'rtol': 1e-8})
        assert opts['primary_solver'] == 'BDF' and opts['fallback_solver'] is None
        assert opts['rtol'] == 1e-8 and opts['atol'] == DEFAULT_SOLVER_OPTIONS['atol']
        # the shared defaults must not be mutated by a call
        assert DEFAULT_SOLVER_OPTIONS['primary_solver'] == 'LSODA'

    @pytest.mark.parametrize("solver", ['LSODA', 'RK45', 'BDF', 'Radau'])
    def test_each_solver_gives_same_ramp_prediction(self, solver):
        # A temperature ramp forces the ODE path (no closed form applies).
        t = np.linspace(0, 30 * 86400, 60)
        T = np.linspace(313.15, 333.15, t.size)
        kwargs = dict(model_name='single_step', model_definition_args={'f_alpha_model': 'F1'},
                      kinetic_params={'Ea': 100e3, 'A': 1e11}, initial_alpha=0.0,
                      temperature_program=(t, T))
        ref = simulate_kinetics(**kwargs, solver_options={'primary_solver': 'LSODA', 'rtol': 1e-9, 'atol': 1e-12})
        out = simulate_kinetics(**kwargs, solver_options={'primary_solver': solver, 'fallback_solver': None})
        np.testing.assert_allclose(out.conversion, ref.conversion, atol=1e-4)


class TestClosedFormPrediction:
    @pytest.mark.parametrize("f_alpha_model", ['F1', 'F2', 'A2', 'R3', 'D3'])
    def test_closed_form_matches_ode(self, f_alpha_model):
        t = np.linspace(0, 200 * 86400, 80)
        common = dict(model_name='single_step', model_definition_args={'f_alpha_model': f_alpha_model},
                      kinetic_params={'Ea': 90e3, 'A': 1e10}, initial_alpha=0.0,
                      temperature_program=lambda tt: 323.15, simulation_time_sec=t)
        closed = simulate_kinetics(**common)
        # A 1 mK wiggle defeats the constant-temperature check and forces the ODE path.
        ode = simulate_kinetics(**{**common, 'temperature_program': lambda tt: 323.15 + 1e-3 * np.sin(np.asarray(tt))},
                                solver_options={'rtol': 1e-9, 'atol': 1e-12})
        np.testing.assert_allclose(closed.conversion, ode.conversion, atol=2e-3)


class TestParamEncoding:
    def test_round_trip(self):
        params = {'Ea': 90e3, 'A': 1e10}
        logA = params_A_to_logA(params, 'single_step')
        assert set(logA) == {'Ea', 'logA'}
        back = params_logA_to_A(logA)
        assert back['Ea'] == 90e3 and back['A'] == pytest.approx(1e10)
        assert all(type(v) is float for v in back.values())

    def test_rejects_non_positive_A(self):
        with pytest.raises(ValueError):
            params_A_to_logA({'Ea': 90e3, 'A': 0.0}, 'single_step')


class TestResidualResampling:
    def test_synthetic_is_fitted_plus_resampled_centered_residuals(self):
        rng = np.random.default_rng(0)
        fitted = [np.linspace(0.1, 0.9, 9), np.linspace(0.2, 0.8, 7)]
        observed = [f + rng.normal(0, 0.01, f.size) for f in fitted]
        pooled = np.concatenate([o - f for o, f in zip(observed, fitted)])
        centered = pooled - pooled.mean()

        synthetic = _resample_residuals(observed, fitted, np.random.default_rng(1))
        for syn, fit in zip(synthetic, fitted):
            added = syn - fit
            # every added value is one of the pooled centered residuals ...
            assert np.all(np.min(np.abs(added[:, None] - centered[None, :]), axis=1) < 1e-12)
        # ... and the result is NOT anchored on the observed data
        assert not np.allclose(synthetic[0], observed[0])

    def test_unevaluated_dataset_is_dropped(self):
        out = _resample_residuals([np.array([0.1, 0.2])], [None], np.random.default_rng(0))
        assert out == [None]


class TestBootstrapReproducibility:
    def test_same_seed_same_result_and_different_seed_differs(self):
        datasets = _isothermal_f1_datasets()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            fit = fit_kinetic_model(datasets, 'single_step', {'f_alpha_model': 'F1'},
                                    {'Ea': 90e3, 'A': 1e10}, verbose=False)
            a = run_bootstrap(datasets, fit, n_iterations=12, n_jobs=2, random_state=42)
            b = run_bootstrap(datasets, fit, n_iterations=12, n_jobs=2, random_state=42)
            c = run_bootstrap(datasets, fit, n_iterations=12, n_jobs=2, random_state=7)
        assert a.parameter_ci == b.parameter_ci
        np.testing.assert_array_equal(a.parameter_distributions['Ea'], b.parameter_distributions['Ea'])
        assert a.parameter_ci != c.parameter_ci

    def test_bootstrap_ci_brackets_true_values(self):
        Ea, A = 100e3, 1e11
        datasets = _isothermal_f1_datasets(seed=3, Ea=Ea, A=A)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            fit = fit_kinetic_model(datasets, 'single_step', {'f_alpha_model': 'F1'},
                                    {'Ea': 90e3, 'A': 1e10}, verbose=False)
            boot = run_bootstrap(datasets, fit, n_iterations=40, n_jobs=2, random_state=0)
        lo, hi = boot.parameter_ci['Ea']
        assert lo < fit.parameters['Ea'] < hi
        # a correct residual bootstrap gives a CI of the order of the parameter's
        # standard error -- not the multi-order-of-magnitude spread of the old
        # (point-shuffling) resampler
        assert (hi - lo) < 0.2 * fit.parameters['Ea']


def test_core_facade_still_exports_everything():
    import akts.core as core
    for name in ['fit_kinetic_model', 'run_bootstrap', 'run_bootstrap_empirical', 'predict_conversion',
                 'predict_conversion_model_free', 'simulate_kinetics', 'discover_kinetic_models',
                 'rank_models', 'rank_replicates', 'calculate_stats_for_replicate',
                 '_simulate_single_dataset', '_simulate_single_dataset_closed_form', '_with_eval_budget',
                 '_get_bootstrap_max_workers', '_calculate_conversion_stats', '_objective_function',
                 '_residual_vector_function', '_objective_function_rate', '_ArrheniusReparam',
                 '_initial_state', '_split_logA_name', '_to_logA_name', '_prepare_full_params_for_ode',
                 'get_log_param_names', 'get_model_info', 'ALPHA_SEED', 'EA_SCALE']:
        assert hasattr(core, name), name
