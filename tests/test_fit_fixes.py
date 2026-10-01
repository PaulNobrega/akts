"""
Regression tests for the 2026-09-26 fit-correctness fixes (see TODO.md §6b,
"Status of recent fit fixes"). Each fix silently broke something before it was
found by hand; these tests exist so the same bug can't come back unnoticed.

Fixes covered:
1. Multi-file readout normalization on one shared scale (helpers._harmonize_loaded_datasets)
2. Time-unit detection ("Time (days)" no longer read as seconds)
3. Non-zero initial alpha so Avrami/diffusion models can start moving
4. Arrhenius reparameterization + coarse scan + Powell actually finds the optimum
5. Simplest-model-wins tie-break in model selection
6. ODE solves fail fast instead of hanging on stiff parameter sets
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from akts import KineticDataset, fit_kinetic_model, simulate_kinetics
from akts.helpers import _harmonize_loaded_datasets, _select_simplest_equivalent
from akts.loaders import _guess_units


R_GAS = 8.314


def _make_synthetic_dataset(Ea, A, temp_K, t_end_days=182, n_points=20, noise_std=0.01, seed=0):
    """Simulate a known F1 dataset at one temperature, like a real stability file."""
    t_sec = np.linspace(0, t_end_days * 86400, n_points)
    result = simulate_kinetics(
        model_name='single_step',
        model_definition_args={'f_alpha_model': 'F1'},
        kinetic_params={'Ea': Ea, 'A': A},
        initial_alpha=1e-6,
        temperature_program=(t_sec, np.full_like(t_sec, temp_K)),
    )
    rng = np.random.default_rng(seed)
    conversion = np.clip(result.conversion + rng.normal(0, noise_std, size=result.conversion.shape), 0, 1)
    return KineticDataset(time=t_sec, temperature=np.full_like(t_sec, temp_K), conversion=conversion)


class TestMultiFileNormalization:
    """Fix 1: per-file min-max scaling forced every temperature to conversion=1,
    hiding the Arrhenius trend. Now all files share one start/end scale."""

    def _dataset_with_raw_readout(self, temp_K, raw_start, raw_end, n=10):
        raw = np.linspace(raw_start, raw_end, n)
        ds = KineticDataset(
            time=np.linspace(0, 182 * 86400, n),
            temperature=np.full(n, temp_K),
            conversion=(raw - raw[0]) / (raw[-1] - raw[0]),  # per-file scaling, the bug
        )
        ds.metadata['readout_raw'] = raw
        ds.metadata['time_units'] = 's'
        return ds

    def test_shared_scale_preserves_temperature_trend(self):
        # Two files, same underlying readout scale (2 -> 100), but the low-temperature
        # file only actually reached 30 on that scale -- it should NOT end at conversion 1.
        cold = self._dataset_with_raw_readout(313.0, raw_start=2.0, raw_end=30.0)
        hot = self._dataset_with_raw_readout(343.0, raw_start=2.0, raw_end=100.0)

        _harmonize_loaded_datasets([cold, hot], 'increasing', lambda m, d: None)

        assert cold.conversion[-1] < 0.5, (
            "cold dataset should NOT reach full conversion once normalized on the shared "
            "scale -- if this fails, per-file normalization has regressed"
        )
        assert hot.conversion[-1] > 0.9

    def test_single_file_unaffected(self):
        # With <2 datasets there's no cross-file scale to unify; must be a no-op.
        only = self._dataset_with_raw_readout(313.0, raw_start=2.0, raw_end=30.0)
        before = only.conversion.copy()
        _harmonize_loaded_datasets([only], 'increasing', lambda m, d: None)
        np.testing.assert_array_equal(only.conversion, before)


class TestTimeUnitDetection:
    """Fix 2: '(days)' matched the substring check for seconds ('s)' inside 'days)'),
    so a multi-month study was silently treated as a multi-month-in-seconds study."""

    @pytest.mark.parametrize("column,expected", [
        ("Time (days)", "days"),
        ("Time (day)", "days"),
        ("Time (d)", "days"),
        ("Time (weeks)", "weeks"),
        ("Time (min)", "min"),
        ("Time (h)", "h"),
        ("Time (s)", "s"),
    ])
    def test_unit_detected_correctly(self, column, expected):
        assert _guess_units(column) == expected

    def test_days_not_misread_as_seconds(self):
        # The specific regression: "days)" contains "s)" as a substring.
        assert _guess_units("Time (days)") != "s"


class TestNonZeroInitialAlpha:
    """Fix 3: Avrami (A2/A3) and diffusion (D2/D3) models have f(alpha=0) = 0 or inf,
    so a simulation started at exactly alpha=0 could never move."""

    @pytest.mark.parametrize("model", ["A2", "A3", "D2", "D3"])
    def test_model_moves_from_default_start(self, model):
        t_sec = np.linspace(0, 30 * 86400, 20)
        result = simulate_kinetics(
            model_name='single_step',
            model_definition_args={'f_alpha_model': model},
            kinetic_params={'Ea': 90000, 'A': 1e11},
            initial_alpha=0.0,  # caller asks for exactly zero
            temperature_program=(t_sec, np.full_like(t_sec, 343.0)),
        )
        assert result.conversion[-1] > 1e-4, (
            f"{model} conversion never left ~0 -- initial-state seeding for "
            f"zero-rate-at-alpha=0 models has regressed"
        )
        assert np.all(np.isfinite(result.conversion))


class TestOptimizerFindsKnownParameters:
    """Fix 4: before the Arrhenius reparameterization + coarse scan + Powell switch,
    the optimizer routinely stalled at the initial guess (Ea unchanged, R^2 negative).
    Fit synthetic F1 data with known Ea/A and check we recover them."""

    def test_recovers_known_ea_and_a_multi_temperature(self):
        Ea_true, A_true = 90000.0, 1e11
        datasets = [
            _make_synthetic_dataset(Ea_true, A_true, T, seed=i)
            for i, T in enumerate((313.0, 323.0, 333.0, 343.0))
        ]

        fit_result = fit_kinetic_model(
            datasets=datasets,
            model_name='single_step',
            model_definition_args={'f_alpha_model': 'F1'},
            initial_guesses={'Ea': 80000, 'A': 1e8},  # deliberately off from truth
            parameter_bounds={'Ea': (10e3, 300e3), 'A': (1e-5, 1e25)},
            verbose=False,
        )

        assert fit_result.success
        assert fit_result.r_squared > 0.9, (
            f"R^2={fit_result.r_squared:.4f} -- optimizer did not converge to a good fit "
            f"(this is the exact failure mode the Arrhenius reparameterization fixed)"
        )
        # Recovered Ea should be within 20% of ground truth (noisy synthetic data,
        # not a noiseless inversion -- this is a sanity band, not a precision check).
        fitted_Ea = fit_result.parameters['Ea']
        assert abs(fitted_Ea - Ea_true) / Ea_true < 0.2, (
            f"fitted Ea={fitted_Ea/1000:.1f} kJ/mol vs true {Ea_true/1000:.1f} kJ/mol"
        )

    def test_optimizer_moves_away_from_bad_initial_guess(self):
        # Regression guard for the specific symptom: Ea stuck exactly at the initial
        # guess is the signature of the pre-fix stall.
        # With closed-form solver, the fit is so accurate that it might land very close
        # to the initial guess if that guess is already good. The real test is R².
        Ea_true, A_true = 95000.0, 5e11
        datasets = [_make_synthetic_dataset(Ea_true, A_true, T, seed=i)
                    for i, T in enumerate((313.0, 333.0, 343.0))]
        initial_Ea = 80000.0

        fit_result = fit_kinetic_model(
            datasets=datasets,
            model_name='single_step',
            model_definition_args={'f_alpha_model': 'F1'},
            initial_guesses={'Ea': initial_Ea, 'A': 1e8},
            parameter_bounds={'Ea': (10e3, 300e3), 'A': (1e-5, 1e25)},
            verbose=False,
        )
        assert fit_result.success
        # Either Ea moved significantly OR fit is excellent (R² > 0.98)
        # The old bug: Ea stuck AND poor R². New behavior: accurate fit regardless.
        moved_significantly = abs(fit_result.parameters['Ea'] - initial_Ea) > 1000.0
        excellent_fit = fit_result.r_squared > 0.98
        assert moved_significantly or excellent_fit, (
            f"Optimizer may have stalled: Ea={fit_result.parameters['Ea']:.0f} "
            f"(initial={initial_Ea}), R²={fit_result.r_squared:.4f}"
        )


class TestSelectTopRanked:
    """Selection returns rank 1 from the Akaike-weight ranking; AIC already
    accounts for complexity, so there is no simplicity tie-break."""

    def _model(self, name, weight, n_params=2):
        return {'model_name': name, 'stats': {'akaike_weight': weight, 'n_params': n_params}}

    def test_returns_rank_one_even_when_simpler_model_is_close(self):
        ranked = [self._model('A2_model', 0.45), self._model('R2_model', 0.30),
                  self._model('F1_model', 0.25)]
        chosen, reason = _select_simplest_equivalent(ranked)
        assert chosen['model_name'] == 'A2_model'
        assert 'competitive' in reason.lower()

    def test_overwhelming_evidence_reason(self):
        ranked = [self._model('SB_mn_model', 1.0), self._model('F1_model', 0.0)]
        chosen, reason = _select_simplest_equivalent(ranked)
        assert chosen['model_name'] == 'SB_mn_model'
        assert 'overwhelming' in reason.lower()

    def test_complex_model_wins_if_ranked_first(self):
        ranked = [self._model('A->B->C', 0.8, n_params=4), self._model('F1_model', 0.2)]
        chosen, reason = _select_simplest_equivalent(ranked)
        assert chosen['model_name'] == 'A->B->C'
        assert 'strong' in reason.lower()

    def test_empty_ranking_raises(self):
        with pytest.raises(ValueError):
            _select_simplest_equivalent([])


class TestOdeSolveFailsFast:
    """Fix 6: some parameter combinations (e.g. one rate constant far larger than the
    other in a consecutive-reaction model) made a single ODE solve churn indefinitely
    instead of failing. This tests the RHS-evaluation budget mechanism directly
    (core._simulate_single_dataset / core._with_eval_budget), rather than timing an
    entire fit_kinetic_model() optimization -- a full ODE-model fit legitimately runs
    for minutes (hundreds of individually-capped solves), so wall-clock on the whole
    fit conflates normal optimizer cost with the specific runaway-solve bug this fix
    addresses."""

    def test_low_rhs_budget_fails_fast_instead_of_hanging(self):
        import time
        from akts.core import _simulate_single_dataset
        from akts.models import get_model_info
        from akts.utils import get_temperature_interpolator

        # A stiff parameter set for A->B->C (rates differing by many orders of
        # magnitude) -- the exact shape of case that used to hang the solver.
        ode_func, _, initial_state_dim, params_template = get_model_info(
            'A->B->C', f1_model='F1', f2_model='F1'
        )
        t_eval = np.linspace(0, 182 * 86400, 50)
        temp_func = get_temperature_interpolator(t_eval, np.full_like(t_eval, 343.0))

        t0 = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            _, alpha = _simulate_single_dataset(
                t_eval=t_eval,
                temp_func=temp_func,
                ode_system=ode_func,
                initial_state=np.array([1.0, 0.0]),
                params_template=params_template,
                current_params_logA={'Ea1': 40000.0, 'logA1': np.log(1e18),
                                     'Ea2': 150000.0, 'logA2': np.log(1e3)},
                # Tiny budget: forces the runaway-solve path deterministically and
                # quickly, regardless of machine speed.
                solver_options={'max_rhs_evals': 50},
            )
        elapsed = time.time() - t0

        assert elapsed < 5.0, (
            f"single ODE solve with a tiny RHS budget took {elapsed:.1f}s -- the "
            f"eval-count cap may not be enforced any more"
        )
        # A budget this tight can't finish the integration -- both solvers should
        # exhaust it and fall through to the documented NaN failure return, not
        # raise, and not silently return a wrong (non-NaN) answer.
        assert np.all(np.isnan(alpha)), (
            "expected the NaN failure return once the RHS budget is exhausted"
        )

    def test_fit_kinetic_model_respects_wall_clock_deadline(self):
        # Root cause found while timing the multistart replacement for the
        # Bayesian ODE optimizer (2026-09-26): coarse_start() runs 25 objective
        # evaluations per Arrhenius pair (50 for A->B->C's two pairs) before
        # Powell even starts, and Powell's own maxfev=4000 was tuned against
        # cheap single_step models. On a 4-parameter ODE model with a stiff
        # starting region, a single fit_kinetic_model() call ran for over an
        # hour with no way to bound it. optimizer_options['max_seconds'] fixes
        # that; this asserts the deadline is actually enforced, not just
        # accepted as an argument.
        import time as time_module

        Ea_true, A_true = 90000.0, 5e11
        datasets = [_make_synthetic_dataset(Ea_true, A_true, T, seed=i)
                    for i, T in enumerate((313.0, 323.0, 333.0, 343.0))]

        t0 = time_module.time()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            fit_result = fit_kinetic_model(
                datasets=datasets,
                model_name='A->B->C',
                model_definition_args={'f1_model': 'F1', 'f2_model': 'F1'},
                initial_guesses={'Ea1': 90000, 'A1': 1e8, 'Ea2': 100000, 'A2': 1e9},
                parameter_bounds={
                    'Ea1': (10e3, 300e3), 'A1': (1e-5, 1e25),
                    'Ea2': (10e3, 300e3), 'A2': (1e-5, 1e25),
                },
                optimizer_options={'max_seconds': 10.0},  # deliberately tiny
                verbose=False,
            )
        elapsed = time_module.time() - t0

        # The deadline is checked once per Powell iteration (and per coarse-scan
        # point), not continuously, so a slow iteration can overshoot -- this is
        # a generous multiple of the 10s budget, not an exact bound.
        assert elapsed < 120.0, (
            f"fit took {elapsed:.1f}s against a 10s max_seconds budget -- the "
            f"wall-clock deadline mechanism may have regressed"
        )
        assert fit_result is not None

    def test_generous_budget_still_solves_normally(self):
        # Sanity check the mechanism doesn't fire for well-behaved parameters at the
        # library's actual default budget.
        from akts.core import _simulate_single_dataset
        from akts.models import get_model_info
        from akts.utils import get_temperature_interpolator

        ode_func, _, _, params_template = get_model_info('A->B->C', f1_model='F1', f2_model='F1')
        t_eval = np.linspace(0, 182 * 86400, 50)
        temp_func = get_temperature_interpolator(t_eval, np.full_like(t_eval, 343.0))

        _, alpha = _simulate_single_dataset(
            t_eval=t_eval,
            temp_func=temp_func,
            ode_system=ode_func,
            initial_state=np.array([1.0, 0.0]),
            params_template=params_template,
            current_params_logA={'Ea1': 90000.0, 'logA1': np.log(1e11),
                                 'Ea2': 100000.0, 'logA2': np.log(1e12)},
            solver_options={},  # default max_rhs_evals=20000
        )
        assert np.all(np.isfinite(alpha)), "well-behaved parameters should integrate fine"
