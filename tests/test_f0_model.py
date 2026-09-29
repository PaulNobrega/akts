"""
Tests for the F0 (zero-order) kinetic model (TODO.md §6b "Models AKTS offers
that akts lacks"). F0 reuses f_n_order with n=0 -- f(alpha) = (1-alpha)^0 = 1,
i.e. a constant rate until conversion completes. Used for controlled-release
and other constant-rate degradation mechanisms.
"""
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from akts import KineticDataset, fit_kinetic_model, simulate_kinetics
from akts.models import F_ALPHA_MODELS


class TestF0FAlpha:
    def test_f_alpha_is_constant_one_below_full_conversion(self):
        f0 = F_ALPHA_MODELS['F0']
        for alpha in (0.0, 0.3, 0.5, 0.9, 0.999):
            assert f0(alpha, {'n': 0.0}) == 1.0

    def test_f_alpha_zero_at_full_conversion(self):
        f0 = F_ALPHA_MODELS['F0']
        assert f0(1.0, {'n': 0.0}) == 0.0


class TestF0Simulation:
    def test_conversion_is_linear_in_time(self):
        # The defining signature of zero-order kinetics: constant dalpha/dt.
        t_sec = np.linspace(0, 5 * 86400, 30)
        result = simulate_kinetics(
            model_name='single_step',
            model_definition_args={'f_alpha_model': 'F0'},
            kinetic_params={'Ea': 90000, 'A': 1e8},
            initial_alpha=1e-6,
            temperature_program=(t_sec, np.full_like(t_sec, 343.0)),
        )
        diffs = np.diff(result.conversion)
        assert np.allclose(diffs, diffs[0], rtol=0.02), (
            "F0 conversion should increase linearly with time (constant rate) "
            "while unsaturated"
        )

    def test_saturates_at_full_conversion(self):
        t_sec = np.linspace(0, 100 * 86400, 30)  # long enough to fully react
        result = simulate_kinetics(
            model_name='single_step',
            model_definition_args={'f_alpha_model': 'F0'},
            kinetic_params={'Ea': 90000, 'A': 1e11},
            initial_alpha=1e-6,
            temperature_program=(t_sec, np.full_like(t_sec, 343.0)),
        )
        assert result.conversion[-1] <= 1.0
        assert np.isclose(result.conversion[-1], 1.0, atol=1e-3)


class TestF0Fit:
    def test_recovers_known_ea_from_synthetic_data(self):
        Ea_true, A_true = 85000.0, 5e9
        datasets = []
        for i, T in enumerate((313.0, 323.0, 333.0)):
            t_sec = np.linspace(0, 20 * 86400, 20)
            sim = simulate_kinetics(
                'single_step', {'f_alpha_model': 'F0'}, {'Ea': Ea_true, 'A': A_true},
                1e-6, (t_sec, np.full_like(t_sec, T)),
            )
            rng = np.random.default_rng(i)
            conv = np.clip(sim.conversion + rng.normal(0, 0.01, size=sim.conversion.shape), 0, 1)
            datasets.append(KineticDataset(time=t_sec, temperature=np.full_like(t_sec, T), conversion=conv))

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            fit_result = fit_kinetic_model(
                datasets=datasets,
                model_name='single_step',
                model_definition_args={'f_alpha_model': 'F0'},
                initial_guesses={'Ea': 80000, 'A': 1e8},
                parameter_bounds={'Ea': (10e3, 300e3), 'A': (1e-5, 1e25)},
                verbose=False,
            )

        assert fit_result.success
        assert fit_result.r_squared > 0.95
        fitted_Ea = fit_result.parameters['Ea']
        assert abs(fitted_Ea - Ea_true) / Ea_true < 0.2, (
            f"fitted Ea={fitted_Ea/1000:.1f} kJ/mol vs true {Ea_true/1000:.1f} kJ/mol"
        )
