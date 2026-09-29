"""
Tests for model-free (Friedman isoconversional) prediction (TODO.md §6b,
"Model-free (isoconversional) prediction -- the core AKTS method").

Covers:
1. IsoResult.ln_A_f_alpha populated by run_friedman() (promoted from the
   regression intercept, previously discarded).
2. Spline-based smoothing in _prepare_iso_data/_spline_dadt_at_alpha fixes
   Friedman on sparse, noisy isothermal data -- raw finite differencing there
   swung wildly (verified: Ea recovered from 24.9 to 137.4 kJ/mol against a
   true value of 90.0 kJ/mol on the same synthetic case used below).
3. core.predict_conversion_model_free(): general ODE-based prediction from an
   IsoResult, dispatched automatically from predict_conversion(iso_result, ...).
4. helpers.time_to_conversion_model_free(): closed-form isothermal quadrature
   shortcut, cross-checked against (3) by forward-simulating to the returned
   time.
5. run_kas()/run_ofw() (integral methods) correctly leave ln_A_f_alpha=None --
   their regression intercepts don't have the ln[A*f(alpha)] interpretation.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from akts import KineticDataset, simulate_kinetics, predict_conversion, predict_conversion_model_free
from akts.isoconversional import run_friedman, run_kas, run_ofw
from akts.helpers import time_to_conversion_model_free


R_GAS = 8.314


def _make_isothermal_datasets(Ea, A, temps_K, t_end_days=182, n_points=27, noise_std=0.005, seed_offset=0):
    """Multi-temperature isothermal stability-study-style synthetic data."""
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


def _make_heating_rate_datasets(Ea, A, betas_K_per_min, T0=300.0, n_points=60):
    """
    DSC/TGA-style constant-heating-rate synthetic data (Friedman's original use
    case). Each dataset runs long enough to fully react (t_end scales inversely
    with beta) so every alpha level is reached at every rate with a wide,
    well-separated temperature spread at each alpha -- a narrow/ill-conditioned
    T-spread (e.g. rates too close together, or one dataset ending before it
    reaches the target alpha) starves the per-alpha Arrhenius regression of
    real leverage and any differentiation method (old or new) recovers Ea
    poorly, which isn't what this test is checking.
    """
    datasets = []
    for beta in betas_K_per_min:
        t_end_min = 400.0 / beta  # same total temperature rise (K) at every rate
        t_min = np.linspace(0, t_end_min, n_points)
        T = T0 + beta * t_min
        sim = simulate_kinetics(
            model_name='single_step', model_definition_args={'f_alpha_model': 'F1'},
            kinetic_params={'Ea': Ea, 'A': A}, initial_alpha=1e-6,
            temperature_program=(t_min * 60, T),
        )
        datasets.append(KineticDataset(time=t_min * 60, temperature=T, conversion=sim.conversion,
                                       heating_rate=beta / 60))
    return datasets


# Shared ground truth for the isothermal case, reused across several tests.
EA_TRUE, A_TRUE = 90000.0, 1e6
TEMPS = (313.0, 323.0, 333.0, 343.0)
ALPHA_LEVELS = np.linspace(0.01, 0.10, 10)


class TestFriedmanOnIsothermalData:
    """Regression coverage for the spline-smoothing fix -- the actual blocker
    found this session that made items further down this file possible."""

    def test_recovers_ea_within_tolerance(self):
        datasets = _make_isothermal_datasets(EA_TRUE, A_TRUE, TEMPS)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = run_friedman(datasets, alpha_levels=ALPHA_LEVELS)

        finite = np.isfinite(result.Ea)
        assert finite.sum() >= 5, "expected Friedman to resolve Ea at most of the requested alpha levels"
        # Generous tolerance: recovering a global "true" Ea from noisy per-alpha
        # regressions on sparse data is not going to be exact. The point of this
        # test is "no longer wildly wrong" (raw differencing gave 24.9-137.4
        # kJ/mol against a true 90.0), not "recovers with model-fit precision".
        recovered = result.Ea[finite]
        assert np.all(np.abs(recovered - EA_TRUE) / EA_TRUE < 0.25), (
            f"Ea(alpha) = {np.round(recovered/1000, 1)} kJ/mol strayed >25% from "
            f"true {EA_TRUE/1000:.1f} kJ/mol -- spline smoothing fix may have regressed"
        )

    def test_ln_a_f_alpha_populated(self):
        datasets = _make_isothermal_datasets(EA_TRUE, A_TRUE, TEMPS)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = run_friedman(datasets, alpha_levels=ALPHA_LEVELS)
        assert result.ln_A_f_alpha is not None
        assert np.isfinite(result.ln_A_f_alpha).sum() >= 5

    def test_still_works_on_heating_rate_data(self):
        # Non-isothermal (DSC/TGA-style) data is Friedman's original intended
        # use case -- the spline change must not regress it.
        datasets = _make_heating_rate_datasets(EA_TRUE, 1e11, (5.0, 10.0, 20.0))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = run_friedman(datasets, alpha_levels=np.linspace(0.1, 0.8, 8))
        finite = np.isfinite(result.Ea)
        assert finite.sum() >= 5
        recovered = result.Ea[finite]
        assert np.all(np.abs(recovered - EA_TRUE) / EA_TRUE < 0.25)

    def test_beyond_data_range_is_nan_not_extrapolated(self):
        # Alpha levels no dataset ever reaches should stay NaN, not silently
        # extrapolate past the measured data.
        datasets = _make_isothermal_datasets(EA_TRUE, A_TRUE, TEMPS)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = run_friedman(datasets, alpha_levels=np.array([0.5, 0.7, 0.9]))
        # None of these datasets get anywhere near 50-90% conversion in 182 days
        # at these temperatures -- see the per-temperature max-conversion values
        # from this session's exploration (313K maxed out under 2%).
        assert np.all(np.isnan(result.Ea))


class TestKasOfwUnaffected:
    """KAS/OFW are integral methods -- their regression intercepts don't mean
    ln[A*f(alpha)], so they must not have inherited a value from the IsoResult
    field addition."""

    def test_kas_leaves_ln_a_f_alpha_none(self):
        datasets = _make_heating_rate_datasets(EA_TRUE, 1e11, (5.0, 10.0, 20.0))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = run_kas(datasets, alpha_levels=np.linspace(0.1, 0.8, 8))
        assert result.ln_A_f_alpha is None

    def test_ofw_leaves_ln_a_f_alpha_none(self):
        datasets = _make_heating_rate_datasets(EA_TRUE, 1e11, (5.0, 10.0, 20.0))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = run_ofw(datasets, alpha_levels=np.linspace(0.1, 0.8, 8))
        assert result.ln_A_f_alpha is None

    def test_kas_still_recovers_reasonable_ea(self):
        # Sanity check the field addition didn't disturb KAS's own Ea calc.
        datasets = _make_heating_rate_datasets(EA_TRUE, 1e11, (5.0, 10.0, 20.0))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = run_kas(datasets, alpha_levels=np.linspace(0.3, 0.6, 4))
        finite = np.isfinite(result.Ea)
        assert finite.sum() >= 2
        assert np.all(np.abs(result.Ea[finite] - EA_TRUE) / EA_TRUE < 0.3)


class TestPredictConversionModelFree:
    def _fit_friedman(self):
        datasets = _make_isothermal_datasets(EA_TRUE, A_TRUE, TEMPS)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return run_friedman(datasets, alpha_levels=ALPHA_LEVELS)

    def test_predicts_in_sane_range_vs_true_parameters(self):
        iso = self._fit_friedman()
        t_eval = np.linspace(0, 2 * 365 * 86400, 50)

        pred = predict_conversion_model_free(
            iso, temperature_program=lambda t: 298.15, simulation_time_sec=t_eval
        )
        true_sim = simulate_kinetics(
            'single_step', {'f_alpha_model': 'F1'}, {'Ea': EA_TRUE, 'A': A_TRUE}, 0.0,
            (t_eval, np.full_like(t_eval, 298.15)),
        )
        # Not claiming precision recovery from noisy data -- checking the method
        # is unbiased (same order of magnitude), not exact.
        assert 0.3 < pred.conversion[-1] / true_sim.conversion[-1] < 3.0
        assert np.all(np.isfinite(pred.conversion))
        assert np.all((pred.conversion >= 0) & (pred.conversion <= 1))

    def test_dispatches_from_predict_conversion(self):
        iso = self._fit_friedman()
        t_eval = np.linspace(0, 365 * 86400, 30)
        direct = predict_conversion_model_free(iso, temperature_program=lambda t: 298.15, simulation_time_sec=t_eval)
        dispatched = predict_conversion(iso, temperature_program=lambda t: 298.15, simulation_time_sec=t_eval)
        np.testing.assert_array_equal(direct.conversion, dispatched.conversion)

    def test_kas_result_falls_back_to_warning(self):
        datasets = _make_heating_rate_datasets(EA_TRUE, 1e11, (5.0, 10.0, 20.0))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            iso_kas = run_kas(datasets, alpha_levels=np.linspace(0.1, 0.8, 8))
        with pytest.warns(UserWarning):
            result = predict_conversion(iso_kas, temperature_program=lambda t: 298.15,
                                        simulation_time_sec=np.array([0, 100]))
        assert len(result.conversion) == 0

    def test_rejects_iso_result_missing_ln_a_f_alpha(self):
        iso = self._fit_friedman()
        iso.ln_A_f_alpha = None
        with pytest.raises(ValueError):
            predict_conversion_model_free(iso, temperature_program=lambda t: 298.15,
                                          simulation_time_sec=np.array([0, 100]))


class TestTimeToConversionModelFree:
    def _fit_friedman(self):
        datasets = _make_isothermal_datasets(EA_TRUE, A_TRUE, TEMPS)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return run_friedman(datasets, alpha_levels=ALPHA_LEVELS)

    def test_round_trips_with_predict_conversion_model_free(self):
        iso = self._fit_friedman()
        t = time_to_conversion_model_free(iso, target_conversion=0.05, temperature_K=298.15)
        assert t is not None and t > 0

        check = predict_conversion_model_free(
            iso, temperature_program=lambda tt: 298.15, simulation_time_sec=np.array([0.0, t])
        )
        assert np.isclose(check.conversion[-1], 0.05, rtol=1e-2)

    def test_higher_target_takes_longer(self):
        iso = self._fit_friedman()
        t_low = time_to_conversion_model_free(iso, 0.02, temperature_K=298.15)
        t_high = time_to_conversion_model_free(iso, 0.08, temperature_K=298.15)
        assert t_high > t_low

    def test_target_at_or_below_initial_alpha_is_zero(self):
        iso = self._fit_friedman()
        assert time_to_conversion_model_free(iso, 0.005, temperature_K=298.15, initial_alpha=0.01) == 0.0

    def test_rejects_invalid_target(self):
        iso = self._fit_friedman()
        with pytest.raises(ValueError):
            time_to_conversion_model_free(iso, target_conversion=0.0, temperature_K=298.15)
        with pytest.raises(ValueError):
            time_to_conversion_model_free(iso, target_conversion=1.5, temperature_K=298.15)

    def test_rejects_iso_result_missing_ln_a_f_alpha(self):
        iso = self._fit_friedman()
        iso.ln_A_f_alpha = None
        with pytest.raises(ValueError):
            time_to_conversion_model_free(iso, target_conversion=0.05, temperature_K=298.15)
