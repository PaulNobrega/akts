"""Two-step Sestak-Berggren (SB2) model: dα/dt = k1·α^m1·(1-α)^n1 + k2·α^m2·(1-α)^n2."""
import numpy as np
import pytest

from akts import KineticDataset, simulate_kinetics, models, generate_sb2_grid_models
from akts.helpers import _setup_model_configs, _fit_sb2
from akts.models import get_model_info, parse_sb2_model, model_display_name

TRUE = {'Ea1': 300e3, 'A1': np.exp(300e3 / (8.314 * 323.15)) * 2e-6, 'm1': 0.5, 'n1': 2.0,
        'Ea2': 70e3, 'A2': np.exp(70e3 / (8.314 * 323.15)) * 2e-8, 'm2': 0.0, 'n2': 1.0}


def _datasets(noise=0.001, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 40 * 86400, 12)
    out = []
    for T in (298.15, 313.15, 323.15, 328.15):
        a = simulate_kinetics('SB2', {'sb2_params': {}}, TRUE, 0.0, lambda _t, T=T: T, t).conversion
        out.append(KineticDataset(time=t, temperature=np.full_like(t, T),
                                  conversion=np.clip(a + rng.normal(0, noise, t.size), 0, 1)))
    return out


def test_grid_names_are_unordered_pairs():
    grid = generate_sb2_grid_models()
    assert len(grid) == 16 * 17 // 2
    assert len(set(grid)) == len(grid)
    assert parse_sb2_model('SB2_m0n3_m1n2_model') == (0, 3, 1, 2)
    assert parse_sb2_model('SB_m0_n3') is None
    assert set(models.kinetic.SB2_grid) <= set(models.all) and 'SB2' in models.all


def test_grid_model_fixes_shapes_and_fits_four_params():
    _, names, _, template = get_model_info('SB2', sb2_params={'m1': 0.0, 'n1': 3.0, 'm2': 1.0, 'n2': 2.0})
    assert names == ['Ea1', 'A1', 'Ea2', 'A2']
    assert template == {'m1': 0.0, 'n1': 3.0, 'm2': 1.0, 'n2': 2.0}
    _, names, _, _ = get_model_info('SB2', sb2_params={})
    assert names == ['Ea1', 'A1', 'Ea2', 'A2', 'm1', 'n1', 'm2', 'n2']
    assert 'm1=0' in model_display_name('SB2_m0n3_m1n2_model')


def test_sb2_reduces_to_single_step_when_one_step_off():
    t = np.linspace(0, 3600, 20)
    params = {'Ea1': 80e3, 'A1': 1e11, 'Ea2': 80e3, 'A2': 1e-30, 'm1': 0.0, 'n1': 1.0, 'm2': 0.0, 'n2': 1.0}
    sb2 = simulate_kinetics('SB2', {'sb2_params': {}}, params, 0.0, lambda _t: 323.15, t).conversion
    f1 = simulate_kinetics('single_step', {'f_alpha_model': 'F1'}, {'Ea': 80e3, 'A': 1e11}, 0.0,
                           lambda _t: 323.15, t).conversion
    np.testing.assert_allclose(sb2, f1, atol=1e-4)


def test_continuous_sb2_recovers_two_step_kinetics():
    datasets = _datasets()
    cfg, guesses, bounds = _setup_model_configs(['SB2'], {'Ea': 80000, 'A': 1e12}, 1.0)
    fit = _fit_sb2(datasets, cfg[0]['def_args'], guesses[cfg[0]['name']], bounds[cfg[0]['name']])
    assert fit.success
    assert fit.r_squared > 0.999
    eas = sorted([fit.parameters['Ea1'], fit.parameters['Ea2']])
    assert eas[0] == pytest.approx(70e3, rel=0.3)
    assert eas[1] == pytest.approx(300e3, rel=0.3)
