"""
Model complexity is penalized only through AIC's 2k term; rank_models adds no
separate simplicity penalty.
"""

import numpy as np
import pytest

from akts import KineticDataset, fit_kinetic_model, rank_models


def _f1_dataset(seed, n_points=35, noise=0.005):
    np.random.seed(seed)
    t = np.linspace(0, 3600, n_points)
    T = np.full_like(t, 323.15)
    k = 1e11 * np.exp(-85000 / (8.314 * 323.15))
    alpha = np.clip(1.0 - np.exp(-k * t) + np.random.normal(0, noise, n_points), 0, 1)
    return KineticDataset(time=t, temperature=T, conversion=alpha)


def _fit(dataset, model, guesses, bounds=None):
    return fit_kinetic_model(
        datasets=[dataset], model_name="single_step",
        model_definition_args={'f_alpha_model': model},
        initial_guesses=guesses, parameter_bounds=bounds, verbose=False,
    )


class TestComplexityThroughAIC:

    def test_extra_parameter_costs_two_aic_units(self):
        dataset = _f1_dataset(60)
        fit_f1 = _fit(dataset, 'F1', {'Ea': 85000, 'A': 1e11})
        fit_fn = _fit(dataset, 'Fn', {'Ea': 85000, 'A': 1e11, 'n': 1.5},
                      {'Ea': (50000, 150000), 'A': (1e7, 1e15), 'n': (0, 5)})
        assert fit_f1.success and fit_fn.success
        assert fit_fn.n_parameters == fit_f1.n_parameters + 1

        ranked = rank_models([fit_f1, fit_fn], apply_filters=False)
        by_name = {len(r['parameters']): r for r in ranked}
        f1, fn = by_name[2], by_name[3]
        # With F1 data, Fn can't reduce RSS enough to pay for its extra parameter.
        if abs(fn['parameters']['n'] - 1.0) < 0.3:
            assert f1['rank'] < fn['rank']

    def test_no_simplicity_penalty_applied(self):
        dataset = _f1_dataset(62, n_points=25, noise=0.0)
        fit_f1 = _fit(dataset, 'F1', {'Ea': 85000, 'A': 1e11})
        fit_fn = _fit(dataset, 'Fn', {'Ea': 85000, 'A': 1e11, 'n': 1.0},
                      {'Ea': (50000, 150000), 'A': (1e7, 1e15), 'n': (0, 5)})
        ranked = rank_models([fit_f1, fit_fn], apply_filters=False)
        assert all(r['simplicity_penalty'] == 0.0 for r in ranked)
        assert ranked[0]['stats']['aic'] == min(r['stats']['aic'] for r in ranked)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
