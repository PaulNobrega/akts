"""
Tests for Akaike-weight ranking in rank_models().
"""

import numpy as np
import pytest

from akts import KineticDataset, fit_kinetic_model, rank_models
from akts.datatypes import FitResult


def _mock_fit(name, aic, r_squared=0.95, n_params=2, plausible=True, n_points=50):
    return FitResult(
        model_name=name, parameters={'Ea': 80000, 'A': 1e10}, success=True, message='mock',
        rss=0.01, r_squared=r_squared, aic=aic, bic=aic + 5.0, n_parameters=n_params,
        n_datapoints=n_points, is_physically_plausible=plausible, model_definition_args={},
    )


class TestAkaikeRanking:

    @pytest.fixture
    def fit_results(self):
        np.random.seed(70)
        t = np.linspace(0, 3600, 30)
        T = np.full_like(t, 323.15)
        k = 1e11 * np.exp(-85000 / (8.314 * 323.15))
        alpha = np.clip(1.0 - np.exp(-k * t) + np.random.normal(0, 0.005, len(t)), 0, 1)
        dataset = KineticDataset(time=t, temperature=T, conversion=alpha)
        results = []
        for model in ['F0', 'F1', 'F2', 'F3']:
            fit = fit_kinetic_model(
                datasets=[dataset], model_name="single_step",
                model_definition_args={'f_alpha_model': model},
                initial_guesses={'Ea': 85000, 'A': 1e11}, verbose=False,
            )
            if fit.success:
                results.append(fit)
        return results

    def test_sorted_by_akaike_weight(self, fit_results):
        ranked = rank_models(fit_results, apply_filters=False)
        weights = [r['stats']['akaike_weight'] for r in ranked]
        assert weights == sorted(weights, reverse=True)
        assert [r['rank'] for r in ranked] == list(range(1, len(ranked) + 1))
        assert abs(sum(weights) - 1.0) < 1e-6

    def test_score_is_negative_akaike_weight(self, fit_results):
        ranked = rank_models(fit_results, apply_filters=False)
        for r in ranked:
            assert r['score'] == pytest.approx(-r['stats']['akaike_weight'])

    def test_rank_one_has_lowest_aic(self, fit_results):
        ranked = rank_models(fit_results, apply_filters=False)
        assert ranked[0]['stats']['aic'] == min(r['stats']['aic'] for r in ranked)

    def test_no_extra_simplicity_penalty(self):
        # AIC already penalizes parameters; a complex model with far better AIC must win.
        ranked = rank_models([_mock_fit('SB_m0_n1_model', aic=-280.0),
                              _mock_fit('SB_m3_n3_model', aic=-400.0)], apply_filters=False)
        assert ranked[0]['model_name'] == 'SB_m3_n3_model'
        assert all(r['simplicity_penalty'] == 0.0 for r in ranked)

    def test_removed_ranking_method_argument_rejected(self, fit_results):
        with pytest.raises(TypeError):
            rank_models(fit_results, ranking_method='bic')


class TestRankingFilters:

    def test_low_r_squared_filtered(self):
        ranked = rank_models([_mock_fit('good', aic=-100.0, r_squared=0.90),
                              _mock_fit('poor', aic=-200.0, r_squared=0.50)])
        assert [r['model_name'] for r in ranked] == ['good']

    def test_implausible_filtered_when_plausible_alternative_exists(self):
        ranked = rank_models([_mock_fit('plausible', aic=-100.0),
                              _mock_fit('implausible', aic=-200.0, plausible=False)])
        assert [r['model_name'] for r in ranked] == ['plausible']
        assert 'filter_warning' not in ranked[0]

    def test_implausible_kept_with_warning_when_no_plausible_model(self):
        with pytest.warns(UserWarning):
            ranked = rank_models([_mock_fit('a', aic=-100.0, plausible=False),
                                  _mock_fit('b', aic=-200.0, plausible=False)])
        assert ranked[0]['model_name'] == 'b'
        assert ranked[0]['filter_warning']['type'] == 'no_plausible_models'

    def test_all_kept_with_critical_warning_when_no_model_passes_r_squared(self):
        with pytest.warns(UserWarning):
            ranked = rank_models([_mock_fit('a', aic=-100.0, r_squared=0.4),
                                  _mock_fit('b', aic=-150.0, r_squared=0.5)])
        assert len(ranked) == 2
        assert ranked[0]['filter_warning']['type'] == 'no_good_models'

    def test_min_r_squared_threshold_configurable(self):
        fits = [_mock_fit('a', aic=-100.0, r_squared=0.80), _mock_fit('b', aic=-200.0, r_squared=0.75)]
        assert {r['model_name'] for r in rank_models(fits, min_r_squared=0.70)} == {'a', 'b'}
        assert {r['model_name'] for r in rank_models(fits, min_r_squared=0.78)} == {'a'}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
