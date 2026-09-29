"""
Tests for the model-selection/validation additions to akts.core.rank_models():
Akaike weights, adjusted R^2, and RMSE (TODO.md §6b "Model selection and
validation"). These are purely additive stats fields -- they must not change
the existing 'score'-based ranking/ordering behavior.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from akts.datatypes import FitResult
from akts.core import rank_models
from akts.utils import calculate_akaike_weights, calculate_adjusted_r_squared


def _fit(name, rss, n_params, n_points, r_squared, aic, bic):
    return FitResult(
        model_name=name, parameters={}, success=True, message='',
        rss=rss, n_datapoints=n_points, n_parameters=n_params,
        r_squared=r_squared, aic=aic, bic=bic,
    )


class TestAkaikeWeights:
    def test_weights_sum_to_one(self):
        weights = calculate_akaike_weights([-140.0, -125.0, -120.0])
        assert pytest.approx(sum(weights), abs=1e-9) == 1.0

    def test_best_aicc_gets_most_weight(self):
        weights = calculate_akaike_weights([-140.0, -125.0, -120.0])
        assert weights[0] == max(weights)

    def test_identical_aicc_gives_equal_weights(self):
        weights = calculate_akaike_weights([-100.0, -100.0, -100.0])
        assert weights == pytest.approx([1 / 3, 1 / 3, 1 / 3])

    def test_non_finite_aicc_gets_zero_weight(self):
        weights = calculate_akaike_weights([-140.0, np.inf, -120.0])
        assert weights[1] == 0.0
        assert pytest.approx(sum(weights), abs=1e-9) == 1.0

    def test_all_non_finite_returns_all_zero(self):
        weights = calculate_akaike_weights([np.inf, np.inf])
        assert weights == [0.0, 0.0]


class TestAdjustedRSquared:
    def test_penalizes_extra_parameters_for_same_raw_r2(self):
        # Same raw R^2, more parameters -> lower adjusted R^2.
        adj_simple = calculate_adjusted_r_squared(0.95, n_params=2, n_datapoints=40)
        adj_complex = calculate_adjusted_r_squared(0.95, n_params=6, n_datapoints=40)
        assert adj_complex < adj_simple

    def test_matches_raw_r2_at_n_minus_2_params_limit_behavior(self):
        # Sanity: adjusted R^2 should be close to (but not above) raw R^2 for a
        # large n relative to k (little penalty when data is abundant).
        adj = calculate_adjusted_r_squared(0.90, n_params=2, n_datapoints=1000)
        assert 0.89 < adj <= 0.90

    def test_undefined_when_too_few_points(self):
        # n_datapoints <= n_params + 1 makes the adjustment undefined.
        assert np.isnan(calculate_adjusted_r_squared(0.9, n_params=4, n_datapoints=5))

    def test_non_finite_r2_returns_nan(self):
        assert np.isnan(calculate_adjusted_r_squared(np.nan, n_params=2, n_datapoints=40))
        assert np.isnan(calculate_adjusted_r_squared(-np.inf, n_params=2, n_datapoints=40))


class TestRankModelsAdditiveStats:
    """rank_models() must keep its existing ordering/scoring behavior; the new
    fields are additive only."""

    def _sample_fits(self):
        return [
            _fit('F1', rss=0.05, n_params=2, n_points=40, r_squared=0.95, aic=-120.0, bic=-115.0),
            _fit('F2', rss=0.04, n_params=2, n_points=40, r_squared=0.96, aic=-125.0, bic=-120.0),
            _fit('ABC', rss=0.01, n_params=4, n_points=40, r_squared=0.99, aic=-140.0, bic=-128.0),
        ]

    def test_new_fields_present_and_valid(self):
        ranked = rank_models(self._sample_fits())
        for item in ranked:
            stats = item['stats']
            assert 'akaike_weight' in stats
            assert 'r_squared_adj' in stats
            assert 'rmse' in stats
            assert 0.0 <= stats['akaike_weight'] <= 1.0
            assert stats['rmse'] >= 0.0

    def test_akaike_weights_sum_to_one_across_ranked_models(self):
        ranked = rank_models(self._sample_fits())
        total = sum(item['stats']['akaike_weight'] for item in ranked)
        assert pytest.approx(total, abs=1e-9) == 1.0

    def test_ranking_order_unchanged_by_new_fields(self):
        # This is the regression guard: adding the new stats must not perturb
        # which model wins or the rank ordering, since 'score' still drives it.
        ranked = rank_models(self._sample_fits())
        assert [item['model_name'] for item in ranked] == ['ABC', 'F2', 'F1']
        assert [item['rank'] for item in ranked] == [1, 2, 3]

    def test_score_weights_parameter_still_respected(self):
        # Existing public parameter; must still work after the additive change.
        fits = self._sample_fits()
        ranked_default = rank_models(fits)
        ranked_r2_only = rank_models(fits, score_weights={'bic': 0.0, 'r_squared': 1.0, 'rss': 0.0, 'n_params': 0.0})
        # Both should still be valid, ordered rankings (not asserting they differ,
        # since ABC also wins on R^2 alone here -- just that the call succeeds and
        # produces a complete ranking).
        assert len(ranked_default) == len(ranked_r2_only) == 3
        assert {item['rank'] for item in ranked_r2_only} == {1, 2, 3}
