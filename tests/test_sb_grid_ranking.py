"""
Tests for SB grid model ranking under Akaike-weight ranking.
"""
import numpy as np
import pytest
from akts.datatypes import FitResult
from akts.ranking import rank_models


def create_mock_fit_result(model_name, aic, r_squared=0.95, n_params=2, parameters=None):
    return FitResult(
        model_name=model_name,
        parameters=parameters or {'Ea': 80000, 'A': 1e10},
        success=True, message='Test fit', rss=0.01, r_squared=r_squared,
        aic=aic, bic=aic + (np.log(50) - 2) * n_params,
        n_parameters=n_params, n_datapoints=50, durbin_watson=2.0, model_definition_args={},
    )


def test_identical_aic_gives_equal_weights():
    ranked = rank_models([create_mock_fit_result('SB_m0_n1_model', aic=100.0),
                          create_mock_fit_result('SB_m2_n3_model', aic=100.0)])
    weights = [r['stats']['akaike_weight'] for r in ranked]
    assert weights == pytest.approx([0.5, 0.5])


def test_better_aic_wins_regardless_of_exponents():
    ranked = rank_models([create_mock_fit_result('SB_m0_n1_model', aic=120.0),
                          create_mock_fit_result('SB_m2_n3_model', aic=100.0)])
    assert [r['model_name'] for r in ranked] == ['SB_m2_n3_model', 'SB_m0_n1_model']
    assert ranked[0]['stats']['akaike_weight'] > 0.99


def test_small_aic_difference_shares_weight():
    ranked = rank_models([create_mock_fit_result('SB_m0_n1_model', aic=100.0),
                          create_mock_fit_result('SB_m1_n2_model', aic=101.0)])
    assert ranked[0]['model_name'] == 'SB_m0_n1_model'
    w = ranked[0]['stats']['akaike_weight']
    assert w == pytest.approx(1.0 / (1.0 + np.exp(-0.5)))


def test_extra_parameters_penalized_through_aic_only():
    grid = create_mock_fit_result('SB_m1_n1_model', aic=100.0)
    cont = create_mock_fit_result('SB_model', aic=104.0, n_params=4,
                                  parameters={'Ea': 80000, 'A': 1e10, 'm': 1.5, 'n': 1.5})
    ranked = rank_models([grid, cont])
    assert ranked[0]['model_name'] == 'SB_m1_n1_model'
    assert all(r['simplicity_penalty'] == 0.0 for r in ranked)


def test_multiple_grid_models_ordered_by_aic():
    fits = [
        create_mock_fit_result('SB_m0_n1_model', aic=95.0),
        create_mock_fit_result('SB_m1_n1_model', aic=96.0),
        create_mock_fit_result('SB_m0_n2_model', aic=95.5),
        create_mock_fit_result('SB_m3_n3_model', aic=100.0),
        create_mock_fit_result('SB_m0_n0_model', aic=98.0),
    ]
    ranked = rank_models(fits)
    assert [r['model_name'] for r in ranked] == [
        'SB_m0_n1_model', 'SB_m0_n2_model', 'SB_m1_n1_model', 'SB_m0_n0_model', 'SB_m3_n3_model']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
