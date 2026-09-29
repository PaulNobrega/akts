"""
Tests for different ranking methods in rank_models().

Verifies that BIC-only, AIC-only, Akaike weight, and R²-only
ranking methods work correctly.
"""

import numpy as np
import pytest

from akts import KineticDataset, fit_kinetic_model, rank_models


class TestRankingMethods:
    """Test different ranking methods."""

    @pytest.fixture
    def simple_dataset(self):
        """Create simple F1 dataset for testing."""
        np.random.seed(70)
        t = np.linspace(0, 3600, 30)
        T = np.full_like(t, 323.15)
        k = 1e11 * np.exp(-85000 / (8.314 * 323.15))
        alpha = 1.0 - np.exp(-k * t)
        alpha = alpha + np.random.normal(0, 0.005, len(t))
        alpha = np.clip(alpha, 0, 1)

        return KineticDataset(time=t, temperature=T, conversion=alpha)

    @pytest.fixture
    def fit_results(self, simple_dataset):
        """Fit multiple models to get fit results for ranking."""
        models = ['F0', 'F1', 'F2', 'F3']
        results = []

        for model in models:
            fit_result = fit_kinetic_model(
                datasets=[simple_dataset],
                model_name="single_step",
                model_definition_args={'f_alpha_model': model},
                initial_guesses={'Ea': 85000, 'A': 1e11}
            )
            if fit_result.success:
                results.append(fit_result)

        return results

    def test_combined_ranking_default(self, fit_results):
        """Test default combined ranking method."""
        ranked = rank_models(fit_results, ranking_method='combined')

        # Check all results have scores
        assert all('score' in r for r in ranked)
        assert all('rank' in r for r in ranked)

        # Check ranks are sequential
        ranks = [r['rank'] for r in ranked]
        assert ranks == list(range(1, len(ranked) + 1))

        # Check Akaike weights are present
        assert all('akaike_weight' in r['stats'] for r in ranked)

    def test_bic_ranking(self, fit_results):
        """Test BIC-only ranking."""
        ranked = rank_models(fit_results, ranking_method='bic')

        # Check scores equal BIC values
        for r in ranked:
            assert abs(r['score'] - r['stats']['bic']) < 1e-9

        # Check sorted by BIC (ascending)
        bic_values = [r['stats']['bic'] for r in ranked]
        assert bic_values == sorted(bic_values)

        # Best model should have lowest BIC
        assert ranked[0]['rank'] == 1
        assert ranked[0]['stats']['bic'] == min(bic_values)

    def test_aic_ranking(self, fit_results):
        """Test AIC-only ranking."""
        ranked = rank_models(fit_results, ranking_method='aic')

        # Check scores equal AIC values
        for r in ranked:
            assert abs(r['score'] - r['stats']['aic']) < 1e-9

        # Check sorted by AIC (ascending)
        aic_values = [r['stats']['aic'] for r in ranked]
        assert aic_values == sorted(aic_values)

        # Best model should have lowest AIC
        assert ranked[0]['rank'] == 1
        assert ranked[0]['stats']['aic'] == min(aic_values)

    def test_akaike_weight_ranking(self, fit_results):
        """Test Akaike weight ranking."""
        ranked = rank_models(fit_results, ranking_method='akaike_weight')

        # Check scores are negative Akaike weights
        for r in ranked:
            assert r['score'] == -r['stats']['akaike_weight']

        # Check sorted by Akaike weight (descending, i.e., negative score ascending)
        akaike_weights = [r['stats']['akaike_weight'] for r in ranked]
        assert akaike_weights == sorted(akaike_weights, reverse=True)

        # Best model should have highest Akaike weight
        assert ranked[0]['rank'] == 1
        assert ranked[0]['stats']['akaike_weight'] == max(akaike_weights)

        # Akaike weights should sum to 1
        assert abs(sum(akaike_weights) - 1.0) < 1e-6

    def test_r_squared_ranking(self, fit_results):
        """Test R² only ranking."""
        ranked = rank_models(fit_results, ranking_method='r_squared')

        # Check scores are negative R²
        for r in ranked:
            r2 = r['stats']['r_squared']
            if np.isfinite(r2):
                assert abs(r['score'] + r2) < 1e-9

        # Check sorted by R² (descending, i.e., negative R² ascending)
        r2_values = [r['stats']['r_squared'] for r in ranked if np.isfinite(r['stats']['r_squared'])]
        assert r2_values == sorted(r2_values, reverse=True)

        # Best model should have highest R²
        assert ranked[0]['rank'] == 1
        finite_r2 = [r['stats']['r_squared'] for r in ranked if np.isfinite(r['stats']['r_squared'])]
        assert ranked[0]['stats']['r_squared'] == max(finite_r2)

    def test_invalid_ranking_method(self, fit_results):
        """Test that invalid ranking method raises error."""
        with pytest.raises(ValueError, match="ranking_method must be one of"):
            rank_models(fit_results, ranking_method='invalid')

    def test_ranking_consistency(self, fit_results):
        """Test that different ranking methods can give different orders."""
        ranked_bic = rank_models(fit_results, ranking_method='bic')
        ranked_aic = rank_models(fit_results, ranking_method='aic')
        ranked_r2 = rank_models(fit_results, ranking_method='r_squared')

        # All should have same models
        assert len(ranked_bic) == len(ranked_aic) == len(ranked_r2)
        assert len(ranked_bic) > 0

        # All ranking methods should produce valid ranks
        for ranked in [ranked_bic, ranked_aic, ranked_r2]:
            assert ranked[0]['rank'] == 1
            assert all('score' in r for r in ranked)
            assert all('stats' in r for r in ranked)

    def test_simplicity_penalty_applies_to_all_methods(self, simple_dataset):
        """Test that simplicity penalty applies regardless of ranking method."""
        # Fit F1 and Fn
        fit_f1 = fit_kinetic_model(
            datasets=[simple_dataset],
            model_name="single_step",
            model_definition_args={'f_alpha_model': 'F1'},
            initial_guesses={'Ea': 85000, 'A': 1e11}
        )

        fit_fn = fit_kinetic_model(
            datasets=[simple_dataset],
            model_name="single_step",
            model_definition_args={'f_alpha_model': 'Fn'},
            initial_guesses={'Ea': 85000, 'A': 1e11, 'n': 1.0},
            parameter_bounds={'Ea': (50000, 150000), 'A': (1e7, 1e15), 'n': (0, 5)}
        )

        # Test with different ranking methods
        for method in ['combined', 'bic', 'aic', 'akaike_weight', 'r_squared']:
            ranked = rank_models([fit_f1, fit_fn], ranking_method=method)

            # All should have simplicity_penalty field
            for r in ranked:
                assert 'simplicity_penalty' in r
                assert r['simplicity_penalty'] >= 0

    def test_bic_delta_interpretation(self, fit_results):
        """Test that BIC ranking allows direct ΔBIC interpretation."""
        ranked = rank_models(fit_results, ranking_method='bic')

        if len(ranked) >= 2:
            # Calculate ΔBIC from best model
            best_bic = ranked[0]['stats']['bic']

            for r in ranked[1:]:
                delta_bic = r['stats']['bic'] - best_bic

                # ΔBIC interpretation (Kass & Raftery 1995):
                # 0-2: weak evidence
                # 2-6: positive evidence
                # 6-10: strong evidence
                # >10: very strong evidence

                # Just verify delta is positive (worse models have higher BIC)
                assert delta_bic >= 0, "ΔBIC should be non-negative"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
