"""
Test simplicity penalty in model ranking.

Verifies that simpler models (lower fitted parameters) are preferred
when fit quality is similar.
"""

import numpy as np
import pytest

from akts import KineticDataset, fit_kinetic_model, rank_models


class TestSimplicityPenalty:
    """Test that simpler models are preferred when fit quality is similar."""

    def test_fn_prefers_lower_n_when_similar_fit(self):
        """Fn with n≈1 should rank higher than n≈3 if R² is similar."""
        np.random.seed(60)

        # Generate F1 data (n=1) with slight noise
        t = np.linspace(0, 3600, 35)
        T = np.full_like(t, 323.15)
        k_true = 1e11 * np.exp(-85000 / (8.314 * 323.15))
        alpha_true = 1.0 - np.exp(-k_true * t)
        alpha = alpha_true + np.random.normal(0, 0.005, len(t))
        alpha = np.clip(alpha, 0, 1)

        dataset = KineticDataset(time=t, temperature=T, conversion=alpha)

        # Fit F1 (fixed n=1)
        fit_f1 = fit_kinetic_model(
            datasets=[dataset],
            model_name="single_step",
            model_definition_args={'f_alpha_model': 'F1'},
            initial_guesses={'Ea': 85000, 'A': 1e11}
        )

        # Fit Fn (n as fitted parameter)
        fit_fn = fit_kinetic_model(
            datasets=[dataset],
            model_name="single_step",
            model_definition_args={'f_alpha_model': 'Fn'},
            initial_guesses={'Ea': 85000, 'A': 1e11, 'n': 1.5},
            parameter_bounds={'Ea': (50000, 150000), 'A': (1e7, 1e15), 'n': (0, 5)}
        )

        # Both should succeed
        assert fit_f1.success
        assert fit_fn.success

        # Rank models
        ranked = rank_models([fit_f1, fit_fn])

        # If Fn fitted n close to 1, both should have similar R²
        # But F1 should rank higher (lower score) due to simplicity (fewer parameters)
        # OR if Fn fitted n>1, simplicity penalty should make F1 rank higher

        # Check that simplicity penalty was applied to Fn
        fn_result = next(r for r in ranked if 'n' in r['parameters'])
        assert 'simplicity_penalty' in fn_result

        # If fitted n is close to 1 (say within 0.5), F1 should still rank better
        # due to having fewer parameters
        fitted_n = fn_result['parameters']['n']

        if abs(fitted_n - 1.0) < 0.3:
            # Very close to F1, so F1 should rank higher (fewer params)
            f1_result = next(r for r in ranked if 'n' not in r['parameters'])
            assert f1_result['rank'] <= fn_result['rank'], \
                "F1 should rank equal or higher when fitted n≈1 (simpler model)"

    def test_sb_prefers_lower_exponents(self):
        """SB with m,n near standard values should rank higher than extreme values."""
        np.random.seed(61)

        # Generate SB_mn data (m=0.5, n=1.0)
        t = np.linspace(0, 10000, 40)
        T = np.full_like(t, 333.15)

        k_true = 1e12 * np.exp(-100000 / (8.314 * 333.15))
        dt = t[1] - t[0]
        alpha_true = np.zeros_like(t)
        alpha_true[0] = 0.001
        for i in range(1, len(t)):
            if alpha_true[i-1] >= 0.999:
                alpha_true[i:] = 1.0
                break
            f_alpha = (alpha_true[i-1]**0.5) * ((1 - alpha_true[i-1])**1.0)
            dalpha = k_true * f_alpha * dt
            alpha_true[i] = min(alpha_true[i-1] + dalpha, 1.0)

        alpha = alpha_true + np.random.normal(0, 0.01, len(t))
        alpha = np.clip(alpha, 0.001, 0.999)

        dataset = KineticDataset(time=t, temperature=T, conversion=alpha)

        # Fit SB_mn (fixed m=0.5, n=1.0)
        fit_sb_mn = fit_kinetic_model(
            datasets=[dataset],
            model_name="single_step",
            model_definition_args={'f_alpha_model': 'SB_mn'},
            initial_guesses={'Ea': 100000, 'A': 1e12}
        )

        # Fit SB (fitted m, n)
        fit_sb = fit_kinetic_model(
            datasets=[dataset],
            model_name="single_step",
            model_definition_args={'f_alpha_model': 'SB'},
            initial_guesses={'Ea': 100000, 'A': 1e12, 'm': 0.5, 'n': 1.0},
            parameter_bounds={
                'Ea': (50000, 200000),
                'A': (1e8, 1e16),
                'm': (0, 3),
                'n': (0, 3)
            }
        )

        assert fit_sb_mn.success
        assert fit_sb.success

        # Rank models
        ranked = rank_models([fit_sb_mn, fit_sb])

        # SB should have simplicity penalty
        sb_result = next(r for r in ranked if 'm' in r['parameters'])
        assert 'simplicity_penalty' in sb_result

        # If fitted m,n are close to 0.5, 1.0, simplicity penalty should be small
        fitted_m = sb_result['parameters']['m']
        fitted_n = sb_result['parameters']['n']

        simplicity_penalty = sb_result['simplicity_penalty']

        # Penalty should scale with distance from standard values
        expected_penalty_approx = 0.01 * (abs(fitted_m - 0.5) + abs(fitted_n - 1.0))
        assert abs(simplicity_penalty - expected_penalty_approx) < 0.005, \
            f"Simplicity penalty {simplicity_penalty} doesn't match expected {expected_penalty_approx}"

    def test_simplicity_penalty_transparency(self):
        """Verify simplicity_penalty is included in ranked results."""
        np.random.seed(62)

        # Simple F1 data
        t = np.linspace(0, 3600, 25)
        T = np.full_like(t, 323.15)
        k = 1e11 * np.exp(-85000 / (8.314 * 323.15))
        alpha = 1.0 - np.exp(-k * t)

        dataset = KineticDataset(time=t, temperature=T, conversion=alpha)

        # Fit multiple models
        fit_f1 = fit_kinetic_model(
            datasets=[dataset],
            model_name="single_step",
            model_definition_args={'f_alpha_model': 'F1'},
            initial_guesses={'Ea': 85000, 'A': 1e11}
        )

        fit_fn = fit_kinetic_model(
            datasets=[dataset],
            model_name="single_step",
            model_definition_args={'f_alpha_model': 'Fn'},
            initial_guesses={'Ea': 85000, 'A': 1e11, 'n': 1.0},
            parameter_bounds={'Ea': (50000, 150000), 'A': (1e7, 1e15), 'n': (0, 5)}
        )

        ranked = rank_models([fit_f1, fit_fn])

        # Check all results have simplicity_penalty field
        for result in ranked:
            assert 'simplicity_penalty' in result
            assert isinstance(result['simplicity_penalty'], (int, float))
            assert result['simplicity_penalty'] >= 0

        # F1 (no fitted shape params) should have zero penalty
        f1_result = next(r for r in ranked if 'n' not in r['parameters'])
        assert f1_result['simplicity_penalty'] == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
