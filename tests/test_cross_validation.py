"""
Tests for cross-validation functions.

Tests leave-one-temperature-out cross-validation for model validation.
"""

import numpy as np
import pytest

from akts import KineticDataset, run_leave_one_out_cv


class TestLeaveOneOutCV:
    """Test leave-one-temperature-out cross-validation."""

    def test_loo_cv_with_three_temperatures(self):
        """LOO-CV should work with 3 temperatures."""
        np.random.seed(50)

        # Generate F1 synthetic data at 3 temperatures
        Ea_true = 85000
        A_true = 1e11
        R = 8.314

        datasets = []
        for T in [313.15, 323.15, 333.15]:
            t = np.linspace(0, 3600, 30)
            k_true = A_true * np.exp(-Ea_true / (R * T))
            alpha_true = 1.0 - np.exp(-k_true * t)
            alpha = alpha_true + np.random.normal(0, 0.01, len(t))
            alpha = np.clip(alpha, 0, 1)

            datasets.append(KineticDataset(
                time=t,
                temperature=np.full_like(t, T),
                conversion=alpha
            ))

        # Run LOO-CV
        cv_result = run_leave_one_out_cv(
            datasets=datasets,
            model_name="single_step",
            model_definition_args={'f_alpha_model': 'F1'},
            initial_guesses={'Ea': 85000, 'A': 1e11},
            parameter_bounds={'Ea': (50000, 150000), 'A': (1e7, 1e15)},
            optimizer_options={'method': 'L-BFGS-B'}
        )

        # Check structure
        assert 'loo_results' in cv_result
        assert 'mean_held_out_r2' in cv_result
        assert 'cv_score' in cv_result
        assert 'n_folds' in cv_result

        # Check we ran 3 folds
        assert cv_result['n_folds'] == 3
        assert len(cv_result['loo_results']) == 3

        # Check all folds succeeded
        assert cv_result['n_successful_folds'] == 3

        # Check reasonable CV score
        assert np.isfinite(cv_result['cv_score'])
        assert cv_result['cv_score'] > 0.9, f"CV score should be >0.9, got {cv_result['cv_score']}"

        # Check mean R² is reasonable
        assert np.isfinite(cv_result['mean_held_out_r2'])
        assert cv_result['mean_held_out_r2'] > 0.85

    def test_loo_cv_with_four_temperatures(self):
        """LOO-CV should work with 4 temperatures."""
        np.random.seed(51)

        # Generate F2 synthetic data at 4 temperatures
        Ea_true = 80000
        A_true = 1e10
        R = 8.314

        datasets = []
        for T in [313.15, 323.15, 333.15, 343.15]:
            t = np.linspace(0, 7200, 30)
            k_true = A_true * np.exp(-Ea_true / (R * T))
            # F2: α = kt/(1+kt)
            kt = k_true * t
            alpha_true = kt / (1 + kt)
            alpha = alpha_true + np.random.normal(0, 0.01, len(t))
            alpha = np.clip(alpha, 0, 0.99)

            datasets.append(KineticDataset(
                time=t,
                temperature=np.full_like(t, T),
                conversion=alpha
            ))

        # Run LOO-CV
        cv_result = run_leave_one_out_cv(
            datasets=datasets,
            model_name="single_step",
            model_definition_args={'f_alpha_model': 'F2'},
            parameter_bounds={'Ea': (50000, 150000), 'A': (1e7, 1e15)},
            optimizer_options={'method': 'L-BFGS-B'}
        )

        # Check 4 folds
        assert cv_result['n_folds'] == 4
        assert len(cv_result['loo_results']) == 4

        # Check most folds succeeded
        assert cv_result['n_successful_folds'] >= 3

        # Check reasonable CV score
        if cv_result['n_successful_folds'] >= 3:
            assert np.isfinite(cv_result['cv_score'])
            assert cv_result['cv_score'] > 0.80

    def test_loo_cv_with_insufficient_data(self):
        """LOO-CV should warn with fewer than 3 datasets."""
        # Only 2 temperatures - not enough for meaningful CV
        t = np.linspace(0, 3600, 20)
        datasets = [
            KineticDataset(
                time=t,
                temperature=np.full_like(t, 313.15),
                conversion=0.5 * (1.0 - np.exp(-0.0001 * t))
            ),
            KineticDataset(
                time=t,
                temperature=np.full_like(t, 323.15),
                conversion=0.5 * (1.0 - np.exp(-0.0002 * t))
            )
        ]

        cv_result = run_leave_one_out_cv(
            datasets=datasets,
            model_name="single_step",
            model_definition_args={'f_alpha_model': 'F1'},
            optimizer_options={'method': 'L-BFGS-B'}
        )

        # Should return with warning
        assert cv_result['n_folds'] == 2
        assert cv_result['n_successful_folds'] == 0
        assert np.isnan(cv_result['cv_score'])

    def test_loo_cv_with_fitted_parameters(self):
        """LOO-CV should work with Fn (fitted n parameter)."""
        np.random.seed(52)

        # Generate F1 data (n≈1) at 3 temperatures
        Ea_true = 85000
        A_true = 1e11
        R = 8.314

        datasets = []
        for T in [313.15, 323.15, 333.15]:
            t = np.linspace(0, 3600, 30)
            k_true = A_true * np.exp(-Ea_true / (R * T))
            alpha_true = 1.0 - np.exp(-k_true * t)
            alpha = alpha_true + np.random.normal(0, 0.01, len(t))
            alpha = np.clip(alpha, 0, 1)

            datasets.append(KineticDataset(
                time=t,
                temperature=np.full_like(t, T),
                conversion=alpha
            ))

        # Run LOO-CV with Fn model
        cv_result = run_leave_one_out_cv(
            datasets=datasets,
            model_name="single_step",
            model_definition_args={'f_alpha_model': 'Fn'},
            initial_guesses={'Ea': 85000, 'A': 1e11, 'n': 1.0},
            parameter_bounds={'Ea': (50000, 150000), 'A': (1e7, 1e15), 'n': (0, 5)},
            optimizer_options={'method': 'L-BFGS-B'}
        )

        # Should succeed
        assert cv_result['n_folds'] == 3
        assert cv_result['n_successful_folds'] >= 2

        # Check reasonable performance
        if cv_result['n_successful_folds'] >= 2:
            assert np.isfinite(cv_result['cv_score'])
            assert cv_result['cv_score'] > 0.80

    def test_loo_cv_per_fold_details(self):
        """Verify per-fold results contain expected information."""
        np.random.seed(53)

        # Generate simple F1 data
        Ea_true = 85000
        A_true = 1e11
        R = 8.314

        datasets = []
        for T in [313.15, 323.15, 333.15]:
            t = np.linspace(0, 3600, 25)
            k_true = A_true * np.exp(-Ea_true / (R * T))
            alpha_true = 1.0 - np.exp(-k_true * t)
            alpha = alpha_true + np.random.normal(0, 0.005, len(t))
            alpha = np.clip(alpha, 0, 1)

            datasets.append(KineticDataset(
                time=t,
                temperature=np.full_like(t, T),
                conversion=alpha
            ))

        cv_result = run_leave_one_out_cv(
            datasets=datasets,
            model_name="single_step",
            model_definition_args={'f_alpha_model': 'F1'},
            initial_guesses={'Ea': 85000, 'A': 1e11},
            parameter_bounds={'Ea': (50000, 150000), 'A': (1e7, 1e15)},
            optimizer_options={'method': 'L-BFGS-B'}
        )

        # Check each fold result
        for i, fold_result in enumerate(cv_result['loo_results']):
            assert 'excluded_index' in fold_result
            assert fold_result['excluded_index'] == i

            assert 'excluded_temp_K' in fold_result
            assert 'training_fit_success' in fold_result
            assert 'held_out_rss' in fold_result
            assert 'held_out_r2' in fold_result
            assert 'held_out_n_points' in fold_result

            # Temperature should match
            expected_temp = np.mean(datasets[i].temperature)
            assert abs(fold_result['excluded_temp_K'] - expected_temp) < 0.1

            # If fit succeeded, metrics should be finite
            if fold_result['training_fit_success']:
                assert np.isfinite(fold_result['held_out_rss'])
                assert fold_result['held_out_rss'] >= 0
                # R² may be negative if model is bad, but should be finite
                assert np.isfinite(fold_result['held_out_r2'])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
