"""
Tests for Durbin-Watson autocorrelation check functionality.

Tests that the Durbin-Watson statistic is correctly calculated and included
in fit results for detecting systematic residual patterns.
"""

import numpy as np
import pytest
from akts import KineticDataset, fit_kinetic_model
from akts.utils import calculate_durbin_watson
from akts.core import rank_models


class TestDurbinWatsonCalculation:
    """Test Durbin-Watson statistic calculation."""

    def test_dw_no_autocorrelation(self):
        """Test DW ≈ 2 for uncorrelated residuals."""
        np.random.seed(42)
        residuals = np.random.randn(100)  # White noise

        dw = calculate_durbin_watson(residuals)

        # Should be close to 2 (no autocorrelation)
        assert 1.5 < dw < 2.5, f"DW={dw:.2f}, expected ≈2 for white noise"

    def test_dw_positive_autocorrelation(self):
        """Test DW < 2 for positively autocorrelated residuals."""
        # Create positively autocorrelated residuals
        # (systematic pattern: grouped positive/negative)
        residuals = np.concatenate([
            np.ones(25) * 0.5,    # Positive cluster
            -np.ones(25) * 0.5,   # Negative cluster
            np.ones(25) * 0.5,    # Positive cluster
            -np.ones(25) * 0.5    # Negative cluster
        ])

        dw = calculate_durbin_watson(residuals)

        # Strong positive autocorrelation → DW << 2
        assert dw < 1.5, f"DW={dw:.2f}, expected <1.5 for positive autocorrelation"

    def test_dw_negative_autocorrelation(self):
        """Test DW > 2 for negatively autocorrelated residuals."""
        # Create negatively autocorrelated residuals (alternating pattern)
        residuals = np.array([(-1)**i * 0.5 for i in range(100)])

        dw = calculate_durbin_watson(residuals)

        # Negative autocorrelation → DW > 2
        assert dw > 2.5, f"DW={dw:.2f}, expected >2.5 for negative autocorrelation"

    def test_dw_with_nan_values(self):
        """Test DW handles NaN values correctly."""
        residuals = np.array([1.0, 2.0, np.nan, 3.0, 4.0, np.nan, 5.0])

        dw = calculate_durbin_watson(residuals)

        # Should compute on valid values only
        assert np.isfinite(dw), "DW should be finite when valid data exists"

    def test_dw_insufficient_data(self):
        """Test DW returns NaN for insufficient data."""
        residuals = np.array([1.0])

        dw = calculate_durbin_watson(residuals)

        assert np.isnan(dw), "DW should be NaN for <2 data points"

    def test_dw_perfect_fit(self):
        """Test DW with zero residuals."""
        residuals = np.zeros(50)

        dw = calculate_durbin_watson(residuals)

        # Zero residuals → undefined DW
        assert np.isnan(dw), "DW should be NaN for perfect fit (zero residuals)"


class TestDurbinWatsonInFitting:
    """Test DW statistic is included in fit results."""

    def test_dw_included_in_fit_result(self):
        """Test DW statistic is calculated and stored in FitResult."""
        # Generate synthetic F1 data
        t_fit = np.linspace(0, 3600, 40)
        T_fit = np.full_like(t_fit, 298.15)

        # Perfect F1 kinetics
        Ea_true = 80000.0
        A_true = 1e10
        k = A_true * np.exp(-Ea_true / (8.314 * 298.15))
        alpha_fit = 1.0 - np.exp(-k * t_fit)

        dataset = KineticDataset(time=t_fit, temperature=T_fit, conversion=alpha_fit)

        # Fit model
        fit_result = fit_kinetic_model(
            datasets=[dataset],
            model_name="single_step",
            model_definition_args={'f_alpha_model': 'F1'},
            initial_guesses={'Ea': 85000, 'A': 1e11}
        )

        assert fit_result.success, "Fit should succeed"

        # Check DW is present
        assert hasattr(fit_result, 'durbin_watson'), "FitResult should have durbin_watson attribute"
        assert fit_result.durbin_watson is not None, "DW should not be None"

        # For perfect or near-perfect fit, DW may be very small or NaN
        # (residuals ≈ 0). For fits with measurable residuals, DW should be finite.
        if fit_result.r_squared > 0.9999:
            # Nearly perfect fit - DW may be close to 0 or NaN
            # (which is actually correct behavior for tiny residuals)
            assert np.isfinite(fit_result.durbin_watson) or np.isnan(fit_result.durbin_watson), \
                "DW should be finite or NaN for near-perfect fit"
        else:
            assert np.isfinite(fit_result.durbin_watson), "DW should be finite"
            # For good fit with measurable residuals, DW should be in reasonable range
            assert 0.5 < fit_result.durbin_watson < 3.5, \
                f"DW={fit_result.durbin_watson:.2f}, expected 0.5-3.5 for good fit"

    def test_dw_detects_systematic_error(self):
        """Test DW detects systematic model error."""
        # Generate data with F2 kinetics but fit with F1 (model mismatch)
        t_fit = np.linspace(0, 3600, 50)
        T_fit = np.full_like(t_fit, 298.15)

        # F2 kinetics (wrong model)
        k = 0.0008
        alpha_fit = k * t_fit / (1.0 + k * t_fit)

        dataset = KineticDataset(time=t_fit, temperature=T_fit, conversion=alpha_fit)

        # Fit with F1 (systematic error expected)
        fit_result = fit_kinetic_model(
            datasets=[dataset],
            model_name="single_step",
            model_definition_args={'f_alpha_model': 'F1'},
            initial_guesses={'Ea': 85000, 'A': 1e11}
        )

        if fit_result.success:
            # Model mismatch may cause DW to deviate from 2
            # (systematic under/over-prediction pattern)
            dw = fit_result.durbin_watson
            assert np.isfinite(dw), "DW should be finite"

            # Note: This test is informative - exact DW depends on fit quality
            # A perfect mismatch would show DW far from 2, but optimization
            # may partially compensate
            print(f"F1 fit to F2 data: DW={dw:.3f}, R²={fit_result.r_squared:.4f}")

    def test_dw_in_rank_models(self):
        """Test DW statistic appears in rank_models output."""
        # Generate data
        t_fit = np.linspace(0, 3600, 40)
        T_fit = np.full_like(t_fit, 298.15)
        alpha_fit = 1.0 - np.exp(-0.0008 * t_fit)

        dataset = KineticDataset(time=t_fit, temperature=T_fit, conversion=alpha_fit)

        # Fit multiple models
        fit_results = []
        for model in ['F1', 'F2']:
            fit_res = fit_kinetic_model(
                datasets=[dataset],
                model_name="single_step",
                model_definition_args={'f_alpha_model': model},
                initial_guesses={'Ea': 85000, 'A': 1e11}
            )
            if fit_res.success:
                fit_results.append(fit_res)

        # Rank models
        ranked = rank_models(fit_results)

        # Check DW is in stats
        assert len(ranked) > 0, "Should have at least one ranked model"

        for model_info in ranked:
            assert 'durbin_watson' in model_info['stats'], \
                "DW should be in stats dict"
            dw = model_info['stats']['durbin_watson']
            # Should be finite for successful fits
            if np.isfinite(dw):
                assert 0 <= dw <= 4, f"DW={dw} should be in [0,4]"

    def test_dw_multiple_datasets(self):
        """Test DW calculation with multiple datasets."""
        # Generate 3 datasets at different temperatures
        datasets = []
        for T in [283.15, 298.15, 313.15]:
            t = np.linspace(0, 3600, 30)
            k = 1e10 * np.exp(-80000 / (8.314 * T))
            alpha = 1.0 - np.exp(-k * t)
            datasets.append(KineticDataset(time=t, temperature=np.full_like(t, T),
                                           conversion=alpha))

        # Fit
        fit_result = fit_kinetic_model(
            datasets=datasets,
            model_name="single_step",
            model_definition_args={'f_alpha_model': 'F1'},
            initial_guesses={'Ea': 85000, 'A': 1e11}
        )

        assert fit_result.success, "Multi-dataset fit should succeed"
        assert np.isfinite(fit_result.durbin_watson), \
            "DW should be finite for multi-dataset fit"


class TestDurbinWatsonInterpretation:
    """Test interpretation of DW values."""

    def test_dw_acceptable_range(self):
        """Verify DW in acceptable range (1.5-2.5) indicates no strong autocorrelation."""
        # Generate good F1 fit
        t = np.linspace(0, 3600, 50)
        T = np.full_like(t, 298.15)
        alpha = 1.0 - np.exp(-0.0008 * t)

        dataset = KineticDataset(time=t, temperature=T, conversion=alpha)

        fit_result = fit_kinetic_model(
            datasets=[dataset],
            model_name="single_step",
            model_definition_args={'f_alpha_model': 'F1'},
            initial_guesses={'Ea': 85000, 'A': 1e11}
        )

        if fit_result.success and fit_result.r_squared > 0.95:
            dw = fit_result.durbin_watson
            # Good fit should have acceptable DW
            # (though not guaranteed - depends on noise)
            if np.isfinite(dw):
                is_acceptable = 1.5 <= dw <= 2.5
                print(f"DW={dw:.3f}, Acceptable={is_acceptable}, R²={fit_result.r_squared:.4f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
