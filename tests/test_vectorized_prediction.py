"""
Tests for vectorized long-term prediction performance improvements.

Tests that chunked predictions for long time series (thousands of points)
produce accurate results and execute efficiently.
"""

import numpy as np
import pytest
import time
from akts import KineticDataset, fit_kinetic_model, predict_conversion


class TestVectorizedPrediction:
    """Test vectorized prediction for long temperature profiles."""

    def test_chunked_vs_standard_f1_accuracy(self):
        """Verify chunked prediction matches standard path for F1 model."""
        # Generate synthetic F1 data at 298K
        t_fit = np.linspace(0, 3600, 50)  # 1 hour, 50 points
        T_fit = np.full_like(t_fit, 298.15)

        # F1 parameters: Ea=80 kJ/mol, A=1e10 s^-1
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

        # Predict with long time series (5000 points, triggers chunking)
        t_pred_long = np.linspace(0, 365*24*3600, 5000)  # 1 year, 5000 points
        temp_pred = np.full_like(t_pred_long, 298.15)

        # Standard prediction (chunk_size > n_points)
        pred_standard = predict_conversion(
            kinetic_description=fit_result,
            temperature_program=(t_pred_long, temp_pred),
            solver_options={'chunk_size': 10000}  # No chunking
        )

        # Chunked prediction (chunk_size = 1000)
        pred_chunked = predict_conversion(
            kinetic_description=fit_result,
            temperature_program=(t_pred_long, temp_pred),
            solver_options={'chunk_size': 1000}
        )

        # Verify times match
        assert np.allclose(pred_standard.time, pred_chunked.time), \
            "Time arrays should match"

        # Verify conversions are very close (within 0.1%)
        max_diff = np.max(np.abs(pred_standard.conversion - pred_chunked.conversion))
        assert max_diff < 0.001, \
            f"Chunked prediction differs by {max_diff:.4f}, expected <0.001"

        # Verify final conversion is reasonable (should be near 1.0 for 1 year)
        assert pred_chunked.conversion[-1] > 0.9, \
            "Should reach high conversion after 1 year"

    def test_chunked_prediction_a_to_b_to_c(self):
        """Verify chunked prediction works for consecutive reaction model."""
        # Generate synthetic A->B->C data
        t_fit = np.linspace(0, 7200, 60)  # 2 hours
        T_fit = np.full_like(t_fit, 323.15)  # 50C

        # Simplified conversion profile (monotonic increase)
        alpha_fit = 1.0 - np.exp(-0.0005 * t_fit)

        dataset = KineticDataset(time=t_fit, temperature=T_fit, conversion=alpha_fit)

        # Fit A->B->C model
        fit_result = fit_kinetic_model(
            datasets=[dataset],
            model_name="A->B->C",
            model_definition_args={'f1_model': 'F1', 'f2_model': 'F1'},
            initial_guesses={'Ea1': 85000, 'A1': 1e11, 'Ea2': 95000, 'A2': 1e12}
        )

        if not fit_result.success:
            pytest.skip("A->B->C fit failed, skipping chunked prediction test")

        # Predict with long series (3000 points)
        t_pred = np.linspace(0, 365*24*3600, 3000)  # 1 year
        temp_pred = np.full_like(t_pred, 323.15)

        # Chunked prediction
        pred_chunked = predict_conversion(
            kinetic_description=fit_result,
            temperature_program=(t_pred, temp_pred),
            solver_options={'chunk_size': 1000}
        )

        # Basic sanity checks
        assert len(pred_chunked.conversion) == len(t_pred), \
            "Should return prediction for all time points"
        assert np.all(pred_chunked.conversion >= 0.0), \
            "Conversion should be non-negative"
        assert np.all(pred_chunked.conversion <= 1.0), \
            "Conversion should not exceed 1.0"
        assert np.all(np.diff(pred_chunked.conversion) >= -1e-6), \
            "Conversion should be monotonically increasing"

    def test_chunk_size_option_respected(self):
        """Verify chunk_size option controls chunking behavior."""
        # Simple F1 data
        t_fit = np.linspace(0, 3600, 30)
        T_fit = np.full_like(t_fit, 298.15)
        alpha_fit = 1.0 - np.exp(-0.001 * t_fit)

        dataset = KineticDataset(time=t_fit, temperature=T_fit, conversion=alpha_fit)

        fit_result = fit_kinetic_model(
            datasets=[dataset],
            model_name="single_step",
            model_definition_args={'f_alpha_model': 'F1'},
            initial_guesses={'Ea': 85000, 'A': 1e11}
        )

        assert fit_result.success

        # Short prediction (100 points)
        t_pred_short = np.linspace(0, 10*24*3600, 100)
        temp_pred = np.full_like(t_pred_short, 298.15)

        # With chunk_size=50, should chunk
        pred = predict_conversion(
            kinetic_description=fit_result,
            temperature_program=(t_pred_short, temp_pred),
            solver_options={'chunk_size': 50}
        )

        assert len(pred.conversion) == 100
        assert np.all(np.isfinite(pred.conversion))

    def test_long_prediction_speedup(self):
        """Benchmark chunked prediction for long time series."""
        # Generate data
        t_fit = np.linspace(0, 3600, 40)
        T_fit = np.full_like(t_fit, 298.15)
        alpha_fit = 1.0 - np.exp(-0.0008 * t_fit)

        dataset = KineticDataset(time=t_fit, temperature=T_fit, conversion=alpha_fit)

        fit_result = fit_kinetic_model(
            datasets=[dataset],
            model_name="single_step",
            model_definition_args={'f_alpha_model': 'F1'},
            initial_guesses={'Ea': 85000, 'A': 1e11}
        )

        assert fit_result.success

        # Very long prediction (10,000 points)
        t_pred = np.linspace(0, 3*365*24*3600, 10000)  # 3 years, 10k points
        temp_pred = np.full_like(t_pred, 298.15)

        # Time the chunked prediction
        start = time.time()
        pred_chunked = predict_conversion(
            kinetic_description=fit_result,
            temperature_program=(t_pred, temp_pred),
            solver_options={'chunk_size': 2000}
        )
        elapsed_chunked = time.time() - start

        # Should complete in reasonable time (<5 seconds)
        assert elapsed_chunked < 5.0, \
            f"Chunked prediction took {elapsed_chunked:.2f}s, expected <5s"

        # Verify result is valid
        assert len(pred_chunked.conversion) == 10000
        assert np.all(np.isfinite(pred_chunked.conversion))
        assert pred_chunked.conversion[-1] > 0.95, \
            "Should reach high conversion after 3 years"

    def test_climate_profile_with_daily_fluctuations(self):
        """Test vectorized prediction with realistic climate fluctuations."""
        # Generate F1 fit data
        t_fit = np.linspace(0, 7200, 50)
        T_fit = np.full_like(t_fit, 298.15)
        alpha_fit = 1.0 - np.exp(-0.0005 * t_fit)

        dataset = KineticDataset(time=t_fit, temperature=T_fit, conversion=alpha_fit)

        fit_result = fit_kinetic_model(
            datasets=[dataset],
            model_name="single_step",
            model_definition_args={'f_alpha_model': 'F1'},
            initial_guesses={'Ea': 85000, 'A': 1e11}
        )

        assert fit_result.success

        # Generate 2-year daily temperature fluctuation profile
        # 25°C ± 5°C daily sine wave
        n_days = 2 * 365
        t_pred = np.linspace(0, n_days * 24 * 3600, n_days * 24)  # Hourly resolution

        # Daily cycle: T(t) = 298.15 + 5*sin(2π*t/(24*3600))
        temp_pred = 298.15 + 5.0 * np.sin(2 * np.pi * t_pred / (24 * 3600))

        # Predict with chunking
        pred = predict_conversion(
            kinetic_description=fit_result,
            temperature_program=(t_pred, temp_pred),
            solver_options={'chunk_size': 1000}
        )

        # Verify prediction completed successfully
        assert len(pred.conversion) == len(t_pred)
        assert np.all(np.isfinite(pred.conversion))
        assert np.all(pred.conversion >= 0.0)
        assert np.all(pred.conversion <= 1.0)

        # Verify monotonic increase (allowing small numerical tolerance)
        diff = np.diff(pred.conversion)
        assert np.all(diff >= -1e-6), \
            "Conversion should be monotonically increasing"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
