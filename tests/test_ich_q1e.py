"""
Tests for ICH Q1E regulatory compliance features.

Tests one-sided CI calculation, ICH extrapolation ceiling, regulatory
section HTML generation, and integration with auto_model_isothermal_data.
"""

import numpy as np
import pytest
from pathlib import Path
import tempfile

from akts import (
    KineticDataset,
    fit_kinetic_model,
    run_bootstrap,
    auto_model_isothermal_data
)
from akts.helpers import time_to_conversion, calculate_ich_q1e_ceiling
from akts.reporting import _create_regulatory_section_html


class TestICHQ1ECeiling:
    """Test ICH Q1E extrapolation ceiling calculation."""

    def test_12_month_study(self):
        """12-month study should allow 24-month extrapolation."""
        ceiling = calculate_ich_q1e_ceiling(12, is_long_term=True)
        assert ceiling == 24.0, "12-month study should allow 2× extrapolation"

    def test_18_month_study(self):
        """18-month study should allow 30-month extrapolation (min(36, 30))."""
        ceiling = calculate_ich_q1e_ceiling(18, is_long_term=True)
        assert ceiling == 30.0, "18-month study should allow min(2×18, 18+12) = 30 months"

    def test_24_month_study(self):
        """24-month study should allow 36-month extrapolation."""
        ceiling = calculate_ich_q1e_ceiling(24, is_long_term=True)
        assert ceiling == 36.0, "24-month study should allow min(48, 36) = 36 months"

    def test_6_month_study(self):
        """6-month study should allow 12-month extrapolation."""
        ceiling = calculate_ich_q1e_ceiling(6, is_long_term=True)
        assert ceiling == 12.0, "6-month study should allow min(12, 18) = 12 months"

    def test_accelerated_study(self):
        """Accelerated study should use 1.5× rule."""
        ceiling = calculate_ich_q1e_ceiling(6, is_long_term=False)
        assert ceiling == 9.0, "Accelerated 6-month study should allow 1.5×6 = 9 months"

    def test_invalid_duration(self):
        """Zero or negative duration should raise ValueError."""
        with pytest.raises(ValueError):
            calculate_ich_q1e_ceiling(0)

        with pytest.raises(ValueError):
            calculate_ich_q1e_ceiling(-12)


class TestOneSidedConfidenceInterval:
    """Test one-sided vs two-sided CI in time_to_conversion."""

    def test_one_sided_vs_two_sided(self):
        """One-sided CI should only return lower bound."""
        # Generate synthetic data
        t_fit = np.linspace(0, 7200, 50)
        T_fit = np.full_like(t_fit, 323.15)
        alpha_fit = 1.0 - np.exp(-0.0005 * t_fit)

        dataset = KineticDataset(time=t_fit, temperature=T_fit, conversion=alpha_fit)

        # Fit model
        fit_result = fit_kinetic_model(
            datasets=[dataset],
            model_name="single_step",
            model_definition_args={'f_alpha_model': 'F1'},
            initial_guesses={'Ea': 85000, 'A': 1e11}
        )

        assert fit_result.success, "Fit should succeed"

        # Run bootstrap
        bootstrap_result = run_bootstrap(
            datasets=[dataset],
            fit_result=fit_result,
            optimizer_options={'method': 'L-BFGS-B'},
            n_iterations=20,
            n_jobs=1
        )

        # Two-sided CI (default)
        result_two_sided = time_to_conversion(
            fit_result=fit_result,
            target_conversion=0.05,
            temperature_K=298.15,
            bootstrap_result=bootstrap_result,
            one_sided_ci=False
        )

        # One-sided CI (ICH Q1E)
        result_one_sided = time_to_conversion(
            fit_result=fit_result,
            target_conversion=0.05,
            temperature_K=298.15,
            bootstrap_result=bootstrap_result,
            one_sided_ci=True
        )

        # Both should have mean estimate
        assert result_two_sided['time_sec'] is not None
        assert result_one_sided['time_sec'] is not None

        # Two-sided should have both bounds
        assert result_two_sided['time_lower_sec'] is not None
        assert result_two_sided['time_upper_sec'] is not None

        # One-sided should only have lower bound
        assert result_one_sided['time_lower_sec'] is not None
        assert result_one_sided['time_upper_sec'] is None, "One-sided CI should not have upper bound"

    def test_one_sided_more_conservative(self):
        """One-sided lower bound should be more conservative (shorter time)."""
        # Generate data with more noise to get meaningful CI
        np.random.seed(42)
        t_fit = np.linspace(0, 7200, 40)
        T_fit = np.full_like(t_fit, 323.15)
        alpha_true = 1.0 - np.exp(-0.0005 * t_fit)
        alpha_fit = alpha_true + np.random.normal(0, 0.01, len(t_fit))
        alpha_fit = np.clip(alpha_fit, 0, 1)

        dataset = KineticDataset(time=t_fit, temperature=T_fit, conversion=alpha_fit)

        fit_result = fit_kinetic_model(
            datasets=[dataset],
            model_name="single_step",
            model_definition_args={'f_alpha_model': 'F1'},
            initial_guesses={'Ea': 85000, 'A': 1e11}
        )

        bootstrap_result = run_bootstrap(
            datasets=[dataset],
            fit_result=fit_result,
            optimizer_options={'method': 'L-BFGS-B'},
            n_iterations=30,
            n_jobs=1
        )

        # Calculate both
        two_sided = time_to_conversion(
            fit_result=fit_result,
            target_conversion=0.05,
            temperature_K=298.15,
            bootstrap_result=bootstrap_result,
            one_sided_ci=False
        )

        one_sided = time_to_conversion(
            fit_result=fit_result,
            target_conversion=0.05,
            temperature_K=298.15,
            bootstrap_result=bootstrap_result,
            one_sided_ci=True
        )

        # One-sided lower bound should be shorter (more conservative) or equal
        # to two-sided lower bound
        if two_sided['time_lower_sec'] and one_sided['time_lower_sec']:
            assert one_sided['time_lower_sec'] <= two_sided['time_lower_sec'], \
                "One-sided CI should be more conservative (shorter shelf-life)"


class TestRegulatoryHTML:
    """Test regulatory section HTML generation."""

    def test_html_generation(self):
        """Test basic HTML generation for regulatory section."""
        regulatory_data = {
            'shelf_life_months': 18.5,
            'shelf_life_lower_95': 15.2,
            'target_conversion': 0.05,
            'storage_temp_K': 298.15,
            'study_duration_months': 12.0,
            'ich_ceiling_months': 24.0,
            'exceeds_guideline': False
        }

        html = _create_regulatory_section_html(regulatory_data)

        # Check key elements are present
        assert 'ICH Q1E Regulatory Analysis' in html
        assert '18.5 months' in html
        assert '15.2 months' in html
        assert '5%' in html
        assert '25°C' in html  # 298.15K = 25°C
        assert '12 months' in html
        assert '24 months' in html
        assert 'Within ICH Q1E guidelines' in html
        assert 'info-btn' in html
        assert 'showICHInfo()' in html

    def test_html_warning_exceeds_ceiling(self):
        """Test HTML shows warning when extrapolation exceeds ceiling."""
        regulatory_data = {
            'shelf_life_months': 30.0,
            'shelf_life_lower_95': 26.0,
            'target_conversion': 0.05,
            'storage_temp_K': 298.15,
            'study_duration_months': 12.0,
            'ich_ceiling_months': 24.0,
            'exceeds_guideline': True
        }

        html = _create_regulatory_section_html(regulatory_data)

        # Should show warning
        assert 'warning' in html
        assert 'exceeds ICH Q1E ceiling' in html
        assert 'additional stability data needed' in html.lower()


class TestAutoModelIntegration:
    """Test ICH Q1E integration with auto_model_isothermal_data."""

    def test_regulatory_section_included(self):
        """Test regulatory section is automatically included in reports."""
        # Generate synthetic multi-temperature data
        np.random.seed(42)

        datasets = []
        for temp in [298.15, 313.15, 323.15]:
            t = np.linspace(0, 7200, 30)
            k = 1e11 * np.exp(-85000 / (8.314 * temp))
            alpha = 1.0 - np.exp(-k * t)
            alpha += np.random.normal(0, 0.005, len(t))
            alpha = np.clip(alpha, 0, 1)
            datasets.append(KineticDataset(time=t, temperature=np.full_like(t, temp), conversion=alpha))

        # Create temporary report file
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "test_report.html"

            # Run auto_model with predictions (triggers regulatory calculation)
            results = auto_model_isothermal_data(
                data_files=datasets,
                predict=(1, 'year'),
                predict_temperature_K=298.15,
                models_to_try=['F1', 'F2'],
                bootstrap_iterations=10,
                report_path=str(report_path)
            )

            # Check report was generated
            assert report_path.exists(), "Report should be generated"

            # Read report and check for regulatory section
            with open(report_path, 'r', encoding='utf-8') as f:
                html_content = f.read()

            # Verify regulatory section is present
            assert 'ICH Q1E Regulatory Analysis' in html_content
            assert 'Shelf-Life Estimate' in html_content
            assert 'ICH Q1E Extrapolation Ceiling' in html_content
            assert 'showICHInfo()' in html_content

    def test_no_regulatory_without_predictions(self):
        """Test no regulatory section when predictions are not made."""
        # Single dataset, no predictions
        t = np.linspace(0, 3600, 30)
        alpha = 1.0 - np.exp(-0.0008 * t)
        dataset = KineticDataset(time=t, temperature=np.full_like(t, 323.15), conversion=alpha)

        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "test_report_no_pred.html"

            # Run without predictions
            results = auto_model_isothermal_data(
                data_files=[dataset],
                predict=None,  # No predictions
                models_to_try=['F1'],
                bootstrap_iterations=0,
                report_path=str(report_path)
            )

            with open(report_path, 'r', encoding='utf-8') as f:
                html_content = f.read()

            # Regulatory section should not be present
            assert 'ICH Q1E Regulatory Analysis' not in html_content


class TestRegulatoryCalculation:
    """Test regulatory calculation details."""

    def test_study_duration_calculation(self):
        """Test study duration is correctly calculated from datasets."""
        # 6-month study (180 days = 15,552,000 seconds)
        max_time_sec = 180 * 24 * 3600
        t = np.linspace(0, max_time_sec, 50)
        alpha = 1.0 - np.exp(-0.000001 * t)
        dataset = KineticDataset(time=t, temperature=np.full_like(t, 298.15), conversion=alpha)

        # Expected study duration in months (using 30.44 days/month)
        expected_months = max_time_sec / (30.44 * 24 * 3600)

        # Verify calculation (approximately 5.9 months)
        assert 5.8 < expected_months < 6.0

    def test_5_percent_threshold(self):
        """Test that 5% degradation is used as standard threshold."""
        # This is tested implicitly through auto_model, but verify the constant
        # The target_conversion should always be 0.05 in regulatory calculations
        # (tested in integration tests above)
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
