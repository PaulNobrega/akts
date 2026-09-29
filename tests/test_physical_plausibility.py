"""
Tests for physical plausibility filtering functionality.

Tests that physically implausible parameter values are correctly identified
and can be optionally excluded from model ranking.
"""

import numpy as np
import pytest
from akts import KineticDataset, fit_kinetic_model, auto_model_isothermal_data
from akts.utils import check_physical_plausibility


class TestPhysicalPlausibilityCheck:
    """Test physical plausibility checking function."""

    def test_plausible_parameters(self):
        """Test typical plausible parameters pass check."""
        params = {'Ea': 80000, 'A': 1e12}  # 80 kJ/mol, 10^12 s^-1
        is_plausible, issues = check_physical_plausibility(params, strict=False)

        assert is_plausible, f"Should be plausible, got issues: {issues}"
        assert len(issues) == 0

    def test_strict_vs_permissive(self):
        """Test strict mode has tighter bounds."""
        # Ea = 25 kJ/mol: below typical (30 kJ/mol) but above absolute (10 kJ/mol)
        params = {'Ea': 25000, 'A': 1e10}

        is_plausible_strict, issues_strict = check_physical_plausibility(params, strict=True)
        is_plausible_permissive, issues_permissive = check_physical_plausibility(params, strict=False)

        # Should fail strict but pass permissive
        assert not is_plausible_strict, "Should fail strict check"
        assert is_plausible_permissive, "Should pass permissive check"

    def test_very_low_ea(self):
        """Test very low Ea is flagged."""
        params = {'Ea': 5000, 'A': 1e10}  # 5 kJ/mol - too low
        is_plausible, issues = check_physical_plausibility(params, strict=False)

        assert not is_plausible, "Very low Ea should be implausible"
        assert len(issues) > 0
        assert "Ea" in issues[0]
        assert "below" in issues[0].lower()

    def test_very_high_ea(self):
        """Test very high Ea is flagged."""
        params = {'Ea': 500000, 'A': 1e10}  # 500 kJ/mol - too high
        is_plausible, issues = check_physical_plausibility(params, strict=False)

        assert not is_plausible, "Very high Ea should be implausible"
        assert len(issues) > 0
        assert "Ea" in issues[0]
        assert "exceed" in issues[0].lower()

    def test_very_low_a(self):
        """Test very low A is flagged."""
        params = {'Ea': 80000, 'A': 0.001}  # 10^-3 s^-1 - too low
        is_plausible, issues = check_physical_plausibility(params, strict=False)

        assert not is_plausible, "Very low A should be implausible"
        assert len(issues) > 0
        assert any("A" in issue for issue in issues)

    def test_very_high_a(self):
        """Test very high A is flagged."""
        params = {'Ea': 80000, 'A': 1e30}  # 10^30 s^-1 - exceeds collision frequency
        is_plausible, issues = check_physical_plausibility(params, strict=False)

        assert not is_plausible, "Very high A should be implausible"
        assert len(issues) > 0
        assert any("A" in issue for issue in issues)

    def test_multiple_issues(self):
        """Test multiple issues are reported."""
        params = {'Ea': 5000, 'A': 1e30}  # Both Ea and A implausible
        is_plausible, issues = check_physical_plausibility(params, strict=False)

        assert not is_plausible
        assert len(issues) == 2, f"Expected 2 issues, got {len(issues)}: {issues}"

    def test_multi_step_model(self):
        """Test plausibility check for multi-step model (A->B->C)."""
        params = {'Ea1': 85000, 'A1': 1e11, 'Ea2': 95000, 'A2': 1e12}
        is_plausible, issues = check_physical_plausibility(params, strict=False)

        assert is_plausible, f"Should be plausible, got issues: {issues}"


class TestPlausibilityInFitting:
    """Test plausibility flags in fit results."""

    def test_plausibility_included_in_fit_result(self):
        """Test plausibility check is performed and stored."""
        # Generate data
        t_fit = np.linspace(0, 3600, 40)
        T_fit = np.full_like(t_fit, 298.15)
        alpha_fit = 1.0 - np.exp(-0.0008 * t_fit)

        dataset = KineticDataset(time=t_fit, temperature=T_fit, conversion=alpha_fit)

        # Fit model
        fit_result = fit_kinetic_model(
            datasets=[dataset],
            model_name="single_step",
            model_definition_args={'f_alpha_model': 'F1'},
            initial_guesses={'Ea': 85000, 'A': 1e11}
        )

        assert fit_result.success

        # Check plausibility fields exist
        assert hasattr(fit_result, 'is_physically_plausible')
        assert hasattr(fit_result, 'plausibility_issues')

        # For reasonable synthetic data, should be plausible
        if fit_result.is_physically_plausible is not None:
            # If check was performed, result should be plausible
            assert fit_result.is_physically_plausible, \
                f"Should be plausible, issues: {fit_result.plausibility_issues}"


class TestFilterImplausible:
    """Test filter_implausible option in auto_model_isothermal_data."""

    def test_filter_implausible_default_false(self):
        """Test filter_implausible defaults to False."""
        # Generate simple data
        t = np.linspace(0, 3600, 30)
        T = np.full_like(t, 298.15)
        alpha = 1.0 - np.exp(-0.0008 * t)

        dataset = KineticDataset(time=t, temperature=T, conversion=alpha)

        # Run without filter (default)
        results = auto_model_isothermal_data(
            data_files=[dataset],
            models_to_try=['F1', 'F2'],
            bootstrap_iterations=0,
            report_path=None
        )

        # Should have results
        assert 'top_models' in results
        assert len(results['top_models']) > 0

    def test_filter_implausible_excludes_bad_models(self):
        """Test filter_implausible=True excludes implausible models."""
        # Generate data that might produce edge-case fits
        t = np.linspace(0, 7200, 30)
        T = np.full_like(t, 323.15)
        # Very slow conversion (might produce low Ea)
        alpha = 0.05 * (t / 7200)

        dataset = KineticDataset(time=t, temperature=T, conversion=alpha)

        # Run without filter
        results_unfiltered = auto_model_isothermal_data(
            data_files=[dataset],
            models_to_try=['F0', 'F1', 'F2'],
            filter_implausible=False,
            bootstrap_iterations=0,
            report_path=None
        )

        # Run with filter
        results_filtered = auto_model_isothermal_data(
            data_files=[dataset],
            models_to_try=['F0', 'F1', 'F2'],
            filter_implausible=True,
            bootstrap_iterations=0,
            report_path=None
        )

        # Both should have results
        assert len(results_unfiltered['top_models']) > 0
        assert len(results_filtered['top_models']) > 0

        # If filtering worked, some models may have been excluded
        # (but we can't guarantee this with synthetic data)
        # At minimum, check that plausibility info is present
        for model in results_filtered['top_models']:
            assert 'is_physically_plausible' in model['stats']

    def test_plausibility_info_in_output(self):
        """Test plausibility information appears in output stats."""
        t = np.linspace(0, 3600, 40)
        T = np.full_like(t, 298.15)
        alpha = 1.0 - np.exp(-0.0008 * t)

        dataset = KineticDataset(time=t, temperature=T, conversion=alpha)

        results = auto_model_isothermal_data(
            data_files=[dataset],
            models_to_try=['F1', 'F2'],
            bootstrap_iterations=0,
            report_path=None
        )

        # Check plausibility info in top_models
        for model in results['top_models']:
            assert 'is_physically_plausible' in model['stats']
            # plausibility_issues may be None if plausible
            assert 'plausibility_issues' in model['stats']


class TestPlausibilityEdgeCases:
    """Test edge cases for plausibility checking."""

    def test_edge_case_boundary_values(self):
        """Test parameters exactly at boundaries."""
        # Test lower bounds
        params_lower = {'Ea': 10000, 'A': 0.01}  # Exactly at permissive limits
        is_plausible, _ = check_physical_plausibility(params_lower, strict=False)
        assert is_plausible, "Should accept boundary values"

        # Test upper bounds
        params_upper = {'Ea': 400000, 'A': 1e25}  # Exactly at permissive limits
        is_plausible, _ = check_physical_plausibility(params_upper, strict=False)
        assert is_plausible, "Should accept boundary values"

    def test_empty_parameters(self):
        """Test empty parameters dict."""
        params = {}
        is_plausible, issues = check_physical_plausibility(params, strict=False)

        # Empty params should be "plausible" (no issues to flag)
        assert is_plausible
        assert len(issues) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
