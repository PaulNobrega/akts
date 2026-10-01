"""
Tests for ICH Q1E compliance: one-sided confidence intervals and shelf-life calculation.

These tests verify that the implementation meets ICH Q1E regulatory requirements:
1. One-sided confidence bounds (not two-sided intervals)
2. Confidence-band crossing method for shelf-life
3. Appropriate bound selection (lower for decreasing, upper for increasing)
4. Conservative estimates (shorter shelf-life than mean)
"""
import numpy as np
import pytest
from akts.shelf_life import calculate_shelf_life_ich_q1e, time_to_specification, _interpolate_crossing
from akts.prediction import _ci_from_replicate_curves
from akts.datatypes import PredictionResult


class TestOneSidedCI:
    """Test one-sided confidence interval calculation per ICH Q1E."""

    def test_decreasing_attribute_uses_upper_percentile(self):
        """
        For decreasing attributes, conservative bound should be 95th percentile.

        When tracking DEGRADATION (conversion increases over time):
        - 95th percentile = faster degradation = conservative shelf-life
        - This corresponds to ICH Q1E "lower bound on potency"
        """
        # Create bootstrap distribution
        np.random.seed(42)
        n_bootstrap = 1000
        n_timepoints = 50

        # Simulate conversion curves with some variability
        base_curve = np.linspace(0, 0.3, n_timepoints)
        curves = np.random.normal(base_curve, 0.02, size=(n_bootstrap, n_timepoints))

        # Calculate one-sided CI for decreasing attribute
        lower, upper = _ci_from_replicate_curves(
            curves, confidence_level=0.95, attribute_direction='decreasing', ci_type='one-sided'
        )

        # Lower bound should be mean (for reference)
        expected_mean = np.mean(curves, axis=0)
        np.testing.assert_allclose(lower, expected_mean, rtol=0.01)

        # Upper bound should be 95th percentile (conservative)
        expected_upper = np.percentile(curves, 95, axis=0)
        np.testing.assert_allclose(upper, expected_upper, rtol=0.01)

    def test_increasing_attribute_uses_upper_percentile(self):
        """
        For increasing attributes, conservative bound should be 95th percentile.

        When tracking IMPURITIES (conversion increases over time):
        - 95th percentile = faster increase = conservative shelf-life
        """
        np.random.seed(42)
        n_bootstrap = 1000
        n_timepoints = 50

        base_curve = np.linspace(0, 0.3, n_timepoints)
        curves = np.random.normal(base_curve, 0.02, size=(n_bootstrap, n_timepoints))

        # Calculate one-sided CI for increasing attribute
        lower, upper = _ci_from_replicate_curves(
            curves, confidence_level=0.95, attribute_direction='increasing', ci_type='one-sided'
        )

        # Lower bound should be mean (for reference)
        expected_mean = np.mean(curves, axis=0)
        np.testing.assert_allclose(lower, expected_mean, rtol=0.01)

        # Upper bound should be 95th percentile (conservative)
        expected_upper = np.percentile(curves, 95, axis=0)
        np.testing.assert_allclose(upper, expected_upper, rtol=0.01)

    def test_one_sided_not_symmetric(self):
        """One-sided CI should NOT be symmetric around mean (unlike two-sided)."""
        np.random.seed(42)
        n_bootstrap = 1000
        n_timepoints = 50

        base_curve = np.linspace(0, 0.3, n_timepoints)
        curves = np.random.normal(base_curve, 0.02, size=(n_bootstrap, n_timepoints))

        # Calculate one-sided CI
        lower, upper = _ci_from_replicate_curves(
            curves, confidence_level=0.95, attribute_direction='decreasing', ci_type='one-sided'
        )

        # Mean should be lower bound (for reference)
        mean = np.mean(curves, axis=0)
        np.testing.assert_allclose(lower, mean, rtol=0.01)

        # Distance from mean to upper should be substantial
        upper_distance = upper - mean

        # Upper distance should be substantial (95th percentile above mean)
        assert np.mean(upper_distance) > 0.01

    def test_confidence_level_affects_percentile(self):
        """Different confidence levels should give different percentiles."""
        np.random.seed(42)
        n_bootstrap = 1000
        n_timepoints = 50

        base_curve = np.linspace(0, 0.3, n_timepoints)
        curves = np.random.normal(base_curve, 0.02, size=(n_bootstrap, n_timepoints))

        # 95% CI (95th percentile)
        _, upper_95 = _ci_from_replicate_curves(
            curves, confidence_level=0.95, attribute_direction='decreasing', ci_type='one-sided'
        )

        # 90% CI (90th percentile)
        _, upper_90 = _ci_from_replicate_curves(
            curves, confidence_level=0.90, attribute_direction='decreasing', ci_type='one-sided'
        )

        # 95% CI upper bound should be higher than 90% CI upper bound
        assert np.all(upper_95 > upper_90 - 1e-6)  # Allow small numerical tolerance
        assert np.mean(upper_95 - upper_90) > 0.003  # Should be meaningfully different

    def test_invalid_attribute_direction_raises(self):
        """Invalid attribute_direction should raise ValueError."""
        np.random.seed(42)
        curves = np.random.normal(0.1, 0.02, size=(100, 50))

        with pytest.raises(ValueError, match="attribute_direction must be"):
            _ci_from_replicate_curves(
                curves, confidence_level=0.95, attribute_direction='invalid'
            )


class TestInterpolateCrossing:
    """Test linear interpolation for specification crossing."""

    def test_exact_crossing(self):
        """When value exactly equals threshold, return that time."""
        time = np.array([0, 1, 2, 3, 4])
        values = np.array([0.0, 0.05, 0.10, 0.15, 0.20])
        threshold = 0.10

        # Crossing at index 2
        t_cross = _interpolate_crossing(time, values, threshold, idx=2)
        assert t_cross == 2.0

    def test_linear_interpolation(self):
        """Interpolate between two points."""
        time = np.array([0, 1, 2, 3, 4])
        values = np.array([0.0, 0.05, 0.12, 0.15, 0.20])
        threshold = 0.10

        # Crossing between index 1 and 2
        # At t=1: y=0.05, at t=2: y=0.12
        # Linear interpolation: t = 1 + (0.10 - 0.05) / (0.12 - 0.05) * (2 - 1)
        t_cross = _interpolate_crossing(time, values, threshold, idx=2)
        expected = 1 + (0.10 - 0.05) / (0.12 - 0.05)
        assert abs(t_cross - expected) < 1e-10

    def test_crossing_at_first_point(self):
        """When crossing at first point, return first time."""
        time = np.array([0, 1, 2, 3, 4])
        values = np.array([0.15, 0.20, 0.25, 0.30, 0.35])
        threshold = 0.10

        t_cross = _interpolate_crossing(time, values, threshold, idx=0)
        assert t_cross == 0.0

    def test_constant_values(self):
        """When values are constant, return left endpoint."""
        time = np.array([0, 1, 2, 3, 4])
        values = np.array([0.10, 0.10, 0.10, 0.10, 0.10])
        threshold = 0.10

        t_cross = _interpolate_crossing(time, values, threshold, idx=2)
        assert t_cross == 1.0  # Left endpoint of interval


class TestShelfLifeDecreasing:
    """Test shelf-life calculation for decreasing attributes (potency, monomer)."""

    def test_basic_decreasing_shelf_life(self):
        """Basic shelf-life calculation for decreasing attribute."""
        # Create synthetic data: degradation over time
        time = np.linspace(0, 365 * 2, 100)  # 2 years in days
        conversion_mean = np.linspace(0, 0.15, 100)  # 0-15% degradation
        conversion_lower = conversion_mean  # Mean (for reference)
        conversion_upper = conversion_mean + 0.03  # Upper bound degrades faster (conservative)

        spec_limit = 0.10  # 10% degradation limit

        result = calculate_shelf_life_ich_q1e(
            time=time,
            conversion_mean=conversion_mean,
            conversion_lower=conversion_lower,
            conversion_upper=conversion_upper,
            specification_limit=spec_limit,
            attribute_direction='decreasing',
            time_units='days'
        )

        # Basic checks
        assert result['attribute_direction'] == 'decreasing'
        assert result['method'] == 'ich_q1e_one_sided_ci'
        assert result['has_confidence_interval'] is True
        assert result['confidence_level'] == 0.95

        # Conservative shelf-life should be shorter than mean
        assert result['shelf_life_conservative'] < result['shelf_life_mean']

        # Both should be within [0, max_time]
        assert 0 < result['shelf_life_conservative'] <= time[-1]
        assert 0 < result['shelf_life_mean'] <= time[-1]

        # Alias should work
        assert result['shelf_life_ich_q1e'] == result['shelf_life_conservative']

    def test_conservative_shorter_for_decreasing(self):
        """Conservative shelf-life must be shorter than mean for decreasing."""
        time = np.linspace(0, 1000, 200)
        conversion_mean = time / 5000  # Linear degradation
        conversion_lower = conversion_mean  # Mean (for reference)
        conversion_upper = conversion_mean * 1.2  # Degrades faster (conservative)

        spec_limit = 0.10

        result = calculate_shelf_life_ich_q1e(
            time=time,
            conversion_mean=conversion_mean,
            conversion_lower=conversion_lower,
            conversion_upper=conversion_upper,
            specification_limit=spec_limit,
            attribute_direction='decreasing'
        )

        # Conservative must be strictly shorter (or equal if no CI)
        assert result['shelf_life_conservative'] <= result['shelf_life_mean']

        # Should be meaningfully shorter (not just numerical noise)
        reduction = (result['shelf_life_mean'] - result['shelf_life_conservative']) / result['shelf_life_mean']
        assert reduction > 0.15  # At least 15% reduction with this factor


class TestShelfLifeIncreasing:
    """Test shelf-life calculation for increasing attributes (aggregates, impurities)."""

    def test_basic_increasing_shelf_life(self):
        """Basic shelf-life calculation for increasing attribute."""
        time = np.linspace(0, 365 * 2, 100)
        conversion_mean = np.linspace(0, 0.08, 100)  # 0-8% aggregates
        conversion_lower = conversion_mean  # Mean (for reference)
        conversion_upper = conversion_mean + 0.02  # Upper bound increases faster (conservative)

        spec_limit = 0.05  # 5% aggregates limit

        result = calculate_shelf_life_ich_q1e(
            time=time,
            conversion_mean=conversion_mean,
            conversion_lower=conversion_lower,
            conversion_upper=conversion_upper,
            specification_limit=spec_limit,
            attribute_direction='increasing'
        )

        assert result['attribute_direction'] == 'increasing'
        assert result['method'] == 'ich_q1e_one_sided_ci'
        assert result['shelf_life_conservative'] < result['shelf_life_mean']

    def test_conservative_shorter_for_increasing(self):
        """Conservative shelf-life must be shorter for increasing attribute."""
        time = np.linspace(0, 1000, 200)
        conversion_mean = time / 10000
        conversion_lower = conversion_mean  # Mean (for reference)
        conversion_upper = conversion_mean * 1.3  # Increases faster (conservative)

        spec_limit = 0.05

        result = calculate_shelf_life_ich_q1e(
            time=time,
            conversion_mean=conversion_mean,
            conversion_lower=conversion_lower,
            conversion_upper=conversion_upper,
            specification_limit=spec_limit,
            attribute_direction='increasing'
        )

        assert result['shelf_life_conservative'] <= result['shelf_life_mean']
        reduction = (result['shelf_life_mean'] - result['shelf_life_conservative']) / result['shelf_life_mean']
        assert reduction > 0.20  # At least 20% reduction with this factor


class TestShelfLifeEdgeCases:
    """Test edge cases and error conditions."""

    def test_no_crossing_within_horizon(self):
        """When spec is not crossed, return inf with warning."""
        time = np.linspace(0, 1000, 100)
        conversion_mean = np.linspace(0, 0.05, 100)  # Only reaches 5%
        conversion_lower = conversion_mean  # Mean (for reference)
        conversion_upper = conversion_mean + 0.01  # Still below spec

        spec_limit = 0.10  # Never reached

        with pytest.warns(UserWarning, match="does not cross specification limit"):
            result = calculate_shelf_life_ich_q1e(
                time=time,
                conversion_mean=conversion_mean,
                conversion_lower=conversion_lower,
                conversion_upper=conversion_upper,
                specification_limit=spec_limit,
                attribute_direction='decreasing'
            )

        assert result['shelf_life_mean'] == np.inf
        assert result['shelf_life_conservative'] == np.inf

    def test_no_ci_provided(self):
        """When no CI provided, return mean only with warning."""
        time = np.linspace(0, 1000, 100)
        conversion_mean = np.linspace(0, 0.15, 100)

        with pytest.warns(UserWarning, match="No confidence interval provided"):
            result = calculate_shelf_life_ich_q1e(
                time=time,
                conversion_mean=conversion_mean,
                conversion_lower=None,
                conversion_upper=None,
                specification_limit=0.10,
                attribute_direction='decreasing'
            )

        assert result['has_confidence_interval'] is False
        assert result['method'] == 'mean_only'
        assert result['shelf_life_conservative'] == result['shelf_life_mean']

    def test_invalid_attribute_direction(self):
        """Invalid attribute_direction should raise ValueError."""
        time = np.linspace(0, 1000, 100)
        conversion = np.linspace(0, 0.15, 100)

        with pytest.raises(ValueError, match="attribute_direction must be"):
            calculate_shelf_life_ich_q1e(
                time=time,
                conversion_mean=conversion,
                conversion_lower=conversion,
                conversion_upper=conversion,
                specification_limit=0.10,
                attribute_direction='invalid'
            )

    def test_mismatched_array_shapes(self):
        """Mismatched array shapes should raise ValueError."""
        time = np.linspace(0, 1000, 100)
        conversion_mean = np.linspace(0, 0.15, 100)
        conversion_upper = np.linspace(0, 0.15, 50)  # Wrong length

        with pytest.raises(ValueError, match="must match time shape"):
            calculate_shelf_life_ich_q1e(
                time=time,
                conversion_mean=conversion_mean,
                conversion_lower=None,
                conversion_upper=conversion_upper,
                specification_limit=0.10,
                attribute_direction='decreasing'
            )

    def test_multidimensional_time_raises(self):
        """Time must be 1D array."""
        time = np.zeros((10, 10))
        conversion = np.zeros((10, 10))

        with pytest.raises(ValueError, match="time must be 1D array"):
            calculate_shelf_life_ich_q1e(
                time=time,
                conversion_mean=conversion,
                conversion_lower=None,
                conversion_upper=None,
                specification_limit=0.10,
                attribute_direction='decreasing'
            )


class TestTimeToSpecification:
    """Test user-friendly wrapper function."""

    def test_wrapper_with_prediction_result(self):
        """Test wrapper accepts PredictionResult."""
        # Create mock PredictionResult
        time = np.linspace(0, 1000, 100)
        conversion = np.linspace(0, 0.15, 100)
        lower = conversion  # Mean (for reference)
        upper = conversion + 0.02  # Upper bound (conservative: faster degradation)

        pred_result = PredictionResult(
            time=time,
            temperature=np.ones_like(time) * 298.15,
            conversion=conversion,
            conversion_ci=(lower, upper)
        )

        result = time_to_specification(
            pred_result,
            specification_limit=0.10,
            attribute_direction='decreasing'
        )

        assert 'shelf_life_ich_q1e' in result
        assert result['has_confidence_interval'] is True
        assert result['shelf_life_conservative'] < result['shelf_life_mean']

    def test_wrapper_without_ci(self):
        """Test wrapper with no confidence intervals."""
        time = np.linspace(0, 1000, 100)
        conversion = np.linspace(0, 0.15, 100)

        pred_result = PredictionResult(
            time=time,
            temperature=np.ones_like(time) * 298.15,
            conversion=conversion,
            conversion_ci=None
        )

        with pytest.warns(UserWarning):
            result = time_to_specification(
                pred_result,
                specification_limit=0.10,
                attribute_direction='decreasing'
            )

        assert result['has_confidence_interval'] is False
        assert result['shelf_life_conservative'] == result['shelf_life_mean']


class TestRegulatoryCompliance:
    """Integration tests to verify overall ICH Q1E compliance."""

    def test_full_workflow_ich_q1e_compliant(self):
        """
        Full workflow test: simulate data → calculate CI → determine shelf-life.
        Verify the entire process is ICH Q1E compliant.
        """
        # Simulate bootstrap replicates for potency degradation
        # with realistic variability where some replicates degrade faster
        np.random.seed(42)
        n_bootstrap = 500
        n_timepoints = 100
        time = np.linspace(0, 730, n_timepoints)  # 2 years in days

        # Create degradation curves with realistic variability
        # Some products degrade faster (lower bound will be conservative)
        base_degradation = time / 3650  # ~20% in 2 years

        # Add both random noise and systematic variation in degradation rate
        curves = []
        for _ in range(n_bootstrap):
            # Each replicate has slightly different degradation rate
            rate_factor = np.random.uniform(0.8, 1.2)  # 20% variation in rates
            curve = base_degradation * rate_factor + np.random.normal(0, 0.005, n_timepoints)
            curves.append(curve)
        curves = np.array(curves)

        # Calculate one-sided CI (ICH Q1E requirement)
        lower, upper = _ci_from_replicate_curves(
            curves, confidence_level=0.95, attribute_direction='decreasing'
        )

        mean = np.mean(curves, axis=0)

        # Calculate shelf-life
        result = calculate_shelf_life_ich_q1e(
            time=time,
            conversion_mean=mean,
            conversion_lower=lower,
            conversion_upper=upper,
            specification_limit=0.10,  # 10% degradation
            attribute_direction='decreasing',
            time_units='days'
        )

        # Verify ICH Q1E compliance
        assert result['method'] == 'ich_q1e_one_sided_ci'
        assert result['confidence_level'] == 0.95
        assert result['attribute_direction'] == 'decreasing'

        # Conservative estimate should be shorter or equal to mean
        # (For decreasing, lower bound should cross first or at same time)
        assert result['shelf_life_conservative'] <= result['shelf_life_mean']

        # Shelf-life should be reasonable (not inf or 0)
        assert 0 < result['shelf_life_conservative'] < time[-1]

        # With realistic variability, conservative should be meaningfully shorter
        # (but allow for small differences due to random variation)
        if result['shelf_life_conservative'] < result['shelf_life_mean']:
            reduction_pct = (
                (result['shelf_life_mean'] - result['shelf_life_conservative'])
                / result['shelf_life_mean']
            ) * 100
            # Should see at least 1% reduction with this variability
            assert reduction_pct >= 1

    def test_decreasing_vs_increasing_use_correct_bounds(self):
        """
        Verify that both decreasing and increasing use upper bound (95th percentile).
        """
        time = np.linspace(0, 1000, 100)
        mean = np.linspace(0, 0.15, 100)
        lower = mean  # Mean (for reference)
        upper = mean + 0.03  # Upper bound is worse (higher degradation/impurity)

        spec_limit = 0.10

        # Decreasing: should use upper bound (faster degradation = conservative)
        result_dec = calculate_shelf_life_ich_q1e(
            time=time,
            conversion_mean=mean,
            conversion_lower=lower,
            conversion_upper=upper,
            specification_limit=spec_limit,
            attribute_direction='decreasing'
        )

        # Increasing: should also use upper bound (faster increase = conservative)
        result_inc = calculate_shelf_life_ich_q1e(
            time=time,
            conversion_mean=mean,
            conversion_lower=lower,
            conversion_upper=upper,
            specification_limit=spec_limit,
            attribute_direction='increasing'
        )

        # Both should use the conservative bound (upper = 95th percentile)
        assert result_dec['shelf_life_conservative'] < result_dec['shelf_life_mean']
        assert result_inc['shelf_life_conservative'] < result_inc['shelf_life_mean']
