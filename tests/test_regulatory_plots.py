"""
Tests for regulatory-compliant plotting functions.

Verifies that ICH Q1E plots are generated correctly with all required elements.
"""

import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing

from akts import KineticDataset
from akts.datatypes import PredictionResult
from akts.regulatory_plots import create_regulatory_shelf_life_plot, create_regulatory_comparison_plot


class TestRegulatoryShelfLifePlot:
    """Test regulatory shelf-life plot generation."""

    @pytest.fixture
    def prediction_result(self):
        """Create sample prediction result for testing."""
        # 24 months of prediction data
        time_sec = np.linspace(0, 24 * 30.44 * 24 * 3600, 100)
        conversion = 0.05 * (1.0 - np.exp(-time_sec / (12 * 30.44 * 24 * 3600)))

        # Create confidence intervals
        ci_lower = conversion * 0.8
        ci_upper = conversion * 1.2

        return PredictionResult(
            time=time_sec,
            temperature=np.full_like(time_sec, 298.15),
            conversion=conversion,
            conversion_ci=(ci_lower, ci_upper)
        )

    def test_basic_plot_generation(self, prediction_result):
        """Test that basic plot is generated without errors."""
        shelf_life_mean_sec = 18 * 30.44 * 24 * 3600  # 18 months
        shelf_life_lower_sec = 15 * 30.44 * 24 * 3600  # 15 months

        fig = create_regulatory_shelf_life_plot(
            prediction=prediction_result,
            target_conversion=0.05,
            shelf_life_mean_sec=shelf_life_mean_sec,
            shelf_life_lower_sec=shelf_life_lower_sec
        )

        assert fig is not None
        assert len(fig.axes) == 1

        # Check that figure was created
        ax = fig.axes[0]
        assert ax.get_xlabel() == 'Time (months)'
        assert ax.get_ylabel() == 'Degradation (%)'

    def test_plot_with_all_elements(self, prediction_result):
        """Test plot with all regulatory elements."""
        shelf_life_mean_sec = 18 * 30.44 * 24 * 3600
        shelf_life_lower_sec = 15 * 30.44 * 24 * 3600
        study_duration_sec = 12 * 30.44 * 24 * 3600
        ich_ceiling_sec = 24 * 30.44 * 24 * 3600

        fig = create_regulatory_shelf_life_plot(
            prediction=prediction_result,
            target_conversion=0.05,
            shelf_life_mean_sec=shelf_life_mean_sec,
            shelf_life_lower_sec=shelf_life_lower_sec,
            study_duration_sec=study_duration_sec,
            ich_ceiling_sec=ich_ceiling_sec,
            storage_temp_K=298.15
        )

        assert fig is not None

        # Check that legend has multiple entries
        ax = fig.axes[0]
        legend = ax.get_legend()
        assert legend is not None
        assert len(legend.get_texts()) >= 5  # Prediction, CI, threshold, mean, conservative
        ci_edge_lines = [line for line in ax.lines if line.get_label() == '_nolegend_']
        assert len(ci_edge_lines) == 2
        assert np.allclose(ci_edge_lines[0].get_ydata(), prediction_result.conversion_ci[0] * 100)
        assert np.allclose(ci_edge_lines[1].get_ydata(), prediction_result.conversion_ci[1] * 100)

    def test_plot_without_confidence_interval(self):
        """Test plot when no confidence interval is available."""
        time_sec = np.linspace(0, 24 * 30.44 * 24 * 3600, 100)
        conversion = 0.05 * (1.0 - np.exp(-time_sec / (12 * 30.44 * 24 * 3600)))

        prediction = PredictionResult(
            time=time_sec,
            temperature=np.full_like(time_sec, 298.15),
            conversion=conversion,
            conversion_ci=None  # No CI
        )

        shelf_life_mean_sec = 18 * 30.44 * 24 * 3600

        fig = create_regulatory_shelf_life_plot(
            prediction=prediction,
            target_conversion=0.05,
            shelf_life_mean_sec=shelf_life_mean_sec,
            shelf_life_lower_sec=None
        )

        assert fig is not None
        # Should still create a valid plot without CI

    def test_plot_exceeding_ceiling(self, prediction_result):
        """Test plot when shelf-life exceeds ICH ceiling (warning zone)."""
        shelf_life_mean_sec = 30 * 30.44 * 24 * 3600  # 30 months
        shelf_life_lower_sec = 27 * 30.44 * 24 * 3600  # 27 months
        study_duration_sec = 12 * 30.44 * 24 * 3600  # 12 months
        ich_ceiling_sec = 24 * 30.44 * 24 * 3600  # 24 months ceiling

        fig = create_regulatory_shelf_life_plot(
            prediction=prediction_result,
            target_conversion=0.05,
            shelf_life_mean_sec=shelf_life_mean_sec,
            shelf_life_lower_sec=shelf_life_lower_sec,
            study_duration_sec=study_duration_sec,
            ich_ceiling_sec=ich_ceiling_sec
        )

        assert fig is not None
        # Warning zone should be added (orange shaded region)


class TestRegulatoryComparisonPlot:
    """Test regulatory comparison plot (observed vs predicted)."""

    @pytest.fixture
    def datasets(self):
        """Create sample datasets at different temperatures."""
        datasets = []
        for temp in [313.15, 323.15, 333.15]:
            t = np.linspace(0, 6 * 30.44 * 24 * 3600, 30)
            k = 1e11 * np.exp(-85000 / (8.314 * temp))
            alpha = 1.0 - np.exp(-k * t)
            alpha = alpha + np.random.normal(0, 0.01, len(t))
            alpha = np.clip(alpha, 0, 1)

            datasets.append(KineticDataset(
                time=t,
                temperature=np.full_like(t, temp),
                conversion=alpha
            ))

        return datasets

    @pytest.fixture
    def prediction_at_storage(self):
        """Create prediction at storage temperature."""
        time_sec = np.linspace(0, 24 * 30.44 * 24 * 3600, 100)
        conversion = 0.05 * (1.0 - np.exp(-time_sec / (15 * 30.44 * 24 * 3600)))

        return PredictionResult(
            time=time_sec,
            temperature=np.full_like(time_sec, 298.15),
            conversion=conversion,
            conversion_ci=None
        )

    def test_comparison_plot_generation(self, datasets, prediction_at_storage):
        """Test that comparison plot is generated correctly."""
        fig = create_regulatory_comparison_plot(
            datasets=datasets,
            prediction=prediction_at_storage,
            storage_temp_K=298.15
        )

        assert fig is not None
        assert len(fig.axes) == 1

        ax = fig.axes[0]
        assert ax.get_xlabel() == 'Time (months)'
        assert ax.get_ylabel() == 'Degradation (%)'

        # Should have both scatter (data) and line (prediction)
        assert len(ax.lines) > 0  # Prediction line
        assert len(ax.collections) > 0  # Scatter points

    def test_comparison_plot_with_ci(self, datasets):
        """Test comparison plot with confidence intervals."""
        time_sec = np.linspace(0, 24 * 30.44 * 24 * 3600, 100)
        conversion = 0.05 * (1.0 - np.exp(-time_sec / (15 * 30.44 * 24 * 3600)))
        ci_lower = conversion * 0.8
        ci_upper = conversion * 1.2

        prediction = PredictionResult(
            time=time_sec,
            temperature=np.full_like(time_sec, 298.15),
            conversion=conversion,
            conversion_ci=(ci_lower, ci_upper)
        )

        fig = create_regulatory_comparison_plot(
            datasets=datasets,
            prediction=prediction,
            storage_temp_K=298.15
        )

        assert fig is not None
        # Should include CI shading


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
