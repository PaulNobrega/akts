"""
Tests for plotting convenience functions.

Tests all plotting helpers: Ea(α), fit overlay, bootstrap CI bands,
Arrhenius, parameter distributions, and multi-temperature data.
"""

import numpy as np
import pytest
import tempfile
import os
from pathlib import Path

# Set matplotlib backend to Agg for headless environments
import matplotlib
matplotlib.use('Agg')

from akts import (KineticDataset, fit_kinetic_model, predict_conversion,
                  run_bootstrap, run_friedman)
from akts import (plot_ea_vs_alpha, plot_fit_overlay, plot_bootstrap_ci_bands,
                  plot_arrhenius, plot_parameter_distributions, plot_multi_temperature_data)
from akts.datatypes import IsoResult, BootstrapResult, PredictionResult


class TestPlotEaVsAlpha:
    """Test Ea(α) plotting."""

    @pytest.fixture
    def iso_result_friedman(self):
        """Create datasets and run Friedman analysis."""
        np.random.seed(42)

        # Generate synthetic data at multiple heating rates
        heating_rates = [5, 10, 20]  # K/min
        datasets = []

        for beta in heating_rates:
            t = np.linspace(0, 600, 50)
            T = 298.15 + beta * t / 60  # Linear heating
            Ea_true = 85000
            A_true = 1e11
            alpha = 1 - np.exp(-A_true * np.exp(-Ea_true / (8.314 * T)) * t)
            alpha = np.clip(alpha + np.random.normal(0, 0.01, len(t)), 0, 1)

            datasets.append(KineticDataset(
                time=t,
                temperature=T,
                conversion=alpha,
                heating_rate=beta
            ))

        return run_friedman(datasets, alpha_levels=np.linspace(0.1, 0.9, 9))

    def test_basic_ea_plot(self, iso_result_friedman):
        """Test basic Ea(α) plot generation."""
        fig = plot_ea_vs_alpha(iso_result_friedman)

        assert fig is not None
        axes = fig.get_axes()
        assert len(axes) == 1

        ax = axes[0]
        assert 'Conversion' in ax.get_xlabel()
        assert 'Activation Energy' in ax.get_ylabel()

    def test_ea_plot_without_error_bars(self, iso_result_friedman):
        """Test Ea(α) plot without error bars."""
        fig = plot_ea_vs_alpha(iso_result_friedman, show_error_bars=False)

        assert fig is not None
        axes = fig.get_axes()
        # Should only have line plot, no error bars
        lines = axes[0].get_lines()
        assert len(lines) >= 1

    def test_ea_plot_custom_figsize(self, iso_result_friedman):
        """Test custom figure size."""
        fig = plot_ea_vs_alpha(iso_result_friedman, figsize=(8, 5))

        assert fig is not None
        # Matplotlib figsize returns inches
        width, height = fig.get_size_inches()
        assert abs(width - 8) < 0.1
        assert abs(height - 5) < 0.1

    def test_ea_plot_on_existing_axes(self, iso_result_friedman):
        """Test plotting on existing axes."""
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()

        returned_fig = plot_ea_vs_alpha(iso_result_friedman, ax=ax)

        assert returned_fig == fig
        plt.close(fig)


class TestPlotFitOverlay:
    """Test fit overlay plotting."""

    @pytest.fixture
    def datasets_and_fit(self):
        """Create datasets and fit result."""
        np.random.seed(123)

        # Generate data at multiple temperatures
        temps = [313.15, 323.15, 333.15]
        datasets = []

        for T in temps:
            t = np.linspace(0, 3600, 30)
            k = 1e11 * np.exp(-85000 / (8.314 * T))
            alpha = 1.0 - np.exp(-k * t)
            alpha = np.clip(alpha + np.random.normal(0, 0.005, len(t)), 0, 1)

            datasets.append(KineticDataset(
                time=t,
                temperature=np.full_like(t, T),
                conversion=alpha
            ))

        fit_result = fit_kinetic_model(
            datasets=datasets,
            model_name="single_step",
            model_definition_args={'f_alpha_model': 'F1'},
            initial_guesses={'Ea': 85000, 'A': 1e11}
        )

        return datasets, fit_result

    def test_fit_overlay_without_prediction(self, datasets_and_fit):
        """Test fit overlay with data only (no prediction line)."""
        datasets, fit_result = datasets_and_fit

        fig = plot_fit_overlay(datasets, fit_result, prediction=None)

        assert fig is not None
        axes = fig.get_axes()
        assert len(axes) == 1
        assert 'Time' in axes[0].get_xlabel()
        assert 'Degradation' in axes[0].get_ylabel()

    def test_fit_overlay_with_prediction(self, datasets_and_fit):
        """Test fit overlay with model prediction."""
        datasets, fit_result = datasets_and_fit

        # Generate prediction at middle temperature
        prediction = predict_conversion(
            kinetic_description=fit_result,
            temperature_program=lambda t: 323.15,
            simulation_time_sec=np.linspace(0, 3600, 100)
        )

        fig = plot_fit_overlay(datasets, fit_result, prediction=prediction)

        assert fig is not None
        axes = fig.get_axes()
        # Should have scatter plots (data) + line plot (prediction)
        lines = axes[0].get_lines()
        assert len(lines) >= 1  # At least the prediction line

    def test_fit_overlay_with_per_dataset_predictions_and_confidence_intervals(self, datasets_and_fit):
        datasets, fit_result = datasets_and_fit
        predictions = [
            PredictionResult(
                time=dataset.time,
                temperature=dataset.temperature,
                conversion=dataset.conversion,
                conversion_ci=(
                    np.clip(dataset.conversion - 0.02, 0, 1),
                    np.clip(dataset.conversion + 0.02, 0, 1),
                ),
            )
            for dataset in datasets
        ]

        fig = plot_fit_overlay(
            datasets, fit_result, prediction=predictions, time_units='hours'
        )

        ax = fig.axes[0]
        assert len(ax.lines) == len(datasets)
        assert len(ax.collections) == 2 * len(datasets)  # data markers + CI bands
        legend_labels = [text.get_text() for text in ax.get_legend().get_texts()]
        assert sum('95% CI at' in label for label in legend_labels) == len(datasets)

    def test_fit_overlay_time_units(self, datasets_and_fit):
        """Test different time units."""
        datasets, fit_result = datasets_and_fit

        fig = plot_fit_overlay(datasets, fit_result, time_units='hours')

        assert fig is not None
        axes = fig.get_axes()
        xlabel = axes[0].get_xlabel().lower()
        assert 'hour' in xlabel


class TestPlotBootstrapCIBands:
    """Test bootstrap CI band plotting."""

    @pytest.fixture
    def prediction_with_ci(self):
        """Create prediction with bootstrap CI."""
        np.random.seed(99)

        # Simple dataset
        t = np.linspace(0, 3600, 30)
        T = np.full_like(t, 323.15)
        k = 1e11 * np.exp(-85000 / (8.314 * 323.15))
        alpha = 1.0 - np.exp(-k * t)
        alpha = np.clip(alpha + np.random.normal(0, 0.005, len(t)), 0, 1)

        dataset = KineticDataset(time=t, temperature=T, conversion=alpha)

        fit_result = fit_kinetic_model(
            datasets=[dataset],
            model_name="single_step",
            model_definition_args={'f_alpha_model': 'F1'},
            initial_guesses={'Ea': 85000, 'A': 1e11}
        )

        bootstrap_result = run_bootstrap(
            fit_result=fit_result,
            datasets=[dataset],
            n_iterations=30,
            optimizer_options={'method': 'L-BFGS-B'}
        )

        prediction = predict_conversion(
            kinetic_description=fit_result,
            bootstrap_result=bootstrap_result,
            temperature_program=lambda t: 298.15,
            simulation_time_sec=np.linspace(0, 7200, 50)
        )

        return prediction

    def test_bootstrap_ci_plot(self, prediction_with_ci):
        """Test basic bootstrap CI plot."""
        fig = plot_bootstrap_ci_bands(prediction_with_ci)

        assert fig is not None
        axes = fig.get_axes()
        assert len(axes) == 1
        assert 'Confidence Interval' in axes[0].get_title()

    def test_bootstrap_ci_time_units(self, prediction_with_ci):
        """Test different time units."""
        fig = plot_bootstrap_ci_bands(prediction_with_ci, time_units='days')

        assert fig is not None
        axes = fig.get_axes()
        assert 'days' in axes[0].get_xlabel()

    def test_bootstrap_ci_requires_ci_data(self):
        """Test that function raises error without CI data."""
        from akts.datatypes import PredictionResult

        # Prediction without CI
        prediction = PredictionResult(
            time=np.linspace(0, 3600, 50),
            temperature=np.full(50, 298.15),
            conversion=np.linspace(0, 0.1, 50),
            conversion_ci=None
        )

        with pytest.raises(ValueError, match="conversion_ci is None"):
            plot_bootstrap_ci_bands(prediction)


class TestPlotArrhenius:
    """Test Arrhenius plot."""

    @pytest.fixture
    def fit_result_with_ea_a(self):
        """Create fit result with Ea and A parameters."""
        np.random.seed(55)

        # Multi-temperature dataset
        temps = [313.15, 323.15, 333.15]
        datasets = []

        for T in temps:
            t = np.linspace(0, 3600, 30)
            k = 1e11 * np.exp(-85000 / (8.314 * T))
            alpha = 1.0 - np.exp(-k * t)
            alpha = np.clip(alpha + np.random.normal(0, 0.005, len(t)), 0, 1)

            datasets.append(KineticDataset(
                time=t,
                temperature=np.full_like(t, T),
                conversion=alpha
            ))

        return fit_kinetic_model(
            datasets=datasets,
            model_name="single_step",
            model_definition_args={'f_alpha_model': 'F1'},
            initial_guesses={'Ea': 85000, 'A': 1e11}
        )

    def test_arrhenius_plot(self, fit_result_with_ea_a):
        """Test basic Arrhenius plot."""
        fig = plot_arrhenius(fit_result_with_ea_a)

        assert fig is not None
        axes = fig.get_axes()
        assert len(axes) == 1
        assert '1000/T' in axes[0].get_xlabel()
        assert 'ln(k)' in axes[0].get_ylabel()

    def test_arrhenius_requires_ea_a(self):
        """Test that function requires Ea and A parameters."""
        from akts.datatypes import FitResult

        # Fit result without Ea/A (only has k parameter)
        fit_result = FitResult(
            model_name='dummy',
            parameters={'k': 0.1},
            success=True,
            message='Success',
            rss=0.1,
            n_datapoints=100,
            n_parameters=1,
            r_squared=0.9,
            aic=10,
            bic=12,
            model_definition_args={}
        )

        # Should raise ValueError due to missing Ea and A
        with pytest.raises(ValueError) as exc_info:
            plot_arrhenius(fit_result)

        assert "Ea" in str(exc_info.value) and "A" in str(exc_info.value)


class TestPlotParameterDistributions:
    """Test parameter distribution plotting."""

    @pytest.fixture
    def bootstrap_result(self):
        """Create sample bootstrap result."""
        n_bootstrap = 100
        return BootstrapResult(
            model_name='single_step',
            parameter_distributions={
                'Ea': np.random.normal(85000, 3000, n_bootstrap),
                'A': np.random.lognormal(np.log(1e11), 0.5, n_bootstrap)
            },
            parameter_ci={
                'Ea': (80000, 90000),
                'A': (5e10, 2e11)
            },
            n_iterations=n_bootstrap,
            confidence_level=0.95
        )

    def test_plot_all_parameters(self, bootstrap_result):
        """Test plotting all parameters."""
        fig = plot_parameter_distributions(bootstrap_result)

        assert fig is not None
        axes = fig.get_axes()
        assert len(axes) == 2  # Ea and A

    def test_plot_selected_parameters(self, bootstrap_result):
        """Test plotting specific parameters only."""
        fig = plot_parameter_distributions(bootstrap_result, parameters=['Ea'])

        assert fig is not None
        axes = fig.get_axes()
        assert len(axes) == 1

    def test_plot_custom_bins(self, bootstrap_result):
        """Test custom histogram bins."""
        fig = plot_parameter_distributions(bootstrap_result, bins=50)

        assert fig is not None


class TestPlotMultiTemperatureData:
    """Test multi-temperature data plotting."""

    @pytest.fixture
    def multi_temp_datasets(self):
        """Create datasets at multiple temperatures."""
        np.random.seed(77)

        temps = [313.15, 323.15, 333.15, 343.15]
        datasets = []

        for T in temps:
            t = np.linspace(0, 3600, 25)
            k = 1e11 * np.exp(-85000 / (8.314 * T))
            alpha = 1.0 - np.exp(-k * t)
            alpha = np.clip(alpha + np.random.normal(0, 0.005, len(t)), 0, 1)

            datasets.append(KineticDataset(
                time=t,
                temperature=np.full_like(t, T),
                conversion=alpha
            ))

        return datasets

    def test_multi_temp_plot(self, multi_temp_datasets):
        """Test basic multi-temperature plot."""
        fig = plot_multi_temperature_data(multi_temp_datasets)

        assert fig is not None
        axes = fig.get_axes()
        assert len(axes) == 1

        # Should have one line per temperature
        legend = axes[0].get_legend()
        assert legend is not None

    def test_multi_temp_time_units(self, multi_temp_datasets):
        """Test different time units."""
        fig = plot_multi_temperature_data(multi_temp_datasets, time_units='hours')

        assert fig is not None
        axes = fig.get_axes()
        assert 'hours' in axes[0].get_xlabel()


class TestPlottingIntegration:
    """Integration tests combining multiple plotting functions."""

    def test_save_plots_to_file(self):
        """Test saving plots to files."""
        import matplotlib.pyplot as plt

        # Simple dataset
        np.random.seed(200)
        t = np.linspace(0, 3600, 30)
        T = np.full_like(t, 323.15)
        k = 1e11 * np.exp(-85000 / (8.314 * 323.15))
        alpha = 1.0 - np.exp(-k * t)
        alpha = np.clip(alpha + np.random.normal(0, 0.005, len(t)), 0, 1)

        dataset = KineticDataset(time=t, temperature=T, conversion=alpha)

        fit_result = fit_kinetic_model(
            datasets=[dataset],
            model_name="single_step",
            model_definition_args={'f_alpha_model': 'F1'},
            initial_guesses={'Ea': 85000, 'A': 1e11}
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            # Test Arrhenius plot save
            fig = plot_arrhenius(fit_result)
            path = os.path.join(tmpdir, 'arrhenius.png')
            fig.savefig(path)
            assert os.path.exists(path)
            plt.close(fig)

            # Test multi-temp plot save
            fig = plot_multi_temperature_data([dataset])
            path = os.path.join(tmpdir, 'data.png')
            fig.savefig(path)
            assert os.path.exists(path)
            plt.close(fig)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
