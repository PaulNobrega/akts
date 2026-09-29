"""
Tests for tabulated export functionality.

Tests DataFrame and CSV export methods for PredictionResult and BootstrapResult,
plus the export_prediction_report() convenience function.
"""

import numpy as np
import pytest
import tempfile
import os
from pathlib import Path

from akts import KineticDataset, fit_kinetic_model, predict_conversion, run_bootstrap
from akts.datatypes import PredictionResult, BootstrapResult
from akts import export_prediction_report


class TestPredictionResultExport:
    """Test PredictionResult.to_dataframe() and to_csv()."""

    @pytest.fixture
    def prediction_without_ci(self):
        """Create prediction result without confidence intervals."""
        time = np.linspace(0, 3600, 50)
        temp = np.full_like(time, 298.15)
        conversion = 0.05 * (1 - np.exp(-time / 1800))

        return PredictionResult(
            time=time,
            temperature=temp,
            conversion=conversion,
            conversion_ci=None
        )

    @pytest.fixture
    def prediction_with_ci(self):
        """Create prediction result with confidence intervals."""
        time = np.linspace(0, 3600, 50)
        temp = np.full_like(time, 298.15)
        conversion = 0.05 * (1 - np.exp(-time / 1800))
        ci_lower = conversion * 0.9
        ci_upper = conversion * 1.1

        return PredictionResult(
            time=time,
            temperature=temp,
            conversion=conversion,
            conversion_ci=(ci_lower, ci_upper)
        )

    def test_to_dataframe_without_ci(self, prediction_without_ci):
        """Test DataFrame export without confidence intervals."""
        df = prediction_without_ci.to_dataframe()

        # Check columns
        assert 'time' in df.columns
        assert 'temperature' in df.columns
        assert 'conversion' in df.columns
        assert 'conversion_ci_lower' not in df.columns
        assert 'conversion_ci_upper' not in df.columns

        # Check data integrity
        assert len(df) == len(prediction_without_ci.time)
        assert np.allclose(df['time'].values, prediction_without_ci.time)
        assert np.allclose(df['conversion'].values, prediction_without_ci.conversion)

    def test_to_dataframe_with_ci(self, prediction_with_ci):
        """Test DataFrame export with confidence intervals."""
        df = prediction_with_ci.to_dataframe()

        # Check all columns present
        assert 'time' in df.columns
        assert 'temperature' in df.columns
        assert 'conversion' in df.columns
        assert 'conversion_ci_lower' in df.columns
        assert 'conversion_ci_upper' in df.columns

        # Check CI data
        assert np.allclose(df['conversion_ci_lower'].values, prediction_with_ci.conversion_ci[0])
        assert np.allclose(df['conversion_ci_upper'].values, prediction_with_ci.conversion_ci[1])

    def test_to_csv(self, prediction_with_ci):
        """Test CSV export."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, 'prediction.csv')

            prediction_with_ci.to_csv(csv_path)

            # Check file exists
            assert os.path.exists(csv_path)

            # Read back and verify
            import pandas as pd
            df = pd.read_csv(csv_path)

            assert len(df) == len(prediction_with_ci.time)
            assert 'conversion_ci_lower' in df.columns


class TestBootstrapResultExport:
    """Test BootstrapResult.to_dataframe() and summary_dataframe()."""

    @pytest.fixture
    def bootstrap_result(self):
        """Create sample bootstrap result."""
        n_bootstrap = 100
        return BootstrapResult(
            model_name='single_step',
            parameter_distributions={
                'Ea': np.random.normal(85000, 5000, n_bootstrap),
                'A': np.random.lognormal(np.log(1e11), 0.5, n_bootstrap)
            },
            parameter_ci={
                'Ea': (80000, 90000),
                'A': (5e10, 2e11)
            },
            n_iterations=n_bootstrap,
            confidence_level=0.95
        )

    def test_to_dataframe(self, bootstrap_result):
        """Test parameter distributions export."""
        df = bootstrap_result.to_dataframe()

        # Check columns
        assert 'Ea' in df.columns
        assert 'A' in df.columns

        # Check dimensions
        assert len(df) == bootstrap_result.n_iterations
        assert len(df.columns) == len(bootstrap_result.parameter_distributions)

        # Check data
        assert np.allclose(df['Ea'].values, bootstrap_result.parameter_distributions['Ea'])

    def test_summary_dataframe(self, bootstrap_result):
        """Test CI summary export."""
        summary = bootstrap_result.summary_dataframe()

        # Check structure
        assert 'parameter' in summary.columns
        assert 'median' in summary.columns
        assert 'ci_lower' in summary.columns
        assert 'ci_upper' in summary.columns
        assert 'confidence_level' in summary.columns

        # Check content
        assert len(summary) == 2  # Ea and A
        assert set(summary['parameter']) == {'Ea', 'A'}
        assert all(summary['confidence_level'] == 0.95)


class TestExportPredictionReport:
    """Test export_prediction_report() convenience function."""

    @pytest.fixture
    def simple_dataset(self):
        """Create simple dataset for integration testing."""
        np.random.seed(80)
        t = np.linspace(0, 3600, 30)
        T = np.full_like(t, 323.15)
        k = 1e11 * np.exp(-85000 / (8.314 * 323.15))
        alpha = 1.0 - np.exp(-k * t)
        alpha = alpha + np.random.normal(0, 0.005, len(t))
        alpha = np.clip(alpha, 0, 1)

        return KineticDataset(time=t, temperature=T, conversion=alpha)

    def test_export_prediction_only(self, simple_dataset):
        """Test exporting prediction without bootstrap."""
        fit_result = fit_kinetic_model(
            datasets=[simple_dataset],
            model_name="single_step",
            model_definition_args={'f_alpha_model': 'F1'},
            initial_guesses={'Ea': 85000, 'A': 1e11}
        )

        prediction = predict_conversion(
            kinetic_description=fit_result,
            temperature_program=lambda t: 298.15,
            simulation_time_sec=np.linspace(0, 7200, 50)
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path_prefix = os.path.join(tmpdir, 'test')

            files = export_prediction_report(
                prediction=prediction,
                path_prefix=path_prefix,
                include_plot=False  # Skip plot for speed
            )

            # Check files created
            assert 'prediction_csv' in files
            assert os.path.exists(files['prediction_csv'])
            assert 'bootstrap_csv' not in files  # No bootstrap provided

    def test_export_with_bootstrap(self, simple_dataset):
        """Test exporting prediction with bootstrap."""
        fit_result = fit_kinetic_model(
            datasets=[simple_dataset],
            model_name="single_step",
            model_definition_args={'f_alpha_model': 'F1'},
            initial_guesses={'Ea': 85000, 'A': 1e11}
        )

        bootstrap_result = run_bootstrap(
            fit_result=fit_result,
            datasets=[simple_dataset],
            n_iterations=4,  # Enough replicates to exercise bootstrap export
            optimizer_options={'method': 'L-BFGS-B'},
            n_jobs=2
        )

        prediction = predict_conversion(
            kinetic_description=fit_result,
            bootstrap_result=bootstrap_result,
            temperature_program=lambda t: 298.15,
            simulation_time_sec=np.linspace(0, 7200, 50)
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path_prefix = os.path.join(tmpdir, 'test')

            files = export_prediction_report(
                prediction=prediction,
                bootstrap_result=bootstrap_result,
                path_prefix=path_prefix,
                include_plot=False
            )

            # Check both files created
            assert 'prediction_csv' in files
            assert 'bootstrap_csv' in files
            assert os.path.exists(files['prediction_csv'])
            assert os.path.exists(files['bootstrap_csv'])

            # Verify bootstrap summary
            import pandas as pd
            summary = pd.read_csv(files['bootstrap_csv'])
            assert 'Ea' in summary['parameter'].values
            assert 'A' in summary['parameter'].values

    def test_export_with_plot(self, simple_dataset):
        """Test plot generation."""
        fit_result = fit_kinetic_model(
            datasets=[simple_dataset],
            model_name="single_step",
            model_definition_args={'f_alpha_model': 'F1'},
            initial_guesses={'Ea': 85000, 'A': 1e11}
        )

        prediction = predict_conversion(
            kinetic_description=fit_result,
            temperature_program=lambda t: 298.15,
            simulation_time_sec=np.linspace(0, 7200, 50)
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path_prefix = os.path.join(tmpdir, 'test')

            files = export_prediction_report(
                prediction=prediction,
                path_prefix=path_prefix,
                include_plot=True,
                time_units='hours'
            )

            # Check plot created
            assert 'plot' in files
            assert os.path.exists(files['plot'])
            assert files['plot'].endswith('.png')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
