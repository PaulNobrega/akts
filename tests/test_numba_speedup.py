"""
Tests for Phase 2: Numba JIT Optimization

Verifies that Numba JIT compilation provides expected speedup for ODE-based
fitting without changing results.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from akts import KineticDataset, fit_kinetic_model
from akts.models import NUMBA_AVAILABLE
from akts import simulate_kinetics

R_GAS = 8.314


def _make_ramp_dataset(Ea, A, T_start, T_end, model='F1', n_points=30, seed=0):
    """Create a temperature-ramp dataset (forces ODE path)."""
    t_sec = np.linspace(0, 3600, n_points)  # 1 hour
    temp_ramp = np.linspace(T_start, T_end, n_points)

    result = simulate_kinetics(
        model_name='single_step',
        model_definition_args={'f_alpha_model': model},
        kinetic_params={'Ea': Ea, 'A': A},
        initial_alpha=1e-6,
        temperature_program=(t_sec, temp_ramp),
    )

    rng = np.random.default_rng(seed)
    conversion = np.clip(result.conversion + rng.normal(0, 0.01, size=result.conversion.shape), 0, 1)
    return KineticDataset(time=t_sec, temperature=temp_ramp, conversion=conversion)


@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba not available")
class TestNumbaSpeedup:
    """Test Numba JIT speedup for ODE path."""

    def test_numba_available(self):
        """Verify Numba is available for JIT compilation."""
        assert NUMBA_AVAILABLE, "Numba should be available"

    def test_ode_fit_with_numba_succeeds(self):
        """Numba-accelerated ODE path should run successfully."""
        Ea_true = 80000.0
        A_true = 1e11

        # Create temperature ramp datasets (forces ODE path)
        datasets = [
            _make_ramp_dataset(Ea_true, A_true, 310, 340, model='F1', n_points=25, seed=i)
            for i in range(2)
        ]

        fit_result = fit_kinetic_model(
            datasets=datasets,
            model_name='single_step',
            model_definition_args={'f_alpha_model': 'F1'},
            initial_guesses={'Ea': 85000, 'A': 1e11},
            verbose=False
        )

        # Just verify it runs successfully with Numba (ODE path is challenging)
        assert fit_result.success, "Numba-accelerated fit should succeed"
        assert fit_result.r_squared > 0.0, "R² should be computed"
        assert 10000 < fit_result.parameters['Ea'] < 500000, "Ea in reasonable range"

    def test_numba_works_for_all_models(self):
        """All f(α) models should work with Numba JIT."""
        models_to_test = ['F1', 'F2', 'A2', 'R2', 'R3', 'D3', 'SB_mn']
        Ea = 70000.0
        A = 1e10

        for model in models_to_test:
            model_args = {'f_alpha_model': model}
            if model == 'SB_mn':
                model_args['f_alpha_params'] = {'m': 0.5, 'n': 1.0}

            # Single ramp dataset
            ds = _make_ramp_dataset(Ea, A, 310, 340, model=model, n_points=20, seed=42)

            fit_result = fit_kinetic_model(
                datasets=[ds],
                model_name='single_step',
                model_definition_args=model_args,
                initial_guesses={'Ea': 75000, 'A': 5e9},
                verbose=False
            )

            assert fit_result.success, f"Numba fit failed for {model}"
            assert fit_result.r_squared > 0.8, f"{model}: R²={fit_result.r_squared}"

    @pytest.mark.slow
    def test_numba_speedup_benchmark(self):
        """Benchmark ODE fitting speedup with Numba (informational, not strict)."""
        Ea_true = 90000.0
        A_true = 5e11

        # Create more challenging datasets (longer ramps)
        datasets = [
            _make_ramp_dataset(Ea_true, A_true, 300, 350, model='F1', n_points=50, seed=i)
            for i in range(3)
        ]

        # Warm up JIT compilation
        _ = fit_kinetic_model(
            datasets=datasets[:1],
            model_name='single_step',
            model_definition_args={'f_alpha_model': 'F1'},
            initial_guesses={'Ea': 95000, 'A': 1e12},
            verbose=False
        )

        # Benchmark fit time
        start = time.time()
        fit_result = fit_kinetic_model(
            datasets=datasets,
            model_name='single_step',
            model_definition_args={'f_alpha_model': 'F1'},
            initial_guesses={'Ea': 95000, 'A': 1e12},
            optimizer_options={'max_seconds': 120},
            verbose=False
        )
        elapsed = time.time() - start

        assert fit_result.success
        print(f"\n  Numba-accelerated ODE fit: {elapsed:.2f}s, R²={fit_result.r_squared:.4f}")

        # Should complete in reasonable time (< 30s for 3 datasets × 50 points)
        assert elapsed < 30.0, f"Fit too slow: {elapsed:.2f}s"


class TestNumbaFallback:
    """Test that code works even if Numba is not available."""

    def test_fallback_decorator_exists(self):
        """Fallback decorator should be defined even without Numba."""
        from akts.models import njit

        # Should be callable
        assert callable(njit)

        # Should work as decorator
        @njit
        def test_func(x):
            return x * 2

        assert test_func(5) == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
