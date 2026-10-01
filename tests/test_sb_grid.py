"""
Tests for SB grid search functionality.
"""
import pytest
import numpy as np
from akts.models import generate_sb_grid_models, f_sb_mn, get_model_info, model_display_name


def test_generate_sb_grid_models_default():
    """Test default grid generation (m,n = 0-3)."""
    grid = generate_sb_grid_models()
    assert len(grid) == 16, "Should generate 16 models for 4x4 grid"
    assert 'SB_m0_n0' in grid
    assert 'SB_m3_n3' in grid
    assert 'SB_m1_n2' in grid


def test_generate_sb_grid_models_custom():
    """Test custom grid generation."""
    grid = generate_sb_grid_models(m_range=range(2), n_range=range(3))
    assert len(grid) == 6, "Should generate 6 models for 2x3 grid"
    assert 'SB_m0_n0' in grid
    assert 'SB_m1_n2' in grid
    assert 'SB_m2_n0' not in grid  # m only goes to 1


def test_sb_grid_model_function():
    """Test that SB function evaluates correctly with fixed m,n."""
    # Test SB(m=0, n=1) should be equivalent to F1: f(α) = (1-α)^1
    alpha = 0.5
    params = {'m': 0.0, 'n': 1.0}
    result = f_sb_mn(alpha, params)
    expected = 0.5  # (1-0.5)^1 = 0.5
    assert np.isclose(result, expected), f"Expected {expected}, got {result}"

    # Test SB(m=1, n=0) - power law: f(α) = α^1
    params = {'m': 1.0, 'n': 0.0}
    result = f_sb_mn(alpha, params)
    expected = 0.5  # 0.5^1 = 0.5
    assert np.isclose(result, expected), f"Expected {expected}, got {result}"

    # Test SB(m=1, n=1) - autocatalytic: f(α) = α * (1-α)
    params = {'m': 1.0, 'n': 1.0}
    result = f_sb_mn(alpha, params)
    expected = 0.25  # 0.5 * 0.5 = 0.25
    assert np.isclose(result, expected), f"Expected {expected}, got {result}"


def test_get_model_info_sb_grid():
    """Test that get_model_info handles grid SB models correctly."""
    # Test grid model SB_m1_n2
    ode_func, base_params, state_dim, template = get_model_info(
        model_name='single_step',
        f_alpha_model='SB_m1_n2'
    )

    # Should only fit Ea and A (m,n are fixed)
    assert base_params == ['Ea', 'A'], f"Expected ['Ea', 'A'], got {base_params}"
    assert state_dim == 1, "Single-step model should have state_dim=1"

    # Template should have fixed m=1, n=2
    assert 'f_alpha_params' in template
    assert template['f_alpha_params']['m'] == 1.0
    assert template['f_alpha_params']['n'] == 2.0


def test_model_display_name_sb_grid():
    """Test display names for grid SB models."""
    # Grid model
    name = model_display_name('SB_m1_n2_model')
    assert 'SB(m=1, n=2)' in name, f"Expected SB(m=1, n=2) in name, got: {name}"

    # Grid model without _model suffix
    name = model_display_name('SB_m0_n1')
    assert 'SB(m=0, n=1)' in name, f"Expected SB(m=0, n=1) in name, got: {name}"

    # Standard model (should still work)
    name = model_display_name('F1_model')
    assert 'F1' in name and 'first-order' in name.lower(), f"Unexpected name: {name}"


def test_sb_grid_edge_cases():
    """Test edge cases for SB grid models."""
    # m=0, n=0 should give constant (zero-order)
    params = {'m': 0.0, 'n': 0.0}
    result = f_sb_mn(0.5, params)
    assert np.isclose(result, 1.0), "SB(0,0) should be constant = 1"

    # At alpha=0
    params = {'m': 1.0, 'n': 1.0}
    result = f_sb_mn(0.0, params)
    assert np.isclose(result, 0.0), "At α=0, SB(1,1) should be 0"

    # At alpha=1
    result = f_sb_mn(1.0, params)
    assert np.isclose(result, 0.0), "At α=1, SB(1,1) should be 0"


def test_model_selector_sb_grid():
    """Test that model selector provides SB_grid."""
    from akts import models

    # Check SB_grid exists
    sb_grid = models.kinetic.SB_grid
    assert isinstance(sb_grid, list)
    assert len(sb_grid) == 16

    # Check all_with_sb_grid
    comprehensive = models.kinetic.all_with_sb_grid
    assert isinstance(comprehensive, list)
    assert len(comprehensive) > 16  # Should have SB grid + other models

    # Check continuous SB is separate
    continuous_sb = models.kinetic.SB
    assert continuous_sb == ['SB']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
