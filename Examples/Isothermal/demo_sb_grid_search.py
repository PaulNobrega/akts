"""
Demo: Sestak-Berggren Grid Search for Mechanistic Screening
============================================================

This example demonstrates the new SB grid search approach for kinetic modeling.
Instead of fitting m,n continuously (which can overfit), we sample integer values
m,n ∈ {0,1,2,3} to ensure mechanistic interpretability.

Benefits:
1. Samples all practical mechanisms (16 models)
2. Integer m,n values have clear mechanistic meaning
3. Prevents overfitting to noise
4. More robust parameter space sampling
5. Faster convergence (fewer parameters to fit)

Common mechanistic interpretations:
- SB_m0_n0: Zero-order (F0) - constant rate
- SB_m0_n1: First-order (F1) - decay kinetics
- SB_m0_n2: Second-order (F2) - bimolecular
- SB_m0_n3: Third-order (F3)
- SB_m1_n0: Power law/autocatalytic (accelerating)
- SB_m1_n1: Mixed autocatalytic
- SB_m2_n1: Strong autocatalytic
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from akts import models, auto_model_isothermal_data, generate_sb_grid_models
import numpy as np


def main():
    """Main function - required for Windows multiprocessing support."""
    print("=" * 80)
    print(" AKTS: Sestak-Berggren Grid Search Demo")
    print("=" * 80)
    print()

    # =========================================================================
    # OPTION 1: Use the pre-built model selector
    # =========================================================================
    print("Option 1: Using models.kinetic.SB_grid")
    print("-" * 80)

    # Get all SB grid models (m,n = 0-3, integer combinations)
    sb_grid_models = models.kinetic.SB_grid
    print(f"Generated {len(sb_grid_models)} SB grid models:")
    print(f"  {sb_grid_models[:4]} ...")
    print()

    # Use them in analysis (combine with other models if desired)
    models_to_try = models.kinetic.SB_grid + ['A2', 'A3', 'D2', 'D3']
    print(f"Will try {len(models_to_try)} models total (16 SB + 4 others)")
    print()

    # =========================================================================
    # OPTION 2: Generate custom grid
    # =========================================================================
    print("Option 2: Generate custom SB grid")
    print("-" * 80)

    # Custom grid: m = 0-2, n = 0-3
    custom_grid = generate_sb_grid_models(m_range=range(3), n_range=range(4))
    print(f"Custom grid (m=0-2, n=0-3): {len(custom_grid)} models")
    print(f"  {custom_grid}")
    print()

    # =========================================================================
    # OPTION 3: Use pre-configured comprehensive set
    # =========================================================================
    print("Option 3: Use all_with_sb_grid (replaces F0-F3 with SB grid)")
    print("-" * 80)

    comprehensive_models = models.kinetic.all_with_sb_grid
    print(f"Comprehensive set: {len(comprehensive_models)} models")
    print(f"  Includes: A2, A3, R2, R3, D2, D3, Bna + SB grid")
    print()

    # =========================================================================
    # COMPARISON: Grid vs Continuous
    # =========================================================================
    print("=" * 80)
    print(" Grid Search vs Continuous Optimization Comparison")
    print("=" * 80)
    print()

    comparison = {
        "SB (continuous)": {
            "models": ['SB'],
            "params_fitted": "Ea, A, m, n (4 parameters)",
            "m_n_values": "Continuous (e.g., m=0.73, n=1.42)",
            "pros": "Maximum flexibility, potentially better fit",
            "cons": "Can overfit noise, less interpretable, slower convergence"
        },
        "SB_grid (recommended)": {
            "models": models.kinetic.SB_grid,
            "params_fitted": "Only Ea, A (2 parameters per model)",
            "m_n_values": "Integer only (e.g., m=1, n=1)",
            "pros": "Mechanistically interpretable, prevents overfitting, faster",
            "cons": "Might miss optimal non-integer values"
        }
    }

    for approach, info in comparison.items():
        print(f"{approach}:")
        print(f"  Models: {len(info['models']) if isinstance(info['models'], list) else 1}")
        print(f"  Parameters fitted: {info['params_fitted']}")
        print(f"  m,n values: {info['m_n_values']}")
        print(f"  Pros: {info['pros']}")
        print(f"  Cons: {info['cons']}")
        print()

    # =========================================================================
    # EXAMPLE USAGE WITH REAL DATA
    # =========================================================================
    print("=" * 80)
    print(" Example Usage (when you have data)")
    print("=" * 80)
    print()

    example_code = '''
# Example with actual data files:
from akts import models, auto_model_isothermal_data

# Use SB grid for mechanistic screening
results = auto_model_isothermal_data(
    data_files=['data_40C.csv', 'data_60C.csv', 'data_80C.csv'],
    models_to_try=models.kinetic.SB_grid,  # Try all 16 SB combinations
    top_n=5,  # Rank top 5 best fits
    report_path='sb_grid_results.html'
)

# Or combine with other mechanistic models
results = auto_model_isothermal_data(
    data_files=['data_40C.csv', 'data_60C.csv', 'data_80C.csv'],
    models_to_try=models.kinetic.all_with_sb_grid,  # SB grid + A2/A3/R2/R3/D2/D3/Bna
    top_n=10,
    report_path='comprehensive_results.html'
)

# Access the best SB model
best_model = results['selected_model']
print(f"Best model: {best_model['model_name']}")
print(f"Parameters: {best_model['parameters']}")

# If it's a grid model, the m,n values are fixed:
# e.g., "SB_m1_n1_model" means m=1, n=1 (only Ea and A were fitted)
'''
    print(example_code)

    # =========================================================================
    # MECHANISTIC INTERPRETATION GUIDE
    # =========================================================================
    print("=" * 80)
    print(" Mechanistic Interpretation Guide")
    print("=" * 80)
    print()

    interpretations = {
        "SB_m0_n0": "Zero-order (constant degradation rate)",
        "SB_m0_n1": "First-order (exponential decay) - most common",
        "SB_m0_n2": "Second-order (bimolecular reaction)",
        "SB_m0_n3": "Third-order",
        "SB_m1_n0": "Power law (accelerating reaction)",
        "SB_m1_n1": "Autocatalytic (S-shaped curve)",
        "SB_m2_n1": "Strong autocatalysis (sharp S-curve)",
        "SB_m1_n2": "Mixed autocatalytic + decay",
    }

    print("Common SB(m,n) mechanistic meanings:")
    for model, meaning in interpretations.items():
        print(f"  {model}: {meaning}")
    print()

    # =========================================================================
    # RECOMMENDATIONS
    # =========================================================================
    print("=" * 80)
    print(" Recommendations")
    print("=" * 80)
    print()
    print("1. START with models.kinetic.SB_grid for mechanistic screening")
    print("   - Covers all practical mechanisms with integer m,n")
    print("   - More interpretable than continuous optimization")
    print()
    print("2. If no SB model fits well, try models.kinetic.all")
    print("   - Includes Avrami (A2, A3), diffusion (D2, D3), etc.")
    print()
    print("3. For comprehensive screening: models.kinetic.all_with_sb_grid")
    print("   - Tests 20 mechanistic models total")
    print()
    print("4. Only use 'SB' (continuous) if:")
    print("   - Grid models fit poorly (Delta-AICc > 10)")
    print("   - You have high-quality, low-noise data")
    print("   - Non-integer m,n is theoretically justified")
    print()
    print("5. Hybrid approach (recommended for publication):")
    print("   - First: Grid search to identify best integer m,n")
    print("   - Then: Fine-tune with continuous optimization around those values")
    print("   - Compare: If Delta-AICc < 2, use integer values (more interpretable)")
    print()

    print("=" * 80)


if __name__ == '__main__':
    main()
