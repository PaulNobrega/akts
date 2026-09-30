"""
Example 1: Pharmaceutical Shelf-Life Analysis with ICH Q1E-Style Regression
========================================================================

Scientific Context:
- Drug product stability study at 25°C, 30°C, and 40°C
- Regression-based shelf-life estimate at label storage temperature
- Automatic linear-versus-quadratic trend selection
- One-sided 95% confidence limit (conservative)
- Extrapolation ceiling validation

Key Features Demonstrated:
✓ ICH Q1E-style regression/ANCOVA shelf-life analysis
✓ Separate bootstrap confidence intervals for kinetic model predictions
✓ Shelf-life prediction at label storage temperature
✓ Model selection with simplicity preference
✓ Professional HTML report with selected shelf-life trend

This synthetic demonstration is not a claim of regulatory compliance. Validate
the study design, trend assumptions, and specification limit for real submissions.

Data: Aspirin tablet degradation (API content loss over time)
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from akts import auto_model_isothermal_data, KineticDataset


def generate_aspirin_data():
    """
    Generate realistic aspirin degradation data.

    True kinetics: Ea = 80 kJ/mol, first-order degradation
    """
    np.random.seed(42)

    # Storage conditions: 25°C (long-term), 30°C (intermediate), 40°C (accelerated)
    temps_C = [25, 30, 40]
    temps_K = [T + 273.15 for T in temps_C]

    # Time points: 0, 1, 2, 3, 6, 9, 12, 18, 24 months
    time_months = np.array([0, 1, 2, 3, 6, 9, 12, 18, 24])

    datasets = []

    for T_K in temps_K:
        # True kinetic parameters
        Ea = 80000  # J/mol
        A = 1e10    # s^-1

        # Calculate degradation (API loss)
        time_sec = time_months * 30.44 * 24 * 3600  # months to seconds
        k = A * np.exp(-Ea / (8.314 * T_K))

        # API content: starts at 100%, decreases via first-order
        api_content = 100 * np.exp(-k * time_sec)

        # Add realistic measurement noise (0.5% RSD)
        api_content += np.random.normal(0, 0.5, len(api_content))
        api_content = np.clip(api_content, 85, 100)

        # Convert to degradation (0-100 scale for conversion)
        degradation = 100 - api_content
        conversion = degradation / 100  # 0-1 scale

        datasets.append(KineticDataset(
            time=time_sec,
            temperature=np.full_like(time_sec, T_K),
            conversion=conversion
        ))

    return datasets


def main():
    """Demonstrate model selection and regression-based shelf-life analysis."""

    print("=" * 80)
    print(" Pharmaceutical Shelf-Life Regression Example")
    print("=" * 80)
    print()
    print("Scenario: Aspirin tablet stability study")
    print("  - Long-term: 25°C/60% RH (24 months)")
    print("  - Intermediate: 30°C/65% RH (12 months)")
    print("  - Accelerated: 40°C/75% RH (6 months)")
    print("  - Specification: >= 95% API content (<= 5% degradation)")
    print()

    # Generate or load data
    datasets = generate_aspirin_data()
    print(f"Loaded {len(datasets)} stability datasets")
    print()

    # Output directory
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    # Run regression-based shelf-life analysis
    print("Running shelf-life analysis...")
    print("-" * 80)

    results = auto_model_isothermal_data(
        data_files=datasets,

        # Prediction horizon and shelf-life storage condition are configured separately.
        predict=(3, 'year', 25),
        shelf_life_temperature_C=25.0,
        shelf_life_target_conversion=0.05,
        shelf_life_confidence_level=0.95,
        shelf_life_is_long_term=True,
        shelf_life_nonlinearity_p_threshold=0.05,

        # Unit conversions
        input_temperature_units='K',
        output_temperature_units='C',

        # Model selection: try simpler models first
        models_to_try=['F0', 'F1', 'F2', 'A2', 'A3', 'R2', 'R3', 'D2', 'D3'],
        include_ode_models=False,  # Not needed for first-order degradation

        # Bootstrap is for kinetic model prediction uncertainty, not shelf-life regression.
        bootstrap_iterations=100,

        # Report with regulatory compliance section
        report_path=output_dir / 'pharmaceutical_ich_q1e_report.html',
        report_format='interactive',

        # Progress tracking
        progress_callback=lambda msg, data: print(f"  [{data.get('timestamp', '')}] {msg}")
    )

    print()
    print("=" * 80)
    print(" Analysis Complete - Shelf-Life Results")
    print("=" * 80)
    print()

    # Extract results
    selected = results['selected_model']
    params = selected['parameters']
    stats = selected['statistics']

    print(f"Selected Model: {selected['model_name']}")
    print(f"  Ea = {params['Ea']/1000:.1f} kJ/mol")
    print(f"  A = {params['A']:.2e} s⁻¹")
    print(f"  R² = {stats['r_squared']:.4f}")
    print(f"  BIC = {stats['bic']:.1f}")
    print()

    # Shelf-life estimate is computed separately from bootstrap prediction intervals.
    print("Regression-Based Shelf-Life Estimation:")
    print("-" * 80)
    shelf_life = results.get('regulatory')
    if shelf_life:
        print(f"  Selected trend: {shelf_life['trend_type']}")
        print(f"  Regression method: {shelf_life['regression_method']}")
        print(f"  Storage temperature: {shelf_life['storage_temp_K'] - 273.15:.1f}°C")
        print(f"  Mean shelf-life: {shelf_life['shelf_life_months']:.1f} months")
        confidence = shelf_life['shelf_life_confidence_level']
        print(f"  One-sided {confidence:.0%} shelf-life bound: {shelf_life['shelf_life_lower_95']:.1f} months")
        print(f"  Extrapolation ceiling: {shelf_life['ich_ceiling_months']:.1f} months")
        if shelf_life['curvature_p_value'] is not None:
            print(f"  Curvature test p-value: {shelf_life['curvature_p_value']:.4g}")
    else:
        print("  Estimate unavailable: no usable observations at the configured shelf-life temperature.")
    print()

    # Prediction results
    if results['predictions']:
        pred = results['predictions']
        print("3-Year Prediction at 25°C (Label Storage):")
        final_deg = pred['conversion_mean'][-1] * 100
        print(f"  Predicted degradation: {final_deg:.1f}%")

        if 'conversion_lower' in pred:
            lower = pred['conversion_lower'][-1] * 100
            upper = pred['conversion_upper'][-1] * 100
            print(f"  95% CI: [{lower:.1f}%, {upper:.1f}%]")

            if upper < 5.0:
                print(f"  ✓ Within specification (< 5% degradation)")
            else:
                print(f"  ⚠ May exceed specification at 3 years")
        print()

    # Report location
    print("=" * 80)
    print(f"Regulatory Report: {results['report_path']}")
    print("  - Includes regression-based shelf-life analysis when matching data are available")
    print("  - Reports selected linear/quadratic trend and one-sided confidence limit")
    print("  - Review study design and trend assumptions before regulatory use")
    print("=" * 80)


if __name__ == '__main__':
    main()
