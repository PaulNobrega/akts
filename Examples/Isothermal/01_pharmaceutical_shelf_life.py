"""
Example 1: Pharmaceutical Shelf-Life Prediction with ICH Q1E Compliance
========================================================================

Scientific Context:
- Drug product stability study at 25°C, 30°C, and 40°C
- ICH Q1E guideline compliance for shelf-life estimation
- One-sided 95% confidence intervals (conservative)
- Extrapolation ceiling validation

Key Features Demonstrated:
✓ ICH Q1E regulatory compliance
✓ Bootstrap confidence intervals
✓ Shelf-life prediction at label storage temperature
✓ Model selection with simplicity preference
✓ Professional HTML report for regulatory submission

Data: Aspirin tablet degradation (API content loss over time)
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from akts import (auto_model_isothermal_data, KineticDataset,
                  time_to_conversion, calculate_ich_q1e_ceiling)


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
    """Pharmaceutical stability analysis with ICH Q1E compliance."""

    print("=" * 80)
    print(" ICH Q1E Pharmaceutical Shelf-Life Prediction")
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

    # Run ICH Q1E compliant analysis
    print("Running ICH Q1E compliant stability analysis...")
    print("-" * 80)

    results = auto_model_isothermal_data(
        data_files=datasets,

        # ICH Q1E: Predict at label storage (25°C) for regulatory shelf-life
        predict=(3, 'year', 25),

        # Unit conversions
        input_temperature_units='K',
        output_temperature_units='C',

        # Model selection: try simpler models first
        models_to_try=['F0', 'F1', 'F2', 'A2', 'A3', 'R2', 'R3', 'D2', 'D3'],
        include_ode_models=False,  # Not needed for first-order degradation

        # ICH Q1E: Bootstrap for confidence intervals
        bootstrap_iterations=100,

        # Report with regulatory compliance section
        report_path=output_dir / 'pharmaceutical_ich_q1e_report.html',
        report_format='interactive',

        # Progress tracking
        progress_callback=lambda msg, data: print(f"  [{data.get('timestamp', '')}] {msg}")
    )

    print()
    print("=" * 80)
    print(" Analysis Complete - ICH Q1E Results")
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

    # ICH Q1E shelf-life calculation
    print("ICH Q1E Shelf-Life Estimation:")
    print("-" * 80)

    # Get bootstrap result
    bootstrap_result = results['bootstrap_results'].get(selected['model_name'])

    if bootstrap_result:
        # Calculate shelf-life at 5% degradation (95% API content)
        from akts import fit_kinetic_model

        # Recreate fit result for time_to_conversion
        fit_result_selected = None
        if 'fit_results' in results:
            fit_result_selected = results['fit_results'].get(selected['model_name'])

        if fit_result_selected:
            # One-sided 95% CI (ICH Q1E requirement)
            shelf_life = time_to_conversion(
                fit_result=fit_result_selected,
                target_conversion=0.05,  # 5% degradation
                temperature_K=298.15,    # 25°C storage
                bootstrap_result=bootstrap_result,
                one_sided_ci=True        # ICH Q1E: one-sided lower bound
            )

            if shelf_life['time_sec']:
                mean_months = shelf_life['time_sec'] / (30.44 * 24 * 3600)
                lower_months = shelf_life['time_lower_sec'] / (30.44 * 24 * 3600) if shelf_life['time_lower_sec'] else None

                print(f"  Mean shelf-life: {mean_months:.1f} months")
                if lower_months:
                    print(f"  95% Lower Bound: {lower_months:.1f} months (ICH Q1E)")
                print()

                # ICH Q1E extrapolation ceiling
                study_duration_months = 24  # 24-month long-term data
                ceiling = calculate_ich_q1e_ceiling(study_duration_months, is_long_term=True)

                print(f"ICH Q1E Extrapolation Guidance:")
                print(f"  Study duration: {study_duration_months} months")
                print(f"  Extrapolation ceiling: {ceiling:.0f} months")

                if lower_months and lower_months <= ceiling:
                    print(f"  ✓ COMPLIANT: Shelf-life within ICH Q1E guidelines")
                    print(f"  -> Proposed label shelf-life: {int(lower_months)} months")
                elif lower_months:
                    print(f"  ⚠ CAUTION: Shelf-life exceeds ICH Q1E ceiling")
                    print(f"  -> Additional stability data recommended")
                    print(f"  -> Conservative shelf-life: {int(ceiling)} months")
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
    print("  - Includes ICH Q1E compliance section")
    print("  - One-sided 95% confidence intervals")
    print("  - Extrapolation ceiling validation")
    print("  - Ready for regulatory submission")
    print("=" * 80)


if __name__ == '__main__':
    main()
