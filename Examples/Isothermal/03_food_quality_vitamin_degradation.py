"""
Example 3: Food Quality - Vitamin C Degradation in Juice
=========================================================

Scientific Context:
- Vitamin C (ascorbic acid) degradation in orange juice
- Temperature-dependent quality loss
- Shelf-life at different storage temperatures
- First-order degradation kinetics

Key Features Demonstrated:
✓ Simple first-order kinetics
✓ Q10 temperature coefficient calculation
✓ Multi-temperature shelf-life prediction
✓ Tabulated export for Excel/R analysis
✓ Publication-ready plotting

Data: Vitamin C retention in pasteurized orange juice
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from akts import (auto_model_isothermal_data, KineticDataset,
                  export_prediction_report, plot_bootstrap_ci_bands)


def generate_vitamin_c_data():
    """
    Generate realistic vitamin C degradation data.

    True kinetics: Ea ≈ 70 kJ/mol, first-order
    """
    np.random.seed(150)

    # Storage temperatures: 4°C (refrigerated), 20°C (room), 35°C (warm)
    temps_C = [4, 20, 35]
    temps_K = [T + 273.15 for T in temps_C]

    datasets = []

    for T_K in temps_K:
        # True kinetic parameters
        Ea = 70000  # J/mol
        A = 1e9     # s^-1

        # Time points in days
        if T_K < 280:  # 4°C
            time_days = np.array([0, 7, 14, 30, 60, 90, 120, 180])
        elif T_K < 300:  # 20°C
            time_days = np.array([0, 3, 7, 14, 21, 30, 45, 60])
        else:  # 35°C
            time_days = np.array([0, 1, 3, 7, 10, 14, 21, 30])

        time_sec = time_days * 24 * 3600

        # First-order degradation
        k = A * np.exp(-Ea / (8.314 * T_K))
        vitamin_retention = np.exp(-k * time_sec)

        # Add measurement noise (HPLC: 2% RSD)
        vitamin_retention += np.random.normal(0, 0.02, len(vitamin_retention))
        vitamin_retention = np.clip(vitamin_retention, 0.5, 1.0)

        # Conversion = loss
        conversion = 1 - vitamin_retention

        datasets.append(KineticDataset(
            time=time_sec,
            temperature=np.full_like(time_sec, T_K),
            conversion=conversion
        ))

    return datasets


def calculate_q10(Ea_J_mol):
    """Calculate Q10 temperature coefficient from activation energy."""
    R = 8.314
    # Q10 = exp(Ea/R * (1/T1 - 1/T2)) where T2 - T1 = 10 K
    T1 = 298.15  # 25°C
    T2 = 308.15  # 35°C
    Q10 = np.exp(Ea_J_mol / R * (1/T1 - 1/T2))
    return Q10


def main():
    """Vitamin C degradation analysis for food quality."""

    print("=" * 80)
    print(" Food Quality: Vitamin C Degradation in Orange Juice")
    print("=" * 80)
    print()
    print("Scenario: Pasteurized orange juice stability")
    print("  - Storage: 4°C (refrigerated), 20°C (room), 35°C (warm)")
    print("  - Measurement: HPLC vitamin C quantification")
    print("  - Quality target: >= 80% retention (<= 20% loss)")
    print()

    # Generate data
    datasets = generate_vitamin_c_data()
    print(f"Generated {len(datasets)} temperature datasets")
    for i, ds in enumerate(datasets):
        temp_C = np.mean(ds.temperature) - 273.15
        duration_days = ds.time[-1] / (24 * 3600)
        retention_final = (1 - ds.conversion[-1]) * 100
        print(f"  {temp_C:.0f}°C: {duration_days:.0f} days, final retention {retention_final:.0f}%")
    print()

    # Output directory
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    # Run analysis
    print("Running degradation kinetics analysis...")
    print("-" * 80)

    results = auto_model_isothermal_data(
        data_files=datasets,

        # Don't predict - we'll do multi-temperature predictions manually
        predict=None,

        # Units
        input_temperature_units='K',
        output_temperature_units='C',

        # Simple models only (first-order expected)
        models_to_try=['F0', 'F1', 'F2', 'A2', 'A3'],
        include_ode_models=False,

        # Bootstrap for confidence intervals
        bootstrap_iterations=100,

        # Report
        report_path=output_dir / 'vitamin_c_report.html',
        report_format='interactive',

        progress_callback=lambda msg, data: print(f"  [{data.get('timestamp', '')}] {msg}")
    )

    print()
    print("=" * 80)
    print(" Analysis Results")
    print("=" * 80)
    print()

    selected = results['selected_model']
    params = selected['parameters']
    stats = selected['statistics']

    print(f"Selected Model: {selected['model_name']}")
    print(f"  Ea = {params['Ea']/1000:.1f} kJ/mol")
    print(f"  A = {params['A']:.2e} s⁻¹")
    print(f"  R² = {stats['r_squared']:.4f}")
    print()

    # Calculate Q10
    Q10 = calculate_q10(params['Ea'])
    print(f"Q10 Temperature Coefficient: {Q10:.2f}")
    print(f"  -> {Q10:.2f}x faster degradation per 10°C increase")
    print()

    # Predict shelf-life at multiple storage temperatures
    print("Shelf-Life Predictions (to 20% vitamin loss):")
    print("-" * 80)

    from akts import predict_conversion, time_to_conversion

    fit_result = results['fit_results'].get(selected['model_name'])
    bootstrap_result = results['bootstrap_results'].get(selected['model_name'])

    storage_temps_C = [4, 10, 20, 25, 30]

    for temp_C in storage_temps_C:
        temp_K = temp_C + 273.15

        # Calculate shelf-life at 20% loss (0.2 conversion)
        if fit_result:
            shelf_life = time_to_conversion(
                fit_result=fit_result,
                target_conversion=0.20,
                temperature_K=temp_K,
                bootstrap_result=bootstrap_result,
                one_sided_ci=False  # Two-sided for food quality
            )

            if shelf_life['time_sec']:
                mean_days = shelf_life['time_sec'] / (24 * 3600)
                print(f"  {temp_C:2d}°C: {mean_days:5.1f} days", end="")

                if shelf_life['time_lower_sec']:
                    lower_days = shelf_life['time_lower_sec'] / (24 * 3600)
                    upper_days = shelf_life['time_upper_sec'] / (24 * 3600)
                    print(f"  (95% CI: {lower_days:.1f} - {upper_days:.1f} days)")
                else:
                    print()

    print()

    # Export predictions for multiple temperatures
    print("Exporting predictions for data analysis...")

    if fit_result and bootstrap_result:
        # Predict at refrigerated temperature (4°C) for 6 months
        prediction = predict_conversion(
            kinetic_description=fit_result,
            bootstrap_result=bootstrap_result,
            temperature_program=lambda t: 277.15,  # 4°C
            simulation_time_sec=np.linspace(0, 180*24*3600, 200)  # 6 months
        )

        # Export to CSV for Excel/R/Python analysis
        files = export_prediction_report(
            prediction=prediction,
            bootstrap_result=bootstrap_result,
            path_prefix=str(output_dir / 'vitamin_c_prediction'),
            include_plot=True,
            time_units='days'
        )

        print(f"  [OK] Prediction CSV: {Path(files['prediction_csv']).name}")
        print(f"  [OK] Bootstrap CI: {Path(files['bootstrap_csv']).name}")
        print(f"  [OK] Plot PNG: {Path(files['plot']).name}")

        # Also generate custom CI band plot
        fig = plot_bootstrap_ci_bands(prediction, time_units='days')
        fig.savefig(output_dir / 'vitamin_c_ci_bands.png', dpi=300, bbox_inches='tight')
        print(f"  [OK] CI bands plot: vitamin_c_ci_bands.png")

    print()
    print("=" * 80)
    print("Recommendations:")
    print("  - Refrigerated storage (4°C) recommended")
    print("  - Avoid room temperature (>20°C) storage")
    print(f"  - Expected quality degradation: {Q10:.1f}x per 10°C")
    print()
    print(f"Report: {results['report_path']}")
    print(f"Data exports: {output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()
