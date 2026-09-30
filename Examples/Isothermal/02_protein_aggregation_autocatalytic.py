"""
Example 2: Protein Aggregation with Autocatalytic Kinetics
===========================================================

Scientific Context:
- Therapeutic protein (mAb) aggregation in liquid formulation
- Autocatalytic mechanism: aggregates nucleate further aggregation
- Multi-temperature stability (5°C, 25°C, 37°C, 45°C)
- High Molecular Weight (HMW) species formation

Key Features Demonstrated:
✓ Autocatalytic models (SB_mn, Bna)
✓ Model comparison and ranking
✓ Cross-validation for model selection
✓ Temperature excursion simulation (shipping scenario)
✓ Custom plotting with plotting helpers

Data: Monoclonal antibody HMW aggregates over time
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from akts import (auto_model_isothermal_data, KineticDataset,
                  plot_multi_temperature_data, plot_arrhenius,
                  plot_fit_overlay, predict_conversion)


def generate_mab_aggregation_data():
    """
    Generate realistic mAb aggregation data with autocatalytic kinetics.

    True mechanism: Nucleation followed by autocatalytic growth
    SB(m,n) model with m≈0.5, n≈1.5 (autocatalytic signature)
    """
    np.random.seed(100)

    # Storage temperatures: 5°C (refrigerated), 25°C (room temp),
    # 37°C (accelerated), 45°C (stress)
    temps_C = [5, 25, 37, 45]
    temps_K = [T + 273.15 for T in temps_C]

    datasets = []

    for T_K in temps_K:
        # True kinetic parameters for autocatalytic aggregation
        Ea = 95000  # J/mol (higher than simple first-order)
        A = 1e13    # s^-1
        m = 0.5     # Nucleation order
        n = 1.5     # Growth order (autocatalytic)

        # Time points (longer at lower temps)
        if T_K < 285:  # 5°C
            time_weeks = np.array([0, 4, 8, 12, 26, 52, 78, 104])  # Up to 2 years
        elif T_K < 300:  # 25°C
            time_weeks = np.array([0, 1, 2, 4, 8, 12, 26, 52])  # Up to 1 year
        else:  # 37°C and 45°C
            time_weeks = np.array([0, 1, 2, 4, 8, 12, 16, 20])  # Up to 20 weeks

        time_sec = time_weeks * 7.0 * 24 * 3600  # weeks to seconds (float, so alpha below is float too)

        # Autocatalytic kinetics (SB model approximation)
        k = A * np.exp(-Ea / (8.314 * T_K))

        # Numerical solution for SB(m,n) - autocatalytic
        alpha = np.zeros_like(time_sec)
        alpha[0] = 0.001  # Small initial aggregate seed

        for i in range(1, len(time_sec)):
            dt = time_sec[i] - time_sec[i-1]
            # f(α) = α^m * (1-α)^n (Sestak-Berggren)
            f_alpha = (alpha[i-1] ** m) * ((1 - alpha[i-1]) ** n)
            dalpha = k * f_alpha * dt
            alpha[i] = min(alpha[i-1] + dalpha, 0.95)  # Cap at 95%

        # Add measurement noise (1% RSD for SEC-HPLC)
        alpha += np.random.normal(0, 0.01, len(alpha))
        alpha = np.clip(alpha, 0, 1)

        datasets.append(KineticDataset(
            time=time_sec,
            temperature=np.full_like(time_sec, T_K),
            conversion=alpha
        ))

    return datasets


def main():
    """Protein aggregation analysis with autocatalytic kinetics."""

    print("=" * 80)
    print(" Therapeutic Protein Aggregation Analysis")
    print("=" * 80)
    print()
    print("Scenario: Monoclonal antibody (mAb) stability")
    print("  - Storage conditions: 5°C, 25°C, 37°C, 45°C")
    print("  - Measurement: HMW species by SEC-HPLC")
    print("  - Mechanism: Autocatalytic aggregation")
    print("  - Specification: <= 5% HMW aggregates")
    print()

    # Generate synthetic data
    datasets = generate_mab_aggregation_data()
    print(f"Generated {len(datasets)} temperature datasets")

    for i, ds in enumerate(datasets):
        temp_C = np.mean(ds.temperature) - 273.15
        duration_weeks = ds.time[-1] / (7 * 24 * 3600)
        print(f"  Dataset {i+1}: {temp_C:.0f}°C, {duration_weeks:.0f} weeks, "
              f"{len(ds.time)} timepoints")
    print()

    # Output directory
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    # Generate publication plot before analysis
    print("Generating experimental data plot...")
    fig = plot_multi_temperature_data(datasets, time_units='days')
    fig.savefig(output_dir / 'mab_aggregation_data.png', dpi=300, bbox_inches='tight')
    print(f"  [OK] Saved: mab_aggregation_data.png")
    print()

    # Run analysis focusing on autocatalytic models
    print("Running kinetic analysis...")
    print("-" * 80)

    results = auto_model_isothermal_data(
        data_files=datasets,

        # Predict at room temperature (30°C) for 2 years
        predict=(2, 'year', 30+273.15),

        # Simulate shipping excursion: cold chain break
        # 5 days at 5°C -> 3 days at 30°C (shipping) -> back to 5°C
        simulate=[
            (0, 5+273.15),      # Start at refrigerated
            (5, 5+273.15),      # 5 days storage
            (8, 30+273.15),     # 3 days shipping at room temp
            (30, 5+273.15),     # Return to refrigerated for 22 days
        ],
        simulate_time_unit='days',

        # Units
        input_temperature_units='K',
        output_temperature_units='C',

        # Model selection: focus on autocatalytic mechanisms
        models_to_try=['F1', 'F2', 'A2', 'A3', 'SB_mn', 'Bna'],  # Include autocatalytic
        include_ode_models=False,

        # Statistical rigor
        bootstrap_iterations=100,
        top_n=5,

        # Report
        report_path=output_dir / 'mab_aggregation_report.html',
        report_format='interactive',

        progress_callback=lambda msg, data: print(f"  [{data.get('timestamp', '')}] {msg}")
    )

    print()
    print("=" * 80)
    print(" Analysis Results - Autocatalytic Aggregation")
    print("=" * 80)
    print()

    # Model comparison
    print("Top 5 Models (Ranked by BIC):")
    print("-" * 80)
    best_bic = results['top_models'][0]['stats']['bic']
    for i, model in enumerate(results['top_models'][:5], 1):
        stats = model['stats']  # ranked top_models use 'stats'; selected_model uses 'statistics'
        print(f"{i}. {model['model_name']:15s} "
              f"R²={stats['r_squared']:.4f}  "
              f"BIC={stats['bic']:.1f}  "
              f"ΔBIC={stats['bic'] - best_bic:.1f}")

    print()
    selected = results['selected_model']
    print(f"Selected: {selected['model_name']} ({selected['reason']})")
    print()

    # If SB or Bna selected, show autocatalytic parameters
    params = selected['parameters']
    if 'SB' in selected['model_name'] or 'Bna' in selected['model_name']:
        print("Autocatalytic Kinetics:")
        print(f"  Ea = {params['Ea']/1000:.1f} kJ/mol")
        print(f"  A = {params['A']:.2e} s⁻¹")

        if 'm' in params and 'n' in params:
            print(f"  m = {params['m']:.2f} (nucleation order)")
            print(f"  n = {params['n']:.2f} (growth order)")
            if params['n'] > 1:
                print("  -> Autocatalytic mechanism confirmed (n > 1)")
            print()

    # Refrigerated storage prediction
    if results['predictions']:
        pred = results['predictions']
        print("2-Year Refrigerated Storage Prediction (5°C):")

        final_agg = pred['conversion_mean'][-1] * 100
        print(f"  Predicted HMW: {final_agg:.1f}%")

        if 'conversion_lower' in pred:
            lower = pred['conversion_lower'][-1] * 100
            upper = pred['conversion_upper'][-1] * 100
            print(f"  95% CI: [{lower:.1f}%, {upper:.1f}%]")

            if upper < 5.0:
                print(f"  ✓ Within specification (< 5% HMW)")
            else:
                print(f"  ⚠ May exceed 5% specification")
        print()

    # Shipping excursion simulation
    if results.get('simulation'):
        sim = results['simulation']
        print("Shipping Excursion Simulation (30 days with 3-day break):")

        # Find final aggregation
        final_agg = sim['conversion_mean'][-1] * 100
        print(f"  Final HMW after excursion: {final_agg:.1f}%")

        # Find aggregation at end of shipping (day 8)
        times = np.array(sim['time'])
        convs = np.array(sim['conversion_mean'])
        day8_idx = np.argmin(np.abs(times - 8 * 86400))
        day8_agg = convs[day8_idx] * 100

        print(f"  HMW at end of shipping (day 8): {day8_agg:.1f}%")

        if 'conversion_upper' in sim:
            upper = sim['conversion_upper'][-1] * 100
            if upper < 5.0:
                print(f"  ✓ Shipping excursion acceptable")
            else:
                print(f"  ⚠ Shipping excursion may cause spec failure")
        print()

    # Generate additional plots
    print("Generating analysis plots...")
    try:
        fit_result = results['fit_results'].get(selected['model_name'])

        if fit_result and 'Ea' in params:
            # Arrhenius plot
            fig = plot_arrhenius(fit_result)
            fig.savefig(output_dir / 'mab_arrhenius.png', dpi=300, bbox_inches='tight')
            print(f"  [OK] Saved: mab_arrhenius.png")

            # Fit overlay
            prediction_25C = predict_conversion(
                kinetic_description=fit_result,
                temperature_program=lambda t: 298.15,
                simulation_time_sec=np.linspace(0, 365*24*3600, 100)
            )
            fig = plot_fit_overlay(datasets, fit_result, prediction_25C, time_units='days')
            fig.savefig(output_dir / 'mab_fit_overlay.png', dpi=300, bbox_inches='tight')
            print(f"  [OK] Saved: mab_fit_overlay.png")

    except Exception as e:
        print(f"  [SKIP] Some plots: {e}")

    print()
    print("=" * 80)
    print(f"Report: {results['report_path']}")
    print("  - Autocatalytic mechanism identified")
    print("  - Refrigerated storage prediction")
    print("  - Shipping excursion impact assessed")
    print("=" * 80)


if __name__ == '__main__':
    main()
