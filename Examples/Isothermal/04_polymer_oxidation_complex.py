"""
Example 4: Polymer Oxidation with Complex Kinetics (A->B->C)
===========================================================

Scientific Context:
- Polyethylene thermal oxidation
- Two-step mechanism: PE -> Hydroperoxides -> Carbonyl products
- Complex ODE kinetics required
- Multi-temperature accelerated aging

Key Features Demonstrated:
✓ Complex ODE models (A->B->C consecutive reactions)
✓ Multistart optimization for global minimum
✓ Model comparison: simple vs. complex mechanisms
✓ Arrhenius parameter interpretation
✓ Long-term service life prediction

Data: Carbonyl index growth in PE films at elevated temperatures
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from akts import (auto_model_isothermal_data, KineticDataset,
                  plot_arrhenius, rank_models)


def generate_polymer_oxidation_data():
    """
    Generate realistic polymer oxidation data with A->B->C kinetics.

    Mechanism:
    - PE -> Hydroperoxides (Ea1 ~ 110 kJ/mol)
    - Hydroperoxides -> Carbonyls (Ea2 ~ 80 kJ/mol)
    - Measure: carbonyl index (final product)
    """
    np.random.seed(200)

    # Accelerated aging: 60°C, 80°C, 100°C, 120°C
    temps_C = [60, 80, 100, 120]
    temps_K = [T + 273.15 for T in temps_C]

    datasets = []

    for T_K in temps_K:
        # True kinetic parameters (two-step)
        Ea1 = 110000  # J/mol (first step)
        A1 = 1e14     # s^-1
        Ea2 = 80000   # J/mol (second step)
        A2 = 1e12     # s^-1

        # Time points (longer at lower temps)
        if T_K < 350:  # 60-80°C
            time_hours = np.array([0, 100, 200, 500, 1000, 2000, 3000, 4000])
        else:  # 100-120°C
            time_hours = np.array([0, 50, 100, 200, 400, 800, 1200, 1600])

        time_sec = time_hours * 3600

        # A->B->C kinetics (numerical solution)
        k1 = A1 * np.exp(-Ea1 / (8.314 * T_K))
        k2 = A2 * np.exp(-Ea2 / (8.314 * T_K))

        # Analytical solution for A->B->C with A0=1, B0=C0=0
        # C(t) = 1 - k1/(k2-k1)*[exp(-k1*t) - exp(-k2*t)] - exp(-k2*t)
        t = time_sec
        if abs(k2 - k1) > 1e-10:
            C = 1 - k1/(k2-k1)*(np.exp(-k1*t) - np.exp(-k2*t)) - np.exp(-k2*t)
        else:
            # Limiting case k1 ≈ k2
            C = 1 - np.exp(-k1*t) - k1*t*np.exp(-k1*t)

        C = np.clip(C, 0, 1)

        # Add measurement noise (FTIR: 3% RSD)
        C += np.random.normal(0, 0.03, len(C))
        C = np.clip(C, 0, 1)

        datasets.append(KineticDataset(
            time=time_sec,
            temperature=np.full_like(time_sec, T_K),
            conversion=C  # Carbonyl formation (final product)
        ))

    return datasets


def main():
    """Polymer oxidation analysis with complex kinetics."""

    print("=" * 80)
    print(" Polymer Thermal Oxidation: A->B->C Kinetics")
    print("=" * 80)
    print()
    print("Scenario: Polyethylene film accelerated aging")
    print("  - Temperatures: 60°C, 80°C, 100°C, 120°C")
    print("  - Measurement: Carbonyl index by FTIR")
    print("  - Mechanism: PE -> Hydroperoxides -> Carbonyls")
    print("  - Service life target: 10 years at 25°C")
    print()

    # Generate data
    datasets = generate_polymer_oxidation_data()
    print(f"Generated {len(datasets)} accelerated aging datasets")
    for i, ds in enumerate(datasets):
        temp_C = np.mean(ds.temperature) - 273.15
        duration_hrs = ds.time[-1] / 3600
        print(f"  {temp_C:.0f}°C: {duration_hrs:.0f} hours, "
              f"{len(ds.time)} timepoints")
    print()

    # Output directory
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    # Run analysis with complex models
    print("Running kinetic analysis (including A->B->C model)...")
    print("  ⚠ Note: Complex ODE models take 5-10 minutes")
    print("-" * 80)

    results = auto_model_isothermal_data(
        data_files=datasets,

        # Predict service life at 75°C for 10 years
        predict=(10, 'year', 75+273.15),

        # Units
        input_temperature_units='K',
        output_temperature_units='C',

        # Include complex ODE models
        models_to_try=['F1', 'F2', 'A2'],  # Simple models for comparison
        include_ode_models=True,  # A->B->C and A+B->C

        # For ODE models: use fewer bootstrap iterations (slow)
        bootstrap_iterations=50,

        # Report
        report_path=output_dir / 'polymer_oxidation_report.html',
        report_format='interactive',

        progress_callback=lambda msg, data: print(f"  [{data.get('timestamp', '')}] {msg}")
    )

    print()
    print("=" * 80)
    print(" Analysis Results - Mechanism Identification")
    print("=" * 80)
    print()

    # Compare simple vs. complex models
    print("Model Comparison:")
    print("-" * 80)
    print(f"{'Model':<20} {'R²':>8} {'BIC':>10} {'ΔBIC':>8} {'Mechanism'}")
    print("-" * 80)

    for model in results['top_models'][:6]:
        name = model['model_name']
        stats = model['stats']  # ranked top_models use 'stats'; selected_model uses 'statistics'
        r2 = stats['r_squared']
        bic = stats['bic']
        delta_bic = bic - results['top_models'][0]['stats']['bic']

        # Categorize mechanism
        if 'A->B->C' in name:
            mech = "Two-step consecutive"
        elif 'A+B->C' in name:
            mech = "Bimolecular"
        elif name in ['F1', 'F2', 'A2', 'A3']:
            mech = "Single-step"
        else:
            mech = "Autocatalytic"

        print(f"{name:<20} {r2:8.4f} {bic:10.1f} {delta_bic:8.1f}  {mech}")

    print()

    selected = results['selected_model']
    print(f"Selected: {selected['model_name']}")
    print(f"  Reason: {selected['reason']}")
    print()

    # If A->B->C selected, show both activation energies
    params = selected['parameters']
    if 'A->B->C' in selected['model_name']:
        print("Two-Step Kinetic Parameters:")
        print(f"  Step 1 (PE -> Hydroperoxides):")
        print(f"    Ea1 = {params['Ea1']/1000:.1f} kJ/mol")
        print(f"    A1 = {params['A1']:.2e} s⁻¹")
        print(f"  Step 2 (Hydroperoxides -> Carbonyls):")
        print(f"    Ea2 = {params['Ea2']/1000:.1f} kJ/mol")
        print(f"    A2 = {params['A2']:.2e} s⁻¹")

        if params['Ea1'] > params['Ea2']:
            print(f"  -> First step is rate-limiting (higher Ea)")
        else:
            print(f"  -> Second step is rate-limiting (higher Ea)")
        print()

    elif 'Ea' in params:
        print("Single-Step Kinetic Parameters:")
        print(f"  Ea = {params['Ea']/1000:.1f} kJ/mol")
        print(f"  A = {params['A']:.2e} s⁻¹")
        print()

    # Service life prediction
    if results['predictions']:
        pred = results['predictions']
        print("10-Year Service Life Prediction at 25°C:")
        print("-" * 80)

        final_ox = pred['conversion_mean'][-1] * 100
        print(f"  Predicted oxidation: {final_ox:.1f}%")

        if 'conversion_lower' in pred:
            lower = pred['conversion_lower'][-1] * 100
            upper = pred['conversion_upper'][-1] * 100
            print(f"  95% CI: [{lower:.1f}%, {upper:.1f}%]")

        # Typical PE oxidation failure: ~30% carbonyl index
        if upper < 30:
            print(f"  ✓ Expected to survive 10 years at 25°C")
        else:
            print(f"  ⚠ May approach failure threshold at 10 years")
        print()

    # Time to 30% oxidation at different service temperatures
    print("Service Life at Various Temperatures (to 30% oxidation):")
    print("-" * 80)

    from akts import time_to_conversion

    fit_result = results['fit_results'].get(selected['model_name'])
    bootstrap_result = results['bootstrap_results'].get(selected['model_name'])

    service_temps_C = [0, 10, 20, 25, 30, 40]

    if fit_result:
        for temp_C in service_temps_C:
            temp_K = temp_C + 273.15

            lifetime = time_to_conversion(
                fit_result=fit_result,
                target_conversion=0.30,
                temperature_K=temp_K,
                bootstrap_result=bootstrap_result
            )

            if lifetime['time_sec']:
                years = lifetime['time_sec'] / (365.25 * 24 * 3600)
                print(f"  {temp_C:3d}°C: {years:6.1f} years", end="")

                if lifetime['time_lower_sec']:
                    lower_yrs = lifetime['time_lower_sec'] / (365.25 * 24 * 3600)
                    upper_yrs = lifetime['time_upper_sec'] / (365.25 * 24 * 3600)
                    print(f"  (95% CI: {lower_yrs:.1f} - {upper_yrs:.1f} years)")
                else:
                    print()

    print()

    # Generate Arrhenius plot
    print("Generating Arrhenius plot...")
    try:
        fit_result = results['fit_results'].get(selected['model_name'])

        if fit_result and 'Ea' in params:
            fig = plot_arrhenius(fit_result)
            fig.savefig(output_dir / 'polymer_arrhenius.png', dpi=300, bbox_inches='tight')
            print(f"  [OK] Saved: polymer_arrhenius.png")
    except Exception as e:
        print(f"  [SKIP] Arrhenius plot: {e}")

    print()
    print("=" * 80)
    print(f"Report: {results['report_path']}")
    print("  - Complex A->B->C kinetics identified")
    print("  - 10-year service life prediction")
    print("  - Temperature-dependent lifetime estimates")
    print("=" * 80)


if __name__ == '__main__':
    main()
