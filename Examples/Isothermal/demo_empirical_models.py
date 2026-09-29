"""
Demo: Empirical degradation models with global Arrhenius fitting.

This example demonstrates the empirical modeling approach where conversion
is fitted directly as a function of time, with shared Arrhenius parameters
across all temperatures.

Models included:
- First Order: α = A·exp(-k(T)·t)
- Linear: α = k(T)·t + C
- Square Root: α = k(T)·√t + C
- Logistic: α = A/(1 + B·exp(-k(T)·t))
- Exponential: α = A·(1 - exp(-k(T)·t)) + C

All use k(T) = A·exp(-Ea/RT) for temperature dependence.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from akts import models, auto_model_isothermal_data


def main():
    """Run empirical model demonstration."""

    print("\n" + "="*70)
    print("AKTS Empirical Models Demo")
    print("Global Arrhenius Fitting Across Multiple Temperatures")
    print("="*70 + "\n")

    # Use protein stability data (4 temperatures)
    data_dir = Path(__file__).parent

    # Check which files exist
    existing_files = []
    for filename in ['protein_stability_313K.csv', 'protein_stability_323K.csv',
                     'protein_stability_333K.csv', 'protein_stability_343K.csv']:
        filepath = data_dir / filename
        if filepath.exists():
            existing_files.append(str(filepath))
            print(f"[OK] Found: {filepath}")
        else:
            print(f"[WARNING] Missing: {filepath}")

    if len(existing_files) < 2:
        print("\n[ERROR] Need at least 2 temperature datasets for empirical models")
        print("Please ensure data files exist in:", data_dir)
        sys.exit(1)

    print(f"\n[INFO] Using {len(existing_files)} datasets for global fit")
    print("[INFO] All temperatures will be fit simultaneously with shared Ea and A")

    try:
        # Run analysis with empirical models
        print("\n" + "-"*70)
        print("Running Analysis with Empirical Models...")
        print("-"*70)

        results = auto_model_isothermal_data(
            data_files=existing_files,
            predict=(3, 'year', 313.15),     # Predict at 40°C for 3 years
            input_temperature_units='K',     # Data files use Kelvin
            output_temperature_units='C',    # Display in Celsius
            top_n=11,                        # Show all models
            convergence_threshold=0.15,      # 15% convergence threshold
            models_to_try= models.empirical.all,              # Use defaults
            bootstrap_iterations=0,          # Skip bootstrap for speed in demo
            report_format="interactive",     # Interactive Plotly plots
            report_path='output/empirical_demo_report.html',
            auto_open=False,                 # Set True to auto-open in browser
            # Column mapping for protein data
            time_col='Time (days)',
            conversion_col='HMW Species (%)',
            temperature_col='Temperature (K)'
        )

        print("\n" + "="*70)
        print("Analysis Complete!")
        print("="*70)

        # Show results
        print("\n[TOP MODELS]")
        for i, model in enumerate(results['top_models'][:5], 1):
            stats = model.get('stats', {})
            print(f"{i}. {model['model_name']:30s} | "
                  f"R² = {stats.get('r_squared', 0.0):.4f} | "
                  f"BIC = {stats.get('bic', 0.0):.1f}")

        print(f"\n[SELECTED MODEL]")
        selected = results['selected_model']
        print(f"Model: {selected['model_name']}")
        print(f"Selection reason: {results['selection_reason']}")
        print(f"\n[PARAMETERS]")
        for param, value in selected['parameters'].items():
            if param == 'Ea':
                print(f"  Ea = {value/1000:.1f} kJ/mol  (Activation Energy)")
            elif param == 'A':
                print(f"  A  = {value:.2e} s^-1  (Pre-exponential Factor)")
            else:
                print(f"  {param} = {value:.4f}")

        if 'predictions' in results and results['predictions']:
            pred = results['predictions']
            req = pred['requested_time']
            final_conv = pred['conversion_mean'][-1]
            temp = pred['temperature']
            temp_unit = pred['temperature_units']
            print(f"\n[PREDICTION]")
            print(f"Time: {req['value']} {req['unit']} at {temp:.1f}{temp_unit}")
            print(f"Predicted conversion: {final_conv:.2%}")

        report_path = Path('output/empirical_demo_report.html')
        if report_path.exists():
            print(f"\n[REPORT] Generated: {report_path.absolute()}")
            print(f"  Size: {report_path.stat().st_size / 1024:.1f} KB")

        print("\n" + "="*70)
        print("Success! Check the HTML report for detailed results.")
        print("="*70 + "\n")

        print("\n[COMPARISON: Mechanistic vs Empirical Models]")
        print("\nMechanistic Models (e.g., F1, A2, R3):")
        print("  + Physical interpretation (nucleation, diffusion, etc.)")
        print("  + Temperature extrapolation via ODE integration")
        print("  + Model-free isoconversional analysis available")
        print("  - Slower (ODE solving required)")

        print("\nEmpirical Models (First Order, Linear, Sqrt, etc.):")
        print("  + Much faster (direct algebraic fit)")
        print("  + Global Arrhenius fit across all temperatures")
        print("  + Simpler, fewer assumptions")
        print("  - Less physical interpretation")
        print("  - Variable temperature simulation approximated")

        print("\nBoth approaches use k(T) = A·exp(-Ea/RT)")
        print("Choose based on your application needs!")

    except Exception as e:
        print(f"\n[ERROR] Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
