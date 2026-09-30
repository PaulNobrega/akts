"""
Demo: Model Selector with IDE Autocomplete

This example demonstrates the new model selector class that provides:
- IDE autocomplete for easy model discovery
- Logical grouping (kinetic, empirical, ODE, model-free)
- Dot notation access: models.kinetic.F1, models.empirical.all, etc.
- Mix and match: models.kinetic.all + [models.empirical.Linear]

No more remembering string names or looking up documentation!
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from akts import models, auto_model_isothermal_data


def main():
    """Demonstrate model selector usage."""

    print("\n" + "="*70)
    print("AKTS Model Selector Demo")
    print("IDE Autocomplete for Easy Model Discovery")
    print("="*70 + "\n")

    # Show available model groups
    print("[MODEL GROUPS]")
    print(f"  models.kinetic.all         -> {len(models.kinetic.all)} mechanistic models")
    print(f"  models.ode.all             -> {len(models.ode.all)} ODE models")
    print(f"  models.empirical.all       -> {len(models.empirical.all)} empirical models")
    print(f"  models.modelfree.all       -> {len(models.modelfree.all)} model-free methods")
    print(f"  models.all                 -> {len(models.all)} total models")

    # Show specific model access (all return lists!)
    print("\n[SPECIFIC MODELS] (all return lists for safe concatenation)")
    print(f"  models.kinetic.F1          -> {models.kinetic.F1}")
    print(f"  models.kinetic.A2          -> {models.kinetic.A2}")
    print(f"  models.empirical.Linear    -> {models.empirical.Linear}")
    print(f"  models.ode.consecutive     -> {models.ode.consecutive}")
    print(f"  models.modelfree.Friedman  -> {models.modelfree.Friedman}")

    # Show sub-categories
    print("\n[SUB-CATEGORIES]")
    print(f"  models.kinetic.nth_order   -> {models.kinetic.nth_order}")
    print(f"  models.kinetic.nucleation  -> {models.kinetic.nucleation}")
    print(f"  models.kinetic.diffusion   -> {models.kinetic.diffusion}")
    print(f"  models.kinetic.autocatalytic -> {models.kinetic.autocatalytic}")

    # Find data files
    data_dir = Path(__file__).parent
    existing_files = []
    for filename in ['protein_stability_313K.csv', 'protein_stability_323K.csv',
                     'protein_stability_333K.csv', 'protein_stability_343K.csv']:
        filepath = data_dir / filename
        if filepath.exists():
            existing_files.append(str(filepath))

    if len(existing_files) < 2:
        print("\n[INFO] Data files not found - skipping analysis examples")
        print("[INFO] Model selector API demonstration complete!")
        return

    print("\n" + "="*70)
    print("USAGE EXAMPLES")
    print("="*70)

    # Example 1: Use all mechanistic models
    print("\n[EXAMPLE 1: All mechanistic models]")
    print("Code: models_to_try = models.kinetic.all")
    print(f"Result: {len(models.kinetic.all)} models")

    # Example 2: Mix model types (safe concatenation!)
    print("\n[EXAMPLE 2: Mix model types - safe concatenation]")
    print("Code: models_to_try = models.kinetic.F1 + models.empirical.Linear")
    mixed = models.kinetic.F1 + models.empirical.Linear
    print(f"Result: {mixed}")
    print("Note: No need for brackets! Each model returns a list.")

    # Example 3: All models from multiple categories
    print("\n[EXAMPLE 3: Multiple categories]")
    print("Code: models_to_try = models.kinetic.nth_order + models.empirical.all")
    combined = models.kinetic.nth_order + models.empirical.all
    print(f"Result: {len(combined)} models → {combined}")

    # Example 4: Run actual analysis with model selector
    print("\n" + "="*70)
    print("RUNNING ANALYSIS WITH MODEL SELECTOR")
    print("="*70)
    print("\nUsing: models.kinetic.nth_order (F0, F1, F2, F3 only)")

    try:
        results = auto_model_isothermal_data(
            data_files=existing_files,
            models_to_try=models.kinetic.nth_order,  # Use model selector!
            predict=(1, 'year', 313.15),
            bootstrap_iterations=0,
            input_temperature_units='K',
            output_temperature_units='C',
            time_col='Time (days)',
            conversion_col='HMW Species (%)',
            temperature_col='Temperature (K)',
            report_path='output/model_selector_demo_report.html',
            auto_open=False
        )

        print("\n[TOP MODELS]")
        for i, model in enumerate(results['top_models'], 1):
            print(f"{i}. {model['model_name']:20s} | R² = {model['stats']['r_squared']:.4f}")

        print(f"\n[SELECTED] {results['selected_model']['model_name']}")
        print(f"Reason: {results['selection_reason']}")

    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*70)
    print("KEY BENEFITS")
    print("="*70)
    print("""
1. IDE AUTOCOMPLETE
   - Type 'models.' and see all available options
   - No need to remember string names or documentation

2. LOGICAL GROUPING
   - models.kinetic      → Solid-state mechanistic models
   - models.empirical    → Direct algebraic fits
   - models.ode          → Multi-step reactions
   - models.modelfree    → Isoconversional methods

3. FLEXIBILITY
   - Use all: models.kinetic.all
   - Use specific: [models.kinetic.F1, models.kinetic.A2]
   - Mix types: models.kinetic.all + models.empirical.all
   - Sub-categories: models.kinetic.nth_order

4. TYPE SAFETY
   - Returns correct model strings
   - No typos: models.kinetic.F1 ✓  vs  'F1' (could be 'f1', 'F_1', etc.)

5. DISCOVERABILITY
   - Browse available models in your IDE
   - See descriptions in docstrings
   - Learn as you code!
    """)

    print("="*70)
    print("Success! Try it in your IDE with autocomplete!")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
