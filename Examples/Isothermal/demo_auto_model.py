"""
Demo: Automated Isothermal Kinetic Modeling
============================================

This example demonstrates the high-level helper function auto_model_isothermal_data()
which automates the entire workflow:
1. Load data from multiple files
2. Fit multiple kinetic models
3. Rank and select best model(s)
4. Generate predictions with confidence intervals
5. Create professional HTML report
6. Export publication-ready plots

This is the user-friendly interface for non-experts!
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from akts import (models, auto_model_isothermal_data, plot_arrhenius,
                  plot_bootstrap_ci_bands)
import numpy as np


def progress_callback(message: str, data: dict):
    """
    Simple progress callback to show what's happening.

    Parameters
    ----------
    message : str
        Progress message
    data : dict
        Additional data (timestamp, step info, etc.)
    """
    timestamp = data.get('timestamp', '')
    print(f"[{timestamp}] {message}", flush=True)


def main():
    """Main function - required for Windows multiprocessing support."""
    print("=" * 80)
    print(" AKTS: Automated Isothermal Kinetic Modeling Demo")
    print("=" * 80)
    print()
    print("This demo shows how easy it is to perform complete kinetic analysis")
    print("with just a few lines of code!")
    print()

    # Create output directory if it doesn't exist
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    # Define data files
    # Note: These files don't exist yet - this is a template for when you have real data
    data_files = [
        Path(__file__).parent / "protein_stability_313K.csv",
        Path(__file__).parent / "protein_stability_323K.csv",
        Path(__file__).parent / "protein_stability_333K.csv",
        Path(__file__).parent / "protein_stability_343K.csv",
    ]

    # Check if files exist
    existing_files = [f for f in data_files if f.exists()]

    if not existing_files:
        print("[ERROR] No data files found!")
        print()
        print("Expected files:")
        for f in data_files:
            print(f"  - {f.name}")
        print()
        print("To run this demo:")
        print("1. Place your isothermal CSV data files in this directory")
        print("2. Update the filenames in this script")
        print("3. Run again!")
        print()
        print("=" * 80)
        print("Showing example usage instead:")
        print("=" * 80)
        print()

        # Show example code
        example_code = '''
# Example usage:
from akts import auto_model_isothermal_data

def progress(msg, data):
    print(f"{data['timestamp']} - {msg}")

results = auto_model_isothermal_data(
    data_files=['data_313K.csv', 'data_323K.csv', 'data_333K.csv'],
    predict=(3, 'year'),  # Predict 3 years ahead
    top_n=3,  # Consider top 3 models
    report_path='stability_report.html',
    report_format='interactive',  # Use interactive plots
    progress_callback=progress
)

# Access results
print(f"Selected model: {results['selected_model']['model_name']}")
print(f"R² = {results['selected_model']['statistics']['r_squared']:.4f}")

if results['predictions']:
    pred = results['predictions']['conversion_mean'][-1]
    print(f"Prediction at 3 years: {pred:.2%} conversion")
'''
        print(example_code)

        sys.exit(0)

    print(f"Found {len(existing_files)} data files:")
    for f in existing_files:
        print(f"  [OK] {f.name}")
    print()

    # Run automated analysis
    print("Starting automated analysis...")
    print("-" * 80)
    print()

    try:
        # Define shipping temperature excursion scenario
        # Simulates: 5 days at 25°C, 2 days shipping at 40°C, then back to 25°C for 23 days
        # Values in kelvin to match input file
        shipping_profile = [
            (0, 30+273.15),      # Day 0: Start at 30°C
            (5, 30+273.15),      # Day 5: Still at 30°C (before shipping)
            (7, 80+273.15),      # Day 7: Heated to 80°C during 2-day shipping
            (10, 40+273.15),     # Day 10: Back to 40°C (after shipping)
            (30, 40+273.15),     # Day 30: End at 40°C
        ]

        results = auto_model_isothermal_data(
            data_files=existing_files,
            predict=(3, 'year', 40+273.15),   # Predict at 313.15 K (40°C) for 3 years (always in kelvin)
            shelf_life_temperature_C=40.0,    # Lowest observed storage condition in this study
            shelf_life_target_conversion=0.05, #5% conversion (degradation)
            shelf_life_confidence_level=0.95,  # CI
            simulate=shipping_profile,     # Simulate shipping temperature excursion
            simulate_time_unit='days',     # Time unit for shipping profile. units: 'seconds', 'minutes', 'hours', 'days', 'weeks', 'months', 'years'
            input_temperature_units='K',   # Input temps in Kelvin. Can be F, C, or K
            output_temperature_units='C',  # Output temps in Celsius. Can be F, C, or K
            top_n=20,  # Consider top 20 models
            convergence_threshold=0.15,  # 15% convergence threshold
            models_to_try=models.all,  # Use model selector! Try: models.kinetic.all, models.empirical.all, models.all
            bootstrap_iterations=100,  # 100 bootstrap samples for confidence intervals
            bootstrap_method='monte_carlo', # Method for bootstrap confidence intervals. Options: 'monte_carlo' (default), 'parametric', 'residual'
            report_path=output_dir / 'isothermal_stability_report.html',
            report_format='interactive',  # Interactive plots with plotly, 'static' is for publication quality images, 'both' for both formats
            progress_callback=progress_callback, #defined above
            # Data loader parameters for file parsing
            time_col='Time (days)',
            temperature_col='Temperature (K)',
            readout_col='HMW Species (%)',
            readout_type='increasing',
            auto_detect=True # tries exact and fuzzy matching for column names
        )

        print()
        print("=" * 80)
        print(" Analysis Complete!")
        print("=" * 80)
        print()

        # Display summary
        summary = results['summary']
        print("Summary:")
        print(f"  Datasets analyzed: {summary['datasets_count']}")
        print(f"  Total data points: {summary['total_datapoints']}")
        temp_range = summary.get('temperature_range')
        if temp_range is None:
            temp_range_K = summary.get('temperature_range_K', [0, 0])
            temp_range = f"{temp_range_K[0]:.1f} - {temp_range_K[1]:.1f} K"
        print(f"  Temperature range: {temp_range}")
        print(f"  Models tried: {summary['models_tried']}")
        print(f"  Successful fits: {summary['models_successful']}")
        print()

        # Display selected model
        selected = results['selected_model']
        print("Selected Model:")
        print(f"  Model: {selected['model_name']}")
        print(f"  Rank: {selected['rank']}")
        print(f"  Reason: {selected['reason']}")
        print()

        # Display parameters
        print("Parameters:")
        params = selected['parameters']
        for name, value in params.items():
            if name == 'Ea':
                print(f"  Activation Energy (Ea): {value/1000:.1f} kJ/mol")
            elif name.startswith('A'):
                print(f"  Pre-exponential Factor ({name}): {value:.4e} s^-1")
            else:
                print(f"  {name}: {value:.4e}")
        print()

        # Display statistics
        stats = selected['statistics']
        print("Goodness of Fit:")
        print(f"  R² = {stats.get('r_squared', 0):.4f}")
        print(f"  AIC = {stats.get('aic', 0):.2f}")
        print(f"  BIC = {stats.get('bic', 0):.2f}")
        print(f"  RSS = {stats.get('rss', 0):.4e}")
        print()

        # Display prediction
        if results['predictions']:
            pred_data = results['predictions']
            temp = pred_data.get('temperature', pred_data.get('temperature_K', 0))
            temp_units = pred_data.get('temperature_units', 'K')
            print(f"Prediction at 3 years (isothermal at {temp:.1f}°{temp_units}):")
            final_conversion = pred_data['conversion_mean'][-1]
            print(f"  Predicted conversion: {final_conversion:.2%}")

            if 'conversion_lower' in pred_data:
                lower = pred_data['conversion_lower'][-1]
                upper = pred_data['conversion_upper'][-1]
                print(f"  95% CI: [{lower:.2%}, {upper:.2%}]")
        print()

        # Display simulation (shipping excursion)
        if results.get('simulation'):
            print("Shipping Temperature Excursion Simulation (30 days):")
            sim_data = results['simulation']
            final_conversion = sim_data['conversion_mean'][-1]
            max_temp = max(sim_data['temperature'])
            temp_units = sim_data.get('temperature_units', 'K')
            print(f"  Final conversion: {final_conversion:.2%}")
            print(f"  Max temperature: {max_temp:.1f}°{temp_units}")

            if 'conversion_lower' in sim_data:
                lower = sim_data['conversion_lower'][-1]
                upper = sim_data['conversion_upper'][-1]
                print(f"  95% CI: [{lower:.2%}, {upper:.2%}]")

            # Find conversion at key time points
            import numpy as np
            times = np.array(sim_data['time'])
            convs = np.array(sim_data['conversion_mean'])
            for day in [7, 10]:  # After shipping, after return to normal
                idx = np.argmin(np.abs(times - day * 86400))  # Convert days to seconds
                print(f"  Day {day}: {convs[idx]:.2%} conversion")
        print()

        # Display report location
        if results['report_path']:
            print(f"[REPORT] HTML Report saved to: {results['report_path']}")
            print("   Open this file in your web browser to view interactive plots and details")
        print()

        # Generate additional publication-ready plots
        print("Generating publication-ready plots...")
        plots_generated = 0
        try:
            # Get the fit result and datasets for plotting
            fit_result = results.get('fit_results', {}).get(selected['model_name'])
            datasets = results.get('datasets')

            if fit_result and datasets:
                # 1. Multi-temperature data plot with fit and CI bands
                try:
                    from akts import predict_conversion
                    from akts.plotting import plot_fit_overlay

                    # Get bootstrap result if available
                    bootstrap_res = results.get('bootstrap_results', {}).get(selected['model_name'])

                    # Predict each series at its own measured temperature profile.
                    predictions_by_dataset = []
                    for dataset in datasets:
                        temperature_profile = lambda t, ds=dataset: np.interp(
                            t, ds.time, ds.temperature
                        )
                        predictions_by_dataset.append(predict_conversion(
                            kinetic_description=fit_result,
                            temperature_program=temperature_profile,
                            simulation_time_sec=dataset.time,
                            bootstrap_result=bootstrap_res
                        ))

                    # Plot every series with its matching fit and bootstrap CI.
                    fig = plot_fit_overlay(
                        datasets=datasets,
                        fit_result=fit_result,
                        prediction=predictions_by_dataset,
                        time_units='days',
                        show_ci=True
                    )

                    plot_path = output_dir / 'data_multi_temperature.png'
                    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
                    print(f"  [OK] Saved: {plot_path.name}")
                    plots_generated += 1
                except Exception as e:
                    print(f"  [SKIP] Multi-temperature plot: {e}")

                # 2. Arrhenius plot. Model-free (Friedman) results have no single
                # Ea/A pair, so show the per-conversion Friedman regression instead.
                if fit_result.model_name == 'Friedman':
                    try:
                        from akts import plot_friedman_arrhenius
                        fig = plot_friedman_arrhenius(
                            fit_result.model_definition_args['iso_result'], datasets
                        )
                        plot_path = output_dir / 'arrhenius_plot.png'
                        fig.savefig(plot_path, dpi=300, bbox_inches='tight')
                        print(f"  [OK] Saved: {plot_path.name}")
                        plots_generated += 1
                    except Exception as e:
                        print(f"  [SKIP] Arrhenius plot: {e}")
                elif 'Ea' in params and any(k.startswith('A') for k in params):
                    try:
                        bootstrap_res = results.get('bootstrap_results', {}).get(selected['model_name'])
                        fig = plot_arrhenius(
                            fit_result,
                            datasets=datasets,
                            bootstrap_result=bootstrap_res
                        )
                        plot_path = output_dir / 'arrhenius_plot.png'
                        fig.savefig(plot_path, dpi=300, bbox_inches='tight')
                        print(f"  [OK] Saved: {plot_path.name}")
                        plots_generated += 1
                    except Exception as e:
                        print(f"  [SKIP] Arrhenius plot: {e}")

                # 3. Prediction with confidence intervals
                if results.get('predictions') and 'conversion_lower' in pred_data:
                    prediction_obj = results.get('prediction')
                    if prediction_obj and hasattr(prediction_obj, 'conversion_ci') and prediction_obj.conversion_ci:
                        try:
                            fig = plot_bootstrap_ci_bands(prediction_obj, time_units='months')
                            plot_path = output_dir / 'prediction_with_ci.png'
                            fig.savefig(plot_path, dpi=300, bbox_inches='tight')
                            print(f"  [OK] Saved: {plot_path.name}")
                            plots_generated += 1
                        except Exception as e:
                            print(f"  [SKIP] Prediction CI plot: {e}")

                if plots_generated > 0:
                    print(f"\n  {plots_generated} publication-ready plot(s) saved to: {output_dir}")
                else:
                    print(f"  [INFO] No additional plots generated (check HTML report for interactive plots)")
                print()
            else:
                print("  [INFO] Plotting data not available (check HTML report for interactive plots)")
                print()

        except Exception as e:
            print(f"  [INFO] Additional plots not generated: {e}")
            print(f"  [INFO] See HTML report for interactive plots")
            print()

        print("=" * 80)
        print(" Tips:")
        print("   - Shelf-life regression is available when data match the configured storage temperature")
        print("   - Use output_format='json' for API integration")
        print("   - Plot PNG files are 300 DPI, ready for publication")
        print("   - Bootstrap intervals are included when requested and successfully estimated")
        print("=" * 80)

    except Exception as e:
        print()
        print("[ERROR] Error during analysis:")
        print(f"   {type(e).__name__}: {e}")
        print()
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
