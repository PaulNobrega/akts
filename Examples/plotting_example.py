"""

if __name__ == "__main__":
    Example: Plotting convenience functions for kinetic analysis

    Demonstrates all plotting helpers:
    - Ea(α) from isoconversional analysis
    - Fit overlay (data vs model)
    - Bootstrap confidence intervals
    - Arrhenius plot
    - Parameter distributions
    - Multi-temperature data
    """

    import numpy as np
    from akts import (KineticDataset, fit_kinetic_model, run_bootstrap,
                      predict_conversion, run_friedman)
    from akts import (plot_ea_vs_alpha, plot_fit_overlay, plot_bootstrap_ci_bands,
                      plot_arrhenius, plot_parameter_distributions, plot_multi_temperature_data)

    # Generate synthetic data at multiple temperatures
    np.random.seed(42)
    temps = [313.15, 323.15, 333.15]  # 40°C, 50°C, 60°C
    datasets = []

    Ea_true = 85000  # J/mol
    A_true = 1e11    # s^-1

    for T in temps:
        t = np.linspace(0, 3600, 30)
        k = A_true * np.exp(-Ea_true / (8.314 * T))
        alpha = 1.0 - np.exp(-k * t)
        alpha = np.clip(alpha + np.random.normal(0, 0.005, len(t)), 0, 1)

        datasets.append(KineticDataset(
            time=t,
            temperature=np.full_like(t, T),
            conversion=alpha
        ))

    print("Generated datasets at:", [f"{np.mean(ds.temperature)-273.15:.0f}°C" for ds in datasets])

    # ============================================================
    # 1. Multi-Temperature Data Plot
    # ============================================================
    print("\n1. Plotting multi-temperature data...")
    fig = plot_multi_temperature_data(datasets, time_units='hours')
    fig.savefig('plot_multi_temp_data.png', dpi=150)
    print("   Saved: plot_multi_temp_data.png")

    # ============================================================
    # 2. Fit Model
    # ============================================================
    print("\n2. Fitting kinetic model...")
    fit_result = fit_kinetic_model(
        datasets=datasets,
        model_name="single_step",
        model_definition_args={'f_alpha_model': 'F1'},
        initial_guesses={'Ea': 85000, 'A': 1e11}
    )

    print(f"   Ea = {fit_result.parameters['Ea']/1000:.1f} kJ/mol")
    print(f"   A = {fit_result.parameters['A']:.2e} s^-1")
    print(f"   R² = {fit_result.r_squared:.4f}")

    # ============================================================
    # 3. Arrhenius Plot
    # ============================================================
    print("\n3. Plotting Arrhenius relationship...")
    fig = plot_arrhenius(fit_result)
    fig.savefig('plot_arrhenius.png', dpi=150)
    print("   Saved: plot_arrhenius.png")

    # ============================================================
    # 4. Fit Overlay (Data vs Model)
    # ============================================================
    print("\n4. Plotting fit overlay...")
    # Generate prediction at middle temperature
    prediction = predict_conversion(
        kinetic_description=fit_result,
        temperature_program=lambda t: 323.15,
        simulation_time_sec=np.linspace(0, 3600, 100)
    )

    fig = plot_fit_overlay(datasets, fit_result, prediction=prediction, time_units='hours')
    fig.savefig('plot_fit_overlay.png', dpi=150)
    print("   Saved: plot_fit_overlay.png")

    # ============================================================
    # 5. Bootstrap Analysis
    # ============================================================
    print("\n5. Running bootstrap analysis (50 iterations)...")
    bootstrap_result = run_bootstrap(
        fit_result=fit_result,
        datasets=datasets,
        n_iterations=50,
        optimizer_options={'method': 'L-BFGS-B'},
        verbose=False
    )

    print(f"   Ea CI: [{bootstrap_result.parameter_ci['Ea'][0]/1000:.1f}, "
          f"{bootstrap_result.parameter_ci['Ea'][1]/1000:.1f}] kJ/mol")

    # ============================================================
    # 6. Parameter Distributions
    # ============================================================
    print("\n6. Plotting parameter distributions...")
    fig = plot_parameter_distributions(bootstrap_result, parameters=['Ea', 'A'])
    fig.savefig('plot_parameter_distributions.png', dpi=150)
    print("   Saved: plot_parameter_distributions.png")

    # ============================================================
    # 7. Prediction with Bootstrap CI
    # ============================================================
    print("\n7. Plotting prediction with confidence intervals...")
    # Predict at storage temperature (25°C) with bootstrap CI
    prediction_with_ci = predict_conversion(
        kinetic_description=fit_result,
        bootstrap_result=bootstrap_result,
        temperature_program=lambda t: 298.15,
        simulation_time_sec=np.linspace(0, 365*24*3600, 100)  # 1 year
    )

    fig = plot_bootstrap_ci_bands(prediction_with_ci, time_units='days')
    fig.savefig('plot_bootstrap_ci.png', dpi=150)
    print("   Saved: plot_bootstrap_ci.png")

    # ============================================================
    # 8. Isoconversional Analysis (Friedman)
    # ============================================================
    print("\n8. Running Friedman isoconversional analysis...")
    # Generate non-isothermal data for isoconversional analysis
    heating_rates = [5, 10, 20]  # K/min
    noniso_datasets = []

    for beta in heating_rates:
        t = np.linspace(0, 600, 50)
        T = 298.15 + beta * t / 60  # Linear heating
        alpha = 1 - np.exp(-A_true * np.exp(-Ea_true / (8.314 * T)) * t)
        alpha = np.clip(alpha + np.random.normal(0, 0.01, len(t)), 0, 1)

        noniso_datasets.append(KineticDataset(
            time=t,
            temperature=T,
            conversion=alpha,
            heating_rate=beta
        ))

    friedman = run_friedman(noniso_datasets, alpha_levels=np.linspace(0.1, 0.9, 9))

    # ============================================================
    # 9. Ea(α) Plot
    # ============================================================
    print("\n9. Plotting Ea(α) from isoconversional analysis...")
    fig = plot_ea_vs_alpha(friedman, show_error_bars=True)
    fig.savefig('plot_ea_vs_alpha.png', dpi=150)
    print("   Saved: plot_ea_vs_alpha.png")

    print("\n" + "="*60)
    print("All plots generated successfully!")
    print("="*60)
    print("\nGenerated files:")
    print("  - plot_multi_temp_data.png")
    print("  - plot_arrhenius.png")
    print("  - plot_fit_overlay.png")
    print("  - plot_parameter_distributions.png")
    print("  - plot_bootstrap_ci.png")
    print("  - plot_ea_vs_alpha.png")
    print("\nThese plotting functions provide quick, publication-ready")
    print("visualizations for kinetic analysis without writing matplotlib code.")
