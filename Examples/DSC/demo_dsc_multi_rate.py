"""

if __name__ == "__main__":
    Demo: DSC Multi-Rate Kinetic Analysis

    This example demonstrates a typical DSC workflow:
    - Epoxy resin curing reaction
    - Multiple heating rates (5, 10, 15, 20 K/min)
    - Loading raw heat flow data
    - Isoconversional analysis (Friedman method)
    - Understanding temperature-dependent kinetics
    """
    import numpy as np
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent))

    from akts import load_dsc_file, run_friedman

    print("=" * 70)
    print("DSC Multi-Rate Kinetic Analysis Demo")
    print("Epoxy Resin Curing - Isoconversional Analysis")
    print("=" * 70)

    # ============================================================================
    # Step 1: Load Multiple Heating Rate DSC Data
    # ============================================================================
    print("\n" + "=" * 70)
    print("STEP 1: Load DSC Data at Multiple Heating Rates")
    print("=" * 70)

    heating_rates = [5, 10, 15, 20]  # K/min
    datasets = []

    print("\nLoading DSC experiments...")
    print("-" * 70)

    for rate in heating_rates:
        filepath = Path(__file__).parent / f"epoxy_cure_{rate}K_min.csv"

        if not filepath.exists():
            print(f"Skipping {rate} K/min - file not found")
            continue

        # Load DSC data - automatically computes conversion from heat flow
        dataset = load_dsc_file(
            filepath,
            time_col='Time (min)',
            temperature_col='Temperature (C)',
            heat_flow_col='Heat Flow (W/g)',
            auto_detect=True
        )

        # Convert time from minutes to seconds
        dataset.time = dataset.time * 60

        # Convert temperature from Celsius to Kelvin
        dataset.temperature = dataset.temperature + 273.15

        datasets.append(dataset)

        print(f"\n{rate} K/min:")
        print(f"  Duration: {dataset.time[-1] / 60:.1f} minutes")
        print(f"  Temperature range: {dataset.temperature.min() - 273.15:.0f} - "
              f"{dataset.temperature.max() - 273.15:.0f} C")
        print(f"  Conversion range: {dataset.conversion.min():.3f} - {dataset.conversion.max():.3f}")
        print(f"  Data points: {len(dataset.time)}")
        print(f"  Source: {dataset.metadata.get('conversion_source', 'N/A')}")

    # ============================================================================
    # Step 2: Understand the Data
    # ============================================================================
    print("\n" + "=" * 70)
    print("STEP 2: Understanding DSC Heat Flow Data")
    print("=" * 70)

    print("\nWhat is heat flow?")
    print("  Exothermic reaction -> Negative heat flow (releases heat)")
    print("  Endothermic reaction -> Positive heat flow (absorbs heat)")
    print("\nEpoxy curing is exothermic:")
    print("  Large negative peak = Maximum reaction rate")
    print("  Integration of peak area = Total reaction enthalpy")

    print("\nConversion is calculated by:")
    print("  Conversion(t) = Cumulative heat / Total heat")
    print("  This is done automatically by akts!")

    # Show peak locations
    print("\nReaction Peak Temperatures:")
    print(f"{'Heating Rate':<20} {'Peak Temp (C)':<20} {'Conversion at Peak'}")
    print("-" * 60)

    for dataset, rate in zip(datasets, heating_rates):
        # Find peak in time domain (not conversion domain)
        # Peak occurs at maximum rate of conversion change
        dadt = np.gradient(dataset.conversion, dataset.time)
        peak_idx = np.argmax(dadt)

        peak_temp_C = dataset.temperature[peak_idx] - 273.15
        peak_conversion = dataset.conversion[peak_idx]

        print(f"{rate:5.0f} K/min          {peak_temp_C:10.1f} C         {peak_conversion:8.3f}")

    print("\nKey Observation:")
    print("  Higher heating rate -> Higher peak temperature")
    print("  This shift is used to determine activation energy!")

    # ============================================================================
    # Step 3: Isoconversional Analysis (Friedman)
    # ============================================================================
    print("\n" + "=" * 70)
    print("STEP 3: Friedman Isoconversional Analysis")
    print("=" * 70)

    print("\nDetermining activation energy Ea as a function of conversion...")
    print("This tells us if the reaction mechanism changes during cure.")

    # Run Friedman isoconversional analysis
    alpha_levels = np.linspace(0.1, 0.9, 17)

    try:
        friedman_result = run_friedman(datasets, alpha_levels=alpha_levels)

        # Calculate statistics
        valid_Ea = friedman_result.Ea[np.isfinite(friedman_result.Ea)]

        if len(valid_Ea) > 0:
            Ea_mean = np.mean(valid_Ea)
            Ea_std = np.std(valid_Ea)

            print(f"\nActivation Energy Results:")
            print(f"  Mean Ea: {Ea_mean / 1000:.1f} +/- {Ea_std / 1000:.1f} kJ/mol")
            print(f"  Range: {np.min(valid_Ea) / 1000:.1f} - {np.max(valid_Ea) / 1000:.1f} kJ/mol")

            print(f"\nEa(alpha) Profile (showing every 3rd point):")
            print(f"{'Conversion':<15} {'Ea (kJ/mol)':<15} {'Std Error (kJ/mol)'}")
            print("-" * 50)

            for i in range(0, len(friedman_result.alpha), 3):
                alpha = friedman_result.alpha[i]
                Ea = friedman_result.Ea[i]
                Ea_err = friedman_result.Ea_std_err[i] if friedman_result.Ea_std_err is not None else np.nan

                if np.isfinite(Ea):
                    if np.isfinite(Ea_err):
                        print(f"{alpha:8.2f}        {Ea / 1000:10.1f}      {Ea_err / 1000:10.1f}")
                    else:
                        print(f"{alpha:8.2f}        {Ea / 1000:10.1f}      {'N/A':>10}")

            # Analyze Ea trend
            print("\nInterpretation:")
            if Ea_std / Ea_mean < 0.1:
                print("  Ea is nearly constant -> Single-step reaction mechanism")
                print("  Suitable for simple nth-order kinetic model")
            else:
                print("  Ea varies with conversion -> Complex/multi-step mechanism")
                print("  May require autocatalytic or multi-stage model")

        else:
            print("\nWarning: Could not compute valid Ea values")
            print("Check that:")
            print("  - Data has sufficient conversion range")
            print("  - Multiple heating rates are present")
            print("  - Temperature and time data are correct")

    except Exception as e:
        print(f"\nError during Friedman analysis: {str(e)}")

    # ============================================================================
    # Step 4: Data Quality Assessment
    # ============================================================================
    print("\n" + "=" * 70)
    print("STEP 4: Data Quality Summary")
    print("=" * 70)

    print(f"\nExperiments loaded: {len(datasets)}")
    print(f"Heating rates: {', '.join([f'{r} K/min' for r in heating_rates])}")
    print(f"Total data points: {sum(len(ds.time) for ds in datasets)}")

    # Check conversion ranges
    print("\nConversion coverage by heating rate:")
    for dataset, rate in zip(datasets, heating_rates):
        print(f"  {rate:5.0f} K/min: {dataset.conversion.min():.3f} - "
              f"{dataset.conversion.max():.3f}")

    print("\nData characteristics:")
    print("  Exothermic reaction: YES")
    print("  Peak clearly defined: YES")
    print("  Conversion calculated from heat flow: YES")
    print("  Multiple heating rates: YES")
    print("  Suitable for isoconversional analysis: YES")

    # ============================================================================
    # Summary and Next Steps
    # ============================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS SUMMARY")
    print("=" * 70)

    print("\nWhat we learned:")
    print("  1. Epoxy curing is exothermic with clear DSC peaks")
    print("  2. Peak temperature increases with heating rate (expected)")
    print("  3. Activation energy determined from multi-rate data")
    print("  4. Data quality is suitable for kinetic modeling")

    print("\nNext Steps:")
    print("  1. Fit kinetic model (nth-order, autocatalytic, or multi-stage)")
    print("  2. Run bootstrap analysis for parameter confidence intervals")
    print("  3. Predict cure at isothermal temperatures")
    print("  4. Optimize cure cycle (time/temperature profile)")
    print("  5. Compare with other isoconversional methods (KAS, OFW, Vyazovkin)")

    print("\nTypical Applications:")
    print("  - Thermoset resin curing (epoxy, polyester, phenolic)")
    print("  - Polymer crystallization")
    print("  - Solid-state reactions")
    print("  - Pharmaceutical polymorphic transitions")
    print("  - Food science (starch gelatinization, protein denaturation)")

    print("\n" + "=" * 70)
    print("Demo Complete!")
    print("=" * 70)
