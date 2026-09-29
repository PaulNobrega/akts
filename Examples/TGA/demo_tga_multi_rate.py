"""

if __name__ == "__main__":
    Demo: TGA Multi-Rate Kinetic Analysis

    This example demonstrates a typical TGA workflow:
    - Polymer thermal decomposition
    - Multiple heating rates (5, 10, 15, 20 K/min)
    - Loading raw mass data
    - Isoconversional analysis
    - Understanding degradation kinetics
    """
    import numpy as np
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent))

    from akts import load_tga_file, run_friedman

    print("=" * 70)
    print("TGA Multi-Rate Kinetic Analysis Demo")
    print("Polymer Thermal Decomposition - Isoconversional Analysis")
    print("=" * 70)

    # ============================================================================
    # Step 1: Load Multiple Heating Rate TGA Data
    # ============================================================================
    print("\n" + "=" * 70)
    print("STEP 1: Load TGA Data at Multiple Heating Rates")
    print("=" * 70)

    heating_rates = [5, 10, 15, 20]  # K/min
    datasets = []

    print("\nLoading TGA experiments...")
    print("-" * 70)

    for rate in heating_rates:
        filepath = Path(__file__).parent / f"polymer_decomp_{rate}K_min.csv"

        if not filepath.exists():
            print(f"Skipping {rate} K/min - file not found")
            continue

        # Load TGA data - automatically computes conversion from mass loss
        dataset = load_tga_file(
            filepath,
            time_col='Time (min)',
            temperature_col='Temperature (C)',
            mass_col='Mass (mg)',
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
        print(f"  Mass loss: {(dataset.conversion.max() - dataset.conversion.min()) * 100:.1f}%")
        print(f"  Data points: {len(dataset.time)}")

    # ============================================================================
    # Step 2: Understand the Data
    # ============================================================================
    print("\n" + "=" * 70)
    print("STEP 2: Understanding TGA Mass Loss Data")
    print("=" * 70)

    print("\nWhat is TGA measuring?")
    print("  TGA tracks sample mass as temperature increases")
    print("  Mass loss indicates thermal degradation or volatilization")
    print("\nConversion in TGA:")
    print("  Conversion = (m_initial - m(t)) / (m_initial - m_final)")
    print("  This is calculated automatically by akts!")

    # Show decomposition characteristics
    print("\nDecomposition Characteristics:")
    print(f"{'Heating Rate':<20} {'T at 50% loss (C)':<25} {'Onset Temp (C)'}")
    print("-" * 70)

    for dataset, rate in zip(datasets, heating_rates):
        # Find temperature at 50% conversion
        idx_50 = np.argmin(np.abs(dataset.conversion - 0.5))
        T_50_C = dataset.temperature[idx_50] - 273.15

        # Estimate onset temperature (10% conversion)
        idx_10 = np.argmin(np.abs(dataset.conversion - 0.1))
        T_onset_C = dataset.temperature[idx_10] - 273.15

        print(f"{rate:5.0f} K/min          {T_50_C:15.1f} C        {T_onset_C:15.1f} C")

    print("\nKey Observation:")
    print("  Higher heating rate -> Higher decomposition temperature")
    print("  This is the basis for kinetic analysis!")

    # ============================================================================
    # Step 3: Isoconversional Analysis (Friedman)
    # ============================================================================
    print("\n" + "=" * 70)
    print("STEP 3: Friedman Isoconversional Analysis")
    print("=" * 70)

    print("\nDetermining activation energy Ea as a function of conversion...")
    print("This reveals if decomposition is single-step or multi-step.")

    # Run Friedman isoconversional analysis
    alpha_levels = np.linspace(0.15, 0.85, 15)

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

            print(f"\nEa(alpha) Profile:")
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
            Ea_trend = np.gradient(valid_Ea)
            avg_trend = np.mean(Ea_trend)

            if abs(Ea_std / Ea_mean) < 0.15:
                print("  Ea is relatively constant -> Likely single-step decomposition")
                print("  Suitable for simple nth-order model")
            else:
                if avg_trend > 0:
                    print("  Ea increases with conversion -> Multi-step process")
                    print("  Later stages have higher energy barrier")
                else:
                    print("  Ea decreases with conversion -> Competing reactions")
                    print("  Initial stage may be rate-limiting")

        else:
            print("\nWarning: Could not compute valid Ea values")
            print("Check data quality and heating rate coverage")

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

    # Check mass loss
    print("\nMass loss by heating rate:")
    for dataset, rate in zip(datasets, heating_rates):
        mass_loss_pct = dataset.conversion.max() * 100
        print(f"  {rate:5.0f} K/min: {mass_loss_pct:5.1f}% mass loss")

    print("\nDecomposition stages detected:")
    # Simple heuristic: look for inflection points
    for i, (dataset, rate) in enumerate(zip(datasets, heating_rates)):
        dadt = np.gradient(dataset.conversion, dataset.time)
        # Count peaks in derivative
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(dadt, height=np.max(dadt) * 0.3, distance=10)
        n_stages = len(peaks)
        print(f"  {rate:5.0f} K/min: ~{n_stages} decomposition stage(s)")

    # ============================================================================
    # Summary and Next Steps
    # ============================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS SUMMARY")
    print("=" * 70)

    print("\nWhat we learned:")
    print("  1. Polymer shows clear thermal decomposition")
    print("  2. Decomposition temperature increases with heating rate")
    print("  3. Activation energy determined from multi-rate data")
    print("  4. Multi-stage decomposition possible (check Ea profile)")

    print("\nNext Steps:")
    print("  1. Try other isoconversional methods (KAS, OFW, Vyazovkin)")
    print("  2. Fit kinetic models (nth-order, diffusion, multi-step)")
    print("  3. Predict thermal stability at service temperatures")
    print("  4. Determine safe operating temperature limits")
    print("  5. Compare with manufacturer specifications")

    print("\nTypical Applications:")
    print("  - Polymer thermal stability assessment")
    print("  - Lifetime prediction for materials")
    print("  - Quality control (batch-to-batch comparison)")
    print("  - Failure analysis (degraded vs. fresh samples)")
    print("  - Formulation optimization (additive effects)")
    print("  - Regulatory compliance (thermal hazard assessment)")

    print("\n" + "=" * 70)
    print("Demo Complete!")
    print("=" * 70)
