"""
Simple Demo: Multi-Temperature Isothermal Protein Stability Analysis

This example shows how to load and analyze protein stability data
from isothermal experiments at multiple temperatures.
"""
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from akts import load_instrument_file


if __name__ == '__main__':
    print("=" * 70)
    print("Multi-Temperature Isothermal Stability Study")
    print("Protein Aggregation Monitoring via %HMW")
    print("=" * 70)

    # Load isothermal datasets at different temperatures
    temperatures = [313, 323, 333, 343]  # K (40, 50, 60, 70°C)
    datasets = []

    print("\nLoading isothermal stability data...")
    print("-" * 70)

    for temp_K in temperatures:
        filepath = Path(__file__).parent / f"protein_stability_{temp_K}K.csv"

        if not filepath.exists():
            print(f"Skipping {temp_K}K - file not found")
            continue

        # Load with readout column (%HMW increases as protein aggregates)
        dataset = load_instrument_file(
            filepath,
            time_col='Time (days)',
            temperature_col='Temperature (K)',
            readout_col='HMW Species (%)',
            readout_type='increasing',  # HMW increases with degradation
            auto_detect=True
        )

        datasets.append(dataset)

        temp_C = temp_K - 273.15
        print(f"\n{temp_K} K ({temp_C:.0f}°C):")
        print(f"  Duration: {dataset.time[-1]:.0f} days")
        print(f"  Data points: {len(dataset.time)}")
        print(f"  Conversion range: {dataset.conversion.min():.3f} - {dataset.conversion.max():.3f}")
        print(f"  Temperature: {dataset.temperature.mean():.2f} ± {dataset.temperature.std():.2f} K")

    # Show how conversion relates to %HMW
    print("\n" + "=" * 70)
    print("Understanding the Data")
    print("=" * 70)

    print("\nConversion represents the progress of protein aggregation:")
    print("  Conversion = 0.0 -> Initial state (low %HMW)")
    print("  Conversion = 1.0 -> Final state (high %HMW)")

    print("\nExample: 323K (50°C) dataset")
    ds_323 = datasets[1] if len(datasets) > 1 else datasets[0]

    print(f"\n{'Time (days)':<15} {'Conversion':<15} {'Status'}")
    print("-" * 50)

    sample_indices = [0, len(ds_323.time)//4, len(ds_323.time)//2,
                      3*len(ds_323.time)//4, len(ds_323.time)-1]

    for idx in sample_indices:
        time_days = ds_323.time[idx]
        conversion = ds_323.conversion[idx]

        if conversion < 0.2:
            status = "Early stage"
        elif conversion < 0.5:
            status = "Moderate aggregation"
        elif conversion < 0.8:
            status = "Advanced aggregation"
        else:
            status = "Near completion"

        print(f"{time_days:<15.0f} {conversion:<15.3f} {status}")

    # Temperature dependence
    print("\n" + "=" * 70)
    print("Temperature Dependence of Aggregation Rate")
    print("=" * 70)

    print("\nTime to reach 50% conversion:")
    print(f"{'Temperature':<20} {'Time to 50% conversion'}")
    print("-" * 50)

    for dataset in datasets:
        temp_K = dataset.temperature.mean()
        temp_C = temp_K - 273.15

        # Find time to 50% conversion
        idx_50 = np.argmin(np.abs(dataset.conversion - 0.5))
        time_50 = dataset.time[idx_50]

        print(f"{temp_C:5.0f}°C ({temp_K:6.1f} K)  {time_50:10.1f} days")

    print("\nKey Observation:")
    print("  Higher temperatures -> Faster aggregation")
    print("  Lower temperatures -> Slower aggregation (better stability)")

    # Data quality summary
    print("\n" + "=" * 70)
    print("Data Summary")
    print("=" * 70)

    print(f"\nTotal datasets loaded: {len(datasets)}")
    print(f"Temperature range: {temperatures[0] - 273.15:.0f}°C - {temperatures[-1] - 273.15:.0f}°C")
    print(f"Total data points: {sum(len(ds.time) for ds in datasets)}")

    print("\nNext Steps:")
    print("  1. Run isoconversional analysis (Friedman, KAS, OFW) to determine Ea")
    print("  2. Fit kinetic model (F1, A2, etc.) to describe aggregation mechanism")
    print("  3. Use bootstrap analysis for parameter uncertainty")
    print("  4. Predict long-term stability at storage temperatures")
    print("  5. Extrapolate to refrigerated conditions (2-8°C)")

    print("\nThese data are ready for kinetic analysis with akts!")
    print("=" * 70)
