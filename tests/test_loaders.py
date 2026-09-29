"""
Test script for data loaders (Section 4).

Tests loading of DSC, TGA, and isothermal experiment data from CSV files.
"""
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from akts.loaders import (
    load_data_file,
    load_dsc_file,
    load_tga_file,
    load_isothermal_file
)


def print_dataset_info(dataset, title="Dataset"):
    """Print summary information about a loaded dataset."""
    print(f"\n{title}")
    print("=" * 60)
    print(f"Number of data points: {len(dataset.time)}")
    print(f"Time range: {dataset.time[0]:.1f} - {dataset.time[-1]:.1f} s")
    print(f"  Duration: {(dataset.time[-1] - dataset.time[0]) / 60:.1f} minutes")
    print(f"Temperature range: {dataset.temperature.min():.2f} - {dataset.temperature.max():.2f} K")
    print(f"  (= {dataset.temperature.min() - 273.15:.2f} - {dataset.temperature.max() - 273.15:.2f} °C)")
    print(f"Conversion range: {dataset.conversion.min():.4f} - {dataset.conversion.max():.4f}")
    print(f"Heating rate: {dataset.heating_rate:.6f} K/s ({dataset.heating_rate * 60:.2f} K/min)")

    print("\nMetadata:")
    for key, value in dataset.metadata.items():
        if isinstance(value, str):
            print(f"  {key}: {value}")
        else:
            print(f"  {key}: {value}")

    # Calculate some statistics
    if len(dataset.time) > 1:
        avg_timestep = np.mean(np.diff(dataset.time))
        print(f"\nAverage time step: {avg_timestep:.2f} s")

    # Show first and last few points
    print("\nFirst 3 data points:")
    print("  Time (s)  | Temp (K) | Conversion")
    print("  " + "-" * 40)
    for i in range(min(3, len(dataset.time))):
        print(f"  {dataset.time[i]:8.1f}  | {dataset.temperature[i]:8.2f} | {dataset.conversion[i]:10.6f}")

    print("\nLast 3 data points:")
    print("  Time (s)  | Temp (K) | Conversion")
    print("  " + "-" * 40)
    for i in range(max(0, len(dataset.time) - 3), len(dataset.time)):
        print(f"  {dataset.time[i]:8.1f}  | {dataset.temperature[i]:8.2f} | {dataset.conversion[i]:10.6f}")


def test_dsc_loader():
    """Test DSC data loading."""
    print("\n" + "=" * 60)
    print("TEST 1: DSC Data Loader")
    print("=" * 60)

    dsc_file = Path(__file__).parent / "DSC" / "example_dsc_10K_min.csv"

    if not dsc_file.exists():
        print(f"[FAIL] DSC file not found: {dsc_file}")
        return False

    try:
        # Load using specific loader
        dataset = load_dsc_file(dsc_file)
        print_dataset_info(dataset, "DSC Data (10 K/min heating)")

        # Verify conversion was computed from heat flow
        assert dataset.metadata.get('conversion_source') == 'dsc_heat_flow', \
            "Conversion should be computed from DSC heat flow"

        # Check that conversion makes sense
        assert 0 <= dataset.conversion.min() <= 0.1, "Minimum conversion should be near 0"
        assert 0.9 <= dataset.conversion.max() <= 1.0, "Maximum conversion should be near 1"

        print("\n[PASS] DSC loader test passed!")
        return True

    except Exception as e:
        print(f"\n[FAIL] DSC loader test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_tga_loader():
    """Test TGA data loading."""
    print("\n" + "=" * 60)
    print("TEST 2: TGA Data Loader")
    print("=" * 60)

    tga_file = Path(__file__).parent / "TGA" / "example_tga_5K_min.csv"

    if not tga_file.exists():
        print(f"[FAIL] TGA file not found: {tga_file}")
        return False

    try:
        # Load using specific loader
        dataset = load_tga_file(tga_file)
        print_dataset_info(dataset, "TGA Data (5 K/min heating)")

        # Verify conversion was computed from mass
        assert dataset.metadata.get('conversion_source') == 'tga_mass', \
            "Conversion should be computed from TGA mass"

        # Check that conversion makes sense
        assert 0 <= dataset.conversion.min() <= 0.1, "Minimum conversion should be near 0"
        assert 0.9 <= dataset.conversion.max() <= 1.0, "Maximum conversion should be near 1"

        # Check monotonic increase
        assert np.all(np.diff(dataset.conversion) >= -0.01), \
            "Conversion should be monotonically increasing (allowing small noise)"

        print("\n[PASS] TGA loader test passed!")
        return True

    except Exception as e:
        print(f"\n[FAIL] TGA loader test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_isothermal_loader():
    """Test isothermal data loading."""
    print("\n" + "=" * 60)
    print("TEST 3: Isothermal Data Loader")
    print("=" * 60)

    iso_file = Path(__file__).parent / "Isothermal" / "example_isothermal_373K.csv"

    if not iso_file.exists():
        print(f"[FAIL] Isothermal file not found: {iso_file}")
        return False

    try:
        # Load using specific loader
        dataset = load_isothermal_file(iso_file)
        print_dataset_info(dataset, "Isothermal Data (373 K)")

        # Verify conversion was provided directly
        assert dataset.metadata.get('conversion_source') == 'direct', \
            "Conversion should be directly from file"

        # Check temperature is roughly constant
        temp_std = np.std(dataset.temperature)
        assert temp_std < 0.5, f"Temperature should be constant (std = {temp_std:.3f} K)"

        # Check mean temperature
        mean_temp = np.mean(dataset.temperature)
        assert 372 < mean_temp < 374, f"Mean temperature should be ~373 K (got {mean_temp:.2f} K)"

        # Check conversion reaches near 1
        assert dataset.conversion.max() > 0.8, "Conversion should reach high values"

        print("\n[PASS] Isothermal loader test passed!")
        return True

    except Exception as e:
        print(f"\n[FAIL] Isothermal loader test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_generic_loader():
    """Test generic load_data_file function."""
    print("\n" + "=" * 60)
    print("TEST 4: Generic Instrument File Loader")
    print("=" * 60)

    dsc_file = Path(__file__).parent / "DSC" / "example_dsc_10K_min.csv"

    try:
        # Test with auto-detection
        dataset = load_data_file(dsc_file, heat_flow_col='Heat Flow', auto_detect=True)

        print("\nAuto-detection results:")
        print(f"  Time column: Found")
        print(f"  Temperature column: Found")
        print(f"  Heat Flow column: Found")
        print(f"  Conversion source: {dataset.metadata.get('conversion_source')}")

        assert len(dataset.time) > 0, "Should have loaded data"

        print("\n[PASS] Generic loader test passed!")
        return True

    except Exception as e:
        print(f"\n[FAIL] Generic loader test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_column_auto_detection():
    """Test automatic column name detection."""
    print("\n" + "=" * 60)
    print("TEST 5: Column Auto-Detection")
    print("=" * 60)

    tga_file = Path(__file__).parent / "TGA" / "example_tga_5K_min.csv"

    try:
        # Test without specifying exact column names
        dataset = load_data_file(
            tga_file,
            time_col='Time',  # Will try to find this or alternatives
            temperature_col='Temperature',
            mass_col='Mass',
            auto_detect=True
        )

        print("\nDetected columns:")
        for key in ['time_units', 'temperature_units', 'mass_units']:
            if key in dataset.metadata:
                print(f"  {key}: {dataset.metadata[key]}")

        assert len(dataset.time) > 0, "Should have loaded data with auto-detection"

        print("\n[PASS] Column auto-detection test passed!")
        return True

    except Exception as e:
        print(f"\n[FAIL] Column auto-detection test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_data_quality():
    """Test that loaded data has reasonable quality."""
    print("\n" + "=" * 60)
    print("TEST 6: Data Quality Checks")
    print("=" * 60)

    files_to_test = [
        ("DSC", Path(__file__).parent / "DSC" / "example_dsc_10K_min.csv", load_dsc_file),
        ("TGA", Path(__file__).parent / "TGA" / "example_tga_5K_min.csv", load_tga_file),
        ("Isothermal", Path(__file__).parent / "Isothermal" / "example_isothermal_373K.csv",
         load_isothermal_file),
    ]

    all_passed = True

    for name, filepath, loader in files_to_test:
        print(f"\nChecking {name} data quality...")

        try:
            dataset = loader(filepath)

            # Check for NaNs
            assert not np.any(np.isnan(dataset.time)), f"{name}: Time contains NaNs"
            assert not np.any(np.isnan(dataset.temperature)), f"{name}: Temperature contains NaNs"
            assert not np.any(np.isnan(dataset.conversion)), f"{name}: Conversion contains NaNs"

            # Check ranges
            assert np.all(dataset.time >= 0), f"{name}: Time should be non-negative"
            assert np.all(dataset.temperature > 0), f"{name}: Temperature should be positive"
            assert np.all(dataset.conversion >= 0), f"{name}: Conversion should be >= 0"
            assert np.all(dataset.conversion <= 1.05), f"{name}: Conversion should be <= 1.0 (allowing 5% error)"

            # Check monotonicity
            assert np.all(np.diff(dataset.time) >= 0), f"{name}: Time should be monotonic"

            # Check sufficient data points
            assert len(dataset.time) >= 50, f"{name}: Should have at least 50 data points"

            print(f"  [PASS] {name} data quality OK")

        except AssertionError as e:
            print(f"  [FAIL] {name} data quality check failed: {str(e)}")
            all_passed = False
        except Exception as e:
            print(f"  [FAIL] {name} loader failed: {str(e)}")
            all_passed = False

    if all_passed:
        print("\n[PASS] All data quality checks passed!")
    else:
        print("\n[FAIL] Some data quality checks failed")

    return all_passed


def main():
    """Run all loader tests."""
    print("\n" + "=" * 60)
    print("AKTS Data Loader Tests (Section 4)")
    print("Testing CSV/XLSX file loading for DSC, TGA, and Isothermal data")
    print("=" * 60)

    results = []

    # Run all tests
    results.append(("DSC Loader", test_dsc_loader()))
    results.append(("TGA Loader", test_tga_loader()))
    results.append(("Isothermal Loader", test_isothermal_loader()))
    results.append(("Generic Loader", test_generic_loader()))
    results.append(("Column Auto-Detection", test_column_auto_detection()))
    results.append(("Data Quality", test_data_quality()))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    for name, passed in results:
        status = "[PASS] PASS" if passed else "[FAIL] FAIL"
        print(f"{status}: {name}")

    all_passed = all(result[1] for result in results)

    if all_passed:
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("SOME TESTS FAILED")
        print("=" * 60)

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
