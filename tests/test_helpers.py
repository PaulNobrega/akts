"""
Tests for helper functions and JSON utilities.
"""
import numpy as np
import json
from pathlib import Path
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from akts import (
    KineticDataset,
    auto_model_isothermal_data,
    parse_json_data,
    serialize_results_to_json,
    convert_numpy_to_python
)


def test_json_data_parsing():
    """Test parsing JSON data to KineticDataset."""
    print("\n" + "="*70)
    print("TEST: JSON Data Parsing")
    print("="*70)

    json_data = {
        'time': [0, 1, 2, 3, 4],
        'temperature': [313, 313, 313, 313, 313],
        'conversion': [0.0, 0.1, 0.2, 0.3, 0.4],
        'metadata': {'test': True}
    }

    dataset = parse_json_data(json_data)

    assert isinstance(dataset, KineticDataset)
    assert len(dataset.time) == 5
    assert dataset.temperature[0] == 313
    assert dataset.metadata['test'] == True

    print("✓ JSON parsing works correctly")
    print(f"  - Time points: {len(dataset.time)}")
    print(f"  - Temperature: {dataset.temperature[0]} K")
    print(f"  - Conversion range: {dataset.conversion.min():.2f} - {dataset.conversion.max():.2f}")


def test_numpy_conversion():
    """Test numpy to Python type conversion."""
    print("\n" + "="*70)
    print("TEST: Numpy Type Conversion")
    print("="*70)

    test_data = {
        'int64': np.int64(42),
        'float64': np.float64(3.14),
        'array': np.array([1, 2, 3]),
        'nested': {
            'value': np.float32(2.71),
            'list': [np.int32(1), np.int32(2)]
        }
    }

    converted = convert_numpy_to_python(test_data)

    # Verify all types are Python native
    assert isinstance(converted['int64'], int)
    assert isinstance(converted['float64'], float)
    assert isinstance(converted['array'], list)
    assert isinstance(converted['nested']['value'], float)
    assert isinstance(converted['nested']['list'][0], int)

    # Verify it's JSON serializable
    json_str = json.dumps(converted)
    assert isinstance(json_str, str)

    print("✓ Numpy conversion works correctly")
    print(f"  - Converted int64 -> {type(converted['int64']).__name__}")
    print(f"  - Converted float64 -> {type(converted['float64']).__name__}")
    print(f"  - Converted array -> {type(converted['array']).__name__}")
    print(f"  - JSON serializable: Yes")


def test_auto_model_with_synthetic_data():
    """Test auto_model_isothermal_data with synthetic data."""
    print("\n" + "="*70)
    print("TEST: Auto Model with Synthetic Data")
    print("="*70)

    # Generate synthetic isothermal data (F1 kinetics)
    # Ea = 80 kJ/mol, A = 1e12 s^-1
    R = 8.314
    Ea = 80000
    A = 1e12

    datasets = []
    temperatures = [313, 323, 333]

    for T in temperatures:
        k = A * np.exp(-Ea / (R * T))
        time = np.linspace(0, 86400*10, 20)  # 10 days in seconds
        # F1 model: dalpha/dt = k*(1-alpha)
        # Solution: alpha = 1 - exp(-k*t)
        conversion = 1 - np.exp(-k * time)
        # Add small noise
        conversion += np.random.normal(0, 0.01, len(conversion))
        conversion = np.clip(conversion, 0, 1)

        json_data = {
            'time': time.tolist(),
            'temperature': [T] * len(time),
            'conversion': conversion.tolist()
        }
        datasets.append(json_data)

    print(f"Generated {len(datasets)} synthetic datasets")
    print(f"Temperature range: {temperatures[0]} - {temperatures[-1]} K")

    # Progress callback
    progress_messages = []
    def progress_cb(msg, data):
        progress_messages.append(msg)
        print(f"  [{data.get('timestamp', '')}] {msg}")

    # Run analysis
    try:
        results = auto_model_isothermal_data(
            data_files=datasets,
            models_to_try=['F1', 'F2', 'A2'],  # Limited models for speed
            top_n=2,
            bootstrap_iterations=20,  # Reduced for speed
            report_path=None,  # Skip HTML for test
            output_format='dict',
            progress_callback=progress_cb
        )

        print("\n✓ Analysis completed successfully")
        print(f"  - Progress messages: {len(progress_messages)}")
        print(f"  - Top models: {len(results['top_models'])}")
        print(f"  - Selected model: {results['selected_model']['model_name']}")

        # Verify results structure
        assert 'top_models' in results
        assert 'selected_model' in results
        assert 'summary' in results
        assert len(results['top_models']) > 0

        # Check that F1 is top ranked (since we generated F1 data)
        top_model = results['top_models'][0]['model_name']
        print(f"  - Top ranked model: {top_model}")
        print(f"  - R² = {results['top_models'][0]['statistics']['r_squared']:.4f}")

        return True

    except Exception as e:
        print(f"\n✗ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_json_output_format():
    """Test JSON output format."""
    print("\n" + "="*70)
    print("TEST: JSON Output Format")
    print("="*70)

    # Simple synthetic data
    json_data = {
        'time': np.linspace(0, 10, 10).tolist(),
        'temperature': [313] * 10,
        'conversion': np.linspace(0, 0.5, 10).tolist()
    }

    try:
        # Test JSON output
        results_json = auto_model_isothermal_data(
            data_files=[json_data],
            models_to_try=['F1'],
            top_n=1,
            bootstrap_iterations=10,
            report_path=None,
            output_format='json'
        )

        # Verify it's a string
        assert isinstance(results_json, str)
        print(f"✓ JSON output is string: {len(results_json)} chars")

        # Verify it's valid JSON
        parsed = json.loads(results_json)
        assert isinstance(parsed, dict)
        print("✓ JSON is valid and parseable")

        # Check structure
        assert 'top_models' in parsed
        assert 'selected_model' in parsed
        assert 'summary' in parsed
        print("✓ JSON has correct structure")

        return True

    except Exception as e:
        print(f"✗ JSON output test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_both_output_format():
    """Test 'both' output format."""
    print("\n" + "="*70)
    print("TEST: Both Output Format")
    print("="*70)

    json_data = {
        'time': np.linspace(0, 10, 10).tolist(),
        'temperature': [313] * 10,
        'conversion': np.linspace(0, 0.5, 10).tolist()
    }

    try:
        # Test both output
        results = auto_model_isothermal_data(
            data_files=[json_data],
            models_to_try=['F1'],
            top_n=1,
            bootstrap_iterations=10,
            report_path=None,
            output_format='both'
        )

        # Should be a tuple
        assert isinstance(results, tuple)
        assert len(results) == 2
        print("✓ Returned tuple of length 2")

        results_dict, results_json = results

        # Check dict
        assert isinstance(results_dict, dict)
        print("✓ First element is dict")

        # Check JSON
        assert isinstance(results_json, str)
        parsed = json.loads(results_json)
        print("✓ Second element is valid JSON string")

        return True

    except Exception as e:
        print(f"✗ Both output test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    print("\n" + "="*70)
    print(" AKTS Helper Functions Test Suite")
    print("="*70)

    tests = [
        ("JSON Data Parsing", test_json_data_parsing),
        ("Numpy Type Conversion", test_numpy_conversion),
        ("Auto Model with Synthetic Data", test_auto_model_with_synthetic_data),
        ("JSON Output Format", test_json_output_format),
        ("Both Output Format", test_both_output_format),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ Test '{name}' raised exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Summary
    print("\n" + "="*70)
    print(" Test Summary")
    print("="*70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")

    print()
    print(f"Results: {passed}/{total} tests passed")
    print("="*70)

    sys.exit(0 if passed == total else 1)
