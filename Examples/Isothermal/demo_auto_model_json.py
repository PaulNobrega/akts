"""
Demo: Automated Isothermal Modeling with JSON Input/Output
===========================================================

This example demonstrates JSON input/output for API integration.
Perfect for web applications and REST APIs!
"""
import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from akts import models, auto_model_isothermal_data


if __name__ == '__main__':
    print("=" * 80)
    print(" AKTS: JSON API Demo for Web Integration")
    print("=" * 80)
    print()

    # Example 1: JSON input data (e.g., from a web API request)
    print("Example 1: JSON Input Data")
    print("-" * 80)

    # Simulate data from a web API POST request
    # Data has realistic experimental noise to demonstrate CI bands
    json_data_313K = {
        "time": [0, 1, 2, 3, 4, 5, 7, 10, 14, 21],  # days
        "temperature": [313, 313, 313, 313, 313, 313, 313, 313, 313, 313],  # K (40°C)
        "conversion": [0.00, 0.04, 0.11, 0.15, 0.24, 0.26, 0.40, 0.50, 0.70, 0.83],  # With measurement variability
        "metadata": {
            "sample_id": "PROT001",
            "temperature_C": 40,
            "units": {"time": "days", "temperature": "K"}
        }
    }

    json_data_323K = {
        "time": [0, 1, 2, 3, 4, 5, 7, 10],  # days
        "temperature": [323, 323, 323, 323, 323, 323, 323, 323],  # K (50°C)
        "conversion": [0.00, 0.10, 0.26, 0.33, 0.48, 0.54, 0.74, 0.86],  # With measurement variability
        "metadata": {
            "sample_id": "PROT001",
            "temperature_C": 50,
            "units": {"time": "days", "temperature": "K"}
        }
    }

    # Add a third temperature for better parameter estimation
    json_data_303K = {
        "time": [0, 2, 4, 7, 10, 14, 21, 28],  # days
        "temperature": [303, 303, 303, 303, 303, 303, 303, 303],  # K (30°C)
        "conversion": [0.00, 0.02, 0.05, 0.09, 0.14, 0.21, 0.35, 0.46],  # Slower degradation
        "metadata": {
            "sample_id": "PROT001",
            "temperature_C": 30,
            "units": {"time": "days", "temperature": "K"}
        }
    }

    print("Input data (3 temperature conditions with measurement variability):")
    print(f"  303 K (30°C): {len(json_data_303K['time'])} data points")
    print(f"  313 K (40°C): {len(json_data_313K['time'])} data points")
    print(f"  323 K (50°C): {len(json_data_323K['time'])} data points")
    print()

    # Run analysis with JSON input
    print("Running automated analysis...")

    try:
        results_dict = auto_model_isothermal_data(
            data_files=[json_data_303K, json_data_313K, json_data_323K],  # JSON dicts as input
            predict=(365, 'days'),  # Predict 1 year ahead
            top_n=2,
            models_to_try=['F1', 'F2', 'A2'],  # Try fewer models for demo
            bootstrap_iterations=50,  # Fewer iterations for speed
            report_path=None,  # Skip HTML report for API use
            output_format='dict'  # Get dict output
        )

        print("[OK] Analysis complete!")
        print()

        # Display results
        print("Results:")
        selected = results_dict['selected_model']
        print(f"  Selected Model: {selected['model_name']}")
        print(f"  R² = {selected['statistics']['r_squared']:.4f}")
        print()

    except Exception as e:
        print(f"[ERROR] Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


    # Create output directory
    from pathlib import Path
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    # Example 2: JSON output for API response (with shipping simulation)
    print()
    print("=" * 80)
    print("Example 2: JSON Output for API Response (with Shipping Simulation)")
    print("-" * 80)
    print()

    try:
        # Define shipping temperature excursion scenario (summer shipping)
        # Baseline storage at 77°F (25°C), heat excursion during transport
        shipping_profile = [
            (0, 77),      # Day 0: Initial storage at 77°F (25°C)
            (3, 77),      # Day 3: Still at controlled storage
            (3.5, 95),    # Day 3.5: Loading/transport begins, temp rises to 95°F (35°C)
            (5, 104),     # Day 5: Peak shipping temperature 104°F (40°C) - hot truck
            (7, 95),      # Day 7: Cooling down during delivery
            (8, 77),      # Day 8: Return to controlled storage 77°F
            (30, 77),     # Day 30: End monitoring period
        ]

        # Run analysis with JSON output (using Fahrenheit for input/output)
        results_json = auto_model_isothermal_data(
            data_files=[json_data_303K, json_data_313K, json_data_323K],
            predict=(365, 'days', 77),       # Predict 1 year at 77°F (25°C)
            simulate=shipping_profile,       # Simulate shipping excursion
            simulate_time_unit='days',
            input_temperature_units='K',     # Input temps in Kelvin
            output_temperature_units='F',    # Output temps in Fahrenheit
            top_n=2,
            models_to_try=['F1', 'F2', 'A2'],
            bootstrap_iterations=50,
            report_path=output_dir / 'json_demo_report.html',
            output_format='json'  # Get JSON string output
        )

        print("JSON output generated:")
        print(f"  Type: {type(results_json)}")
        print(f"  Length: {len(results_json)} characters")
        print()

        # Parse JSON to verify it's valid
        parsed = json.loads(results_json)
        print("JSON is valid! [OK]")
        print()

        print("Top-level keys:")
        for key in parsed.keys():
            print(f"  - {key}")
        print()

        # Show formatted JSON excerpt
        print("Selected model (formatted JSON):")
        print(json.dumps(parsed['selected_model'], indent=2))
        print()

        # Show how to use this in a web API
        print("=" * 80)
        print("Example Web API Integration:")
        print("=" * 80)
        print()

        api_example = '''
# Flask API Example:
from flask import Flask, request, jsonify
from akts import models, auto_model_isothermal_data
import json

app = Flask(__name__)

@app.route('/api/analyze', methods=['POST'])
def analyze_kinetics():
    """
    POST /api/analyze
    Body: {
      "datasets": [
        {"time": [...], "temperature": [...], "conversion": [...]},
        ...
      ],
      "predict_time": 365,
      "predict_unit": "days"
    }
    """
    data = request.get_json()

    # Extract parameters
    datasets = data['datasets']
    predict_time = data.get('predict_time', None)
    predict_unit = data.get('predict_unit', 'days')

    # Run analysis
    results_json = auto_model_isothermal_data(
        data_files=datasets,
        predict=(predict_time, predict_unit) if predict_time else None,
        output_format='json',
        report_path=None
    )

    # Return JSON response
    return results_json, 200, {'Content-Type': 'application/json'}

if __name__ == '__main__':
    app.run(debug=True)
'''
        print(api_example)

    except Exception as e:
        print(f"[ERROR] Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


    # Example 3: Both dict and JSON
    print()
    print("=" * 80)
    print("Example 3: Get Both Dict and JSON")
    print("-" * 80)
    print()

    try:
        results_dict, results_json = auto_model_isothermal_data(
            data_files=[json_data_303K, json_data_313K, json_data_323K],
            predict=(365, 'days'),
            top_n=2,
            models_to_try=['F1', 'F2'],
            bootstrap_iterations=50,
            report_path=None,
            output_format='both'  # Get both formats
        )

        print("Got both formats:")
        print(f"  Dict type: {type(results_dict)}")
        print(f"  JSON type: {type(results_json)}")
        print()

        # Use dict for Python processing
        print("Using dict in Python:")
        for i, model in enumerate(results_dict['top_models'][:2]):
            stats = model.get('stats', model.get('statistics', {}))
            print(f"  {i+1}. {model['model_name']} (R² = {stats.get('r_squared', 0):.4f})")
        print()

        # JSON for API response or storage
        print("JSON ready for:")
        print("  [OK] API response")
        print("  [OK] Database storage")
        print("  [OK] File export")
        print("  [OK] Client-side processing")
        print()

    except Exception as e:
        print(f"[ERROR] Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("=" * 80)
    print(" Summary: JSON API Integration")
    print("=" * 80)
    print()
    print("[OK] JSON input: Pass dicts directly to data_files parameter")
    print("[OK] JSON output: Use output_format='json' for API responses")
    print("[OK] Both formats: Use output_format='both' when you need flexibility")
    print()
    print("Perfect for:")
    print("  • Web APIs (Flask, FastAPI, Django)")
    print("  • Microservices")
    print("  • Cloud functions (AWS Lambda, Azure Functions)")
    print("  • Mobile app backends")
    print("  • Data pipelines")
    print()
    print("=" * 80)
