# JSON Input/Output Specification

Complete specification for using `auto_model_isothermal_data()` with JSON for web API integration.

## Overview

The `auto_model_isothermal_data()` function fully supports JSON input and output, making it perfect for:
- REST APIs (Flask, FastAPI, Django)
- Microservices
- Cloud functions (AWS Lambda, Azure Functions)
- Mobile app backends
- Web applications

## Table of Contents

- [Input Format](#input-format)
- [Output Format](#output-format)
- [Complete Example](#complete-example)
- [Flask API Example](#flask-api-example)
- [FastAPI Example](#fastapi-example)
- [Error Handling](#error-handling)
- [Performance Considerations](#performance-considerations)

---

## Input Format

### Dataset JSON Structure

Each dataset is a JSON object with three required arrays:

```json
{
  "time": [0, 86400, 172800, 259200],
  "temperature": [313.15, 313.15, 313.15, 313.15],
  "conversion": [0.0, 0.05, 0.12, 0.20]
}
```

**Fields**:
- `time` (array of numbers): Time points in specified units
- `temperature` (array of numbers): Temperature at each time point
- `conversion` (array of numbers): Fractional conversion (0.0 to 1.0)

**Optional metadata**:
```json
{
  "time": [...],
  "temperature": [...],
  "conversion": [...],
  "metadata": {
    "sample_id": "SAMPLE-001",
    "batch": "2024-01",
    "notes": "Accelerated aging study"
  }
}
```

### Function Call with JSON

```python
from akts import auto_model_isothermal_data

# JSON data (from API request)
json_datasets = [
    {
        "time": [0, 86400, 172800],
        "temperature": [313, 313, 313],
        "conversion": [0.0, 0.1, 0.2]
    },
    {
        "time": [0, 43200, 86400],
        "temperature": [333, 333, 333],
        "conversion": [0.0, 0.15, 0.3]
    }
]

# Call function
results = auto_model_isothermal_data(
    data_files=json_datasets,  # Pass JSON directly
    predict=(1, 'year'),
    output_format='json'  # Get JSON output
)
```

---

## Output Format

### Output Modes

```python
# Mode 1: Dictionary (default)
results_dict = auto_model_isothermal_data(..., output_format='dict')
# Returns: Dict

# Mode 2: JSON string
results_json = auto_model_isothermal_data(..., output_format='json')
# Returns: str (valid JSON)

# Mode 3: Both
results_dict, results_json = auto_model_isothermal_data(..., output_format='both')
# Returns: Tuple[Dict, str]
```

### Complete Output Structure

```json
{
  "top_models": [
    {
      "rank": 1,
      "model_name": "F2_model",
      "parameters": {
        "Ea": 95234.5,
        "A": 1.23e12
      },
      "stats": {
        "r_squared": 0.9982,
        "r_squared_adj": 0.9978,
        "aic": -150.23,
        "bic": -145.12,
        "rss": 0.00123,
        "rmse": 0.0091,
        "n_params": 2,
        "n_points": 24,
        "durbin_watson": 1.92,
        "is_physically_plausible": true,
        "plausibility_issues": [],
        "akaike_weight": 0.87
      },
      "n_params": 2,
      "score": -0.87,
      "simplicity_penalty": 0.0
    }
  ],

  "selected_model": {
    "model_name": "F2_model",
    "rank": 1,
    "reason": "Strong evidence (Akaike weight = 87.0%)",
    "parameters": {
      "Ea": 95234.5,
      "A": 1.23e12
    },
    "statistics": {
      "r_squared": 0.9982,
      "r_squared_adj": 0.9978,
      "aic": -150.23,
      "bic": -145.12,
      "rss": 0.00123,
      "rmse": 0.0091,
      "n_params": 2,
      "n_points": 24,
      "akaike_weight": 0.87
    },
    "physical_sanity_flags": []
  },

  "predictions": {
    "time": [0, 31536, 63072, ...],
    "conversion_mean": [0.0, 0.05, 0.12, ...],
    "conversion_lower": [0.0, 0.04, 0.10, ...],
    "conversion_upper": [0.0, 0.06, 0.14, ...],
    "temperature": 298.15,
    "temperature_units": "K",
    "temperature_K": 298.15,
    "time_unit": "seconds",
    "requested_time": {
      "value": 1,
      "unit": "year"
    }
  },

  "simulation": {
    "time": [0, 432, 864, ...],
    "conversion_mean": [0.0, 0.001, 0.003, ...],
    "conversion_lower": [0.0, 0.0009, 0.0025, ...],
    "conversion_upper": [0.0, 0.0011, 0.0035, ...],
    "temperature": [298, 313, 298, ...],
    "temperature_units": "K",
    "time_unit": "days",
    "input_profile": [[0, 298], [5, 313], [10, 298]]
  },

  "bootstrap_results": null,

  "report_path": null,

  "summary": {
    "datasets_count": 2,
    "total_datapoints": 24,
    "temperature_range": "313.00 - 333.00 °K",
    "models_tried": 12,
    "models_successful": 12,
    "top_n_selected": 3,
    "bootstrap_iterations": 0
  }
}
```

### Field Descriptions

#### `top_models` (array)
The `top_n` highest-ranked models that passed the ranking filters (R² ≥
`min_r_squared` and physical plausibility, relaxed with a warning if nothing
passes). Models are ranked by Akaike weight only.

**Fields per model**:
- `rank` (int): Ranking (1 = best)
- `model_name` (string): Model identifier (e.g., "F2_model")
- `parameters` (object): Model parameters (Ea in J/mol, A in s⁻¹)
- `stats` (object): Statistical metrics
  - `r_squared` (float): Coefficient of determination
  - `r_squared_adj` (float): Adjusted R², penalized for parameter count
  - `aic` (float): Akaike Information Criterion (lower is better)
  - `bic` (float): Bayesian Information Criterion (lower is better)
  - `rss` (float): Residual sum of squares
  - `rmse` (float): Root-mean-square error (`sqrt(rss / n_points)`)
  - `n_params` (int): Number of fitted parameters
  - `n_points` (int): Number of data points used in the fit
  - `akaike_weight` (float): Probability this model is the best among the candidate
    set, given AICc (computed jointly across the models that passed the filters;
    sums to 1 across `top_models`' underlying candidate set, not just the entries shown)
  - `durbin_watson` (float): Residual autocorrelation statistic (≈2 is ideal)
  - `is_physically_plausible` (bool or null): Parameter sanity check
    (A < 1e20 s⁻¹, 5 < Ea < 1000 kJ/mol)
  - `plausibility_issues` (array of strings or null): Descriptions of any issues
- `n_params` (int): Number of fitted parameters
- `score` (float): Negated Akaike weight (`-akaike_weight`); lower is better and
  determines the rank
- `simplicity_penalty` (float): Always `0.0`. AIC's 2k term already penalizes
  complexity; the field is kept for compatibility
- `filter_warning` (object, rank 1 only, optional): Present when the filters had to
  be relaxed. `type` is `'no_plausible_models'` (no model passed both R² and
  plausibility, so R²-passing models were kept) or `'no_good_models'` (no model
  reached `min_r_squared`, so all were kept), plus a `message`

#### `selected_model` (object)
The best model selected by the algorithm.

**Fields**: Same as `top_models` entry (with `stats` renamed to `statistics`), plus:
- `reason` (string): Why this model was selected. The selected model is always
  rank 1; the reason reports the strength of evidence from its Akaike weight:
  `"Overwhelming evidence (...)"` (≥ 0.90), `"Strong evidence (...)"` (≥ 0.70),
  `"Substantial support (...)"` (≥ 0.50), otherwise
  `"Best of N competitive models (...)"`. The same string is also returned as the
  top-level `selection_reason` in the dict output
- `physical_sanity_flags` (array of strings): Warnings when a fitted activation
  energy (`Ea`/`Ea1`/`Ea2`/...) falls outside the plausible 30-180 kJ/mol range
  typical for drug degradation kinetics. Empty list if nothing was flagged. A
  flagged fit is not necessarily wrong — e.g. diffusion-limited or unusually
  stable formulations can genuinely fall outside this range — but it is worth a
  second look.

#### `predictions` (object)
Predicted conversion over time at specified temperature.

**Present only if** `predict` parameter was provided.

**Fields**:
- `time` (array of floats): Time points in seconds
- `conversion_mean` (array of floats): Mean predicted conversion (0-1)
- `conversion_lower` (array of floats): Lower CI bound (if bootstrap ran)
- `conversion_upper` (array of floats): Upper CI bound (if bootstrap ran)

With the default `ci_type='two-sided'`, the bounds are equal-tailed bootstrap
percentiles (2.5th and 97.5th at `confidence_level=0.95`). With
`ci_type='one-sided'`, the band is (fitted curve, 95th percentile). The ICH Q1E
bootstrap shelf-life always uses its own one-sided 95% bound, whichever
`ci_type` is set.
- `temperature` (float): Prediction temperature in output units
- `temperature_units` (string): 'K', 'C', or 'F'
- `temperature_K` (float): Prediction temperature in Kelvin (for compatibility)
- `time_unit` (string): Always "seconds"
- `requested_time` (object): Original prediction request

#### `simulation` (object)
Conversion under variable temperature profile.

**Present only if** `simulate` parameter was provided.

**Fields**:
- `time` (array of floats): Time points
- `conversion_mean` (array of floats): Mean conversion at each time
- `conversion_lower` (array of floats): Lower CI bound (same `ci_type` as `predictions`)
- `conversion_upper` (array of floats): Upper CI bound
- `temperature` (array of floats): Temperature at each time point
- `temperature_units` (string): Output temperature units
- `time_unit` (string): Time units (from `simulate_time_unit`)
- `input_profile` (array of arrays): Original profile [[time, temp], ...]

#### `summary` (object)
Analysis summary statistics.

**Fields**:
- `datasets_count` (int): Number of datasets analyzed
- `total_datapoints` (int): Total data points across all datasets
- `temperature_range` (string): Formatted minimum and maximum temperatures,
  including the output unit, for example `"25.00 - 40.00 °C"`
- `models_tried` (int): Number of models attempted
- `models_successful` (int): Number that converged
- `top_n_selected` (int): Number of top models kept
- `bootstrap_iterations` (int): Actual bootstrap iterations completed

---

## Complete Example

### Request

```python
import requests
import json

# Prepare data
request_data = {
    "datasets": [
        {
            "time": [0, 86400, 172800, 259200],
            "temperature": [313, 313, 313, 313],
            "conversion": [0.0, 0.05, 0.12, 0.20]
        }
    ],
    "predict": {
        "value": 2,
        "unit": "year",
        "temperature": 25
    },
    "temperature_units": {
        "input": "K",
        "output": "C"
    },
    "models": ["F1", "F2", "A2"],
    "bootstrap": 50
}

# Send to API
response = requests.post(
    'https://api.example.com/analyze',
    json=request_data,
    headers={'Content-Type': 'application/json'}
)

# Get results
results = response.json()
print(f"Best model: {results['selected_model']['model_name']}")
print(f"2-year prediction: {results['predictions']['conversion_mean'][-1]:.1%}")
```

### Response

```json
{
  "top_models": [
    {
      "rank": 1,
      "model_name": "F2_model",
      "parameters": {"Ea": 95234, "A": 1.23e12},
      "stats": {
        "r_squared": 0.9982, "r_squared_adj": 0.9978,
        "aic": -150.2, "bic": -145.1, "rss": 0.0012, "rmse": 0.0091,
        "n_params": 2, "n_points": 4, "akaike_weight": 0.87
      },
      "n_params": 2,
      "score": -0.87,
      "simplicity_penalty": 0.0
    }
  ],
  "selected_model": {
    "model_name": "F2_model",
    "rank": 1,
    "reason": "Strong evidence (Akaike weight = 87.0%)",
    "parameters": {"Ea": 95234, "A": 1.23e12},
    "statistics": {
      "r_squared": 0.9982, "r_squared_adj": 0.9978,
      "aic": -150.2, "bic": -145.1, "rss": 0.0012, "rmse": 0.0091,
      "n_params": 2, "n_points": 4, "akaike_weight": 0.87
    },
    "physical_sanity_flags": []
  },
  "predictions": {
    "time": [0, 315360, 630720, ..., 63072000],
    "conversion_mean": [0.0, 0.01, 0.02, ..., 0.45],
    "conversion_lower": [0.0, 0.009, 0.018, ..., 0.42],
    "conversion_upper": [0.0, 0.011, 0.022, ..., 0.48],
    "temperature": 25.0,
    "temperature_units": "C",
    "temperature_K": 298.15,
    "time_unit": "seconds",
    "requested_time": {"value": 2, "unit": "year"}
  },
  "simulation": null,
  "bootstrap_results": null,
  "report_path": null,
  "summary": {
    "datasets_count": 1,
    "total_datapoints": 4,
    "temperature_range": "40.00 - 40.00 °C",
    "models_tried": 3,
    "models_successful": 3,
    "top_n_selected": 3,
    "bootstrap_iterations": 50
  }
}
```

---

## Flask API Example

```python
from flask import Flask, request, jsonify
from akts import auto_model_isothermal_data
import json

app = Flask(__name__)

@app.route('/api/analyze', methods=['POST'])
def analyze_kinetics():
    """
    Kinetic analysis API endpoint.

    POST /api/analyze
    Body: {
        "datasets": [...],
        "predict": {"value": 1, "unit": "year", "temperature": 25},
        "simulate": [[0, 25], [5, 40], [10, 25]],
        "simulate_time_unit": "days",
        "temperature_units": {"input": "C", "output": "C"},
        "models": ["F1", "F2", "A2"],  // optional model names
        "bootstrap": 50  // optional
    }

    Returns: JSON with analysis results
    """
    try:
        data = request.json

        # Extract parameters
        datasets = data['datasets']
        predict_info = data.get('predict')
        simulate = data.get('simulate')
        simulate_time_unit = data.get('simulate_time_unit', 'days')
        temp_units = data.get('temperature_units', {})
        input_temp_units = temp_units.get('input', 'K')
        output_temp_units = temp_units.get('output', 'K')
        models = data.get('models')
        bootstrap = data.get('bootstrap', 50)

        # Build predict tuple
        predict = None
        if predict_info:
            predict = (
                predict_info['value'],
                predict_info['unit'],
                predict_info.get('temperature')
            ) if 'temperature' in predict_info else (
                predict_info['value'],
                predict_info['unit']
            )

        # Convert simulate to list of tuples
        simulate_tuples = [tuple(p) for p in simulate] if simulate else None

        # Run analysis
        results_json = auto_model_isothermal_data(
            data_files=datasets,
            predict=predict,
            simulate=simulate_tuples,
            simulate_time_unit=simulate_time_unit,
            input_temperature_units=input_temp_units,
            output_temperature_units=output_temp_units,
            models_to_try=models,
            bootstrap_iterations=bootstrap,
            output_format='json',
            report_path=None  # No HTML for API
        )

        # Return JSON response
        return results_json, 200, {'Content-Type': 'application/json'}

    except KeyError as e:
        return jsonify({'error': f'Missing required field: {e}'}), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
```

---

## FastAPI Example

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Tuple
from akts import auto_model_isothermal_data
import json

app = FastAPI(title="AKTS Kinetic Analysis API")

class Dataset(BaseModel):
    time: List[float]
    temperature: List[float]
    conversion: List[float]
    metadata: Optional[dict] = None

class PredictInfo(BaseModel):
    value: float
    unit: str
    temperature: Optional[float] = None

class TemperatureUnits(BaseModel):
    input: str = "K"
    output: str = "K"

class AnalysisRequest(BaseModel):
    datasets: List[Dataset]
    predict: Optional[PredictInfo] = None
    simulate: Optional[List[Tuple[float, float]]] = None
    simulate_time_unit: str = "days"
    temperature_units: Optional[TemperatureUnits] = None
    models: Optional[List[str]] = None
    bootstrap: int = 50

@app.post("/api/analyze")
async def analyze_kinetics(request: AnalysisRequest):
    """
    Perform kinetic analysis on isothermal data.

    Returns analysis results including model fit, predictions, and statistics.
    """
    try:
        # Convert datasets to JSON format
        datasets_json = [d.dict() for d in request.datasets]

        # Build predict tuple
        predict = None
        if request.predict:
            if request.predict.temperature is not None:
                predict = (
                    request.predict.value,
                    request.predict.unit,
                    request.predict.temperature
                )
            else:
                predict = (
                    request.predict.value,
                    request.predict.unit
                )

        # Get temperature units
        temp_units = request.temperature_units or TemperatureUnits()

        # Run analysis
        results_json = auto_model_isothermal_data(
            data_files=datasets_json,
            predict=predict,
            simulate=request.simulate,
            simulate_time_unit=request.simulate_time_unit,
            input_temperature_units=temp_units.input,
            output_temperature_units=temp_units.output,
            models_to_try=request.models,
            bootstrap_iterations=request.bootstrap,
            output_format='json',
            report_path=None
        )

        # Parse and return
        return json.loads(results_json)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

## Error Handling

### Common Errors

**Invalid JSON structure**:
```json
{
  "error": "Missing required field: 'time'",
  "status": 400
}
```

**Invalid temperature units**:
```json
{
  "error": "Unknown temperature unit: X. Supported: 'K', 'C', 'F'",
  "status": 400
}
```

**No successful model fits**:
```json
{
  "error": "No models were successfully fitted. Check your data and try again.",
  "status": 500
}
```

### Validation Checklist

Before sending data to the API, ensure:

- **Array lengths**: `time`, `temperature`, and `conversion` must have the same length.
- **Conversion range**: Values must be between 0.0 and 1.0.
- **Time order**: Values should increase monotonically.
- **Temperature units**: Use `K`, `C`, or `F`.
- **Data points**: Each dataset must contain at least three points.
- **Temperature coverage**: Use at least two datasets at different temperatures to estimate Ea reliably.

---

## Performance Considerations

### Response Time

| Configuration | Typical Time | Notes |
|--------------|-------------|-------|
| **Fast** (3 models, no bootstrap) | 1-3 seconds | Good for real-time APIs |
| **Standard** (12 default models, 50 bootstrap) | 30-60 seconds | Balanced |
| **Comprehensive** (12 default + 2 ODE models, 100 bootstrap) | 5-10 minutes | Background processing |

`models.all` (about 173 models, 136 of them from `models.kinetic.SB2_grid`) is much
slower: each SB2 grid model is ODE-integrated, and the grid alone typically adds
10-15 minutes. Avoid it for synchronous API calls.

### Optimization Tips

**For Fast APIs** (<5 seconds):
```python
results = auto_model_isothermal_data(
    data_files=json_data,
    models_to_try=['F1', 'F2'],  # Only 2 models
    bootstrap_iterations=0,  # Skip bootstrap
    report_path=None,
    output_format='json'
)
```

**For Background Jobs**:
```python
results = auto_model_isothermal_data(
    data_files=json_data,
  models_to_try=['F1', 'F2', 'A2', 'A->B->C', 'A+B->C'],
    bootstrap_iterations=100,  # Full uncertainty
    report_path='results/report.html',  # Save report
    output_format='both'  # Both dict and JSON
)
```

### Caching Strategy

Cache results based on:
- Dataset hash (time + temperature + conversion)
- Model selection
- Bootstrap iterations

Example:
```python
import hashlib
import json

def cache_key(data, models, bootstrap):
    """Generate cache key for results."""
    data_str = json.dumps(data, sort_keys=True)
    key_str = f"{data_str}_{models}_{bootstrap}"
    return hashlib.sha256(key_str.encode()).hexdigest()

# Use with Redis/Memcached
cache_key = cache_key(datasets, models_to_try, bootstrap_iterations)
cached_result = redis.get(cache_key)
if cached_result:
    return json.loads(cached_result)
```

---

## Summary

The JSON I/O specification enables seamless integration of AKTS into web applications and APIs:

- JSON datasets can be passed directly.
- Output is available as a dictionary, JSON string, or both.
- Result parameters are available in the serialized output.
- NumPy values are converted to native Python types.
- Errors are returned with diagnostic information.

See [Automated Analysis Guide](automated_analysis.md) for more examples and [API Reference](api_reference.md) for complete parameter details.
