# Automated Analysis

## `auto_model_isothermal_data()`

Automated kinetic analysis for isothermal data.

### What It Does

`auto_model_isothermal_data()` provides complete automated kinetic analysis:

1. **Loads data** from files, objects, or JSON (all replicate points kept by default)
2. **Fits multiple models** automatically (F0, F1, F2, F3, A2, A3, R2, R3, D2, D3, SB_mn, Bna by default; A→B→C, A+B→C, SB2 and others on request)
3. **Filters models** by R² ≥ 0.70 and physical plausibility
4. **Ranks models** by Akaike weights (probability each is best)
5. **Calculates confidence intervals** via bootstrap resampling
6. **Generates predictions** at any temperature and time
7. **Simulates temperature excursions** (shipping, storage cycles)
8. **Creates professional HTML reports** with interactive plots

### Basic Usage

```python
from akts import auto_model_isothermal_data

# Simplest usage
results = auto_model_isothermal_data(
    data_files=['data_25C.csv', 'data_40C.csv', 'data_60C.csv'],
    predict=(2, 'year'),           # Predict 2 years ahead
    report_path='analysis.html'
)

print(f"Best model: {results['selected_model']['model_name']}")
print(f"Activation energy: {results['selected_model']['parameters']['Ea']/1000:.1f} kJ/mol")
```

### Key Parameters

#### Data Input
```python
data_files : List[Union[str, Path, KineticDataset, Dict]]
average_replicates : bool = False  # Keep every replicate point (recommended)
```
- **File paths**: `['data1.csv', 'data2.csv']`
- **KineticDataset objects**: Pre-loaded data
- **JSON dicts**: `[{'time': [...], 'temperature': [...], 'conversion': [...]}]`

**Replicates**: By default every replicate point is kept for fitting and bootstrap. With equal replicate counts per time point, least squares on replicates gives the same fit as on means, and the replicates preserve the scatter that the bootstrap resamples and the prediction interval uses. `average_replicates=True` averages measurements at identical (time, temperature) coordinates and uses their standard deviation for inverse-variance weighting. Fitting on averages while bootstrapping on replicates is not recommended because the two are inconsistent. See [Data Quality Weighting](data_quality_weighting.md) for details.

**Conversion scaling**: When several files are loaded, readouts are put on one shared conversion scale, `conversion = (readout - readout_initial) / (readout_final - readout_initial)`. By default the start is the mean first readout across files and the end is the largest observed readout (smallest for `readout_type='decreasing'`). Pass `readout_initial` / `readout_final` as loader keywords to override. For %HMW data, `readout_final=100.0` gives the commercial AKTS scaling (HMW - HMW0)/(100 - HMW0).

#### Predictions
```python
predict : Tuple[float, str] or Tuple[float, str, float]
```
- **Simple**: `(3, 'year')` - uses first dataset temperature
- **With temperature**: `(3, 'year', 298.15)` - specify 298.15 K (25°C)
- **Separate parameter**: `predict_temperature_K=298.15`

**Supported time units**: `'seconds'`, `'minutes'`, `'hours'`, `'days'`, `'weeks'`, `'months'`, `'years'`

#### Temperature Excursion Simulation
```python
simulate : List[Tuple[float, float]]
simulate_time_unit : str
```

Simulate degradation under **variable temperature** conditions:

```python
# Shipping scenario: 25°C → 40°C for 2 days → back to 25°C
shipping_profile = [
    (0, 298),      # Day 0: 25°C
    (5, 298),      # Day 5: Still 25°C
    (7, 313),      # Day 7: 40°C (2-day shipping)
    (10, 298),     # Day 10: Back to 25°C
    (30, 298),     # Day 30: End
]

results = auto_model_isothermal_data(
    data_files=['stability_data.csv'],
    simulate=shipping_profile,
    simulate_time_unit='days'
)
```

**Applications:**
- Shipping temperature excursions
- Daily/seasonal temperature cycles
- Equipment failure scenarios
- Accelerated aging protocols

#### Model Selection and Ranking
```python
models_to_try : List[str] = None  # Default: all common models
top_n : int = 3                    # Top N models to analyze
min_r_squared : float = 0.70       # Minimum R² threshold (filter)
apply_filters : bool = True         # Apply R² and plausibility filters
```

**Model ranking**: Uses **Akaike weights** only (probability each model is best). There is no separate simplicity penalty; AIC's 2k term already penalizes complexity. See [Model Selection](model_selection.md).

**Filters** (when `apply_filters=True`):
- R² ≥ `min_r_squared` and physically plausible (A < 10²⁰ s⁻¹, 5 < Ea < 1000 kJ/mol): only these models are ranked
- If no model passes both, models passing R² are kept and a `filter_warning` (`'no_plausible_models'`) is attached to rank 1
- If no model reaches `min_r_squared`, all models are kept with a `'no_good_models'` warning

Every `Ea` parameter is fitted within `akts.utils.EA_BOUNDS` = 5-1000 kJ/mol. A warning is issued when a fitted `Ea` sits at a bound.

**Default models**: `models.default` (the 12 kinetic models in `models.kinetic.all`; see the [Model Selector Guide](model_selector.md)).

**ODE models** (optional, slower): `models.ode.all`
- Add them to `models_to_try`, for example `models.default + models.ode.all`.
- Fits use multistart local optimization, retaining the result with the highest R².

**Two-step Sestak-Berggren** (optional, slow): `models.kinetic.SB2` and `models.kinetic.SB2_grid`
- dα/dt = k1(T)·α^m1·(1-α)^n1 + k2(T)·α^m2·(1-α)^n2: two parallel SB steps on one conversion (the AKTS commercial two-step form), ODE-integrated.
- `SB2` fits Ea1, A1, Ea2, A2, m1, n1, m2, n2 (m in 0-3, n in 0-8). `SB2_grid` is 136 models named `SB2_m{m1}n{n1}_m{m2}n{n2}`, every unordered pair of integer steps with m, n in {0, 1, 2, 3}; each fits only Ea1, A1, Ea2, A2.
- Both are included in `models.all` (about 173 models), which makes `models.all` runs much slower; the SB2 grid adds roughly 10-15 minutes.
- SB2 pre-exponential factors typically exceed the 10²⁰ s⁻¹ plausibility limit, so a "No physically plausible models" warning is expected when SB2 is best. The model is still selected.

**Friedman model-free analysis**: `'Friedman'` (opt-in)
- Friedman is never added automatically. Include it in `models_to_try` when you want to run it. A meaningful fit requires **at least three distinct mean temperatures**, rounded to the nearest kelvin, with overlapping conversion ranges.
- Fit as a real `FitResult` (`model_name='Friedman'`) so it competes in ranking/selection like any other model, and gets bootstrap confidence intervals like any other model — see [Model-free (isoconversional) prediction](api_reference.md#model-free-prediction-no-reaction-model-assumed).
- The `alpha_levels` used internally adapt to how much conversion the data actually reaches (useful for accelerated-aging studies that only reach a few % conversion within practical timeframes) rather than assuming every dataset gets to ~95% conversion.
- If explicitly selected with fewer than three distinct temperatures, it may not produce a valid fit.

**Custom selection**:
```python
models_to_try=['F1', 'F2', 'A2']  # Only these three; Friedman is not added
models_to_try=['F1', 'Friedman']  # Explicitly include Friedman
```

#### Bimolecular Reactant Ratio
```python
initial_ratio_r : float = 1.0
```

Fixed `[B]₀/[A]₀` ratio used by the `A+B→C` bimolecular ODE model when it is included in `models_to_try`. At the default `r=1.0` (stoichiometric) with the default reaction orders `m=n=1`, `A+B→C` is mathematically identical to `F2`, so this only matters if the true starting ratio of the two reactants is known and is not 1:1.

#### Bootstrap Confidence Intervals
```python
bootstrap_iterations : int = 100   # Confidence interval samples
confidence_level : float = 0.95    # 95% confidence intervals
ci_type : str = 'two-sided'        # 'two-sided' or 'one-sided'
bootstrap_method : str = 'monte_carlo'  # 'monte_carlo', 'parametric', 'residual'
n_jobs : int = -1                  # Bootstrap workers; default leaves one CPU core free
```

`ci_type='two-sided'` (default) gives equal-tailed bootstrap percentile bands
([2.5th, 97.5th] at 95%) for the fit, prediction, and simulation plots and for
`conversion_lower`/`conversion_upper`. `ci_type='one-sided'` gives (MLE curve, 95th
percentile). The ICH Q1E bootstrap shelf-life always uses a separately computed
one-sided 95th-percentile bound, whichever `ci_type` is set.

Bootstrap refits whose RSS exceeds 10x the median replicate RSS are excluded as
non-converged (a warning gives the count). Replicate predictions whose final
conversion is below min(1%, 10% of the model's own prediction) are excluded as
degenerate, so low-conversion predictions keep their CI.

`monte_carlo` is the default case bootstrap: complete observations are sampled
with replacement within each dataset, preserving its row count. `parametric`
generates Gaussian errors around fitted curves using residual-based standard
deviations; `residual` resamples centered residuals with conversion-transition
weights. Case resampling can omit measured time points and repeat others, so it
may be unreliable for very sparse datasets.

Select the method in the analysis call:

```python
from akts import auto_model_isothermal_data, models

results = auto_model_isothermal_data(
    data_files=data_files,
    models_to_try=models.empirical.all,
    bootstrap_iterations=200,
    confidence_level=0.95,
    bootstrap_method='residual',
    random_state=12345,
)

print(results['summary']['bootstrap_method'])
```

The selected method is also recorded on each returned `BootstrapResult` as
`bootstrap_method`. For direct use with a fitted empirical or Friedman model,
see the [`run_bootstrap_empirical()` and `run_bootstrap_friedman()` API examples](api_reference.md#run_bootstrap).

The worker count is capped at the number of bootstrap iterations. Set `n_jobs=1`
to run bootstrap fits in a single worker process. These methods apply to kinetic
model uncertainty; shelf-life trend analysis at the requested storage condition
does not use bootstrap resampling. It requires observations within 0.5 K of that
temperature and selects a linear or quadratic trend per batch using a nested F-test.
When all batch trends are linear, ANCOVA tests slope poolability at α=0.25.
Increasing degradation conversion uses its one-sided upper confidence limit for
the conservative shelf-life time. The result is not itself a compliance determination.

#### Shelf-Life Regression Settings
```python
shelf_life_temperature_C : float = 20.0
shelf_life_target_conversion : float = 0.05
shelf_life_confidence_level : float = 0.95
shelf_life_is_long_term : bool = True
shelf_life_nonlinearity_p_threshold : float = 0.05
```

Shelf life is evaluated independently of `predict_temperature_K`. For each usable
batch at the requested temperature, a nested F-test compares linear and quadratic
trends; quadratic is selected when the added curvature has `p` below the configured
threshold. If all batches select linear trends, the ANCOVA common-slope check is
applied. The report records the selected trend, method, curvature p-value,
storage temperature, specification conversion, confidence level, and extrapolation
category. The 20°C default requires data within 0.5 K of 20°C; otherwise the shelf-life
estimate is unavailable unless another temperature is configured.

This regression shelf-life does not depend on the bootstrap or on `ci_type`. Its
plot shades a two-sided 90% band (at the default `shelf_life_confidence_level=0.95`),
whose edges are one-sided 95% limits; the legend
reads "One-sided 95% confidence limits".

This is an automated regression choice, not a determination of regulatory compliance.
Review the observed data, residuals, specification limit, storage condition, and model
assumptions before using a shelf-life estimate in a regulatory submission.

#### Output Format
```python
output_format : str = 'dict'  # Options: 'dict', 'json', 'both'
```

**Three output modes:**

1. **Dictionary** (default):
```python
results = auto_model_isothermal_data(..., output_format='dict')
# Returns: Dict
print(results['selected_model'])
```

2. **JSON string** (for APIs):
```python
results_json = auto_model_isothermal_data(..., output_format='json')
# Returns: str (JSON)
return jsonify(results_json)  # Send to web client
```

3. **Both**:
```python
results_dict, results_json = auto_model_isothermal_data(..., output_format='both')
# Returns: Tuple[Dict, str]
# Use dict locally, send JSON to client
```

#### HTML Report
```python
report_path : str = None          # Path to HTML file
report_format : str = 'interactive'  # 'interactive', 'static', or 'both'
auto_open : bool = False           # Open in browser automatically
```

**Report includes:**
- Executive summary cards
- Model comparison table (all fitted models with statistics)
- Fit plots: one panel per temperature with its own y-scale, time in days, residuals below. Each panel shows the data, the fitted curve, a darker 95% confidence interval (CI) and a lighter 95% prediction interval (PI)
- Reported Ea with a note table of typical activation energy ranges (thermal denaturation/unfolding 400-800 kJ/mol; enzymatic/proteolytic cleavage 20-100 kJ/mol; spontaneous/pyrolytic hydrolysis 90-140 kJ/mol)
- Prediction plot with confidence intervals
- **Temperature excursion simulation plot** (if `simulate` provided)
- Statistical details

**Reading the bands**: The CI covers uncertainty in the fitted curve only, so data points scattering outside it are expected. The PI adds each temperature's own residual scatter: half-width = t·sqrt(se_fit² + s_i²), with s_i² that dataset's residual variance and se_fit from the bootstrap CI half-width. About 95% of data points should fall inside the PI. Separate panels keep narrow low-temperature bands visible.

### Complete Example

```python
from akts import auto_model_isothermal_data

def progress(msg, data):
    """Optional progress callback"""
    print(f"[{data['timestamp']}] {msg}")

# Define shipping temperature profile
shipping = [
    (0, 298),      # 25°C storage
    (5, 313),      # Heat to 40°C
    (7, 298),      # Back to 25°C
    (30, 298),     # End
]

results = auto_model_isothermal_data(
    # Data sources (mix file paths and JSON!)
    data_files=[
        'stability_25C.csv',
        'stability_40C.csv',
        {'time': [0, 86400, 172800], 'temperature': [333, 333, 333], 'conversion': [0.0, 0.15, 0.28]}
    ],

    # Predictions
    predict=(2, 'year', 298),        # 2 years at 25°C
    simulate=shipping,                # Shipping excursion
    simulate_time_unit='days',

    # Model selection
    models_to_try=None,               # Try all default models
    top_n=3,                          # Top 3 models

    # Statistics
    bootstrap_iterations=100,         # 100 bootstrap samples
    confidence_level=0.95,            # 95% confidence

    # Output
    output_format='both',             # Dict + JSON
    report_path='output/analysis.html',
    report_format='interactive',      # Interactive Plotly plots
    auto_open=False,

    # Progress tracking
    progress_callback=progress,

    # Data loader options (passed to load_data_file)
    time_col='Time (days)',
    temperature_col='Temperature (K)',
    readout_col='Conversion',
    auto_detect=True                  # Fuzzy column name matching
)

# Unpack results (since output_format='both')
results_dict, results_json = results

# Access Python dict
print(f"Best model: {results_dict['selected_model']['model_name']}")
print(f"Ea: {results_dict['selected_model']['parameters']['Ea']/1000:.1f} kJ/mol")
print(f"2-year prediction: {results_dict['predictions']['conversion_mean'][-1]:.1%}")

if results_dict['simulation']:
    print(f"Shipping degradation: {results_dict['simulation']['conversion_mean'][-1]:.1%}")

# Send JSON to API client
# return jsonify(results_json)  # In Flask/FastAPI
```

### Results Structure

```python
results = {
    'top_models': [
        {
            'rank': 1,
            'model_name': 'F2_model',
            'parameters': {'Ea': 95000.0, 'A': 1.2e12},
            'stats': {
                'r_squared': 0.998, 'aic': -150.2, 'bic': -145.1, 'rss': 0.0012,
                'r_squared_adj': 0.997,   # R² penalized for parameter count
                'rmse': 0.0089,           # sqrt(rss / n_points)
                'akaike_weight': 0.62     # probability this model is best, given AICc, among the fitted candidates
            },
            'score': -0.62,              # -akaike_weight; lower ranks first
            'simplicity_penalty': 0.0    # always 0.0 (kept for compatibility)
        },
        # ... more models
    ],

    'selected_model': {
        'model_name': 'F2_model',
        'rank': 1,
        # 'reason' reports the Akaike-weight evidence -- see "Model Selection Logic" below.
        'reason': 'Best of 2 competitive models (Akaike weight = 62.0%)',
        'parameters': {'Ea': 95000.0, 'A': 1.2e12},
        'statistics': {'r_squared': 0.998, ...},
        'physical_sanity_flags': []  # e.g. ["Ea=25.0 kJ/mol is below the plausible 30-180 kJ/mol
                                      #        range for drug degradation -- may indicate diffusion
                                      #        control or model misfit"] if Ea looks implausible
    },

    'predictions': {
        'time': [0, 1000, 2000, ...],              # seconds
        'conversion_mean': [0.0, 0.05, 0.12, ...],
        'conversion_lower': [0.0, 0.04, 0.10, ...], # 95% CI (ci_type)
        'conversion_upper': [0.0, 0.06, 0.14, ...], # 95% CI (ci_type)
        'temperature_K': 298.15,
        'time_unit': 'seconds',
        'requested_time': {'value': 2, 'unit': 'year'}
    },

    'simulation': {  # If simulate parameter provided
        'time': [0, 432, 864, ...],
        'conversion_mean': [0.0, 0.001, 0.003, ...],
        'temperature': [298, 299, 301, ...],       # K at each time
        'conversion_lower': [...],                  # 95% CI
        'conversion_upper': [...],                  # 95% CI
        'time_unit': 'days',
        'input_profile': [(0, 298), (5, 313), ...] # Original input
    },

    'bootstrap_results': {
        # Bootstrap distribution data (if bootstrap_iterations > 0)
    },

    'report_path': 'output/analysis.html',

    'summary': {
        'datasets_count': 4,
        'total_datapoints': 108,
        'temperature_range': '24.85 - 59.85 °C',
        'models_tried': 10,
        'models_successful': 10,
        'bootstrap_iterations': 100,
        'top_n_selected': 3
    }
}
```

### Model Selection Logic

`auto_model_isothermal_data()` selects the rank-1 model, i.e. the one with the highest Akaike weight after filtering. There is no additional tie-break toward simpler models; AIC already penalizes each extra parameter. `selected_model['reason']` (also returned as `selection_reason`) states the strength of evidence: `"Overwhelming evidence (Akaike weight = ...)"` for a weight ≥ 0.90, `"Strong evidence"` for ≥ 0.70, `"Substantial support"` for ≥ 0.50, and otherwise `"Best of N competitive models"`, where N counts models among the top five with weight > 0.10. A low weight means several models describe the data about equally well; compare their predictions before relying on one.

`selected_model['physical_sanity_flags']` also lists any warnings (not exclusions) when a fitted activation energy falls outside the ~30-180 kJ/mol range considered plausible for drug degradation kinetics — a flagged fit isn't necessarily wrong, but is worth a second look.

[Model Selection Guide](model_selection_guide.md) explains this selection logic.

### Model-Based Time to Target Conversion

To answer "how long until X% degradation at a given storage temperature" directly (rather than reading it off a `predict` curve), use `time_to_conversion()`. It is the inverse of `predict_conversion()` and can propagate bootstrap uncertainty for a fitted kinetic model. This is a model-based extrapolation utility, not the ICH Q1E shelf-life calculation.

`time_to_conversion()` takes a `FitResult` directly (the kind returned by `fit_kinetic_model()`), so it's typically used alongside the manual/advanced fitting workflow:

```python
from akts import fit_kinetic_model, run_bootstrap, time_to_conversion

fit_result = fit_kinetic_model(
    datasets=datasets,
    model_name='single_step',
    model_definition_args={'f_alpha_model': 'F1'},
    initial_guesses={'Ea': 85000, 'A': 1e11}
)

bootstrap_result = run_bootstrap(datasets=datasets, fit_result=fit_result, n_iterations=100)

result = time_to_conversion(
    fit_result=fit_result,
    target_conversion=0.05,       # 5% degradation
    temperature_K=298.15,         # storage temperature
    bootstrap_result=bootstrap_result  # optional: adds a confidence interval
)

print(f"Time to 5% degradation: {result['time_sec'] / 86400:.1f} days")
if result['time_lower_sec'] is not None:
    print(f"95% CI: [{result['time_lower_sec']/86400:.1f}, {result['time_upper_sec']/86400:.1f}] days")
```

`result['time_sec']` is `None` if the target conversion isn't reached within the (automatically extended) search window — this can happen for a very stable formulation at a low storage temperature.

The shelf-life result generated by `auto_model_isothermal_data()` is separate from
`time_to_conversion()`: it auto-selects a linear or quadratic time trend at the
configured shelf-life temperature and applies an adverse-direction one-sided
confidence limit, without bootstrap resampling.

## JSON Input/Output for Web APIs

### Input: JSON from Web Client

```python
# Client sends JSON
json_data = {
    "time": [0, 3600, 7200, 10800],      # seconds
    "temperature": [313, 313, 313, 313],  # K
    "conversion": [0.0, 0.05, 0.12, 0.20]
}

# Server processes
results = auto_model_isothermal_data(
    data_files=[json_data],         # Pass JSON dict directly
    output_format='json',           # Get JSON string back
    report_path=None                # Skip HTML for API
)

# Send back to client
return jsonify(results)  # Flask
# or
return JSONResponse(content=json.loads(results))  # FastAPI
```

### Output: JSON to Web Client

```python
# Get JSON output
results_json = auto_model_isothermal_data(
    data_files=['data.csv'],
    output_format='json'  # Returns JSON string
)

# All numpy types converted to native Python
# {
#   "top_models": [...],
#   "selected_model": {...},
#   "predictions": {...},
#   ...
# }

# Ready to send over HTTP
response = requests.post('https://api.example.com/results',
                         data=results_json,
                         headers={'Content-Type': 'application/json'})
```

### Full Flask API Example

```python
from flask import Flask, request, jsonify
from akts import auto_model_isothermal_data
import json

app = Flask(__name__)

@app.route('/api/analyze', methods=['POST'])
def analyze_kinetics():
    """
    Analyze kinetic data from JSON.

    POST /api/analyze
    Body: {
        "datasets": [
            {"time": [...], "temperature": [...], "conversion": [...]},
            ...
        ],
        "predict": [1, "year", 298],
        "simulate": [[0, 298], [5, 313], [10, 298]],
        "simulate_time_unit": "days"
    }
    """
    data = request.json

    # Run analysis
    results_json = auto_model_isothermal_data(
        data_files=data['datasets'],
        predict=tuple(data.get('predict', [])) if data.get('predict') else None,
        simulate=[tuple(p) for p in data.get('simulate', [])] if data.get('simulate') else None,
        simulate_time_unit=data.get('simulate_time_unit', 'days'),
        models_to_try=data.get('models', None),
        bootstrap_iterations=data.get('bootstrap', 50),
        output_format='json',  # JSON output
        report_path=None       # No HTML for API
    )

    # Return JSON response
    return jsonify(json.loads(results_json))

if __name__ == '__main__':
    app.run(debug=True)
```

## Key Advantages

### For Routine Analysis
- **One function** does everything
- **Sensible defaults** for all parameters
- **Clear, actionable output** (HTML reports)
- **Progress tracking** to show what's happening

### For Web Integration
- **JSON in/out** for APIs
- **No file I/O required** (use dicts)
- **Stateless** operation
- **All numpy types** converted automatically

### For Scientific Workflows
- **Multiple models** tried automatically
- **Statistical ranking** (Akaike weights after R² and plausibility filters)
- **Bootstrap confidence intervals** included
- **Temperature excursion simulation** for real-world scenarios
- **Publication-quality reports** (interactive or static)

### General Guidance
- **Flexible input**: Files, objects, or JSON
- **Flexible output**: Dict, JSON, or both
- **Temperature control**: Specify prediction temperature
- **Simulation**: Model real storage/shipping conditions
- **Professional reports**: Interactive HTML with embedded plots

## Backward Compatibility

All existing AKTS code continues to work! The auto-helper is an **addition**, not a replacement:

```python
# Old way (still works!)
from akts import load_data_file, fit_kinetic_model, run_bootstrap
dataset = load_data_file('data.csv')
fit_result = fit_kinetic_model(...)
bootstrap_result = run_bootstrap(...)

# New way (easier!)
from akts import auto_model_isothermal_data
results = auto_model_isothermal_data(['data.csv'])
```
