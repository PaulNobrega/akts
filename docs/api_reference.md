# API Reference

Quick reference for akts functions. For detailed examples, see [Getting Started](getting_started.md) and [Examples](examples.md).

## One-Function Workflow

### auto_model_isothermal_data()

Complete automated analysis for isothermal kinetic data.

[Automated analysis tutorial](automated_analysis.md)

```python
from akts import auto_model_isothermal_data, models

results = auto_model_isothermal_data(
    data_files,                    # List of files, KineticDataset, or JSON dicts
    predict=(2, 'year'),           # Optional prediction
    models_to_try=models.default,  # Or models.default + models.ode.all
    initial_ratio_r=1.0,           # Fixed [B]0/[A]0 ratio for A+B->C (only matters if != 1.0)
    filter_implausible=False,      # Exclude models with unrealistic parameters
    input_temperature_units='K',   # 'K', 'C', or 'F'
    output_temperature_units='K',
    shelf_life_temperature_C=20.0,
    shelf_life_target_conversion=0.05,
    shelf_life_confidence_level=0.95,
    shelf_life_is_long_term=True,
    shelf_life_nonlinearity_p_threshold=0.05,
    report_path='report.html',
    output_format='dict',          # 'dict', 'json', or 'both'
    bootstrap_iterations=100,
    bootstrap_method='monte_carlo',
    n_jobs=-1,                     # Bootstrap workers: all but one CPU core
    progress_callback=None
)
```

`models_to_try=None` (the default) uses `models.default` (12 kinetic models).
Friedman is not added automatically; include `'Friedman'` in `models_to_try` to
select it. A meaningful Friedman fit requires at least three distinct
temperatures. To include ODE or empirical models, pass a selector list such as
`models.default + models.ode.all`. See
[Model-free prediction](#model-free-prediction-no-reaction-model-assumed) below
and [automated_analysis.md](automated_analysis.md#model-selection) for details.

When `predict` is provided, the report also attempts a shelf-life estimate at
`shelf_life_temperature_C` (default 20°C), independently of the prediction
temperature. It uses `shelf_life_target_conversion` (default 5% degradation),
the one-sided `shelf_life_confidence_level` (default 95%), and the selected
long-term/accelerated ceiling formula. For each batch, a nested F-test compares
linear and quadratic time trends; quadratic is selected when its p-value is below
`shelf_life_nonlinearity_p_threshold` (default 0.05). If all batch trends are
linear, ANCOVA tests common-slope pooling at α=0.25. Usable observations must be
within 0.5 K of the shelf-life temperature; otherwise no estimate is returned.
These automatic trend choices are diagnostics and do not by themselves establish
regulatory compliance.

## Core Functions

### fit_kinetic_model()

Fit a single model to data. Isothermal data with a closed-form model (F0-F3,
A2, A3, R2, R3, D2-D4) is fitted directly with `least_squares`. Everything else is
integrated with `solve_ivp` (LSODA by default, RK45 fallback). See
[Numerical Solvers and Performance](advanced_usage.md#numerical-solvers-and-performance)
for `solver_options`.

```python
from akts import fit_kinetic_model

fit_result = fit_kinetic_model(
    datasets,                      # List[KineticDataset]
    model_name,                    # 'single_step', 'A->B->C', 'A+B->C'
    model_definition_args,         # See Model Parameters below
    initial_guesses,               # Dict[str, float]
    parameter_bounds=None,
    solver_options=None,           # e.g. {'primary_solver': 'BDF', 'fallback_solver': 'Radau'}
    optimizer_options=None,        # e.g. {'method': 'Powell', 'max_seconds': 120}
)
```

**Model Parameters:**

```python
# single_step (fixed shape parameters)
model_definition_args={'f_alpha_model': 'F1'}  # F0, F1, F2, F3, A2, A3, R2, R3, D2, D3, SB_mn, Bna
initial_guesses={'Ea': 85000, 'A': 1e11}

# single_step with fitted shape parameters
model_definition_args={'f_alpha_model': 'Fn'}  # Fn: n is fitted, SB: m,n fitted, SB_mnp: m,n,p fitted
initial_guesses={'Ea': 85000, 'A': 1e11, 'n': 1.0}  # n as fitted parameter
parameter_bounds={'Ea': (50e3, 150e3), 'A': (1e7, 1e15), 'n': (0, 5)}

# SB (fitted Sestak-Berggren)
model_definition_args={'f_alpha_model': 'SB'}
initial_guesses={'Ea': 100000, 'A': 1e12, 'm': 0.5, 'n': 1.0}
parameter_bounds={'Ea': (50e3, 200e3), 'A': (1e8, 1e16), 'm': (0, 3), 'n': (0, 3)}

# SB_mnp (extended Sestak-Berggren with p exponent)
model_definition_args={'f_alpha_model': 'SB_mnp'}
initial_guesses={'Ea': 100000, 'A': 1e12, 'm': 0.5, 'n': 1.0, 'p': 0.0}
parameter_bounds={'m': (0, 3), 'n': (0, 3), 'p': (-2, 2)}

# A->B->C (consecutive)
model_definition_args={'f1_model': 'F1', 'f2_model': 'F1'}
initial_guesses={'Ea1': 85000, 'A1': 1e11, 'Ea2': 95000, 'A2': 1e12}

# A+B->C (bimolecular)
model_definition_args={'bimol_params': {'initial_ratio_r': 1.0, 'm': 1.0, 'n': 1.0}}
initial_guesses={'Ea': 85000, 'A': 1e11}
```

### run_bootstrap()

Generate confidence intervals.

```python
from akts import run_bootstrap

if __name__ == '__main__':  # Required on Windows!
    bootstrap_result = run_bootstrap(
        datasets,
        fit_result,
        optimizer_options={'method': 'L-BFGS-B'},
        parameter_bounds={'Ea': (50000, 150000), 'A': (1e7, 1e14)},
        n_iterations=100,
        bootstrap_method='monte_carlo',
        n_jobs=-1,            # Use all but one CPU core
        random_state=12345,   # Optional: reproducible resampling
    )
```

`bootstrap_method` accepts three strategies and defaults to `monte_carlo`:

- **`monte_carlo`** (case bootstrap): sample complete `(time, temperature,
    conversion)` rows with replacement within each dataset, keeping its row count.
    Some observed times may be omitted and others repeated, so this can be unstable
    for very sparse datasets.
- **`parametric`**: add independent Gaussian conversion errors around the fitted
    curves, using residual standard deviations estimated per dataset (with a pooled
    estimate when a dataset has too few finite residuals), then refit.
- **`residual`**: add centered residuals sampled with replacement from the
    pooled residuals, weighted toward the conversion-transition region, then refit.

Each replicate is refit with the same model and optimizer. `BootstrapResult`
records the chosen method in `bootstrap_method`. The specialized helpers
`run_bootstrap_empirical()` and `run_bootstrap_friedman()` are also exported from
`akts` and accept the same option; Friedman reruns its isoconversional regressions
on each synthetic dataset.

For an empirical fit, call its specialized helper directly:

```python
from akts import run_bootstrap_empirical

bootstrap_result = run_bootstrap_empirical(
    datasets,
    empirical_fit_result,
    n_iterations=200,
    confidence_level=0.95,
    bootstrap_method='residual',
    random_state=12345,
)
```

For Friedman, pass the `IsoResult` stored on the Friedman fit:

```python
from akts import run_bootstrap_friedman

iso_result = friedman_fit_result.model_definition_args['iso_result']
bootstrap_result = run_bootstrap_friedman(
    datasets,
    iso_result,
    n_iterations=200,
    confidence_level=0.95,
    bootstrap_method='monte_carlo',
    random_state=12345,
)
```

The selected method is available as `bootstrap_result.bootstrap_method` and is
included in bootstrap JSON serialization.

The shelf-life estimate in `auto_model_isothermal_data()` is separate from these
bootstrap methods. At `shelf_life_temperature_C`, it selects a linear or quadratic
time trend per batch using a nested F-test (`p < shelf_life_nonlinearity_p_threshold`).
When all batches are linear, slope poolability is tested by ANCOVA at α=0.25;
otherwise batch trends are kept separate. Increasing degradation uses a one-sided
upper confidence limit to find the conservative time to the target conversion.
No bootstrap is used for this estimate, and the automatic trend choice alone does
not establish regulatory compliance.

### predict_conversion()

Extrapolate to arbitrary temperature program.

```python
from akts import predict_conversion
from akts.utils import construct_profile

# Build temperature profile
segments = [{'type': 'isothermal', 'duration': 2*365*24*3600, 'temperature': 278}]
time_s, temp_K = construct_profile(segments)

# Predict with confidence intervals
prediction = predict_conversion(
    kinetic_description=fit_result,
    temperature_program=(time_s, temp_K),
    bootstrap_result=bootstrap_result  # Optional, adds .conversion_ci
)
```

### time_to_conversion()

Inverse of `predict_conversion()`: given a fitted model and a fixed storage
temperature, find the time to reach a target conversion fraction. Optionally
propagates a confidence interval from a `bootstrap_result` for model-based
prediction uncertainty. This generic utility is not the ICH Q1E shelf-life
calculation.

The `one_sided_ci` parameter changes the bootstrap time-interval calculation to
a one-sided limit; it does not make the result an ICH Q1E analysis.

```python
from akts import time_to_conversion

result = time_to_conversion(
    fit_result=fit_result,        # A successful FitResult from fit_kinetic_model()
    target_conversion=0.05,       # Target fraction in (0, 1], e.g. 0.05 = 5% degradation
    temperature_K=298.15,         # Fixed storage temperature
    bootstrap_result=bootstrap_result,  # Optional, adds time_lower_sec/time_upper_sec
    max_search_time_sec=None,     # Optional override of the search window
    n_eval_points=500,            # Points simulated across the search window
    one_sided_ci=False            # Optional one-sided bootstrap time interval
)

print(f"Time to 5% degradation: {result['time_sec'] / 86400:.1f} days")
# result also has 'time_lower_sec' and 'time_upper_sec' (None if no bootstrap_result
# was given, or if a bound was never reached within the search window)

# For a one-sided model-based bootstrap time interval:
result_regulatory = time_to_conversion(
    fit_result=fit_result,
    target_conversion=0.05,
    temperature_K=298.15,
    bootstrap_result=bootstrap_result,
    one_sided_ci=True  # Use one-sided 95% lower bound on bootstrap crossing times
)
# With one_sided_ci=True, time_upper_sec is None
```

`auto_model_isothermal_data()` computes its shelf-life trend result separately:
it selects linear or quadratic time trends with a nested F-test and uses ANCOVA
for common-slope pooling when all batch trends are linear. It does not use
`time_to_conversion()` or bootstrap resampling for this result; automated trend
selection is not itself a determination of regulatory compliance.

Returns a dict with `time_sec`, `time_lower_sec`, `time_upper_sec` (all `Optional[float]`,
`None` if the target conversion is never reached within the search window). When `one_sided_ci=True`,
`time_upper_sec` is always `None`.

### calculate_ich_q1e_ceiling()

Calculate the maximum allowable extrapolation for stability shelf-life claims according to ICH Q1E guidelines.

ICH Q1E analysis is included in HTML reports generated by `auto_model_isothermal_data()`. Reports include:
- Regulatory-compliant shelf-life plots with confidence intervals
- Visual display of study duration, ICH ceiling, and predicted shelf-life
- Compliance status and recommendations

```python
from akts.helpers import calculate_ich_q1e_ceiling

# Long-term stability study: 12 months of data
ceiling = calculate_ich_q1e_ceiling(12, is_long_term=True)
print(f"Maximum shelf-life claim: {ceiling} months")  # 24 months

# 18-month study
ceiling = calculate_ich_q1e_ceiling(18, is_long_term=True)
print(f"Maximum shelf-life claim: {ceiling} months")  # 30 months (min(36, 30))

# Accelerated stability study: 6 months of data
ceiling = calculate_ich_q1e_ceiling(6, is_long_term=False)
print(f"Maximum shelf-life claim: {ceiling} months")  # 9 months (1.5×6)
```

**ICH Q1E Extrapolation Rules:**
- **Long-term data**: Up to min(2 × study duration, study duration + 12 months)
- **Accelerated data**: Up to 1.5 × study duration

This ensures regulatory shelf-life claims are adequately supported by stability data.

## Data Classes

```python
from akts import KineticDataset

# Create dataset manually
dataset = KineticDataset(
    time=time_min,        # np.ndarray (minutes)
    temperature=temp_K,    # np.ndarray (Kelvin)
    conversion=alpha,      # np.ndarray [0, 1]
    heating_rate=None      # Optional float (K/min)
)
```

**FitResult attributes:**
- `.success` - bool
- `.parameters` - Dict[str, float]
- `.r_squared`, `.aic`, `.bic` - float
- `.durbin_watson` - float (autocorrelation statistic, ≈2 is ideal)
- `.is_physically_plausible` - bool (parameter sanity check)
- `.plausibility_issues` - List[str] (descriptions of parameter issues, if any)
- `.message` - str

**`rank_models()` per-model `stats` dict** (used internally by `auto_model_isothermal_data()`
and returned in its `top_models`/`selected_model` output) includes `rss`, `r_squared`,
`aic`, `bic`, `n_params`, `n_points`, plus the additive fields `r_squared_adj` (adjusted
R²), `rmse`, `akaike_weight` (probability this model is the best in the candidate
set, given AICc), `durbin_watson` (residual autocorrelation, ≈2 is ideal, <1.5 or >2.5
indicates systematic error), `is_physically_plausible` (bool), and `plausibility_issues`
(List[str] or None).

**BootstrapResult attributes:**
- `.parameter_ci` - Dict[str, Tuple[float, float]]
- `.median_parameters` - Dict[str, float]
- `.n_iterations` - int

**PredictionResult attributes:**
- `.time`, `.temperature`, `.conversion` - np.ndarray
- `.conversion_ci` - Optional[Tuple[np.ndarray, np.ndarray]]

## Data Loaders

```python
from akts.loaders import load_isothermal_file, load_dsc_file, load_tga_file

# Isothermal stability data
dataset = load_isothermal_file(
    'data.csv',
    time_col='Time (days)',
    temperature_col='Temperature (K)',
    readout_col='HMW%',
    readout_type='increasing',    # or 'decreasing'
    input_temperature_units='K'
)

# DSC data
dataset = load_dsc_file('dsc.csv', ...)

# TGA data
dataset = load_tga_file('tga.csv', ...)
```

## Isoconversional Methods

```python
from akts import run_kas, run_friedman, run_ofw

kas_result = run_kas(datasets)
friedman_result = run_friedman(datasets)
ofw_result = run_ofw(datasets)

# Returns IsoResult with:
# .alpha, .Ea (J/mol), .Ea_std_err
# .ln_A_f_alpha (Friedman only) -- lets you predict without a model, see below
```

### Model-free prediction (no reaction model assumed)

`run_friedman()` returns `Ea(alpha)` and `ln_A_f_alpha(alpha)`, which support
conversion prediction without selecting an `f(alpha)` model:

```python
from akts import predict_conversion, predict_conversion_model_free, time_to_conversion_model_free

friedman_result = run_friedman(datasets)  # needs multiple temperatures/rates

# General case -- any temperature profile, including excursions
prediction = predict_conversion_model_free(
    friedman_result,
    temperature_program=lambda t: 298.15,  # or (time_sec, temp_K) tuple
    simulation_time_sec=time_array,
)
# predict_conversion(friedman_result, ...) dispatches here automatically.

# Isothermal shortcut -- closed-form quadrature, no ODE solve
seconds_to_5pct = time_to_conversion_model_free(
    friedman_result, target_conversion=0.05, temperature_K=298.15
)
```

Only `run_friedman()` populates `ln_A_f_alpha` -- KAS/OFW are integral methods
whose regression intercepts don't have this interpretation, so predicting from
their `IsoResult` still falls back to a warning.

## Model Discovery and Ranking

```python
from akts import discover_kinetic_models, rank_models

models_to_try = [
    {'name': 'F1', 'type': 'single_step', 'def_args': {'f_alpha_model': 'F1'}},
    {'name': 'F2', 'type': 'single_step', 'def_args': {'f_alpha_model': 'F2'}}
]

ranked_models = discover_kinetic_models(
    datasets=datasets,
    models_to_try=models_to_try,
    initial_guesses_pool={'F1': {...}, 'F2': {...}}
)
```

### Ranking Methods

`rank_models()` supports multiple ranking approaches:

```python
from akts import rank_models

# Combined scoring (default): weighted BIC, R², RSS, n_params
ranked = rank_models(fit_results, ranking_method='combined')

# BIC-only: direct ΔBIC interpretation
# ΔBIC < 2: weak evidence, 2-6: positive, 6-10: strong, >10: very strong
ranked = rank_models(fit_results, ranking_method='bic')

# AIC-only: similar to BIC but different penalty
ranked = rank_models(fit_results, ranking_method='aic')

# Akaike weights: probability each model is best (sums to 1)
ranked = rank_models(fit_results, ranking_method='akaike_weight')

# R² only: maximize explained variance
ranked = rank_models(fit_results, ranking_method='r_squared')
```

**When to use each method:**
- **BIC/AIC**: Most interpretable for model comparison (prefer BIC for n>40)
- **Akaike weight**: Multi-model inference, model averaging
- **Combined**: Balances multiple criteria (default)
- **R²**: Maximize explanatory power (ignores model complexity)

## Utilities

### Temperature Profile Builder

```python
from akts.utils import construct_profile

# Temperature profile builder
segments = [
    {'type': 'isothermal', 'duration': 86400, 'temperature': 298},
    {'type': 'ramp', 'duration': 3600, 'start_temp': 298, 'end_temp': 323}
]
time_s, temp_K = construct_profile(segments, points_per_segment=100)
```

### Tabulated Export

Export prediction and bootstrap results to CSV/DataFrame for analysis in Excel, R, or other tools.

```python
from akts import fit_kinetic_model, predict_conversion, run_bootstrap, export_prediction_report

# Fit model
fit_result = fit_kinetic_model(datasets, 'single_step', {'f_alpha_model': 'F1'}, ...)

# Generate predictions with bootstrap CIs
bootstrap_result = run_bootstrap(fit_result, datasets, n_iterations=100)
prediction = predict_conversion(fit_result, lambda t: 298.15, np.linspace(0, 3*365*24*3600, 500),
                               bootstrap_result=bootstrap_result)

# Export everything in one call
files = export_prediction_report(
    prediction=prediction,
    bootstrap_result=bootstrap_result,
    path_prefix='stability_3year',
    time_units='months'
)
# Creates: stability_3year.csv, stability_3year_bootstrap.csv, stability_3year_plot.png

# Or export individually
df = prediction.to_dataframe()
df.to_csv('prediction.csv', index=False)

bootstrap_summary = bootstrap_result.summary_dataframe()
bootstrap_summary.to_csv('bootstrap_ci.csv', index=False)
```

**Methods:**
- `PredictionResult.to_dataframe()` → pandas DataFrame with time, temperature, conversion, CIs
- `PredictionResult.to_csv(path)` → Export directly to CSV
- `BootstrapResult.to_dataframe()` → Parameter distributions (for histograms, statistics)
- `BootstrapResult.summary_dataframe()` → CI summary table
- `export_prediction_report()` → One-call export of CSV + plot

## Plotting Convenience Functions

Thin wrappers around matplotlib for common kinetic analysis plots.

### Ea(α) Isoconversional Plot

```python
from akts import run_friedman, plot_ea_vs_alpha

friedman = run_friedman(datasets)
fig = plot_ea_vs_alpha(friedman)
fig.savefig('ea_vs_alpha.png')
```

### Fit Overlay (Data vs Model)

```python
from akts import plot_fit_overlay, predict_conversion

# Plot experimental data with model prediction
prediction = predict_conversion(fit_result, lambda t: 298.15,
                               simulation_time_sec=np.linspace(0, 7200, 100))
fig = plot_fit_overlay(datasets, fit_result, prediction=prediction, time_units='hours')
fig.savefig('fit_overlay.png')
```

### Bootstrap Confidence Intervals

```python
from akts import plot_bootstrap_ci_bands

# Prediction with bootstrap CI
prediction = predict_conversion(fit_result, lambda t: 298.15,
                               simulation_time_sec=np.linspace(0, 7200, 100),
                               bootstrap_result=bootstrap_result)
fig = plot_bootstrap_ci_bands(prediction, time_units='months')
fig.savefig('prediction_ci.png')
```

### Arrhenius Plot

```python
from akts import plot_arrhenius

fig = plot_arrhenius(fit_result)
fig.savefig('arrhenius.png')
```

### Parameter Distributions

```python
from akts import plot_parameter_distributions

fig = plot_parameter_distributions(bootstrap_result, parameters=['Ea', 'A'])
fig.savefig('parameter_distributions.png')
```

### Multi-Temperature Data

```python
from akts import plot_multi_temperature_data

fig = plot_multi_temperature_data(datasets, time_units='hours')
fig.savefig('multi_temp_data.png')
```

**All plotting functions**:
- Return `matplotlib.Figure` objects for customization
- Accept optional `ax` parameter to plot on existing axes
- Support custom `figsize` parameter
- Use sensible defaults for labels, titles, and styling

## JSON Utilities

```python
from akts.json_utils import parse_json_data, serialize_results_to_json

# JSON dict to KineticDataset
json_data = {'time': [...], 'temperature': [...], 'conversion': [...]}
dataset = parse_json_data(json_data)

# Results to JSON string
json_string = serialize_results_to_json(results)
```

## Complete Example

```python
from akts import auto_model_isothermal_data, models

# Automated analysis example
results = auto_model_isothermal_data(
    data_files=['25C.csv', '40C.csv', '60C.csv'],
    predict=(2, 'year'),
    input_temperature_units='C',
    output_temperature_units='F',
    models_to_try=models.default + models.ode.all,
    report_path='report.html',
    bootstrap_iterations=100
)

print(f"Best model: {results['selected_model']['model_name']}")
print(f"Ea = {results['selected_model']['parameters']['Ea']/1000:.1f} kJ/mol")
print(f"Predicted at 2 years: {results['predictions']['conversion_mean'][-1]:.1%}")
```

## See Also

- **[Getting Started](getting_started.md)** - Installation and tutorials
- **[Automated Analysis](automated_analysis.md)** - Full auto_model guide
- **[Examples](examples.md)** - Working code with data
- **[Kinetic Models](kinetic_models.md)** - Model equations
- **[Advanced Usage](advanced_usage.md)** - Custom workflows
- **[JSON I/O Specification](json_io_specification.md)** - Web API details
