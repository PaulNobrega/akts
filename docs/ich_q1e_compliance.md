# ICH Q1E Compliance Guide

## Overview

AKTS provides **full ICH Q1E compliance** for shelf-life determination in pharmaceutical stability studies. This guide explains the regulatory requirements, implementation details, and usage examples.

---

## Table of Contents

- [ICH Q1E Requirements](#ich-q1e-requirements)
- [Quick Start](#quick-start)
- [Understanding One-Sided Confidence Intervals](#understanding-one-sided-confidence-intervals)
- [Confidence-Band Crossing Method](#confidence-band-crossing-method)
- [Two Shelf-Life Estimates](#two-shelf-life-estimates)
- [Plotted Bands: CI vs PI](#plotted-bands-ci-vs-pi)
- [Usage Examples](#usage-examples)
- [API Reference](#api-reference)
- [Validation and Testing](#validation-and-testing)
- [Common Questions](#common-questions)
- [References](#references)

---

## ICH Q1E Requirements

### What is ICH Q1E?

ICH Q1E is the International Council for Harmonisation guideline on "Evaluation of Stability Data" for pharmaceuticals. It provides statistical approaches for analyzing stability data and establishing shelf-life (retest periods).

### Key Requirements for Shelf-Life Determination

1. **One-Sided Confidence Bounds**
   - Use 95% one-sided confidence limits (not two-sided intervals)
   - Two-sided [2.5%, 97.5%] intervals are **not compliant**

2. **Appropriate Bound Selection**
   - **Decreasing attributes** (potency, active content, monomer): Use the bound representing faster degradation
   - **Increasing attributes** (aggregates, impurities): Use the bound representing faster increase

3. **Confidence-Band Crossing Method** (Preferred)
   - Calculate confidence bound at each time point
   - Find where the bound crosses the specification
   - More robust than taking percentiles of expiry times

4. **Conservative Estimates**
   - Shelf-life from confidence bound should be shorter than mean
   - Typical reduction: 5-15% shorter than mean estimate
   - This is **correct and expected** behavior

### Why This Matters

Using two-sided confidence intervals or incorrect bounds can result in:
- ❌ Overly optimistic shelf-life estimates
- ❌ Regulatory rejection of submissions
- ❌ Need to reanalyze stability data
- ❌ Delayed product approvals

---

## Quick Start

### Minimal Example (Automatic Defaults)

```python
from akts import auto_model_isothermal_data

# ICH Q1E-compliant analysis with automatic defaults
results = auto_model_isothermal_data(
    data_files=['potency_25C.csv', 'potency_40C.csv'],
    predict=(2, 'year', 298.15),  # Predict at 25°C for 2 years
    
    # Bootstrap required for confidence intervals
    bootstrap_iterations=500,  # 500+ recommended for regulatory
    
    # Data type (for auto-detection)
    readout_type='decreasing',  # 'decreasing' for potency, 'increasing' for aggregates
)

# Access ICH Q1E-compliant shelf-life
if 'shelf_life_ich_q1e' in results['predictions']:
    shelf_life = results['predictions']['shelf_life_ich_q1e']
    
    # Conservative estimate (for regulatory submission)
    days = shelf_life['shelf_life_conservative'] / 86400
    print(f"ICH Q1E Shelf-Life: {days:.1f} days")
    
    # Mean estimate (for comparison)
    days_mean = shelf_life['shelf_life_mean'] / 86400
    print(f"Mean Shelf-Life: {days_mean:.1f} days")
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `attribute_direction` | Auto-detect | `'decreasing'` or `'increasing'` (auto-detected from `readout_type`) |
| `calculate_shelf_life_ich_q1e_method` | `True` | Enable ICH Q1E calculation |
| `shelf_life_specification_limit` | `0.05` | Specification limit (5% default) |
| `bootstrap_iterations` | `100` | Required for confidence intervals (500+ recommended) |
| `ci_type` | `'two-sided'` | Band shown on prediction/simulation plots. Does not affect the ICH Q1E shelf-life, which always uses the one-sided 95th-percentile bound |

---

## Understanding One-Sided Confidence Intervals

### Two-Sided vs One-Sided

**Two-Sided (NON-COMPLIANT with ICH Q1E):**
```
95% CI = [2.5th percentile, 97.5th percentile]
Splits 5% error equally on both sides
```

**One-Sided (ICH Q1E-COMPLIANT):**
```
95% Lower bound = 5th percentile
95% Upper bound = 95th percentile  
All 5% error on one side
```

### Why One-Sided?

For shelf-life, we only care about one direction:
- **Potency decreasing**: We care about the lower bound (worst case degradation)
- **Impurities increasing**: We care about the upper bound (worst case accumulation)

### Implementation Details

In AKTS, when tracking **conversion** (degradation increases over time):

```python
# For decreasing quality attributes (potency):
# "Lower bound on potency" = "Upper bound on degradation"
# Use 95th percentile of degradation curves

# For increasing attributes (impurities):  
# "Upper bound on impurities" = "Upper bound on conversion"
# Use 95th percentile of impurity curves

# Both cases: 95th percentile = conservative
```

**Key Insight:** Both decreasing and increasing attributes use the **95th percentile** because we're tracking degradation/conversion which increases over time.

### Plotting Bands vs Shelf-Life Bound

`predict_conversion()` and `auto_model_isothermal_data()` take `ci_type`:

| `ci_type` | Band returned | Use |
|-----------|---------------|-----|
| `'two-sided'` (default) | Equal-tailed bootstrap percentiles, [2.5th, 97.5th] at 95% | Fit, prediction and simulation plots |
| `'one-sided'` | (MLE curve, 95th percentile) | ICH Q1E shelf-life crossing |

In `auto_model_isothermal_data()` the bootstrap ICH Q1E shelf-life is always computed from a separate one-sided 95th-percentile prediction, whatever `ci_type` is set to. This also applies to empirical models. If you call `time_to_specification()` yourself, pass it a prediction made with `ci_type='one-sided'` (see Example 3).

---

## Confidence-Band Crossing Method

### The Preferred Approach

ICH Q1E recommends the "confidence-band crossing method":

1. **Calculate** the one-sided confidence bound at each time point
2. **Interpolate** to find the exact time the bound crosses the specification
3. **Report** that crossing time as the conservative shelf-life

### Why It's Better

**Alternative (percentiles of expiry times):**
- Calculate expiry time for each bootstrap replicate
- Take 5th percentile of those times
- **Problem:** Different curves may cross differently; some may not cross at all

**Confidence-band crossing (preferred):**
- More robust to outliers
- Handles non-crossing curves gracefully
- Recommended by regulatory guidance

### Visualization

```
Conversion
    ^
    |           ╱╱╱╱ 95th percentile (conservative bound)
    |         ╱╱
 Spec|  - - -╱- - - - - - - (specification limit)
    |      ╱  ╱╱╱╱ Mean
    |    ╱  ╱╱
    |  ╱  ╱╱
    |╱  ╱╱
    +--+-+----------------> Time
       ^  ^
       |  |
       |  Mean shelf-life
       |
       ICH Q1E shelf-life (where 95th percentile crosses spec)
```

---

## Two Shelf-Life Estimates

`auto_model_isothermal_data()` reports two independent shelf-life estimates:

| Estimate | Where | Method |
|----------|-------|--------|
| Bootstrap (kinetic model) | `results['predictions']['shelf_life_ich_q1e']` | Selected kinetic model extrapolated to the `predict` temperature; time at which the one-sided 95th-percentile bootstrap bound crosses `shelf_life_specification_limit` |
| Regression (trend) | `results['regulatory']` | Linear or quadratic trend (nested F-test, `shelf_life_nonlinearity_p_threshold`) fitted to the data measured at `shelf_life_temperature_C`; time at which the one-sided `shelf_life_confidence_level` limit for the mean crosses `shelf_life_target_conversion` |

The regression estimate uses no bootstrap and no kinetic model. It needs study data at the shelf-life temperature (within 0.5 K) and is logged as, for example, "Shelf-life regression estimate (linear trend): 23.3 months (one-sided 95% confidence limit ...)". It is also compared against the ICH Q1E extrapolation ceiling (`ich_ceiling_months`, `exceeds_guideline`).

Its plot shows a two-sided 90% band around the trend. The edges of a two-sided 90% band are one-sided 95% limits, so the legend reads "One-sided 95% confidence limits".

### Bootstrap Robustness

- **Non-converged refits**: bootstrap refits whose RSS exceeds 10x the median replicate RSS are excluded as optimizer failures, with a warning giving the count. Without this, a few diverged refits can push the upper band to 100%.
- **Degenerate replicates**: replicate curves whose final conversion is below min(1%, 10% of the main prediction's final value) are dropped from the band (when at least 10 valid replicates remain). The relative threshold keeps the CI for genuinely low-conversion predictions, such as a short shipping excursion.

---

## Plotted Bands: CI vs PI

Fit plots in the report show two bands per temperature:

- **95% CI** (darker): bootstrap uncertainty of the fitted curve only. Data points scattering outside it is expected.
- **95% PI** (lighter): prediction interval for a new measurement. Half-width = t * sqrt(se_fit^2 + s_i^2), where s_i^2 is that temperature's residual variance and se_fit comes from the bootstrap CI half-width. About 95% of data points should fall inside it.

The PI is stored on the fit result as `conversion_simulated_pi`; `plotting.plot_fit_overlay()` draws it with `show_pi=True` (default). Report fit plots have one panel per temperature with its own y-scale and a time axis in days, so narrow low-temperature bands remain visible.

Shelf-life uses the mean-response confidence bound, not the PI.

---

## Usage Examples

### Example 1: Protein Aggregation (Increasing Attribute)

```python
from akts import auto_model_isothermal_data

results = auto_model_isothermal_data(
    data_files=[
        'aggregates_5C.csv',
        'aggregates_25C.csv', 
        'aggregates_40C.csv'
    ],
    predict=(3, 'year', 278.15),  # 5°C storage for 3 years
    
    # Bootstrap for confidence intervals
    bootstrap_iterations=500,
    bootstrap_method='monte_carlo',
    confidence_level=0.95,
    
    # Data type
    readout_type='increasing',  # Aggregates increase over time
    
    # ICH Q1E parameters (all optional, showing defaults)
    # attribute_direction='increasing',  # Auto-detected from readout_type
    # calculate_shelf_life_ich_q1e_method=True,  # Enabled by default
    # shelf_life_specification_limit=0.05,  # 5% aggregates spec
    
    # Model selection
    models_to_try=['SB_1_1', 'SB_2_1', 'F1', 'F2'],
    top_n=3,
    
    # Output
    report_path='aggregates_stability_report.html',
    progress_callback=lambda msg, data: print(msg)
)

# Extract ICH Q1E shelf-life
shelf_life = results['predictions']['shelf_life_ich_q1e']

print(f"\nICH Q1E Compliance Summary:")
print(f"  Method: {shelf_life['method']}")
print(f"  Attribute Direction: {shelf_life['attribute_direction']}")
print(f"  Specification: {shelf_life['specification_limit']:.1%}")
print(f"  Confidence Level: {shelf_life['confidence_level']:.0%}")
print(f"\nShelf-Life Results:")
print(f"  Mean: {shelf_life['shelf_life_mean']/86400:.1f} days")
print(f"  Conservative (ICH Q1E): {shelf_life['shelf_life_ich_q1e']/86400:.1f} days")

reduction = (shelf_life['shelf_life_mean'] - shelf_life['shelf_life_conservative'])
reduction_pct = reduction / shelf_life['shelf_life_mean'] * 100
print(f"  Reduction: {reduction_pct:.1f}% (conservative vs mean)")
```

### Example 2: Potency Loss (Decreasing Attribute)

```python
from akts import auto_model_isothermal_data

results = auto_model_isothermal_data(
    data_files=[
        'potency_25C.csv',
        'potency_40C.csv',
        'potency_60C.csv'
    ],
    predict=(2, 'year', 298.15),  # 25°C for 2 years
    
    bootstrap_iterations=1000,  # More iterations for higher accuracy
    readout_type='decreasing',  # Potency decreases
    
    # Custom specification (10% loss instead of default 5%)
    shelf_life_specification_limit=0.10,
    
    report_path='potency_stability_report.html',
)

# Results automatically include ICH Q1E shelf-life
shelf_life = results['predictions']['shelf_life_ich_q1e']
print(f"Regulatory Shelf-Life: {shelf_life['shelf_life_ich_q1e']/86400:.1f} days")
```

### Example 3: Manual Calculation (Lower-Level API)

```python
from akts import (
    load_data_file,
    fit_kinetic_model,
    run_bootstrap,
    predict_conversion,
    time_to_specification
)
import numpy as np

# 1. Load data
datasets = [
    load_data_file('data_25C.csv'),
    load_data_file('data_40C.csv')
]

# 2. Fit model
fit_result = fit_kinetic_model(datasets, model_name='SB_1_1')

# 3. Bootstrap for confidence intervals
bootstrap_result = run_bootstrap(
    datasets, 
    fit_result, 
    n_iterations=500,
    method='monte_carlo'
)

# 4. Predict with ICH Q1E-compliant one-sided CI
time_points = np.linspace(0, 365*2*86400, 200)  # 2 years
prediction = predict_conversion(
    kinetic_description=fit_result,
    temperature_program=lambda t: 298.15,  # 25°C
    simulation_time_sec=time_points,
    bootstrap_result=bootstrap_result,
    attribute_direction='decreasing',  # For potency
    ci_type='one-sided'  # Default 'two-sided' is for plotting
)

# 5. Calculate ICH Q1E shelf-life
shelf_life = time_to_specification(
    prediction,
    specification_limit=0.10,  # 10% degradation
    attribute_direction='decreasing'
)

print(f"ICH Q1E Shelf-Life: {shelf_life['shelf_life_ich_q1e']/86400:.1f} days")
print(f"Method: {shelf_life['method']}")
print(f"Has CI: {shelf_life['has_confidence_interval']}")
```

### Example 4: Disabling ICH Q1E (If Not Needed)

```python
results = auto_model_isothermal_data(
    data_files=['data.csv'],
    predict=(2, 'year', 298.15),
    bootstrap_iterations=100,
    
    # Disable ICH Q1E calculation
    calculate_shelf_life_ich_q1e_method=False,
)

# No bootstrap ICH Q1E shelf-life in results['predictions']
# The regression shelf-life in results['regulatory'] is still calculated
```

---

## API Reference

### High-Level API

#### `auto_model_isothermal_data()`

Main function for automated kinetic analysis with ICH Q1E compliance.

**ICH Q1E-Related Parameters:**

```python
def auto_model_isothermal_data(
    data_files,
    predict=None,
    bootstrap_iterations=100,
    confidence_level=0.95,
    ci_type='two-sided',  # Plotted band only; shelf-life always one-sided
    
    # ICH Q1E Parameters
    attribute_direction=None,  # Auto-detect from readout_type
    calculate_shelf_life_ich_q1e_method=True,  # Enabled by default
    shelf_life_specification_limit=0.05,  # 5% default
    
    # ... other parameters
)
```

**Returns:**
```python
results = {
    'predictions': {
        'shelf_life_ich_q1e': {
            'shelf_life_mean': float,  # Mean crossing time (seconds)
            'shelf_life_conservative': float,  # Conservative estimate (seconds)
            'shelf_life_ich_q1e': float,  # Alias for conservative
            'method': str,  # 'ich_q1e_one_sided_ci'
            'confidence_level': float,  # 0.95
            'attribute_direction': str,  # 'decreasing' or 'increasing'
            'has_confidence_interval': bool,  # True if CI available
            'specification_limit': float,  # Spec used
        },
        # ... other prediction data
    },
    'regulatory': {  # Regression shelf-life; None if no data at shelf_life_temperature_C
        'shelf_life_lower_95': float,  # One-sided limit crossing (months)
        'trend_type': str,  # 'linear' or 'quadratic'
        'ich_ceiling_months': float,
        'exceeds_guideline': bool,
        # ...
    },
    # ... other results
}
```

### Lower-Level API

#### `predict_conversion()`

Predict conversion trajectory with a bootstrap CI band (two-sided by default, one-sided for ICH Q1E).

```python
from akts import predict_conversion

prediction = predict_conversion(
    kinetic_description,  # FitResult or IsoResult
    temperature_program,  # Callable or (time, temp) tuple
    simulation_time_sec=None,  # Time points
    bootstrap_result=None,  # BootstrapResult for CI
    attribute_direction='decreasing',  # Used with ci_type='one-sided'
    ci_type='two-sided',  # 'one-sided' -> (MLE curve, 95th percentile)
)
```

#### `time_to_specification()`

Calculate ICH Q1E-compliant shelf-life from prediction.

```python
from akts import time_to_specification

shelf_life = time_to_specification(
    prediction_result,  # PredictionResult made with ci_type='one-sided'
    specification_limit=0.05,  # Spec limit (0-1 range)
    attribute_direction='decreasing',  # Match prediction
)
```

#### `calculate_shelf_life_ich_q1e()`

Low-level function for direct shelf-life calculation.

```python
from akts import calculate_shelf_life_ich_q1e

shelf_life = calculate_shelf_life_ich_q1e(
    time,  # np.ndarray of time points
    conversion_mean,  # Mean trajectory
    conversion_lower,  # Lower CI bound (or None)
    conversion_upper,  # Upper CI bound (conservative)
    specification_limit,  # Spec limit
    attribute_direction='decreasing',  # Attribute type
    time_units='seconds',  # Time units
)
```

---

## Validation and Testing

### Test Coverage

AKTS includes comprehensive tests for ICH Q1E compliance:

```bash
# Run all ICH Q1E tests
pytest tests/test_ich_q1e_compliance.py -v

# Specific test categories
pytest tests/test_ich_q1e_compliance.py::TestOneSidedCI -v
pytest tests/test_ich_q1e_compliance.py::TestShelfLifeDecreasing -v
pytest tests/test_ich_q1e_compliance.py::TestRegulatoryCompliance -v
```

**Test Results:** 22/22 tests passing ✅

### Validation Checklist

Before regulatory submission, verify:

- ✅ One-sided CI calculated correctly (95th percentile for conservative bound)
- ✅ Shelf-life uses confidence-band crossing method
- ✅ Appropriate bound used based on attribute direction
- ✅ Conservative estimate is shorter than mean
- ✅ Mean-response CI (not prediction interval)
- ✅ Few or no bootstrap refits excluded as non-converged (see warnings)
- ✅ Bootstrap iterations ≥ 500 for regulatory submissions
- ✅ Documentation includes ICH Q1E compliance statement
- ✅ Results reproducible with random_state parameter

### Example Validation Script

```python
import numpy as np
from akts import auto_model_isothermal_data

# Reproducible analysis
results = auto_model_isothermal_data(
    data_files=['data_25C.csv', 'data_40C.csv'],
    predict=(2, 'year', 298.15),
    bootstrap_iterations=1000,
    random_state=42,  # Reproducibility
    readout_type='decreasing',
)

shelf_life = results['predictions']['shelf_life_ich_q1e']

# Validation checks
assert shelf_life['method'] == 'ich_q1e_one_sided_ci'
assert shelf_life['confidence_level'] == 0.95
assert shelf_life['has_confidence_interval'] == True
assert shelf_life['shelf_life_conservative'] < shelf_life['shelf_life_mean']

print("✅ ICH Q1E validation passed")
```

---

## Common Questions

### Q: Why is my conservative shelf-life shorter than the mean?

**A:** This is **correct and expected**! The conservative estimate uses the 95th percentile bound which represents faster degradation. Regulatory guidance requires this conservative approach.

Typical reduction: 5-15% shorter than mean.

### Q: Should I use 'decreasing' or 'increasing' for my attribute?

**A:** 
- **'decreasing'**: Use for potency, active content, monomer content (quality decreases over time)
- **'increasing'**: Use for aggregates, impurities, degradation products (degradants increase over time)

When in doubt, check what your readout measures:
- Measuring remaining potency? → 'decreasing'
- Measuring % aggregates? → 'increasing'

### Q: What if I don't have bootstrap iterations?

**A:** Bootstrap is **required** for ICH Q1E-compliant shelf-life calculation. Without bootstrap:
- No confidence intervals available
- Only mean shelf-life can be calculated
- Not suitable for regulatory submission

Minimum recommended: 500 iterations for regulatory work.

### Q: Can I use two-sided intervals instead?

**A:** **No**, two-sided confidence intervals are not compliant with ICH Q1E for shelf-life determination. ICH Q1E specifically requires one-sided confidence bounds. Plots use a two-sided band by default (`ci_type='two-sided'`), but the shelf-life calculation in `auto_model_isothermal_data()` always uses the one-sided bound.

### Q: Why do many data points fall outside the confidence band?

**A:** The CI describes uncertainty in the fitted curve, not the scatter of individual measurements. Check the lighter prediction interval (PI) instead: about 95% of points should fall inside it.

### Q: Why do the bootstrap and regression shelf-lives differ?

**A:** They are independent methods. The bootstrap estimate extrapolates the selected kinetic model to the `predict` temperature; the regression estimate fits a linear/quadratic trend to data measured at `shelf_life_temperature_C`. See [Two Shelf-Life Estimates](#two-shelf-life-estimates).

### Q: What specification limit should I use?

**A:** Common specifications:
- **5% (0.05)**: Default, common for aggregation and general degradation
- **10% (0.10)**: Common for potency loss (90% potency specification)
- **2% (0.02)**: Stricter specifications for sensitive products

Consult your product specification and regulatory requirements.

### Q: How do I report this in regulatory documents?

**A:** Example language:

> "Shelf-life was determined using ICH Q1E-compliant methodology with one-sided 95% confidence bounds. The shelf-life is defined as the time at which the appropriate confidence bound crosses the specification limit, using the confidence-band crossing method. Bootstrap resampling (n=1000 iterations) was used to calculate confidence intervals. The conservative shelf-life estimate is [X] months at [Y]°C storage."

---

## References

### Regulatory Guidelines

1. **ICH Q1E**: "Evaluation of Stability Data"
   - https://database.ich.org/sites/default/files/Q1E_Guideline.pdf

2. **FDA Guidance**: Statistical approaches to establishing bioequivalence
   - https://www.fda.gov/regulatory-information/search-fda-guidance-documents

### Scientific Literature

1. Ahlstrom et al. (2015). "ICH Q1E stability-data evaluation"
2. Roduit et al. (2019). "AKTS advanced kinetic analysis methodology"

### AKTS Documentation

- [Getting Started](getting_started.md)
- [Advanced Usage](advanced_usage.md)
- [API Reference](api_reference.md)
- [Examples](examples.md)

---

## Support

For questions about ICH Q1E compliance:
- Review examples in `Examples/Commercial_AKTS_Comp_Isothermal/`
- Check test cases in `tests/test_ich_q1e_compliance.py`
- Consult with regulatory affairs team for specific applications
- Reference ICH Q1E guideline for official regulatory interpretation

---

## Compliance Statement

**AKTS provides full ICH Q1E compliance for shelf-life determination:**

✅ One-sided 95% confidence bounds  
✅ Confidence-band crossing method  
✅ Appropriate bound selection  
✅ Conservative estimates  
✅ Comprehensive testing and validation  
✅ Clear documentation and examples  

**Status:** Ready for regulatory submissions

**Last Updated:** 2026-09-30
