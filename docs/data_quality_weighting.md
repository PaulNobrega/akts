# Data Quality Weighting and Replicate Handling

## Overview

AKTS fits kinetic models by **weighted least squares**. Every point gets a transition weight, which emphasizes the kinetically informative part of the curve. If replicates are averaged, each point also gets an inverse-variance data quality weight.

By default, `auto_model_isothermal_data()` keeps **all replicate points** (`average_replicates=False`). Averaging is opt-in.

---

## Replicates: Keep or Average?

### Default Behavior

| Function | `average_replicates` default |
|----------|------------------------------|
| `auto_model_isothermal_data()` | `False` (keep all replicate points) |
| `load_data_file()` | `True` (average) |

`auto_model_isothermal_data()` passes its own `average_replicates` value to the loader, so its default of `False` applies to every file it loads.

### Why Keep Replicates

- **Same fit**: with an equal number of replicates at each time point, least squares on the replicates gives the same parameter estimates as least squares on the means.
- **Scatter is preserved**: the bootstrap resamples individual observations, and the prediction interval (PI) uses each temperature's residual variance. Both need the real replicate scatter, which averaging removes.
- **Consistency**: fitting on averages but bootstrapping on replicates mixes two different data sets. This is not recommended.

Averaging is only useful when replicate counts differ strongly between time points and you want inverse-variance weighting (see below).

### Example

**Raw Data:**
```
Time (days)    HMW (%)
0              1.20
0              1.22
0              1.19
7              1.25
7              1.28
7              1.24
```

**After Averaging (`average_replicates=True`):**
```
Time (days)    HMW (mean)    Std Dev    n_replicates    Weight (before normalization)
0              1.203         0.015      3               n/σ² = 12857
7              1.257         0.021      3               n/σ² = 6923
```

Weights are then normalized to a mean of 1.0.

### Usage

```python
from akts import auto_model_isothermal_data, load_data_file

# Default: all replicate points used for fitting and bootstrap
results = auto_model_isothermal_data(
    data_files=['HMW_5C.csv', 'HMW_25C.csv'],
    time_col='Time(day)',
    temperature_col='Temperature(°C)',
    readout_col='HMW(%)',
    readout_type='increasing',
)

# Loading a single file directly averages by default; keep replicates with:
dataset = load_data_file(
    'stability_data.csv',
    time_col='Time(day)',
    temperature_col='Temperature(°C)',
    readout_col='HMW(%)',
    readout_type='increasing',
    average_replicates=False,
)
```

---

## Weighting Scheme

AKTS uses **combined weighting** in the fitting objective function:

### Total Weight Formula

```python
total_weight = transition_weight × data_quality_weight   # data_quality_weight = 1 if not averaged
```

### 1. Transition Weight (Always Applied)

Points in the transition region carry 10x the weight of baseline/plateau points (`akts/fitting.py`):

```python
TRANSITION_WEIGHT = 10.0
BASELINE_WEIGHT = 1.0
TRANSITION_ALPHA_RANGE = (0.05, 0.95)  # exclusive bounds on observed conversion

def transition_weights(conversion):
    lo, hi = TRANSITION_ALPHA_RANGE
    return np.where((conversion > lo) & (conversion < hi), TRANSITION_WEIGHT, BASELINE_WEIGHT)
```

**Rationale:** Baseline and plateau points carry little information about the rate. Without this weighting, a fit can score well by matching the plateau alone. The same weights are used in the bootstrap's residual resampling.

### 2. Data Quality Weight (Only When Averaging)

From replicate standard deviation (inverse-variance weighting):

```python
# For replicated points (n > 1):
weight = n / σ²

# For single measurements:
weight = 1 / mean(σ²)

# Normalize to mean = 1.0
weights = weights / mean(weights)
```

**Rationale:**
- Points with **more replicates** get higher weight
- Points with **less scatter** get higher weight
- Standard practice in weighted least squares regression

---

## Mathematical Justification

### Weighted Least Squares (WLS)

The objective function minimizes:

```
χ² = Σᵢ wᵢ (yᵢ - ŷᵢ)²
```

Where:
- `yᵢ` = observed conversion
- `ŷᵢ` = model prediction
- `wᵢ` = combined weight (transition × data quality)

### Why Inverse-Variance?

When measurement errors have different variances, the optimal estimator is:

```
w_i = 1 / Var(y_i)
```

This gives the **Best Linear Unbiased Estimator (BLUE)** by the Gauss-Markov theorem.

For replicated measurements:
```
Var(ȳ) = σ² / n
```

Therefore:
```
w = 1 / Var(ȳ) = n / σ²
```

Fitting the individual replicates with equal weight is equivalent when every time point has the same number of replicates with similar scatter.

---

## Verification

### Check if Averaging Occurred

```python
dataset = load_data_file('data.csv', ...)

print(f"Averaged: {dataset.metadata.get('replicates_averaged')}")
print(f"Points before: {dataset.metadata.get('n_points_before_averaging')}")
print(f"Points after: {dataset.metadata.get('n_points_after_averaging')}")
print(f"Replicated points: {dataset.metadata.get('n_replicated_points')}")

if dataset.weights is not None:
    print(f"Weight range: {dataset.weights.min():.2f} - {dataset.weights.max():.2f}")
    print(f"Weight mean: {dataset.weights.mean():.2f}")
```

### Example Output

```
Averaged: True
Points before: 56
Points after: 7
Replicated points: 7
Weight range: 0.28 - 1.12
Weight mean: 1.00
```

---

## Integration with Bootstrap and Prediction Intervals

- **Bootstrap**: `bootstrap_method='monte_carlo'` resamples complete observations within each dataset; `'residual'` resamples centered residuals with transition weighting. With replicates kept, the resampling reflects the real measurement scatter.
- **Prediction interval (PI)**: report fit plots show a 95% PI with half-width t * sqrt(se_fit^2 + s_i^2), where s_i^2 is the residual variance of that temperature's data. With averaged data, s_i^2 describes the scatter of means rather than of individual measurements, so the PI is too narrow for new single measurements.

See [ICH Q1E Compliance](ich_q1e_compliance.md#plotted-bands-ci-vs-pi) for CI vs PI.

---

## Advanced Usage

### Custom Weighting

If you have custom weights (e.g., from instrument precision specs):

```python
from akts import KineticDataset
import numpy as np

# Manual dataset creation
dataset = KineticDataset(
    time=time_array,
    temperature=temp_array,
    conversion=conv_array,
    weights=custom_weights,  # Your custom weights
    metadata={'custom_weighting': True}
)

# Fitting will use your custom weights
from akts import fit_kinetic_model
result = fit_kinetic_model(
    datasets=[dataset],
    model_name='F1',
    ...
)
```

Bootstrap replicate datasets are rebuilt without the `weights` field, so custom weights affect the main fit but not the bootstrap refits.

### Disable Transition Weighting

Not exposed as a parameter; transition weighting is always applied. To change it, edit `TRANSITION_WEIGHT` / `TRANSITION_ALPHA_RANGE` in `akts/fitting.py` (setting `TRANSITION_WEIGHT = 1.0` gives uniform weighting).

---

## Implementation Details

### Files

1. **akts/datatypes.py**: `weights` field on `KineticDataset`
2. **akts/loaders.py**: `_average_replicates()` function
3. **akts/fitting.py**: `transition_weights()`; objective functions multiply by `ds.weights` when present

### Algorithms

**Replicate Detection:**
```python
# Round to avoid floating-point issues
time_rounded = np.round(time, decimals=6)
temp_rounded = np.round(temperature, decimals=3)

# Group by (time, temp)
grouped = df.groupby(['time_rounded', 'temp_rounded']).agg({
    'conversion': ['mean', 'std', 'count']
})
```

**Weight Calculation:**
```python
# For replicated points
mask_replicated = n_replicates > 1
weights[mask_replicated] = n_replicates[mask_replicated] / (std[mask_replicated] ** 2)

# For single measurements
mean_variance = np.mean(std[mask_replicated] ** 2)
weights[~mask_replicated] = 1.0 / mean_variance

# Normalize
weights = weights / np.mean(weights)
```

---

## Edge Cases

### All Identical Replicates

If replicates are perfectly identical (σ = 0):
- Replace with minimum non-zero σ from other points
- If all σ = 0: uniform weighting

### No Replicates Detected

If no (time, temperature) combinations are repeated:
- `weights = None`
- Fitting proceeds with transition weighting only
- `metadata['replicates_averaged'] = False`

### Single Dataset Point

If only one measurement exists:
- No averaging possible
- Weight = 1.0 (neutral)

---

## Best Practices

### Do:
- Keep replicates (the `auto_model_isothermal_data()` default)
- Use the same data (replicates or means) for both fitting and bootstrap
- Check that about 95% of points fall inside the report's PI
- Use bootstrap for uncertainty quantification

### Don't:
- Fit on averages and bootstrap on replicates
- Pre-average data externally if you want the PI and bootstrap to reflect measurement scatter
- Ignore warnings about measurement quality
- Trust fits with only 1-2 unique time points per temperature

---

## References

1. **Gauss-Markov Theorem**: Weighted least squares optimality
2. **Aitken (1936)**: "On Least Squares and Linear Combination of Observations"
3. **ICH Q1E**: Evaluation of stability data
4. **ASTM E698**: Standard test method for kinetic parameters (weighted fitting)

---

## See Also

- [Getting Started](getting_started.md) - Basic usage
- [Advanced Usage](advanced_usage.md) - Custom workflows
- [ICH Q1E Compliance](ich_q1e_compliance.md) - Regulatory compliance, CI vs PI
- [API Reference](api_reference.md) - Complete API documentation

---

**Summary**: AKTS always applies transition weighting (10x inside 0.05 < α < 0.95). `auto_model_isothermal_data()` keeps all replicate points by default, so fitting, bootstrap, and prediction intervals all see the real measurement scatter. With `average_replicates=True`, replicates are averaged and weighted by n/σ².
