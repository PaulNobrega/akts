# Fit Plots and Prediction Temperature Fixes

## Issues Reported

1. **Fit lines not appearing** - Only data points visible in fit plot
2. **Residuals plot blank** - No residuals shown
3. **Prediction needs temperature** - How to specify temperature for predictions?

## Root Causes

### Issue 1 & 2: Missing Simulated Conversion Data

The `FitResult` objects returned by `fit_kinetic_model()` did not include simulated conversion data (`conversion_simulated` attribute). The reporting code expected this attribute to exist:

```python
# In reporting.py:
if hasattr(fit_result, 'conversion_simulated') and fit_result.conversion_simulated is not None:
    # Plot fit lines and residuals
```

Without this attribute:
-  No fit lines drawn (only data points)
-  No residuals calculated or plotted

### Issue 3: Prediction Temperature Not Configurable

The `predict` parameter only accepted `(time_value, time_unit)`, defaulting to the first dataset's temperature. Users had no way to specify:
- A specific temperature for extrapolation
- Predictions at conditions different from experimental data

## Solutions Implemented

### Fix 1: Simulate Conversion for Plotting

**File:** `akts/helpers.py`

**Added Step 7.5** after model selection to simulate conversion for each dataset:

```python
# Step 7.5: Simulate conversion for top models for plotting
for fit_res in top_fit_results:
    try:
        simulated_conversions = []
        for ds in datasets:
            temp_func = lambda t: np.interp(t, ds.time, ds.temperature)
            pred_result = predict_conversion(
                kinetic_description=fit_res,
                temperature_program=temp_func,
                simulation_time_sec=ds.time,
                initial_alpha=0.0
            )
            simulated_conversions.append(pred_result.conversion)
        # Add as attribute to FitResult object
        fit_res.conversion_simulated = simulated_conversions
    except Exception as e:
        warnings.warn(f"Failed to simulate conversion for plotting: {e}")
        fit_res.conversion_simulated = None
```

**Why This Approach:**
- Doesn't modify core `FitResult` dataclass (cleaner separation)
- Only simulates when needed for reporting
- Uses existing `predict_conversion()` function
- Handles each temperature dataset separately

**Result:**
-  Fit lines now appear on plots (solid lines matching data point colors)
-  Residuals now calculated and plotted
-  Both use the same color scheme per temperature

### Fix 2: Temperature Parameter for Predictions

**File:** `akts/helpers.py`

**Updated function signature** to accept temperature in multiple ways:

```python
def auto_model_isothermal_data(
    data_files: List[Union[str, Path, KineticDataset, Dict]],
    predict: Optional[Union[Tuple[float, str], Tuple[float, str, float]]] = None,
    predict_temperature_K: Optional[float] = None,
    ...
```

**Three ways to specify temperature:**

1. **Default (no temperature specified):**
   ```python
   predict=(3, 'year')  # Uses first dataset's average temperature
   ```

2. **Temperature in predict tuple:**
   ```python
   predict=(3, 'year', 298.15)  # Predict at 298.15 K (25°C)
   ```

3. **Separate parameter:**
   ```python
   predict=(3, 'year'),
   predict_temperature_K=298.15  # Predict at 298.15 K
   ```

**Updated prediction logic:**

```python
# Parse predict parameter (time_value, time_unit) or (time_value, time_unit, temp_K)
if len(predict) == 3:
    pred_value, pred_unit, temp_K = predict
elif len(predict) == 2:
    pred_value, pred_unit = predict
    # Use predict_temperature_K parameter if provided, else first dataset temperature
    if predict_temperature_K is not None:
        temp_K = predict_temperature_K
    else:
        temp_K = datasets[0].temperature.mean()
```

**Updated progress messages:**
```python
progress(f"Prediction for {pred_value} {pred_unit} at {temp_K:.1f} K: {pred_result.conversion[-1]:.2%} conversion",
        {'time_value': pred_value, 'time_unit': pred_unit, 'temperature_K': temp_K})
```

**Result:**
-  Temperature now included in predictions dictionary (`predictions_dict['temperature_K']`)
-  Users can specify any temperature for extrapolation
-  Clear progress messages show which temperature is being used
-  Backward compatible (existing code still works)

## Test Results

### Test 1: Default Temperature
```python
predict=(1, 'year')
# Output: Prediction for 1 year at 313.2 K: 100.00% conversion
# Uses: First dataset average (313.16 K)
```

### Test 2: Temperature in Tuple
```python
predict=(1, 'year', 310.0)
# Output: Prediction for 1 year at 310.0 K: 100.00% conversion
# Uses: Specified 310.0 K
```

### Test 3: Separate Parameter
```python
predict=(1, 'year'),
predict_temperature_K=350.0
# Output: Prediction for 1 year at 350.0 K: 100.00% conversion
# Uses: Specified 350.0 K
```

### HTML Report Verification

```bash
# Check for fit lines
grep -c '"mode":"lines"' test_fit_default_temp.html
# Output: 2 (fit plot lines + prediction plot line)

# Check for fit traces
grep -o '"name":"Fit @ [^"]*"' test_fit_default_temp.html
# Output:
# "name":"Fit @ 40°C"
# "name":"Fit @ 50°C"
# "name":"Fit @ 60°C"
# "name":"Fit @ 70°C"

# Check for residuals
grep -o '"name":"Residuals @ [^"]*"' test_fit_default_temp.html
# Output:
# "name":"Residuals @ 40°C"
# "name":"Residuals @ 50°C"
# "name":"Residuals @ 60°C"
# "name":"Residuals @ 70°C"
```

## Visual Verification

### Before Fix
```
Fit Plot:
- Data: ● ● ● ● (dots only)
- Fit:  (nothing - BLANK)

Residuals Plot:
- (completely blank - NO DATA)
```

### After Fix
```
Fit Plot:
- Data: ● ● ● ● (dots with markers)
- Fit:  ━━━━━━━ (solid lines matching dot colors)

Residuals Plot:
- Residuals: ● ● ● ● (dots showing fit quality)
- Zero line: ╌╌╌╌╌╌╌ (dashed reference)
```

## Files Modified

### 1. akts/helpers.py (2 additions)

**Addition 1: Simulate conversion for plotting** (after line ~509)
- Adds conversion_simulated attribute to top_fit_results
- Called before report generation
- Uses predict_conversion() with dataset's temperature profile

**Addition 2: Temperature handling in predictions** (lines ~236-240, ~475-495)
- Updated function signature with predict_temperature_K parameter
- Enhanced predict tuple parsing (2 or 3 elements)
- Added temperature to predictions_dict
- Updated progress messages to show temperature

### 2. Examples/Isothermal/demo_auto_model.py (documentation update)

**Added comments** showing temperature usage examples:
```python
predict=(3, 'year'),  # Uses first dataset temperature
# Optionally specify temperature:
# predict=(3, 'year', 298.15),  # Predict at 298.15 K (25°C)
# Or use separate parameter:
# predict_temperature_K=298.15,  # Predict at 298.15 K
```

## Use Cases

### Use Case 1: Shelf Life at Storage Temperature
```python
# Tested at 40°C, 50°C, 60°C, 70°C
# Want to predict at 25°C (room temperature)
results = auto_model_isothermal_data(
    data_files=['test_313K.csv', 'test_323K.csv', ...],
    predict=(2, 'year', 298.15),  # 2 years at 25°C
    report_path='shelf_life_25C.html'
)
```

### Use Case 2: Accelerated Aging Extrapolation
```python
# Tested at ambient conditions
# Want to predict at elevated temperature for accelerated testing
results = auto_model_isothermal_data(
    data_files=['ambient_data.csv'],
    predict=(6, 'month', 323.15),  # 6 months at 50°C
    report_path='accelerated_aging.html'
)
```

### Use Case 3: Worst-Case Scenario Analysis
```python
# Test across range, predict at worst-case (highest) temperature
results = auto_model_isothermal_data(
    data_files=temperature_range_files,
    predict=(1, 'year', 343.15),  # 1 year at 70°C (worst case)
    report_path='worst_case.html'
)
```

## Documentation Updates Needed

### README.md
Add section on prediction temperature:
```markdown
### Specifying Prediction Temperature

By default, predictions use the average temperature from the first dataset:
```python
predict=(3, 'year')  # Uses first dataset temperature
```

You can specify a custom temperature in two ways:

**1. Include in predict tuple:**
```python
predict=(3, 'year', 298.15)  # 3 years at 298.15 K (25°C)
```

**2. Use separate parameter:**
```python
predict=(3, 'year'),
predict_temperature_K=298.15
```

This is useful for:
- Predicting at storage conditions different from test conditions
- Extrapolating to worst-case temperatures
- Shelf-life estimates at room temperature
```

### API Reference
Update function signature and parameters documentation.

## Summary

### All Issues Resolved

1. **Fit lines appear** - Simulated conversion added to FitResult objects before plotting
2. **Residuals plotted** - Calculated from simulated conversion vs. actual data
3. **Temperature configurable** - Three ways to specify prediction temperature

### Quality Improvements

-  Plots now show complete fit information (data + model + residuals)
-  Users have full control over prediction conditions
-  Progress messages show exactly what temperature is being used
-  Backward compatible (existing code still works)
-  Well-documented with examples
-  Tested with multiple temperature specifications

The implementation is now complete and ready for production use!
