# Troubleshooting Guide

Common issues and solutions when using akts.

## Installation Issues

### ModuleNotFoundError: No module named 'akts'

**Problem:**
```python
>>> import akts
ModuleNotFoundError: No module named 'akts'
```

**Solutions:**

1. **Install in editable mode:**
   ```bash
   cd akts
   pip install -e .
   ```

2. **Check virtual environment:**
   - Make sure your virtual environment is activated
   - Windows: `.venv\Scripts\activate`
   - macOS/Linux: `source .venv/bin/activate`

3. **Verify Python path:**
   ```python
   import sys
   print(sys.path)
   # Should include the akts directory
   ```

### Dependency Installation Failures

**Problem:** `pip install -r requirements.txt` fails

**Solutions:**

1. **Update pip:**
   ```bash
   python -m pip install --upgrade pip
   ```

2. **Install build tools (Windows):**
   - Install Microsoft C++ Build Tools
   - Or use Anaconda: `conda install numpy scipy matplotlib`

3. **Try installing dependencies individually:**
   ```bash
   pip install numpy scipy pandas
   pip install matplotlib plotly
   pip install openpyxl
   ```

## Data Loading Issues

### FileNotFoundError

**Problem:**
```python
FileNotFoundError: [Errno 2] No such file or directory: 'data.csv'
```

**Solutions:**

1. **Use absolute paths:**
   ```python
   from pathlib import Path
   data_path = Path('C:/Users/username/data/data.csv')
   results = auto_model_isothermal_data(data_files=[data_path])
   ```

2. **Check current directory:**
   ```python
   import os
   print(os.getcwd())  # Where Python is looking
   ```

3. **Use relative paths from script location:**
   ```python
   from pathlib import Path
   script_dir = Path(__file__).parent
   data_path = script_dir / 'data' / 'data.csv'
   ```

### ValueError: Cannot detect column names

**Problem:** Loader can't auto-detect columns

**Solution:** Specify column names explicitly:
```python
dataset = load_isothermal_file(
    'data.csv',
    time_col='Time (days)',
    temperature_col='Temperature (K)',
    readout_col='Conversion',
    auto_detect=False  # Disable auto-detection
)
```

### ValueError: Conversion out of range

**Problem:** Conversion values outside [0, 1] range

**Solutions:**

1. **Check readout_type:**
   ```python
   # If readout INCREASES with degradation
   readout_type='increasing'  # e.g., HMW%, degraded%

   # If readout DECREASES with degradation
   readout_type='decreasing'  # e.g., intact%, remaining mass
   ```

2. **Set the conversion scale explicitly:**
   ```python
   # conversion = (readout - readout_initial) / (readout_final - readout_initial)
   results = auto_model_isothermal_data(
       data_files=files,
       readout_type='increasing',
       readout_final=100.0,   # commercial AKTS %HMW scaling: (HMW - HMW0)/(100 - HMW0)
   )
   ```
   Without `readout_initial` / `readout_final`, all files share one scale: the
   mean first readout is conversion 0 and the maximum observed readout (minimum
   for decreasing readouts) is conversion 1. Fitted Ea and A depend on this
   scale, so set `readout_final` when comparing with another tool.

3. **Normalize manually:**
   ```python
   import pandas as pd
   df = pd.read_csv('data.csv')
   df['conversion'] = (df['readout'] - df['readout'].min()) / (df['readout'].max() - df['readout'].min())
   ```

4. **Check for outliers:**
   ```python
   print(df['conversion'].describe())
   # Look for values < 0 or > 1
   ```

## Fitting Issues

### All Models Fail to Fit

**Problem:** Every model returns `success=False`

**Solutions:**

1. **Check data quality:**
   ```python
   import matplotlib.pyplot as plt
   for ds in datasets:
       plt.plot(ds.time, ds.conversion, 'o-', label=f'{ds.temperature}K')
   plt.xlabel('Time')
   plt.ylabel('Conversion')
   plt.legend()
   plt.show()
   # Look for: outliers, plateaus, decreasing conversion
   ```

2. **Verify sufficient data:**
   - Minimum 6-8 points per temperature
   - Conversion should span at least 0.1-0.8 range
   - Multiple temperatures (3+ recommended)

3. **Widen parameter bounds:**
   ```python
   parameter_bounds={
       'Ea': (5e3, 1000e3),    # Full default range, akts.utils.EA_BOUNDS
       'A': (1e3, 1e25)        # Wider A range
   }
   ```

4. **Try different initial guesses:**
   ```python
   initial_guesses={
       'Ea': 70000,   # Try different values
       'A': 1e10      # Based on literature
   }
   ```

### Negative R² Values

**Problem:** Model fits show negative R² (worse than horizontal line)

**Explanation:** Model is completely inappropriate for the data

**Solutions:**

1. **Try simpler models first:**
   ```python
   models_to_try=['F1', 'F2']  # Start with simple models
   ```

2. **Check if data is actually kinetic:**
   - Should show smooth degradation/reaction
   - Not step functions or discontinuities
   - Temperature dependence should exist

3. **Verify temperature units:**
   ```python
   input_temperature_units='C'  # Don't mix K and C!
   ```

### RuntimeWarning: overflow encountered in exp

**Problem:** Warnings about overflow during fitting

**Solution:** Overflow warnings can indicate parameter values outside a useful range. If they occur repeatedly:

1. **Tighten A bounds:**
   ```python
   parameter_bounds={
       'Ea': (40000, 200000),
       'A': (1e6, 1e18)  # Tighter range
   }
   ```

2. **Check for bad data:**
   - Extremely high or low temperatures
   - Very short or long time scales

### ODE Models Taking Too Long

**Problem:** A→B→C or A+B→C fitting is slow

**Explanation:** `auto_model_isothermal_data()` automatically uses multistart local
fitting for ODE models: `fit_kinetic_model()` (which internally uses an Arrhenius
reparameterization, a coarse rate-constant scan, and the Powell method) is run from
several starting points and the best-R² result is kept. See
[Fitting ODE Models Reliably](bayesian_optimization.md) for details.

**Solutions:**

1. **Bound a single fit's wall-clock time** (default is 60 seconds per parameter,
   e.g. 240s for the 4-parameter A->B->C model):
   ```python
   fit_result = fit_kinetic_model(
       ...,
       optimizer_options={'max_seconds': 60}  # tighten the deadline
   )
   ```

2. **Skip ODE models if not needed:**
   ```python
   models_to_try=models.default  # Use default kinetic models only
   ```

### Fitted Ea is at the fitting bound

**Problem:** Warning `Fitted Ea1 (1000.0 kJ/mol) is at the fitting bound (5-1000 kJ/mol); the optimum may lie outside it.`

**Explanation:** Every Ea parameter is fitted within `akts.utils.EA_BOUNDS` (5-1000 kJ/mol). A value at the bound means the data do not pin Ea down inside that range. This is common for the steep step of SB2, whose Ea is poorly identified when only one or two temperatures show that step.

**Solutions:**
- Treat the bound-limited Ea as a lower (or upper) limit, not an estimate
- Compare with the typical ranges in the report (thermal denaturation/unfolding 400-800 kJ/mol, enzymatic/proteolytic cleavage 20-100, spontaneous/pyrolytic hydrolysis 90-140)
- Add temperatures in the region where that step dominates, or try a simpler model

### Warning: No physically plausible models achieved R² ≥ 0.70

**Problem:** Ranking warns that the selected model has questionable plausibility.

**Explanation:** Plausibility requires every A parameter below 1e20 s⁻¹ and Ea at most 1000 kJ/mol. When no model with R² ≥ `min_r_squared` passes, the models that meet the R² threshold are ranked anyway and this warning is attached (`filter_warning['type'] == 'no_plausible_models'`). SB2 always triggers it when it is the only candidate: its fast step needs A of about 1e166 or more. The model is still selected.

**Solutions:**
- For SB2, the warning refers to the A values and is expected
- For other models, check the parameters for a bound-limited or extreme fit before relying on predictions

### Conversion-based fit failed, falling back to RATE

**Problem:** Warning appears during fitting

**Explanation:** The optimizer first fits conversion directly. If that fit fails, it attempts a rate-based fit. The warning reports this fallback.

No action is required if the rate-based fit succeeds and its diagnostics are acceptable.

## Bootstrap Issues

### BrokenProcessPool on Windows

**Problem:**
```python
RuntimeError: An attempt has been made to start a new process before the current process has finished its bootstrapping phase
```

**Solution:** Wrap in `if __name__ == '__main__':`:
```python
if __name__ == '__main__':
    bootstrap_result = run_bootstrap(
        datasets=datasets,
        fit_result=fit_result,
        n_iterations=100
    )
```

### Bootstrap Taking Forever

**Problem:** Bootstrap stuck or very slow

**Solutions:**

1. **Check CPU usage:**
   - Bootstrap uses all cores by default (`n_jobs=-1`)
   - Should see high CPU usage
   - If not, may be serialized: try `n_jobs=1`

2. **Reduce iterations:**
   ```python
   n_iterations=50  # Faster, less accurate
   ```

3. **Increase timeout:**
   ```python
   timeout_per_replicate=120  # 2 minutes per replicate
   ```

4. **Check for failed replicates:**
   ```python
   print(f"Successful replicates: {bootstrap_result.n_iterations}/100")
   # Should be close to requested number
   ```

### All Bootstrap Replicates Failing

**Problem:** `bootstrap_result` is `None`

**Solutions:**

1. **Check original fit quality:**
   ```python
   print(f"Original R²: {fit_result.r_squared}")
   # Should be > 0.5 for good bootstrap
   ```

2. **Widen bounds:**
   ```python
   parameter_bounds={
       'Ea': (5e3, 1000e3),  # Default Ea range (akts.utils.EA_BOUNDS)
       'A': (1e5, 1e20)
   }
   ```

3. **Increase replicate timeout:**
   ```python
   timeout_per_replicate=180  # 3 minutes
   ```

## Prediction Issues

### Prediction Returns NaN or Inf

**Problem:** `prediction.conversion` contains NaN or Inf values

**Solutions:**

1. **Check temperature program:**
   ```python
   import matplotlib.pyplot as plt
   plt.plot(time_s, temp_K)
   plt.xlabel('Time (s)')
   plt.ylabel('Temperature (K)')
   plt.show()
   # Should be smooth, no discontinuities
   ```

2. **Check for extreme temperatures:**
   - Very high temp causes overflow
   - Very low temp causes underflow
   - Stay within ±50K of fit data range

3. **Reduce extrapolation distance:**
   ```python
   predict=(1, 'year')  # Instead of (10, 'year')
   ```

### Confidence Bands Invisible in Plots

**Problem:** Fit plots show no band at low-conversion temperatures.

**Explanation:** A refrigerated series that reaches 0.5% conversion can have a band only about 0.2 percentage points wide, which disappears on a shared 0-100% axis. The report's fit plots (interactive and static) draw one panel per temperature with its own y-scale and a time axis in days, so these bands are visible there. For your own figures, plot each temperature separately.

### Data Points Fall Outside the Confidence Band

**Problem:** Many measured points lie outside the fitted CI.

**Explanation:** The 95% CI covers uncertainty of the fitted curve only, so data scatter outside it is expected. The lighter 95% prediction interval (PI) drawn behind the CI adds each temperature's residual scatter (half-width `t·sqrt(se_fit² + s_i²)`, where s_i² is that dataset's residual variance). About 95% of points should fall inside the PI. The PI is stored on the fit as `fit_result.conversion_simulated_pi`; `plot_fit_overlay(..., show_pi=True)` draws it (default).

If far fewer than 95% of points fall in the PI, the model does not describe the data.

### Confidence Bands Jump to 100% or Are Missing

**Problem:** Bands reach 100% conversion, or a low-conversion prediction has no CI.

**Explanation and checks:**
- Bootstrap refits whose RSS exceeds 10x the median replicate RSS are excluded as non-converged; a warning reports how many. These used to push bands to 100%.
- Prediction CIs drop degenerate replicates whose final conversion is below min(1%, 10% of the main prediction's final value). A warning such as `Filtered 89/89 degenerate bootstrap samples` means the whole prediction is close to that threshold; low-conversion predictions otherwise keep their CI.

### One-Sided vs Two-Sided Bands

Fit, prediction and simulation plots use `ci_type='two-sided'` (default): the 2.5th-97.5th bootstrap percentiles at 95%. `ci_type='one-sided'` gives (MLE curve, 95th percentile). The ICH Q1E bootstrap shelf-life always uses a separately computed one-sided 95th-percentile bound, whatever `ci_type` is. The regression-based ICH Q1E shelf-life plot shows a two-sided 90% band; its edges are the one-sided 95% limits, so the legend reads "One-sided 95% confidence limits".

### Prediction Confidence Intervals Too Wide

**Problem:** Bootstrap confidence bands are huge

**Explanation:** High uncertainty in parameters

**Solutions:**

1. **More data:**
   - Add more time points
   - Add more temperatures
   - Extend experiments longer

2. **Better quality data:**
   - Reduce measurement noise
   - Use more precise instruments
   - Improve sampling consistency

3. **Increase bootstrap iterations:**
   ```python
   bootstrap_iterations=200  # More samples
   ```

## Report Generation Issues

### FileNotFoundError when creating report

**Problem:** Can't write HTML report

**Solution:** AKTS creates the report's parent directory automatically. You can also create it explicitly:
```python
from pathlib import Path
report_path = Path('output/stability_report.html')
report_path.parent.mkdir(parents=True, exist_ok=True)
```

### Report Opens Blank Page

**Problem:** HTML report shows nothing

**Solutions:**

1. **Check file size:**
   ```bash
   ls -lh stability_report.html
   # Should be >100 KB
   ```

2. **Open in different browser:**
   - Try Chrome, Firefox, or Edge
   - Safari may have security restrictions

3. **Check JavaScript console:**
   - Open browser dev tools (F12)
   - Look for errors in console
   - Plotly errors indicate data format issues

### Report Plots Not Interactive

**Problem:** Can't zoom or hover on plots

**Solution:** Ensure Plotly format:
```python
report_format='interactive'  # Not 'static'
```

Plotly is embedded in interactive reports, so viewing them does not require an internet connection.

## Temperature Unit Issues

### Results Don't Match Expected Values

**Problem:** Parameters seem wrong by orders of magnitude

**Solution:** Check temperature units:
```python
# Data in Celsius but forgot to specify
input_temperature_units='C'  # NOT 'K'!

# Example:
# 25°C = 298 K
# Treating 25 degrees Celsius as 25 kelvin produces invalid results.
```

### Temperature Conversion Errors

**Problem:** Converted temperatures are wrong

**Solutions:**

1. **Use correct unit strings:**
   ```python
   # Valid:
   'K', 'k', 'kelvin', 'Kelvin'
   'C', 'c', 'celsius', 'Celsius'
   'F', 'f', 'fahrenheit', 'Fahrenheit'

   # Invalid:
   'degC', '°C', 'deg_C'
   ```

2. **Verify conversions:**
   ```python
   # 25°C should be 298.15 K
   # 77°F should be 298.15 K
   ```

## JSON I/O Issues

### JSON Serialization Errors

**Problem:** `TypeError: Object of type 'ndarray' is not JSON serializable`

**Solution:** Use built-in serialization:
```python
results_json = auto_model_isothermal_data(
    ...,
    output_format='json'  # Returns JSON string
)

# Or manually:
from akts.json_utils import serialize_results_to_json
json_string = serialize_results_to_json(results)
```

### numpy.int64 or numpy.float64 Errors

**Problem:** JSON doesn't accept NumPy types

**Solution:** The AKTS JSON serializer converts NumPy values to native Python types. If the error occurs when serializing a custom result:
```python
from akts.json_utils import convert_numpy_to_python
clean_dict = convert_numpy_to_python(results)
```

## Performance Issues

### Analysis Very Slow

**Problem:** Takes hours to complete

**Solutions:**

1. **Disable ODE models:**
   ```python
   models_to_try=models.default  # Omit slower ODE candidates
   ```
   `models.all` includes SB2 and the 136-model SB2 grid (173 models in total).
   The SB2 grid alone adds roughly 10-15 minutes; list `models.kinetic.SB2`
   explicitly instead if you only need the two-step model.

2. **Reduce bootstrap iterations:**
   ```python
   bootstrap_iterations=50  # Default 100
   ```

3. **Use fewer models:**
   ```python
   models_to_try=['F1', 'F2', 'A2']  # Top 3 most common
   ```

4. **Reduce data points:**
   ```python
   # Downsample if you have many points
   df_downsampled = df[::2]  # Every other point
   ```

5. **Check multistart fitting is being used for ODE models:**
   ```python
   # Should see this in output:
   # "Using multistart local fitting (faster for ODE models)..."
   ```

### High Memory Usage

**Problem:** Python using too much RAM

**Solutions:**

1. **Reduce bootstrap iterations:**
   ```python
   n_iterations=50
   ```

2. **Reduce n_jobs:**
   ```python
   n_jobs=2  # Instead of -1 (all cores)
   ```

3. **Process datasets separately:**
   ```python
   # Instead of all at once
   for data_file in data_files:
       results = auto_model_isothermal_data(
           data_files=[data_file],
           ...
       )
   ```

## Warning Messages

### OptimizeWarning: Unknown solver options

**Problem:** Warnings during optimization

**Solution:** Check that your SciPy version supports the optimizer options in use. If the warning persists:
```python
# Update to latest version
git pull
pip install -e .
```

### RuntimeWarning: divide by zero

**Problem:** Division by zero in calculations

**Solutions:**

1. **Check for zero conversion:**
   ```python
   # Ensure data starts at conversion = 0
   # Not exactly 0.0000 (numerical precision)
   initial_conversion = 1e-6  # Small non-zero
   ```

2. **Check for zero rates:**
   - Data should show continuous reaction
   - No flat plateaus at beginning

### UserWarning: Dataset too short for rate calc

**Problem:** Not enough points for numerical differentiation

**Solution:** Need more data points:
```python
# Minimum 5 points per dataset
# Recommended 8-12 points
```

## Common Mistakes

### 1. Wrong Time Units

**Problem:**
```python
# Data is in hours, but you treat as minutes
time = [0, 1, 2, 3]  # hours
# But loader assumes minutes by default!
```

**Solution:**
```python
# Convert to minutes
df['Time (min)'] = df['Time (h)'] * 60
```

### 2. Temperature Not Constant

**Problem:**
```python
# Using "isothermal" analysis on ramping data
# Temperature changes during experiment
```

**Solution:**
- Use DSC/TGA analysis for ramping data
- Isothermal analysis requires constant T (±1K)

### 3. Mixing Data Types

**Problem:**
```python
# Mixing isothermal and ramping data
datasets = [isothermal_dataset, dsc_dataset]  # Wrong!
```

**Solution:**
- Keep isothermal and non-isothermal separate
- Use appropriate analysis for each type

### 4. Forgetting Windows Multiprocessing Guard

**Problem:**
```python
# No if __name__ == '__main__'
bootstrap_result = run_bootstrap(...)  # Fails on Windows!
```

**Solution:**
```python
if __name__ == '__main__':
    bootstrap_result = run_bootstrap(...)
```

### 5. Widening `max_seconds` unnecessarily for simple models

**Problem:**
```python
# Setting a large optimizer_options={'max_seconds': ...} budget for a fast
# single_step (F1, F2, ...) fit -- unnecessary, since these fits are already fast
```

**Solution:**
- The default wall-clock budget (`60s * n_parameters`) only matters in practice for
  multi-parameter ODE models (`A->B->C`, `A+B->C`)
- `single_step` models have 2 parameters and converge quickly; there's no need to
  tune `max_seconds` for them

## Getting Help

### Check Documentation

- [Getting Started](getting_started.md) - Installation and first steps
- [API Reference](api_reference.md) - Function signatures
- [Examples](examples.md) - Working code

### Enable Debugging

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# More verbose output
fit_result = fit_kinetic_model(..., verbose=True)
```

### Run the Regression Test Suite

akts ships a pytest-based regression test suite under `tests/`. Running it is a
good first check when something behaves unexpectedly — if the suite fails on a
clean checkout, the problem is likely environmental (Python/dependency versions)
rather than in your own code:

```bash
pip install -e ".[test]"   # installs pytest as a dev dependency
pytest tests/
```

### Report Issues

If you found a bug:

1. **Check if already reported:** [GitHub Issues](https://github.com/PaulNobrega/akts/issues)

2. **Provide minimal example:**
   ```python
   # Minimal code that reproduces the problem
   from akts import auto_model_isothermal_data
   results = auto_model_isothermal_data(...)
   # Error occurs here
   ```

3. **Include:**
   - Python version: `python --version`
   - akts version: `git log -1 --oneline`
   - Operating system
   - Full error traceback
   - Sample data (if possible)

### Community

- **GitHub Discussions:** Ask questions
- **Issues:** Report bugs
- **Pull Requests:** Contribute fixes

## Quick Diagnostic Checklist

When something goes wrong:

- [ ] Python 3.8+ installed?
- [ ] All dependencies installed? (`pip install -r requirements.txt`)
- [ ] Virtual environment activated?
- [ ] Data files exist at specified paths?
- [ ] Temperature units specified correctly?
- [ ] Conversion values in [0, 1] range?
- [ ] At least 6-8 data points per temperature?
- [ ] At least 3 temperatures?
- [ ] Windows: Used `if __name__ == '__main__':` for bootstrap?
- [ ] Latest version? (`git pull`, `pip install -e .`)

---

**Still having issues?** [Open an issue on GitHub](https://github.com/PaulNobrega/akts/issues) with:
- Your code
- Error message
- Python/OS version
- Sample data (if possible)
