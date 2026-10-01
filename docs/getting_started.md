# Getting Started with AKTS

This guide covers installation and a first kinetic analysis with AKTS.

## Prerequisites

- Python 3.8 or higher
- Basic familiarity with Python and command line
- Kinetic data from DSC, TGA, or isothermal stability experiments

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/PaulNobrega/akts.git
cd akts
```

### 2. Create Virtual Environment (Recommended)

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS/Linux:**
```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

All dependencies are required (no optional packages):
- NumPy ≥1.20.0
- SciPy ≥1.7.0
- Matplotlib ≥3.4.0
- Plotly ≥5.0.0
- pandas ≥1.3.0
- openpyxl ≥3.0.0

### 4. Install AKTS in Development Mode

```bash
pip install -e .
```

This installs akts so you can import it from anywhere while still being able to edit the source code.

### 5. Verify Installation

```python
python -c "import akts; print('AKTS installed successfully!')"
```

## Your First Analysis

### Automated Analysis

Use `auto_model_isothermal_data()` for an automated analysis:

```python
from akts import auto_model_isothermal_data

# Analyze stability data with one function
results = auto_model_isothermal_data(
    data_files=['data_25C.csv', 'data_40C.csv', 'data_60C.csv'],
    predict=(2, 'year'),  # Predict 2 years ahead
    report_path='stability_report.html'
)

# View results
print(f"Best model: {results['selected_model']['model_name']}")
print(f"Activation energy: {results['selected_model']['parameters']['Ea']/1000:.1f} kJ/mol")
print(f"R² = {results['selected_model']['statistics']['r_squared']:.4f}")

# Open the HTML report in your browser
import webbrowser
webbrowser.open('stability_report.html')
```

**What this does:**
1. Loads your CSV files (keeping all replicate points) and scales every file to one shared conversion range
2. Tries 12 common kinetic models
3. Filters models by R² ≥ 0.70 and physical plausibility
4. Ranks survivors by Akaike weights (probability each is best)
5. Selects the top-ranked model
6. Generates predictions with two-sided 95% bootstrap confidence intervals
7. Creates an interactive HTML report with plots

[Automated analysis guide](automated_analysis.md)

### Option 2: Step-by-Step Manual Analysis

For more control over the analysis:

```python
import numpy as np
from akts import KineticDataset, fit_kinetic_model, run_bootstrap, predict_conversion
from akts.loaders import load_isothermal_file

# 1. Load your data
dataset_313K = load_isothermal_file(
    'protein_stability_313K.csv',
    time_col='Time (days)',
    temperature_col='Temperature (K)',
    readout_col='HMW Species (%)',
    readout_type='increasing'  # HMW increases with degradation
)

dataset_323K = load_isothermal_file('protein_stability_323K.csv', ...)
dataset_333K = load_isothermal_file('protein_stability_333K.csv', ...)

datasets = [dataset_313K, dataset_323K, dataset_333K]

# 2. Fit a kinetic model
fit_result = fit_kinetic_model(
    datasets=datasets,
    model_name='single_step',
    model_definition_args={'f_alpha_model': 'F1'},  # First-order kinetics
    initial_guesses={'Ea': 85000, 'A': 1e11},  # J/mol, 1/s
    parameter_bounds={
        'Ea': (50000, 150000),  # Reasonable for proteins
        'A': (1e7, 1e14)
    }
)

print(f"Fit successful: {fit_result.success}")
print(f"Ea = {fit_result.parameters['Ea']/1000:.1f} kJ/mol")
print(f"A = {fit_result.parameters['A']:.2e} 1/s")
print(f"R² = {fit_result.r_squared:.4f}")

# 3. Get confidence intervals via bootstrap
if __name__ == '__main__':  # Required on Windows
    bootstrap_result = run_bootstrap(
        datasets=datasets,
        fit_result=fit_result,
        parameter_bounds={'Ea': (50000, 150000), 'A': (1e7, 1e14)},
        n_iterations=100,
        n_jobs=-1  # Use all CPU cores
    )

    print("\n95% Confidence Intervals:")
    for param, (lo, hi) in bootstrap_result.parameter_ci.items():
        print(f"  {param}: ({lo:.3g}, {hi:.3g})")

# 4. Predict long-term stability
from akts.utils import construct_profile

# Storage profile: 2 years at 5°C
segments = [
    {'type': 'isothermal', 'duration': 2 * 365 * 24 * 3600, 'temperature': 278.15}
]
time_s, temp_K = construct_profile(segments, points_per_segment=100)

prediction = predict_conversion(
    kinetic_description=fit_result,
    temperature_program=(time_s, temp_K),
    bootstrap_result=bootstrap_result  # Adds confidence intervals
)

print(f"\nPredicted conversion after 2 years at 5°C: {prediction.conversion[-1]:.1%}")
if prediction.conversion_ci:
    lower, upper = prediction.conversion_ci
    print(f"95% CI: ({lower[-1]:.1%}, {upper[-1]:.1%})")
```

[API reference](api_reference.md)

## Data Format Requirements

### CSV File Format

Your CSV files should have columns for:
- **Time** - Numeric values (days, hours, minutes, seconds)
- **Temperature** - Numeric values in K, °C, or °F
- **Readout** - Your measurement (%, mass, signal, etc.)

Example CSV:
```csv
Time (days),Temperature (K),HMW Species (%)
0,313.15,0.0
1,313.15,2.3
3,313.15,5.8
7,313.15,11.2
14,313.15,18.9
```

**Column names are flexible** - the loader will try to auto-detect them, or you can specify:

```python
dataset = load_isothermal_file(
    'data.csv',
    time_col='Time (days)',
    temperature_col='Temp (C)',
    readout_col='Degradation (%)',
    input_temperature_units='C'  # Convert to Kelvin automatically
)
```

### Temperature Units

AKTS works internally in Kelvin but can accept input in any unit:

```python
results = auto_model_isothermal_data(
    data_files=['data_25C.csv', 'data_40C.csv'],
    input_temperature_units='C',  # Input is Celsius
    output_temperature_units='F'  # Report in Fahrenheit
)
```

[Temperature units guide](temperature_units.md)

## Running the Examples

The repository includes working examples with real data:

```bash
# Isothermal stability example
cd Examples/Isothermal
python demo_auto_model.py

# DSC example
cd Examples/DSC
python demo_dsc_analysis.py

# TGA example
cd Examples/TGA
python demo_tga_analysis.py
```

Each example generates an HTML report in the `output/` directory.

[Examples guide](examples.md)

## Common Workflows

### Pharmaceutical Stability Study

```python
from akts import auto_model_isothermal_data

# Accelerated stability data at 3 temperatures
results = auto_model_isothermal_data(
    data_files=[
        'stability_25C.csv',  # 25°C/60% RH (long-term)
        'stability_40C.csv',  # 40°C/75% RH (accelerated)
        'stability_60C.csv'   # 60°C (stress)
    ],
    predict=(24, 'month'),  # Predict 2 years
    input_temperature_units='C',
    report_path='shelf_life_report.html',
    bootstrap_iterations=100
)

print(f"Predicted degradation at 24 months (25°C): {results['predictions']['conversion_mean'][-1]:.1%}")
```

### Polymer Thermal Stability

```python
# DSC data at multiple heating rates
results = auto_model_dsc_data(
    data_files=[
        'polymer_5Kmin.csv',
        'polymer_10Kmin.csv',
        'polymer_20Kmin.csv'
    ],
    heating_rates=[5, 10, 20],  # K/min
    report_path='thermal_stability.html'
)
```

### Protein Formulation Screening

```python
# Compare multiple formulations
formulations = ['pH6', 'pH7', 'pH8', 'with_stabilizer']

for formulation in formulations:
    results = auto_model_isothermal_data(
        data_files=[
            f'{formulation}_40C.csv',
            f'{formulation}_50C.csv'
        ],
        predict=(12, 'month'),
        report_path=f'formulation_{formulation}.html'
    )

    print(f"{formulation}: Ea = {results['selected_model']['parameters']['Ea']/1000:.1f} kJ/mol")
```

## Next Steps

### Learn More

- **[Automated Analysis Guide](automated_analysis.md)** - Master the one-function workflow
- **[Kinetic Models Guide](kinetic_models.md)** - Understand the models
- **[Model Selection Guide](model_selection_guide.md)** - Choose the right model
- **[Experimental Design](experimental_design.md)** - Design better experiments

### Advanced Topics

- **[API Reference](api_reference.md)** - Complete function reference
- **[Advanced Usage](advanced_usage.md)** - Custom workflows
- **[Multistart Fitting](bayesian_optimization.md)** - Reliable ODE fitting
- **[JSON I/O](json_io_specification.md)** - Web API integration

### Get Help

- **[Troubleshooting Guide](troubleshooting.md)** - Common issues
- **[GitHub Issues](https://github.com/PaulNobrega/akts/issues)** - Report bugs
- **[Examples](examples.md)** - Working code with real data

## Recommendations

### 1. Start Simple

Begin with `auto_model_isothermal_data()` and use the manual workflow when you need more control.

### 2. Check Your Data

Before fitting:
- Plot your raw data to check for outliers
- Ensure temperatures are isothermal or linear ramps
- Verify conversion is in [0, 1] range
- Check that time units are consistent

### 3. Use Multiple Temperatures

For reliable kinetic parameters:
- **Minimum:** 3 temperatures spanning 20-30°C
- **Recommended:** 4-5 temperatures spanning 30-50°C
- Include your target storage temperature for validation

### 4. Bootstrap for Confidence

Always run bootstrap to get confidence intervals:
```python
bootstrap_iterations=100  # Standard
bootstrap_iterations=200  # Better accuracy
```

### 5. Inspect the HTML Report

The HTML report shows:
- Model comparison table
- Fit plots, one panel per temperature, with a 95% confidence interval (curve uncertainty) and a wider 95% prediction interval (where about 95% of data points should fall)
- Arrhenius plot
- Predictions with confidence bands
- Parameter uncertainties

Review it to ensure the fit makes sense!

### 6. Temperature Units

Work in whatever units are convenient:
```python
input_temperature_units='C'   # Celsius
output_temperature_units='F'  # Report in Fahrenheit
```

### 7. ODE Models Optional

ODE models (A→B→C, A+B→C) and the two-step Sestak-Berggren models (SB2, SB2 grid) are powerful but slow. Enable only if simple models don't fit:

```python
models_to_try=models.default + models.ode.all      # Include ODE models
models_to_try=models.default + models.kinetic.SB2  # Two-step SB, fitted orders
```

`models.all` includes the 136-model `models.kinetic.SB2_grid` and can take 10-15 minutes longer.

## Troubleshooting

### Import Error

**Problem:** `ModuleNotFoundError: No module named 'akts'`

**Solution:** Make sure you installed with `pip install -e .` from the akts directory.

### Fit Failure

**Problem:** All models show `success=False`

**Solutions:**
- Check data quality (outliers, noise)
- Verify temperature units
- Try wider parameter bounds
- Ensure sufficient data points (minimum 6-8 per temperature)

### Slow Performance

**Problem:** Analysis taking too long

**Solutions:**
- Use `models.default` in `models_to_try` to omit ODE and SB2 models (avoid `models.all` unless needed).
- Reduce bootstrap iterations: `bootstrap_iterations=50`
- Use fewer models: `models_to_try=['F1', 'F2', 'A2']`

[Troubleshooting guide](troubleshooting.md)

## Quick Reference Card

### Import Essentials
```python
from akts import (
    auto_model_isothermal_data,  # One-function workflow
    models,                       # Model selector
    KineticDataset,              # Data container
    fit_kinetic_model,           # Fit one model
    run_bootstrap,               # Confidence intervals
    predict_conversion,          # Extrapolation
    discover_kinetic_models      # Try multiple models
)
from akts.loaders import (
    load_isothermal_file,        # Load stability data
    load_dsc_file,               # Load DSC data
    load_tga_file                # Load TGA data
)
```

### Key Parameters
```python
# Temperature units
input_temperature_units='K'   # 'K', 'C', or 'F'
output_temperature_units='K'

# Model selection
models_to_try=models.kinetic.F1 + models.kinetic.F2  # Specific models
models_to_try=models.default + models.ode.all         # Add ODE models

# Bootstrap
bootstrap_iterations=100   # Confidence intervals
confidence_level=0.95      # 95% CI

# Prediction
predict=(2, 'year')        # Extrapolate 2 years
predict=(6, 'month')       # Extrapolate 6 months
```

### Common Models
- **F0** - Zero-order (constant rate, e.g. controlled release)
- **F1** - First-order reaction (most common)
- **F2** - Second-order reaction
- **A2, A3** - Nucleation and growth
- **D2, D3** - Diffusion-controlled
- **R2, R3** - Contracting geometry
- **SB_mn** - Sestak-Berggren (general autocatalytic)
- **SB2** - Two parallel Sestak-Berggren steps (ODE)
- **Bna** - Prout-Tompkins (autocatalytic)
- **A→B→C** - Consecutive reactions (ODE)
- **A+B→C** - Bimolecular reaction (ODE)

---

For an automated workflow, see the [automated analysis guide](automated_analysis.md) or run an [example](examples.md).
