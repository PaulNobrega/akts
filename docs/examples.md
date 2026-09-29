# Examples Guide

Complete guide to the example scripts and data included with akts.

## Overview

The `Examples/` directory contains working examples with real data demonstrating typical workflows for:

- **Isothermal** - Protein stability studies, accelerated aging
- **DSC** - Differential scanning calorimetry (curing, decomposition)
- **TGA** - Thermogravimetric analysis (thermal degradation)

All examples generate interactive HTML reports in the `output/` subdirectory.

## Directory Structure

```
Examples/
├── Isothermal/
│   ├── demo_auto_model.py           # Automated analysis (recommended)
│   ├── demo_manual_analysis.py       # Step-by-step workflow
│   ├── protein_stability_313K.csv    # 40°C data
│   ├── protein_stability_323K.csv    # 50°C data
│   ├── protein_stability_333K.csv    # 60°C data
│   ├── protein_stability_343K.csv    # 70°C data
│   ├── output/                       # Generated reports
│   └── README.md                     # Dataset description
│
├── DSC/
│   ├── demo_dsc_analysis.py
│   ├── epoxy_cure_5Kmin.csv
│   ├── epoxy_cure_10Kmin.csv
│   ├── epoxy_cure_20Kmin.csv
│   ├── output/
│   └── README.md
│
└── TGA/
    ├── demo_tga_analysis.py
    ├── polymer_decomp_5Kmin.csv
    ├── polymer_decomp_10Kmin.csv
    ├── polymer_decomp_20Kmin.csv
    ├── output/
    └── README.md
```

## Isothermal Examples

### 1. Automated Analysis (Recommended)

**File:** `Examples/Isothermal/demo_auto_model.py`

**What it does:**
- Loads 4 protein stability datasets (40°C, 50°C, 60°C, 70°C)
- Includes ODE models (A→B→C, A+B→C) in the candidate list
- Uses multistart local fitting for reliable ODE fitting
- Generates predictions for 2-year storage at 5°C
- Creates interactive HTML report

**Run it:**
```bash
cd Examples/Isothermal
python demo_auto_model.py
```

**Key code:**
```python
from akts import auto_model_isothermal_data, models

results = auto_model_isothermal_data(
    data_files=[
        'protein_stability_313K.csv',  # 40°C
        'protein_stability_323K.csv',  # 50°C
        'protein_stability_333K.csv',  # 60°C
        'protein_stability_343K.csv'   # 70°C
    ],
    predict=(2, 'year'),
    input_temperature_units='K',
    output_temperature_units='C',
    models_to_try=models.default + models.ode.all,
    bootstrap_iterations=100,
    report_path='output/isothermal_stability_report.html',
    progress_callback=progress
)
```

**Output:**
- Console progress updates with timestamps
- `output/isothermal_stability_report.html` - Interactive report
- Model comparison table, fit plots, Arrhenius plot, predictions

**Expected runtime:** ~7-8 minutes (includes ODE models with multistart fitting)

### 2. Manual Step-by-Step Analysis

**File:** `Examples/Isothermal/demo_manual_analysis.py`

**What it does:**
- Demonstrates full control workflow
- Shows how to fit individual models
- Runs bootstrap for confidence intervals
- Creates custom temperature programs
- Generates predictions with uncertainty

**Key code:**
```python
# 1. Load data
datasets = [load_isothermal_file(f) for f in data_files]

# 2. Fit model
fit_result = fit_kinetic_model(
    datasets=datasets,
    model_name='single_step',
    model_definition_args={'f_alpha_model': 'F1'},
    initial_guesses={'Ea': 85000, 'A': 1e11}
)

# 3. Bootstrap
bootstrap_result = run_bootstrap(
    datasets=datasets,
    fit_result=fit_result,
    n_iterations=100
)

# 4. Predict
prediction = predict_conversion(
    kinetic_description=fit_result,
    temperature_program=(time_s, temp_K),
    bootstrap_result=bootstrap_result
)
```

**When to use:** When you need fine control over every step

### Dataset Description

**Source:** Synthetic protein stability data based on typical biologic degradation

**Experimental conditions:**
- 4 temperatures: 313K (40°C), 323K (50°C), 333K (60°C), 343K (70°C)
- Measured parameter: High molecular weight (HMW) species %
- Time points: 0, 1, 3, 7, 14, 28 days
- Readout increases with degradation

**True kinetic parameters** (used to generate data):
- Ea = 95 kJ/mol
- A = 1.2×10¹² s⁻¹
- Mechanism: First-order (F1)

**Use case:** Accelerated stability study for shelf-life prediction

## DSC Examples

**File:** `Examples/DSC/demo_dsc_analysis.py`

**What it does:**
- Loads DSC data at 3 heating rates (5, 10, 20 K/min)
- Analyzes epoxy resin curing kinetics
- Fits activation energy from peak shift
- Compares models

**Dataset:** Epoxy cure exotherm
- Heating rates: 5, 10, 20 K/min
- Temperature range: 25-250°C
- Measurement: Heat flow (exothermic curing reaction)

**Run it:**
```bash
cd Examples/DSC
python demo_dsc_analysis.py
```

**Key concepts:**
- Multi-rate DSC requires ≥3 heating rates
- Isoconversional methods (Friedman, KAS, OFW) extract Ea(α)
- Model-based fitting finds best kinetic model

## TGA Examples

**File:** `Examples/TGA/demo_tga_analysis.py`

**What it does:**
- Loads TGA data at 3 heating rates
- Analyzes polymer thermal decomposition
- Determines degradation mechanism
- Predicts service lifetime

**Dataset:** Polymer decomposition
- Heating rates: 5, 10, 20 K/min
- Temperature range: 200-600°C
- Measurement: Mass loss (%)

**Run it:**
```bash
cd Examples/TGA
python demo_tga_analysis.py
```

**Key concepts:**
- TGA tracks mass loss vs. temperature
- Multi-step decomposition may require A→B→C model
- Service lifetime predicted from degradation onset

## Data File Formats

All example CSV files follow this structure:

### Isothermal Data
```csv
Time (days),Temperature (K),HMW Species (%)
0,313.15,0.0
1,313.15,2.3
3,313.15,5.8
7,313.15,11.2
14,313.15,18.9
28,313.15,28.5
```

### DSC Data
```csv
Time (min),Temperature (C),Heat Flow (W/g)
0.0,25.0,0.00
2.0,35.0,0.05
4.0,45.0,0.15
...
```

### TGA Data
```csv
Time (min),Temperature (C),Mass (%)
0.0,200.0,100.0
5.0,225.0,98.5
10.0,250.0,95.2
...
```

## Customizing Examples

### Change Temperature Units

```python
# Input data in Celsius
results = auto_model_isothermal_data(
    ...,
    input_temperature_units='C',  # Data is in Celsius
    output_temperature_units='F'  # Report in Fahrenheit
)
```

### Try Different Models

```python
# Test only specific models
results = auto_model_isothermal_data(
    ...,
    models_to_try=['F1', 'F2', 'A2']  # Select only these models
)
```

### Adjust Prediction Time

```python
# Predict different durations
predict=(6, 'month')   # 6 months
predict=(5, 'year')    # 5 years
predict=(10, 'week')   # 10 weeks
```

### Custom Temperature Profile

```python
from akts.utils import construct_profile

# Complex storage profile
segments = [
    {'type': 'isothermal', 'duration': 30*24*3600, 'temperature': 278},  # 30 days at 5°C
    {'type': 'ramp', 'duration': 3600, 'start_temp': 278, 'end_temp': 298},  # Ramp to 25°C
    {'type': 'isothermal', 'duration': 7*24*3600, 'temperature': 298},  # 7 days at 25°C
    {'type': 'ramp', 'duration': 3600, 'start_temp': 298, 'end_temp': 278},  # Back to 5°C
    {'type': 'isothermal', 'duration': 365*24*3600, 'temperature': 278}  # 1 year at 5°C
]
time_s, temp_K = construct_profile(segments)
```

### Change Bootstrap Settings

```python
results = auto_model_isothermal_data(
    ...,
    bootstrap_iterations=200,  # More iterations = better CI
    confidence_level=0.99      # 99% confidence intervals
)
```

## Using Your Own Data

### 1. Prepare CSV Files

Your CSV should have 3 columns:
- Time (any unit, specify in loader)
- Temperature (K, °C, or °F)
- Readout (%, mass, intensity, etc.)

### 2. Create Analysis Script

```python
from akts import auto_model_isothermal_data

results = auto_model_isothermal_data(
    data_files=[
        'your_data_temp1.csv',
        'your_data_temp2.csv',
        'your_data_temp3.csv'
    ],
    time_col='Time (h)',  # Your column name
    temperature_col='Temp (C)',
    readout_col='% Intact',
    readout_type='decreasing',  # % intact decreases
    input_temperature_units='C',
    predict=(2, 'year'),
    report_path='your_report.html'
)
```

### 3. Run and Review

```bash
python your_analysis.py
```

Open `your_report.html` to review:
- Model comparison - which fits best?
- Fit quality plots - does it match data?
- Arrhenius plot - linear (good) or curved (complex mechanism)?
- Predictions - reasonable extrapolation?

## Common Modifications

### Add Progress Bar

```python
from tqdm import tqdm

progress_bar = None

def progress(msg, data):
    global progress_bar
    if 'iteration' in data:
        if progress_bar is None:
            progress_bar = tqdm(total=data.get('n_calls', 100))
        progress_bar.update(1)
        progress_bar.set_description(msg)
    else:
        if progress_bar:
            progress_bar.close()
            progress_bar = None
        print(msg)

results = auto_model_isothermal_data(..., progress_callback=progress)
```

### Save Results to JSON

```python
import json

results = auto_model_isothermal_data(..., output_format='json')

with open('results.json', 'w') as f:
    f.write(results)
```

### Batch Process Multiple Datasets

```python
import glob
from pathlib import Path

# Find all CSV files
data_dirs = glob.glob('data/*/')

for data_dir in data_dirs:
    csv_files = list(Path(data_dir).glob('*.csv'))
    name = Path(data_dir).stem

    results = auto_model_isothermal_data(
        data_files=csv_files,
        report_path=f'reports/{name}_report.html'
    )

    print(f"{name}: Ea = {results['selected_model']['parameters']['Ea']/1000:.1f} kJ/mol")
```

## Expected Output

### Console Output

```
================================================================================
 AKTS: Automated Isothermal Kinetic Modeling Demo
================================================================================

Found 4 data files:
  [OK] protein_stability_313K.csv
  [OK] protein_stability_323K.csv
  [OK] protein_stability_333K.csv
  [OK] protein_stability_343K.csv

[20:52:19] Loading data files...
[20:52:19] Loaded dataset 1/4: protein_stability_313K.csv
[20:52:19] Loaded dataset 2/4: protein_stability_323K.csv
...
[20:52:23] Fitting kinetic models (this may take a while)...
[20:52:23] Fitting F1 (first-order)...
[20:52:27] F1 (first-order) fit successful (R²=-0.3633)
...
[20:52:37] Fitting A->B->C (consecutive reactions)...
[20:52:37]   Using multistart local fitting (faster for ODE models)...
[20:52:41]   Multistart: 1/4 R²=0.9723 * new best [4.1s]
[20:52:45]   Multistart: 2/4 R²=0.9872 * new best [3.9s]
[20:52:49]   Multistart: 3/4 R²=0.9801 (best 0.9872) [4.0s]
[20:52:53]   Multistart: 4/4 R²=0.9756 (best 0.9872) [3.8s]
...
[20:55:30] Generating HTML report...
[20:55:32] Completed. Report saved to: output/isothermal_stability_report.html
```

### HTML Report Contents

1. **Executive Summary**
   - Selected model and parameters
   - Activation energy, pre-exponential factor
   - Fit statistics (R², AIC, BIC)

2. **Model Comparison Table**
   - All models ranked by score
   - Statistical metrics for each

3. **Fit Quality Plots**
   - Experimental data vs. model fit
   - One plot per temperature
   - Residuals shown below

4. **Arrhenius Plot**
   - ln(k) vs. 1/T
   - Shows temperature dependence
   - Linear = simple Arrhenius behavior

5. **Prediction Plot**
   - Extrapolated conversion
   - Confidence intervals (shaded)
   - Extended time axis

6. **Parameter Details**
   - Best-fit values
   - Bootstrap confidence intervals
   - Correlation matrix

## Troubleshooting Examples

### Example Won't Run

**Check:**
1. Python version: `python --version` (need 3.8+)
2. Dependencies: `pip install -r requirements.txt`
3. Working directory: `cd Examples/Isothermal`

### FileNotFoundError

**Solution:**
```python
from pathlib import Path
script_dir = Path(__file__).parent
data_file = script_dir / 'protein_stability_313K.csv'
```

### Example Takes Too Long

**Speed it up:**
```python
models_to_try=models.default  # Use default kinetic models without ODE models
bootstrap_iterations=50   # Fewer iterations
```

## Next Steps

After running the examples:

1. **Understand the models** - [Kinetic Models Guide](kinetic_models.md)
2. **Choose the right model** - [Model Selection Guide](model_selection_guide.md)
3. **Design better experiments** - [Experimental Design](experimental_design.md)
4. **Use your own data** - [Getting Started](getting_started.md)
5. **Advanced workflows** - [Advanced Usage](advanced_usage.md)

## Additional Resources

- **[API Reference](api_reference.md)** - Complete function documentation
- **[JSON I/O Specification](json_io_specification.md)** - Web API integration
- **[Troubleshooting](troubleshooting.md)** - Common issues

---

For a first analysis, see the [getting started guide](getting_started.md).
