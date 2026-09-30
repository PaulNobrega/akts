# Isothermal Kinetic Analysis Examples

This directory contains **realistic scientific examples** demonstrating automated isothermal kinetic modeling across various application domains.

## 📚 Example Scripts

### 🧪 Example 1: Pharmaceutical Shelf-Life Regression
**File**: `01_pharmaceutical_shelf_life.py`

**Scientific Context**: Drug product stability study with an automated shelf-life trend assessment

**Key Features**:
- ✅ Automatic linear/quadratic trend selection
- ✅ One-sided 95% confidence intervals
- ✅ Configurable extrapolation ceiling
- ⚠️ Example output is not a determination of regulatory compliance
- ✅ Shelf-life prediction at label storage
- ✅ Model simplicity preference (Occam's Razor)

**Data**: Aspirin tablet degradation (API content loss)
- Storage: 25°C, 30°C, 40°C (ICH conditions)
- Duration: Up to 24 months
- Measurement: API assay by HPLC
- Specification: ≥ 95% API content

**Run**: `python 01_pharmaceutical_shelf_life.py`

---

### 🧬 Example 2: Protein Aggregation (Autocatalytic Kinetics)
**File**: `02_protein_aggregation_autocatalytic.py`

**Scientific Context**: Therapeutic mAb aggregation with nucleation-growth mechanism

**Key Features**:
- ✅ Autocatalytic models (SB_mn, Bna)
- ✅ Temperature excursion simulation (shipping)
- ✅ Model comparison and ranking
- ✅ Custom plotting with helpers
- ✅ HMW species quantification

**Data**: Monoclonal antibody HMW aggregates
- Storage: 5°C, 25°C, 37°C, 45°C
- Duration: Up to 2 years at 5°C
- Measurement: SEC-HPLC
- Specification: ≤ 5% HMW

**Run**: `python 02_protein_aggregation_autocatalytic.py`

---

### 🍊 Example 3: Food Quality (Vitamin Degradation)
**File**: `03_food_quality_vitamin_degradation.py`

**Scientific Context**: Vitamin C stability in orange juice for shelf-life determination

**Key Features**:
- ✅ First-order kinetics
- ✅ Q10 temperature coefficient
- ✅ Multi-temperature shelf-life table
- ✅ Tabulated export (CSV for Excel/R)
- ✅ Publication-ready plots

**Data**: Vitamin C retention in pasteurized juice
- Storage: 4°C, 20°C, 35°C
- Duration: Up to 6 months at 4°C
- Measurement: HPLC vitamin C assay
- Quality target: ≥ 80% retention

**Run**: `python 03_food_quality_vitamin_degradation.py`

---

### 🔬 Example 4: Polymer Oxidation (Complex A→B→C Kinetics)
**File**: `04_polymer_oxidation_complex.py`

**Scientific Context**: Polyethylene thermal oxidation with two-step mechanism

**Key Features**:
- ✅ Complex ODE models (A→B→C)
- ✅ Multistart optimization
- ✅ Mechanism identification
- ✅ Service life prediction (10 years)
- ✅ Arrhenius parameter interpretation

**Data**: PE film carbonyl index growth
- Accelerated aging: 60°C, 80°C, 100°C, 120°C
- Duration: Up to 4000 hours
- Measurement: FTIR carbonyl index
- Failure: ~30% oxidation

**Run**: `python 04_polymer_oxidation_complex.py`

---

### 🎯 Original Demo: General Template
**File**: `demo_auto_model.py`

**Purpose**: Template for custom data analysis

**Features**: Complete workflow demonstration with all features

**Usage**: See detailed instructions in original README section below

---

## 🚀 Quick Start

### Run Any Example

All examples are self-contained and generate synthetic data:

```bash
# Navigate to Examples/Isothermal directory
cd Examples/Isothermal

# Run any example
python 01_pharmaceutical_shelf_life.py
python 02_protein_aggregation_autocatalytic.py
python 03_food_quality_vitamin_degradation.py
python 04_polymer_oxidation_complex.py
```

Each script will:
1. Generate realistic synthetic data
2. Perform automated kinetic analysis
3. Create HTML report in `output/` directory
4. Export publication-ready plots (300 DPI PNG)
5. Print summary to console

### Expected Runtime

- **Examples 1 & 3**: ~30-60 seconds (simple models)
- **Example 2**: ~2-3 minutes (autocatalytic models)
- **Example 4**: ~5-10 minutes (complex ODE models)

Bootstrap iterations can be reduced for faster testing:
```python
bootstrap_iterations=50  # Default is 100
bootstrap_iterations=0   # Skip bootstrap (no CI)
```

---

## 📊 Output Files

All examples save results to `output/` directory:

### HTML Reports
- **Interactive plots** with plotly
- **Model comparison tables**
- **Statistical details**
- Shelf-life trend and extrapolation summary (Example 1)
- **Arrhenius analysis**

### PNG Plots (300 DPI)
- `*_data.png` - Experimental data at all temperatures
- `*_arrhenius.png` - Arrhenius plot (ln(k) vs 1/T)
- `*_fit_overlay.png` - Data vs model fit
- `*_ci_bands.png` - Predictions with confidence intervals

### CSV Data Exports
- `*_prediction.csv` - Time, temperature, conversion, CI
- `*_bootstrap_ci.csv` - Parameter uncertainty summary

---

## 🎓 Learning Path

### Beginner
Start with **Example 3** (Vitamin C):
- Simple first-order kinetics
- Easy to understand Q10 concept
- Clear shelf-life interpretation
- Fast runtime (~30 seconds)

### Intermediate
Move to **Example 1** (Pharmaceutical):
- ICH Q1E regulatory concepts
- One-sided confidence intervals
- Extrapolation ceiling validation
- Professional reporting

### Advanced - Mechanism
Try **Example 2** (Protein):
- Autocatalytic kinetics (SB model)
- Temperature excursion simulation
- Model comparison (simple vs autocatalytic)
- Custom plotting

### Advanced - Complex
Finish with **Example 4** (Polymer):
- A→B→C consecutive reactions
- ODE system solution
- Multistart optimization
- Long-term service life prediction

---

## 🔧 Customization Guide

### Using Your Own Data

Replace data generation function with file loading:

```python
# Instead of generate_*_data():
from akts import auto_model_isothermal_data

results = auto_model_isothermal_data(
    data_files=['your_data_40C.csv', 'your_data_50C.csv'],
    # ... rest of parameters
)
```

### Key Parameters to Adjust

```python
# Temperature units
input_temperature_units='C'   # 'K', 'C', or 'F'
output_temperature_units='C'

# Model selection
models_to_try=['F1', 'F2', 'SB_mn']  # Specific models
include_ode_models=True              # A→B→C, A+B→C

# Prediction
predict=(2, 'year', 25)   # Time, unit, temperature
predict=None              # Skip prediction

# Temperature excursion
simulate=[
    (0, 5),    # Start at 5°C
    (7, 30),   # Heat to 30°C at day 7
    (10, 5),   # Return to 5°C at day 10
]
simulate_time_unit='days'

# Statistical rigor
bootstrap_iterations=100    # CI precision (0-1000)
top_n=5                    # Models to analyze

# Report format
report_format='interactive'  # Plotly plots
report_format='static'       # Matplotlib plots
report_format='both'         # Both formats
```

---

## 📈 Feature Matrix

| Feature | Ex 1 | Ex 2 | Ex 3 | Ex 4 | Demo |
|---------|------|------|------|------|------|
| Shelf-Life Trend Regression | ✅ | — | — | — | — |
| Autocatalytic Models | — | ✅ | — | — | — |
| Temperature Excursion | — | ✅ | — | — | ✅ |
| Q10 Calculation | — | — | ✅ | — | — |
| Tabulated Export | — | — | ✅ | — | — |
| Complex ODE (A→B→C) | — | — | — | ✅ | ✅ |
| Multi-temp Shelf-Life | — | — | ✅ | ✅ | — |
| Custom Plotting | — | ✅ | ✅ | ✅ | ✅ |
| Bootstrap CI | ✅ | ✅ | ✅ | ✅ | ✅ |
| Model Comparison | ✅ | ✅ | ✅ | ✅ | ✅ |

---

## 🧪 Scientific Validation

All examples use realistic kinetic parameters from literature:

| Application | Ea Range | Mechanism | Reference Type |
|-------------|----------|-----------|----------------|
| Aspirin hydrolysis | 70-90 kJ/mol | First-order | USP stability |
| mAb aggregation | 90-110 kJ/mol | Autocatalytic | Biopharmaceutics |
| Vitamin C oxidation | 65-75 kJ/mol | First-order | Food science |
| PE oxidation | 80-120 kJ/mol | Two-step | Polymer degradation |

Synthetic data includes realistic noise levels:
- HPLC: 0.5-2% RSD
- SEC: 1-2% RSD  
- FTIR: 2-3% RSD

---

## 🚨 Common Issues

### "Analysis too slow"
```python
include_ode_models=False      # Skip A→B→C (10× faster)
bootstrap_iterations=50       # Reduce from 100
models_to_try=['F1', 'F2']   # Fewer models
```

### "ODE solver warnings"
**Normal for complex models** - fallback solver (LSODA) will handle it automatically. See console output explanation.

### "Low R² values"
- Check temperature units (must be Kelvin or set `input_temperature_units`)
- Verify data quality (no NaN, reasonable ranges)
- Try different models: `models_to_try=['F0', 'F1', 'F2', 'A2', 'SB_mn']`

### "Bootstrap failures"
- Some replicates may fail (normal if >80% succeed)
- Reduce iterations if too many fail: `bootstrap_iterations=50`
- Check parameter bounds in error messages

---

## 📚 Documentation

- **API Reference**: See `docs/api_reference.md`
- **Model Equations**: See `docs/kinetic_models.md`
- **Advanced Guide**: See `docs/advanced_usage.md`
- **ICH Q1E Details**: See `PHASE5_ICH_Q1E_SUMMARY.md`

---

## 💡 Tips for Success

1. **Start simple**: Try Example 3 first (fastest, clearest)
2. **Check units**: Temperature must be Kelvin or specify `input_temperature_units`
3. **Inspect HTML**: Interactive plots reveal data quality issues
4. **Read warnings**: ODE solver warnings are usually OK (fallback works)
5. **Bootstrap CI**: Use ≥100 iterations for publication
6. **Model selection**: Trust BIC ranking (penalizes complexity)

---

## 🎯 Using for Your Research

These examples demonstrate best practices for:

**Pharmaceutical**:
- Regression-based shelf-life exploration
- Configurable storage condition and specification threshold
- Review trend assumptions and applicable regulations before submission

**Biotechnology**:
- Protein stability assessment
- Cold chain validation
- Formulation development

**Food Science**:
- Shelf-life prediction
- Storage condition optimization
- Quality degradation modeling

**Materials Science**:
- Service life prediction
- Accelerated aging extrapolation
- Mechanism elucidation

---

## 📞 Need Help?

- **Issues**: Report at [GitHub repository]
- **Questions**: Check project README
- **Custom Analysis**: Review `demo_auto_model.py` template

---

# Original Demo Documentation

## 🔧 Quick Start (Custom Data)

## Quick Start

### 1. Prepare Your Data

Place your isothermal CSV data files in this directory. The data files should contain:
- **Time column**: Time values (can be in various units)
- **Temperature column**: Temperature values (K, °C, or °F)
- **Readout column**: Measurement values (concentration, degradation %, etc.)

Example file structure (`protein_stability_313K.csv`):
```csv
Time (days),Temperature (K),HMW Species (%)
0,313.15,2.1
1,313.15,3.5
2,313.15,5.2
...
```

### 2. Update the Demo Script

Edit `demo_auto_model.py` and update the file paths to match your data:

```python
data_files = [
    Path(__file__).parent / "your_data_313K.csv",
    Path(__file__).parent / "your_data_323K.csv",
    Path(__file__).parent / "your_data_333K.csv",
]
```

Also update the column names to match your CSV:
```python
time_col='Time (days)',
temperature_col='Temperature (K)',
readout_col='HMW Species (%)',
readout_type='increasing',  # or 'decreasing'
```

### 3. Run the Analysis

```bash
python demo_auto_model.py
```

The script will:
1. ✅ Load all data files
2. ✅ Fit 15+ kinetic models automatically
3. ✅ Select the best model using statistical ranking (BIC, AICc, R²)
4. ✅ Generate predictions with 95% confidence intervals (bootstrap)
5. ✅ Create ICH Q1E regulatory compliance report
6. ✅ Export publication-ready plots (300 DPI PNG)
7. ✅ Generate interactive HTML report

## Output Files

After running, you'll find in the `output/` directory:

- **`isothermal_stability_report.html`** - Complete interactive HTML report with:
  - Model comparison table
  - Best fit plots
  - Arrhenius analysis
  - Predictions with confidence intervals
    - Shelf-life trend and extrapolation summary (not a compliance determination)
  - Statistical details

- **`data_multi_temperature.png`** - Experimental data at all temperatures

- **`arrhenius_plot.png`** - Arrhenius plot (ln(k) vs 1/T)

- **`prediction_with_ci.png`** - Prediction with bootstrap confidence intervals

All PNG files are 300 DPI, ready for publication or regulatory submission.

## Features Demonstrated

### Automated Model Selection

The demo tries 15+ models including:
- **Single-step models**: F0, F1, F2, F3 (nth order)
- **Nucleation models**: A2, A3 (Avrami-Erofeev)
- **Geometric models**: R2, R3 (contracting area/volume)
- **Diffusion models**: D2, D3 (Jander equation)
- **Autocatalytic**: SB(m,n), Bna (Prout-Tompkins)
- **Complex reactions**: A→B→C (consecutive), A+B→C (bimolecular)
- **Model-free**: Friedman isoconversional analysis

### Statistical Ranking

Models are ranked using:
- **BIC** (Bayesian Information Criterion) - penalizes complexity
- **AICc** (Corrected Akaike Information Criterion)
- **R²** (Coefficient of determination)
- **Simplicity penalty** - prefers simpler models when fit quality is similar

### Bootstrap Confidence Intervals

- 100 bootstrap resamples (configurable)
- 95% confidence intervals on all predictions
- Parameter uncertainty quantification
- Conservative shelf-life estimates

### ICH Q1E Regulatory Compliance

Automatically included in HTML reports:
- One-sided 95% confidence intervals (lower bound)
- Extrapolation ceiling calculations
- Clear guidance on regulatory acceptability
- Professional formatting for submission

### Temperature Excursion Simulation

The demo includes a shipping excursion scenario:
```python
shipping_profile = [
    (0, 25),   # Start at 25°C
    (5, 25),   # Storage before shipping
    (7, 40),   # 2-day shipping at elevated temperature
    (10, 25),  # Return to storage
    (30, 25),  # End of simulation
]
```

This models real-world scenarios like:
- Temperature-controlled shipping
- Cold chain excursions
- Accelerated stability testing
- Climate zone variations

## Customization Options

### Model Selection

```python
models_to_try=['F1', 'F2', 'A2', 'SB_mn']  # Try specific models
include_ode_models=False  # Skip complex models (faster)
```

### Prediction Settings

```python
predict=(2, 'year', 25)  # 2 years at 25°C
predict=(6, 'month', 5)  # 6 months at 5°C
predict=(180, 'day', 40)  # 180 days at 40°C
```

### Bootstrap Settings

```python
bootstrap_iterations=50   # Faster (less accurate CI)
bootstrap_iterations=200  # Slower (more accurate CI)
bootstrap_iterations=0    # Skip bootstrap (no CI)
```

### Report Format

```python
report_format='interactive'  # Plotly interactive plots (default)
report_format='static'       # Matplotlib static images
report_format='both'         # Both formats
```

### Data Loading

```python
auto_detect=True  # Fuzzy column name matching
auto_detect=False  # Exact column names only

readout_type='increasing'  # Degradation increases (%)
readout_type='decreasing'  # Concentration decreases
```

## Advanced Usage

For more control, you can access individual results:

```python
results = auto_model_isothermal_data(...)

# Access specific components
fit_result = results['fit_results']['F2']
datasets = results['datasets']
prediction = results['prediction']
bootstrap = results['bootstrap_results']['F2']

# Generate custom plots
from akts import plot_arrhenius, plot_parameter_distributions

fig = plot_arrhenius(fit_result)
fig.savefig('custom_arrhenius.png', dpi=300)

fig = plot_parameter_distributions(bootstrap, parameters=['Ea', 'A'])
fig.savefig('parameter_distributions.png', dpi=300)
```

## Troubleshooting

### "No data files found"
- Ensure CSV files are in the `Examples/Isothermal/` directory
- Update file paths in `demo_auto_model.py`

### "Column not found"
- Check column names in your CSV match the script
- Enable `auto_detect=True` for fuzzy matching

### "All models failed"
- Check data quality (no NaN values)
- Ensure temperature is in Kelvin (or set `input_temperature_units='C'`)
- Try fewer models: `include_ode_models=False`

### "Bootstrap too slow"
- Reduce `bootstrap_iterations=50` (default is 100)
- Or skip bootstrap: `bootstrap_iterations=0`

### "ODE solver warnings"
- Normal for complex models (A→B→C)
- The solver will fall back to LSODA automatically
- Final fit will still be accurate

## Need Help?

- **Documentation**: See `docs/` directory
- **Examples**: More examples in `Examples/` directory
- **Issues**: Report bugs at GitHub repository
- **Questions**: Check `README.md` in project root

## Citation

If you use this software for publication or regulatory submission, please cite:

```
AKTS Python Library for Kinetic Analysis
[https://github.com/PaulNobrega/akts]
```
