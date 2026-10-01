# Two-Step Kinetics Comparison - Isothermal Example

This example runs automated kinetic analysis on isothermal %HMW aggregation data and
checks whether the two-step Sestak-Berggren model (SB2), the form used by AKTS
commercial software, recovers the kinetics that generated the data.

The data are **synthetic**: they are simulated from a known two-step model with
duplicate measurements and assay noise (see `generate_synthetic_data.py`), so they can
be published and the fitted parameters can be compared against the true values.

## Files

- **commercial_comp.py** - Main example script
- **generate_synthetic_data.py** - Regenerates the CSV files from the known model (fixed seed)
- **HMW_5C.csv**, **HMW_15C.csv**, **HMW_25C.csv** - 8 time points (0-56 days) x 2 replicates
- **HMW_30C.csv** - 7 time points (0-42 days) x 2 replicates
- **HMW_40C.csv** - 7 time points (0-21 days) x 2 replicates (accelerated condition)

## Data Format

- `Time(day)` - Time in days
- `Temperature(°C)` - Temperature in Celsius
- `HMW(%)` - High Molecular Weight species percentage

All files are UTF-8 encoded.

## Generating Model

    dα/dt = k1(T)·α^m1·(1-α)^n1 + k2(T)·α^m2·(1-α)^n2,   ki(T) = Ai·exp(-Eai/RT)
    HMW(%) = 1.2 + α·(100 - 1.2)

| Step | Ea (kJ/mol) | ln(A·s) | m | n |
|------|-------------|---------|-----|-----|
| 1 (steep aggregation) | 250 | 82.611 | 0.4 | 2.0 |
| 2 (slow, low temperature) | 85 | 15.357 | 0.0 | 4.0 |

Assay noise: Gaussian, SD 0.05 % HMW per measurement.

## Running the Example

```bash
cd Examples/Commercial_AKTS_Comp_Isothermal
python generate_synthetic_data.py   # optional: rewrites the CSVs (same seed, same data)
python commercial_comp.py
```

## What the Script Does

1. **Loads data** from the five CSV files and puts them on one conversion scale,
   conversion = (HMW - HMW0)/(100 - HMW0) via `readout_final=100.0`
2. **Fits models** from `models.all`, including SB2 and the 136-model SB2 grid
   (this takes several minutes; use `models.kinetic.SB2` for a quick run)
3. **Ranks** models by Akaike weight after R² and plausibility filters
4. **Bootstraps** the selected model for 95% confidence and prediction bands
5. **Predicts** 3 years at 5°C and simulates a shipping temperature excursion
6. **Calculates shelf life** (regression-based ICH Q1E and bootstrap one-sided bound)
7. **Writes** an interactive HTML report and PNG plots to `output/`

## Expected Results

Fitting the synthetic data with `SB2` (checked with `models_to_try=['SB2', 'SB_mn', 'R2', 'F1']`):

| | Ea1 (kJ/mol) | ln(A1·s) | m1 | n1 | Ea2 (kJ/mol) | ln(A2·s) | m2 | n2 |
|---|---|---|---|---|---|---|---|---|
| True | 250 | 82.6 | 0.4 | 2.0 | 85 | 15.4 | 0.0 | 4.0 |
| SB2 fit | 248 | 81.9 | 0.40 | 2.02 | 83 | 14.9 | 0.05 | 8 (at bound) |

SB2 ranks first (R² > 0.9999) with 100% Akaike weight. The two steps are
interchangeable, so the fit may report them in either order (above, the steep step is
shown first). The steep step is recovered closely. The slow step's Ea and A are
recovered, but its n goes to the bound: at 5-15°C conversion stays below 1%, where
(1-α)^n is close to 1 whatever n is, so the data cannot identify it.

The SB2 pre-exponential factors exceed the 1e20 s⁻¹ plausibility limit, so the run
prints a "no physically plausible models" warning; the model is still selected.

## Customization

```python
from akts import models

results = auto_model_isothermal_data(
    ...
    models_to_try=models.kinetic.SB2,          # fast: continuous SB2 only
    shelf_life_specification_limit=0.10,       # 10% instead of default 5%
    bootstrap_iterations=500,                  # more iterations for regulatory work
    calculate_shelf_life_ich_q1e_method=False, # skip ICH Q1E shelf-life
)
```

## Troubleshooting

### Column Name Mismatch
Column names must match: `Time(day)`, `Temperature(°C)`, `HMW(%)`.

### Temperature Units
Input data is in Celsius; set `input_temperature_units='C'`. Prediction and simulation
values are also in Celsius. Internal calculations use Kelvin.

## Further Reading

- [ICH Q1E Compliance Guide](../../docs/ich_q1e_compliance.md)
- [Automated Analysis Guide](../../docs/automated_analysis.md)
- [SB Grid Search and SB2](../../docs/SB_GRID_SEARCH.md)
