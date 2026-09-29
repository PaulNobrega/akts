# Experimental Design for Kinetic Analysis

Guidance for designing isothermal, DSC, and TGA experiments to estimate kinetic parameters.

## Core Principle

**Quality kinetic data requires:**
1. Multiple temperatures (≥3)
2. Sufficient conversion range (0.1-0.8 ideal)
3. Enough time points (≥8 per temperature)
4. Appropriate temperature spacing (20-50°C range)

## Isothermal Experiments

### Accelerated Stability Studies

**Objective:** Predict long-term stability from short-term accelerated data

**Minimum Design:**
```
Temperature: 3 levels spanning 20-30°C
Time points: 8-12 per temperature
Duration: Until 20-80% conversion
Example: 40°C, 50°C, 60°C for 28 days
```

**Recommended Design:**
```
Temperature: 4-5 levels spanning 30-50°C
Time points: 10-15 per temperature
Duration: 0.3-0.7 conversion range
Include target storage temp for validation
Example: 25°C (validation), 40°C, 50°C, 60°C, 70°C
```

### Temperature Selection

**Rules of thumb:**

1. **Span activation energy range:**
   - ΔT ≥ 20°C for reliable Ea estimation
   - ΔT = 30-50°C optimal
   - Avoid exceeding mechanism-change temperature

2. **Include target condition:**
   - Always include intended storage temperature (5°C, 25°C, etc.)
   - Use as validation, not for fitting
   - Confirms no mechanism change

3. **Evenly space in 1/T:**
   - Not equally spaced in °C
   - Example (good): 40°C (313K), 50°C (323K), 60°C (333K), 70°C (343K)
   - Example (bad): 25°C, 50°C, 75°C, 100°C (too wide)

**Temperature calculator:**
```python
import numpy as np

# Target: 4 temperatures spanning 30-60°C
T_low = 30 + 273.15  # 303 K
T_high = 60 + 273.15  # 333 K

# Space evenly in 1/T
inv_T = np.linspace(1/T_high, 1/T_low, 4)
temps_K = 1 / inv_T
temps_C = temps_K - 273.15

print("Optimal temperatures (°C):", temps_C)
# Output: [30.0, 39.1, 49.3, 60.0]
```

### Time Point Selection

**Early phase (0-20% conversion):**
- More frequent sampling (every 1-3 days)
- Captures onset kinetics
- Critical for accurate rate determination

**Middle phase (20-60% conversion):**
- Moderate sampling (every 3-7 days)
- Main fitting region
- Most reliable data

**Late phase (>60% conversion):**
- Less frequent (weekly)
- Often noisy or plateau
- Can omit if needed

**Example timeline for 60°C:**
```
Days: 0, 1, 3, 7, 14, 21, 28
Sampling pattern: ▓░░▓░░░▓░░░░░░▓░░░░░░░▓░░░░░░▓░░░░░░▓
Conversion: 0%, 10%, 25%, 45%, 60%, 70%, 75%
```

### Sample Size

**Per temperature:**
- Minimum: n=2 (duplicates)
- Recommended: n=3 (triplicates)
- High-value: n=4+ if resources allow

**Total samples:**
- Minimum viable: 3 temps × 8 timepoints × 2 reps = 48 samples
- Recommended: 4 temps × 10 timepoints × 3 reps = 120 samples

### Measurement Frequency

**Don't oversample early:**
- No need for hourly measurements
- First 24h: 2-3 timepoints sufficient
- Focus on covering full conversion range

**Don't undersample:**
- Minimum 8 points per temperature
- Fewer points = unreliable fit
- More points = better statistics

## Multi-Rate Experiments (DSC/TGA)

### Differential Scanning Calorimetry (DSC)

**Objective:** Determine curing kinetics, decomposition energy

**Standard Design:**
```
Heating rates: 3-5 rates spanning 4-10x
Common: 5, 10, 20 K/min
Advanced: 2, 5, 10, 15, 20 K/min
```

**Why multiple rates?**
- Isoconversional methods require multi-rate data
- Single rate cannot discriminate models
- Standard practice (ASTM E698)

**Rate selection:**

1. **Minimum: 3 rates**
   - Example: 5, 10, 20 K/min (4x span)
   - Allows basic Ea(α) determination

2. **Recommended: 4-5 rates**
   - Example: 2, 5, 10, 15, 20 K/min (10x span)
   - Better statistical power
   - Model discrimination

3. **Avoid extremes:**
   - Too slow (< 2 K/min): Long experiments, baseline drift
   - Too fast (> 30 K/min): Instrument lag, thermal gradients

**Temperature range:**
- Start 50°C before reaction onset
- End 50°C after completion
- Include full baseline before and after

### Thermogravimetric Analysis (TGA)

**Objective:** Thermal decomposition kinetics

**Standard Design:**
```
Heating rates: 3-4 rates
Common: 5, 10, 20 K/min
Mass loss range: 5-95%
```

**Additional considerations:**

1. **Purge gas:**
   - N₂ or Ar for pyrolysis
   - Air/O₂ for oxidation
   - Flow rate: 20-50 mL/min

2. **Sample mass:**
   - 5-15 mg typical
   - Smaller = less thermal lag
   - Consistent mass across runs

3. **Crucible type:**
   - Pt for high temperature
   - Al for organic samples
   - Open or pinhole lid

## Data Quality Checklist

### Before Starting

- [ ] Instrument calibrated recently?
- [ ] Sample preparation method validated?
- [ ] Storage conditions controlled?
- [ ] Enough material for all timepoints?

### During Experiment

- [ ] Temperature stability ±0.5°C (isothermal)?
- [ ] Samples protected from light/oxygen/moisture?
- [ ] Consistent sampling procedure?
- [ ] Blank/control samples included?

### After Data Collection

- [ ] Data shows smooth degradation curves?
- [ ] No obvious outliers?
- [ ] Conversion ranges 0.2-0.8 achieved?
- [ ] Replicates agree well (RSD < 10%)?

## Common Experimental Errors

### 1. Temperature Not Truly Isothermal

**Problem:** Temperature fluctuates ±2-3°C

**Impact:** Rate constant uncertainty, poor model fits

**Solution:**
- Use temperature-controlled chamber
- Monitor continuously
- Check calibration

### 2. Insufficient Conversion Range

**Problem:** Only measured 0-20% conversion

**Impact:** Cannot discriminate models, extrapolation unreliable

**Solution:**
- Run experiments longer
- Use higher temperatures
- Plan for 0.2-0.8 conversion

### 3. Too Few Temperatures

**Problem:** Only 2 temperatures tested

**Impact:** Cannot determine Ea accurately, no model validation

**Solution:**
- Minimum 3 temperatures
- 4-5 temperatures recommended
- Span 20-30°C range

### 4. Poor Sampling Strategy

**Problem:** Too many early points, none at high conversion

**Impact:** Biased fits, miss late-stage behavior

**Solution:**
- Plan sampling in advance
- Cover full conversion range
- More points in 20-60% region

### 5. Inconsistent Measurements

**Problem:** Different methods/instruments between timepoints

**Impact:** Systematic errors, artifact trends

**Solution:**
- Same instrument/method throughout
- Include quality controls
- Validate method reproducibility

## Statistical Considerations

### Replication

**Technical replicates:**
- Same sample, measured multiple times
- Tests measurement precision
- Minimum: n=2

**Biological/batch replicates:**
- Different preparations/batches
- Tests true variability
- Recommended: n=3

### Sample Size Calculation

For target parameter uncertainty of ±10%:

```
Required samples ≈ 100 / (desired_precision%)²
For ±10%: n ≈ 100 samples total
For ±5%: n ≈ 400 samples total
```

**Practical compromise:** 3-4 temperatures × 10 timepoints × 3 replicates = 90-120 samples

### Outlier Detection

**When to remove data:**
- Technical failure (e.g., instrument error)
- Invalid values (for example, conversion greater than 1.0)
- More than 3σ from mean

**When NOT to remove:**
- "Doesn't fit the model" is not a reason
- High variability is real information
- Document all exclusions

## Application-Specific Guidance

### Pharmaceutical Stability (ICH Guidelines)

**ICH Q1A requirements:**
```
Long-term: 25°C/60% RH for 12 months (minimum 3 timepoints)
Accelerated: 40°C/75% RH for 6 months (minimum 3 timepoints)
```

**Kinetic enhancement:**
```
Add intermediate: 30°C, 50°C
Add stress: 60°C, 70°C
More timepoints: 0, 1, 3, 7, 14, 28 days
Enable Arrhenius extrapolation
```

### Polymer Thermal Stability

**Service lifetime prediction:**
```
Use temperatures: 20-40°C above max service temp
Example: Service at 120°C → test at 140-180°C
Duration: Until measurable degradation
Timepoints: Focus on early phase (0-30%)
```

### Protein Formulation Development

**Rapid screening:**
```
Temperatures: 40°C, 50°C, 60°C
Duration: 1-2 weeks
Timepoints: 0, 1, 3, 7, 14 days
Readout: Aggregation, potency, etc.
Gives Ea for formulation ranking
```

**Full stability study:**
```
Add: 25°C (long-term), 5°C (validation)
Duration: 6-12 months at 25°C
Monthly sampling
Full characterization
```

## Cost-Benefit Trade-offs

### Budget Constrained

**Minimum viable:**
- 3 temperatures
- 8 timepoints each
- Duplicates only
- ~50 samples total

**Provides:**
- Basic Ea estimate (±20%)
- Model discrimination
- Reasonable shelf-life prediction

### Well-Resourced

**Optimal:**
- 5 temperatures (including target)
- 12 timepoints each
- Triplicates
- ~180 samples total

**Provides:**
- Precise Ea (±5%)
- Confidence intervals via bootstrap
- Model validation
- Robust predictions

### Pilot Study Strategy

1. **Phase 1 - Quick screen (3 temps, n=2):**
   - Determine if degradation is measurable
   - Estimate Ea (rough)
   - Cost: ~50 samples

2. **Phase 2 - Full study (4-5 temps, n=3):**
   - Precise kinetics
   - Model selection
   - Predictions
   - Cost: ~150 samples

## Tools for Planning

### Calculate experiment duration

```python
def estimate_duration(T_celsius, Ea_kJ_mol, A_per_s, target_conversion):
    """Estimate time to reach target conversion."""
    import numpy as np

    T_K = T_celsius + 273.15
    Ea = Ea_kJ_mol * 1000  # Convert to J/mol
    R = 8.314  # J/(mol·K)
    k = A_per_s * np.exp(-Ea / (R * T_K))

    # For F1 model: α = 1 - exp(-kt)
    t_seconds = -np.log(1 - target_conversion) / k
    t_days = t_seconds / (24 * 3600)

    return t_days

# Example: Ea=95 kJ/mol, A=1e12 s⁻¹
print(f"60°C to 50% conversion: {estimate_duration(60, 95, 1e12, 0.5):.1f} days")
print(f"40°C to 50% conversion: {estimate_duration(40, 95, 1e12, 0.5):.1f} days")
```

### Experimental Design Checklist

Before starting your study:

- [ ] Determined required precision (±5%, ±10%, ±20%)?
- [ ] Calculated sample size needed?
- [ ] Selected temperatures (≥3, spanning 20-30°C)?
- [ ] Planned timepoints (≥8, covering 0.2-0.8 conversion)?
- [ ] Decided on replication (n≥2)?
- [ ] Estimated experiment duration?
- [ ] Confirmed method precision (RSD < 10%)?
- [ ] Have enough material for all samples?
- [ ] Planned for controls/blanks?

## Further Reading

- **[Getting Started](getting_started.md)** - Analyze your data
- **[Model Selection Guide](model_selection_guide.md)** - Choose the right model
- **[Examples](examples.md)** - See working analyses
- **[Troubleshooting](troubleshooting.md)** - Common data issues

---

Experimental design strongly influences the reliability of kinetic estimates.
