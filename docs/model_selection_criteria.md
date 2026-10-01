# Model Selection Criteria

## Overview

AKTS uses a **filter-then-rank** approach for model selection, combining physical plausibility checks with statistical quality metrics.

---

## Selection Strategy

### Two-Stage Process

```
Stage 1: FILTERS (Hard Requirements)
  ↓
  Remove models that fail minimum criteria
  ↓
Stage 2: RANKING (Relative Quality)
  ↓
  Rank remaining models by information criteria
```

---

## Stage 1: Filters (Hard Requirements)

### 1. Physical Plausibility Filter

**Purpose**: Reject models with parameters that violate physical principles

**Criteria:**
- **Pre-exponential factor**: A < 10²⁰ s⁻¹
  - Theoretical maximum from transition state theory
  - Larger values are physically impossible
  
- **Activation energy**: 5 kJ/mol < Ea < 1000 kJ/mol
  - Same range as the fitting bounds for every Ea (`akts.utils.EA_BOUNDS`)
  - Wide enough for cooperative protein unfolding (typically 400-800 kJ/mol)
  - A fitted Ea at a bound triggers a warning from `fit_kinetic_model`

- **Shape parameters**: m, n ≥ 0 (enforced by the fitting bounds)

**Why hard filter?**
- Models with A > 10²⁰ produce unrealistic extrapolations
- Better to use a model with lower R² but physical parameters
- Prevents catastrophic prediction failures

### 2. Goodness-of-Fit Filter

**Purpose**: Reject models that don't adequately describe the data

**Criteria:**
- **R² < 0.70**: Model is too poor for reliable extrapolation
  
**Conditional Logic:**

```python
if any model has R² ≥ 0.70 AND is_physically_plausible:
    # Filter out R² < 0.70 OR implausible models
    keep models with: R² ≥ 0.70 AND is_physically_plausible
    
elif any model has R² ≥ 0.70:
    # No good plausible models available
    # Relax plausibility filter but keep R² filter
    keep models with: R² ≥ 0.70 (any plausibility)
    ADD WARNING to output/report:
        "No physically plausible models achieved R² ≥ 0.70.
         Selected model has questionable energetic plausibility.
         Physical plausibility criteria: A < 1e20 s⁻¹, 5 < Ea < 1000 kJ/mol
         Use predictions with caution."
    (filter_warning type 'no_plausible_models')

else:
    # No model reaches R² ≥ 0.70
    keep all models
    ADD WARNING: "CRITICAL: No models achieved R² ≥ 0.70 ..."
    (filter_warning type 'no_good_models')
```

The warning is attached to the rank-1 entry as `filter_warning`.

**Rationale:**
- **R² < 0.70**: Explains < 70% of variance → unreliable for extrapolation
- These models extrapolate far into the future (shelf-life predictions)
- Poor fit → large uncertainty → unrealistic predictions

---

## Stage 2: Ranking (Relative Quality)

After filtering, rank remaining models by **Akaike weights**. This is the only ranking criterion.

### Akaike Weights

**Formula:**
```
Δᵢ = AICc_i - AICc_min

wᵢ = exp(-0.5 × Δᵢ) / Σⱼ exp(-0.5 × Δⱼ)
```

**Interpretation:**
- **w** = probability that model i is the best in the candidate set
- Sums to 1.0 across all models
- Incorporates both fit quality AND model complexity

**Example:**
```
Model        AICc      Δ_AIC    Akaike Weight    Interpretation
F1          -45.2      0.0         100%          Overwhelming evidence
F2          -35.1     10.1          <1%          Essentially no support
SB_m0_n2    -32.4     12.8          <1%          Very strong evidence against
```

**Why one model gets 100%?**

This is **correct behavior** when one model is vastly superior:
- Δ_AIC > 10: "Essentially no support" for worse model (Burnham & Anderson)
- Δ_AIC > 20: exp(-0.5 × 20) = exp(-10) ≈ 0.00005 (negligible)
- **This is the signal**: One model is clearly best

**When weights are distributed:**
```
Model        AICc      Δ_AIC    Akaike Weight    Interpretation
F1          -45.2      0.0          52%          Slight preference
F2          -44.8      0.4          45%          Nearly equivalent
SB_m0_n1    -42.1      3.1           3%          Some support
```

**Interpretation**: Multiple models have similar support → model uncertainty

---

## No Alternative Ranking Methods

Akaike weight is the only ranking criterion. `rank_models()` has no
`ranking_method` or `score_weights` argument (passing either raises
`TypeError`), and there is no BIC, R²-only, or weighted combined-score ranking.
BIC and R² are still computed and reported for each model.

There is also no simplicity penalty or ΔAIC/ΔBIC < 2 tie-break toward simpler
models. The 2k term in AIC already penalizes complexity; a second penalty would
count it twice. The `simplicity_penalty` field in each ranked entry is kept for
compatibility and is always 0.0. `auto_model_isothermal_data()` selects the
rank-1 model.

---

## Recommended Workflow

```python
from akts import auto_model_isothermal_data, models

results = auto_model_isothermal_data(
    data_files=['25C.csv', '40C.csv'],
    models_to_try=models.all,  # All 173 models (slow; see below)
    top_n=10,  # Return top 10 for inspection
    
    # Uses Akaike weights by default
    # Applies R² ≥ 0.70 filter
    # Applies plausibility filter (with fallback)
)

# Check if warning was issued
selected = results['selected_model']
print(f"Model: {selected['model_name']}")
print(f"R²: {selected['statistics']['r_squared']:.3f}")
print(f"Plausible: {selected['statistics']['is_physically_plausible']}")

# Check Akaike weight
akaike_wt = selected['statistics']['akaike_weight']
print(f"Akaike weight: {akaike_wt:.1%}")

print(selected['reason'])  # e.g. "Overwhelming evidence (Akaike weight = 97.3%)"
```

`selected['reason']` uses these thresholds: ≥ 90% "Overwhelming evidence",
≥ 70% "Strong evidence", ≥ 50% "Substantial support", otherwise
"Best of N competitive models".

`models.all` includes the 136-model two-step `SB2_grid` plus `SB2`, all
ODE-integrated; they add roughly 10-15 minutes to a run. Use
`models.default` or a smaller selection for quick screening.

---

## Decision Tree

```
                     Start: All fitted models
                              |
                              ↓
                    ┌─────────────────────┐
                    │ Any model with:     │
                    │ R² ≥ 0.70 AND       │
                    │ Plausible?          │
                    └─────────────────────┘
                         /            \
                       YES             NO
                        ↓               ↓
              ┌──────────────────┐   ┌──────────────────┐
              │ Filter to keep:  │   │ Filter to keep:  │
              │ R² ≥ 0.70 AND    │   │ R² ≥ 0.70        │
              │ Plausible        │   │ (any plausibility)│
              └──────────────────┘   └──────────────────┘
                        |                     |
                        ↓                     ↓
              ┌──────────────────┐   ┌──────────────────┐
              │ Rank by Akaike   │   │ Rank by Akaike   │
              │ weights          │   │ weights          │
              └──────────────────┘   │ + ADD WARNING    │
                        |             └──────────────────┘
                        ↓                     |
              ┌──────────────────┐           |
              │ Select top model │←──────────┘
              └──────────────────┘
                        |
                        ↓
                   FINAL MODEL
```

---

## Output Messages

### Normal Case (Plausible Model Found)

```
Selected Model: F1 (First-order)
Rank: 1 / 36
R² = 0.982
Akaike weight: 99.8%
Physical plausibility: PASS
  Ea = 147 kJ/mol (5-1000 kJ/mol range)
  A = 1.2e13 s⁻¹ (< 1e20 s⁻¹ limit)
```

### Warning Case (No Plausible Models with R² ≥ 0.70)

```
⚠ WARNING: Model Selection Quality Issue

No physically plausible models achieved R² ≥ 0.70.
Selected model has questionable energetic plausibility.

Selected Model: SB_m0_n0 (Zero-order)
Rank: 1 / 36
R² = 0.976
Akaike weight: 100%
Physical plausibility: FAIL
  Issues detected:
  - Pre-exponential factor A = 1.6e45 s⁻¹ (exceeds 1e20 s⁻¹ physical limit)
  
Physical Plausibility Criteria:
  - Pre-exponential factor (A): Must be < 10²⁰ s⁻¹
    Rationale: Maximum rate from transition state theory
    
  - Activation energy (Ea): Must be 5-1000 kJ/mol
    
⚠ Predictions from this model may be unrealistic for long-term extrapolation.

Recommendations:
  1. Collect more data (more time points, temperatures, or replicates)
  2. Try models.all to explore all available models
  3. Consider if data quality issues are present (outliers, systematic errors)
  4. Use predictions with caution — validate against independent data if available
```

An SB2 selection normally comes with this warning: the fast step needs A far
above 10²⁰ s⁻¹ (about 10¹⁶⁶ s⁻¹ or more on the commercial HMW example). An implausible SB2 fit
is ranked only when no plausible model reaches R² ≥ `min_r_squared`; otherwise the
plausibility filter removes it.

---

## Scoring vs. Filtering: Why the Change?

### Old Approach (Scoring, removed)
```python
score = 0.4×BIC + 0.4×R² + 0.1×RSS + 0.1×n_params + 1000×(not plausible)
```

**Problems:**
- Arbitrary weights (why 0.4?)
- Less interpretable
- Penalty can be overcome by very good fit

### New Approach (Filtering + Ranking)
```python
# Stage 1: Remove R² < 0.70 OR implausible (conditional)
filtered_models = apply_filters(all_models)

# Stage 2: Rank survivors by Akaike weights
ranked = rank_by_akaike_weights(filtered_models)
```

**Benefits:**
- ✅ Hard requirements (R², plausibility) are explicit
- ✅ Akaike weights have clear interpretation
- ✅ No arbitrary weight tuning
- ✅ Follows statistical best practices
- ✅ Clear warning when requirements not met

---

## Technical Details

### R² Threshold Justification

**Why 0.70?**
- R² < 0.70 = less than 70% variance explained
- For long-term extrapolation (shelf-life), need strong fit
- Rule of thumb: R² > 0.70 is "acceptable", > 0.90 is "excellent"

**Adjustable via parameter:**
```python
results = auto_model_isothermal_data(
    ...,
    min_r_squared=0.80,  # Stricter requirement
)
```

### Plausibility Cutoffs

**A < 10²⁰ s⁻¹:**
- Transition state theory: ν_max = k_B T / h ≈ 10¹³ s⁻¹ at 300K
- Entropy factor: exp(ΔS‡/R) ≈ 10⁷ (maximum reasonable)
- Combined: A_max ≈ 10²⁰ s⁻¹

**5 < Ea < 1000 kJ/mol:**
- Below 5 kJ/mol: effectively no activation barrier
- Upper limit set high enough for protein denaturation/unfolding (typically 400-800 kJ/mol)
- The HTML report shows typical ranges next to the reported Ea: thermal denaturation/unfolding 400-800, enzymatic/proteolytic cleavage 20-100, spontaneous/pyrolytic hydrolysis 90-140 kJ/mol
- `results['selected_model']['physical_sanity_flags']` separately warns (without filtering) when Ea is outside 30-180 kJ/mol, the range typical of drug degradation

### Akaike Weight Distribution

**One model at 100%:**
- Means Δ_AIC > ~20 for next best model
- This is **strong evidence** — not a bug
- Interpretation: "Best model is overwhelmingly supported"

**Even distribution:**
- Means multiple models have similar AIC
- Interpretation: "Model uncertainty — consider model averaging"

---

## References

1. **Burnham & Anderson (2002)**: "Model Selection and Multimodel Inference"
   - Canonical reference for AIC and Akaike weights
   
2. **Schwarz (1978)**: "Estimating the Dimension of a Model"
   - Original BIC paper
   
3. **Eyring (1935)**: "The Activated Complex in Chemical Reactions"
   - Transition state theory → A max
   
4. **ICH Q1E**: Pharmaceutical stability guidelines
   - Recommends information criteria for model selection

---

## See Also

- [Data Quality Weighting](data_quality_weighting.md) - Weighted fitting
- [ICH Q1E Compliance](ich_q1e_compliance.md) - Regulatory requirements
- [Advanced Usage](advanced_usage.md) - Custom workflows
- [API Reference](api_reference.md) - Function signatures

---

**Summary**: Use filter-then-rank approach with R² ≥ 0.70 and plausibility filters, then rank by Akaike weights. Akaike weight = 100% is correct when one model is vastly superior. Clear warnings issued when no plausible models meet R² requirements.
