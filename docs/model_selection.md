# Model Selection with Akaike Weights

## Overview

AKTS uses a **filter-then-rank** approach with **Akaike weights** as the sole ranking criterion. This provides optimal model selection that balances fit quality and complexity.

---

## Two-Stage Process

### Stage 1: Filtering (Hard Requirements)

Models must meet quality thresholds before ranking:

```python
# R² filter (default: 0.70)
if R² < 0.70:
    REJECT  # Insufficient fit quality for extrapolation

# Physical plausibility filter
if any(A >= 10²⁰ s⁻¹ OR Ea outside 5-1000 kJ/mol):
    FLAG as implausible
```

**Filter Logic:**
1. If plausible models with R² ≥ 0.70 exist → Use them
2. Otherwise → Keep R² ≥ 0.70 models, warn about plausibility (`filter_warning` type `'no_plausible_models'`)
3. If no model reaches R² ≥ 0.70 → Keep all models, warn about fit quality (`filter_warning` type `'no_good_models'`)

### Stage 2: Ranking (Akaike Weights)

**Formula:**
```
AIC_i = -2×ln(likelihood) + 2×k
Δ_i = AIC_i - AIC_min
w_i = exp(-0.5 × Δ_i) / Σ exp(-0.5 × Δ_j)
```

AIC here is the small-sample corrected AICc.

**Interpretation:**
- w = probability that model i is the best in the candidate set
- All weights sum to 100%
- Automatically balances fit quality and model complexity

Ranking uses the Akaike weight only. There is no separate simplicity penalty
(the `simplicity_penalty` field in each ranked entry is always 0.0): the 2k term
in AIC already penalizes extra parameters, and adding another penalty would count
complexity twice.

### Stage 3: Selection

`auto_model_isothermal_data()` selects the rank-1 model. There is no tie-break
that swaps in a simpler model when ΔAIC < 2. `results['selected_model']['reason']`
describes the evidence from the Akaike weight:

| Akaike weight | Reason text |
|---------------|-------------|
| ≥ 90% | "Overwhelming evidence" |
| ≥ 70% | "Strong evidence" |
| ≥ 50% | "Substantial support" |
| < 50% | "Best of N competitive models" (N = top-5 models with weight > 10%) |

---

## Why Akaike Weights?

### Compared to R² Alone

**R² problems:**
```python
# Example:
Model A: R² = 0.95, k = 2 params  ← Better choice!
Model B: R² = 0.96, k = 10 params ← Overfitting

# R² alone → selects Model B (overfitting)
# Akaike weights → selects Model A (better predictive performance)
```

**Why R² fails:**
- ❌ No penalty for complexity
- ❌ Always favors more parameters
- ❌ Leads to overfitting
- ❌ Poor extrapolation

**Why Akaike works:**
- ✅ Penalizes unnecessary parameters
- ✅ Balances bias-variance tradeoff
- ✅ Optimal for prediction (your use case!)
- ✅ Statistical interpretation (probabilities)

### Compared to BIC

BIC is similar but penalizes complexity more strongly:
```
BIC = -2×ln(L) + k×ln(n)
```

- BIC: Stronger penalty → favors simpler models
- AIC: Balanced penalty → better prediction
- **For kinetics**: AIC is better (small n, prediction focus)

---

## Usage

### Basic:
```python
from akts import auto_model_isothermal_data

results = auto_model_isothermal_data(
    data_files=['25C.csv', '40C.csv'],
    models_to_try=models.all,
    # Akaike weights used automatically
    # R² ≥ 0.70 filter applied
    # Plausibility filter applied with fallback
)

selected = results['selected_model']
print(f"Model: {selected['model_name']}")
print(f"Akaike weight: {selected['statistics']['akaike_weight']:.1%}")
print(f"Plausible: {selected['statistics']['is_physically_plausible']}")
```

### Custom Thresholds:
```python
results = auto_model_isothermal_data(
    data_files=['25C.csv', '40C.csv'],
    min_r_squared=0.80,  # Stricter R² requirement
    apply_filters=True    # Enable filters (default)
)
```

### Disable Filters (Not Recommended):
```python
results = auto_model_isothermal_data(
    data_files=['25C.csv', '40C.csv'],
    apply_filters=False  # Keep all models, even poor/implausible
)
```

---

## Interpreting Akaike Weights

### One Model Dominates (Weight ≈ 100%)

```python
Model A: AIC = -45.0  →  Weight = 100.0%
Model B: AIC = -25.0  →  Weight ≈ 0.0%  (Δ_AIC = 20)
Model C: AIC = -23.0  →  Weight ≈ 0.0%  (Δ_AIC = 22)
```

**Interpretation:**
- ✅ **This is CORRECT behavior, not a bug!**
- Model A is vastly superior (Δ_AIC > 20)
- Overwhelming evidence for Model A
- Confidently use Model A

### Multiple Models Competitive

```python
Model A: AIC = -45.0  →  Weight = 47%  (Δ_AIC = 0.0)
Model B: AIC = -44.5  →  Weight = 36%  (Δ_AIC = 0.5)
Model C: AIC = -43.0  →  Weight = 17%  (Δ_AIC = 2.0)
```

**Interpretation:**
- Model uncertainty exists
- Multiple models have similar support
- Consider:
  - Using top model but acknowledge uncertainty
  - Model averaging (advanced)
  - Collecting more data

### Δ_AIC Guidelines (Burnham & Anderson)

| Δ_AIC | Evidence | Weight (approx) |
|-------|----------|-----------------|
| 0-2   | Substantial support | 25-100% |
| 2-4   | Considerable support | 10-25% |
| 4-7   | Much less support | 1-10% |
| 7-10  | Very little support | 0.1-1% |
| > 10  | Essentially no support | < 0.1% |

---

## Physical Plausibility Flags

Models are flagged (but not automatically rejected) if:

### Pre-exponential Factor (A)
```
A ≥ 10²⁰ s⁻¹  →  FLAG: Exceeds transition state theory maximum
```

**Why this matters:**
- A represents maximum rate from molecular vibrations
- Theoretical maximum ≈ k_B×T/h × exp(ΔS‡/R) ≈ 10²⁰ s⁻¹
- Higher values are physically impossible
- Indicates overfitting or model misspecification

### Activation Energy (Ea)
```
Ea > 1000 kJ/mol  →  FLAG: Too high
```

The plausibility range is 5-1000 kJ/mol, the same as the fitting bounds
(`akts.utils.EA_BOUNDS`), so a fit cannot go below 5 kJ/mol. When a fitted Ea
ends at one of the bounds, `fit_kinetic_model` warns that the optimum may lie
outside the bounds.

**Why the range is wide:**
- Below 5 kJ/mol there is effectively no activation barrier
- Cooperative protein unfolding/denaturation typically has Ea of 400-800 kJ/mol, so a tighter upper limit would reject real systems
- The HTML report lists typical ranges next to the reported Ea: thermal denaturation/unfolding 400-800 kJ/mol, enzymatic/proteolytic cleavage 20-100 kJ/mol, spontaneous/pyrolytic hydrolysis 90-140 kJ/mol

`results['selected_model']['physical_sanity_flags']` separately lists Ea values
outside 30-180 kJ/mol, the range typical of drug degradation. These are
warnings only and do not affect filtering.

### What Happens When Flagged?

**If plausible alternatives exist:**
```
Implausible models are FILTERED OUT before ranking
→ Only plausible models ranked
→ Best plausible model selected
```

**If NO plausible alternatives exist:**
```
Keep R² ≥ 0.70 models even if implausible
→ Rank by Akaike weights
→ ADD WARNING to output:

⚠ WARNING: No physically plausible models achieved R² ≥ 0.70.
Selected model has questionable energetic plausibility.

Physical plausibility criteria:
  - Pre-exponential factor (A): Must be < 10²⁰ s⁻¹
    Rationale: Maximum rate from transition state theory
    
  - Activation energy (Ea): Must be 5-1000 kJ/mol

⚠ Use predictions with caution.

Recommendations:
  1. Collect more data (more time points, temperatures, replicates)
  2. Try models.all to explore all available models
  3. Validate against independent data if available
```

---

## Example Output

### Normal Case (Plausible Model Found):

```
================================================================================
Selected Model: SB(m=0, n=1) (First-order kinetic model)
================================================================================

Rank: 1 / 24
Akaike Weight: 85.3% ← High confidence!

Parameters:
  Ea = 147 kJ/mol     ✓ Plausible (5-1000 kJ/mol range)
  A  = 1.2e13 s⁻¹    ✓ Plausible (< 1e20 s⁻¹ limit)
  
Goodness of Fit:
  R² = 0.942
  AIC = -42.5
  BIC = -38.2
  
Physical Plausibility: ✓ PASS

Interpretation:
  This model has 85% probability of being the best in the candidate set.
  High Akaike weight indicates strong evidence for this model.
  Parameters are physically plausible for chemical degradation.
```

### Warning Case (No Plausible Models):

```
================================================================================
⚠ WARNING: Model Selection Quality Issue
================================================================================

No physically plausible models achieved R² ≥ 0.70.
Selected model has questionable energetic plausibility.

Selected Model: SB(m=0, n=0) (Zero-order)
Rank: 1 / 24
Akaike Weight: 100%

Parameters:
  Ea = 290 kJ/mol     ✓ Within 5-1000 kJ/mol
  A  = 1.6e45 s⁻¹    ✗ Exceeds physical limit (> 1e20 s⁻¹)

Goodness of Fit:
  R² = 0.978  ← Good fit statistically
  AIC = -48.3
  BIC = -44.0

Physical Plausibility: ✗ FAIL
  Issue: Pre-exponential factor exceeds 1e20 s⁻¹ physical limit

⚠ Predictions may be unrealistic for long-term extrapolation.

Recommendations:
  • Collect more data (more time points or temperatures)
  • This dataset shows extremely slow degradation
  • Consider if measurement precision is limiting factor
```

**Two-step Sestak-Berggren (SB2) fits** usually trigger this warning when they
are selected. The fast step of an SB2 fit needs a pre-exponential factor far
above 10²⁰ s⁻¹ (about 10¹⁶⁶ s⁻¹ or more on the commercial HMW example), so SB2 fits
usually fail the A limit. Such a fit is ranked only when no plausible model
reaches R² ≥ `min_r_squared`; otherwise the plausibility filter removes it.

---

## Advanced: Direct Ranking

For custom workflows, use `rank_models()` directly:

```python
from akts import rank_models
from akts.fitting import fit_kinetic_model

# Fit models manually
fit_results = []
for model_name in ['F1', 'F2', 'A2']:
    result = fit_kinetic_model(
        datasets=datasets,
        model_name=model_name,
        ...
    )
    fit_results.append(result)

# Rank with Akaike weights
ranked = rank_models(
    fit_results,
    min_r_squared=0.70,  # Filter threshold
    apply_filters=True    # Enable filters
)

# Access results
for model in ranked[:3]:  # Top 3
    print(f"{model['rank']}. {model['model_name']}")
    print(f"   Akaike weight: {model['stats']['akaike_weight']:.1%}")
    print(f"   R²: {model['stats']['r_squared']:.3f}")
    print(f"   Plausible: {model['stats']['is_physically_plausible']}")
```

`rank_models()` accepts only `fit_results`, `min_r_squared`, and `apply_filters`.
The former `ranking_method` and `score_weights` arguments were removed; passing
them raises `TypeError`. `discover_kinetic_models()` likewise takes
`min_r_squared` and `apply_filters` instead of `score_weights`.

---

## References

### Theory:
1. **Akaike (1974)**: "A new look at the statistical model identification"
   - Original AIC paper
   
2. **Burnham & Anderson (2002)**: "Model Selection and Multimodel Inference"
   - Canonical reference for Akaike weights
   - Guidelines for Δ_AIC interpretation

3. **Eyring (1935)**: "The Activated Complex in Chemical Reactions"
   - Transition state theory → A_max limit

### Practice:
4. **ICH Q1E**: Pharmaceutical stability guidelines
   - Recommends information criteria for model selection
   
5. **ASTM E698**: Kinetic parameter testing
   - R² thresholds for quality

---

## See Also

- [Getting Started](getting_started.md) - Basic usage
- [Advanced Usage](advanced_usage.md) - Custom workflows
- [Data Quality Weighting](data_quality_weighting.md) - Weighted fitting
- [API Reference](api_reference.md) - Complete API

---

## Summary

**AKTS uses Akaike weights exclusively for model selection because:**

1. ✅ **Optimal for prediction** - Your primary use case (shelf-life)
2. ✅ **Prevents overfitting** - Automatic complexity penalty
3. ✅ **Statistical interpretation** - Probabilities, not arbitrary scores
4. ✅ **Standard practice** - Peer-reviewed, regulatory-accepted
5. ✅ **No tuning needed** - No arbitrary weights to adjust

**Physical plausibility is enforced as a filter** - models with implausible parameters are excluded when better alternatives exist, with clear warnings when they're not.

**Akaike weight = 100% is correct** - it means one model is vastly superior. Use it with confidence!
