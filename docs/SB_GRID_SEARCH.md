# Sestak-Berggren Grid Search

## Overview

The Sestak-Berggren (SB) model is a general form that encompasses many solid-state kinetic mechanisms:

```
f(α) = α^m · (1-α)^n
```

Where:
- `α` is the degree of conversion (0 to 1)
- `m` controls the autocatalytic character (acceleration at low conversion)
- `n` controls the reaction order (deceleration as conversion increases)

## Problem with Continuous Optimization

The traditional approach fits `m` and `n` as continuous parameters alongside `Ea` and `A`. This has several drawbacks:

1. **Overfitting**: With 4 parameters (Ea, A, m, n), the model can fit noise rather than mechanism
2. **Interpretability**: Non-integer values (e.g., m=0.73, n=1.42) lack clear mechanistic meaning
3. **Convergence**: More parameters → slower, less reliable convergence
4. **Comparability**: Continuous values make it harder to compare across studies

## Solution: Grid Search Approach

Instead of fitting m,n continuously, we sample **integer values** in the range 0-3. This:

✅ **Ensures mechanistic meaning** - Integer values correspond to known mechanisms  
✅ **Prevents overfitting** - Only 2 fitted parameters (Ea, A) per model  
✅ **Faster convergence** - Simpler optimization problem  
✅ **Comprehensive coverage** - 16 models cover all practical mechanisms  
✅ **Better interpretability** - Results are mechanistically meaningful  

## Usage

### Basic Usage

```python
from akts import models, auto_model_isothermal_data

# Use SB grid for mechanistic screening
results = auto_model_isothermal_data(
    data_files=['data_40C.csv', 'data_60C.csv', 'data_80C.csv'],
    models_to_try=models.kinetic.SB_grid,  # All 16 SB(m,n) combinations
    top_n=5,
    report_path='results.html'
)
```

### Combined with Other Models

```python
# Comprehensive mechanistic screening
results = auto_model_isothermal_data(
    data_files=['data_40C.csv', 'data_60C.csv', 'data_80C.csv'],
    models_to_try=models.kinetic.all_with_sb_grid,  # SB grid + Avrami, diffusion, etc.
    top_n=10,
    report_path='comprehensive_results.html'
)
```

### Custom Grid

```python
from akts import generate_sb_grid_models

# Custom range: m = 0-2, n = 0-3
custom_grid = generate_sb_grid_models(m_range=range(3), n_range=range(4))

results = auto_model_isothermal_data(
    data_files=['data_40C.csv', 'data_60C.csv'],
    models_to_try=custom_grid,
    top_n=3
)
```

## Mechanistic Interpretations

| Model | Formula | Mechanistic Meaning | Common Use Cases |
|-------|---------|-------------------|------------------|
| `SB_m0_n0` | `f(α) = 1` | Zero-order (constant rate) | Photodegradation, surface reactions |
| `SB_m0_n1` | `f(α) = (1-α)` | First-order (exponential decay) | **Most pharmaceuticals, proteins** |
| `SB_m0_n2` | `f(α) = (1-α)²` | Second-order | Dimerization, bimolecular |
| `SB_m0_n3` | `f(α) = (1-α)³` | Third-order | Rare, complex reactions |
| `SB_m1_n0` | `f(α) = α` | Power law | Autocatalytic (accelerating) |
| `SB_m1_n1` | `f(α) = α·(1-α)` | Autocatalytic | Prout-Tompkins, thermal polymerization |
| `SB_m2_n1` | `f(α) = α²·(1-α)` | Strong autocatalytic | Crystallization, phase transitions |
| `SB_m1_n2` | `f(α) = α·(1-α)²` | Mixed autocatalytic + decay | Complex degradation pathways |

### Special Cases

- **m=0, n=0**: Zero-order (F0)
- **m=0, n=1**: First-order (F1) - identical to classical first-order model
- **m=0, n=2**: Second-order (F2)
- **m=0, n=3**: Third-order (F3)
- **m=1, n=1**: Classic autocatalytic (similar to Prout-Tompkins)

## Comparison: Grid vs Continuous

| Aspect | Grid Search (Recommended) | Continuous Optimization |
|--------|---------------------------|-------------------------|
| **Parameters fitted** | 2 per model (Ea, A) | 4 (Ea, A, m, n) |
| **Number of models** | 16 (one per m,n combo) | 1 |
| **m,n values** | Integer (0-3) | Continuous (e.g., 0.73, 1.42) |
| **Interpretability** | ✅ Clear mechanistic meaning | ❌ Unclear meaning |
| **Overfitting risk** | ✅ Low (fewer params) | ⚠️ Higher (more params) |
| **Convergence** | ✅ Fast & reliable | ⚠️ Slower, can get stuck |
| **Recommended for** | Drug stability, material science | High-quality data, theoretical work |

## Model Selection API

The `models` selector provides several ways to access SB models:

```python
from akts import models

# 1. All 16 SB grid models (m,n = 0-3)
models.kinetic.SB_grid  
# → ['SB_m0_n0', 'SB_m0_n1', ..., 'SB_m3_n3']

# 2. Continuous SB (fits m,n as parameters)
models.kinetic.SB  
# → ['SB']

# 3. Fixed SB(m=0.5, n=1.0)
models.kinetic.SB_mn  
# → ['SB_mn']

# 4. All standard models (F0-F3, A2-A3, R2-R3, D2-D3, SB_mn, Bna)
models.kinetic.all  
# → 12 models (excludes grid to avoid duplication)

# 5. Comprehensive screening (SB grid + Avrami, diffusion, etc.)
models.kinetic.all_with_sb_grid  
# → 23 models (replaces F0-F3 with 16 SB grid variants)

# 6. Two-step SB (see "Two-Step Sestak-Berggren (SB2)" below)
models.kinetic.SB2        # ['SB2']
models.kinetic.SB2_grid   # 136 models
```

## Model Selection Criteria

SB grid models are ranked like every other model: after the R² (default 0.70)
and physical-plausibility filters, by **Akaike weight** (from AICc) only, and
`auto_model_isothermal_data()` selects the rank-1 model.

```python
results = auto_model_isothermal_data(
    data_files=files,
    models_to_try=models.kinetic.SB_grid,
    min_r_squared=0.70,   # default
    apply_filters=True    # default
)
```

There is no `ranking_method` or `score_weights` option (both were removed) and
no simplicity tie-break toward lower m+n. All 16 grid models have the same two
fitted parameters, so AICc compares them on fit quality alone; against models
with more parameters (continuous SB, SB2) the 2k term in AICc provides the
complexity penalty. See [Model Selection](model_selection.md).

## When to Use Each Approach

### Use Grid Search (✅ Recommended) when:
- ✅ You want mechanistically interpretable results
- ✅ You're screening for the best mechanism
- ✅ You're working with pharmaceutical/drug stability data
- ✅ You want to prevent overfitting
- ✅ You need robust, reproducible results

### Use Continuous SB (⚠️ Advanced) when:
- You have high-quality, low-noise data
- The grid search gives poor fits (ΔAICc > 10 vs continuous)
- Non-integer m,n values are theoretically justified for your system
- You're doing fundamental kinetic research

### Hybrid Approach (🏆 Best for Publication):
1. **First pass**: Run grid search to identify the best integer m,n
2. **Second pass**: Use continuous SB initialized near the best grid point
3. **Compare**: If ΔAICc < 2, you may prefer the integer values (more interpretable). This is a manual judgment; automatic selection uses the Akaike weight only

```python
# Step 1: Grid search
grid_results = auto_model_isothermal_data(
    data_files=files,
    models_to_try=models.kinetic.SB_grid,
    top_n=3
)

# Step 2: If best grid model has, e.g., m=1, n=1...
# Step 3: Optionally refine with continuous SB
continuous_results = auto_model_isothermal_data(
    data_files=files,
    models_to_try=['SB'],  # Continuous optimization
    top_n=1
)

# Compare AICc values and choose the most parsimonious
```

## Two-Step Sestak-Berggren (SB2)

Some data need two processes on one conversion, for example a slow process that
dominates at low temperature and a steep one that takes over at high
temperature. SB2 is the AKTS commercial two-step form: two parallel SB steps
summed on a single conversion.

```
dα/dt = k1(T)·α^m1·(1-α)^n1 + k2(T)·α^m2·(1-α)^n2,    ki(T) = Ai·exp(-Eai / RT)
```

### Variants

| Selector | Models | Fitted parameters | Shape parameters |
|----------|--------|-------------------|------------------|
| `models.kinetic.SB2` | 1 (`'SB2'`) | Ea1, A1, Ea2, A2, m1, n1, m2, n2 | Fitted: m in 0-3, n in 0-8 |
| `models.kinetic.SB2_grid` | 136 (`'SB2_m{m1}n{n1}_m{m2}n{n2}'`) | Ea1, A1, Ea2, A2 | Fixed integers, m, n in {0,1,2,3} |

The grid contains every unordered pair of the 16 integer SB steps (16·17/2 = 136,
including pairs of identical steps). The sum is symmetric, so each pair appears
once. A step with m > 0 cannot start from α = 0 on its own; the other step
supplies the initial conversion.

```python
from akts import models, generate_sb2_grid_models
from akts.models import parse_sb2_model

models.kinetic.SB2          # ['SB2']
models.kinetic.SB2_grid     # ['SB2_m0n0_m0n0', 'SB2_m0n0_m0n1', ...]

custom = generate_sb2_grid_models(m_range=range(2), n_range=range(4))
parse_sb2_model('SB2_m0n1_m1n3')   # (0, 1, 1, 3)

results = auto_model_isothermal_data(
    data_files=files,
    models_to_try=models.kinetic.SB2 + models.kinetic.SB2_grid,
)
```

Both are included in `models.all`, but not in `models.kinetic.all`,
`models.default`, or `models.kinetic.all_with_sb_grid`.

### Fitting

- SB2 models are ODE-integrated, unlike the single-step SB grid, so each fit is much slower. SB2_grid adds roughly 10-15 minutes to a `models.all` run.
- Optimization uses scipy `least_squares` (TRF) on the ODE residuals. Powell does not converge on the 8-parameter continuous model.
- Starting values come from zero-order Arrhenius fits of the low-conversion datasets (maximum conversion < 0.2) and the high-conversion ones, tried in both step orderings, plus the default guess. The fit with the lowest AIC is kept.
- Bounds: Ea1, Ea2 in 5-1000 kJ/mol (`akts.utils.EA_BOUNDS`, as for every model); A1, A2 in 10⁻¹⁰ to 10²⁰⁰ s⁻¹, because the fast step needs A of about 10¹⁶⁶ s⁻¹ or more.

### Plausibility Warning

The fitted A of the fast step is usually far above the 10²⁰ s⁻¹ plausibility limit, so a
"No physically plausible models achieved R² ≥ 0.70" warning usually appears when an
SB2 model is selected. An implausible SB2 fit is ranked only when no plausible
model reaches R² ≥ `min_r_squared`; otherwise the plausibility filter removes it
before ranking.

### Validation on Synthetic Data

`Examples/Commercial_AKTS_Comp_Isothermal` contains synthetic %HMW data (5-40°C, duplicates, 0.05 % HMW noise) simulated from a known two-step model. With `readout_final=100.0` (the commercial AKTS %HMW scaling), continuous SB2 ranks first (R² > 0.9999):

| Parameter | True (generating) | akts `SB2` fit |
|-----------|-------------------|----------------|
| Ea, steep step (kJ/mol) | 250 | 248 |
| ln(A·s), steep step | 82.6 | 81.9 |
| m / n, steep step | 0.4 / 2.0 | 0.40 / 2.02 |
| Ea, slow step (kJ/mol) | 85 | 83 |
| ln(A·s), slow step | 15.4 | 14.9 |
| m / n, slow step | 0 / 4.0 | 0.05 / 8 (at bound) |

The two steps are interchangeable, so the fit may report them in either order. The steep step is recovered closely. The slow step's Ea and A are recovered, but its n goes to the bound: at 5-15°C conversion stays below 1%, where (1-α)^n is close to 1 for any n.

## Technical Details

### How It Works

1. **Model naming**: Grid models are named `SB_m{m}_n{n}` (e.g., `SB_m1_n2`)
2. **Recognition**: The fitting code recognizes the pattern via regex
3. **Fixed parameters**: m and n are **fixed** (not fitted) at the specified integer values
4. **Optimization**: Only Ea and A are fitted (2 parameters instead of 4)
5. **Display**: Reports show `SB(m=1, n=2)` for clarity

### Computational Cost

- **Grid search**: 16 models × 2 parameters = 32 optimizations (fast)
- **Continuous**: 1 model × 4 parameters = 1 optimization (slower, less reliable)

Despite trying more models, grid search is often faster overall because each individual fit converges more quickly.

## Examples

See:
- [`Examples/Isothermal/demo_sb_grid_search.py`](../Examples/Isothermal/demo_sb_grid_search.py) - Complete demo with explanations
- [`tests/test_sb_grid.py`](../tests/test_sb_grid.py) - Unit tests
- [`tests/test_sb2.py`](../tests/test_sb2.py) - SB2 unit tests
- [`Examples/Commercial_AKTS_Comp_Isothermal/commercial_comp.py`](../Examples/Commercial_AKTS_Comp_Isothermal/commercial_comp.py) - SB2 fit to synthetic two-step data with known parameters

## Frequently Asked Questions

**Q: Why not use continuous optimization for best fit?**  
A: Continuous optimization has 4 parameters vs 2 for grid models. The extra flexibility often fits noise rather than mechanism, leading to worse predictions and less interpretable results.

**Q: What if my mechanism isn't in the 0-3 range?**  
A: The range 0-3 covers >99% of solid-state reactions in literature. If you genuinely need m or n > 3, generate a custom grid with `generate_sb_grid_models(m_range=range(5), n_range=range(5))`.

**Q: Can I mix grid and continuous models?**  
A: Yes! `models_to_try = models.kinetic.SB_grid + ['SB']` will try both.

**Q: How do I interpret negative m or n?**  
A: The grid search only uses non-negative integers (0-3) by design. Negative m or n values are physically meaningless for degradation kinetics.

**Q: Which is equivalent to first-order?**  
A: `SB_m0_n1` is exactly equivalent to the F1 (first-order) model.

## References

- Šesták, J., & Berggren, G. (1971). Study of the kinetics of the mechanism of solid-state reactions at increasing temperatures. *Thermochimica Acta*, 3(1), 1-12.
- Khawam, A., & Flanagan, D. R. (2006). Solid-state kinetic models: basics and mathematical fundamentals. *The journal of physical chemistry B*, 110(35), 17315-17328.

## Summary

✅ **Use `models.kinetic.SB_grid` as your default choice**  
✅ **Integer m,n = mechanistically meaningful results**  
✅ **Fewer parameters = less overfitting = better predictions**  
✅ **16 models comprehensively screen the mechanistic space**  

The grid search approach provides the sweet spot between mechanistic interpretability and fitting flexibility, making it the recommended choice for most applications.
