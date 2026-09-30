# Kinetic Model Reference Table

## Single-Step f(α) Models

| Model ID | Model Name | Equation f(α) | Description | Typical Applications |
|----------|------------|---------------|-------------|---------------------|
| **F0** | Zero-order | `1` (constant) | Constant-rate reaction, independent of remaining fraction | **Controlled-release formulations**, zero-order drug release |
| **F1** | First-order | `(1-α)` | Simple first-order kinetics | **Protein denaturation**, enzyme deactivation, drug degradation, simple chemical reactions |
| **F2** | Second-order | `(1-α)²` | Second-order reaction kinetics | **Protein aggregation**, bimolecular reactions, polymer crosslinking |
| **F3** | Third-order | `(1-α)³` | Third-order reaction kinetics | Complex degradation pathways, multi-component reactions |
| **Fn** | Flexible Nth-order | `(1-α)^n` (n fitted) | Generalized nth-order model where reaction order n is fitted from data | When reaction order is uncertain, F1/F2/F3 don't fit well |
| **A2** | Avrami-Erofeev (n=2) | `2(1-α)[-ln(1-α)]^(1/2)` | Nucleation and growth (2D) | **Crystallization**, phase transitions, polymer crystallization |
| **A3** | Avrami-Erofeev (n=3) | `3(1-α)[-ln(1-α)]^(2/3)` | Nucleation and growth (3D) | Solid-state reactions, crystallization from melt |
| **R2** | Contracting area | `2(1-α)^(1/2)` | Reaction proceeds on surface (2D) | Surface reactions, oxidation of metals, **tablet dissolution** |
| **R3** | Contracting volume | `3(1-α)^(2/3)` | Reaction proceeds throughout volume (3D) | Decomposition of solids, combustion, particle reactions |
| **D1** | 1D diffusion | `1/(2α)` | Diffusion-limited (planar) | Thin film degradation, coating oxidation |
| **D2** | 2D diffusion (Valensi) | `1/[-ln(1-α)]` | Diffusion-limited (cylindrical) | Cylinder oxidation, fiber degradation |
| **D3** | 3D diffusion (Jander) | `(3/2)(1-α)^(2/3)/[1-(1-α)^(1/3)]` | Diffusion-limited (spherical) | **Particle oxidation**, slow solid-state reactions, controlled-release |
| **D4** | 3D diffusion (Ginstling-Brounshtein) | `(3/2)/[(1-α)^(-1/3)-1]` | Alternative diffusion (spherical) | Similar to D3, different boundary conditions |
| **SB_mn** | Sestak-Berggren SB(m,n) | `α^m·(1-α)^n` (fixed m=0.5, n=1.0) | General empirical autocatalytic model | Complex/ambiguous mechanisms, autocatalytic degradation not well described by Bna |
| **SB** | Fitted Sestak-Berggren | `α^m·(1-α)^n` (m, n fitted) | Flexible autocatalytic model where both exponents are fitted from data | Complex degradation mechanisms with both autocatalytic and inhibition effects |
| **SB_mnp** | Extended Sestak-Berggren | `α^m·(1-α)^n·[-ln(1-α)]^p` (m, n, p fitted) | Most general empirical model combining power-law, autocatalytic, and nucleation terms | Very complex mechanisms, research applications requiring maximum flexibility |
| **Bna** | Prout-Tompkins (autocatalytic) | `α^c·(1-α)` (fixed c=1.0) | Autocatalytic reaction (product accelerates further reaction) | Autocatalytic drug degradation, self-accelerating solid-state decomposition |

## Multi-Step ODE Models

| Model ID | Model Name | Description | Parameters | Typical Applications |
|----------|------------|-------------|------------|---------------------|
| **A->B->C** | Consecutive reactions | Two-step sequential reactions A→B→C, each step follows f(α) mechanism | Ea1, A1, Ea2, A2, f1_model, f2_model | **Multi-step protein degradation**, intermediate species formation, enzymatic cascades |
| **A+B->C** | Bimolecular | Two species react to form product | Ea, A, initial_ratio_r, m, n | **Protein-ligand binding**, enzyme-substrate reactions, chemical synthesis |

## Empirical Algebraic Models

These models fit conversion directly as a function of time. Their rate
coefficient `k(T)` follows a shared Arrhenius relationship across datasets;
unlike the `f(α)` and ODE models above, they do not specify a reaction
mechanism.

| Model ID | Conversion equation | Description |
|----------|---------------------|-------------|
| **First_Order** | `α = A_scale·exp(-k(T)·t)` | Exponential decay form with fitted amplitude |
| **Linear** | `α = k(T)·t + C` | Linear change with time |
| **Sqrt** | `α = k(T)·√t + C` | Square-root time dependence |
| **Logistic** | `α = A_max / (1 + B·exp(-k(T)·t))` | Sigmoidal curve with fitted upper asymptote |
| **Exponential** | `α = A_amp·(1 - exp(-k(T)·t)) + C` | Exponential approach from a fitted baseline |

These are not included in `models.default`. Select them through
`models.empirical` or combine them with other candidates. See the
[Model Selector Guide](model_selector.md) for examples and the
[Model Selection Guide](model_selection_guide.md) for guidance on when to use
them. Their fitted parameters describe the empirical curve and should not be
interpreted as evidence of a reaction mechanism.

## Model Selection Guide

### By Application Domain

**Pharmaceutical Stability:**
- **Protein therapeutics**: F1, F2 (most common), A->B->C (if intermediates observed)
- **Small molecules**: F1 (primary), F2 (if autocatalytic)
- **Tablets/Solid dosage**: R2, R3 (surface/bulk dissolution)
- **Controlled release**: D3, D4 (diffusion-limited)

**Polymer Degradation:**
- **Chain scission**: F1 (random), F2 (autocatalytic)
- **Oxidation**: R2, R3 (surface/bulk)
- **Crystallization**: A2, A3 (nucleation-growth)
- **Crosslinking**: F2, F3 (multi-molecular)

**Food Science:**
- **Enzyme inactivation**: F1, F2
- **Lipid oxidation**: R2, R3
- **Protein denaturation**: F1 (primary)
- **Microbial inactivation**: F1, A2

**Chemical Engineering:**
- **Catalytic reactions**: F1, F2
- **Solid-state decomposition**: R2, R3, D3
- **Crystallization**: A2, A3
- **Phase transitions**: A2, A3, R3

### By Observed Behavior

**Accelerating Rate** (autocatalytic):
- Try: **F2, F3**, A2, A3, SB_mn, Bna

**Decelerating Rate** (surface reaction):
- Try: **F1**, R2, R3, D2, D3, D4

**Sigmoidal Curve** (nucleation):
- Try: **A2, A3**

**Linear Progress** (zero-order):
- Try: **R2, R3** (contracting geometry)

**Multiple Stages** (intermediates):
- Try: **A->B->C** (consecutive)

## Automated Model Selection

When `models_to_try` is omitted, `auto_model_isothermal_data()` uses
`models.default` from the [Model Selector Guide](model_selector.md). The default
contains the 12 kinetic models in `models.kinetic.all`.

To include other model categories, compose their lists in `models_to_try`:

```python
from akts import models, auto_model_isothermal_data

# Default kinetic models plus both ODE models
models_to_try = models.default + models.ode.all

# Default kinetic models plus empirical models
models_to_try = models.default + models.empirical.all

results = auto_model_isothermal_data(
    data_files=data_files,
    models_to_try=models_to_try
)
```

Friedman model-free isoconversional analysis is never added automatically;
include `'Friedman'` explicitly in `models_to_try` to select it. A meaningful
fit requires at least three distinct mean temperatures, rounded to the nearest
kelvin, and sufficient overlapping conversion range. Its conversion levels
adapt to the conversion range in the data. See
[Model-free prediction](api_reference.md#model-free-prediction-no-reaction-model-assumed).

See the [Model Selector Guide](model_selector.md) for available model groups and
selection examples.

**Ranking and selection:** The default combined score uses normalized BIC (weight 0.4), R² (0.4), RSS (0.1), and parameter count (0.1). AIC is reported but is not part of this score. After ranking, AKTS selects the simplest model among those with BIC less than 2 above the lowest BIC. If no other model meets that threshold, the lowest-BIC model is selected.

The statistics are interpreted as follows:
1. **R²** (coefficient of determination) - higher is better
2. **AIC** (Akaike Information Criterion) - lower is better; reported for comparison
3. **BIC** (Bayesian Information Criterion) - lower is better; also used for the final simplicity comparison
4. **RSS** (Residual Sum of Squares) - lower is better
5. **Parameter count** - fewer parameters are preferred when BIC indicates comparable fit

## Parameter Interpretation

### For Single-Step Models:
- **Ea** (activation energy): Arrhenius temperature-sensitivity parameter, reported in J/mol. Interpret it in the context of the reaction, data range, and fitted model; a threshold alone does not establish whether a fit is physically plausible.
- **A** (pre-exponential factor): Arrhenius rate prefactor, reported in s⁻¹ for single-step fits. Its interpretation depends on the model and parameterization; compare values only when units and model definitions are consistent.

### For A->B->C Model:
- **Ea1, A1**: Parameters for first step (A→B)
- **Ea2, A2**: Parameters for second step (B→C)
- **f1_model**: Mechanism for first step (default: 'F1')
- **f2_model**: Mechanism for second step (default: 'F1')

## References

1. Brown, M. E., et al. (2000). *Handbook of Thermal Analysis and Calorimetry*. Vol. 1.
2. Vyazovkin, S., et al. (2011). *ICTAC Kinetics Committee recommendations*. Thermochimica Acta.
3. Khawam, A. & Flanagan, D. R. (2006). *Solid-state kinetic models*. J. Phys. Chem. B, 110(35).

## Scientific Background

### Rate Law
```
dα/dt = k(T) · f(α)
```

Where:
- `α` = conversion (0 to 1)
- `k(T) = A·exp(-Ea/RT)` = Arrhenius rate constant
- `f(α)` = reaction model function (see table above)
- `T` = absolute temperature (K)
- `R` = gas constant (8.314 J/(mol·K))

### Isothermal Closed-Form Solutions

For **constant temperature**, most models have explicit solutions α(t) = g⁻¹(kt):

| Model | Closed-Form Solution α(t) |
|-------|---------------------------|
| F0 | `α = kt` |
| F1 | `α = 1 - exp(-kt)` |
| F2 | `α = kt/(1+kt)` |
| F3 | `α = 1 - 1/√(1+2kt)` |
| A2 | `α = 1 - exp(-(kt)²)` |
| A3 | `α = 1 - exp(-(kt)³)` |
| R2 | `α = 1 - (1-kt)²` |
| R3 | `α = 1 - (1-kt)³` |
| D2 | Numerical inversion of g(α) = -ln(1-α) - α |
| D3 | `α = 1 - (1-√kt)³` |
| D4 | Numerical inversion of g(α) = 1 - 2α/3 - (1-α)^(2/3) |

For isothermal data, AKTS uses closed-form evaluation where available; for temperature-varying profiles, it uses Numba JIT-compiled ODE integration. Closed-form evaluation is approximately 1000× faster than pure-Python ODE integration in the implementation benchmark; compiled integration is typically 2-5× faster than pure Python.

### Integral Form
```
g(α) = ∫[0→α] dα'/f(α') = A·exp(-Ea/RT)·t
```

For constant-temperature data, AKTS uses closed-form solutions where implemented. For temperature-varying profiles, it numerically integrates the differential rate law.

