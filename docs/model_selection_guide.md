# Model Selection Guide

How to choose the right kinetic model for your system.

> **Note:** This guide covers **which models to try**. For how models are **ranked and selected**, see the [Model Selection](model_selection.md) guide (Akaike weights, filters, plausibility).

## Quick Decision Tree

```
START: What type of experiment?
│
├─ Isothermal (constant temperature)
│  │
│  ├─ Simple degradation → Try F1, F2, A2
│  ├─ Complex/multi-phase → Try A→B→C, A+B→C (ODE models)
│  └─ Autocatalytic → Try A2, A3 (Avrami), Bna (Prout-Tompkins)
│
└─ Non-isothermal (DSC/TGA)
   │
   ├─ Single peak → F1, F2
   ├─ Multiple peaks → A→B→C (consecutive)
   └─ Broad/complex → D2, D3 (diffusion) or SB_mn

Start with `auto_model_isothermal_data()` to compare the default set of models.
```

## Model Categories

### 1. Reaction-Order Models (F-series)

**F0 (Zero-Order)**
- **When to use:** Constant-rate degradation, independent of remaining fraction
- **Characteristics:** Linear conversion vs. time
- **Examples:** Controlled-release drug formulations, zero-order release kinetics

**F1 (First-Order)** - Most common for degradation
- **When to use:** Simple chemical degradation, most pharmaceutical stability
- **Characteristics:** Exponential decay, constant half-life
- **Examples:** Protein aggregation, API degradation, oxidation
- **Equation:** dα/dt = k(1-α)

**F2 (Second-Order)**
- **When to use:** Concentration-dependent reactions
- **Characteristics:** Decay rate slows with time more than F1
- **Examples:** Dimerization, some enzymatic reactions

**F3 (Third-Order)**
- **When to use:** Rare, complex concentration dependencies
- **Characteristics:** Very slow decay at high conversion

[Kinetic models guide](kinetic_models.md) provides the model equations.

### 2. Nucleation & Growth (Avrami)

**A2, A3 (Avrami-Erofeev)**
- **When to use:** Crystallization, autocatalytic reactions
- **Characteristics:** Sigmoidal curves, acceleration period
- **Examples:** Protein crystallization, solid-state transitions, polymer crystallization
- **Equation:** dα/dt = k·n·(1-α)·[-ln(1-α)]^((n-1)/n)

### 3. Contracting Geometry (R-series)

**R2 (Contracting Area)** - 2D interface-controlled
- **When to use:** Surface-limited reactions, thin films
- **Examples:** Oxide layer growth, surface corrosion

**R3 (Contracting Volume)** - 3D interface-controlled
- **When to use:** Particle dissolution, shrinking-core reactions
- **Examples:** Tablet dissolution, particle degradation

### 4. Diffusion-Controlled (D-series)

**D2, D3, D4**
- **When to use:** Mass transport limitations, solid-state diffusion
- **Characteristics:** Very slow at high conversion, often poor fits
- **Examples:** Diffusion through polymer matrix, solid-state reactions
- **Note:** Often indicates experimental artifacts (mass transfer limitations)

### 5. Multi-Step ODE Models

**A→B→C (Consecutive Reactions)**
- **When to use:** Multi-phase degradation, sequential reactions
- **Characteristics:** Two distinct kinetic phases, intermediate formation
- **Examples:** Protein unfolding→aggregation, multi-step decomposition, API→degradant→further degradation
- **Parameters:** Ea1, A1, Ea2, A2 (4 parameters)
- **Fitting:** Uses multistart local optimization (several fits from different starting points, keeping the best by R²) for reliability against a bad local optimum, automatic inside `auto_model_isothermal_data()`

**A+B→C (Bimolecular)**
- **When to use:** Two-component reactions, binding events
- **Characteristics:** Both reactants consumed
- **Examples:** Protein-ligand binding, enzyme-substrate complex, cross-linking reactions
- **Parameters:** Ea, A, initial_ratio, m, n
- **Fitting:** Uses multistart local optimization (several fits from different starting points, keeping the best by R²) for reliability against a bad local optimum, automatic inside `auto_model_isothermal_data()`

[ODE fitting guide](bayesian_optimization.md) describes the fitting procedure.

### 6. Model-Free Isoconversional Analysis

**Friedman** estimates activation energy as a function of conversion, `Ea(α)`,
without assuming a specific reaction model.

- **Use it when:** the reaction mechanism is unknown, may change during the
    reaction, or you want to examine how activation energy varies with conversion.
- **Data needed:** measurements at three or more distinct mean temperatures
    (rounded to the nearest kelvin) that cover overlapping conversion ranges.
- **Interpretation:** changing `Ea(α)` can indicate a changing or multi-step
    process; it does not identify a unique reaction mechanism.
- **Limitations:** the method estimates rates from measured conversion curves,
    so sparse, noisy, or non-overlapping data can make estimates unreliable. It
    may be omitted or fail when there is insufficient usable data.

Use the [model selector](model_selector.md) to request Friedman explicitly with
`models.modelfree.Friedman`. Automated analysis never adds it unless selected.

### Choosing Between ODE and Model-Free Methods

ODE and model-free analyses answer different questions:

| Approach | Use when | Provides | Main limitation |
|----------|----------|----------|-----------------|
| ODE (`A→B→C`, `A+B→C`) | There is a plausible sequential or two-reactant mechanism to test | Mechanism-specific kinetic parameters and predicted behavior | More parameters and slower fitting; needs data informative enough to distinguish the steps |
| Friedman | The mechanism is unknown or may vary with conversion | Activation energy versus conversion without choosing a reaction model | Requires multi-temperature data with adequate, overlapping conversion coverage; does not identify a mechanism |

Prefer ODE models when the proposed reaction structure is supported by
independent chemical knowledge or measurable intermediates. Use Friedman to
characterize conversion-dependent kinetics without imposing that structure.
They can also be used together: compare an ODE model's adequacy with the
conversion dependence in `Ea(α)`. Do not choose between them solely by R²;
consider the data requirements, fitted parameter plausibility, and intended
interpretation.

### 7. Algebraic Empirical Models

The empirical family (`First_Order`, `Linear`, `Sqrt`, `Logistic`, and
`Exponential`) fits conversion directly as a function of time, with shared
Arrhenius temperature dependence across datasets. Unlike the mechanistic
`f(α)` and ODE models, these algebraic forms are not tied to a proposed reaction
mechanism.

- **Use them when:** the goal is a compact description of observed conversion
  curves, or mechanistic models do not describe the measured curve shape.
- **Compare them with mechanistic models:** include both families when the
  mechanism is uncertain and compare residuals, information criteria, and
  parameter plausibility.
- **Interpretation:** treat their fitted parameters as empirical descriptors,
  not direct evidence of a physical reaction mechanism.
- **Extrapolation:** check predictions against independent data or plausible
  physical limits; an algebraic curve that fits the observed range may behave
  poorly outside it.

They are not included in `models.default`. Add them explicitly with the
[model selector](model_selector.md):

```python
from akts import models, auto_model_isothermal_data

results = auto_model_isothermal_data(
    data_files=data_files,
    models_to_try=models.default + models.empirical.all
)
```

### 8. Autocatalytic / Flexible `f(α)` Models

**SB_mn (Sestak-Berggren)**
- **When to use:** Complex or ambiguous mechanisms where F-series/Avrami/contracting-geometry models don't capture the observed shape
- **Characteristics:** General empirical autocatalytic form `α^m·(1-α)^n` (fixed m=0.5, n=1.0 in akts)
- **Examples:** Autocatalytic drug degradation not well described by Bna, mixed-mechanism solid-state reactions

**Bna (Prout-Tompkins)**
- **When to use:** Autocatalytic reactions where the product itself accelerates further reaction
- **Characteristics:** `α^c·(1-α)` (fixed c=1.0 in akts) — sigmoidal, similar in shape to Avrami-Erofeev but derived differently
- **Examples:** Self-accelerating solid-state decomposition, autocatalytic drug degradation

**SB2 (two-step Sestak-Berggren)**
- **When to use:** Two processes acting on one conversion, for example a slow low-temperature process plus a steep high-temperature one (the AKTS commercial two-step form)
- **Equation:** dα/dt = k1(T)·α^m1·(1-α)^n1 + k2(T)·α^m2·(1-α)^n2
- **Variants:** `SB2` fits Ea1, A1, Ea2, A2, m1, n1, m2, n2 (8 parameters); `SB2_grid` is 136 models with integer m, n fixed and only Ea1, A1, Ea2, A2 fitted
- **Caveats:** ODE-integrated and slow; the fast step's A is usually far above 10²⁰ s⁻¹, so SB2 is filtered out when any plausible model reaches the R² threshold, and the plausibility warning appears whenever SB2 is selected. See [SB grid search](SB_GRID_SEARCH.md#two-step-sestak-berggren-sb2).

[Kinetic models guide](kinetic_models.md) provides the full equations.

## Selection Strategy

### Step 1: Start with Automated Analysis

```python
from akts import auto_model_isothermal_data

results = auto_model_isothermal_data(
    data_files=['25C.csv', '40C.csv', '60C.csv'],
    report_path='initial_screen.html'
)

print(f"Top 3 models:")
for model in results['top_models'][:3]:
    print(f"  {model['model_name']}: R² = {model['stats']['r_squared']:.4f}")
```

### Step 2: Interpret Results

**If simple models fit well (R² > 0.95):**
- Prefer the simplest adequate model.
- No need for ODE models
- Done!

**If simple models fit poorly (R² < 0.8):**
- Consider ODE models (A→B→C, A+B→C).
- Add the ODE models to `models_to_try`, for example `models.default + models.ode.all`.
- Check for multi-phase degradation

**If multiple models fit similarly:**
- Look at the Akaike weights of the top models: several with weight > 10% means the data do not clearly separate them
- Check physical meaning and parameter plausibility
- Consider collecting data that discriminates between them (more temperatures, higher conversion)

### How akts Picks the Winning Model

`auto_model_isothermal_data()` filters the fitted models (R² ≥ `min_r_squared`, default 0.70, and physically plausible, with fallbacks described in [Model Selection](model_selection.md)), ranks the survivors by Akaike weight, and selects the rank-1 model. There is no ΔBIC or ΔAIC tie-break toward simpler models: the 2k term in AIC already penalizes complexity.

`results['selected_model']['reason']` states the evidence, e.g.:
- `"Overwhelming evidence (Akaike weight = 97.3%)"` (weight ≥ 90%)
- `"Strong evidence (...)"` (≥ 70%) or `"Substantial support (...)"` (≥ 50%)
- `"Best of 3 competitive models (Akaike weight = 41.0%)"` — model uncertainty; inspect `results['top_models']`

`results['selected_model']['physical_sanity_flags']` is also worth checking: it lists warnings (not exclusions) when a fitted `Ea` falls outside the ~30-180 kJ/mol range typical for drug degradation kinetics (see "Check parameter values" below) — an automatic version of that manual sanity check. This is separate from the plausibility filter, which only rejects Ea outside 5-1000 kJ/mol or A ≥ 10²⁰ s⁻¹. A flagged fit isn't necessarily wrong (diffusion-limited or unusually stable formulations can genuinely fall outside this range), but it's worth a second look.

### Step 3: Validate Choice

**Check Arrhenius plot:**
- Should be linear (straight line)
- Curved = complex mechanism, try ODE models
- Scattered = bad data quality

**Check residuals:**
- Random scatter = good fit
- Systematic patterns = wrong model
- Trends = missing physics

**Check parameter values:**
- Ea = 40-150 kJ/mol (typical)
- Ea < 40 kJ/mol = diffusion-limited
- Ea > 150 kJ/mol = suspect for small-molecule degradation (check units); protein unfolding/denaturation is typically 400-800 kJ/mol
- Ea at 5 or 1000 kJ/mol = at the fitting bound; `fit_kinetic_model` warns, and the true optimum may lie outside
- `auto_model_isothermal_data()` flags this automatically in `results['selected_model']['physical_sanity_flags']` (using a slightly wider 30-180 kJ/mol plausible range) — see "How akts Picks the Winning Model" above

## Application-Specific Guidance

### Pharmaceutical Stability

**Typical progression:**
1. Start with F1 (first-order) - most common
2. If poor fit, try F2 (second-order)
3. If multi-phase visible, try A→B→C

**Common models:**
- API degradation: F1, F2
- Protein aggregation: F1, A2, A→B→C
- Oxidation: F1
- Hydrolysis: F1, F2

### Polymer Degradation

**Thermal decomposition:**
- Single-step: F1, F2
- Multi-step: A→B→C
- Autocatalytic: A2, A3

**Oxidative degradation:**
- Initial: F1
- Accelerated: A2 (autocatalytic)

### Protein Stability

**Unfolding/Aggregation:**
- Simple: F1
- With intermediate: A→B→C (native→unfolded→aggregate)
- Autocatalytic: A2

**Binding studies:**
- Protein-ligand: A+B→C (bimolecular)

## Common Mistakes

### Avoid choosing a model by name alone

Do not select R2 or R3 solely because the system involves a reaction. Compare
multiple candidate models against the data.

### Use ODE models when justified by the data

ODE models:
- 4+ parameters (overfitting risk)
- 5-7 minutes to fit
- Only justified if data shows multi-phase

### Check physical plausibility

Example: Ea = 300 kJ/mol is unusual for small-molecule chemical degradation, but not for protein unfolding. Judge Ea against the process you expect.

### Use diffusion models with supporting evidence

Diffusion models rarely apply to:
- Liquid-phase reactions
- Well-mixed systems
- Homogeneous degradation

## Model Comparison Metrics

### R² (Coefficient of Determination)

- Range: -∞ to 1.0
- Interpretation:
  - R² > 0.95: Excellent fit
  - R² = 0.8-0.95: Good fit
  - R² = 0.5-0.8: Poor fit, try other models
  - R² < 0.5: Very poor, wrong model
  - R² < 0: Worse than horizontal line!

### AIC (Akaike Information Criterion)

- Lower is better
- Penalizes additional parameters
- Use for comparing models with different parameters

- akts uses the small-sample corrected AICc

### Akaike Weight

- w_i = exp(-0.5·ΔAICc_i) / Σ exp(-0.5·ΔAICc_j)
- Probability that a model is the best in the candidate set; weights sum to 1
- **akts ranks models by Akaike weight only** (after the R² and plausibility filters)

### BIC (Bayesian Information Criterion)

- Lower is better
- Penalizes parameters more than AIC
- Reported for reference; not used for ranking

## Advanced: Custom Model Selection

If auto_model_isothermal_data() doesn't work:

```python
from akts import discover_kinetic_models, fit_kinetic_model

# Try specific models
models_to_try = [
    {'name': 'F1', 'type': 'single_step', 'def_args': {'f_alpha_model': 'F1'}},
    {'name': 'F2', 'type': 'single_step', 'def_args': {'f_alpha_model': 'F2'}},
    {'name': 'A->B->C', 'type': 'A->B->C', 'def_args': {'f1_model': 'F1', 'f2_model': 'F1'}}
]

ranked = discover_kinetic_models(
    datasets=datasets,
    models_to_try=models_to_try,
    initial_guesses_pool={...},
    parameter_bounds_pool={...},
    min_r_squared=0.70,   # R² filter (default)
    apply_filters=True    # R² and plausibility filters (default)
)
```

[API reference](api_reference.md) provides details.

## Decision Checklist

Before finalizing model choice:

- [ ] Tried automated analysis first?
- [ ] Checked R² > 0.90?
- [ ] Verified Arrhenius plot is linear?
- [ ] Confirmed residuals are random?
- [ ] Checked parameters are physically reasonable?
- [ ] Checked the Akaike weight (low weight = model uncertainty)?
- [ ] Only used ODE models if necessary (R² < 0.8 for simple models)?
- [ ] Validated with independent data (if available)?

## Further Reading

- **[Kinetic Models Guide](kinetic_models.md)** - Model equations and mechanisms
- **[Getting Started](getting_started.md)** - First analysis
- **[Examples](examples.md)** - Working code with model selection
- **[Experimental Design](experimental_design.md)** - Design for model discrimination

---

Use `auto_model_isothermal_data()` as a starting point, then assess fit quality,
parameter plausibility, and the physical basis for the selected model.
