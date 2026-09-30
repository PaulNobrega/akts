# Model Selector Guide

The AKTS model selector provides IDE autocomplete and type-safe model selection through structured dot notation.

---

## Quick Start

```python
from akts import models, auto_model_isothermal_data

# Select models with IDE autocomplete
results = auto_model_isothermal_data(
    data_files=['25C.csv', '40C.csv', '60C.csv'],
    models_to_try=models.kinetic.all,
    ...
)
```

---

## Model Categories

### 1. Kinetic (Mechanistic) Models

Solid-state kinetic models based on reaction mechanisms:

```python
from akts import models

# Individual models (each returns a list)
models.kinetic.F0        # ['F0'] - Zero-order
models.kinetic.F1        # ['F1'] - First-order
models.kinetic.F2        # ['F2'] - Second-order
models.kinetic.F3        # ['F3'] - Third-order
models.kinetic.A2        # ['A2'] - Avrami-Erofeev n=2
models.kinetic.A3        # ['A3'] - Avrami-Erofeev n=3
models.kinetic.R2        # ['R2'] - Contracting area
models.kinetic.R3        # ['R3'] - Contracting volume
models.kinetic.D2        # ['D2'] - 2D diffusion
models.kinetic.D3        # ['D3'] - 3D diffusion (Jander)
models.kinetic.SB_mn     # ['SB_mn'] - Sestak-Berggren
models.kinetic.Bna       # ['Bna'] - Prout-Tompkins

# All kinetic models
models.kinetic.all       # All 12 models

# Sub-categories
models.kinetic.nth_order      # ['F0', 'F1', 'F2', 'F3']
models.kinetic.nucleation     # ['A2', 'A3']
models.kinetic.diffusion      # ['D2', 'D3']
models.kinetic.autocatalytic  # ['SB_mn', 'Bna']
```

### 2. Empirical Models

Direct algebraic fits with global Arrhenius temperature dependence:

```python
# Individual models
models.empirical.First_Order  # ['First_Order'] - α = A·exp(-k(T)·t)
models.empirical.Linear       # ['Linear'] - α = k(T)·t + C
models.empirical.Sqrt         # ['Sqrt'] - α = k(T)·√t + C
models.empirical.Logistic     # ['Logistic'] - α = A/(1+B·exp(-k(T)·t))
models.empirical.Exponential  # ['Exponential'] - α = A·(1-exp(-k(T)·t))+C

# All empirical models
models.empirical.all          # All 5 empirical models
```

To try empirical models, include them in `models_to_try`, for example
`models_to_try=models.empirical.all`.

### 3. ODE Models

Multi-step reaction models (slower but more flexible):

```python
models.ode.consecutive   # ['A->B->C'] - Consecutive reactions
models.ode.bimolecular   # ['A+B->C'] - Bimolecular reaction
models.ode.all           # Both ODE models
```

To try ODE models, include them in `models_to_try`, for example
`models_to_try=models.default + models.ode.all`.

### 4. Model-Free Methods

Isoconversional analysis (no reaction model assumed):

```python
models.modelfree.Friedman  # ['Friedman'] - Friedman differential method
models.modelfree.all       # All model-free methods
```

**Note**: Friedman is opt-in; add `models.modelfree.Friedman` to `models_to_try`.
Meaningful fits require at least three distinct temperatures with overlapping
conversion ranges.

---

## All Models

```python
models.all  # Everything: kinetic + empirical + ODE + model-free
```

---

## Safe Concatenation

All model attributes return lists, which can be concatenated directly.

```python
# Examples:
models_to_try = models.kinetic.F1 + models.kinetic.A2
# Result: ['F1', 'A2']

models_to_try = models.empirical.all + models.kinetic.F0
# Result: ['First_Order', 'Linear', 'Sqrt', 'Logistic', 'Exponential', 'F0']

models_to_try = models.kinetic.nth_order + models.empirical.Linear
# Result: ['F0', 'F1', 'F2', 'F3', 'Linear']

# No brackets needed - each attribute returns a list!
```

**Before** (old way - error-prone):
```python
# Had to remember exact strings
models_to_try = ['F1', 'A2']  # Could typo: 'f1', 'F_1', etc.

# Concatenation required manual list construction
models_to_try = ['First_Order'] + ['F0']
```

**After** (model selector - safe):
```python
# IDE autocomplete shows all options
models_to_try = models.kinetic.F1 + models.kinetic.A2

# Safe concatenation
models_to_try = models.empirical.all + models.kinetic.F0  # Works!
```

---

## Usage Examples

### Example 1: Use All Mechanistic Models

```python
from akts import models, auto_model_isothermal_data

results = auto_model_isothermal_data(
    data_files=['25C.csv', '40C.csv', '60C.csv'],
    models_to_try=models.kinetic.all,  # All 12 mechanistic models
    predict=(2, 'year', 298.15),
    report_path='report.html'
)
```

### Example 2: Compare Specific Models

```python
# Compare first-order and Avrami-Erofeev
results = auto_model_isothermal_data(
    data_files=files,
    models_to_try=models.kinetic.F1 + models.kinetic.A2,
    ...
)
```

### Example 3: Mix Mechanistic and Empirical

```python
# Try mechanistic nth-order plus empirical linear
results = auto_model_isothermal_data(
    data_files=files,
    models_to_try=models.kinetic.nth_order + models.empirical.Linear,
    ...
)
```

### Example 4: Use Sub-Categories

```python
# Only autocatalytic models
results = auto_model_isothermal_data(
    data_files=files,
    models_to_try=models.kinetic.autocatalytic,  # SB_mn, Bna
    ...
)
```

### Example 5: Everything

```python
# Try all available models
results = auto_model_isothermal_data(
    data_files=files,
    models_to_try=models.all,  # All models
    ...
)
```

---

## Benefits

### 1. IDE Autocomplete

Type `models.` and your IDE shows all available model groups:
- `models.kinetic`
- `models.empirical`
- `models.ode`
- `models.modelfree`
- `models.all`

Type `models.kinetic.` and see all kinetic models with docstrings.

### 2. No Typos

**Before**: `'F1'` could be mistyped as `'f1'`, `'F_1'`, `'F-1'`
**After**: `models.kinetic.F1` is validated by Python - typos cause immediate errors

### 3. Discoverability

No need to look up documentation to find available models. Browse them in your IDE!

### 4. Type Safety

Returns correct model strings guaranteed. No string manipulation errors.

### 5. Logical Grouping

Models organized by:
- **Mechanism** (kinetic, empirical, ODE)
- **Sub-category** (nth-order, nucleation, diffusion, autocatalytic)
- **All**

---

## Advanced Usage

### Aliases

```python
# 'mechanistic' is an alias for 'kinetic'
models.mechanistic.all == models.kinetic.all  # True
```

### Default Models

```python
# Get default models used by auto_model_isothermal_data()
models.default  # Same as models.kinetic.all
```

### Dynamic Selection

```python
# Build model list dynamically
user_choice = "empirical"
if user_choice == "empirical":
    models_to_try = models.empirical.all
elif user_choice == "kinetic":
    models_to_try = models.kinetic.all
else:
    models_to_try = models.all
```

---

## Migration from String Lists

### Old Code (Still Works)

```python
# Still valid - backward compatible
results = auto_model_isothermal_data(
    data_files=files,
    models_to_try=['F1', 'F2', 'A2'],  # Old way
    ...
)
```

### New Code (Recommended)

```python
# New way - type-safe and autocomplete
results = auto_model_isothermal_data(
    data_files=files,
    models_to_try=models.kinetic.F1 + models.kinetic.F2 + models.kinetic.A2,
    ...
)
```

---

## API Reference

### Model Selector Class

```python
from akts import models

# Top-level instance
models = _ModelSelector()

# Model groups (properties)
models.kinetic       # _KineticModels instance
models.empirical     # _EmpiricalModels instance
models.ode           # _ODEModels instance
models.modelfree     # _ModelFree instance
models.mechanistic   # Alias for kinetic
models.all           # List[str] - all models
models.default       # List[str] - default models
```

### Model Group Properties

All model attributes return `List[str]`:

```python
models.kinetic.F1    # Returns: ['F1']
models.kinetic.all   # Returns: ['F0', 'F1', 'F2', ...]
```

This design enables safe concatenation:
```python
models.kinetic.F1 + models.empirical.Linear  # Concatenate model lists
```

---

## FAQ

**Q: Why do single models return lists?**

A: To enable safe concatenation. If `models.kinetic.F0` returned `'F0'` (string), then `models.empirical.all + models.kinetic.F0` would fail (list + string). By returning `['F0']`, concatenation always works.

**Q: Can I still use string lists?**

A: Yes! The old way still works for backward compatibility:
```python
models_to_try = ['F1', 'A2', 'R3']  # Still valid
```

**Q: How do I include empirical or ODE models?**

A: Add their selector lists to `models_to_try`, or use `models.all` to select
every model category. Friedman is not added automatically; include
`models.modelfree.Friedman` when you want to run it.

**Q: What if I mistype a model name?**

A: Your IDE will show an error immediately. `models.kinetic.F99` doesn't exist, so Python catches it before running.

**Q: Can I create custom model groups?**

A: Custom groups are not currently supported. Combine model lists directly:
```python
my_favorites = models.kinetic.F1 + models.empirical.Linear + models.kinetic.A2
```

---

## See Also

- [Kinetic Models Guide](kinetic_models.md) - Model equations and mechanisms
- [Model Selection Guide](model_selection_guide.md) - Choosing the right model
- [Automated Analysis Guide](automated_analysis.md) - Using auto_model_isothermal_data()
- [Automated analysis example](../Examples/Isothermal/demo_auto_model.py) - Uses the model selector

---

