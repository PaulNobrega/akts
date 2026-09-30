# Advanced Usage

Custom workflows for users who need control over individual analysis steps.

For routine analyses, use [auto_model_isothermal_data()](automated_analysis.md).

**Use advanced workflows when you need:**
- Custom model definitions
- Specific optimization strategies
- Integration with existing pipelines
- Batch processing automation
- Fine control over every step
- A different ODE solver or reproducible bootstraps ([Numerical Solvers](#numerical-solvers-and-performance))

## Manual Step-by-Step Workflow

### Complete Example

```python
from akts import fit_kinetic_model, run_bootstrap, predict_conversion
from akts.loaders import load_isothermal_file
from akts.utils import construct_profile

# 1. Load data
datasets = []
for temp in [313, 323, 333, 343]:
    dataset = load_isothermal_file(
        f'data_{temp}K.csv',
        input_temperature_units='K'
    )
    datasets.append(dataset)

# 2. Fit model
fit_result = fit_kinetic_model(
    datasets=datasets,
    model_name='single_step',
    model_definition_args={'f_alpha_model': 'F1'},
    initial_guesses={'Ea': 85000, 'A': 1e11},
    parameter_bounds={'Ea': (50000, 150000), 'A': (1e7, 1e14)}
)

# 3. Bootstrap confidence intervals
if __name__ == '__main__':  # Required on Windows
    bootstrap_result = run_bootstrap(
        datasets=datasets,
        fit_result=fit_result,
        optimizer_options={'method': 'L-BFGS-B'},
        parameter_bounds={'Ea': (50000, 150000), 'A': (1e7, 1e14)},
        n_iterations=100,
        n_jobs=-1
    )

# 4. Predict with custom temperature profile
segments = [
    {'type': 'isothermal', 'duration': 2*365*24*3600, 'temperature': 278}
]
time_s, temp_K = construct_profile(segments)

prediction = predict_conversion(
    kinetic_description=fit_result,
    temperature_program=(time_s, temp_K),
    bootstrap_result=bootstrap_result
)

print(f"Final conversion: {prediction.conversion[-1]:.1%}")
```

## ODE Models (A→B→C and A+B→C)

### A→B→C (Consecutive Reactions)

```python
from akts import fit_kinetic_model

fit_result = fit_kinetic_model(
    datasets=datasets,
    model_name='A->B->C',
    model_definition_args={'f1_model': 'F1', 'f2_model': 'F1'},
    initial_guesses={
        'Ea1': 85000, 'A1': 1e11,  # A→B
        'Ea2': 95000, 'A2': 1e12   # B→C
    }
)
```

For ODE models, `fit_kinetic_model()` reparameterizes `(Ea, logA)` pairs to
reduce correlation, scans candidate starting values, and then optimizes with
Powell. See the [multistart fitting guide](bayesian_optimization.md) for details.

When ODE models are included in `models_to_try`,
`auto_model_isothermal_data()` runs fits from perturbed starting points and
retains the result with the highest R². Use `models.default + models.ode.all` to
select the default kinetic models and ODE models. This excludes empirical and
model-free models; use `models.all` to select every model category. See the
[Model Selector Guide](model_selector.md) for details.

Direct calls to `fit_kinetic_model()` do not run multistart automatically. To
apply a similar strategy, run several fits from perturbed initial guesses and
retain the successful result with the highest R²:

```python
import numpy as np

best_result = None
for _ in range(4):
    guesses = {k: v * np.random.uniform(0.6, 1.4) for k, v in {
        'Ea1': 85000, 'A1': 1e11, 'Ea2': 95000, 'A2': 1e12
    }.items()}
    result = fit_kinetic_model(
        datasets=datasets,
        model_name='A->B->C',
        model_definition_args={'f1_model': 'F1', 'f2_model': 'F1'},
        initial_guesses=guesses
    )
    if result.success and (best_result is None or result.r_squared > best_result.r_squared):
        best_result = result
```

### A+B→C (Bimolecular)

```python
fit_result = fit_kinetic_model(
    datasets=datasets,
    model_name='A+B->C',
    model_definition_args={
        'bimol_params': {
            'initial_ratio_r': 1.0,  # [B]₀/[A]₀
            'm': 1.0,  # Order in A
            'n': 1.0   # Order in B
        }
    },
    initial_guesses={'Ea': 85000, 'A': 1e11}
)
```

In the automated workflow, set the fixed ratio with the top-level `initial_ratio_r` parameter (default `1.0`). See [Automated Analysis](automated_analysis.md).

## Batch Processing

```python
import glob
from pathlib import Path
from akts import auto_model_isothermal_data, models
import pandas as pd

# Process multiple formulations
formulations = glob.glob('data/formulation_*/')

results_summary = []

for formulation_dir in formulations:
    name = Path(formulation_dir).stem
    csv_files = list(Path(formulation_dir).glob('*.csv'))

    results = auto_model_isothermal_data(
        data_files=csv_files,
        predict=(12, 'month'),
        models_to_try=models.default,
        bootstrap_iterations=50,
        report_path=f'reports/{name}_report.html'
    )

    results_summary.append({
        'Formulation': name,
        'Model': results['selected_model']['model_name'],
        'Ea (kJ/mol)': results['selected_model']['parameters']['Ea']/1000,
        'R²': results['selected_model']['statistics']['r_squared'],
        '12mo Conversion': results['predictions']['conversion_mean'][-1]
    })

# Save summary
df = pd.DataFrame(results_summary)
df.to_csv('formulation_summary.csv', index=False)
print(df)
```

## Custom Model Comparison

```python
from akts import discover_kinetic_models

models_to_try = [
    {'name': 'F1', 'type': 'single_step', 'def_args': {'f_alpha_model': 'F1'}},
    {'name': 'F2', 'type': 'single_step', 'def_args': {'f_alpha_model': 'F2'}},
    {'name': 'A2', 'type': 'single_step', 'def_args': {'f_alpha_model': 'A2'}}
]

initial_guesses_pool = {
    'F1': {'Ea': 85000, 'A': 1e11},
    'F2': {'Ea': 88000, 'A': 5e11},
    'A2': {'Ea': 90000, 'A': 1e12}
}

ranked_models = discover_kinetic_models(
    datasets=datasets,
    models_to_try=models_to_try,
    initial_guesses_pool=initial_guesses_pool
)

for model in ranked_models:
    print(f"{model['rank']}. {model['model_name']}: R²={model['stats']['r_squared']:.4f}")
```

## JSON API Integration

```python
from flask import Flask, request, jsonify
from akts import auto_model_isothermal_data

app = Flask(__name__)

@app.route('/analyze', methods=['POST'])
def analyze():
    json_data = request.json
    datasets = json_data['datasets']

    results = auto_model_isothermal_data(
        data_files=datasets,
        output_format='json',
        report_path=None
    )

    return jsonify(results)

if __name__ == '__main__':
    app.run()
```

[JSON I/O specification](json_io_specification.md)

## Custom Temperature Profiles

```python
from akts.utils import construct_profile

# Multi-segment shipping profile
segments = [
    # Storage at 5°C
    {'type': 'isothermal', 'duration': 30*24*3600, 'temperature': 278},
    # Ramp to 25°C (shipping)
    {'type': 'ramp', 'duration': 3600, 'start_temp': 278, 'end_temp': 298},
    # At 25°C for 7 days
    {'type': 'isothermal', 'duration': 7*24*3600, 'temperature': 298},
    # Back to 5°C
    {'type': 'ramp', 'duration': 3600, 'start_temp': 298, 'end_temp': 278},
    # Long-term storage
    {'type': 'isothermal', 'duration': 2*365*24*3600, 'temperature': 278}
]

time_s, temp_K = construct_profile(segments, points_per_segment=100)

prediction = predict_conversion(
    kinetic_description=fit_result,
    temperature_program=(time_s, temp_K),
    bootstrap_result=bootstrap_result
)
```

## Progress Callbacks

```python
def my_callback(message, data):
    timestamp = data.get('timestamp', '')
    print(f"[{timestamp}] {message}")

results = auto_model_isothermal_data(
    data_files=files,
    progress_callback=my_callback
)
```

## Numerical Solvers and Performance

Every model that is not solved in closed form is integrated with
[`scipy.integrate.solve_ivp`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html).
You can choose the integration method through the `solver_options` dictionary,
which is accepted by `fit_kinetic_model()`, `run_bootstrap()`,
`predict_conversion()`, `predict_conversion_model_free()` and `simulate_kinetics()`.

### Closed-form fast path (no ODE solve)

Single-step models with an analytic isothermal solution are evaluated exactly
instead of being integrated:

| Models | When it applies |
|---|---|
| F0, F1, F2, F3, A2, A3, R2, R3, D2, D3, D4 | Fitting and bootstrap: all datasets are isothermal (temperature std < 0.5 K). Prediction: constant temperature program and `initial_alpha=0`. |

This is automatic and needs no configuration. It is the reason a typical
isothermal stability fit takes a fraction of a second. Models without a closed
form (Fn, SB, SB_mn, SB_mnp, Bna, D1, A->B->C, A+B->C), and any non-isothermal
data, use the ODE path below.

### Available solvers

| Method | Type | Use when |
|---|---|---|
| `'LSODA'` (**default primary**) | Automatic stiff/non-stiff switching (Adams ↔ BDF) | General purpose. Handles both the benign and the stiff parameter regions an optimizer visits. |
| `'RK45'` (**default fallback**) | Explicit Runge-Kutta 5(4) | Non-stiff problems. Robust fallback when LSODA reports failure. |
| `'RK23'` | Explicit Runge-Kutta 3(2) | Non-stiff problems at loose tolerances. |
| `'DOP853'` | Explicit Runge-Kutta 8 | Non-stiff problems at very tight tolerances. |
| `'BDF'` | Implicit backward differentiation | Stiff problems, e.g. A->B->C with rate constants many orders of magnitude apart. |
| `'Radau'` | Implicit Runge-Kutta (Radau IIA, order 5) | Stiff problems that need high accuracy. |

Each ODE solve tries the **primary** solver first and, only if that fails, the
**fallback** solver. A warning is issued when the fallback is used.

### `solver_options` reference

| Key | Default | Meaning |
|---|---|---|
| `primary_solver` | `'LSODA'` | Method tried first. |
| `fallback_solver` | `'RK45'` | Method tried if the primary fails. Set to `None` to disable the fallback. |
| `rtol` | `1e-6` | Relative tolerance. |
| `atol` | `1e-9` | Absolute tolerance. |
| `max_rhs_evals` | `20000` | Per-solve cap on right-hand-side evaluations. A pathological parameter set fails fast instead of hanging the optimizer. |
| `chunk_size` | `2000` | Predictions longer than this many time points are integrated in chunks. |

Any key you omit keeps its default (see `akts.simulation.DEFAULT_SOLVER_OPTIONS`).

```python
from akts import fit_kinetic_model, predict_conversion

# Stiff consecutive reaction: implicit solvers for both attempts
fit = fit_kinetic_model(
    datasets, 'A->B->C', {'f1_model': 'F1', 'f2_model': 'F1'},
    initial_guesses={'Ea1': 85e3, 'A1': 1e11, 'Ea2': 95e3, 'A2': 1e12},
    solver_options={'primary_solver': 'BDF', 'fallback_solver': 'Radau'},
)

# Previous default order (RK45 first, LSODA as fallback)
prediction = predict_conversion(
    fit, temperature_program=lambda t: 298.15, simulation_time_sec=t_eval,
    solver_options={'primary_solver': 'RK45', 'fallback_solver': 'LSODA'},
)

# Single solver, no fallback, tighter tolerances
opts = {'primary_solver': 'LSODA', 'fallback_solver': None, 'rtol': 1e-8, 'atol': 1e-11}
```

Changing the solver changes fitted parameters only within the integration
tolerance. If two solvers give noticeably different fits, tighten `rtol`/`atol`
until they agree.

### Optimizer time budget

`fit_kinetic_model(..., optimizer_options={'max_seconds': 120})` caps the
wall-clock time of one fit. The default is 60 s per fitted parameter. In
`run_bootstrap(..., timeout_per_replicate=30)` the timeout also becomes each
replicate's optimizer budget.

### Reproducible bootstraps

Bootstrap resampling is random. Pass `random_state` to get identical results
on every run, independent of `n_jobs` or the order in which workers finish:

```python
boot = run_bootstrap(datasets, fit, n_iterations=200, random_state=12345)

results = auto_model_isothermal_data(files, predict=(2, 'year'), random_state=12345)
```

`random_state` accepts an `int`, a `numpy.random.SeedSequence` or a
`numpy.random.Generator`. `run_bootstrap_friedman()` accepts it as well.

## See Also

- **[API Reference](api_reference.md)** - Complete function reference
- **[Automated Analysis](automated_analysis.md)** - High-level workflow
- **[Examples](examples.md)** - Working code with data
- **[JSON I/O Specification](json_io_specification.md)** - Web API details
- **[Multistart Fitting](bayesian_optimization.md)** - Reliability approach for ODE models

---

For routine analyses, use `auto_model_isothermal_data()`. Use the manual
workflows in this guide when individual fitting or prediction steps need to be
controlled directly.
