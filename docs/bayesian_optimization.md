# ODE Model Fitting

> **Note:** This document previously described a Bayesian-optimization (Gaussian
> process / `scikit-optimize`) approach for fitting the ODE models (`A->B->C`,
> `A+B->C`). That approach has been removed from the codebase: `akts/bayesian_opt.py`
> no longer exists, and `scikit-optimize` is no longer a dependency (see
> `requirements.txt` / `pyproject.toml`). This page now describes the approach
> that replaced it.

## Fitting Challenges

ODE models (`A->B->C`, `A+B->C`) are the slowest and hardest-to-fit models in akts,
because each objective-function evaluation requires numerically integrating an ODE
system, and the fitted parameters (`Ea`, `A`) are highly correlated. A single fit
from a single starting point can land in a poor local optimum, or run for a long
time on a difficult region of parameter space.

## Fitting Approach

### 1. Arrhenius Reparameterization (in `fit_kinetic_model()`)

`Ea` and `log(A)` are almost perfectly correlated in the Arrhenius equation and
differ in magnitude by orders of magnitude, which makes gradient-style optimizers
stall right at the initial guess. `akts/core.py`'s `fit_kinetic_model()` internally
reparameterizes each `(Ea, logA)` pair into `(Ea/scale, ln k(T_ref))`, where `T_ref`
is the mean temperature across the datasets — `ln k(T_ref)` is far less correlated
with `Ea` than `logA` is. This is handled by the `_ArrheniusReparam` class and is
transparent to callers: you still pass and receive plain `Ea`/`A` values.

### 2. Coarse Grid-Scan Starting Point

Before the main optimization, `_ArrheniusReparam.coarse_start()` scans candidate
values of `ln k(T_ref)` (25 evaluations per Arrhenius pair) to find a starting point
inside the basin of the true optimum, rather than starting from a value where the
objective is flat (e.g. the model predicts either "fully reacted" or "not reacted
at all" everywhere).

### 3. Powell Optimization Method

`fit_kinetic_model()` now defaults to the derivative-free Powell method instead of
L-BFGS-B. The ODE solver's own tolerance makes the objective function noisy at
small scales, which makes finite-difference gradients (used by L-BFGS-B) unreliable
and prone to stalling early; Powell does not depend on gradients.

### 4. Wall-Clock Deadline (`optimizer_options['max_seconds']`)

A single `fit_kinetic_model()` call is now bounded by a wall-clock deadline instead
of only an evaluation-count cap, because ODE objective calls vary widely in cost
depending on the parameter region. The default budget is `60 seconds * number of
parameters`, which can be overridden via `optimizer_options={'max_seconds': ...}`.
This mainly matters for ODE models with 4+ parameters (e.g. `A->B->C`), where a
single fit could otherwise run for a long time from an unlucky starting point.

### 5. Multistart Local Fitting (`akts/helpers.py`'s `_multistart_fit()`)

Because the reparameterization, coarse scan, Powell method, and deadline improve
individual fit behavior, `auto_model_isothermal_data()` uses multistart local
fitting instead of Bayesian optimization for ODE models. The helper
`_multistart_fit()` runs `fit_kinetic_model()` from several starting points — one
unperturbed guess plus a few randomly perturbed ones (perturbing log-scaled
parameters like `A` multiplicatively, and other parameters like `Ea` by a random
scale factor) — and keeps whichever result has the best R². This is used for the
ODE models (`A->B->C`, `A+B->C`) and is a limited local search, not an exhaustive
global optimization.

## Usage

Multistart fitting is used automatically for ODE models inside
`auto_model_isothermal_data()` — there is nothing to opt into:

```python
from akts import auto_model_isothermal_data, models

results = auto_model_isothermal_data(
    data_files=['data_25C.csv', 'data_40C.csv', 'data_60C.csv'],
    models_to_try=models.default + models.ode.all,
    report_path='report.html'
)
```

When calling `fit_kinetic_model()` directly for an ODE model, fit from several
`initial_guesses` and retain the result with the highest R². The internal
`akts.helpers._multistart_fit()` helper is not part of the public API.

### Bounding fit time directly

If a single ODE fit is taking too long, tighten the wall-clock budget explicitly
rather than reaching for a different optimizer:

```python
from akts import fit_kinetic_model

fit_result = fit_kinetic_model(
    datasets=datasets,
    model_name='A->B->C',
    model_definition_args={'f1_model': 'F1', 'f2_model': 'F1'},
    initial_guesses={'Ea1': 90000, 'A1': 1e12, 'Ea2': 110000, 'A2': 1e13},
    parameter_bounds={
        'Ea1': (10000, 300000), 'A1': (1e3, 1e20),
        'Ea2': (10000, 300000), 'A2': (1e3, 1e20)
    },
    optimizer_options={'max_seconds': 60}  # override the 60s/parameter default
)
```

## Migration Notes (from the removed Bayesian-optimization approach)

- `scikit-optimize` is no longer a dependency and is not used by AKTS.
- `akts.bayesian_opt` and `bayesian_optimize_ode_model()` **no longer exist**.
  Code importing from `akts.bayesian_opt` will raise `ModuleNotFoundError` and
  should call `fit_kinetic_model()` or
  `auto_model_isothermal_data()` as shown above.
- There is no `should_use_bayesian_opt()` to override — ODE models always use
  multistart fitting inside `auto_model_isothermal_data()`, and there is no
  scikit-optimize fallback path to worry about.

## Troubleshooting

### ODE model fit is slow or times out

1. Reduce the wall-clock budget so a bad start fails fast rather than running the
   default `60s * n_parameters`: `optimizer_options={'max_seconds': 30}`.
2. Reduce the number of multistart attempts if you're calling the internal
   multistart helper directly (default is 1 unperturbed + 3 perturbed starts).
3. Tighten `parameter_bounds` — an overly wide search space makes both the coarse
   grid-scan and Powell's search less effective.
4. Omit ODE models if they aren't needed by setting `models_to_try=models.default`.

### "ModuleNotFoundError: No module named 'akts.bayesian_opt'"

This module was removed. Update any code that imported
`bayesian_optimize_ode_model` from `akts.bayesian_opt` to instead call
`fit_kinetic_model()` (optionally with `optimizer_options={'max_seconds': ...}`),
which now includes the reparameterization and coarse-start logic that Bayesian
optimization was previously compensating for.
