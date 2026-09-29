# Isothermal Examples: Quick Reference

The four scripts use generated datasets to demonstrate common isothermal kinetic-analysis workflows. Each script writes its report and other outputs to the `output/` directory.

## Choose an Example

| Example | Application | Main topics |
|----------|-------------|-------------|
| [01 Pharmaceutical Shelf Life](01_pharmaceutical_shelf_life.py) | Drug stability | ICH Q1E, shelf-life estimates, one-sided confidence intervals |
| [02 Protein Aggregation](02_protein_aggregation_autocatalytic.py) | Protein formulation | Autocatalytic kinetics, model comparison, temperature excursions |
| [03 Food Quality](03_food_quality_vitamin_degradation.py) | Vitamin C degradation | First-order kinetics, Q10, tabulated exports |
| [04 Polymer Oxidation](04_polymer_oxidation_complex.py) | Polymer aging | Consecutive reactions, ODE fitting, service-life prediction |

## Run an Example

Run commands from the repository root:

```bash
python Examples/Isothermal/01_pharmaceutical_shelf_life.py
python Examples/Isothermal/02_protein_aggregation_autocatalytic.py
python Examples/Isothermal/03_food_quality_vitamin_degradation.py
python Examples/Isothermal/04_polymer_oxidation_complex.py
```

## Select Models

Use the model selector to choose candidate models. `models.default` contains the default kinetic models; add other categories explicitly.

```python
from akts import auto_model_isothermal_data, models

results = auto_model_isothermal_data(
    data_files=['data_25C.csv', 'data_40C.csv', 'data_60C.csv'],
    models_to_try=models.default,
    bootstrap_iterations=100,
    report_path='analysis.html'
)
```

To include ODE models, use `models.default + models.ode.all`. To include empirical models, use `models.default + models.empirical.all`. `models.all` selects every model category. See the [Model Selector Guide](../../docs/model_selector.md) for details.

## Adjust Runtime

For a shorter development run, use fewer bootstrap iterations and omit ODE models:

```python
bootstrap_iterations=20  # Fewer samples; confidence-interval estimates are less stable
models_to_try=models.default
```

For a more thorough analysis, increase the number of bootstrap samples and include ODE models when justified by the data:

```python
bootstrap_iterations=200
models_to_try=models.default + models.ode.all
```

Control bootstrap parallelism with `n_jobs`. The default, `-1`, uses all but one available CPU core; the worker count is capped at the number of bootstrap iterations. Set `n_jobs=1` to use one worker process.

Runtime depends on the dataset, selected models, hardware, bootstrap configuration, and worker count.

## Output

Generated files are written under `Examples/Isothermal/output/`. Filenames and available exports vary by example. HTML reports may include model comparisons, fit and Arrhenius plots, predictions, confidence intervals, and regulatory analysis where applicable.

## Common Adjustments

### Prediction Conditions

The optional third value in `predict` is the prediction temperature, in the temperature units specified by `input_temperature_units`:

```python
predict=(2, 'year', 25)  # Two years at 25 degrees Celsius when input units are C
```

Supported time units include seconds, minutes, hours, days, weeks, months, and years.

### Temperature Excursions

Supply time-temperature points through `simulate`. The time unit and temperature units must match the corresponding input settings:

```python
simulate=[
    (0, 25),
    (5, 40),
    (10, 25),
],
simulate_time_unit='days',
input_temperature_units='C',
```

### Report Format

```python
report_format='interactive'  # Interactive Plotly report
report_format='static'       # Static plots
report_format='both'         # Both formats
```

## Troubleshooting

### Analysis Is Slow

Use `models.default` to omit ODE models, reduce `bootstrap_iterations`, or limit `models_to_try` to a smaller candidate set. Adjust `n_jobs` if process-pool parallelism is not appropriate for the available CPU and memory resources.

### ODE Solver Warnings

Review the warning and fit diagnostics. A solver fallback may still produce a fit; verify that the fit succeeded and inspect its residuals and parameters before using the result.

### Poor Fit

Check the input temperature units, data quality, conversion range, and residual plots. Compare additional candidate models when the current model does not describe the observations.

### Bootstrap Replicates Fail

Inspect the reported number of successful replicates and the warnings. Confidence-interval reliability depends on the number and quality of successful replicates; there is no single success percentage that is sufficient for every analysis.

## Further Reading

- [Getting Started](../../docs/getting_started.md)
- [Model Selection Guide](../../docs/model_selection_guide.md)
- [Kinetic Models](../../docs/kinetic_models.md)
- [Troubleshooting](../../docs/troubleshooting.md)