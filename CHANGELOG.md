# Changelog

All notable changes to the AKTS Python library are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased]

### Added
- **ICH Q1E Full Compliance**: Complete implementation of ICH Q1E regulatory requirements for shelf-life determination
  - **One-sided 95% confidence bounds** (not two-sided intervals) per ICH Q1E guidelines
  - **Confidence-band crossing method** for conservative shelf-life estimates
  - **Auto-detection** of attribute direction from readout_type ('decreasing' for potency, 'increasing' for aggregates)
  - **Default enabled** with sensible defaults (5% specification limit)
  - New `shelf_life.py` module with `calculate_shelf_life_ich_q1e()` and `time_to_specification()` functions
  - New `attribute_direction` parameter on `predict_conversion()` for one-sided CI calculation
  - Comprehensive test suite (22 tests) verifying ICH Q1E compliance
  - Full documentation in `docs/ich_q1e_compliance.md`
  - ICH Q1E parameters now have automatic defaults: `attribute_direction=None` (auto-detect), `calculate_shelf_life_ich_q1e_method=True`, `shelf_life_specification_limit=0.05`
  - Conservative shelf-life estimates are typically 5-15% shorter than mean (correct per ICH Q1E)
  - Ready for regulatory submissions

### Performance
- **ODE fits are several times faster.** Temperature lookups inside the ODE right-hand side now use a constant function (isothermal data) or a pure-Python/`np.interp` piecewise-linear function instead of scipy `interp1d`: ~1 µs instead of ~110 µs per call. Objectives now build the model and temperature functions once per optimization instead of once per evaluation. Benchmark (3 temperatures × 20 points): SB fit 19.8 s → 1.1 s; A->B->C fit 171 s → 6.3 s.
- **Bootstrap replicates use the same optimizer as the original fit.** Isothermal closed-form models (F0–F3, A2, A3, R2, R3, D2–D4) refit with `least_squares` on the analytic solution instead of Powell + ODE. The coarse ln k scan is skipped for replicates, which start next to the best fit. 30-replicate F1 bootstrap: 269 s → 12 s.
- **Closed-form prediction.** `simulate_kinetics()` / `predict_conversion()` evaluate closed-form models analytically for constant-temperature programs. This speeds up bootstrap CI bands and `time_to_conversion()`.

### Added
- **`random_state`** on `run_bootstrap()`, `run_bootstrap_empirical()`, `run_bootstrap_friedman()` and `auto_model_isothermal_data()` gives reproducible results. Each replicate gets an independent stream from `numpy.random.SeedSequence.spawn`, so results don't depend on `n_jobs`.
- **Solver documentation**: new "Numerical Solvers and Performance" section in `docs/advanced_usage.md` (available methods, `solver_options` keys, choosing primary/fallback solvers).
- **Bootstrap confidence interval quality control**: Automatic filtering of degenerate bootstrap replicates (predicting less than 1% final conversion) before calculating confidence intervals. Prevents lower CI bounds from incorrectly collapsing to zero.
- **Enhanced HTML report statistics**: Statistical Details section now includes degrees of freedom, data point counts, and bootstrap quality control information with clear explanations.
- **Model selector with IDE autocomplete**: Type `models.` to discover all available kinetic models organized by category (kinetic, empirical, ODE, model-free). Enables safe concatenation like `models.kinetic.F1 + models.empirical.Linear`.
- **Empirical kinetic models**: Fast algebraic fits with global Arrhenius fitting across all temperatures. Includes First Order, Linear, Square Root, Logistic, and Exponential models.
- **ICH Q1E regulatory compliance**: Automatic shelf-life estimation with one-sided 95% confidence intervals, ICH Q1E extrapolation ceiling calculation, and regulatory analysis section in HTML reports.
- **Dynamic y-axis scaling for shelf-life plots**: Plots automatically scale to show all data instead of cutting off at fixed limits.

### Changed
- **Default ODE solver is now LSODA, with RK45 as fallback** (previously RK45 with LSODA fallback). LSODA switches automatically between stiff and non-stiff methods and needs fewer right-hand-side evaluations. Fitted parameters change only within solver tolerance. Restore the old order with `solver_options={'primary_solver': 'RK45', 'fallback_solver': 'LSODA'}`.
- **`akts/core.py` split into focused modules**: `simulation`, `fitting`, `bootstrap`, `prediction`, `ranking`. `akts.core` re-exports every previous name, so existing imports keep working.
- Mutable default arguments (`solver_options={}`, `optimizer_options={}`, `iteration_counter=[0]`) replaced with `None`.
- Model display names consolidated into `models.MODEL_DISPLAY_NAMES`, shared by progress messages and reports.
- **Activation energy display**: Reports now show Ea in kJ/mol (more readable) instead of J/mol in the Statistical Details section.
- **Parameter table formatting**: Enhanced with units column and better organization of activation energy and pre-exponential factor.
- **Report section ordering**: Methods section moved to end of HTML reports for better flow (results first, methodology last).

### Fixed
- **ODE-model residual bootstrap did not resample.** Residual values and target points were swapped, so each replicate only shuffled residuals between points. Parameter CIs could be orders of magnitude too wide: a 30-replicate F1 bootstrap gave an Ea CI of 20–1,788 kJ/mol, now 98–104 kJ/mol.
- **Empirical-model bootstrap added residuals to the observed data** instead of the fitted values, which inflated the noise twice over. Both bootstraps now share one resampler.
- **Bootstrap is reproducible**: workers no longer draw from the unseeded global `np.random` state.
- **Closed-form fits could stop at the initial guess.** When the data saturates between samples, the closed-form `least_squares` path now runs the same coarse ln k scan as the ODE path first (e.g. F1 R² 0.961 → 0.9999 on such data).
- **Bootstrap replicate refits** now use `fit_kinetic_model`'s default physical bounds. Their starting points are jittered in (Ea, ln k(T_ref)) coordinates, so the start keeps the fitted rate. Previously, replicates on single-temperature data could wander to negative Ea.
- **HTML report confidence bands**: the Prediction/Extrapolation and Temperature Excursion plots built the CI polygon by adding numpy arrays element-wise instead of concatenating them, so the band never rendered. The Model Fit Visualization now also shows a per-temperature bootstrap CI band.
- **Friedman shelf-life on isothermal data was about 2× too short.** `auto_model_isothermal_data` resolved Ea(α) only from α = 5% (linear grid). Below that, predictions held the 5% rate constant, which overstates the early rate. The grid is now log-spaced from α = 0.5%. On the bundled protein data, the mean time to 5% at 40 °C moved from 34 to 60 days (observed: 71 days); the ICH lower bound moved from 0.6 to 1.7 months.
- **Friedman bootstrap** no longer emits divide-by-zero warnings. Resamples that repeat a single temperature, which leaves no 1/T spread, are skipped.
- **Friedman fits with only 2 temperatures** are now reported as failed. A 2-point Arrhenius line has zero degrees of freedom.
- **Arrhenius plot**: the legend and parameter box now sit in the corners the fitted line never crosses. The CI band also renders without `return_replicate_params=True`.
- **Bootstrap confidence intervals**: Lower bound no longer incorrectly shows zero across all time points when some bootstrap samples produce degenerate predictions.
- **Shelf-life plot y-axis scaling**: Plots no longer cut off data at y=15% when degradation exceeds this value.
- **Temperature unit display in plots**: Temperature annotations in simulation plots now correctly display converted units.
- **Empirical model fitting**: Fixed FitResult parameter error that prevented empirical models from running. Datasets and predictions now stored correctly in model_definition_args.

### Improved
- **Bootstrap analysis robustness**: Quality control automatically excludes numerical artifacts (typically 2-3 out of 100 replicates) that represent unrealistic parameter combinations rather than genuine uncertainty.
- **Report clarity**: Bootstrap Analysis section includes clear explanations of confidence interval methodology and quality control procedures.
- **Code maintainability**: Removed process-oriented comments, leaving only code-focused documentation for programmers.

---

## [0.2.0] - Previous Release

Previous features and functionality as documented in original release.

---

## Notes

### Bootstrap Quality Control

The bootstrap confidence interval filtering introduced in this release addresses a numerical issue where 2-3% of bootstrap replicates could produce parameter combinations predicting negligible degradation. These are filtered automatically before calculating percentile-based confidence intervals, resulting in more realistic uncertainty estimates.

### Model Selector

The model selector provides a structured way to discover and select kinetic models:

```python
from akts import models

# Discover models with IDE autocomplete
models_to_try = models.kinetic.all      # All mechanistic models
models_to_try = models.empirical.all    # All empirical models
models_to_try = models.all              # Everything

# Safe concatenation
models_to_try = models.kinetic.F1 + models.empirical.Linear
```

All model selector properties return lists, enabling safe concatenation without type errors.

### ICH Q1E Compliance

Regulatory analysis follows ICH Q1E guidelines for stability data evaluation:
- One-sided 95% confidence intervals (more conservative than two-sided)
- Automatic extrapolation ceiling calculation: min(2 × study duration, study duration + 12 months)
- Clear indication when shelf-life estimates exceed guideline limits

---

For detailed documentation, see [docs/README.md](docs/README.md).
