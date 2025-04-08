# akts: Python Library for Advanced Kinetic Analysis

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

akts is a Python library designed for the comprehensive analysis of thermoanalytical data (DSC, TGA, etc.) to determine reaction kinetics and predict material behavior under various temperature conditions. It implements robust methods for isoconversional analysis, multi-model fitting with parameter uncertainty estimation via bootstrapping, and flexible simulations based on the derived kinetic parameters.

This library aims to provide a powerful, accessible, and transparent alternative or complement to commercial software packages for researchers in materials science, chemistry, pharmaceuticals, and related fields.

---

## Key Features

- **Isoconversional (Model-Free) Analysis**:
  - Determine activation energy (`Ea`) as a function of conversion (`alpha`) without prior model assumptions.
  - Implemented methods: Friedman (differential), KAS (Kissinger-Akahira-Sunose - integral), OFW (Ozawa-Flynn-Wall - integral).

- **Model-Based Fitting**:
  - Globally fit kinetic models to multiple experimental datasets simultaneously.
  - Supports standard single-step models (F_n, A_n, SB(m,n), etc. - easily extensible).
  - Supports common multi-step models (e.g., consecutive A->B->C).
  - Handles bimolecular reactions (A+B->C).
  - Uses `logA` scaling internally for improved optimizer stability.
  - Option to optimize based on conversion residuals (default, weighted) or rate residuals.

- **Bootstrap Analysis**:
  - Estimate parameter confidence intervals and distributions using parallelized residual resampling.
  - Provides robust uncertainty quantification, especially useful when dealing with parameter correlations (KCE).
  - Returns ranked replicate results and median parameter estimates.

- **Model Discovery & Ranking**:
  - Fit multiple predefined or custom models to the data sequentially.
  - Rank fitted models based on a combined score using statistical criteria (BIC/AICc), goodness-of-fit (R², RSS), and parameter count.

- **Prediction & Simulation**:
  - Simulate conversion (`alpha`) vs. time for arbitrary user-defined temperature programs (isothermal, ramps, steps, complex profiles, excursions).
  - Generate prediction confidence intervals based on bootstrap results.
  - Includes helper function (`construct_profile`) to easily build complex temperature profiles.

- **Callbacks**:
  - Provides hooks for monitoring the progress of optimization routines (main fit and bootstrap replicates).

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core Concepts](#core-concepts)
  - [Kinetic Models](#kinetic-models)
  - [Isoconversional Analysis](#isoconversional-analysis)
  - [Model Fitting](#model-fitting)
  - [Bootstrapping](#bootstrapping)
  - [Model Discovery & Ranking](#model-discovery--ranking)
  - [Prediction](#prediction)
- [Usage Examples](#usage-examples)
  - [1. Loading Data](#1-loading-data)
  - [2. Isoconversional Analysis](#2-isoconversional-analysis)
  - [3. Single Model Fit](#3-single-model-fit)
  - [4. Bootstrap Analysis](#4-bootstrap-analysis)
  - [5. Model Discovery](#5-model-discovery)
  - [6. Prediction with Excursion](#6-prediction-with-excursion)
- [API Reference (Overview)](#api-reference-overview)
- [Contributing](#contributing)
- [License](#license)
- [Citation (Placeholder)](#citation-placeholder)

---

## Installation

```bash
pip install akts  # Placeholder - Replace with actual PyPI name when available
```

Or, install directly from GitHub:

```bash
pip install git+https://github.com/your_username/akts.git  # Placeholder - Replace URL
```

### Dependencies

- **NumPy**
- **SciPy**

---

## Quick Start

```python
import numpy as np
import matplotlib.pyplot as plt
from akts import (
    KineticDataset,
    discover_kinetic_models,
    run_bootstrap,
    predict_conversion,
    simulate_kinetics,
    construct_profile,
    run_kas  # Example isoconversional method
)

# --- 1. Load Your Data ---
# Replace with your actual data loading
# Data should be a list of KineticDataset objects
# Each KineticDataset needs time (min), temperature (K), conversion (0-1)
datasets = []

# Example: Using Dummy Data for Demonstration
T0 = 298.15
true_params = {'Ea1': 80e3, 'A1': 1e10, 'Ea2': 95e3, 'A2': 5e11}
model_def_args_gen = {'f1_model': 'F1', 'f2_model': 'F1'}
kinetic_params_gen = {k: v for k, v in true_params.items()}
heating_rates = [2.0, 5.0, 10.0, 15.0, 20.0]
end_times = [90, 45, 30, 25, 20]

for beta, t_end in zip(heating_rates, end_times):
    beta_K_s = beta / 60.0
    time_s = np.linspace(0, t_end * 60.0, 25)
    temperature_K = T0 + beta_K_s * time_s
    temp_prog_sec = (time_s, temperature_K)
    sim_result = simulate_kinetics(
        model_name="A->B->C",
        model_definition_args=model_def_args_gen,
        kinetic_params=kinetic_params_gen,
        initial_alpha=0.0,
        temperature_program=temp_prog_sec
    )
    noise = np.random.normal(0, 0.015, size=sim_result.conversion.shape)
    alpha_noisy = np.clip(sim_result.conversion + noise, 0.0, 1.0)
    datasets.append(KineticDataset(
        time=time_s / 60.0,
        temperature=temperature_K,
        conversion=alpha_noisy,
        heating_rate=beta
    ))
```

---

## Core Concepts

### Kinetic Models

The rate of solid-state transformations is generally described by:

```
dα/dt = k(T) * f(α)
```

Where:
- `α` is the conversion fraction.
- `k(T)` is the temperature-dependent rate constant (Arrhenius: `k(T) = A * exp(-Ea / RT)`).
- `f(α)` is the reaction model function describing the dependence on conversion.

akts allows fitting various models by specifying `f(α)`.

#### Single-Step Models
- **F1, F2, F3**: Nth-order reactions (`f(α) = (1-α)^n`).
- **A2, A3**: Avrami-Erofeev models (`f(α) = n(1-α)[-ln(1-α)]^(1-1/n)`).
- **SB(m,n)**: Sestak-Berggren model (`f(α) = α^m * (1-α)^n`).

#### Multi-Step Models
- **A->B->C**: Consecutive first-order steps.
- **A+B->C**: Bimolecular reaction.

---

## Usage Examples

### 1. Loading Data

```python
from akts import KineticDataset
import pandas as pd

# Example loading from CSVs
data_files = ['ramp_2kmin.csv', 'ramp_5kmin.csv', 'ramp_10kmin.csv']
heating_rates = [2.0, 5.0, 10.0]  # K/min
datasets = []

for i, file in enumerate(data_files):
    df = pd.read_csv(file)
    datasets.append(KineticDataset(
        time=df['Time_min'].values,
        temperature=df['Temperature_K'].values,
        conversion=df['Conversion'].values,
        heating_rate=heating_rates[i]
    ))
```

### 2. Isoconversional Analysis

```python
from akts import run_kas
import matplotlib.pyplot as plt
import numpy as np

try:
    kas_result = run_kas(datasets, alpha_levels=np.linspace(0.05, 0.95, 19))
    print("KAS Analysis Results:")
    print(f" Alpha: {np.round(kas_result.alpha, 2)}")
    print(f" Ea (kJ/mol): {np.round(kas_result.Ea / 1000, 1)}")

    # Plot Ea vs alpha
    plt.figure()
    plt.plot(kas_result.alpha, kas_result.Ea / 1000, 'o-')
    plt.xlabel("Conversion (alpha)")
    plt.ylabel("Activation Energy (kJ/mol)")
    plt.title("KAS Isoconversional Ea")
    plt.grid(True)
    plt.show()
except Exception as e:
    print(f"Isoconversional analysis failed: {e}")
```

### 3. Single Model Fit

```python
from akts import fit_kinetic_model, FitResult

# Define model and guesses/bounds (A-scale)
model_name_fit = "single_step"
# For F1 model, n=1 is default in f_n_order, no f_alpha_params needed
model_def_args_fit = {'f_alpha_model': 'F1'}
initial_guesses_fit = {'Ea': 85e3, 'A': 1e11}
parameter_bounds_fit = {'Ea': (50e3, 150e3), 'A': (1e7, 1e14)}
solver_options_fit = {'primary_solver': 'LSODA'}
optimizer_options_fit = {'method': 'L-BFGS-B'}

try:
    fit_result = fit_kinetic_model(
        datasets=datasets,
        model_name=model_name_fit,
        model_definition_args=model_def_args_fit,
        initial_guesses=initial_guesses_fit,
        parameter_bounds=parameter_bounds_fit,
        solver_options=solver_options_fit,
        optimizer_options=optimizer_options_fit,
        # optimize_on_rate=False # Use default conversion objective (True)
    )

    if isinstance(fit_result, FitResult) and fit_result.success:
        print("\nFit Successful!")
        print(f" Model: {fit_result.model_name}")
        print(f" Parameters: {fit_result.parameters}")
        print(f" RSS: {fit_result.rss:.4e}")
        print(f" R-squared: {fit_result.r_squared:.4f}")
        print(f" BIC: {fit_result.bic:.2f}")
    elif isinstance(fit_result, FitResult):
        print(f"\nFit Failed: {fit_result.message}")
    else:
         print("\nFit Failed: Unknown error.")

except Exception as e:
    print(f"Fitting failed: {e}")
```

### 4. Bootstrap Analysis

```python
from akts import run_bootstrap, FitResult, BootstrapResult

# Assumes 'fit_result' is a successful FitResult object from step 3
if isinstance(fit_result, FitResult) and fit_result.success:
    print(f"\nRunning Bootstrap for {fit_result.model_name}...")
    # Use bounds relevant to the fitted model (if defined previously)
    # Example: bounds_A_boot = parameter_bounds_pool.get(fit_result.model_name)
    bounds_A_boot = {'Ea': (50e3, 150e3), 'A': (1e7, 1e14)} # Example bounds for F1

    bootstrap_optimizer_options = { 'method': 'L-BFGS-B', 'options': {'maxiter': 100} } # Faster options

    bootstrap_result = run_bootstrap(
        datasets=datasets,
        fit_result=fit_result,
        optimizer_options=bootstrap_optimizer_options,
        parameter_bounds=bounds_A_boot, # Pass A-scale bounds
        solver_options=solver_options_fit, # Use same solver options
        n_iterations=100, # Use more iterations for better CIs
        n_jobs=-1, # Use parallel processing
        timeout_per_replicate=120.0, # 2 min timeout per replicate
        return_replicate_params=True # Get raw params list
    )

    if isinstance(bootstrap_result, BootstrapResult):
        print(f"\nBootstrap completed: {bootstrap_result.n_iterations} successful replicates.")
        print("Confidence Intervals (95%):")
        for name, ci in bootstrap_result.parameter_ci.items():
            if 'Ea' in name: print(f"  {name}: ({ci[0]/1000:.2f}, {ci[1]/1000:.2f}) kJ/mol")
            elif 'A' in name: print(f"  {name}: ({ci[0]:.2e}, {ci[1]:.2e}) 1/s")
            if 'Ea' in name: print(f"  {name}: ({ci[0]/1000:.2f}, {ci[1]/1000:.2f}) kJ/mol")
            elif 'A' in name: print(f"  {name}: ({ci[0]:.2e}, {ci[1]:.2e}) 1/s")
            else: print(f"  {name}: ({ci[0]:.3f}, {ci[1]:.3f})")d
        if bootstrap_result.ranked_replicates:
        # Access ranked replicates or median parameters if needed
        if bootstrap_result.ranked_replicates:ed_replicates[:5]:
             print("\n Top 5 Replicates by RSS:")SS={item['stats']['rss']:.3e}, Params={item['parameters']}")
             for item in bootstrap_result.ranked_replicates[:5]:
                 print(f"  Rank {item['rank']}: RSS={item['stats']['rss']:.3e}, Params={item['parameters']}")
        if bootstrap_result.median_parameters:
             print(f"\n Median Parameters: {bootstrap_result.median_parameters}")
        print("\nBootstrap analysis failed or returned no results.")
    else:
        print("\nBootstrap analysis failed or returned no results.")
```

### 5. Model Discovery
from akts import discover_kinetic_models
```python
from akts import discover_kinetic_models Quick Start
models_to_try = [
# Define models, guesses, bounds etc. as in Quick Start'f_alpha_model': 'F1'}},
models_to_try = [, 'type': 'single_step', 'def_args': {'f_alpha_model': 'A2', 'f_alpha_params': {'n': 2.0}}},
    {'name': 'F1', 'type': 'single_step', 'def_args': {'f_alpha_model': 'F1'}},', 'f2_model': 'F1'}},
    {'name': 'A2', 'type': 'single_step', 'def_args': {'f_alpha_model': 'A2', 'f_alpha_params': {'n': 2.0}}},
    {'name': 'A->B->C (F1,F1)', 'type': 'A->B->C', 'def_args': {'f1_model': 'F1', 'f2_model': 'F1'}},
]   'F1': {'Ea': 85e3, 'A': 1e11},
initial_guesses_pool = {A': 5e11},
    'F1': {'Ea': 85e3, 'A': 1e11},3, 'A1': 1e10, 'Ea2': 95e3, 'A2': 5e11},
    'A2': {'Ea': 88e3, 'A': 5e11},
    'A->B->C (F1,F1)': {'Ea1': 80e3, 'A1': 1e10, 'Ea2': 95e3, 'A2': 5e11},
}   'F1': {'Ea': (10e3, 200e3), 'A': (1e5, 1e15)},
parameter_bounds_pool = {00e3), 'A': (1e5, 1e15)},
    'F1': {'Ea': (10e3, 200e3), 'A': (1e5, 1e15)},: (1e5, 1e15), 'Ea2': (10e3, 200e3), 'A2': (1e5, 1e15)},
    'A2': {'Ea': (10e3, 200e3), 'A': (1e5, 1e15)},
    'A->B->C (F1,F1)': {'Ea1': (10e3, 200e3), 'A1': (1e5, 1e15), 'Ea2': (10e3, 200e3), 'A2': (1e5, 1e15)},
}ptimizer_options_disc = { 'method': 'L-BFGS-B', 'options': {'maxiter': 500} } # Faster discovery fits
solver_options_disc = { 'primary_solver': 'LSODA', 'fallback_solver': 'BDF'}
optimizer_options_disc = { 'method': 'L-BFGS-B', 'options': {'maxiter': 500} } # Faster discovery fits
    ranked_models = discover_kinetic_models(
try:    datasets=datasets,
    ranked_models = discover_kinetic_models(
        datasets=datasets,ol=initial_guesses_pool,
        models_to_try=models_to_try,ter_bounds_pool,
        initial_guesses_pool=initial_guesses_pool,
        parameter_bounds_pool=parameter_bounds_pool,
        solver_options=solver_options_disc,version objective
        optimizer_options=optimizer_options_disc,
        # optimize_on_rate=False, # Use conversion objective
        score_weights=None # Use default scoring
    )rint("\n--- Ranked Model Discovery Results ---")
    if ranked_models:
    print("\n--- Ranked Model Discovery Results ---")
    if ranked_models:tem['stats']
        for item in ranked_models:rank']} (Score={item['score']:.3f}): {item['model_name']}")
            stats = item['stats']ats.get('bic', np.nan):.2f}, R² = {stats.get('r_squared', np.nan):.4f}")
            print(f"  Rank {item['rank']} (Score={item['score']:.3f}): {item['model_name']}")
            print(f"    BIC = {stats.get('bic', np.nan):.2f}, R² = {stats.get('r_squared', np.nan):.4f}")
    else:
        print("  No models fitted successfully.")
    print(f"Model discovery failed: {e}")
except Exception as e:
    print(f"Model discovery failed: {e}")
```

### 6. Prediction with Excursion
from akts import simulate_kinetics, construct_profile, FitResult
```python
from akts import simulate_kinetics, construct_profile, FitResult
if isinstance(fit_result, FitResult) and fit_result.success:
# Assumes 'fit_result' contains the FitResult for the chosen model
if isinstance(fit_result, FitResult) and fit_result.success:
    print("\nPredicting profile with excursion...")0, 'temperature': 278.15}, # 2 days at 5C
    segments = [ 'ramp', 'duration': 1*3600, 'start_temp': 278.15, 'end_temp': 298.15}, # 1hr ramp to 25C
        {'type': 'isothermal', 'duration': 2*24*3600, 'temperature': 278.15}, # 2 days at 5C
        {'type': 'ramp', 'duration': 1*3600, 'start_temp': 278.15, 'end_temp': 298.15}, # 1hr ramp to 25C
        {'type': 'isothermal', 'duration': 8*3600, 'temperature': 298.15}, # 8hr at 25Cat 5C
        {'type': 'ramp', 'duration': 1*3600, 'start_temp': 298.15, 'end_temp': 278.15}, # 1hr ramp to 5C
        {'type': 'isothermal', 'duration': 5*24*3600, 'temperature': 278.15}, # 5 days at 5C
    ]xcursion_prog = (excursion_time_sec, excursion_temp_K)
    excursion_time_sec, excursion_temp_K = construct_profile(segments)
    excursion_prog = (excursion_time_sec, excursion_temp_K)ion_args', {})

    sim_model_def_args = getattr(fit_result, 'model_definition_args', {})
        model_name=fit_result.model_name, # Use model name from fit result
    excursion_prediction = simulate_kinetics(rgs,
        model_name=fit_result.model_name, # Use model name from fit result
        model_definition_args=sim_model_def_args,
        kinetic_params=fit_result.parameters,
        initial_alpha=0.0,ver_options_fit # Use solver options from fit
        temperature_program=excursion_prog,
        solver_options=solver_options_fit # Use solver options from fit
    ) Plotting
    plt.figure(figsize=(10, 6))
    # Plottingsubplot(2, 1, 1)
    plt.figure(figsize=(10, 6))on.time / 3600.0, excursion_prediction.temperature, 'r-')
    ax1 = plt.subplot(2, 1, 1))")
    plt.plot(excursion_prediction.time / 3600.0, excursion_prediction.temperature, 'r-')
    plt.ylabel("Temperature (K)")
    plt.title("Prediction for Profile with Temperature Excursion")
    plt.grid(True)sion_prediction.time / 3600.0, excursion_prediction.conversion, 'b-')
    plt.subplot(2, 1, 2, sharex=ax1) # Share x-axis
    plt.plot(excursion_prediction.time / 3600.0, excursion_prediction.conversion, 'b-')
    plt.ylabel("Predicted Conversion (alpha)")
    plt.xlabel("Time (hours)")
    plt.ylim(-0.05, 1.05)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
```


## API Reference (Overview) link to generated API documentation, e.g., using Sphinx)

(This section would ideally link to generated API documentation, e.g., using Sphinx)

### Main Functions:tic_model(...)`: Fits a single specified kinetic model.
- `akts.run_bootstrap(...)`: Performs bootstrap analysis on a previous fit.
- `akts.fit_kinetic_model(...)`: Fits a single specified kinetic model.hem.
- `akts.run_bootstrap(...)`: Performs bootstrap analysis on a previous fit.d temperature program.
- `akts.discover_kinetic_models(...)`: Fits multiple models and ranks them.optionally adds bootstrap CIs.
- `akts.simulate_kinetics(...)`: Simulates kinetics for given parameters and temperature program.
- `akts.predict_conversion(...)`: Simulates kinetics using a FitResult and optionally adds bootstrap CIs.
- `akts.run_kas(...)`, `run_ofw(...)`, `run_friedman(...)`: Perform isoconversional analysis.

### Helper Functions:truct_profile(...)`: Builds complex temperature profiles.

- `akts.utils.construct_profile(...)`: Builds complex temperature profiles.

### Data Classes:cDataset`: Holds experimental data.
- `akts.FitResult`: Stores results from `fit_kinetic_model`.
- `akts.KineticDataset`: Holds experimental data.n_bootstrap`.
- `akts.FitResult`: Stores results from `fit_kinetic_model`.ediction.
- `akts.BootstrapResult`: Stores results from `run_bootstrap`.is.
- `akts.PredictionResult`: Stores results from simulation/prediction.
- `akts.IsoResult`: Stores results from isoconversional analysis.

## API Reference (Overview)ls on the main user-facing functions and data structures. For complete details, please refer to the docstrings within the code or generated API documentation (link placeholder).

This section provides details on the main user-facing functions and data structures. For complete details, please refer to the docstrings within the code or generated API documentation (link placeholder).

### Data Classess are used to structure input data and results.

These dataclasses are used to structure input data and results.run.
    *   `time` (np.ndarray): Time points (typically in **minutes** for input, converted internally).
*   **`KineticDataset`**: Holds experimental data for a single run..
    *   `time` (np.ndarray): Time points (typically in **minutes** for input, converted internally).
    *   `temperature` (np.ndarray): Temperature points (**Kelvin**).n), used by some isoconversional methods if available.
    *   `conversion` (np.ndarray): Conversion fraction (`alpha`, 0 to 1).
    *   `heating_rate` (Optional[float]): Nominal heating rate (K/min), used by some isoconversional methods if available.
    *   `metadata` (Dict): Optional dictionary for user metadata.
    *   `model_name` (str): User-provided name for the fitted model.
*   **`FitResult`**: Stores results from `fit_kinetic_model`. parameters (A-scale, e.g., `{'Ea': 80e3, 'A': 1e10}`). Ea is in J/mol, A is in 1/s.
    *   `model_name` (str): User-provided name for the fitted model.ly.
    *   `parameters` (Dict[str, float]): Dictionary of fitted parameters (A-scale, e.g., `{'Ea': 80e3, 'A': 1e10}`). Ea is in J/mol, A is in 1/s.
    *   `success` (bool): True if the optimizer converged successfully.n *conversion* (`sum(alpha_exp - alpha_sim)**2`).
    *   `message` (str): Message from the optimizer regarding termination status.nversion RSS and stats.
    *   `rss` (float): Final unweighted Residual Sum of Squares based on *conversion* (`sum(alpha_exp - alpha_sim)**2`).
    *   `n_datapoints` (int): Number of data points used for calculating final conversion RSS and stats., if calculable from optimizer's Hessian.
    *   `n_parameters` (int): Number of parameters fitted by the optimizer.Cc) based on conversion RSS. Lower is better.
    *   `param_std_err` (Optional[Dict[str, float]]): Estimated standard errors for parameters (A-scale), if calculable from optimizer's Hessian.
    *   `aic` (Optional[float]): Corrected Akaike Information Criterion (AICc) based on conversion RSS. Lower is better.
    *   `bic` (Optional[float]): Bayesian Information Criterion based on conversion RSS. Lower is better, penalizes parameters more than AICc.
    *   `r_squared` (Optional[float]): R-squared value based on conversion fit.pecific model structure and fixed parameters used for this fit.
    *   `initial_ratio_r` (Optional[float]): The initial reactant ratio `r = B0/A0` used if fitting the "A+B->C" model.use the conversion-based fit failed.
    *   `model_definition_args` (Optional[Dict]): The dictionary defining the specific model structure and fixed parameters used for this fit.
    *   `used_rate_fallback` (bool): True if the optimization fell back to using the rate-based objective function because the conversion-based fit failed.
    *   `model_name` (str): Name of the model that was bootstrapped.
*   **`BootstrapResult`**: Stores results from `run_bootstrap`.ionary where keys are parameter names (A-scale) and values are arrays containing the parameter values from each successful bootstrap replicate.
    *   `model_name` (str): Name of the model that was bootstrapped.containing the calculated confidence intervals (lower, upper bounds) for each parameter (A-scale).
    *   `parameter_distributions` (Dict[str, np.ndarray]): Dictionary where keys are parameter names (A-scale) and values are arrays containing the parameter values from each successful bootstrap replicate.
    *   `parameter_ci` (Dict[str, Tuple[float, float]]): Dictionary containing the calculated confidence intervals (lower, upper bounds) for each parameter (A-scale).
    *   `n_iterations` (int): Number of *successful* bootstrap replicates completed._params=True`), a list containing the fitted parameter dictionaries (A-scale) for each successful replicate (unsorted).
    *   `confidence_level` (float): The confidence level used for calculating CIs (e.g., 0.95).ul replicate, ranked by RSS on their *resampled* data. Each dict contains `'rank'`, `'source'`, `'parameters'` (A-scale), and `'stats'` (RSS, R², AICc, BIC calculated on the *resampled* data).
    *   `raw_parameter_list` (Optional[List[Dict]]): If requested (`return_replicate_params=True`), a list containing the fitted parameter dictionaries (A-scale) for each successful replicate (unsorted).
    *   `ranked_replicates` (Optional[List[Dict]]): List of dictionaries, one for each successful replicate, ranked by RSS on their *resampled* data. Each dict contains `'rank'`, `'source'`, `'parameters'` (A-scale), and `'stats'` (RSS, R², AICc, BIC calculated on the *resampled* data).
    *   `median_parameters` (Optional[Dict[str, float]]): Dictionary of median parameter values (A-scale) calculated from the successful replicates.
    *   `median_stats` (Optional[Dict]): Dictionary of stats (RSS, R², AICc, BIC) calculated by simulating the `median_parameters` against the *original* experimental data.
    *   `time` (np.ndarray): Time points (seconds) of the simulation.
*   **`PredictionResult`**: Stores results from `simulate_kinetics` or `predict_conversion`.ints.
    *   `time` (np.ndarray): Time points (seconds) of the simulation.each time point.
    *   `temperature` (np.ndarray): Temperature points (Kelvin) corresponding to the time points.version` with bootstrap results, a tuple containing (lower_ci_alpha_array, upper_ci_alpha_array).
    *   `conversion` (np.ndarray): Predicted conversion (`alpha`) at each time point.
    *   `conversion_ci` (Optional[Tuple[np.ndarray, np.ndarray]]): If calculated via `predict_conversion` with bootstrap results, a tuple containing (lower_ci_alpha_array, upper_ci_alpha_array).
    *   `method` (str): Name of the isoconversional method used.
*   **`IsoResult`**: Stores results from isoconversional functions (`run_kas`, etc.).
    *   `method` (str): Name of the isoconversional method used.es (J/mol) corresponding to `alpha`.
    *   `alpha` (np.ndarray): Array of conversion levels at which Ea was calculated.lable from the method's regression.
    *   `Ea` (np.ndarray): Array of calculated activation energies (J/mol) corresponding to `alpha`.tics (e.g., r_value, stderr) for each alpha level.
    *   `Ea_std_err` (Optional[np.ndarray]): Estimated standard error of Ea, if available from the method's regression.
    *   `regression_stats` (Optional[List[Dict]]): List of dictionaries containing regression statistics (e.g., r_value, stderr) for each alpha level.

### Main Functions_model(datasets, model_name, model_definition_args, initial_guesses, parameter_bounds=None, solver_options={}, optimizer_options={}, callback=None)`**
    *   Fits a *single* kinetic model globally to one or more datasets.
*   **`fit_kinetic_model(datasets, model_name, model_definition_args, initial_guesses, parameter_bounds=None, solver_options={}, optimizer_options={}, callback=None)`**
    *   Fits a *single* kinetic model globally to one or more datasets.tep', 'A->B->C').
    *   `datasets`: List of `KineticDataset` objects. exact model variant (e.g., `{'f_alpha_model': 'F1'}` or `{'f1_model': 'F1', 'f2_model': 'F1'}`). See [Core Concepts](#core-concepts).
    *   `model_name` (str): The internal type of model (e.g., 'single_step', 'A->B->C').Ea': 80e3, 'A': 1e10}`). Must include all parameters defined by the combination of `model_name` and `model_definition_args`.
    *   `model_definition_args` (Dict): Specifies the exact model variant (e.g., `{'f_alpha_model': 'F1'}` or `{'f1_model': 'F1', 'f2_model': 'F1'}`). See [Core Concepts](#core-concepts).s if None or for missing parameters.
    *   `initial_guesses` (Dict): Initial guesses for parameters (**A-scale**, e.g., `{'Ea': 80e3, 'A': 1e10}`). Must include all parameters defined by the combination of `model_name` and `model_definition_args`.
    *   `parameter_bounds` (Optional[Dict]): Bounds for parameters (**A-scale**). Keys must match `initial_guesses`. Example: `{'Ea': (10e3, 200e3), 'A': (1e5, 1e15)}`. Uses default bounds if None or for missing parameters.
    *   `solver_options` (Dict): Options for the ODE solver (`scipy.integrate.solve_ivp`), e.g., `{'primary_solver': 'LSODA', 'fallback_solver': 'BDF', 'rtol': 1e-6, 'atol': 1e-9}`.
    *   `optimizer_options` (Dict): Options for the optimizer (`scipy.optimize.minimize`), e.g., `{'method': 'L-BFGS-B', 'options': {'maxiter': 1000, 'ftol': 1e-9}}`.
    *   `callback` (Optional[Callable]): Function called after each optimizer iteration: `callback(iteration_count, params_dict_A)`.
    *   Returns: `FitResult` object.esult, optimizer_options, parameter_bounds=None, n_iterations=100, confidence_level=0.95, n_jobs=-1, solver_options={}, end_callback=None, timeout_per_replicate=None, return_replicate_params=False)`**
    *   Performs bootstrap analysis for a model previously fitted.
*   **`run_bootstrap(datasets, fit_result, optimizer_options, parameter_bounds=None, n_iterations=100, confidence_level=0.95, n_jobs=-1, solver_options={}, end_callback=None, timeout_per_replicate=None, return_replicate_params=False)`**
    *   Performs bootstrap analysis for a model previously fitted.urned by `fit_kinetic_model`. Contains the parameters and model definition to use.
    *   `datasets`: Original list of `KineticDataset` objects used for the fit. bootstrap replicate*. Often uses reduced `maxiter`.
    *   `fit_result` (FitResult): The successful result object returned by `fit_kinetic_model`. Contains the parameters and model definition to use.
    *   `optimizer_options` (Dict): Options for the optimizer used *within each bootstrap replicate*. Often uses reduced `maxiter`.
    *   `parameter_bounds` (Optional[Dict]): Bounds for parameters (**A-scale**). Should match the bounds used for the original fit if possible.
    *   `n_iterations` (int): Number of bootstrap replicates to run.
    *   `confidence_level` (float): Confidence level for CIs (e.g., 0.95 for 95%)..
    *   `n_jobs` (int): Number of CPU cores to use (-1 for all).n each replicate *finishes* (or fails/times out): `callback(iter_index, status_str, data_dict)`. `data_dict` contains 'rss' on success or 'message' on failure.
    *   `solver_options` (Dict): Options for the ODE solver used within replicates.vidual replicate fit.
    *   `end_callback` (Optional[Callable]): Function called when each replicate *finishes* (or fails/times out): `callback(iter_index, status_str, data_dict)`. `data_dict` contains 'rss' on success or 'message' on failure.
    *   `timeout_per_replicate` (Optional[float]): Timeout in seconds for each individual replicate fit.
    *   `return_replicate_params` (bool): If True, the `raw_parameter_list` attribute of the result will be populated.
    *   Returns: `BootstrapResult` object or `None` if bootstrap fails completely.meter_bounds_pool=None, solver_options={}, optimizer_options={}, score_weights=None)`**
    *   Fits multiple models and ranks them.
*   **`discover_kinetic_models(datasets, models_to_try, initial_guesses_pool, parameter_bounds_pool=None, solver_options={}, optimizer_options={}, score_weights=None)`**
    *   Fits multiple models and ranks them.efining models, e.g., `[{'name': 'F1', 'type': 'single_step', 'def_args': {'f_alpha_model': 'F1'}}, ...]`. `name` is user-friendly, `type` matches internal ODE system keys, `def_args` defines specifics.
    *   `datasets`: List of `KineticDataset` objects.s user-defined model `name` to its A-scale initial guess dictionary.
    *   `models_to_try` (List[Dict]): List defining models, e.g., `[{'name': 'F1', 'type': 'single_step', 'def_args': {'f_alpha_model': 'F1'}}, ...]`. `name` is user-friendly, `type` matches internal ODE system keys, `def_args` defines specifics.
    *   `initial_guesses_pool` (Dict[str, Dict]): Maps user-defined model `name` to its A-scale initial guess dictionary.
    *   `parameter_bounds_pool` (Optional[Dict[str, Dict]]): Maps user-defined model `name` to its A-scale bounds dictionary.faults if None.
    *   `solver_options`, `optimizer_options`: Applied to each model fit.'`, `'model_name'`, `'parameters'`, `'stats'`.
    *   `score_weights` (Optional[Dict]): Weights for ranking criteria (e.g., `{'bic': 0.4, 'r_squared': 0.4, ...}`). Uses defaults if None.
    *   Returns: List of dictionaries, ranked by score, containing `'rank'`, `'model_name'`, `'parameters'`, `'stats'`.ion_time_sec=None, solver_options={})`**
    *   Performs a single kinetic simulation.
*   **`simulate_kinetics(model_name, model_definition_args, kinetic_params, initial_alpha, temperature_program, simulation_time_sec=None, solver_options={})`**
    *   Performs a single kinetic simulation.ents defining the specific model variant.
    *   `model_name` (str): Internal model type name (e.g., 'single_step').e**).
    *   `model_definition_args` (Dict): Arguments defining the specific model variant.
    *   `kinetic_params` (Dict): Dictionary of kinetic parameters (**A-scale**).T(t)` or tuple `(time_sec, temp_K)`.
    *   `initial_alpha` (float): Starting conversion. Specific times (seconds) for output. Required if `temperature_program` is a callable.
    *   `temperature_program` (Callable | Tuple): Temperature profile function `T(t)` or tuple `(time_sec, temp_K)`.
    *   `simulation_time_sec` (Optional[np.ndarray]): Specific times (seconds) for output. Required if `temperature_program` is a callable.
    *   `solver_options` (Dict): ODE solver options.
    *   Returns: `PredictionResult` object (without CIs).e_program, simulation_time_sec=None, initial_alpha=0.0, solver_options={}, bootstrap_result=None)`**
    *   Convenience function using a `FitResult`. Calculates CIs if `BootstrapResult` is provided.
*   **`predict_conversion(kinetic_description, temperature_program, simulation_time_sec=None, initial_alpha=0.0, solver_options={}, bootstrap_result=None)`**
    *   Convenience function using a `FitResult`. Calculates CIs if `BootstrapResult` is provided.
    *   `kinetic_description` (FitResult): The result from `fit_kinetic_model`.trap` for the *same model* as in `kinetic_description`.
    *   Other arguments are similar to `simulate_kinetics`.luding `conversion_ci`.
    *   `bootstrap_result` (Optional[BootstrapResult]): Results from `run_bootstrap` for the *same model* as in `kinetic_description`.
    *   Returns: `PredictionResult` object, potentially including `conversion_ci`.
    *   Perform isoconversional analysis.
*   **`run_kas(datasets, alpha_levels=...)`, `run_ofw(...)`, `run_friedman(...)`**g rates.
    *   Perform isoconversional analysis.y of conversion values (0-1) at which to calculate Ea.
    *   `datasets`: List of `KineticDataset` objects from runs at different heating rates.
    *   `alpha_levels` (np.ndarray): Array of conversion values (0-1) at which to calculate Ea.
    *   Returns: `IsoResult` object.

### Helper Functions.construct_profile(segments, points_per_segment=50)`**
    *   Builds a time-temperature profile tuple `(time_sec, temp_K)` from a list of segments.
*   **`akts.utils.construct_profile(segments, points_per_segment=50)`**
    *   Builds a time-temperature profile tuple `(time_sec, temp_K)` from a list of segments.
    *   `segments` (List[Dict]): List defining segments:art_temp': float_K, 'end_temp': float_K}`
        *   `{'type': 'isothermal', 'duration': float_sec, 'temperature': float_K}`ime relative to segment start)
        *   `{'type': 'ramp', 'duration': float_sec, 'start_temp': float_K, 'end_temp': float_K}`
        *   `{'type': 'custom', 'time_array': array_sec, 'temp_array': array_K}` (time relative to segment start)
    *   Returns: `Tuple[np.ndarray, np.ndarray]` suitable for `temperature_program`.
---

## Contributing
re welcome! Please feel free to submit pull requests, report issues, or suggest enhancements on the GitHub repository: [Link to Your Repo].
Contributions are welcome! Please feel free to submit pull requests, report issues, or suggest enhancements on the GitHub repository: [Link to Your Repo].


## License
ct is licensed under the MIT License - see the LICENSE file for details.
This project is licensed under the MIT License - see the LICENSE file for details.


## Citation (Placeholder)
r research, please cite it as follows:
If you use akts in your research, please cite it as follows:[Provide citation details here once available - e.g., Zenodo DOI, paper reference].
[Provide citation details here once available - e.g., Zenodo DOI, paper reference].