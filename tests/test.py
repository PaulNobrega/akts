# test.py
import numpy as np
import matplotlib.pyplot as plt
import time as timer
import traceback
import sys # For flushing output
# import threading # No longer needed for lock

# Import necessary components directly from the AKTSlib package
from akts import (
    KineticDataset,
    fit_kinetic_model,
    run_bootstrap,
    predict_conversion,
    run_kas,
    run_friedman,
    run_ofw,
    FitResult,
    simulate_kinetics, # Import the new simulation function
    discover_kinetic_models
)
from akts.utils import construct_profile  # Import the new helper

# --- Define Constants ---
T0 = 298.15

# --- 1. Generate Synthetic Data ---
true_params = {
    'Ea1': 80e3, 'A1': 1e10, 'f1_model': 'F1',
    'Ea2': 95e3, 'A2': 5e11, 'f2_model': 'F1',
}

def generate_data(heating_rate_K_min, t_end_min, n_points=10, noise_level=0.01):
    """Generates one dataset for A->B->C using the public simulate_kinetics"""
    beta_K_s = heating_rate_K_min / 60.0
    time_s = np.linspace(0, t_end_min * 60.0, n_points)
    temperature_K = T0 + beta_K_s * time_s
    temp_program_sec = (time_s, temperature_K)
    model_name = "A->B->C"
    model_def_args = {'f1_model': true_params['f1_model'], 'f2_model': true_params['f2_model']}
    kinetic_params_A = {k: v for k, v in true_params.items() if k in ['Ea1', 'A1', 'Ea2', 'A2']}
    try:
        sim_result = simulate_kinetics(model_name=model_name, model_definition_args=model_def_args, kinetic_params=kinetic_params_A, initial_alpha=0.0, temperature_program=temp_program_sec)
        alpha_sim_true = sim_result.conversion
    except Exception as e: print(f"Error during data generation simulation: {e}"); return KineticDataset(time=np.array([]), temperature=np.array([]), conversion=np.array([]))
    noise = np.random.normal(0, noise_level, size=alpha_sim_true.shape)
    alpha_noisy = np.clip(alpha_sim_true + noise, 0.0, 1.0)
    return KineticDataset(time=time_s / 60.0, temperature=temperature_K, conversion=alpha_noisy, heating_rate=heating_rate_K_min)

# --- Define Callbacks ---
def fit_callback(iteration, params_A, replicate_idx=None):
    """Callback function for fit_kinetic_model."""
    prefix = f"  [Rep {replicate_idx+1}]" if replicate_idx is not None else " "
    if iteration % 10 == 0:
        param_str_parts = []
        # **** Iterate through all items, format based on key ****
        for k, v in params_A.items():
            if k.startswith('A') and not k.startswith('Ai'): # Handle A, A1, A2 etc.
                param_str_parts.append(f"{k}={v:.2e}")
            elif k.startswith('Ea'): # Handle Ea, Ea1, Ea2 etc.
                param_str_parts.append(f"{k}={v/1000:.1f}")
            elif k in ['n', 'm'] or k.startswith('p1_') or k.startswith('p2_'): # Handle n, m, p1_n etc.
                param_str_parts.append(f"{k}={v:.2f}") # Format with 2 decimal places
            else: # Handle other params like initial_ratio_r
                param_str_parts.append(f"{k}={v:.3f}")
        param_str = ", ".join(param_str_parts)
        print(f"{prefix} Fit Iter: {iteration}, Params: {param_str}", flush=True)

# **** REVERTED bootstrap_end_callback ****
def bootstrap_end_callback(iteration, status, data_dict):
    """Callback function for run_bootstrap (reports END status)."""
    details = ""
    if data_dict:
        if 'rss' in data_dict: details = f" RSS={data_dict['rss']:.3e}"
        elif 'message' in data_dict: msg = data_dict['message']; details = f" Msg={msg}"
    # Just print the status of the finished iteration on a new line
    print(f"\n  Bootstrap Iter: {iteration+1} finished, Status: {status}{details}", flush=True)

# **** ADD MAIN GUARD ****
if __name__ == '__main__':
    # --- Generate Data loop ---
    heating_rates = [2.0, 5.0, 10.0, 15.0, 20.0]
    end_times = [90, 45, 30, 25, 20]
    datasets = []
    print("Generating synthetic data...")
    for beta, t_end in zip(heating_rates, end_times):
        ds = generate_data(beta, t_end, n_points=25, noise_level=0.015)
        if len(ds.time) > 0: datasets.append(ds); print(f"  Generated data for {beta} K/min")
        else: print(f"  Failed to generate data for {beta} K/min")
    if not datasets: print("Error: No datasets generated."); exit()
    print("Data generation complete.")

    # --- Isoconversional Analysis ---
    print("\nRunning Isoconversional Analysis...")
    try:
        kas_result = run_kas(datasets)
        friedman_result = run_friedman(datasets)
        print("KAS Ea (kJ/mol):", np.round(kas_result.Ea[np.isfinite(kas_result.Ea)] / 1000, 1))
        print("Friedman Ea (kJ/mol):", np.round(friedman_result.Ea[np.isfinite(friedman_result.Ea)] / 1000, 1))
        plt.figure(figsize=(8, 6))
        if np.any(np.isfinite(kas_result.Ea)): plt.plot(kas_result.alpha[np.isfinite(kas_result.Ea)], kas_result.Ea[np.isfinite(kas_result.Ea)] / 1000, 'bo-', label='KAS')
        if np.any(np.isfinite(friedman_result.Ea)): plt.plot(friedman_result.alpha[np.isfinite(friedman_result.Ea)], friedman_result.Ea[np.isfinite(friedman_result.Ea)] / 1000, 'rs--', label='Friedman')
        plt.xlabel("Conversion (alpha)"); plt.ylabel("Activation Energy (kJ/mol)"); plt.title("Isoconversional Activation Energy")
        plt.legend(); plt.grid(True); plt.ylim(bottom=50, top=150)
        plt.savefig("test_isoconversional_Ea.png"); print("Saved isoconversional plot to test_isoconversional_Ea.png"); plt.close()
    except Exception as e: print(f"Isoconversional analysis failed: {e}"); traceback.print_exc()


    # --- Model Fitting ---
    print("\nRunning Model Fitting (A->B->C with F1 steps, logA scaling)...")
    fit_model_name = "A->B->C"
    fit_model_def_args = {'f1_model': 'F1', 'f2_model': 'F1'}

    # **** PROVIDE GUESSES/BOUNDS FOR A (not logA) ****
    initial_guesses_A = {
        'Ea1': 80e3, 'A1': 1e10, 'p1_n': 1.0,  # Add p1_n
        'Ea2': 95e3, 'A2': 5e11, 'p2_n': 1.0   # Add p2_n
    }
    # Define bounds for A
    bounds_A = {
        'Ea1': (10e3, 200e3), 'A1': (1e5, 1e15), 'p1_n': (1.0, 1.0),  # Fix n=1
        'Ea2': (10e3, 200e3), 'A2': (1e5, 1e15), 'p2_n': (1.0, 1.0)   # Fix n=1
    }

    optimizer_options = { 'method': 'L-BFGS-B', 'options': {'disp': False, 'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8} }
    solver_options = { 'primary_solver': 'LSODA', 'fallback_solver': 'BDF', 'rtol': 1e-6, 'atol': 1e-9 }

    start_time = timer.time()
    fit_result = None
    try:
        # By default, optimize_on_rate=True is now used in fit_kinetic_model
        fit_result = fit_kinetic_model(
            datasets=datasets, model_name=fit_model_name,
            model_definition_args=fit_model_def_args,
            initial_guesses=initial_guesses_A,  # Pass A guesses
            parameter_bounds=bounds_A,  # Pass A bounds
            optimizer_options=optimizer_options,
            solver_options=solver_options,
            callback=fit_callback
            # optimize_on_rate=True  # Explicitly use rate objective (optional, now default)
            # optimize_on_rate=False  # Explicitly use conversion objective
        )
        print()
        end_time = timer.time()
        print(f"Fitting completed in {end_time - start_time:.2f} seconds.")
        if isinstance(fit_result, FitResult) and fit_result.success:
            print("Fit Successful!")
            print("Fitted Parameters (A converted back from logA):")
            print(f"  Param | Fitted       | True")
            print(f"  ------|--------------|--------------")
            for name, val in fit_result.parameters.items():
                true_val = true_params.get(name, np.nan)
                if 'Ea' in name: print(f"  {name:<5}| {val/1000:>12.2f} | {true_val/1000:>12.2f} kJ/mol")
                elif 'A' in name: print(f"  {name:<5}| {val:>12.2e} | {true_val:>12.2e} 1/s")
                else: print(f"  {name:<5}| {val:>12.3f} | {true_val:>12.3f}")
            print(f"RSS: {fit_result.rss:.4e}")
            print(f"R-squared: {fit_result.r_squared:.4f}" if fit_result.r_squared is not None else "R-squared: nan")
            print(f"AICc: {fit_result.aic:.2f}" if fit_result.aic is not None else "AICc: nan")
            print(f"BIC: {fit_result.bic:.2f}" if fit_result.bic is not None else "BIC: nan")
        elif isinstance(fit_result, FitResult): print(f"Fit Failed: {fit_result.message}")
        else: print("Fit Failed: No result returned.")
    except Exception as e: print(f"Model fitting failed: {e}"); traceback.print_exc()


    # --- Run Model Discovery ---
    print("\n--- Running Model Discovery ---")
    # Define models to try
    models_to_try = [
        {'name': 'F1', 'type': 'single_step', 'def_args': {'f_alpha_model': 'F1'}},  # n=1 is default in f_n_order
        {'name': 'A2', 'type': 'single_step', 'def_args': {'f_alpha_model': 'A2', 'f_alpha_params': {'n': 2.0}}},  # Specify n=2
        {'name': 'A3', 'type': 'single_step', 'def_args': {'f_alpha_model': 'A3', 'f_alpha_params': {'n': 3.0}}},  # Specify n=3
        {'name': 'SB(m,n)', 'type': 'single_step', 'def_args': {'f_alpha_model': 'SB_mn', 'f_alpha_params': {'m': 0.5, 'n': 0.5}}},  # Specify m, n
        {'name': 'A->B->C (F1,F1)', 'type': 'A->B->C', 'def_args': {'f1_model': 'F1', 'f2_model': 'F1'}},  # n=1 default for both steps
    ]

    # Define initial guesses pool (A scale) - ONLY Ea and A
    initial_guesses_pool = {
        'F1': {'Ea': 85e3, 'A': 1e11},
        'A2': {'Ea': 88e3, 'A': 5e11},  # No guess for n needed
        'A3': {'Ea': 90e3, 'A': 1e12},  # No guess for n needed
        'SB(m,n)': {'Ea': 85e3, 'A': 1e11},  # No guess for m, n needed
        'A->B->C (F1,F1)': {'Ea1': 80e3, 'A1': 1e10, 'Ea2': 95e3, 'A2': 5e11},  # No guess for p1_n, p2_n
    }

    # Define bounds pool (A scale) - ONLY Ea and A
    parameter_bounds_pool = {
        'F1': {'Ea': (10e3, 200e3), 'A': (1e5, 1e15)},
        'A2': {'Ea': (10e3, 200e3), 'A': (1e5, 1e15)},  # No bound for n needed
        'A3': {'Ea': (10e3, 200e3), 'A': (1e5, 1e15)},  # No bound for n needed
        'SB(m,n)': {'Ea': (10e3, 200e3), 'A': (1e5, 1e15)},  # No bound for m, n needed
        'A->B->C (F1,F1)': {'Ea1': (10e3, 200e3), 'A1': (1e5, 1e15), 'Ea2': (10e3, 200e3), 'A2': (1e5, 1e15)},  # No bound for p1_n, p2_n
    }

    # Define common solver/optimizer options
    solver_options = {'primary_solver': 'LSODA', 'fallback_solver': 'BDF', 'rtol': 1e-6, 'atol': 1e-9}
    optimizer_options = {'method': 'L-BFGS-B', 'options': {'disp': False, 'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8}}

    ranked_models = []
    try:
        custom_score_weights = {'bic': 0.4, 'r_squared': 0.4, 'rss': 0.1, 'n_params': 0.1}
        # Call discover_kinetic_models
        ranked_models = discover_kinetic_models(
            datasets=datasets,
            models_to_try=models_to_try,
            initial_guesses_pool=initial_guesses_pool,
            parameter_bounds_pool=parameter_bounds_pool,
            solver_options=solver_options,
            optimizer_options=optimizer_options,
            score_weights=custom_score_weights
        )
        # Print ranked models
        if ranked_models:
            for item in ranked_models:
                stats = item['stats']
                print(f"  Rank {item['rank']} (Score={item['score']:.3f}): {item['model_name']}")
                print(f"    BIC = {stats.get('bic', np.nan):.2f}, AICc = {stats.get('aic', np.nan):.2f}, R² = {stats.get('r_squared', np.nan):.4f}, RSS = {stats.get('rss', np.nan):.3e}")
                param_str = ", ".join([f"{k}={v:.2e}" if 'A' in k else f"{k}={v/1000:.1f}" for k, v in item['parameters'].items()])
                print(f"    Params: [{param_str}]")
            print("\n--- Model Discovery Complete ---")
        else:
            print("  No models fitted successfully.")
    except Exception as e:
        print(f"Model discovery failed: {e}")
        traceback.print_exc()

    # --- Select Best Model and Run Bootstrap ---
    best_model_fit = None  # Initialize best_model_fit

    if ranked_models:
        best_model_info = ranked_models[0]  # Select the top-ranked model by SCORE
        print(f"\n--- Selecting top model '{best_model_info['model_name']}' (Rank {best_model_info['rank']}) for bootstrapping ---")
        selected_model_def = next((m for m in models_to_try if m['name'] == best_model_info['model_name']), None)
        if selected_model_def:
            selected_guesses = initial_guesses_pool.get(selected_model_def['name'])
            selected_bounds = parameter_bounds_pool.get(selected_model_def['name']) if parameter_bounds_pool else None
            print("Re-fitting the best model...")
            # Assign result to best_model_fit
            best_model_fit = fit_kinetic_model(
                datasets=datasets, model_name=selected_model_def['type'],
                model_definition_args=selected_model_def['def_args'],
                initial_guesses=selected_guesses,  # Pass A-scale guesses
                parameter_bounds=selected_bounds,  # Pass A-scale bounds
                optimizer_options=optimizer_options, solver_options=solver_options,
                callback=fit_callback
            )
            print()
            if isinstance(best_model_fit, FitResult) and best_model_fit.success:
                print("Re-fit successful.")
            else:
                print("Re-fit failed. Cannot run bootstrap.")
                best_model_fit = None
        else:
            print(f"Could not find definition for selected model '{best_model_info['model_name']}'.")
            best_model_fit = None

    # --- Bootstrapping ---
    bootstrap_result = None
    if isinstance(best_model_fit, FitResult) and best_model_fit.success:
        print(f"\nRunning Bootstrap Analysis for {best_model_fit.model_name}...")
        start_time = timer.time()
        try:
            parameter_bounds_A = parameter_bounds_pool.get(best_model_fit.model_name) if parameter_bounds_pool else None
            n_iterations_bootstrap = 20
            timeout_bootstrap = 180.0
            bootstrap_optimizer_options = {'method': 'L-BFGS-B', 'options': {'disp': False, 'maxiter': 100, 'ftol': 1e-9, 'gtol': 1e-7}}

            bootstrap_result = run_bootstrap(
                datasets=datasets,
                fit_result=best_model_fit,  # Pass best_model_fit
                optimizer_options=bootstrap_optimizer_options,
                parameter_bounds=parameter_bounds_A,
                solver_options=solver_options,
                n_iterations=n_iterations_bootstrap,
                confidence_level=0.95, n_jobs=-1,
                end_callback=bootstrap_end_callback,
                timeout_per_replicate=timeout_bootstrap,
                return_replicate_params=True
            )
            # ...existing bootstrap processing...
        except Exception as e:
            print(f"Bootstrap failed: {e}")
            traceback.print_exc()
            bootstrap_result = None
    else:
        print("\nSkipping Bootstrap Analysis because model selection/re-fit failed.")

    # --- Prediction & Plotting ---
    if isinstance(best_model_fit, FitResult) and best_model_fit.success:
        print("\nRunning Prediction and Plotting...")
        try:
            plt.figure(figsize=(10, 7))
            colors = plt.cm.viridis(np.linspace(0, 1, len(datasets)))
            for i, ds in enumerate(datasets):
                time_sec = ds.time * 60.0
                temp_prog_sec = (time_sec, ds.temperature)
                prediction = predict_conversion(
                    kinetic_description=best_model_fit,  # Use best_model_fit
                    temperature_program=temp_prog_sec,
                    bootstrap_result=bootstrap_result,
                    solver_options=solver_options
                )
                pred_time_min = prediction.time / 60.0
                plt.plot(ds.time, ds.conversion, 'o', color=colors[i], markersize=5, label=f'{ds.heating_rate} K/min (Exp.)')
                plt.plot(pred_time_min, prediction.conversion, '-', color=colors[i], linewidth=2, label=f'{ds.heating_rate} K/min (Fit)')
                if bootstrap_result and prediction.conversion_ci is not None:
                    lower_ci, upper_ci = prediction.conversion_ci
                    if lower_ci is not None and upper_ci is not None and len(lower_ci) == len(pred_time_min) and len(upper_ci) == len(pred_time_min):
                        valid_ci = np.isfinite(lower_ci) & np.isfinite(upper_ci)
                        plt.fill_between(pred_time_min[valid_ci], lower_ci[valid_ci], upper_ci[valid_ci], color=colors[i], alpha=0.2)
                    else:
                        print(f"Warning: Invalid CIs for dataset {i}.")
            plt.xlabel("Time (min)")
            plt.ylabel("Conversion (alpha)")
            plt.title(f"Global Fit ({best_model_fit.model_name}, R²={best_model_fit.r_squared:.3f}) and CIs")
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys(), fontsize='small', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True)
            plt.ylim(-0.05, 1.05)
            plt.tight_layout(rect=[0, 0, 0.80, 1])
            plt.savefig("test_fit_and_prediction.png")
            print("Saved fit plot to test_fit_and_prediction.png")
            plt.close()

            # --- New Program Prediction ---
            print("\nPredicting for a new program: 10 K/min ramp then 400K isotherm...")
            ramp_rate_K_s = 10.0 / 60.0
            target_iso_temp = 400.0
            t_ramp_end_s = (target_iso_temp - T0) / ramp_rate_K_s if ramp_rate_K_s > 1e-9 else 0
            if t_ramp_end_s < 0:
                t_ramp_end_s = 0
            t_iso_duration_s = 30 * 60
            t_total_s = t_ramp_end_s + t_iso_duration_s
            pred_time_s = np.linspace(0, t_total_s, 200)
            pred_temp_K = np.piecewise(
                pred_time_s,
                [pred_time_s <= t_ramp_end_s, pred_time_s > t_ramp_end_s],
                [lambda t: T0 + ramp_rate_K_s * t, target_iso_temp]
            )
            pred_temp_prog_sec = (pred_time_s, pred_temp_K)
            sim_model_def_args = getattr(best_model_fit, 'model_definition_args', {})
            new_prediction = simulate_kinetics(
                model_name=best_model_fit.model_name,
                model_definition_args=sim_model_def_args,
                kinetic_params=best_model_fit.parameters,
                initial_alpha=0.0,
                temperature_program=pred_temp_prog_sec,
                solver_options=solver_options
            )
            pred_time_min_new = new_prediction.time / 60.0
            plt.figure(figsize=(10, 6))
            plt.subplot(2, 1, 1)
            plt.plot(pred_time_min_new, new_prediction.temperature, 'r-')
            plt.ylabel("Temperature (K)")
            plt.title("Prediction for New Temperature Program")
            plt.grid(True)
            plt.subplot(2, 1, 2)
            plt.plot(pred_time_min_new, new_prediction.conversion, 'b-')
            plt.ylabel("Predicted Conversion (alpha)")
            plt.xlabel("Time (min)")
            plt.ylim(-0.05, 1.05)
            plt.grid(True)
            plt.tight_layout()
            plt.savefig("test_new_program_prediction.png")
            print("Saved new program prediction plot to test_new_program_prediction.png")
            plt.close()

            # --- Excursion Prediction ---
            print("\nPredicting for profile with excursion...")
            segments = [
                {'type': 'isothermal', 'duration': 2 * 24 * 3600, 'temperature': 278.15},
                {'type': 'ramp', 'duration': 1 * 3600, 'start_temp': 278.15, 'end_temp': 298.15},
                {'type': 'isothermal', 'duration': 8 * 3600, 'temperature': 298.15},
                {'type': 'ramp', 'duration': 1 * 3600, 'start_temp': 298.15, 'end_temp': 278.15},
                {'type': 'isothermal', 'duration': 2 * 24 * 3600, 'temperature': 278.15},
            ]
            excursion_time_sec, excursion_temp_K = construct_profile(segments, points_per_segment=20)
            excursion_temp_prog = (excursion_time_sec, excursion_temp_K)
            excursion_prediction = simulate_kinetics(
                model_name=best_model_fit.model_name,
                model_definition_args=sim_model_def_args,
                kinetic_params=best_model_fit.parameters,
                initial_alpha=0.0,
                temperature_program=excursion_temp_prog,
                solver_options=solver_options
            )
            pred_time_hr = excursion_prediction.time / 3600.0
            plt.figure(figsize=(10, 6))
            plt.subplot(2, 1, 1)
            plt.plot(pred_time_hr, excursion_prediction.temperature, 'r-')
            plt.ylabel("Temperature (K)")
            plt.title("Prediction for Profile with Temperature Excursion")
            plt.grid(True)
            plt.subplot(2, 1, 2)
            plt.plot(pred_time_hr, excursion_prediction.conversion, 'b-')
            plt.ylabel("Predicted Conversion (alpha)")
            plt.xlabel("Time (hours)")
            plt.ylim(-0.05, 1.05)
            plt.grid(True)
            plt.tight_layout()
            plt.savefig("test_excursion_prediction.png")
            print("Saved excursion prediction plot to test_excursion_prediction.png")
            plt.close()

        except Exception as e:
            print(f"Prediction or plotting failed: {e}")
            traceback.print_exc()
    else:
        print("\nSkipping Prediction and Plotting because model selection/re-fit failed.")

    print("\nTest script finished.")