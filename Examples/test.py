# Examples/test.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import time
# **** ADDED Optional ****
from typing import Optional
# **** END ADDED ****

# --- Import necessary components from your AKTS library ---
try:
    import akts.core as core
    import akts.isoconversional as iso
    import akts.models as models
    import akts.utils as utils
    # **** Import needed dataclasses ****
    from akts.datatypes import KineticDataset, FullAnalysisResult, IsoResult
except ImportError:
    import sys
    import os
    # **** ADDED Optional again for except block ****
    from typing import Optional
    # **** END ADDED ****
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    import akts.core as core
    import akts.isoconversional as iso
    import akts.models as models
    import akts.utils as utils
    # **** Import needed dataclasses again ****
    from akts.datatypes import KineticDataset, FullAnalysisResult, IsoResult

# Define constants
R_GAS = 8.314462 # J/(mol*K)

# --- 1. Generate Simulated ISOTHERMAL Accelerated Stability Data ---

def first_order_ode(t, alpha, k):
    """ ODE for first-order reaction d(alpha)/dt = k * (1-alpha). Returns list [dadt]. """
    alpha_val = alpha[0]
    alpha_clipped = np.clip(alpha_val, 0, 1)
    if alpha_clipped >= 1.0: return [0.0]
    dadt = k * (1 - alpha_clipped) # f(alpha) = (1-alpha) for n=1
    return [dadt]

def generate_pharma_stability_data(temperatures_C, duration_days, n_points=50, noise_level=0.005):
    """ Generates dummy isothermal stability data at different temperatures. """
    datasets = []
    # Realistic pharma parameters (example: Ea=90 kJ/mol, A=1e6 s^-1)
    true_A = 1e6 # s^-1
    true_Ea = 90000 # J/mol

    duration_sec = duration_days * 86400.0 # Convert duration to seconds

    print("Generating simulated isothermal stability data...")
    for i, T_C in enumerate(temperatures_C):
        T_K = T_C + 273.15
        # Calculate rate constant at this temperature
        k = true_A * np.exp(-true_Ea / (R_GAS * T_K))
        print(f"  Simulating at {T_C}°C (k={k:.3e} s^-1)...")

        t_eval = np.linspace(0, duration_sec, n_points)

        # Simulate using solve_ivp
        sol = solve_ivp(
            first_order_ode,
            (t_eval[0], t_eval[-1]),
            [0.0], # Initial alpha = 0
            t_eval=t_eval,
            args=(k,), # Pass rate constant k
            method='RK45', # RK45 is usually fine for simple isothermal
            rtol=1e-6, atol=1e-9
        )

        sim_time = sol.t
        sim_alpha = sol.y[0]

        # Add noise
        alpha_noisy = np.clip(sim_alpha + np.random.normal(0, noise_level, size=sim_alpha.shape), 0, 1)
        # Temperature is constant for the run
        temp_array = np.full_like(sim_time, T_K)

        dataset = KineticDataset(
            time=sim_time, # seconds
            temperature=temp_array, # Kelvin (constant)
            conversion=alpha_noisy,
            heating_rate=None, # Isothermal, no heating rate
            metadata={'Run': i+1, 'Simulated': True, 'Condition': f"{T_C}C"}
        )
        datasets.append(dataset)
        print(f"  Generated data for {T_C}°C. Final alpha: {alpha_noisy[-1]:.3f}")

    return datasets

# --- Simulation Parameters ---
accelerated_temps_C = [40.0, 50.0, 60.0] # Typical accelerated temps
study_duration_days = 180 # e.g., 6 months
num_data_points = 30 # Fewer points typical for stability studies
data_noise = 0.005

experimental_runs_data = generate_pharma_stability_data(
    accelerated_temps_C, study_duration_days, num_data_points, data_noise
)
print(f"Generated {len(experimental_runs_data)} isothermal datasets.")

# --- 2. Run the Full Analysis ---
iso_method_to_use = iso.run_iso_isothermal  # Use the updated function
# Target prediction: 25 C storage condition
prediction_temp_C = 25.0
# Target degradation level (e.g., 10% corresponds to 90% remaining)
prediction_alpha = 0.10
num_bootstrap = 100  # Keep low for testing
conf_level = 95.0
pred_unit = "days"  # Predict shelf life in days

# Initial guesses for SB-Sum fit
custom_initial_guesses_sb = {
    'A': 1e6, # Closer to simulated value
    'c1': 0.9, # Still favor first term
    'm1': 0.0,
    'n1': 1.0,
    'm2': 1.0,
    'n2': 1.0
}

print(f"\n--- Starting Full Kinetic Analysis (SB-Sum + Ea Variable on Isothermal Stability Data) ---")
print(f"--- Predicting time to {prediction_alpha*100:.1f}% degradation at {prediction_temp_C}°C ---")
start_time = time.time()

analysis_results: Optional[FullAnalysisResult] = core.run_full_analysis_sb_sum(
    experimental_runs=experimental_runs_data,
    iso_analysis_func=iso_method_to_use,  # This function now handles alpha_levels internally
    initial_params_sb=custom_initial_guesses_sb,
    target_temp_C=prediction_temp_C,
    target_alpha=prediction_alpha,
    n_bootstrap=num_bootstrap,
    confidence_level=conf_level,
    prediction_time_unit=pred_unit,
    perturb_bootstrap_guess=0.05
)

end_time = time.time()
print(f"--- Full analysis completed in {end_time - start_time:.2f} seconds ---")

# --- 3. Display Results ---
# (This section remains largely the same, but interpretation changes)
if analysis_results:
    print("\n--- Analysis Results Summary ---")
    if analysis_results.isoconversional_result:
        iso_res = analysis_results.isoconversional_result
        print(f"\nIsoconversional Method: {iso_res.method}")
        if iso_res.Ea is not None and len(iso_res.Ea) > 0:
             valid_ea = iso_res.Ea[np.isfinite(iso_res.Ea)]
             if len(valid_ea)>0:
                  print(f"  Calculated Ea Range: {np.min(valid_ea)/1000:.1f} - {np.max(valid_ea)/1000:.1f} kJ/mol (Note: Ea from isothermal data can be less reliable than non-iso)")
             else:
                  print("  Calculated Ea values are non-finite.")
             if len(iso_res.alpha) > 5:
                  idx_10 = np.argmin(np.abs(iso_res.alpha - 0.10))
                  idx_50 = np.argmin(np.abs(iso_res.alpha - 0.50))
                  # Only print if index is valid
                  if idx_10 < len(iso_res.Ea): print(f"  Ea at alpha~{iso_res.alpha[idx_10]:.2f}: {iso_res.Ea[idx_10]/1000:.1f} kJ/mol")
                  if idx_50 < len(iso_res.Ea): print(f"  Ea at alpha~{iso_res.alpha[idx_50]:.2f}: {iso_res.Ea[idx_50]/1000:.1f} kJ/mol")
        else:
             print("  Isoconversional Ea calculation failed or yielded no results.")


    if analysis_results.fit_result:
        fit_res = analysis_results.fit_result
        print(f"\nFitting Results (Model: {fit_res.model_name})")
        print(f"  Fit Success: {fit_res.success} ({fit_res.message})")
        print("  Fitted Parameters:")
        if fit_res.parameters:
            for param, value in fit_res.parameters.items():
                print(f"    {param}: {value:.4g}")
        else:
            print("    No parameters fitted.")

    if analysis_results.prediction_bootstrap_result:
        pred_res = analysis_results.prediction_bootstrap_result
        print("\nPrediction Results (Bootstrap):")
        print(f"  Target: {pred_res.target_description}")
        if np.isfinite(pred_res.predicted_value_median):
            print(f"  Median Predicted Value: {pred_res.predicted_value_median:.4g} {pred_res.unit}")
            print(f"  {pred_res.confidence_level}% Confidence Interval: "
                  f"({pred_res.predicted_value_ci[0]:.4g} - {pred_res.predicted_value_ci[1]:.4g}) {pred_res.unit}")
            print(f"  Based on {pred_res.n_iterations} successful bootstrap iterations reaching target.")
        else:
            print(f"  Target alpha not reached in any successful bootstrap iterations (Predicted time > t_max).")


    if analysis_results.prediction_from_original_fit:
        pred_orig = analysis_results.prediction_from_original_fit
        print("\nPrediction Results (Original Fit Parameters):")
        print(f"  Target: {pred_orig.target_description}")
        if np.isfinite(pred_orig.predicted_value):
            print(f"  Predicted Value: {pred_orig.predicted_value:.4g} {pred_orig.unit}")
        else:
            print(f"  Target alpha not reached within t_max.")

    # --- Optional: Check prediction against 6 months ---
    if analysis_results.prediction_from_original_fit and np.isfinite(analysis_results.prediction_from_original_fit.predicted_value):
         pred_days = analysis_results.prediction_from_original_fit.predicted_value
         if pred_unit.lower() != 'days': # Convert if necessary
              time_factor_to_days = 86400.0 / time_conversion_factors.get(pred_unit.lower(), 86400.0)
              pred_days *= time_factor_to_days
         six_months_days = 6 * 30.4 # Approximate
         print("\nComparison to 6 Months:")
         if pred_days > six_months_days:
              print(f"  Predicted time to {prediction_alpha*100:.1f}% degradation ({pred_days:.1f} days) is LONGER than 6 months.")
         else:
              print(f"  Predicted time to {prediction_alpha*100:.1f}% degradation ({pred_days:.1f} days) is SHORTER than 6 months.")


else:
    print("\nAnalysis failed. Please check logs or input data.")


# --- 4. Generate Plots ---
# (Plotting section remains largely the same, titles updated)
if analysis_results:
    print("\nGenerating plots...")
    plt.style.use('seaborn-v0_8-darkgrid')

    # Plot 1: Ea vs Alpha
    if analysis_results.isoconversional_result:
        iso_res = analysis_results.isoconversional_result
        if iso_res.alpha is not None and iso_res.Ea is not None and len(iso_res.alpha)>0:
            valid_mask_iso = np.isfinite(iso_res.Ea)
            if np.any(valid_mask_iso):
                plt.figure(figsize=(8, 6))
                plt.plot(iso_res.alpha[valid_mask_iso], iso_res.Ea[valid_mask_iso] / 1000, marker='o', linestyle='-', label=iso_res.method)
                # ... (error bar plotting remains the same) ...
                if iso_res.Ea_std_err is not None:
                     valid_mask_err = valid_mask_iso & np.isfinite(iso_res.Ea_std_err)
                     if np.any(valid_mask_err):
                          lower_bound = (iso_res.Ea[valid_mask_err] - iso_res.Ea_std_err[valid_mask_err]) / 1000
                          upper_bound = (iso_res.Ea[valid_mask_err] + iso_res.Ea_std_err[valid_mask_err]) / 1000
                          plt.fill_between(iso_res.alpha[valid_mask_err], lower_bound, upper_bound,
                                           alpha=0.2, label='Std Err')
                plt.xlabel("Conversion (alpha)")
                plt.ylabel("Activation Energy (kJ/mol)")
                plt.title("Isoconversional Ea (Simulated Isothermal Data)") # Updated title
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
            else: print("Skipping Ea vs Alpha plot: No finite Ea values.")
        else: print("Skipping Ea vs Alpha plot: No isoconversional results.")


    # Plot 2: Experimental Data vs Fitted Model Curves
    if analysis_results.fit_result and analysis_results.isoconversional_result:
        fit_res = analysis_results.fit_result
        iso_res = analysis_results.isoconversional_result
        if fit_res.success and iso_res.alpha is not None and iso_res.Ea is not None and len(iso_res.alpha) > 1:
            ea_interpolator = utils.create_ea_interpolator(iso_res.alpha, iso_res.Ea)
            fit_params = fit_res.parameters

            plt.figure(figsize=(10, 7))
            colors = plt.cm.viridis(np.linspace(0, 1, len(experimental_runs_data)))

            for i, run_data in enumerate(experimental_runs_data):
                time_exp = run_data.time
                alpha_exp = run_data.conversion
                temp_exp = run_data.temperature
                if len(time_exp) < 2: continue

                # For plotting isothermal fit, simulate under constant temp
                T_K_run = temp_exp[0] # Get the constant temp for this run
                temp_interp_const = lambda t: T_K_run
                alpha_initial = alpha_exp[0]
                t_span = (time_exp[0], time_exp[-1])

                try:
                    required_ode_params = ['A', 'c1', 'm1', 'n1', 'm2', 'n2']
                    if not all(p in fit_params for p in required_ode_params): continue

                    # Simulate using the constant temp interpolator for this run
                    sol = solve_ivp(
                        core._kinetic_ode_ea_variable,
                        t_span, [alpha_initial], t_eval=time_exp,
                        args=(fit_params['A'], models.f_sb_sum,
                              (fit_params['c1'], fit_params['m1'], fit_params['n1'],
                               fit_params['m2'], fit_params['n2']),
                              temp_interp_const, ea_interpolator), # Use constant temp interp
                        method='Radau'
                    )
                    plt.plot(time_exp / 86400.0, alpha_exp, 'o', color=colors[i], markersize=4,
                             label=f"Exp {run_data.metadata.get('Condition','Run '+str(i+1))}") # Plot time in days
                    if sol.status == 0 and len(sol.t) == len(time_exp):
                        plt.plot(sol.t / 86400.0, sol.y[0], '-', color=colors[i], linewidth=1.5) # Plot time in days
                    else: print(f"Warning: ODE integration failed/mismatch for plotting fit of run {i+1}")
                except Exception as plot_e: print(f"Error during simulation for plotting run {i+1}: {plot_e}")

            plt.xlabel("Time (days)") # Updated label
            plt.ylabel("Conversion (alpha)")
            plt.title(f"Isothermal Data vs Fitted '{fit_res.model_name}' Model") # Updated title
            plt.legend(fontsize='small', ncol=1) # Adjusted legend
            plt.grid(True)
            plt.ylim(-0.05, 1.05)
            plt.tight_layout()
        else: print("Skipping data vs fit plot: Fit failed or missing isoconversional results.")


    # Plot 3: Distribution of Bootstrap Prediction
    # (Remains the same, checks for finite values)
    if analysis_results.prediction_bootstrap_result:
        pred_res = analysis_results.prediction_bootstrap_result
        dist = pred_res.predicted_value_distribution
        if dist is not None and len(dist) > 0:
            finite_dist = dist[np.isfinite(dist)]
            if len(finite_dist) > 1:
                plt.figure(figsize=(8, 6))
                plt.hist(finite_dist, bins=30, density=True, alpha=0.7, label='Bootstrap Distribution (Finite Results)')
                if np.isfinite(pred_res.predicted_value_median): plt.axvline(pred_res.predicted_value_median, color='red', linestyle='--', label=f'Median: {pred_res.predicted_value_median:.3g}')
                if np.isfinite(pred_res.predicted_value_ci[0]): plt.axvline(pred_res.predicted_value_ci[0], color='black', linestyle=':', label=f'CI Lower: {pred_res.predicted_value_ci[0]:.3g}')
                if np.isfinite(pred_res.predicted_value_ci[1]): plt.axvline(pred_res.predicted_value_ci[1], color='black', linestyle=':', label=f'CI Upper: {pred_res.predicted_value_ci[1]:.3g}')
                plt.xlabel(f"Predicted Value ({pred_res.unit})")
                plt.ylabel("Density")
                plt.title(f"Bootstrap Distribution for: {pred_res.target_description}")
                plt.legend(fontsize='small')
                plt.grid(True)
                plt.tight_layout()
            else: print("Skipping bootstrap distribution plot: Not enough finite prediction values.")
        else: print("Skipping bootstrap distribution plot: Distribution data not available.")

    plt.show()

else:
    print("Cannot generate plots because analysis failed.")

print("\nExample script finished.")