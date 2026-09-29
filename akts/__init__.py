# AKTSlib/__init__.py

# Import key classes and functions to expose them at the top level
from .datatypes import KineticDataset, FitResult, IsoResult, BootstrapResult, PredictionResult
from .core import (fit_kinetic_model, run_bootstrap, predict_conversion,
                   simulate_kinetics, discover_kinetic_models, predict_conversion_model_free, rank_models)
from .isoconversional import (run_friedman, run_kas, run_ofw, run_vyazovkin,
                               run_kissinger, estimate_compensation_parameters)
# Expose model registration and listing functions
from .models import (register_f_alpha_model, register_ode_system, list_available_models,
                     F_ALPHA_MODELS, ODE_SYSTEMS)
# Expose data loaders
from .loaders import (load_data_file, load_dsc_file, load_tga_file, load_isothermal_file)
# Expose helper functions (high-level user-friendly functions)
from .helpers import auto_model_isothermal_data, time_to_conversion, time_to_conversion_model_free, calculate_ich_q1e_ceiling
# Expose validation functions
from .validation import run_leave_one_out_cv
# Expose export utilities
from .utils import export_prediction_report
# Expose plotting convenience functions
from .plotting import (plot_ea_vs_alpha, plot_fit_overlay, plot_bootstrap_ci_bands,
                       plot_arrhenius, plot_parameter_distributions, plot_multi_temperature_data)
# Expose JSON utilities for API integration
from .json_utils import parse_json_data, serialize_results_to_json, convert_numpy_to_python
# Expose model selector for IDE autocomplete
from .model_selector import models
# Expose load_instrument_file as alias for load_data_file for backward compatibility
load_instrument_file = load_data_file

# Optional: Define __all__ to control 'from AKTSlib import *' behavior
__all__ = [
    # Datatypes
    'KineticDataset',
    'FitResult',
    'IsoResult',
    'BootstrapResult',
    'PredictionResult',
    # Core functions
    'fit_kinetic_model',
    'run_bootstrap',
    'predict_conversion',
    'simulate_kinetics',
    'discover_kinetic_models',
    'predict_conversion_model_free',
    'rank_models',
    # High-level helper functions
    'auto_model_isothermal_data',
    'time_to_conversion',
    'time_to_conversion_model_free',
    'calculate_ich_q1e_ceiling',
    # Validation functions
    'run_leave_one_out_cv',
    # Export utilities
    'export_prediction_report',
    # Plotting functions
    'plot_ea_vs_alpha',
    'plot_fit_overlay',
    'plot_bootstrap_ci_bands',
    'plot_arrhenius',
    'plot_parameter_distributions',
    'plot_multi_temperature_data',
    # Isoconversional functions
    'run_friedman',
    'run_kas',
    'run_ofw',
    'run_vyazovkin',
    'run_kissinger',
    'estimate_compensation_parameters',
    # Model registration and listing
    'register_f_alpha_model',
    'register_ode_system',
    'list_available_models',
    'F_ALPHA_MODELS',
    'ODE_SYSTEMS',
    # Data loaders
    'load_data_file',
    'load_dsc_file',
    'load_tga_file',
    'load_isothermal_file',
    'load_instrument_file',  # Alias for backward compatibility
    # JSON utilities
    'parse_json_data',
    'serialize_results_to_json',
    'convert_numpy_to_python',
    # Model selector
    'models',
]

# Library version
__version__ = "0.2.1"