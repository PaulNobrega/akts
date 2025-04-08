# akts/__init__.py

# Import key classes and functions to expose them at the top level

# --- Import Datatypes ---
from .datatypes import (
    KineticDataset,
    IsoResult,
    FitResult,
    ParameterBootstrapResult,
    PredictionCurveResult,
    BootstrapPredictionResult,
    SingleValuePredictionResult,
    FullAnalysisResult
)

# --- Import Core Functions ---
from .core import (
    # --- Keep existing core functions if they are still relevant ---
    fit_kinetic_model, # Assuming this is kept
    run_bootstrap,     # Assuming this is the parameter bootstrap
    predict_conversion,# Assuming this predicts a curve
    # simulate_kinetics, # Often internal, maybe not needed top-level
    # discover_kinetic_models, # Keep if used

    # --- Add the new main orchestration function ---
    run_full_analysis_sb_sum,

    # --- Add other core functions users might need directly ---
    fit_sb_sum_model_ea_variable,
    predict_time_to_alpha_iso,
)

# --- Import Isoconversional Functions ---
from .isoconversional import (
    run_friedman,
    run_kas,
    run_ofw,
    # --- REPLACE vyazovkin_method with the ACTUAL name ---
    run_vyazovkin, # <<< EXAMPLE: Replace with actual name found in isoconversional.py
    run_iso_isothermal
    # --- END REPLACEMENT ---
)

# --- Import Models & Utils (Optional) ---
from . import models
from . import utils


# Define __all__ to control 'from akts import *' behavior
__all__ = [
    # Datatypes
    'KineticDataset',
    'IsoResult',
    'FitResult',
    'ParameterBootstrapResult',
    'PredictionCurveResult',
    'BootstrapPredictionResult',
    'SingleValuePredictionResult',
    'FullAnalysisResult',

    # Core functions
    'fit_kinetic_model', # Include if kept
    'run_bootstrap',     # Include if kept (parameter bootstrap)
    'predict_conversion',# Include if kept (curve prediction)
    # 'discover_kinetic_models', # Include if kept
    'run_full_analysis_sb_sum',
    'fit_sb_sum_model_ea_variable',
    'predict_time_to_alpha_iso',

    # Isoconversional functions
    'run_friedman',
    'run_kas',
    'run_ofw',
    # --- REPLACE vyazovkin_method with the ACTUAL name ---
    'run_vyazovkin', # <<< EXAMPLE: Replace with actual name
    'run_iso_isothermal',
    # --- END REPLACEMENT ---

    # Exposed Modules (Optional)
    'models',
    'utils',
]

# Define a version for your library
__version__ = "0.2.0"