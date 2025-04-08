# AKTSlib/__init__.py

# Import key classes and functions to expose them at the top level
from .datatypes import KineticDataset, FitResult, IsoResult, BootstrapResult, PredictionResult
# **** ADD discover_kinetic_models ****
from .core import (fit_kinetic_model, run_bootstrap, predict_conversion,
                   simulate_kinetics, discover_kinetic_models)
from .isoconversional import run_friedman, run_kas, run_ofw
# You might also want to expose specific models or the registry if users need them directly
# from .models import F_ALPHA_MODELS, ODE_SYSTEMS

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
    'discover_kinetic_models', # **** ADDED here ****
    # Isoconversional functions
    'run_friedman',
    'run_kas',
    'run_ofw',
    # Add model names or registries here if desired in __all__
]

# Optional: Define a version for your library
__version__ = "0.1.3" # Incremented version slightly