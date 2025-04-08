# datatypes.py
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Callable, Any

# --- Core Data Structures ---

@dataclass
class KineticDataset:
    """Structure to hold experimental kinetic data for a single run."""
    time: np.ndarray
    temperature: np.ndarray # Should be in Kelvin for calculations
    conversion: np.ndarray
    heating_rate: Optional[float] = None # Optional heating rate (K/min or K/s)
    metadata: Dict[str, Any] = field(default_factory=dict) # Allow any metadata type

# --- Analysis Step Results ---

@dataclass
class IsoResult:
    """Structure for isoconversional analysis results."""
    # --- Non-Defaults ---
    method: str
    alpha: np.ndarray
    Ea: np.ndarray # Activation Energy in J/mol
    # --- Defaults ---
    Ea_std_err: Optional[np.ndarray] = None # Standard error if calculated
    regression_stats: Optional[List[Dict[str, Any]]] = None # List of stats per alpha point if available

@dataclass
class FitResult:
    """Structure for model fitting results based on the original dataset."""
    # --- Non-Defaults ---
    model_name: str
    parameters: Dict[str, float] # Fitted parameters (e.g., A, c1, m1, n1, m2, n2)
    success: bool # Did the optimization algorithm converge?
    message: str # Message from the optimizer
    rss: float # Residual Sum of Squares (conversion scale) on original data
    n_datapoints: int
    n_parameters: int
    # --- Defaults ---
    param_std_err: Optional[Dict[str, float]] = None # Std Err from covariance matrix (if reliable)
    covariance_matrix: Optional[np.ndarray] = None # Covariance matrix from fit
    aic: Optional[float] = None # Akaike Information Criterion
    bic: Optional[float] = None # Bayesian Information Criterion
    r_squared: Optional[float] = None # R-squared value
    initial_ratio_r: Optional[float] = None # (If applicable to fitting method)
    model_definition_args: Optional[Dict[str, Any]] = None # Stored definition args
    used_rate_fallback: bool = False # Flag indicating if rate objective was used as fallback

@dataclass
class ParameterBootstrapResult:
    """
    Structure for bootstrap analysis results focused on PARAMETER uncertainty.
    """
    # --- Non-Defaults ---
    model_name: str
    parameter_distributions: Dict[str, np.ndarray] # Distributions of parameters (e.g., A, c1...)
    parameter_ci: Dict[str, Tuple[float, float]] # Confidence intervals for parameters
    n_iterations: int # Number of successful bootstrap iterations completed
    confidence_level: float # e.g., 95.0 for 95%
    # --- Defaults ---
    median_parameters: Optional[Dict[str, float]] = None # Median parameters from successful replicates
    raw_parameter_list: Optional[List[Dict[str, float]]] = None # Optional list of params per success
    ranked_replicates: Optional[List[Dict[str, Any]]] = None # List of dicts, ranked by RSS on resampled data
    median_stats: Optional[Dict[str, Any]] = None # Stats (RSS, AIC, BIC etc. on ORIGINAL data) for median parameters


# --- Prediction Results ---

@dataclass
class PredictionCurveResult:
    """Structure for prediction results representing a full curve."""
    # --- Non-Defaults ---
    time: np.ndarray
    temperature: np.ndarray # Temperature profile used for prediction
    conversion: np.ndarray # Predicted conversion curve
    # --- Defaults ---
    conversion_ci: Optional[Tuple[np.ndarray, np.ndarray]] = None # (lower, upper) CI band if calculated (e.g., via parameter bootstrap + simulation)
    description: str = "Predicted conversion curve" # Description of the prediction scenario

@dataclass
class BootstrapPredictionResult:
    """
    Structure for bootstrap analysis results focused on the uncertainty
    of a SINGLE PREDICTED VALUE (e.g., shelf-life time).
    """
    # --- Non-Defaults ---
    target_description: str # Description (e.g., "Time to 0.1 alpha at 25 C")
    predicted_value_distribution: np.ndarray # Distribution of the predicted scalar value
    predicted_value_median: float # Median of the distribution
    predicted_value_ci: Tuple[float, float] # (lower_bound, upper_bound) CI
    n_iterations: int # Number of successful bootstrap iterations yielding this prediction
    confidence_level: float # e.g., 95.0 for 95%
    # --- Defaults ---
    unit: str = "seconds" # Unit of the predicted value (e.g., "days", "seconds", "years")


@dataclass
class SingleValuePredictionResult:
    """Structure for a single predicted value based on the original fit."""
    # --- Non-Defaults ---
    target_description: str # Description (e.g., "Time to 0.1 alpha at 25 C")
    predicted_value: float # The single value predicted using original fit parameters
    # --- Defaults ---
    unit: str = "seconds" # Unit of the predicted value


# --- Top-Level Analysis Container ---

@dataclass
class FullAnalysisResult:
    """Top-level container for results from the entire analysis workflow."""
    # --- Non-Defaults ---
    input_datasets: List[KineticDataset]
    # --- Defaults ---
    isoconversional_result: Optional[IsoResult] = None
    fit_result: Optional[FitResult] = None
    parameter_bootstrap_result: Optional[ParameterBootstrapResult] = None # Optional: if parameter CIs were calculated separately
    prediction_from_original_fit: Optional[SingleValuePredictionResult] = None # Prediction using original params
    prediction_bootstrap_result: Optional[BootstrapPredictionResult] = None # Prediction with CI from bootstrap
    prediction_curve_result: Optional[PredictionCurveResult] = None # Optional: if a full curve was predicted

# --- Type Hints (Keep as they are useful) ---

# Type hint for f(alpha) functions
# Takes alpha (float or array) and optional parameter dict, returns float or array
FAlphaCallable = Callable[[Any, Optional[Dict[str, float]]], Any]

# Type hint for ODE system functions used by solve_ivp
# Takes t, y (state vector), T_func (interpolated temp), params_dict, maybe other args
OdeSystemCallable = Callable[..., np.ndarray] # Make more general as args might change