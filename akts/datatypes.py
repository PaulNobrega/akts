# datatypes.py
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Callable

@dataclass
class KineticDataset:
    """Structure to hold experimental kinetic data."""
    time: np.ndarray
    temperature: np.ndarray
    conversion: np.ndarray
    heating_rate: Optional[float] = None
    metadata: Dict = field(default_factory=dict)

@dataclass
class IsoResult:
    """Structure for isoconversional analysis results."""
    method: str
    alpha: np.ndarray
    Ea: np.ndarray
    Ea_std_err: Optional[np.ndarray] = None
    regression_stats: Optional[List[Dict]] = None

@dataclass
class FitResult:
    """Structure for model fitting results."""
    # --- Fields without defaults first ---
    model_name: str
    parameters: Dict[str, float] # Parameters in A scale
    success: bool
    message: str
    rss: float # Unweighted Conversion RSS on ORIGINAL data
    n_datapoints: int
    n_parameters: int
    # --- Fields with defaults last ---
    param_std_err: Optional[Dict[str, float]] = None # Std Err for A scale params
    aic: Optional[float] = None
    bic: Optional[float] = None
    r_squared: Optional[float] = None
    initial_ratio_r: Optional[float] = None
    model_definition_args: Optional[Dict] = None # Stored definition args
    # **** ADDED used_rate_fallback ****
    used_rate_fallback: bool = False # Flag indicating if rate objective was used as fallback

@dataclass
class BootstrapResult:
    """Structure for bootstrap analysis results."""
    model_name: str
    parameter_distributions: Dict[str, np.ndarray] # Distributions in A-scale
    parameter_ci: Dict[str, Tuple[float, float]] # Confidence intervals in A-scale
    n_iterations: int # Number of successful iterations
    confidence_level: float
    raw_parameter_list: Optional[List[Dict]] = None # Optional list of A-scale params per success
    # **** ADDED FIELD BACK ****
    ranked_replicates: Optional[List[Dict]] = None # List of dicts, ranked by RSS on resampled data
    # **** Keep median fields ****
    median_parameters: Optional[Dict[str, float]] = None # Median parameters (A-scale) from successful replicates
    median_stats: Optional[Dict] = None # Stats (RSS, AIC, BIC etc. on ORIGINAL data) for median parameters

@dataclass
class PredictionResult:
    """Structure for prediction results."""
    time: np.ndarray
    temperature: np.ndarray
    conversion: np.ndarray
    conversion_ci: Optional[Tuple[np.ndarray, np.ndarray]] = None # (lower, upper) CI band if calculated

# Type hint for f(alpha) functions
FAlphaCallable = Callable[[float, Optional[Dict]], float]

# Type hint for ODE system functions used by solve_ivp
# Takes t, y (state vector), T_func (interpolated temp), params_dict
OdeSystemCallable = Callable[[float, np.ndarray, Callable, Dict], np.ndarray]