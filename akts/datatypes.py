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
    relative_humidity: Optional[np.ndarray] = None  # RH in range [0, 1] for humidity-dependent kinetics
    metadata: Dict = field(default_factory=dict)

@dataclass
class IsoResult:
    """Structure for isoconversional analysis results."""
    method: str
    alpha: np.ndarray
    Ea: np.ndarray
    Ea_std_err: Optional[np.ndarray] = None
    regression_stats: Optional[List[Dict]] = None
    # ln[A*f(alpha)](alpha) from the Friedman differential-method regression
    # intercept. Only meaningful for method="Friedman" -- KAS/OFW are integral
    # methods whose intercepts don't have this interpretation, so they leave
    # this None. Combined with Ea, this is enough to predict conversion without
    # assuming a reaction model (see core.predict_conversion_model_free).
    ln_A_f_alpha: Optional[np.ndarray] = None

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
    durbin_watson: Optional[float] = None # Durbin-Watson statistic for residual autocorrelation
    is_physically_plausible: Optional[bool] = None # Parameters within plausible ranges
    plausibility_issues: Optional[List[str]] = None # Descriptions of parameter issues
    initial_ratio_r: Optional[float] = None
    model_definition_args: Optional[Dict] = None
    used_rate_fallback: bool = False  # True if rate-based objective was used as fallback

@dataclass
class BootstrapResult:
    """Structure for bootstrap analysis results."""
    model_name: str
    parameter_distributions: Dict[str, np.ndarray] # Distributions in A-scale
    parameter_ci: Dict[str, Tuple[float, float]] # Confidence intervals in A-scale
    n_iterations: int  # Number of successful iterations
    confidence_level: float
    raw_parameter_list: Optional[List[Dict]] = None  # A-scale parameters per successful replicate
    ranked_replicates: Optional[List[Dict]] = None  # Replicates ranked by RSS on resampled data
    median_parameters: Optional[Dict[str, float]] = None  # Median parameters (A-scale)
    median_stats: Optional[Dict] = None  # Statistics for median parameters on original data

    def to_dataframe(self):
        """
        Export bootstrap parameter distributions to pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame where each column is a parameter and each row is a bootstrap replicate.
            Useful for statistical analysis of parameter uncertainty.

        Examples
        --------
        >>> df = bootstrap_result.to_dataframe()
        >>> df.describe()  # Summary statistics
        >>> df.plot(kind='hist', alpha=0.5)  # Histograms
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas is required for to_dataframe(). "
                "Install with: pip install pandas"
            )

        return pd.DataFrame(self.parameter_distributions)

    def summary_dataframe(self):
        """
        Export bootstrap confidence interval summary to pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns:
            - parameter: parameter name
            - median: median value from bootstrap
            - ci_lower: lower CI bound
            - ci_upper: upper CI bound
            - confidence_level: CI confidence level

        Examples
        --------
        >>> summary = bootstrap_result.summary_dataframe()
        >>> summary.to_csv('bootstrap_ci_summary.csv', index=False)
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas is required for summary_dataframe(). "
                "Install with: pip install pandas"
            )

        rows = []
        for param_name, (ci_lower, ci_upper) in self.parameter_ci.items():
            # Get median value from distribution
            median_val = np.median(self.parameter_distributions[param_name])

            rows.append({
                'parameter': param_name,
                'median': median_val,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'confidence_level': self.confidence_level
            })

        return pd.DataFrame(rows)

@dataclass
class PredictionResult:
    """Structure for prediction results."""
    time: np.ndarray
    temperature: np.ndarray
    conversion: np.ndarray
    conversion_ci: Optional[Tuple[np.ndarray, np.ndarray]] = None # (lower, upper) CI band if calculated

    def to_dataframe(self):
        """
        Export prediction results to pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns:
            - time: time points
            - temperature: temperature values
            - conversion: predicted conversion
            - conversion_ci_lower: lower CI bound (if available)
            - conversion_ci_upper: upper CI bound (if available)

        Examples
        --------
        >>> df = prediction_result.to_dataframe()
        >>> df.to_csv('prediction.csv', index=False)
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas is required for to_dataframe(). "
                "Install with: pip install pandas"
            )

        data = {
            'time': self.time,
            'temperature': self.temperature,
            'conversion': self.conversion
        }

        if self.conversion_ci is not None:
            data['conversion_ci_lower'] = self.conversion_ci[0]
            data['conversion_ci_upper'] = self.conversion_ci[1]

        return pd.DataFrame(data)

    def to_csv(self, path: str, **kwargs):
        """
        Export prediction results to CSV file.

        Parameters
        ----------
        path : str
            Output CSV file path
        **kwargs
            Additional arguments passed to pandas.DataFrame.to_csv()
            Common options: index=False (default), sep=',', float_format='%.6f'

        Examples
        --------
        >>> prediction_result.to_csv('prediction.csv', index=False)
        >>> prediction_result.to_csv('data.tsv', sep='\\t', float_format='%.4f')
        """
        df = self.to_dataframe()
        kwargs.setdefault('index', False)  # Default to no index column
        df.to_csv(path, **kwargs)

# Type hint for f(alpha) functions
FAlphaCallable = Callable[[float, Optional[Dict]], float]

# Type hint for ODE system functions used by solve_ivp
# Takes t, y (state vector), T_func (interpolated temp), params_dict
OdeSystemCallable = Callable[[float, np.ndarray, Callable, Dict], np.ndarray]