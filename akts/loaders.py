# loaders.py
"""
Data loaders for common thermal analysis instrument formats.

Supports CSV and XLSX files with headers for DSC, TGA, and other thermal analysis data.
"""
import numpy as np
import warnings
from pathlib import Path
from typing import Optional, Dict, Tuple, Union

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    pd = None

from .datatypes import KineticDataset


def load_data_file(
    filepath: Union[str, Path],
    time_col: str = 'Time',
    temperature_col: str = 'Temperature',
    signal_col: Optional[str] = None,
    conversion_col: Optional[str] = 'Conversion',
    mass_col: Optional[str] = None,
    heat_flow_col: Optional[str] = None,
    readout_col: Optional[str] = None,
    readout_type: str = 'increasing',
    readout_initial: Optional[float] = None,
    readout_final: Optional[float] = None,
    file_type: Optional[str] = None,
    heating_rate: Optional[float] = None,
    auto_detect: bool = True,
    **kwargs
) -> KineticDataset:
    """
    Load kinetic data from CSV or XLSX instrument export files.

    Parameters
    ----------
    filepath : str or Path
        Path to the data file (CSV or XLSX)
    time_col : str
        Name of the time column (default: 'Time')
    temperature_col : str
        Name of the temperature column (default: 'Temperature')
    signal_col : str, optional
        Generic signal column name if conversion needs to be computed
    conversion_col : str, optional
        Name of the conversion column if already present (default: 'Conversion')
    mass_col : str, optional
        Name of the mass column for TGA data
    heat_flow_col : str, optional
        Name of the heat flow column for DSC data
    readout_col : str, optional
        Name of experimental readout column (e.g., '%HMW', 'Absorbance', 'Purity')
    readout_type : str
        Type of readout: 'increasing' (default, e.g., %HMW, degradation) or
        'decreasing' (e.g., %Monomer, purity). Determines how conversion is calculated.
    readout_initial, readout_final : float, optional
        Readout values that map to conversion 0 and 1. Defaults to this file's first/last
        values. Set readout_final explicitly when comparing files that stop at different
        extents of reaction, otherwise every file is stretched to end at conversion 1.
    file_type : str, optional
        Force file type: 'csv' or 'xlsx'. If None, auto-detected from extension
    heating_rate : float, optional
        Heating rate in K/s. If None, estimated from data
    auto_detect : bool
        Automatically detect column names if exact names not found (default: True)
    **kwargs
        Additional arguments passed to pandas read functions

    Returns
    -------
    KineticDataset
        Loaded kinetic dataset

    Examples
    --------
    >>> # Load DSC data with conversion already calculated
    >>> ds = load_data_file('dsc_data.csv', time_col='Time (s)',
    ...                           temperature_col='Temp (K)', conversion_col='Alpha')

    >>> # Load TGA data and auto-compute conversion from mass
    >>> ds = load_data_file('tga_data.csv', mass_col='Mass (mg)')

    >>> # Load with auto-detection
    >>> ds = load_data_file('experiment.xlsx', auto_detect=True)

    >>> # Load stability data with %HMW readout
    >>> ds = load_data_file('stability.csv', readout_col='HMW Species (%)',
    ...                           readout_type='increasing')
    """
    if not HAS_PANDAS:
        raise ImportError(
            "pandas is required for data loading. Install with: pip install pandas\n"
            "Or install akts with loader support: pip install akts[loaders]"
        )

    filepath = Path(filepath)

    # Determine file type
    if file_type is None:
        file_type = filepath.suffix.lower().lstrip('.')

    # Read file
    if file_type == 'csv':
        df = pd.read_csv(filepath, **kwargs)
    elif file_type in ['xlsx', 'xls']:
        df = pd.read_excel(filepath, **kwargs)
    else:
        raise ValueError(f"Unsupported file type: {file_type}. Use 'csv' or 'xlsx'")

    # Auto-detect column names if enabled
    if auto_detect:
        time_col = _find_column(df, time_col, ['time', 't', 'time(s)', 'time (s)', 'zeit'])
        temperature_col = _find_column(df, temperature_col,
                                       ['temperature', 'temp', 'T', 'temperature(K)',
                                        'temp (K)', 'temperature (K)', 'temperatur'])

        if conversion_col:
            conversion_col = _find_column(df, conversion_col,
                                         ['conversion', 'alpha', 'α', 'extent', 'x'],
                                         required=False)

        if mass_col:
            mass_col = _find_column(df, mass_col,
                                   ['mass', 'weight', 'mass(mg)', 'mass (mg)',
                                    'weight (mg)', 'masse'],
                                   required=False)

        if heat_flow_col:
            heat_flow_col = _find_column(df, heat_flow_col,
                                         ['heat flow', 'heatflow', 'dsc', 'heat_flow',
                                          'heat flow (mW)', 'DSC (mW/mg)'],
                                         required=False)

        if readout_col:
            readout_col = _find_column(df, readout_col,
                                       ['hmw', 'hmw species', '%hmw', 'hmw (%)',
                                        'absorbance', 'purity', 'monomer', '%monomer'],
                                       required=False)

    # Extract data
    metadata = {'filename': filepath.name}

    # Time
    if time_col not in df.columns:
        raise ValueError(f"Time column '{time_col}' not found. Available: {list(df.columns)}")
    time = df[time_col].values
    metadata['time_units'] = _guess_units(time_col)

    # Temperature
    if temperature_col not in df.columns:
        raise ValueError(f"Temperature column '{temperature_col}' not found. Available: {list(df.columns)}")
    temperature = df[temperature_col].values
    metadata['temperature_units'] = _guess_units(temperature_col)

    # Conversion - multiple strategies
    conversion = None
    readout = None

    # Strategy 1: Conversion already provided
    if conversion_col and conversion_col in df.columns:
        conversion = df[conversion_col].values
        metadata['conversion_source'] = 'direct'

    # Strategy 2: Compute from TGA mass data
    elif mass_col and mass_col in df.columns:
        mass = df[mass_col].values
        conversion = _compute_conversion_from_mass(mass)
        metadata['conversion_source'] = 'tga_mass'
        metadata['mass_units'] = _guess_units(mass_col)

    # Strategy 3: Compute from DSC heat flow (requires integration)
    elif heat_flow_col and heat_flow_col in df.columns:
        heat_flow = df[heat_flow_col].values
        conversion = _compute_conversion_from_heat_flow(time, heat_flow)
        metadata['conversion_source'] = 'dsc_heat_flow'
        metadata['heat_flow_units'] = _guess_units(heat_flow_col)

    # Strategy 4: Experimental readout (e.g., %HMW, purity)
    elif readout_col and readout_col in df.columns:
        readout = df[readout_col].values.astype(float)
        conversion = _compute_conversion_from_readout(readout, readout_type,
                                                      readout_initial, readout_final)
        metadata['conversion_source'] = f'readout_{readout_type}'
        metadata['readout_units'] = _guess_units(readout_col)
        metadata['readout_type'] = readout_type

    # Strategy 5: Generic signal column
    elif signal_col and signal_col in df.columns:
        signal = df[signal_col].values
        conversion = _compute_conversion_from_signal(signal)
        metadata['conversion_source'] = 'signal'

    else:
        raise ValueError(
            "Could not determine conversion. Provide one of: "
            "conversion_col, mass_col, heat_flow_col, readout_col, or signal_col"
        )

    # Estimate heating rate if not provided
    if heating_rate is None:
        heating_rate = _estimate_heating_rate(time, temperature)
        metadata['heating_rate_estimated'] = True
    else:
        metadata['heating_rate_estimated'] = False

    metadata['heating_rate_K_per_s'] = heating_rate

    # Clean data - remove NaNs and ensure monotonic time
    mask = np.isfinite(time) & np.isfinite(temperature) & np.isfinite(conversion)
    time = time[mask]
    temperature = temperature[mask]
    conversion = conversion[mask]
    if readout is not None:
        readout = readout[mask]

    # Ensure time is monotonically increasing
    if len(time) > 1:
        sorted_idx = np.argsort(time)
        time = time[sorted_idx]
        temperature = temperature[sorted_idx]
        conversion = conversion[sorted_idx]
        if readout is not None:
            readout = readout[sorted_idx]

    if readout is not None:
        # Kept so multi-file workflows can renormalize all files onto one scale.
        metadata['readout_raw'] = readout

    # Clip conversion to [0, 1]
    conversion = np.clip(conversion, 0.0, 1.0)

    return KineticDataset(
        time=time,
        temperature=temperature,
        conversion=conversion,
        heating_rate=heating_rate,
        metadata=metadata
    )


def _find_column(df: pd.DataFrame, preferred: str, alternatives: list,
                 required: bool = True) -> Optional[str]:
    """Find column by name, checking alternatives (case-insensitive)."""
    # Check exact match first
    if preferred in df.columns:
        return preferred

    # Check alternatives (case-insensitive)
    df_cols_lower = {col.lower(): col for col in df.columns}

    for alt in alternatives:
        if alt.lower() in df_cols_lower:
            return df_cols_lower[alt.lower()]

    if required:
        raise ValueError(
            f"Column '{preferred}' not found. Tried alternatives: {alternatives}. "
            f"Available columns: {list(df.columns)}"
        )
    return None


def _guess_units(column_name: str) -> str:
    """Guess units from column name."""
    col_lower = column_name.lower()

    # Time units - check longer names first so "(days)" is not read as seconds
    if '(day' in col_lower or '(d)' in col_lower:
        return 'days'
    elif '(week' in col_lower:
        return 'weeks'
    elif '(min' in col_lower:
        return 'min'
    elif '(h' in col_lower:
        return 'h'
    elif '(s' in col_lower:
        return 's'

    # Temperature units
    if 'k)' in col_lower or '(k' in col_lower:
        return 'K'
    elif '°c)' in col_lower or '(°c' in col_lower or 'c)' in col_lower:
        return '°C'

    # Mass units
    if 'mg)' in col_lower or '(mg' in col_lower:
        return 'mg'
    elif 'g)' in col_lower or '(g' in col_lower:
        return 'g'

    # Heat flow units
    if 'mw)' in col_lower or '(mw' in col_lower:
        return 'mW'
    elif 'w)' in col_lower or '(w' in col_lower:
        return 'W'

    return 'unknown'


def _compute_conversion_from_mass(mass: np.ndarray) -> np.ndarray:
    """
    Compute conversion from TGA mass data.

    Conversion = (m_initial - m(t)) / (m_initial - m_final)
    """
    if len(mass) < 2:
        raise ValueError("Need at least 2 mass points to compute conversion")

    # Use first 5% of data for initial mass (average to reduce noise)
    n_initial = max(1, len(mass) // 20)
    m_initial = np.mean(mass[:n_initial])

    # Use last 5% of data for final mass
    n_final = max(1, len(mass) // 20)
    m_final = np.mean(mass[-n_final:])

    # Compute conversion
    if abs(m_initial - m_final) < 1e-9:
        warnings.warn("Initial and final masses are nearly identical - no mass loss detected")
        return np.zeros_like(mass)

    conversion = (m_initial - mass) / (m_initial - m_final)

    return conversion


def _compute_conversion_from_heat_flow(time: np.ndarray, heat_flow: np.ndarray) -> np.ndarray:
    """
    Compute conversion from DSC heat flow data via integration.

    Conversion = ∫(heat_flow) dt / ∫_total(heat_flow) dt
    """
    if len(heat_flow) < 2:
        raise ValueError("Need at least 2 heat flow points to compute conversion")

    # Integrate heat flow using cumulative trapezoidal integration
    heat_cumulative = np.zeros_like(heat_flow)
    for i in range(1, len(heat_flow)):
        dt = time[i] - time[i-1]
        heat_cumulative[i] = heat_cumulative[i-1] + 0.5 * (heat_flow[i] + heat_flow[i-1]) * dt

    # Normalize to [0, 1]
    total_heat = heat_cumulative[-1]

    if abs(total_heat) < 1e-9:
        warnings.warn("Total integrated heat is near zero - no reaction detected")
        return np.zeros_like(heat_flow)

    conversion = heat_cumulative / total_heat

    # Handle negative heat flows (exothermic vs endothermic)
    if total_heat < 0:
        conversion = 1.0 - conversion

    return conversion


def _compute_conversion_from_signal(signal: np.ndarray) -> np.ndarray:
    """
    Compute conversion from generic signal (normalized to [0, 1]).

    Uses min-max normalization.
    """
    signal_min = np.min(signal)
    signal_max = np.max(signal)

    if abs(signal_max - signal_min) < 1e-9:
        warnings.warn("Signal has no variation - returning zero conversion")
        return np.zeros_like(signal)

    conversion = (signal - signal_min) / (signal_max - signal_min)

    return conversion


def _compute_conversion_from_readout(readout: np.ndarray, readout_type: str,
                                     readout_initial: Optional[float] = None,
                                     readout_final: Optional[float] = None) -> np.ndarray:
    """
    Compute conversion from experimental readout.

    For 'increasing' readouts (e.g., %HMW, degradation products):
        conversion = (readout - readout_initial) / (readout_final - readout_initial)

    For 'decreasing' readouts (e.g., %Monomer, purity):
        conversion = (readout_initial - readout) / (readout_initial - readout_final)

    Parameters
    ----------
    readout : np.ndarray
        Experimental readout values
    readout_type : str
        'increasing' or 'decreasing'

    Returns
    -------
    np.ndarray
        Conversion values normalized to [0, 1]
    """
    if len(readout) < 2:
        raise ValueError("Need at least 2 readout points to compute conversion")

    n_edge = max(1, len(readout) // 20)
    if readout_initial is None:
        readout_initial = np.mean(readout[:n_edge])
    if readout_final is None:
        readout_final = np.mean(readout[-n_edge:])

    if abs(readout_final - readout_initial) < 1e-9:
        warnings.warn("Initial and final readouts are nearly identical - no change detected")
        return np.zeros_like(readout)

    if readout_type == 'increasing':
        # For increasing signals (e.g., %HMW increases as aggregation progresses)
        conversion = (readout - readout_initial) / (readout_final - readout_initial)

    elif readout_type == 'decreasing':
        # For decreasing signals (e.g., %Monomer decreases as aggregation progresses)
        conversion = (readout_initial - readout) / (readout_initial - readout_final)

    else:
        raise ValueError(f"Unknown readout_type: {readout_type}. Use 'increasing' or 'decreasing'")

    return conversion


def _estimate_heating_rate(time: np.ndarray, temperature: np.ndarray) -> float:
    """
    Estimate heating rate from temperature vs time data.

    Returns heating rate in K/s.
    """
    if len(time) < 2:
        return 0.0

    # Use linear regression for robustness
    from scipy.stats import linregress

    # Use middle 80% of data to avoid edge effects
    n = len(time)
    start_idx = n // 10
    end_idx = 9 * n // 10

    if end_idx - start_idx < 2:
        start_idx = 0
        end_idx = n

    result = linregress(time[start_idx:end_idx], temperature[start_idx:end_idx])

    heating_rate = result.slope  # K/s

    if heating_rate < 0:
        warnings.warn(f"Negative heating rate detected ({heating_rate:.3e} K/s). Using absolute value.")
        heating_rate = abs(heating_rate)

    return heating_rate


# Convenience functions for specific instrument types
def load_dsc_file(filepath: Union[str, Path], **kwargs) -> KineticDataset:
    """
    Load DSC (Differential Scanning Calorimetry) data file.

    Automatically looks for heat flow columns and computes conversion.

    Parameters
    ----------
    filepath : str or Path
        Path to DSC data file
    **kwargs
        Additional arguments passed to load_data_file

    Returns
    -------
    KineticDataset
    """
    # Common DSC column names
    kwargs.setdefault('heat_flow_col', 'Heat Flow')
    kwargs.setdefault('auto_detect', True)

    return load_data_file(filepath, **kwargs)


def load_tga_file(filepath: Union[str, Path], **kwargs) -> KineticDataset:
    """
    Load TGA (Thermogravimetric Analysis) data file.

    Automatically looks for mass columns and computes conversion.

    Parameters
    ----------
    filepath : str or Path
        Path to TGA data file
    **kwargs
        Additional arguments passed to load_data_file

    Returns
    -------
    KineticDataset
    """
    # Common TGA column names
    kwargs.setdefault('mass_col', 'Mass')
    kwargs.setdefault('auto_detect', True)

    return load_data_file(filepath, **kwargs)


def load_isothermal_file(filepath: Union[str, Path], **kwargs) -> KineticDataset:
    """
    Load isothermal experiment data file.

    For isothermal experiments, heating_rate should be 0 or very small.

    Parameters
    ----------
    filepath : str or Path
        Path to isothermal data file
    **kwargs
        Additional arguments passed to load_data_file

    Returns
    -------
    KineticDataset
    """
    kwargs.setdefault('auto_detect', True)
    dataset = load_data_file(filepath, **kwargs)

    # Override heating rate for isothermal
    if dataset.heating_rate and abs(dataset.heating_rate) > 0.01:
        warnings.warn(
            f"Data appears non-isothermal (heating rate = {dataset.heating_rate:.3e} K/s). "
            f"Consider using load_data_file instead."
        )

    return dataset
