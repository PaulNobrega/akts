"""
JSON utilities for AKTS library - serialization and deserialization of kinetic data and results.
"""
import numpy as np
import json
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from .datatypes import KineticDataset, FitResult, BootstrapResult, PredictionResult


def convert_numpy_to_python(obj: Any) -> Any:
    """
    Recursively convert numpy types and dataclasses to native Python types for JSON serialization.

    Parameters
    ----------
    obj : Any
        Object to convert (can be numpy array, dict, list, dataclass, etc.)

    Returns
    -------
    Any
        Object with numpy types converted to Python types
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_to_python(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_to_python(item) for item in obj]
    elif isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, (BootstrapResult, FitResult, PredictionResult, KineticDataset)):
        # Convert dataclasses to dict recursively
        import dataclasses
        return convert_numpy_to_python(dataclasses.asdict(obj))
    else:
        return obj


def parse_json_data(json_data: Dict) -> KineticDataset:
    """
    Convert JSON dictionary to KineticDataset object.

    Parameters
    ----------
    json_data : Dict
        Dictionary with keys: 'time', 'temperature', 'conversion'
        Optional: 'heating_rate', 'metadata'

    Returns
    -------
    KineticDataset
        Dataset object ready for kinetic analysis

    Example
    -------
    >>> json_data = {
    ...     "time": [0, 1, 2, 3],
    ...     "temperature": [313, 313, 313, 313],
    ...     "conversion": [0.0, 0.1, 0.2, 0.3],
    ...     "metadata": {"filename": "test_data.json"}
    ... }
    >>> dataset = parse_json_data(json_data)
    """
    if not all(key in json_data for key in ['time', 'temperature', 'conversion']):
        raise ValueError("JSON data must contain 'time', 'temperature', and 'conversion' fields")

    time = np.array(json_data['time'], dtype=float)
    temperature = np.array(json_data['temperature'], dtype=float)
    conversion = np.array(json_data['conversion'], dtype=float)

    heating_rate = json_data.get('heating_rate', None)
    metadata = json_data.get('metadata', {})

    return KineticDataset(
        time=time,
        temperature=temperature,
        conversion=conversion,
        heating_rate=heating_rate,
        metadata=metadata
    )


def dataset_to_json(dataset: KineticDataset) -> Dict:
    """
    Convert KineticDataset to JSON-serializable dictionary.

    Parameters
    ----------
    dataset : KineticDataset
        Dataset to convert

    Returns
    -------
    Dict
        JSON-serializable dictionary
    """
    return {
        'time': dataset.time.tolist(),
        'temperature': dataset.temperature.tolist(),
        'conversion': dataset.conversion.tolist(),
        'heating_rate': dataset.heating_rate,
        'metadata': dataset.metadata
    }


def _optional_float(value: Optional[float]) -> Optional[float]:
    """Converts a possibly-None numeric field to float, passing None through.

    FitResult's r_squared/aic/bic default to None (e.g. every early-return
    failure path in fit_kinetic_model() leaves them unset), so a bare
    float(...) call on these fields raises TypeError for unsuccessful fits.
    """
    return None if value is None else float(value)


def fit_result_to_json(fit_result: FitResult) -> Dict[str, Any]:
    """
    Convert FitResult to JSON-serializable dictionary.

    Parameters
    ----------
    fit_result : FitResult
        Fit result to convert

    Returns
    -------
    Dict
        JSON-serializable dictionary with fit parameters and statistics
    """
    result: Dict[str, Any] = {
        'success': fit_result.success,
        'message': fit_result.message,
        'parameters': convert_numpy_to_python(fit_result.parameters),
        'param_std_err': convert_numpy_to_python(fit_result.param_std_err) if fit_result.param_std_err else None,
        'rss': _optional_float(fit_result.rss),
        'r_squared': _optional_float(fit_result.r_squared),
        'aic': _optional_float(fit_result.aic),
        'bic': _optional_float(fit_result.bic),
        'n_datapoints': int(fit_result.n_datapoints),
        'n_parameters': int(fit_result.n_parameters),
        'model_name': fit_result.model_name,
    }

    # Add optional fields if present
    if hasattr(fit_result, 'time_data') and fit_result.time_data is not None:
        result['time_data'] = convert_numpy_to_python(fit_result.time_data)
    if hasattr(fit_result, 'conversion_data') and fit_result.conversion_data is not None:
        result['conversion_data'] = convert_numpy_to_python(fit_result.conversion_data)
    if hasattr(fit_result, 'conversion_simulated') and fit_result.conversion_simulated is not None:
        result['conversion_simulated'] = convert_numpy_to_python(fit_result.conversion_simulated)

    return result


def prediction_to_json(prediction: PredictionResult) -> Dict:
    """
    Convert PredictionResult to JSON-serializable dictionary.

    Parameters
    ----------
    prediction : PredictionResult
        Prediction result to convert

    Returns
    -------
    Dict
        JSON-serializable dictionary
    """
    result: Dict[str, Any] = {
        'time': prediction.time.tolist(),
        'conversion': prediction.conversion.tolist(),
    }

    if prediction.temperature is not None:
        result['temperature'] = np.asarray(prediction.temperature).tolist()

    if prediction.conversion_ci is not None:
        result['conversion_lower'] = prediction.conversion_ci[0].tolist()
        result['conversion_upper'] = prediction.conversion_ci[1].tolist()

    return result


def bootstrap_to_json(bootstrap: BootstrapResult) -> Dict:
    """
    Convert BootstrapResult to JSON-serializable dictionary.

    Parameters
    ----------
    bootstrap : BootstrapResult
        Bootstrap result to convert

    Returns
    -------
    Dict
        JSON-serializable dictionary
    """
    return {
        'parameter_distributions': convert_numpy_to_python(bootstrap.parameter_distributions),
        'parameter_ci': convert_numpy_to_python(bootstrap.parameter_ci),
        'n_iterations': int(bootstrap.n_iterations),
        'confidence_level': float(bootstrap.confidence_level),
        'bootstrap_method': bootstrap.bootstrap_method,
        'median_parameters': convert_numpy_to_python(bootstrap.median_parameters)
    }


def serialize_results_to_json(
    top_models: List[Dict],
    selected_model: Dict,
    predictions: Dict = None,
    report_path: Union[str, Path] = None,
    bootstrap_results: Dict = None,
    summary: Dict = None,
    regulatory: Dict = None
) -> str:
    """
    Serialize complete results to JSON string.

    Parameters
    ----------
    top_models : List[Dict]
        List of top-ranked models with statistics
    selected_model : Dict
        Selected model (single or list if no convergence)
    predictions : Dict, optional
        Prediction results
    report_path : str or Path, optional
        Path to generated HTML report
    bootstrap_results : Dict, optional
        Bootstrap results for each model
    summary : Dict, optional
        Summary statistics

    Returns
    -------
    str
        JSON string
    """
    result = {
        'top_models': convert_numpy_to_python(top_models),
        'selected_model': convert_numpy_to_python(selected_model),
    }

    if predictions is not None:
        result['predictions'] = convert_numpy_to_python(predictions)

    if report_path is not None:
        result['report_path'] = str(report_path)

    if bootstrap_results is not None:
        result['bootstrap_results'] = convert_numpy_to_python(bootstrap_results)

    if summary is not None:
        result['summary'] = convert_numpy_to_python(summary)

    if regulatory is not None:
        result['regulatory'] = convert_numpy_to_python(regulatory)

    return json.dumps(result, indent=2)


def deserialize_results_from_json(json_str: str) -> Dict:
    """
    Deserialize results from JSON string to dictionary.

    Parameters
    ----------
    json_str : str
        JSON string to deserialize

    Returns
    -------
    Dict
        Dictionary with results
    """
    return json.loads(json_str)
