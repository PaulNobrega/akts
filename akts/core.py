"""
Backward-compatible facade for the kinetics engine.

The implementation is split across focused modules:

- :mod:`akts.simulation` -- ODE/closed-form simulation, logA parameter encoding
- :mod:`akts.fitting`    -- objective functions, fit_kinetic_model, discover_kinetic_models
- :mod:`akts.bootstrap`  -- residual-resampling bootstrap (run_bootstrap, ...)
- :mod:`akts.prediction` -- predict_conversion, predict_conversion_model_free
- :mod:`akts.ranking`    -- rank_models

Everything that used to be importable from ``akts.core`` is re-exported here, so
existing ``from akts.core import ...`` statements keep working.
"""
from .models import get_model_info, get_log_param_names, F_ALPHA_MODELS  # noqa: F401
from .isoconversional import run_friedman, run_kas, run_ofw  # noqa: F401

from .simulation import (  # noqa: F401
    _split_logA_name, _to_logA_name, _prepare_full_params_for_ode,
    ALPHA_SEED, _initial_state, EA_SCALE, _ArrheniusReparam,
    _RhsBudgetExceeded, _with_eval_budget,
    _simulate_single_dataset_closed_form, _simulate_single_dataset,
    simulate_kinetics,
)
from .fitting import (  # noqa: F401
    _objective_function, _residual_vector_function, _objective_function_rate,
    _calculate_conversion_stats, fit_kinetic_model, discover_kinetic_models,
)
from .bootstrap import (  # noqa: F401
    _initialize_bootstrap_worker, _execute_bootstrap_worker, _get_bootstrap_max_workers,
    _fit_empirical_on_resampled_data, _fit_on_resampled_data,
    calculate_stats_for_replicate, rank_replicates,
    run_bootstrap, run_bootstrap_empirical,
)
from .prediction import predict_conversion_model_free, predict_conversion  # noqa: F401
from .ranking import rank_models  # noqa: F401
