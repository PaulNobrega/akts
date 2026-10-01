"""
Ranking of fitted models by information criteria / combined score.
"""
import numpy as np
import warnings
from typing import List, Dict, Optional

from .datatypes import FitResult
from .utils import calculate_akaike_weights, calculate_adjusted_r_squared


def rank_models(
    fit_results: List[FitResult],
    min_r_squared: float = 0.70,
    apply_filters: bool = True
) -> List[Dict]:
    """
    Ranks fitted models using filter-then-rank with Akaike weights.

    Stage 1 (Filtering): Apply hard requirements for R² and physical plausibility
    Stage 2 (Ranking): Rank survivors by Akaike weights

    Parameters
    ----------
    fit_results : List[FitResult]
        List of fit results to rank
    min_r_squared : float, default=0.70
        Minimum R² threshold for model selection. Models below this are filtered out
        unless no models meet the requirement.
    apply_filters : bool, default=True
        Apply R² and plausibility filters before ranking

    Returns
    -------
    List[Dict]
        Ranked models with stats, Akaike weights, and filter warnings.
        Each dict contains:
        - 'model_name': Model identifier
        - 'rank': Ranking position (1 = best)
        - 'stats': Dictionary with AIC, BIC, R², Akaike weight, plausibility
        - 'parameters': Fitted parameter values
        - 'filter_warning': Present on rank 1 if plausibility issues exist

    Notes
    -----
    Ranking Method:
    - Uses ONLY Akaike weights (probability each model is best)
    - Balances fit quality and model complexity automatically
    - Simpler models preferred when fit quality is similar

    Filter Logic:
    1. If any model has R² ≥ min_r_squared AND is_physically_plausible:
       Keep only models with R² ≥ min_r_squared AND is_physically_plausible
    2. Otherwise:
       Keep models with R² ≥ min_r_squared (any plausibility)
       Add warning that top model has questionable plausibility

    Akaike Weight Interpretation:
    - Weight = 100% for one model → vastly superior (Δ_AIC > ~20)
    - This is CORRECT behavior, not a bug
    - Even distribution (e.g., 40%, 35%, 25%) → model uncertainty
    - Sum of all weights = 100%

    Physical Plausibility Criteria:
    - Pre-exponential factor: A < 10²⁰ s⁻¹ (transition state theory limit)
    - Activation energy: 5 < Ea < 1000 kJ/mol
    - Shape parameters: m, n ≥ 0 (non-negative)

    Examples
    --------
    >>> from akts import rank_models
    >>> ranked = rank_models(fit_results, min_r_squared=0.70)
    >>> top_model = ranked[0]
    >>> print(f"Best model: {top_model['model_name']}")
    >>> print(f"Akaike weight: {top_model['stats']['akaike_weight']:.1%}")
    >>> print(f"Plausible: {top_model['stats']['is_physically_plausible']}")
    """
    if not fit_results:
        return []

    # --- Prepare data for ranking ---
    valid_fits_data = []
    stat_values = {'rss': [], 'r_squared': [], 'aic': [], 'bic': [], 'n_params': []}

    for res in fit_results:
        rss = getattr(res, 'rss', np.inf)
        r2 = getattr(res, 'r_squared', -np.inf)
        aic = getattr(res, 'aic', np.inf)
        bic = getattr(res, 'bic', np.inf)
        n_params = getattr(res, 'n_parameters', np.inf)
        n_points = getattr(res, 'n_datapoints', 0)

        if not all(np.isfinite([rss, aic, bic, n_params])) or n_points <= n_params:
            warnings.warn(f"Model '{res.model_name}' has invalid stats for ranking. Skipping.")
            continue

        r2 = r2 if np.isfinite(r2) else -np.inf
        # FitResult.durbin_watson defaults to None (not NaN) when unset -- getattr's
        # fallback only fires if the attribute is missing entirely, so this can still
        # legitimately be None here.
        durbin_watson = getattr(res, 'durbin_watson', None)
        durbin_watson = float(durbin_watson) if durbin_watson is not None and np.isfinite(durbin_watson) else float('nan')
        is_plausible = getattr(res, 'is_physically_plausible', None)
        plausibility_issues = getattr(res, 'plausibility_issues', None)
        valid_fits_data.append({
            'model_name': res.model_name,
            'parameters': res.parameters,
            'stats': {
                # Cast every numeric stat to plain float/int here (rather than
                # relying on downstream JSON conversion) so consumers that use
                # this dict directly -- reporting.py, rank_models() callers --
                # also see native Python types, not numpy scalars.
                'rss': float(rss), 'r_squared': float(r2), 'aic': float(aic), 'bic': float(bic),
                'n_params': int(n_params), 'n_points': int(n_points),
                'r_squared_adj': float(calculate_adjusted_r_squared(r2, n_params, n_points)),
                'rmse': float(np.sqrt(rss / n_points)) if n_points > 0 else float('nan'),
                'durbin_watson': durbin_watson,
                'is_physically_plausible': is_plausible,
                'plausibility_issues': plausibility_issues,
            },
            'n_params': int(n_params)
        })
        stat_values['rss'].append(rss)
        stat_values['r_squared'].append(r2)
        stat_values['aic'].append(aic)
        stat_values['bic'].append(bic)
        stat_values['n_params'].append(n_params)

    if not valid_fits_data:
        print("No models with valid stats found for ranking.")
        return []

    # --- Stage 1: Apply Filters (if enabled) ---
    filter_warning = None
    if apply_filters:
        # Check if any model meets both criteria
        good_models = [
            item for item in valid_fits_data
            if item['stats']['r_squared'] >= min_r_squared
            and item['stats']['is_physically_plausible'] is not False
        ]

        if good_models:
            # Keep only models meeting both criteria
            valid_fits_data = good_models
            warnings.warn(
                f"Applied filters: R² ≥ {min_r_squared:.2f} AND physically plausible. "
                f"Kept {len(valid_fits_data)} models out of {len(stat_values['aic'])} total."
            )
        else:
            # No models meet both criteria - try relaxing plausibility
            acceptable_models = [
                item for item in valid_fits_data
                if item['stats']['r_squared'] >= min_r_squared
            ]

            if acceptable_models:
                # Keep models with good R² even if implausible
                valid_fits_data = acceptable_models
                filter_warning = {
                    'type': 'no_plausible_models',
                    'message': (
                        f"⚠ WARNING: No physically plausible models achieved R² ≥ {min_r_squared:.2f}. "
                        f"Selected model has questionable energetic plausibility. "
                        f"Physical plausibility criteria: A < 1e20 s⁻¹, 5 < Ea < 1000 kJ/mol. "
                        f"Use predictions with caution."
                    ),
                    'min_r_squared': min_r_squared,
                    'n_models_before': len(stat_values['aic']),
                    'n_models_after': len(valid_fits_data)
                }
                warnings.warn(filter_warning['message'])
            else:
                # No models meet even R² requirement - keep all and warn
                filter_warning = {
                    'type': 'no_good_models',
                    'message': (
                        f"⚠ CRITICAL: No models achieved R² ≥ {min_r_squared:.2f}. "
                        f"All {len(valid_fits_data)} models retained, but fit quality is poor. "
                        f"Consider collecting more data or adjusting min_r_squared."
                    ),
                    'min_r_squared': min_r_squared,
                    'n_models': len(valid_fits_data)
                }
                warnings.warn(filter_warning['message'])

    # --- Stage 2: Rank by Akaike Weights ---
    # Akaike weight = probability each model is best in candidate set
    # Higher weight is better, so negate for ascending sort
    # NOTE: Akaike weights already include complexity penalty via AIC (AIC = -2*ln(L) + 2*k)
    # No additional simplicity penalty needed - that would double-penalize complexity!
    temp_akaike_weights = calculate_akaike_weights([item['stats']['aic'] for item in valid_fits_data])
    for item, w in zip(valid_fits_data, temp_akaike_weights):
        item['score'] = -w  # Negate so higher weight = lower score = better rank
        item['simplicity_penalty'] = 0.0  # No additional penalty with Akaike-only

    # Note: Physical plausibility is handled as a FILTER, not a penalty
    # Models with implausible parameters are removed before ranking (unless no plausible options exist)

    # --- Sort by Score (ascending) ---
    valid_fits_data.sort(key=lambda x: x['score'])
    for rank, item in enumerate(valid_fits_data):
        item['rank'] = rank + 1

    # Akaike weights: probability each model is the best in this candidate set,
    # given AICc. Computed over the same finite-AICc models used for scoring/ranking
    # above (ties in 'score' do not affect this -- it is a separate, additive stat).
    akaike_weights = calculate_akaike_weights([item['stats']['aic'] for item in valid_fits_data])
    for item, w in zip(valid_fits_data, akaike_weights):
        item['stats']['akaike_weight'] = w

    # Add filter warning to first result if present
    if filter_warning is not None and valid_fits_data:
        valid_fits_data[0]['filter_warning'] = filter_warning

    return valid_fits_data
