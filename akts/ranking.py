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
    score_weights: Optional[Dict[str, float]] = None,
    ranking_method: str = 'combined'
) -> List[Dict]:
    """
    Ranks a list of FitResult objects based on various ranking methods.

    Parameters
    ----------
    fit_results : List[FitResult]
        List of fit results to rank
    score_weights : Dict[str, float], optional
        Custom weights for combined scoring (only used when ranking_method='combined')
        Default: {'bic': 0.4, 'r_squared': 0.4, 'rss': 0.1, 'n_params': 0.1}
    ranking_method : str, default='combined'
        Ranking approach:
        - 'combined': Weighted combination of BIC, R², RSS, n_params (default)
        - 'bic': Rank by BIC alone (lower is better)
        - 'aic': Rank by AICc alone (lower is better)
        - 'akaike_weight': Rank by Akaike weight (higher is better)
        - 'r_squared': Rank by R² alone (higher is better)

    Returns
    -------
    List[Dict]
        Ranked models with stats, scores, and Akaike weights

    Notes
    -----
    - BIC/AIC ranking methods provide direct interpretation: ΔBIC > 10 is "very strong"
      evidence against the higher-BIC model
    - Akaike weights give the probability each model is the best in the candidate set
    - Combined scoring (default) balances multiple criteria but is less interpretable
    - Simplicity penalty (for fitted shape parameters) is applied to all methods
    """
    if not fit_results:
        return []

    # Validate ranking method
    valid_methods = ['combined', 'bic', 'aic', 'akaike_weight', 'r_squared']
    if ranking_method not in valid_methods:
        raise ValueError(f"ranking_method must be one of {valid_methods}, got '{ranking_method}'")

    # --- Define Default Weights (only used for 'combined') ---
    default_weights = {'bic': 0.4, 'r_squared': 0.4, 'rss': 0.1, 'n_params': 0.1}
    weights = score_weights if score_weights and np.isclose(sum(score_weights.values()), 1.0) else default_weights

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

    # --- Calculate Score for each model based on ranking method ---
    if ranking_method == 'combined':
        # Original combined scoring with normalization
        ranges = {key: (np.min(stat_values[key]), np.ptp(stat_values[key])) for key in ['rss', 'aic', 'bic', 'n_params']}
        finite_r2 = [r for r in stat_values['r_squared'] if np.isfinite(r)]
        ranges['r_squared'] = (np.min(finite_r2), np.max(finite_r2), np.ptp(finite_r2)) if finite_r2 else (0, 0, 0)

        for item in valid_fits_data:
            stats = item['stats']
            score = 0.0
            for key, weight in weights.items():
                if key == 'r_squared':
                    min_r2_norm, max_r2_norm, range_r2_norm = ranges['r_squared']
                    val = stats.get('r_squared', -np.inf)
                    if not np.isfinite(val):
                        norm_val = 1.0  # Penalize invalid R2 maximally
                    elif range_r2_norm > 1e-9:
                        norm_val = (max_r2_norm - val) / range_r2_norm  # Higher R2 -> lower score component
                    else:
                        norm_val = 0.0
                    score += weight * norm_val
                elif key in ['rss', 'aic', 'bic', 'n_params']:
                    min_val, range_width = ranges[key]
                    val = stats.get(key, np.inf)
                    if not np.isfinite(val):
                        norm_val = 1.0
                    elif range_width > 1e-9:
                        norm_val = (val - min_val) / range_width
                    else:
                        norm_val = 0.0
                    score += weight * norm_val
            item['score'] = score

    elif ranking_method == 'bic':
        # Rank by BIC alone (lower is better)
        for item in valid_fits_data:
            item['score'] = item['stats']['bic']

    elif ranking_method == 'aic':
        # Rank by AICc alone (lower is better)
        for item in valid_fits_data:
            item['score'] = item['stats']['aic']

    elif ranking_method == 'akaike_weight':
        # Rank by Akaike weight (higher is better, so negate for ascending sort)
        # Compute Akaike weights first
        temp_akaike_weights = calculate_akaike_weights([item['stats']['aic'] for item in valid_fits_data])
        for item, w in zip(valid_fits_data, temp_akaike_weights):
            item['score'] = -w  # Negate so higher weight = lower score = better rank

    elif ranking_method == 'r_squared':
        # Rank by R² alone (higher is better, so negate for ascending sort)
        for item in valid_fits_data:
            r2 = item['stats']['r_squared']
            item['score'] = -r2 if np.isfinite(r2) else np.inf  # Negate for ascending sort

    # --- Apply Simplicity Penalty for Fitted Shape Parameters ---
    # Prefer lower values of fitted parameters (n, m, p) when fit quality is similar
    # Penalty is small (~0.01 per unit) so it only affects ranking when scores are close
    for item in valid_fits_data:
        params = item['parameters']
        simplicity_penalty = 0.0

        # Fn model: prefer lower n (e.g., n=1 over n=3)
        if 'n' in params and item['model_name'] == 'single_step':
            # Check if this is a fitted n (not fixed F1/F2/F3)
            # Penalty scales with distance from n=1 (most common reaction order)
            n_val = params['n']
            if n_val > 1.0:
                simplicity_penalty += 0.01 * (n_val - 1.0)  # Penalize n > 1

        # SB/SB_mnp models: prefer lower m,n values
        if 'm' in params and 'n' in params:
            m_val = params['m']
            n_val = params['n']
            # Penalty scales with distance from standard SB_mn values (m=0.5, n=1.0)
            simplicity_penalty += 0.01 * abs(m_val - 0.5)
            simplicity_penalty += 0.01 * abs(n_val - 1.0)

            # Additional penalty for p in SB_mnp (prefer p=0, i.e., reduces to SB)
            if 'p' in params:
                p_val = params['p']
                simplicity_penalty += 0.01 * abs(p_val)

        item['score'] += simplicity_penalty
        item['simplicity_penalty'] = simplicity_penalty  # Store for transparency

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

    return valid_fits_data
