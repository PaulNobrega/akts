# models.py
import numpy as np
from typing import Dict, Callable, List, Tuple, Optional
from .datatypes import FAlphaCallable, OdeSystemCallable
from .utils import R_GAS
import warnings

# --- Library of f(alpha) functions (Pure Python) ---
def f_n_order(alpha: float, params: Dict = {'n': 1.0}) -> float:
    """N-th order reaction model: f(alpha) = (1 - alpha)^n"""
    n = params.get('n', 1.0) # Get n from params dict
    if alpha >= 1.0 - 1e-9: return 0.0
    base = 1.0 - alpha
    if base < 0 and n % 1 != 0: return 0.0
    try: return base**n
    except ValueError: return 0.0

def f_avrami_n(alpha: float, params: Dict = {'n': 2.0}) -> float:
    """Avrami-Erofeev model: f(alpha) = n * (1 - alpha) * [-ln(1 - alpha)]^(1 - 1/n)"""
    n = params.get('n', 2.0) # Get n from params dict
    if n == 0: return 0.0
    n_eff = n if n != 1.0 else 1.00001
    if alpha >= 1.0 - 1e-9: return 0.0
    safe_alpha = min(alpha, 0.999999999)
    try: neg_log_one_minus_alpha = -np.log(1.0 - safe_alpha)
    except ValueError: return 0.0
    if neg_log_one_minus_alpha <= 0: return 0.0
    exponent = 1.0 - 1.0 / n_eff
    try:
        if neg_log_one_minus_alpha < 0 and exponent % 1 != 0: return 0.0
        term = neg_log_one_minus_alpha**exponent
    except ValueError: return 0.0
    return n_eff * (1.0 - safe_alpha) * term

def f_sb_mn(alpha: float, params: Dict = {'m': 0.5, 'n': 1.0}) -> float:
    """Sestak-Berggren SB(m,n) model: alpha^m * (1-alpha)^n"""
    m = params.get('m', 0.5) # Get m from params dict
    n = params.get('n', 1.0) # Get n from params dict
    term1 = 0.0
    if alpha <= 1e-9: term1 = 1.0 if m == 0 else (0.0 if m > 0 else np.inf)
    else:
        try: term1 = alpha**m
        except ValueError: return 0.0
    term2 = 0.0
    if alpha >= 1.0 - 1e-9: term2 = 1.0 if n == 0 else (0.0 if n > 0 else np.inf)
    else:
        try: term2 = (1.0 - alpha)**n
        except ValueError: return 0.0
    result = term1 * term2
    if not np.isfinite(result): return 0.0
    return max(0.0, result)

# --- Registry of f(alpha) models ---
F_ALPHA_MODELS: Dict[str, FAlphaCallable] = {
    "F1": f_n_order, # Pass function directly
    "F2": f_n_order,
    "F3": f_n_order,
    "A2": f_avrami_n,
    "A3": f_avrami_n,
    "SB_mn": f_sb_mn,
}
# Store default parameters separately if needed by get_model_info
F_ALPHA_DEFAULT_PARAMS = {
    "F1": {'n': 1.0}, "F2": {'n': 2.0}, "F3": {'n': 3.0},
    "A2": {'n': 2.0}, "A3": {'n': 3.0},
    "SB_mn": {'m': 0.5, 'n': 1.0},
}


# --- ODE System Definitions (Pure Python) ---
def ode_system_single_step(t: float, y: np.ndarray, T_func: Callable, params: Dict) -> np.ndarray:
    """ODE system for a single reaction step: alpha."""
    alpha = y[0]
    if alpha >= 1.0 - 1e-9: return np.array([0.0])
    T = T_func(t)
    if T <= 0: return np.array([0.0])
    Ea = params['Ea']; A = params['A']
    f_alpha_func = params.get('f_alpha_func')
    f_alpha_params = params.get('f_alpha_params', {}) # Contains fixed params like {'n': 1.0}
    if f_alpha_func is None: warnings.warn("Missing 'f_alpha_func'"); return np.array([0.0])
    exp_arg = -Ea / (R_GAS * T); k = A * np.exp(exp_arg) if exp_arg > -700 else 0.0
    try: f_val = f_alpha_func(alpha, f_alpha_params) # Pass the fixed params dict
    except Exception as e_falpha: warnings.warn(f"Error evaluating f_alpha: {e_falpha}"); f_val = 0.0
    dalpha_dt = k * f_val; dalpha_dt = max(0.0, dalpha_dt)
    if not np.isfinite(dalpha_dt): dalpha_dt = 0.0
    return np.array([dalpha_dt])

def ode_system_A_B_C(t: float, y: np.ndarray, T_func: Callable, params: Dict) -> np.ndarray:
    """ODE system for consecutive A -> B -> C (Nth order concentration based)."""
    A_conc, B_conc = y[0], y[1]; T = T_func(t)
    if T <= 0: return np.array([0.0, 0.0])
    Ea1, A1 = params['Ea1'], params['A1']; Ea2, A2 = params['Ea2'], params['A2']
    # Get fixed orders n1, n2 from the params dict
    f1_params = params.get('f1_params', {}); f2_params = params.get('f2_params', {})
    n1 = f1_params.get('n', 1.0); n2 = f2_params.get('n', 1.0) # Default to 1 if not specified
    exp_arg1 = -Ea1 / (R_GAS * T); k1 = A1 * np.exp(exp_arg1) if exp_arg1 > -700 else 0.0
    exp_arg2 = -Ea2 / (R_GAS * T); k2 = A2 * np.exp(exp_arg2) if exp_arg2 > -700 else 0.0
    rate1_conc = k1 * (A_conc**n1) if A_conc > 1e-9 else 0.0
    rate2_conc = k2 * (B_conc**n2) if B_conc > 1e-9 else 0.0
    rate1_conc = max(0.0, rate1_conc); rate2_conc = max(0.0, rate2_conc)
    dA_dt = -rate1_conc; dB_dt = rate1_conc - rate2_conc
    if A_conc <= 1e-9 and dA_dt < 0: dA_dt = 0.0
    if B_conc <= 1e-9 and dB_dt < 0: dB_dt = 0.0
    return np.array([dA_dt, dB_dt])

def ode_system_A_plus_B_C(t: float, y: np.ndarray, T_func: Callable, params: Dict) -> np.ndarray:
    """ODE system for bimolecular A + B -> C."""
    alpha = y[0]; r = params['initial_ratio_r']
    max_alpha = min(1.0, r) if r > 0 else 1.0
    if alpha >= max_alpha - 1e-9: return np.array([0.0])
    T = T_func(t);
    if T <= 0: return np.array([0.0])
    Ea = params['Ea']; A_fitted = params['A']
    m = params.get('m', 1.0); n = params.get('n', 1.0) # Get fixed m, n
    exp_arg = -Ea / (R_GAS * T); k_part = A_fitted * np.exp(exp_arg) if exp_arg > -700 else 0.0
    term1 = (1.0 - alpha); term2 = (r - alpha); conc_part = 0.0
    if term1 > 1e-9 and term2 > 1e-9:
        try: term1_pow = term1**m; term2_pow = term2**n; conc_part = term1_pow * term2_pow
        except ValueError: conc_part = 0.0
    dalpha_dt = k_part * conc_part; dalpha_dt = max(0.0, dalpha_dt)
    if not np.isfinite(dalpha_dt): dalpha_dt = 0.0
    return np.array([dalpha_dt])

# --- Registry of ODE Systems ---
ODE_SYSTEMS: Dict[str, Tuple[OdeSystemCallable, List[str], int]] = {
    "single_step": (ode_system_single_step, ['Ea', 'A'], 1), # Only Ea, A are fitted
    "A->B->C": (ode_system_A_B_C, ['Ea1', 'A1', 'Ea2', 'A2'], 2), # Only Ea/A fitted
    "A+B->C": (ode_system_A_plus_B_C, ['Ea', 'A', 'initial_ratio_r'], 1), # Only Ea, A fitted (r fixed)
}

# --- Helper to get model info ---
# **** REVERTED: Only return base kinetic params to be fitted ****
def get_model_info(model_name: str, f_alpha_model: Optional[str] = None, f_alpha_params: Optional[Dict] = None,
                   f1_model: Optional[str] = None, f1_params: Optional[Dict] = None,
                   f2_model: Optional[str] = None, f2_params: Optional[Dict] = None,
                   bimol_params: Optional[Dict] = None) -> Tuple[OdeSystemCallable, List[str], int, Dict]:
    """ Gets the ODE system, BASE parameter names (A scale), state dimension, and template. """
    if model_name == "single_step":
        if not f_alpha_model or f_alpha_model not in F_ALPHA_MODELS: raise ValueError(f"Valid f_alpha_model required")
        ode_func, base_params, state_dim = ODE_SYSTEMS[model_name]
        f_alpha_func = F_ALPHA_MODELS[f_alpha_model]
        # Use provided f_alpha_params if given, otherwise use defaults
        default_f_params = F_ALPHA_DEFAULT_PARAMS.get(f_alpha_model, {})
        actual_f_params = f_alpha_params if f_alpha_params is not None else default_f_params
        # Template stores the function wrapper and the FIXED parameters for it
        full_params_dict_template = {'f_alpha_func': f_alpha_func, 'f_alpha_params': actual_f_params}
        # Return only base params (Ea, A) to be fitted
        return ode_func, base_params, state_dim, full_params_dict_template

    elif model_name == "A->B->C":
        if not f1_model or f1_model not in F_ALPHA_MODELS or not f2_model or f2_model not in F_ALPHA_MODELS: raise ValueError(f"Valid f1/f2_model required")
        ode_func, base_params, state_dim = ODE_SYSTEMS[model_name]
        f1_func = F_ALPHA_MODELS[f1_model]; f2_func = F_ALPHA_MODELS[f2_model]
        def_f1p = F_ALPHA_DEFAULT_PARAMS.get(f1_model, {}); def_f2p = F_ALPHA_DEFAULT_PARAMS.get(f2_model, {})
        f1p = f1_params if f1_params is not None else def_f1p; f2p = f2_params if f2_params is not None else def_f2p
        # Template stores functions and FIXED parameters for them
        full_params_dict_template = {'f1_func': f1_func, 'f2_func': f2_func, 'f1_params': f1p, 'f2_params': f2p}
        # Return only base params (Ea1, A1, Ea2, A2) to be fitted
        return ode_func, base_params, state_dim, full_params_dict_template

    elif model_name == "A+B->C":
        ode_func, base_params_req, state_dim = ODE_SYSTEMS[model_name] # base_params_req = ['Ea', 'A', 'initial_ratio_r']
        bp = bimol_params if bimol_params is not None else {}
        if 'initial_ratio_r' not in bp: raise ValueError("A+B->C requires 'initial_ratio_r'.")
        # Check if m, n are provided, otherwise use defaults
        m_val = bp.get('m', 1.0); n_val = bp.get('n', 1.0)
        # Template stores fixed r, m, n
        full_params_dict_template = {'initial_ratio_r': bp['initial_ratio_r'], 'm': m_val, 'n': n_val}
        # Return only base params (Ea, A) to be fitted (r is fixed via template)
        params_to_fit = [p for p in base_params_req if p != 'initial_ratio_r']
        return ode_func, params_to_fit, state_dim, full_params_dict_template

    else: raise ValueError(f"Unknown model_name: {model_name}")