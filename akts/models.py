# models.py
import numpy as np
from typing import Dict, Callable, List, Tuple, Optional, Union
from .datatypes import FAlphaCallable, OdeSystemCallable
from .utils import R_GAS
import warnings

# Try to import Numba for JIT compilation
try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Fallback: no-op decorator
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator

# --- Numba-JIT Core Implementations (scalar parameters) ---
@njit(cache=True, fastmath=True)
def _f_n_order_jit(alpha: float, n: float) -> float:
    """JIT-compiled n-th order: f(α) = (1-α)^n"""
    if alpha >= 0.999999999:
        return 0.0
    base = 1.0 - alpha
    if base < 0.0:
        return 0.0
    return base ** n

@njit(cache=True, fastmath=True)
def _f_avrami_n_jit(alpha: float, n: float) -> float:
    """JIT-compiled Avrami: f(α) = n·(1-α)·[-ln(1-α)]^(1-1/n)"""
    if n == 0.0 or alpha >= 0.999999999:
        return 0.0
    n_eff = n if abs(n - 1.0) > 1e-5 else 1.00001
    safe_alpha = min(alpha, 0.999999999)
    neg_log = -np.log(1.0 - safe_alpha)
    if neg_log <= 0.0:
        return 0.0
    exponent = 1.0 - 1.0 / n_eff
    term = neg_log ** exponent
    return n_eff * (1.0 - safe_alpha) * term

@njit(cache=True, fastmath=True)
def _f_sb_mn_jit(alpha: float, m: float, n: float) -> float:
    """JIT-compiled Sestak-Berggren: α^m·(1-α)^n"""
    if alpha <= 1e-9:
        term1 = 1.0 if abs(m) < 1e-9 else 0.0
    else:
        term1 = alpha ** m

    if alpha >= 0.999999999:
        term2 = 1.0 if abs(n) < 1e-9 else 0.0
    else:
        term2 = (1.0 - alpha) ** n

    result = term1 * term2
    return max(0.0, result) if np.isfinite(result) else 0.0

@njit(cache=True, fastmath=True)
def _f_diffusion_d1_jit(alpha: float) -> float:
    """JIT-compiled D1: f(α) = 1/(2α)"""
    if alpha <= 1e-9:
        return 1e10  # Large but finite
    return 1.0 / (2.0 * alpha)

@njit(cache=True, fastmath=True)
def _f_diffusion_d2_jit(alpha: float) -> float:
    """JIT-compiled D2 Valensi: f(α) = [-ln(1-α)]^-1"""
    if alpha >= 0.999999999:
        return 0.0
    safe_alpha = min(alpha, 0.999999999)
    neg_log = -np.log(1.0 - safe_alpha)
    if neg_log <= 1e-9:
        return 1e10
    return 1.0 / neg_log

@njit(cache=True, fastmath=True)
def _f_diffusion_d3_jit(alpha: float) -> float:
    """JIT-compiled D3 Jander: f(α) = (3/2)·(1-α)^(2/3)/[1-(1-α)^(1/3)]"""
    if alpha <= 1e-9:
        return 1e10
    if alpha >= 0.999999999:
        return 0.0
    safe_alpha = min(max(alpha, 1e-9), 0.999999999)
    one_minus_alpha = 1.0 - safe_alpha
    term1 = one_minus_alpha ** (2.0/3.0)
    term2 = 1.0 - one_minus_alpha ** (1.0/3.0)
    if abs(term2) < 1e-9:
        return 1e10
    return 1.5 * term1 / term2

@njit(cache=True, fastmath=True)
def _f_diffusion_d4_jit(alpha: float) -> float:
    """JIT-compiled D4 Ginstling-Brounshtein: f(α) = (3/2)/[(1-α)^(-1/3)-1]"""
    if alpha <= 1e-9:
        return 1e10
    if alpha >= 0.999999999:
        return 0.0
    safe_alpha = min(max(alpha, 1e-9), 0.999999999)
    one_minus_alpha = 1.0 - safe_alpha
    term = one_minus_alpha ** (-1.0/3.0) - 1.0
    if abs(term) < 1e-9:
        return 1e10
    return 1.5 / term

@njit(cache=True, fastmath=True)
def _f_contracting_rn_jit(alpha: float, n: float) -> float:
    """JIT-compiled contracting geometry: f(α) = n·(1-α)^((n-1)/n)"""
    if n == 0.0 or alpha >= 0.999999999:
        return 0.0
    exponent = (n - 1.0) / n
    base = 1.0 - alpha
    if base < 0.0:
        return 0.0
    return n * (base ** exponent)

@njit(cache=True, fastmath=True)
def _f_bna_jit(alpha: float, c: float) -> float:
    """JIT-compiled Prout-Tompkins: f(α) = α^c·(1-α)"""
    if alpha <= 1e-9 or alpha >= 0.999999999:
        return 0.0
    safe_alpha = min(max(alpha, 1e-9), 0.999999999)
    return (safe_alpha ** c) * (1.0 - safe_alpha)


# --- Library of f(alpha) functions (Dict interface, calls JIT'd versions) ---
def f_n_order(alpha: float, params: Dict = {'n': 1.0}) -> float:
    """N-th order reaction model: f(alpha) = (1 - alpha)^n"""
    n = params.get('n', 1.0)
    return _f_n_order_jit(alpha, n)

def f_avrami_n(alpha: float, params: Dict = {'n': 2.0}) -> float:
    """Avrami-Erofeev model: f(alpha) = n * (1 - alpha) * [-ln(1 - alpha)]^(1 - 1/n)"""
    n = params.get('n', 2.0)
    return _f_avrami_n_jit(alpha, n)

def f_sb_mn(alpha: float, params: Dict = {'m': 0.5, 'n': 1.0}) -> float:
    """Sestak-Berggren SB(m,n) model: alpha^m * (1-alpha)^n"""
    m = params.get('m', 0.5)
    n = params.get('n', 1.0)
    return _f_sb_mn_jit(alpha, m, n)

@njit(cache=True, fastmath=True)
def _f_sb_mnp_jit(alpha: float, m: float, n: float, p: float) -> float:
    """Extended Sestak-Berggren SB(m,n,p): α^m · (1-α)^n · [-ln(1-α)]^p"""
    if alpha <= 1e-9:
        # At α→0: α^m term dominates
        term1 = 1.0 if abs(m) < 1e-9 else 0.0
        term2 = 1.0
        term3 = 1.0 if abs(p) < 1e-9 else 0.0
    elif alpha >= 0.999999999:
        # At α→1: (1-α)^n and [-ln(1-α)]^p terms dominate
        term1 = 1.0
        term2 = 1.0 if abs(n) < 1e-9 else 0.0
        term3 = 1.0 if abs(p) < 1e-9 else 0.0
    else:
        term1 = alpha ** m if m != 0 else 1.0
        term2 = (1.0 - alpha) ** n if n != 0 else 1.0
        # -ln(1-α) term
        if abs(p) < 1e-9:
            term3 = 1.0
        else:
            log_term = -np.log(1.0 - alpha)
            term3 = log_term ** p

    result = term1 * term2 * term3
    return max(0.0, result) if np.isfinite(result) else 0.0

def f_sb_mnp(alpha: float, params: Dict = {'m': 0.5, 'n': 1.0, 'p': 0.0}) -> float:
    """Extended Sestak-Berggren SB(m,n,p): α^m · (1-α)^n · [-ln(1-α)]^p"""
    m = params.get('m', 0.5)
    n = params.get('n', 1.0)
    p = params.get('p', 0.0)
    return _f_sb_mnp_jit(alpha, m, n, p)

def f_diffusion_d1(alpha: float, params: Dict = None) -> float:
    """1D diffusion model: f(alpha) = 1/(2*alpha)"""
    return _f_diffusion_d1_jit(alpha)

def f_diffusion_d2(alpha: float, params: Dict = None) -> float:
    """2D diffusion model (Valensi): f(alpha) = [-ln(1-alpha)]^-1"""
    return _f_diffusion_d2_jit(alpha)

def f_diffusion_d3(alpha: float, params: Dict = None) -> float:
    """3D diffusion model (Jander): f(alpha) = (3/2) * (1-alpha)^(2/3) / [1 - (1-alpha)^(1/3)]"""
    return _f_diffusion_d3_jit(alpha)

def f_diffusion_d4(alpha: float, params: Dict = None) -> float:
    """3D diffusion model (Ginstling-Brounshtein): f(alpha) = (3/2) * [(1-alpha)^(-1/3) - 1]^-1"""
    return _f_diffusion_d4_jit(alpha)

def f_contracting_rn(alpha: float, params: Dict = {'n': 2.0}) -> float:
    """Contracting geometry model: f(alpha) = n * (1-alpha)^((n-1)/n)"""
    n = params.get('n', 2.0)
    return _f_contracting_rn_jit(alpha, n)

def f_prout_tompkins(alpha: float, params: Dict = {'c': 1.0}) -> float:
    """Prout-Tompkins model (autocatalytic): f(alpha) = alpha^c * (1-alpha)"""
    c = params.get('c', 1.0)
    return _f_bna_jit(alpha, c)

# --- Registry of f(alpha) models ---
F_ALPHA_MODELS: Dict[str, FAlphaCallable] = {
    "F0": f_n_order, # n=0 -> f(alpha)=1, i.e. constant rate (zero-order)
    "F1": f_n_order, # Pass function directly
    "F2": f_n_order,
    "F3": f_n_order,
    "Fn": f_n_order,  # n as fitted parameter
    "A2": f_avrami_n,
    "A3": f_avrami_n,
    "SB_mn": f_sb_mn,
    "SB": f_sb_mn,  # m and n as fitted parameters
    "SB_mnp": f_sb_mnp,  # Extended SB with m, n, p as fitted parameters
    "D1": f_diffusion_d1,
    "D2": f_diffusion_d2,
    "D3": f_diffusion_d3,
    "D4": f_diffusion_d4,
    "R1": f_contracting_rn,
    "R2": f_contracting_rn,
    "R3": f_contracting_rn,
    "Bna": f_prout_tompkins,
}
# User-facing display names, shared by helpers.py progress messages and reporting.py.
MODEL_DISPLAY_NAMES = {
    # f(alpha) models
    'F0': 'F0 (zero-order)',
    'F1': 'F1 (first-order)',
    'F2': 'F2 (second-order)',
    'F3': 'F3 (third-order)',
    'A2': 'A2 (Avrami-Erofeev, n=2)',
    'A3': 'A3 (Avrami-Erofeev, n=3)',
    'R2': 'R2 (contracting area)',
    'R3': 'R3 (contracting volume)',
    'D2': 'D2 (2D diffusion)',
    'D3': 'D3 (3D diffusion, Jander)',
    'D4': 'D4 (3D diffusion, Ginstling-Brounshtein)',
    'D1': 'D1 (1D diffusion)',
    'SB_mn': 'SB(m,n) (Sestak-Berggren, autocatalytic)',
    'Bna': 'Bna (Prout-Tompkins, autocatalytic)',
    # ODE models (multi-step)
    'A->B->C': 'A->B->C (consecutive reactions)',
    'A+B->C': 'A+B->C (bimolecular)',
    # Model-free (isoconversional)
    'Friedman': 'Friedman (model-free isoconversional)',
}


def model_display_name(model_name: str) -> str:
    """'F1_model' / 'F1' -> 'F1 (first-order)'; unknown names are returned without the _model suffix."""
    base = model_name.replace('_model', '')
    return MODEL_DISPLAY_NAMES.get(base, base)


# Store default parameters separately if needed by get_model_info
F_ALPHA_DEFAULT_PARAMS = {
    "F0": {'n': 0.0}, "F1": {'n': 1.0}, "F2": {'n': 2.0}, "F3": {'n': 3.0},
    "Fn": {},  # n fitted, no default
    "A2": {'n': 2.0}, "A3": {'n': 3.0},
    "SB_mn": {'m': 0.5, 'n': 1.0},
    "SB": {},  # m and n fitted, no defaults
    "SB_mnp": {},  # m, n, p fitted, no defaults
    "D1": {}, "D2": {}, "D3": {}, "D4": {},
    "R1": {'n': 1.0}, "R2": {'n': 2.0}, "R3": {'n': 3.0},
    "Bna": {'c': 1.0},
}


# --- Closed-Form Solutions for Isothermal Conditions ---
# For constant temperature, many models have analytic solutions α(t) = g⁻¹(kt)
# where g(α) = ∫[0 to α] 1/f(α') dα' and k = A·exp(-Ea/RT)

def alpha_of_kt_f0(kt: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """F0 (zero-order): α(t) = kt"""
    return np.clip(kt, 0.0, 1.0)

def alpha_of_kt_f1(kt: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """F1 (first-order): α(t) = 1 - exp(-kt)"""
    return 1.0 - np.exp(-np.clip(kt, 0, 700))

def alpha_of_kt_f2(kt: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """F2 (second-order): α(t) = kt/(1+kt)"""
    return kt / (1.0 + kt)

def alpha_of_kt_f3(kt: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """F3 (third-order): α(t) = 1 - 1/√(1+2kt)"""
    return 1.0 - 1.0 / np.sqrt(1.0 + 2.0 * kt)

def alpha_of_kt_a2(kt: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """A2 (Avrami n=2): α(t) = 1 - exp(-(kt)²)"""
    kt_safe = np.clip(kt, 0, 26.5)  # exp(-700) limit → kt < sqrt(700) ≈ 26.5
    return 1.0 - np.exp(-(kt_safe ** 2))

def alpha_of_kt_a3(kt: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """A3 (Avrami n=3): α(t) = 1 - exp(-(kt)³)"""
    kt_safe = np.clip(kt, 0, 8.88)  # exp(-700) limit → kt < cbrt(700) ≈ 8.88
    return 1.0 - np.exp(-(kt_safe ** 3))

def alpha_of_kt_r2(kt: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """R2 (contracting area): α(t) = 1 - (1-kt)²"""
    kt_safe = np.clip(kt, 0, 1.0)
    return 1.0 - (1.0 - kt_safe) ** 2

def alpha_of_kt_r3(kt: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """R3 (contracting volume): α(t) = 1 - (1-kt)³"""
    kt_safe = np.clip(kt, 0, 1.0)
    return 1.0 - (1.0 - kt_safe) ** 3

def alpha_of_kt_d2(kt: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """D2 (2D diffusion, Valensi equation): g(α) = -ln(1-α) - α

    For D2, g(α) = -ln(1-α) - α = kt.
    No closed-form inverse, use numerical root-finding.
    """
    from scipy.optimize import fsolve

    def g_d2(alpha):
        """g(α) = -ln(1-α) - α for D2"""
        if alpha >= 1.0 - 1e-9:
            return 1e9  # g approaches infinity as α→1
        return -np.log(1.0 - alpha) - alpha

    def solve_single(kt_val):
        if kt_val <= 0:
            return 0.0
        if kt_val > 10:
            # For large kt, α approaches 1
            return 0.999
        # Solve g(α) = kt for α
        # Initial guess: use linear approximation for small kt
        alpha_guess = min(0.8, kt_val / (1.0 + kt_val))
        try:
            alpha = fsolve(lambda a: g_d2(a) - kt_val, alpha_guess, full_output=False)[0]
            return float(np.clip(alpha, 0.0, 0.999))
        except:
            return alpha_guess

    if np.isscalar(kt):
        return solve_single(kt)
    else:
        return np.array([solve_single(k) for k in kt])

def alpha_of_kt_d3(kt: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """D3 (3D diffusion, Jander): α from g(α) = [1-(1-α)^(1/3)]²"""
    # g(α) = [1-(1-α)^(1/3)]² = kt
    # Solve: 1-(1-α)^(1/3) = ±√kt
    # → (1-α)^(1/3) = 1 - √kt (take positive root)
    # → α = 1 - (1-√kt)³
    sqrt_kt = np.sqrt(np.clip(kt, 0, 1.0))
    return 1.0 - (1.0 - sqrt_kt) ** 3

def alpha_of_kt_d4(kt: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """D4 (3D diffusion, Ginstling-Brounshtein): α from g(α) = 1 - (2α/3) - (1-α)^(2/3)"""
    from scipy.optimize import fsolve

    def g_d4(alpha):
        if alpha >= 1.0 - 1e-9:
            return 1.0
        return 1.0 - (2.0 * alpha / 3.0) - (1.0 - alpha) ** (2.0 / 3.0)

    def solve_single(kt_val):
        if kt_val <= 0:
            return 0.0
        alpha_guess = min(0.5, kt_val)
        try:
            alpha = fsolve(lambda a: g_d4(a) - kt_val, alpha_guess, full_output=False)[0]
            return float(np.clip(alpha, 0.0, 0.999))
        except:
            return alpha_guess

    if np.isscalar(kt):
        return solve_single(kt)
    else:
        return np.array([solve_single(k) for k in kt])

# Registry mapping model names to their closed-form α(kt) functions
CLOSED_FORM_REGISTRY: Dict[str, Callable] = {
    "F0": alpha_of_kt_f0,
    "F1": alpha_of_kt_f1,
    "F2": alpha_of_kt_f2,
    "F3": alpha_of_kt_f3,
    "A2": alpha_of_kt_a2,
    "A3": alpha_of_kt_a3,
    "R2": alpha_of_kt_r2,
    "R3": alpha_of_kt_r3,
    "D2": alpha_of_kt_d2,
    "D3": alpha_of_kt_d3,
    "D4": alpha_of_kt_d4,
}

def has_closed_form(model_name: str) -> bool:
    """Check if a model has a closed-form solution for isothermal conditions."""
    return model_name in CLOSED_FORM_REGISTRY


# --- ODE System Definitions (Pure Python) ---
def ode_system_single_step(t: float, y: np.ndarray, T_func: Callable, params: Dict) -> np.ndarray:
    """ODE system for a single reaction step: alpha."""
    alpha = y[0]
    if alpha >= 1.0 - 1e-9: return np.array([0.0])
    T = T_func(t)
    if T <= 0: return np.array([0.0])
    Ea = params['Ea']; A = params['A']
    f_alpha_func = params.get('f_alpha_func')
    f_alpha_params = params.get('f_alpha_params', {})

    # Check if shape parameters are being fitted (e.g., 'n' for Fn, 'm' and 'n' for SB)
    fitted_shape_params = params.get('fitted_shape_params', [])
    if fitted_shape_params:
        # Extract fitted shape parameters from main params dict and override f_alpha_params
        f_alpha_params = dict(f_alpha_params)  # Copy to avoid modifying template
        for param_name in fitted_shape_params:
            if param_name in params:
                f_alpha_params[param_name] = params[param_name]

    if f_alpha_func is None: warnings.warn("Missing 'f_alpha_func'"); return np.array([0.0])
    exp_arg = -Ea / (R_GAS * T); k = A * np.exp(exp_arg) if exp_arg > -700 else 0.0
    try: f_val = f_alpha_func(alpha, f_alpha_params) # Pass the params dict (fixed or fitted)
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
    # Smooth RHS: hard concentration thresholds make stiff solvers chatter and never finish.
    rate1_conc = k1 * max(A_conc, 0.0)**n1
    rate2_conc = k2 * max(B_conc, 0.0)**n2
    dA_dt = -rate1_conc; dB_dt = rate1_conc - rate2_conc
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

def ode_system_parallel_competing(t: float, y: np.ndarray, T_func: Callable, params: Dict) -> np.ndarray:
    """
    ODE system for parallel competing reactions: A -> B1 (pathway 1) and A -> B2 (pathway 2).

    State vector: [alpha1, alpha2] where alpha1 + alpha2 <= 1.0 (total conversion)
    Each pathway has its own Ea, A, and f(alpha) model.
    """
    alpha1, alpha2 = y[0], y[1]
    total_alpha = alpha1 + alpha2
    if total_alpha >= 1.0 - 1e-9: return np.array([0.0, 0.0])

    T = T_func(t)
    if T <= 0: return np.array([0.0, 0.0])

    # Pathway 1 parameters
    Ea1 = params['Ea1']; A1 = params['A1']
    f1_func = params.get('f1_func')
    f1_params = params.get('f1_params', {})

    # Pathway 2 parameters
    Ea2 = params['Ea2']; A2 = params['A2']
    f2_func = params.get('f2_func')
    f2_params = params.get('f2_params', {})

    # Calculate rates for each pathway
    exp_arg1 = -Ea1 / (R_GAS * T)
    k1 = A1 * np.exp(exp_arg1) if exp_arg1 > -700 else 0.0

    exp_arg2 = -Ea2 / (R_GAS * T)
    k2 = A2 * np.exp(exp_arg2) if exp_arg2 > -700 else 0.0

    # f(alpha) for remaining reactant (1 - total_alpha)
    remaining_A = 1.0 - total_alpha
    try:
        f1_val = f1_func(remaining_A, f1_params) if f1_func else (remaining_A if remaining_A > 1e-9 else 0.0)
        f2_val = f2_func(remaining_A, f2_params) if f2_func else (remaining_A if remaining_A > 1e-9 else 0.0)
    except Exception:
        f1_val = f2_val = 0.0

    dalpha1_dt = k1 * f1_val
    dalpha2_dt = k2 * f2_val

    # Clamp to physical bounds
    dalpha1_dt = max(0.0, dalpha1_dt) if np.isfinite(dalpha1_dt) else 0.0
    dalpha2_dt = max(0.0, dalpha2_dt) if np.isfinite(dalpha2_dt) else 0.0

    return np.array([dalpha1_dt, dalpha2_dt])

def ode_system_humidity(t: float, y: np.ndarray, T_func: Callable, RH_func: Callable, params: Dict) -> np.ndarray:
    """
    ODE system for single-step with humidity dependence (ASAP/Waterman model).

    k(T, RH) = A · exp(-Ea/RT + B·RH)

    Parameters
    ----------
    t : float
        Time
    y : np.ndarray
        State vector [alpha]
    T_func : Callable
        Temperature interpolation function T(t)
    RH_func : Callable
        Relative humidity interpolation function RH(t), returns values in [0, 1]
    params : Dict
        Must contain 'Ea', 'A', 'B' (humidity coefficient), 'f_alpha_func', 'f_alpha_params'

    Returns
    -------
    np.ndarray
        [dalpha/dt]
    """
    alpha = y[0]
    if alpha >= 1.0 - 1e-9:
        return np.array([0.0])

    T = T_func(t)
    if T <= 0:
        return np.array([0.0])

    RH = RH_func(t)

    Ea = params['Ea']
    A = params['A']
    B = params['B']  # Humidity coefficient

    # Modified Arrhenius with humidity: k(T,RH) = A·exp(-Ea/RT + B·RH)
    exp_arg = -Ea / (R_GAS * T) + B * RH
    k = A * np.exp(exp_arg) if -700 < exp_arg < 700 else 0.0

    # Get f(alpha) function
    f_alpha_func = params.get('f_alpha_func')
    f_alpha_params = params.get('f_alpha_params', {})

    # Handle fitted shape parameters (same as single_step)
    fitted_shape_params = params.get('fitted_shape_params', [])
    if fitted_shape_params:
        f_alpha_params = dict(f_alpha_params)
        for param_name in fitted_shape_params:
            if param_name in params:
                f_alpha_params[param_name] = params[param_name]

    if f_alpha_func is None:
        f_val = 1.0  # Default to zero-order if not specified
    else:
        try:
            f_val = f_alpha_func(alpha, f_alpha_params)
        except Exception:
            f_val = 0.0

    dalpha_dt = k * f_val
    dalpha_dt = max(0.0, dalpha_dt)
    if not np.isfinite(dalpha_dt):
        dalpha_dt = 0.0

    return np.array([dalpha_dt])

# --- Registry of ODE Systems ---
ODE_SYSTEMS: Dict[str, Tuple[OdeSystemCallable, List[str], int]] = {
    "single_step": (ode_system_single_step, ['Ea', 'A'], 1), # Only Ea, A are fitted
    "A->B->C": (ode_system_A_B_C, ['Ea1', 'A1', 'Ea2', 'A2'], 2), # Only Ea/A fitted
    "A+B->C": (ode_system_A_plus_B_C, ['Ea', 'A', 'initial_ratio_r'], 1), # Only Ea, A fitted (r fixed)
    "parallel_competing": (ode_system_parallel_competing, ['Ea1', 'A1', 'Ea2', 'A2'], 2), # Two competing pathways
    "humidity": (ode_system_humidity, ['Ea', 'A', 'B'], 1),  # Humidity-dependent kinetics
}

# --- Explicit declaration of which base parameters are Arrhenius pre-exponential
# factors (fitted internally on a log scale by core.py). This is metadata about
# each model, not something that should be guessed from the parameter's name
# (e.g. a naming heuristic would misclassify a future rate prefactor not named
# "A"/"A1"/"A2", or a non-rate parameter that happens to start with "A").
ODE_SYSTEMS_LOG_PARAMS: Dict[str, frozenset] = {
    "single_step": frozenset({'A'}),
    "A->B->C": frozenset({'A1', 'A2'}),
    "A+B->C": frozenset({'A'}),
    "parallel_competing": frozenset({'A1', 'A2'}),
    "humidity": frozenset({'A'}),  # A is log-scale, B is linear
}

def get_log_param_names(model_name: str) -> frozenset:
    """
    Returns the set of base parameter names for `model_name` that are Arrhenius
    pre-exponential factors, fitted internally on a log scale (see core.py).
    This is explicit model metadata and must not be inferred from a parameter's
    name shape (e.g. "starts with 'A'") — that breaks for any future parameter
    that happens to start with 'A' without being a rate prefactor.
    """
    # Empirical models use 'A' as pre-exponential factor (fitted on log scale)
    if model_name.startswith('Empirical_'):
        return frozenset({'A'})

    if model_name not in ODE_SYSTEMS_LOG_PARAMS:
        raise ValueError(f"Unknown model_name: {model_name}")
    return ODE_SYSTEMS_LOG_PARAMS[model_name]

# --- User-pluggable model registration API ---
def register_f_alpha_model(name: str, func: FAlphaCallable, default_params: Optional[Dict] = None) -> None:
    """
    Register a custom f(alpha) model for use with kinetic fitting.

    Parameters
    ----------
    name : str
        Name for the model (e.g., "MY_MODEL"). Should not conflict with built-in names.
    func : FAlphaCallable
        Function with signature f(alpha: float, params: Dict) -> float that computes
        the reaction model. Should handle edge cases (alpha=0, alpha=1) gracefully.
    default_params : Dict, optional
        Default parameters for the model. If None, defaults to an empty dict.

    Example
    -------
    >>> def my_custom_model(alpha, params):
    ...     k = params.get('k', 1.0)
    ...     return k * alpha * (1 - alpha)
    >>> register_f_alpha_model("CUSTOM", my_custom_model, {'k': 1.0})
    """
    if name in F_ALPHA_MODELS:
        warnings.warn(f"Overwriting existing f(alpha) model: {name}")
    F_ALPHA_MODELS[name] = func
    F_ALPHA_DEFAULT_PARAMS[name] = default_params if default_params is not None else {}

def register_ode_system(name: str, ode_func: OdeSystemCallable, param_names: List[str],
                       state_dim: int, log_param_names: Optional[List[str]] = None) -> None:
    """
    Register a custom ODE system for kinetic modeling.

    Parameters
    ----------
    name : str
        Name for the ODE system (e.g., "parallel_competing").
    ode_func : OdeSystemCallable
        Function with signature f(t, y, T_func, params) -> np.ndarray that returns dy/dt.
    param_names : List[str]
        List of parameter names to be fitted (e.g., ['Ea1', 'A1', 'Ea2', 'A2']).
    state_dim : int
        Dimension of the state vector y.
    log_param_names : List[str], optional
        List of parameter names that are Arrhenius pre-exponential factors
        (will be fitted on log scale). If None, assumes parameters named 'A', 'A1', 'A2', etc.

    Example
    -------
    >>> def parallel_ode(t, y, T_func, params):
    ...     alpha1, alpha2 = y
    ...     # ... compute rates ...
    ...     return np.array([dalpha1_dt, dalpha2_dt])
    >>> register_ode_system("parallel", parallel_ode, ['Ea1', 'A1', 'Ea2', 'A2'],
    ...                     state_dim=2, log_param_names=['A1', 'A2'])
    """
    if name in ODE_SYSTEMS:
        warnings.warn(f"Overwriting existing ODE system: {name}")
    ODE_SYSTEMS[name] = (ode_func, param_names, state_dim)

    # Determine which params are log-scale (Arrhenius pre-factors)
    if log_param_names is None:
        # Default heuristic: parameters named A, A1, A2, etc.
        log_params = frozenset(p for p in param_names if p.startswith('A') and
                               (p[1:].isdigit() or len(p) == 1))
    else:
        log_params = frozenset(log_param_names)
    ODE_SYSTEMS_LOG_PARAMS[name] = log_params

def list_available_models() -> Dict[str, List[str]]:
    """
    List all available models (both built-in and user-registered).

    Returns
    -------
    Dict with keys:
        'f_alpha_models': List of available f(alpha) model names
        'ode_systems': List of available ODE system names
    """
    return {
        'f_alpha_models': sorted(F_ALPHA_MODELS.keys()),
        'ode_systems': sorted(ODE_SYSTEMS.keys())
    }

# --- Helper to get model info ---
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

        # Check if this is a model with fitted shape parameters
        if f_alpha_model == "Fn":
            # Fn model: n is a fitted parameter
            base_params_extended = base_params + ['n']
            # Template stores function but no fixed params (n comes from optimizer)
            full_params_dict_template = {'f_alpha_func': f_alpha_func, 'f_alpha_params': {}, 'fitted_shape_params': ['n']}
            return ode_func, base_params_extended, state_dim, full_params_dict_template
        elif f_alpha_model == "SB":
            # SB model: m and n are fitted parameters
            base_params_extended = base_params + ['m', 'n']
            # Template stores function but no fixed params (m, n come from optimizer)
            full_params_dict_template = {'f_alpha_func': f_alpha_func, 'f_alpha_params': {}, 'fitted_shape_params': ['m', 'n']}
            return ode_func, base_params_extended, state_dim, full_params_dict_template
        elif f_alpha_model == "SB_mnp":
            # Extended SB model: m, n, and p are fitted parameters
            base_params_extended = base_params + ['m', 'n', 'p']
            # Template stores function but no fixed params (m, n, p come from optimizer)
            full_params_dict_template = {'f_alpha_func': f_alpha_func, 'f_alpha_params': {}, 'fitted_shape_params': ['m', 'n', 'p']}
            return ode_func, base_params_extended, state_dim, full_params_dict_template
        else:
            # Standard models with fixed shape parameters
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

    elif model_name == "parallel_competing":
        if not f1_model or f1_model not in F_ALPHA_MODELS or not f2_model or f2_model not in F_ALPHA_MODELS:
            raise ValueError(f"parallel_competing requires valid f1_model and f2_model")
        ode_func, base_params, state_dim = ODE_SYSTEMS[model_name]
        f1_func = F_ALPHA_MODELS[f1_model]; f2_func = F_ALPHA_MODELS[f2_model]
        def_f1p = F_ALPHA_DEFAULT_PARAMS.get(f1_model, {}); def_f2p = F_ALPHA_DEFAULT_PARAMS.get(f2_model, {})
        f1p = f1_params if f1_params is not None else def_f1p
        f2p = f2_params if f2_params is not None else def_f2p
        # Template stores functions and FIXED parameters for them
        full_params_dict_template = {'f1_func': f1_func, 'f2_func': f2_func, 'f1_params': f1p, 'f2_params': f2p}
        # Return only base params (Ea1, A1, Ea2, A2) to be fitted
        return ode_func, base_params, state_dim, full_params_dict_template

    else: raise ValueError(f"Unknown model_name: {model_name}")