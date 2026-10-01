"""
Generate the synthetic %HMW stability data used by this example.

The data are simulated from a known two-step Sestak-Berggren model (see TRUE_PARAMETERS)
with duplicate measurements and Gaussian assay noise, so the example can be published
and the fitted parameters can be checked against the values that generated them.

    dα/dt = k1(T)·α^m1·(1-α)^n1 + k2(T)·α^m2·(1-α)^n2,   ki(T) = Ai·exp(-Eai/RT)
    HMW(%) = HMW0 + α·(100 - HMW0)

Run:  python generate_synthetic_data.py
"""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from akts import simulate_kinetics  # noqa: E402

R_GAS = 8.31446261815324
T_REF_K = 298.15

# Rates are set at 25 °C (k_ref, 1/s) and converted to pre-exponential factors.
EA1, K1_REF, M1, N1 = 250e3, 1.2e-8, 0.4, 2.0   # steep aggregation step
EA2, K2_REF, M2, N2 = 85e3, 6.0e-9, 0.0, 4.0    # slow low-temperature step
TRUE_PARAMETERS = {
    'Ea1': EA1, 'A1': K1_REF * np.exp(EA1 / (R_GAS * T_REF_K)), 'm1': M1, 'n1': N1,
    'Ea2': EA2, 'A2': K2_REF * np.exp(EA2 / (R_GAS * T_REF_K)), 'm2': M2, 'n2': N2,
}

HMW0 = 1.2           # % HMW at time zero
ASSAY_SD = 0.05      # % HMW, absolute assay noise per measurement
REPLICATES = 2
SEED = 20260930

# (temperature °C, sampling days)
STUDY_DESIGN = [
    (5, [0, 7, 14, 21, 28, 35, 42, 56]),
    (15, [0, 7, 14, 21, 28, 35, 42, 56]),
    (25, [0, 7, 14, 21, 28, 35, 42, 56]),
    (30, [0, 7, 14, 21, 28, 35, 42]),
    (40, [0, 3, 7, 10, 14, 17, 21]),
]


def simulate_hmw(temperature_C: float, days) -> np.ndarray:
    t_sec = np.asarray(days, dtype=float) * 86400.0
    T_K = temperature_C + 273.15
    alpha = simulate_kinetics('SB2', {'sb2_params': {}}, TRUE_PARAMETERS, 0.0,
                              lambda _t: T_K, t_sec).conversion
    return HMW0 + alpha * (100.0 - HMW0)


def main():
    rng = np.random.default_rng(SEED)
    out_dir = Path(__file__).parent
    for temperature_C, days in STUDY_DESIGN:
        hmw = simulate_hmw(temperature_C, days)
        lines = ['Time(day),Temperature(°C),HMW(%)']
        for day, value in zip(days, hmw):
            for _ in range(REPLICATES):
                noisy = max(value + rng.normal(0.0, ASSAY_SD), 0.0)
                lines.append(f'{day},{temperature_C},{noisy:.2f}')
        path = out_dir / f'HMW_{temperature_C}C.csv'
        path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
        print(f'Wrote {path.name}: {len(days)} time points x {REPLICATES} replicates, '
              f'final mean {hmw[-1]:.2f}% HMW')

    print('\nGenerating parameters:')
    print(f"  Ea1 = {EA1/1e3:.1f} kJ/mol, ln(A1*s) = {np.log(TRUE_PARAMETERS['A1']):.3f}, m1 = {M1}, n1 = {N1}")
    print(f"  Ea2 = {EA2/1e3:.1f} kJ/mol, ln(A2*s) = {np.log(TRUE_PARAMETERS['A2']):.3f}, m2 = {M2}, n2 = {N2}")


if __name__ == '__main__':
    main()
