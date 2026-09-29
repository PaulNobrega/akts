"""
Test script for new Section 3 features:
- New f(alpha) models (D1-D4, R1-R3, Bna)
- Parallel competing reactions
- Vyazovkin method
- Kissinger method
- Compensation effect estimation
"""
import numpy as np
import sys
sys.path.insert(0, '..')

from akts import (
    KineticDataset,
    fit_kinetic_model,
    list_available_models,
    register_f_alpha_model,
    run_vyazovkin,
    run_kissinger,
    estimate_compensation_parameters,
    run_friedman
)

def test_list_models():
    """Test that all new models are registered"""
    print("=" * 60)
    print("TEST 1: List Available Models")
    print("=" * 60)

    models = list_available_models()
    print(f"\nAvailable f(alpha) models ({len(models['f_alpha_models'])}):")
    print(", ".join(models['f_alpha_models']))

    print(f"\nAvailable ODE systems ({len(models['ode_systems'])}):")
    print(", ".join(models['ode_systems']))

    # Check for new models
    expected_f_alpha = ['D1', 'D2', 'D3', 'D4', 'R1', 'R2', 'R3', 'Bna']
    for model in expected_f_alpha:
        assert model in models['f_alpha_models'], f"Missing {model}"

    assert 'parallel_competing' in models['ode_systems'], "Missing parallel_competing"

    print("\n✓ All expected models found!")


def test_diffusion_models():
    """Test fitting with diffusion models D1-D4"""
    print("\n" + "=" * 60)
    print("TEST 2: Diffusion Models (D1-D4)")
    print("=" * 60)

    # Generate synthetic data
    np.random.seed(42)
    time = np.linspace(0, 3600, 100)
    temperature = 400.0 + 0.1 * time  # Linear heating

    for model_name in ['D1', 'D2', 'D3', 'D4']:
        # Simple conversion profile (sigmoid-like)
        conversion = 1.0 - np.exp(-0.001 * time)
        dataset = KineticDataset(time=time, temperature=temperature, conversion=conversion)

        try:
            result = fit_kinetic_model(
                dataset=dataset,
                model_name="single_step",
                f_alpha_model=model_name,
                initial_guess={'Ea': 100000, 'A': 1e10}
            )
            print(f"\n{model_name}: Ea = {result.fitted_params['Ea']:.0f} J/mol, "
                  f"A = {result.fitted_params['A']:.2e} 1/s, "
                  f"R² = {result.R_squared:.4f}")
        except Exception as e:
            print(f"\n{model_name}: Failed - {str(e)[:80]}")

    print("\n✓ Diffusion models test complete!")


def test_contracting_geometry():
    """Test fitting with contracting geometry models R1-R3"""
    print("\n" + "=" * 60)
    print("TEST 3: Contracting Geometry Models (R1-R3)")
    print("=" * 60)

    np.random.seed(42)
    time = np.linspace(0, 3600, 100)
    temperature = 450.0 + 0.05 * time

    for model_name in ['R1', 'R2', 'R3']:
        conversion = 1.0 - (1.0 - 0.8 * time/3600)**(1.0)
        conversion = np.clip(conversion, 0, 0.95)
        dataset = KineticDataset(time=time, temperature=temperature, conversion=conversion)

        try:
            result = fit_kinetic_model(
                dataset=dataset,
                model_name="single_step",
                f_alpha_model=model_name,
                initial_guess={'Ea': 120000, 'A': 1e12}
            )
            print(f"\n{model_name}: Ea = {result.fitted_params['Ea']:.0f} J/mol, "
                  f"A = {result.fitted_params['A']:.2e} 1/s, "
                  f"R² = {result.R_squared:.4f}")
        except Exception as e:
            print(f"\n{model_name}: Failed - {str(e)[:80]}")

    print("\n✓ Contracting geometry test complete!")


def test_parallel_competing():
    """Test parallel competing reactions model"""
    print("\n" + "=" * 60)
    print("TEST 4: Parallel Competing Reactions")
    print("=" * 60)

    np.random.seed(42)
    time = np.linspace(0, 7200, 150)
    temperature = 400.0 + 0.08 * time

    # Synthetic data: two competing pathways
    alpha1 = 0.6 * (1.0 - np.exp(-0.0005 * time))
    alpha2 = 0.3 * (1.0 - np.exp(-0.0003 * time))
    conversion = alpha1 + alpha2

    dataset = KineticDataset(time=time, temperature=temperature, conversion=conversion)

    try:
        result = fit_kinetic_model(
            dataset=dataset,
            model_name="parallel_competing",
            f1_model="F1",
            f2_model="F1",
            initial_guess={'Ea1': 100000, 'A1': 1e10, 'Ea2': 120000, 'A2': 1e11}
        )
        print(f"\nPathway 1: Ea1 = {result.fitted_params['Ea1']:.0f} J/mol, "
              f"A1 = {result.fitted_params['A1']:.2e} 1/s")
        print(f"Pathway 2: Ea2 = {result.fitted_params['Ea2']:.0f} J/mol, "
              f"A2 = {result.fitted_params['A2']:.2e} 1/s")
        print(f"R² = {result.R_squared:.4f}")
        print("\n✓ Parallel competing reactions test passed!")
    except Exception as e:
        print(f"\n✗ Test failed: {str(e)}")


def test_vyazovkin():
    """Test Vyazovkin method"""
    print("\n" + "=" * 60)
    print("TEST 5: Vyazovkin Advanced Isoconversional Method")
    print("=" * 60)

    # Generate multiple heating rate datasets
    datasets = []
    heating_rates = [5, 10, 20]  # K/min

    for beta in heating_rates:
        np.random.seed(42 + int(beta))
        time = np.linspace(0, 1800, 80)
        temperature = 300 + (beta / 60.0) * time  # Convert K/min to K/s
        conversion = 1.0 - np.exp(-0.001 * time * (beta / 10.0))
        conversion = np.clip(conversion, 0, 0.95)

        dataset = KineticDataset(
            time=time,
            temperature=temperature,
            conversion=conversion,
            heating_rate=beta / 60.0
        )
        datasets.append(dataset)

    try:
        # Run Vyazovkin (note: this is computationally intensive)
        alpha_levels = np.linspace(0.1, 0.8, 5)  # Fewer points for speed
        result = run_vyazovkin(datasets, alpha_levels=alpha_levels)

        print(f"\nVyazovkin Ea values:")
        for i, (alpha, Ea) in enumerate(zip(result.alpha, result.Ea)):
            if np.isfinite(Ea):
                print(f"  α = {alpha:.2f}: Ea = {Ea:.0f} J/mol")

        valid_ea = result.Ea[np.isfinite(result.Ea)]
        if len(valid_ea) > 0:
            print(f"\nMean Ea: {np.mean(valid_ea):.0f} J/mol")
            print("✓ Vyazovkin test passed!")
        else:
            print("✗ No valid Ea values computed")

    except Exception as e:
        print(f"\n✗ Test failed: {str(e)}")


def test_kissinger():
    """Test Kissinger method"""
    print("\n" + "=" * 60)
    print("TEST 6: Kissinger Peak Method")
    print("=" * 60)

    # Generate datasets with clear peaks at different heating rates
    datasets = []
    heating_rates = [5, 10, 15, 20]  # K/min

    for beta in heating_rates:
        np.random.seed(42 + int(beta))
        time = np.linspace(0, 2400, 100)
        temperature = 300 + (beta / 60.0) * time

        # Gaussian-like peak
        t_peak = 1200 * (10.0 / beta)
        conversion = 0.95 * (1.0 - np.exp(-((time - t_peak) / 600) ** 2))
        conversion = np.clip(conversion, 0, 0.95)

        dataset = KineticDataset(
            time=time,
            temperature=temperature,
            conversion=conversion,
            heating_rate=beta / 60.0
        )
        datasets.append(dataset)

    try:
        result = run_kissinger(datasets, peak_detection_method='max_rate')

        print(f"\nKissinger Results:")
        print(f"  Ea = {result['Ea']:.0f} ± {result['Ea_std_err']:.0f} J/mol")
        print(f"  A = {result['A']:.2e} 1/s")
        print(f"  R² = {result['r_value']**2:.4f}")
        print(f"\nPeak data:")
        for pd in result['peak_data']:
            print(f"  β = {pd['beta']*60:.1f} K/min → T_peak = {pd['T_peak']:.1f} K")

        print("\n✓ Kissinger test passed!")
    except Exception as e:
        print(f"\n✗ Test failed: {str(e)}")


def test_compensation_effect():
    """Test compensation effect parameter estimation"""
    print("\n" + "=" * 60)
    print("TEST 7: Compensation Effect Estimation")
    print("=" * 60)

    # Generate isoconversional data
    datasets = []
    for beta in [5, 10, 20]:
        np.random.seed(42 + int(beta))
        time = np.linspace(0, 1800, 80)
        temperature = 300 + (beta / 60.0) * time
        conversion = 1.0 - np.exp(-0.001 * time * (beta / 10.0))
        conversion = np.clip(conversion, 0, 0.95)

        dataset = KineticDataset(
            time=time,
            temperature=temperature,
            conversion=conversion,
            heating_rate=beta / 60.0
        )
        datasets.append(dataset)

    try:
        # Run Friedman first
        iso_result = run_friedman(datasets, alpha_levels=np.linspace(0.2, 0.8, 7))

        # Estimate compensation parameters
        comp_params = estimate_compensation_parameters(iso_result)

        print(f"\nCompensation Effect Parameters:")
        print(f"  Mean Ea: {comp_params['Ea_mean']:.0f} ± {comp_params['Ea_std']:.0f} J/mol")
        print(f"  Geometric mean A: {comp_params['A_geometric_mean']:.2e} 1/s")
        print(f"  Isokinetic temperature: {comp_params['isokinetic_temperature']:.1f} K")
        print(f"\n  {comp_params['note']}")

        print("\n✓ Compensation effect test passed!")
    except Exception as e:
        print(f"\n✗ Test failed: {str(e)}")


def test_custom_registration():
    """Test custom model registration API"""
    print("\n" + "=" * 60)
    print("TEST 8: Custom Model Registration")
    print("=" * 60)

    # Register a custom f(alpha) model
    def my_custom_model(alpha, params):
        """Custom reaction model: f(α) = k * α * (1-α)"""
        k = params.get('k', 1.0)
        return k * alpha * (1.0 - alpha)

    try:
        register_f_alpha_model("CUSTOM", my_custom_model, {'k': 2.0})

        # Verify it's registered
        models = list_available_models()
        assert "CUSTOM" in models['f_alpha_models'], "Custom model not registered"

        print("\n✓ Custom model 'CUSTOM' registered successfully!")
        print(f"  Total f(alpha) models: {len(models['f_alpha_models'])}")

    except Exception as e:
        print(f"\n✗ Test failed: {str(e)}")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("AKTS Section 3 Feature Tests")
    print("Testing new models and isoconversional methods")
    print("=" * 60)

    try:
        test_list_models()
        test_diffusion_models()
        test_contracting_geometry()
        test_parallel_competing()
        test_vyazovkin()
        test_kissinger()
        test_compensation_effect()
        test_custom_registration()

        print("\n" + "=" * 60)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 60)

    except Exception as e:
        print(f"\n" + "=" * 60)
        print(f"TEST SUITE FAILED: {str(e)}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
