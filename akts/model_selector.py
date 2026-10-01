"""
Model selector class for intuitive model selection with IDE autocomplete.

Provides structured access to all available kinetic models with dot notation:
    models.all                    # All available models
    models.mechanistic.all        # All mechanistic models
    models.empirical.all          # All empirical models
    models.kinetic.F1             # Specific model
    models.ode.consecutive        # ODE models
    models.modelfree.Friedman     # Model-free methods

Example usage:
    from akts import models, auto_model_isothermal_data

    results = auto_model_isothermal_data(
        data_files=['25C.csv', '40C.csv'],
        models_to_try=models.mechanistic.all + [models.empirical.Linear],
        ...
    )
"""

from typing import List


class _KineticModels:
    """Solid-state mechanistic kinetic models."""

    # All return lists for safe concatenation: models.F0 + models.F1 works!

    # Zero-order
    @property
    def F0(self) -> List[str]:
        """Zero-order (constant rate)."""
        return ['F0']

    # Nth-order reactions
    @property
    def F1(self) -> List[str]:
        """First-order."""
        return ['F1']

    @property
    def F2(self) -> List[str]:
        """Second-order."""
        return ['F2']

    @property
    def F3(self) -> List[str]:
        """Third-order."""
        return ['F3']

    # Avrami-Erofeev (nucleation and growth)
    @property
    def A2(self) -> List[str]:
        """Avrami-Erofeev, n=2."""
        return ['A2']

    @property
    def A3(self) -> List[str]:
        """Avrami-Erofeev, n=3."""
        return ['A3']

    # Contracting geometry
    @property
    def R2(self) -> List[str]:
        """Contracting area (2D)."""
        return ['R2']

    @property
    def R3(self) -> List[str]:
        """Contracting volume (3D)."""
        return ['R3']

    # Diffusion-controlled
    @property
    def D2(self) -> List[str]:
        """2D diffusion."""
        return ['D2']

    @property
    def D3(self) -> List[str]:
        """3D diffusion (Jander equation)."""
        return ['D3']

    # Autocatalytic
    @property
    def SB_mn(self) -> List[str]:
        """Sestak-Berggren (m=0.5, n=1.0)."""
        return ['SB_mn']

    @property
    def SB(self) -> List[str]:
        """Sestak-Berggren with fitted m,n (continuous optimization)."""
        return ['SB']

    @property
    def SB_grid(self) -> List[str]:
        """
        Sestak-Berggren grid search: all integer combinations m,n ∈ {0,1,2,3}.

        Returns 16 models with fixed integer m,n values. Recommended over continuous
        SB optimization for mechanistic interpretability and to prevent overfitting.

        Mechanistic interpretations:
        - SB_m0_n0: Zero-order (F0)
        - SB_m0_n1: First-order (F1)
        - SB_m0_n2: Second-order (F2)
        - SB_m1_n0: Power law/autocatalytic
        - SB_m0_n3: Third-order (F3)
        """
        from .models import generate_sb_grid_models
        return generate_sb_grid_models(m_range=range(4), n_range=range(4))

    @property
    def Bna(self) -> List[str]:
        """Prout-Tompkins (autocatalytic, c=1.0)."""
        return ['Bna']

    @property
    def SB2(self) -> List[str]:
        """Two parallel Sestak-Berggren steps with fitted m1,n1,m2,n2 (AKTS two-step form)."""
        return ['SB2']

    @property
    def SB2_grid(self) -> List[str]:
        """
        Two-step Sestak-Berggren grid: every unordered pair of integer SB steps,
        m,n in {0,1,2,3}, summed on one conversion. 136 models, each with 4 fitted
        parameters (Ea1, A1, Ea2, A2). Implemented like SB_grid, but ODE-integrated.
        """
        from .models import generate_sb2_grid_models
        return generate_sb2_grid_models(m_range=range(4), n_range=range(4))

    @property
    def all(self) -> List[str]:
        """All mechanistic kinetic models (excludes grid models to avoid duplication)."""
        return [
            'F0', 'F1', 'F2', 'F3',
            'A2', 'A3',
            'R2', 'R3',
            'D2', 'D3',
            'SB_mn', 'Bna'
        ]

    @property
    def nth_order(self) -> List[str]:
        """Nth-order reaction models (F0, F1, F2, F3)."""
        return ['F0', 'F1', 'F2', 'F3']

    @property
    def nucleation(self) -> List[str]:
        """Nucleation and growth models (Avrami-Erofeev)."""
        return ['A2', 'A3']

    @property
    def diffusion(self) -> List[str]:
        """Diffusion-controlled models."""
        return ['D2', 'D3']

    @property
    def autocatalytic(self) -> List[str]:
        """Autocatalytic models (Sestak-Berggren, Prout-Tompkins)."""
        return ['SB_mn', 'Bna']

    @property
    def all_with_sb_grid(self) -> List[str]:
        """
        All mechanistic models INCLUDING SB grid search (replaces standard models).

        Use this instead of `all` when you want comprehensive mechanistic screening
        with the SB(m,n) grid replacing F0-F3 with their SB equivalents plus additional
        autocatalytic models.

        Warning: Returns 20 models (12 standard + 8 unique SB grid).
        """
        # Exclude F0, F1, F2, F3 since they're covered by SB grid
        return [
            'A2', 'A3',
            'R2', 'R3',
            'D2', 'D3',
            'Bna'
        ] + self.SB_grid


class _ODEModels:
    """Multi-step ODE models (slower but more flexible)."""

    @property
    def consecutive(self) -> List[str]:
        """Consecutive reactions A→B→C."""
        return ['A->B->C']

    @property
    def bimolecular(self) -> List[str]:
        """Bimolecular reaction A+B→C."""
        return ['A+B->C']

    @property
    def all(self) -> List[str]:
        """All ODE models."""
        return ['A->B->C', 'A+B->C']


class _EmpiricalModels:
    """Empirical models with global Arrhenius fitting."""

    @property
    def First_Order(self) -> List[str]:
        """α = A·exp(-k(T)·t)"""
        return ['First_Order']

    @property
    def Linear(self) -> List[str]:
        """α = k(T)·t + C"""
        return ['Linear']

    @property
    def Sqrt(self) -> List[str]:
        """α = k(T)·√t + C"""
        return ['Sqrt']

    @property
    def Logistic(self) -> List[str]:
        """α = A/(1+B·exp(-k(T)·t))"""
        return ['Logistic']

    @property
    def Exponential(self) -> List[str]:
        """α = A·(1-exp(-k(T)·t))+C"""
        return ['Exponential']

    @property
    def all(self) -> List[str]:
        """All empirical models."""
        return ['First_Order', 'Linear', 'Sqrt', 'Logistic', 'Exponential']


class _ModelFree:
    """Model-free isoconversional methods."""

    @property
    def Friedman(self) -> List[str]:
        """Friedman differential method."""
        return ['Friedman']

    @property
    def all(self) -> List[str]:
        """All model-free methods."""
        return ['Friedman']


class _ModelSelector:
    """
    Structured model selector with IDE autocomplete support.

    All model attributes return lists for safe concatenation.
    Access models through dot notation for easy discovery:
    - models.kinetic.F1          → ['F1']
    - models.kinetic.all         → ['F0', 'F1', 'F2', ...]
    - models.empirical.Linear    → ['Linear']
    - models.ode.consecutive     → ['A->B->C']
    - models.modelfree.Friedman  → ['Friedman']
    - models.all                 → All models
    - models.mechanistic.all     → All mechanistic (kinetic + ODE)

    Examples
    --------
    >>> from akts import models
    >>> models_to_try = models.kinetic.all
    >>> models_to_try = models.kinetic.F1 + models.kinetic.A2  # Safe concatenation!
    >>> models_to_try = models.empirical.all + models.kinetic.F0  # Works!
    >>> models_to_try = models.all
    >>> models_to_try = models.mechanistic.all + models.empirical.all
    """

    def __init__(self):
        self.kinetic = _KineticModels()
        self.ode = _ODEModels()
        self.empirical = _EmpiricalModels()
        self.modelfree = _ModelFree()

    @property
    def mechanistic(self) -> _KineticModels:
        """
        Alias for kinetic models (mechanistic = kinetic in this context).

        Returns mechanistic solid-state kinetic models (F0-F3, A2-A3, etc.)
        Does NOT include ODE models - use models.ode for those.
        """
        return self.kinetic

    @property
    def all(self) -> List[str]:
        """
        All available models (kinetic + SB_grid + ODE + empirical + model-free).

        Includes BOTH standard models (F0-F3) AND SB_grid even though they overlap,
        because users expect 'all' to mean ALL models without hidden exclusions.

        Total: ~180 models including all variants (136 of them SB2_grid).

        Note: Some models are functionally equivalent:
        - F0 = SB_m0_n0 (zero-order)
        - F1 = SB_m0_n1 (first-order)
        - F2 = SB_m0_n2 (second-order)
        - F3 = SB_m0_n3 (third-order)
        """
        return (
            self.kinetic.all +
            self.kinetic.SB_grid +
            self.kinetic.SB2 +
            self.kinetic.SB2_grid +
            self.ode.all +
            self.empirical.all +
            self.modelfree.all
        )

    @property
    def default(self) -> List[str]:
        """
        Default models used by auto_model_isothermal_data() when models_to_try=None.

        Returns standard kinetic models (excludes ODE and empirical by default).
        """
        return self.kinetic.all


# Singleton instance for user import
models = _ModelSelector()


# For backward compatibility with direct string imports
__all__ = ['models', '_ModelSelector']
