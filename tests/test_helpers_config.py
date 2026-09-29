"""
Tests for small additive features in akts.helpers: exposing initial_ratio_r for
A+B->C, and including SB_mn/Bna in the default isothermal model list
(TODO.md §6b "Models AKTS offers that akts lacks").
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from akts.helpers import (
    _setup_model_configs, DEFAULT_ISOTHERMAL_MODELS, MODEL_DISPLAY_NAMES,
    _physical_sanity_flags,
)
from akts.core import _get_bootstrap_max_workers


class TestBootstrapWorkerCount:
    def test_default_leaves_one_cpu_core_free(self, monkeypatch):
        monkeypatch.setattr('os.cpu_count', lambda: 12)
        assert _get_bootstrap_max_workers(-1, 100) == 11

    def test_default_is_capped_by_replicate_count(self, monkeypatch):
        monkeypatch.setattr('os.cpu_count', lambda: 12)
        assert _get_bootstrap_max_workers(-1, 3) == 3

    def test_explicit_worker_count_is_capped_by_replicate_count(self):
        assert _get_bootstrap_max_workers(8, 2) == 2


class TestInitialRatioR:
    def test_default_ratio_is_stoichiometric(self):
        cfg, _, _ = _setup_model_configs(['A+B->C'])
        assert cfg[0]['def_args']['bimol_params']['initial_ratio_r'] == 1.0

    def test_custom_ratio_flows_through(self):
        cfg, _, _ = _setup_model_configs(['A+B->C'], initial_ratio_r=2.5)
        assert cfg[0]['def_args']['bimol_params']['initial_ratio_r'] == 2.5

    def test_other_models_unaffected_by_ratio_param(self):
        # initial_ratio_r is only meaningful for A+B->C; other models must ignore it.
        cfg, guesses, bounds = _setup_model_configs(['F1', 'A2'], initial_ratio_r=3.0)
        for model in cfg:
            assert model['type'] != 'A+B->C'
            assert 'bimol_params' not in model['def_args']


class TestDefaultModelListIncludesAutocatalytic:
    def test_sb_mn_and_bna_in_default_list(self):
        assert 'SB_mn' in DEFAULT_ISOTHERMAL_MODELS
        assert 'Bna' in DEFAULT_ISOTHERMAL_MODELS

    def test_sb_mn_and_bna_have_display_names(self):
        assert 'SB_mn' in MODEL_DISPLAY_NAMES
        assert 'Bna' in MODEL_DISPLAY_NAMES

    def test_sb_mn_and_bna_produce_valid_single_step_configs(self):
        cfg, guesses, bounds = _setup_model_configs(['SB_mn', 'Bna'])
        names = {c['name']: c for c in cfg}
        assert names['SB_mn_model']['type'] == 'single_step'
        assert names['SB_mn_model']['def_args']['f_alpha_model'] == 'SB_mn'
        assert names['Bna_model']['type'] == 'single_step'
        assert names['Bna_model']['def_args']['f_alpha_model'] == 'Bna'
        # Same generic Ea/A guesses and bounds as any other single_step model.
        assert set(guesses['SB_mn_model'].keys()) == {'Ea', 'A'}
        assert set(bounds['SB_mn_model'].keys()) == {'Ea', 'A'}


class TestDefaultModelListIncludesF0:
    def test_f0_in_default_list(self):
        assert 'F0' in DEFAULT_ISOTHERMAL_MODELS

    def test_f0_has_display_name(self):
        assert 'F0' in MODEL_DISPLAY_NAMES

    def test_f0_produces_valid_single_step_config(self):
        cfg, guesses, bounds = _setup_model_configs(['F0'])
        assert cfg[0]['type'] == 'single_step'
        assert cfg[0]['def_args']['f_alpha_model'] == 'F0'
        assert set(guesses['F0_model'].keys()) == {'Ea', 'A'}


class TestPhysicalSanityFlags:
    def test_no_flags_for_typical_ea(self):
        assert _physical_sanity_flags({'Ea': 90000, 'A': 1e11}) == []

    def test_flags_low_ea(self):
        flags = _physical_sanity_flags({'Ea': 20000, 'A': 1e11})
        assert len(flags) == 1
        assert 'below' in flags[0]

    def test_flags_high_ea(self):
        flags = _physical_sanity_flags({'Ea': 250000, 'A': 1e11})
        assert len(flags) == 1
        assert 'above' in flags[0]

    def test_checks_every_ea_parameter_in_multistep_models(self):
        # A->B->C has Ea1/Ea2 -- both should be checked independently.
        flags = _physical_sanity_flags({'Ea1': 90000, 'Ea2': 250000, 'A1': 1e11, 'A2': 1e12})
        assert len(flags) == 1
        assert 'Ea2' in flags[0]

    def test_non_ea_parameters_ignored(self):
        # A (pre-exponential factor) and other non-Ea params must not trigger flags.
        assert _physical_sanity_flags({'A': 1e30, 'initial_ratio_r': 50.0}) == []

    def test_boundary_values_not_flagged(self):
        # Exactly at the documented 30-180 kJ/mol boundary should not flag.
        assert _physical_sanity_flags({'Ea': 30000}) == []
        assert _physical_sanity_flags({'Ea': 180000}) == []
