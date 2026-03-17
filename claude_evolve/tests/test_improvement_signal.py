"""Tests for the continuous improvement signal G_t."""

import os
import json
import tempfile

from claude_evolve.core.improvement_signal import ImprovementSignal
from claude_evolve.core.stagnation import StagnationLevel


def test_update_increases_gt_on_improvement():
    sig = ImprovementSignal()
    sig.update(child_score=0.9, parent_score=0.5, island_id=0)
    assert sig.g_t > 0.0


def test_update_no_change_on_regression():
    sig = ImprovementSignal()
    sig.update(child_score=0.3, parent_score=0.5, island_id=0)
    assert sig.g_t == 0.0  # delta clamped to 0


def test_exploration_intensity_high_when_stagnant():
    sig = ImprovementSignal()
    # g_t = 0.0 (no improvement) -> max exploration
    assert sig.exploration_intensity >= 0.65


def test_exploration_intensity_low_when_improving():
    sig = ImprovementSignal()
    for _ in range(10):
        sig.update(0.9, 0.5, 0)  # Strong improvement
    assert sig.exploration_intensity < 0.3


def test_meta_guidance_triggers_when_all_islands_stagnant():
    sig = ImprovementSignal()
    sig.per_island_g_t = {0: 0.01, 1: 0.05, 2: 0.02}
    sig.meta_threshold = 0.12
    assert sig.should_trigger_meta_guidance()


def test_meta_guidance_no_trigger_when_one_island_active():
    sig = ImprovementSignal()
    sig.per_island_g_t = {0: 0.5, 1: 0.05, 2: 0.02}
    sig.meta_threshold = 0.12
    assert not sig.should_trigger_meta_guidance()


def test_derive_stagnation_level():
    sig = ImprovementSignal()
    sig.g_t = 0.2
    assert sig.derive_stagnation_level() == StagnationLevel.NONE
    sig.g_t = 0.001
    assert sig.derive_stagnation_level() == StagnationLevel.CRITICAL


def test_save_and_load_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'signal.json')
        sig = ImprovementSignal(g_t=0.42, rho=0.9)
        sig.per_island_g_t = {"0": 0.3, "1": 0.1}
        sig.save(path)
        loaded = ImprovementSignal.load(path)
        assert abs(loaded.g_t - 0.42) < 1e-10
        assert loaded.per_island_g_t == {"0": 0.3, "1": 0.1}


# --- Additional tests beyond the plan ---


def test_ema_decay_reduces_gt_over_neutral_updates():
    """When child == parent repeatedly, g_t should decay toward zero via EMA."""
    sig = ImprovementSignal(rho=0.9)
    # Inject a strong initial signal
    sig.update(child_score=1.0, parent_score=0.5, island_id=0)
    initial_gt = sig.g_t
    assert initial_gt > 0.0
    # Now submit many neutral updates (child == parent -> delta=0)
    for _ in range(50):
        sig.update(child_score=0.5, parent_score=0.5, island_id=0)
    assert sig.g_t < initial_gt * 0.01  # Should have decayed significantly


def test_per_island_tracks_independently():
    """Each island's g_t should reflect only that island's updates."""
    sig = ImprovementSignal(rho=0.5)
    # Island 0 gets improvement, island 1 gets regression
    sig.update(child_score=0.9, parent_score=0.5, island_id=0)
    sig.update(child_score=0.3, parent_score=0.5, island_id=1)
    assert sig.per_island_g_t["0"] > 0.0
    assert sig.per_island_g_t["1"] == 0.0


def test_load_nonexistent_returns_defaults():
    """Loading from a path that does not exist returns a default signal."""
    loaded = ImprovementSignal.load("/nonexistent/path/signal.json")
    assert loaded.g_t == 0.0
    assert loaded.rho == 0.95
    assert loaded.per_island_g_t == {}


def test_derive_stagnation_level_all_thresholds():
    """Verify all five stagnation level boundaries."""
    sig = ImprovementSignal()

    sig.g_t = 0.15
    assert sig.derive_stagnation_level() == StagnationLevel.NONE

    sig.g_t = 0.07
    assert sig.derive_stagnation_level() == StagnationLevel.MILD

    sig.g_t = 0.03
    assert sig.derive_stagnation_level() == StagnationLevel.MODERATE

    sig.g_t = 0.01
    assert sig.derive_stagnation_level() == StagnationLevel.SEVERE

    sig.g_t = 0.002
    assert sig.derive_stagnation_level() == StagnationLevel.CRITICAL


def test_exploration_intensity_bounds():
    """Exploration intensity should always be within [i_min, i_max]."""
    sig = ImprovementSignal(i_min=0.1, i_max=0.7)
    # At g_t=0 -> max intensity
    assert abs(sig.exploration_intensity - 0.7) < 1e-10
    # Drive g_t very high
    sig.g_t = 100.0
    intensity = sig.exploration_intensity
    assert intensity >= sig.i_min
    assert intensity <= sig.i_max


def test_meta_guidance_with_empty_islands_uses_global_gt():
    """When no per-island data exists, should_trigger_meta_guidance uses global g_t."""
    sig = ImprovementSignal(meta_threshold=0.12)
    sig.per_island_g_t = {}
    sig.g_t = 0.05
    assert sig.should_trigger_meta_guidance()
    sig.g_t = 0.5
    assert not sig.should_trigger_meta_guidance()


def test_save_preserves_custom_parameters():
    """Save/load should preserve all custom rho, i_min, i_max, meta_threshold."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'signal.json')
        sig = ImprovementSignal(g_t=0.1, rho=0.8, i_min=0.2, i_max=0.9, meta_threshold=0.05)
        sig.save(path)
        loaded = ImprovementSignal.load(path)
        assert abs(loaded.rho - 0.8) < 1e-10
        assert abs(loaded.i_min - 0.2) < 1e-10
        assert abs(loaded.i_max - 0.9) < 1e-10
        assert abs(loaded.meta_threshold - 0.05) < 1e-10


def test_update_with_zero_parent_score():
    """When parent_score is 0, division should use 1e-10 floor, not crash."""
    sig = ImprovementSignal()
    sig.update(child_score=0.5, parent_score=0.0, island_id=0)
    # delta = (0.5 - 0.0) / 1e-10 = huge number, but no crash
    assert sig.g_t > 0.0
