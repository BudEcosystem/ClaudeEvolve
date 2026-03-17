"""
Tests for claude_evolve.core.ucb_selector.

Covers UCB1 strategy selection: unvisited-arm priority, reward-based
exploitation, capped rewards, decay mechanics, exploration modulation,
persistence round-trip, edge cases, and deterministic behavior.
"""

import json
import math
import os
import tempfile

from claude_evolve.core.ucb_selector import StrategyArm, UCBStrategySelector


# ---------------------------------------------------------------------------
# StrategyArm dataclass
# ---------------------------------------------------------------------------


class TestStrategyArm:
    """Tests for the StrategyArm dataclass."""

    def test_defaults(self):
        arm = StrategyArm(strategy_id="x")
        assert arm.strategy_id == "x"
        assert arm.total_reward == 0.0
        assert arm.visit_count == 0
        assert arm.decayed_reward == 0.0

    def test_custom_values(self):
        arm = StrategyArm(strategy_id="y", total_reward=1.5, visit_count=3, decayed_reward=0.8)
        assert arm.total_reward == 1.5
        assert arm.visit_count == 3
        assert arm.decayed_reward == 0.8


# ---------------------------------------------------------------------------
# UCBStrategySelector -- unvisited arm priority
# ---------------------------------------------------------------------------


def test_unvisited_arms_selected_first():
    """Unvisited arms must be selected before any visited arm."""
    sel = UCBStrategySelector(["a", "b", "c"])
    first = sel.select()
    assert first in ["a", "b", "c"]
    sel.record(first, 0.1)
    second = sel.select()
    assert second != first  # Should pick unvisited


def test_all_unvisited_visited_in_order():
    """With three unvisited arms, the first three selects should cover all."""
    sel = UCBStrategySelector(["x", "y", "z"])
    visited = set()
    for _ in range(3):
        arm_id = sel.select()
        visited.add(arm_id)
        sel.record(arm_id, 0.0)
    assert visited == {"x", "y", "z"}


# ---------------------------------------------------------------------------
# UCBStrategySelector -- exploitation after exploration
# ---------------------------------------------------------------------------


def test_high_reward_arm_preferred_after_exploration():
    """After all arms visited, the one with consistently highest reward
    should dominate under low exploration."""
    sel = UCBStrategySelector(["a", "b", "c"], decay=1.0)  # No decay so reward persists
    # Visit all arms once with zero reward
    for sid in ["a", "b", "c"]:
        sel.select()
        sel.record(sid, 0.0)
    # Give "b" several big rewards to build a clear advantage
    for _ in range(5):
        sel.record("b", 0.8)
    # After enough visits, b should be preferred
    counts = {"a": 0, "b": 0, "c": 0}
    for _ in range(60):
        s = sel.select(exploration_intensity=0.05)  # Very low exploration
        counts[s] += 1
        sel.record(s, 0.0)
    assert counts["b"] > counts["a"]
    assert counts["b"] > counts["c"]


# ---------------------------------------------------------------------------
# UCBStrategySelector -- reward capping
# ---------------------------------------------------------------------------


def test_reward_is_capped_at_one():
    """A large score_delta should be capped at 1.0."""
    sel = UCBStrategySelector(["a"])
    sel.record("a", 5.0)  # Large delta
    assert sel.arms["a"].total_reward <= 1.0


def test_negative_reward_floored_to_zero():
    """Negative score_delta should be treated as 0.0 reward."""
    sel = UCBStrategySelector(["a"])
    sel.record("a", -3.0)
    assert sel.arms["a"].total_reward == 0.0
    assert sel.arms["a"].decayed_reward == 0.0


# ---------------------------------------------------------------------------
# UCBStrategySelector -- decay
# ---------------------------------------------------------------------------


def test_decay_reduces_old_rewards():
    """Decay should reduce the decayed_reward of an arm over successive records."""
    sel = UCBStrategySelector(["a"], decay=0.5)
    sel.record("a", 0.8)
    r1 = sel.arms["a"].decayed_reward
    sel.record("a", 0.0)  # No new reward, but decay applied
    r2 = sel.arms["a"].decayed_reward
    assert r2 < r1


def test_decay_affects_all_arms():
    """When recording for one arm, decay should reduce all arms' decayed_reward."""
    sel = UCBStrategySelector(["a", "b"], decay=0.5)
    sel.record("a", 1.0)
    assert sel.arms["a"].decayed_reward == 1.0
    # Now record for "b" -- "a" should decay
    sel.record("b", 0.0)
    # arm "a" decayed_reward was 1.0 * 0.5 = 0.5
    assert abs(sel.arms["a"].decayed_reward - 0.5) < 1e-9


# ---------------------------------------------------------------------------
# UCBStrategySelector -- exploration modulation
# ---------------------------------------------------------------------------


def test_exploration_intensity_modulates_selection():
    """Higher exploration intensity should increase UCB bonus, causing more
    diversity in arm selection over many rounds."""
    sel_low = UCBStrategySelector(["a", "b"], c=1.414, decay=1.0)
    sel_high = UCBStrategySelector(["a", "b"], c=1.414, decay=1.0)

    # Bootstrap both selectors identically: visit both arms, give "a" reward
    for s in [sel_low, sel_high]:
        s.select()
        s.record("a", 0.8)
        s.select()
        s.record("b", 0.0)

    # Run 50 rounds with low vs high exploration
    counts_low = {"a": 0, "b": 0}
    counts_high = {"a": 0, "b": 0}
    for _ in range(50):
        choice_low = sel_low.select(exploration_intensity=0.05)
        counts_low[choice_low] += 1
        sel_low.record(choice_low, 0.0)

        choice_high = sel_high.select(exploration_intensity=2.0)
        counts_high[choice_high] += 1
        sel_high.record(choice_high, 0.0)

    # High exploration should select "b" more often than low exploration
    assert counts_high["b"] >= counts_low["b"]


# ---------------------------------------------------------------------------
# UCBStrategySelector -- persistence (save/load)
# ---------------------------------------------------------------------------


def test_save_and_load_roundtrip():
    """Saved state should be fully restored on load."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'ucb.json')
        sel = UCBStrategySelector(["a", "b"])
        sel.record("a", 0.3)
        sel.save(path)
        loaded = UCBStrategySelector.load(path)
        assert loaded.arms["a"].visit_count == 1
        assert loaded.arms["a"].total_reward > 0
        assert abs(loaded.c - sel.c) < 1e-9
        assert abs(loaded.decay - sel.decay) < 1e-9
        assert loaded.total_selections == sel.total_selections


def test_load_nonexistent_returns_empty():
    """Loading from a non-existent path should return a selector with no arms."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'does_not_exist.json')
        sel = UCBStrategySelector.load(path)
        assert len(sel.arms) == 0


def test_save_creates_parent_directories():
    """save() should create intermediate directories as needed."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'nested', 'deep', 'ucb.json')
        sel = UCBStrategySelector(["a"])
        sel.record("a", 0.5)
        sel.save(path)
        assert os.path.isfile(path)
        loaded = UCBStrategySelector.load(path)
        assert loaded.arms["a"].visit_count == 1


def test_save_load_preserves_decayed_reward():
    """The decayed_reward field must survive a save/load cycle."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'ucb.json')
        sel = UCBStrategySelector(["a", "b"], decay=0.9)
        sel.record("a", 0.7)
        sel.record("b", 0.3)
        sel.record("a", 0.2)
        original_a = sel.arms["a"].decayed_reward
        original_b = sel.arms["b"].decayed_reward
        sel.save(path)
        loaded = UCBStrategySelector.load(path)
        assert abs(loaded.arms["a"].decayed_reward - original_a) < 1e-9
        assert abs(loaded.arms["b"].decayed_reward - original_b) < 1e-9


# ---------------------------------------------------------------------------
# UCBStrategySelector -- edge cases
# ---------------------------------------------------------------------------


def test_single_arm_always_selected():
    """With only one arm, it should always be selected."""
    sel = UCBStrategySelector(["only"])
    for _ in range(10):
        assert sel.select() == "only"
        sel.record("only", 0.1)


def test_record_unknown_strategy_is_noop():
    """Recording for a strategy_id not in arms should not crash or mutate state."""
    sel = UCBStrategySelector(["a", "b"])
    sel.record("nonexistent", 0.5)
    # Arms unchanged
    assert sel.arms["a"].visit_count == 0
    assert sel.arms["b"].visit_count == 0


def test_total_selections_increments():
    """Each call to select() should increment total_selections."""
    sel = UCBStrategySelector(["a", "b"])
    assert sel.total_selections == 0
    sel.select()
    assert sel.total_selections == 1
    sel.record("a", 0.0)
    sel.select()
    assert sel.total_selections == 2
