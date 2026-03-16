"""
Tests for the cross-run memory module.

Covers Learning dataclass serialization, CrossRunMemory initialization,
load/save round-trips, querying, prompt formatting, and limit enforcement.
"""

import json
import os
import time

import pytest

from claude_evolve.core.memory import CrossRunMemory, Learning


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_learning(
    id: str = "learn-1",
    run_id: str = "run-abc",
    iteration: int = 5,
    category: str = "insight",
    description: str = "Test learning",
    score_before: float = 0.5,
    score_after: float = 0.7,
    score_delta: float = 0.2,
    metadata: dict = None,
) -> Learning:
    """Create a Learning with sensible defaults for testing."""
    return Learning(
        id=id,
        run_id=run_id,
        iteration=iteration,
        timestamp=time.time(),
        category=category,
        description=description,
        score_before=score_before,
        score_after=score_after,
        score_delta=score_delta,
        metadata=metadata or {},
    )


# ---------------------------------------------------------------------------
# Learning dataclass creation and serialization
# ---------------------------------------------------------------------------

class TestLearningCreation:
    """Test Learning dataclass instantiation."""

    def test_create_with_required_fields(self):
        l = Learning(
            id="l1",
            run_id="r1",
            iteration=1,
            timestamp=1000.0,
            category="success",
            description="First success",
        )
        assert l.id == "l1"
        assert l.run_id == "r1"
        assert l.iteration == 1
        assert l.timestamp == 1000.0
        assert l.category == "success"
        assert l.description == "First success"

    def test_optional_fields_default_to_none(self):
        l = Learning(
            id="l2",
            run_id="r1",
            iteration=2,
            timestamp=1001.0,
            category="failure",
            description="A failure",
        )
        assert l.score_before is None
        assert l.score_after is None
        assert l.score_delta is None
        assert l.metadata == {}

    def test_create_with_all_fields(self):
        l = Learning(
            id="l3",
            run_id="r2",
            iteration=10,
            timestamp=2000.0,
            category="approach",
            description="New approach",
            score_before=0.3,
            score_after=0.8,
            score_delta=0.5,
            metadata={"technique": "gradient"},
        )
        assert l.score_before == 0.3
        assert l.score_after == 0.8
        assert l.score_delta == 0.5
        assert l.metadata["technique"] == "gradient"


class TestLearningSerialization:
    """Test Learning.to_dict() and Learning.from_dict()."""

    def test_to_dict_returns_all_fields(self):
        l = _make_learning()
        d = l.to_dict()
        assert d["id"] == "learn-1"
        assert d["run_id"] == "run-abc"
        assert d["iteration"] == 5
        assert d["category"] == "insight"
        assert d["description"] == "Test learning"
        assert d["score_before"] == 0.5
        assert d["score_after"] == 0.7
        assert d["score_delta"] == 0.2
        assert isinstance(d["timestamp"], float)
        assert isinstance(d["metadata"], dict)

    def test_from_dict_required_fields(self):
        d = {
            "id": "x1",
            "run_id": "r1",
            "iteration": 3,
            "timestamp": 999.0,
            "category": "failure",
            "description": "Failed attempt",
        }
        l = Learning.from_dict(d)
        assert l.id == "x1"
        assert l.category == "failure"
        assert l.score_before is None

    def test_from_dict_ignores_extra_keys(self):
        d = {
            "id": "x2",
            "run_id": "r1",
            "iteration": 1,
            "timestamp": 100.0,
            "category": "insight",
            "description": "Has extras",
            "unknown_field": "should be ignored",
            "another_extra": 42,
        }
        l = Learning.from_dict(d)
        assert l.id == "x2"
        assert not hasattr(l, "unknown_field")

    def test_roundtrip_to_dict_from_dict(self):
        original = _make_learning(
            id="rt-1",
            run_id="run-rt",
            iteration=7,
            category="success",
            description="Roundtrip test",
            score_before=0.1,
            score_after=0.9,
            score_delta=0.8,
            metadata={"key": "value"},
        )
        d = original.to_dict()
        restored = Learning.from_dict(d)
        assert restored.id == original.id
        assert restored.run_id == original.run_id
        assert restored.iteration == original.iteration
        assert restored.category == original.category
        assert restored.description == original.description
        assert restored.score_before == original.score_before
        assert restored.score_after == original.score_after
        assert restored.score_delta == original.score_delta
        assert restored.metadata == original.metadata
        # Timestamp is float so compare with tolerance
        assert abs(restored.timestamp - original.timestamp) < 0.001

    def test_to_dict_json_serializable(self):
        """to_dict output must be JSON-serializable."""
        l = _make_learning(metadata={"nested": {"a": 1}})
        d = l.to_dict()
        serialized = json.dumps(d)
        assert isinstance(serialized, str)
        deserialized = json.loads(serialized)
        assert deserialized["id"] == l.id


# ---------------------------------------------------------------------------
# CrossRunMemory initialization
# ---------------------------------------------------------------------------

class TestCrossRunMemoryInit:
    """Test CrossRunMemory constructor defaults."""

    def test_default_initialization(self, tmp_path):
        mem = CrossRunMemory(str(tmp_path / "memory"))
        assert mem.memory_dir == str(tmp_path / "memory")
        assert mem.max_learnings == 100
        assert mem.max_failed_approaches == 50
        assert mem.learnings == []
        assert mem.failed_approaches == []
        assert mem.strategies == []
        assert mem._loaded is False

    def test_custom_limits(self, tmp_path):
        mem = CrossRunMemory(str(tmp_path / "mem"), max_learnings=10, max_failed_approaches=5)
        assert mem.max_learnings == 10
        assert mem.max_failed_approaches == 5


# ---------------------------------------------------------------------------
# load() from empty directory
# ---------------------------------------------------------------------------

class TestCrossRunMemoryLoadEmpty:
    """Test loading when the memory directory does not exist or is empty."""

    def test_load_creates_directory(self, tmp_path):
        mem_dir = str(tmp_path / "nonexistent" / "deep" / "memory")
        mem = CrossRunMemory(mem_dir)
        mem.load()
        assert os.path.isdir(mem_dir)

    def test_load_empty_directory(self, tmp_path):
        mem_dir = str(tmp_path / "empty_mem")
        os.makedirs(mem_dir)
        mem = CrossRunMemory(mem_dir)
        mem.load()
        assert mem.learnings == []
        assert mem.failed_approaches == []
        assert mem.strategies == []
        assert mem._loaded is True


# ---------------------------------------------------------------------------
# save() and load() round-trip
# ---------------------------------------------------------------------------

class TestCrossRunMemoryRoundTrip:
    """Test that save() followed by load() preserves all data."""

    def test_save_creates_files(self, tmp_path):
        mem_dir = str(tmp_path / "mem")
        mem = CrossRunMemory(mem_dir)
        mem.save()
        assert os.path.exists(os.path.join(mem_dir, "learnings.json"))
        assert os.path.exists(os.path.join(mem_dir, "failed_approaches.json"))
        assert os.path.exists(os.path.join(mem_dir, "strategies.json"))

    def test_roundtrip_learnings(self, tmp_path):
        mem_dir = str(tmp_path / "mem")
        mem = CrossRunMemory(mem_dir)
        l1 = _make_learning(id="rt-l1", category="success", description="Worked well")
        l2 = _make_learning(id="rt-l2", category="failure", description="Did not work")
        mem.add_learning(l1)
        mem.add_learning(l2)
        mem.save()

        mem2 = CrossRunMemory(mem_dir)
        mem2.load()
        assert len(mem2.learnings) == 2
        assert mem2.learnings[0].id == "rt-l1"
        assert mem2.learnings[1].id == "rt-l2"
        assert mem2.learnings[0].category == "success"
        assert mem2.learnings[1].category == "failure"

    def test_roundtrip_failed_approaches(self, tmp_path):
        mem_dir = str(tmp_path / "mem")
        mem = CrossRunMemory(mem_dir)
        mem.add_failed_approach("Bad approach A", score=0.1, iteration=3, run_id="r1")
        mem.add_failed_approach("Bad approach B", score=0.2, iteration=5, run_id="r2",
                                metadata={"reason": "timeout"})
        mem.save()

        mem2 = CrossRunMemory(mem_dir)
        mem2.load()
        assert len(mem2.failed_approaches) == 2
        assert mem2.failed_approaches[0]["description"] == "Bad approach A"
        assert mem2.failed_approaches[1]["metadata"]["reason"] == "timeout"

    def test_roundtrip_strategies(self, tmp_path):
        mem_dir = str(tmp_path / "mem")
        mem = CrossRunMemory(mem_dir)
        mem.add_strategy("Greedy", "Use greedy selection", score=0.85, run_id="r1")
        mem.add_strategy("Random", "Use random mutations", score=0.6, run_id="r2")
        mem.save()

        mem2 = CrossRunMemory(mem_dir)
        mem2.load()
        assert len(mem2.strategies) == 2
        assert mem2.strategies[0]["name"] == "Greedy"
        assert mem2.strategies[0]["score"] == 0.85
        assert mem2.strategies[1]["name"] == "Random"

    def test_roundtrip_all_combined(self, tmp_path):
        """Verify all three collections survive a full save/load cycle."""
        mem_dir = str(tmp_path / "mem")
        mem = CrossRunMemory(mem_dir)
        mem.add_learning(_make_learning(id="combined-1"))
        mem.add_failed_approach("fail-1", score=0.0, iteration=1, run_id="r1")
        mem.add_strategy("strat-1", "desc", score=0.9, run_id="r1")
        mem.save()

        mem2 = CrossRunMemory(mem_dir)
        mem2.load()
        assert len(mem2.learnings) == 1
        assert len(mem2.failed_approaches) == 1
        assert len(mem2.strategies) == 1


# ---------------------------------------------------------------------------
# add_learning, add_failed_approach, add_strategy
# ---------------------------------------------------------------------------

class TestCrossRunMemoryAdd:
    """Test adding entries to memory."""

    def test_add_learning(self, tmp_path):
        mem = CrossRunMemory(str(tmp_path / "mem"))
        l = _make_learning(id="add-1")
        mem.add_learning(l)
        assert len(mem.learnings) == 1
        assert mem.learnings[0].id == "add-1"

    def test_add_multiple_learnings(self, tmp_path):
        mem = CrossRunMemory(str(tmp_path / "mem"))
        for i in range(5):
            mem.add_learning(_make_learning(id=f"multi-{i}"))
        assert len(mem.learnings) == 5

    def test_add_failed_approach(self, tmp_path):
        mem = CrossRunMemory(str(tmp_path / "mem"))
        mem.add_failed_approach("Too aggressive pruning", score=0.15, iteration=10, run_id="r1")
        assert len(mem.failed_approaches) == 1
        fa = mem.failed_approaches[0]
        assert fa["description"] == "Too aggressive pruning"
        assert fa["score"] == 0.15
        assert fa["iteration"] == 10
        assert fa["run_id"] == "r1"
        assert "timestamp" in fa
        assert fa["metadata"] == {}

    def test_add_failed_approach_with_metadata(self, tmp_path):
        mem = CrossRunMemory(str(tmp_path / "mem"))
        mem.add_failed_approach("Overflow", score=0.0, iteration=1, run_id="r2",
                                metadata={"error": "stack overflow"})
        assert mem.failed_approaches[0]["metadata"]["error"] == "stack overflow"

    def test_add_strategy(self, tmp_path):
        mem = CrossRunMemory(str(tmp_path / "mem"))
        mem.add_strategy("Hill Climbing", "Iterative hill climbing", score=0.92, run_id="r5")
        assert len(mem.strategies) == 1
        s = mem.strategies[0]
        assert s["name"] == "Hill Climbing"
        assert s["description"] == "Iterative hill climbing"
        assert s["score"] == 0.92
        assert s["run_id"] == "r5"
        assert "timestamp" in s
        assert s["metadata"] == {}

    def test_add_strategy_with_metadata(self, tmp_path):
        mem = CrossRunMemory(str(tmp_path / "mem"))
        mem.add_strategy("SA", "Simulated annealing", score=0.88, run_id="r3",
                         metadata={"temperature": 0.5})
        assert mem.strategies[0]["metadata"]["temperature"] == 0.5


# ---------------------------------------------------------------------------
# get_failed_approaches with limit
# ---------------------------------------------------------------------------

class TestGetFailedApproaches:
    """Test get_failed_approaches with limit parameter."""

    def test_empty_returns_empty(self, tmp_path):
        mem = CrossRunMemory(str(tmp_path / "mem"))
        assert mem.get_failed_approaches() == []

    def test_returns_all_when_under_limit(self, tmp_path):
        mem = CrossRunMemory(str(tmp_path / "mem"))
        for i in range(3):
            mem.add_failed_approach(f"fail-{i}", score=0.1 * i, iteration=i, run_id="r1")
        result = mem.get_failed_approaches(limit=10)
        assert len(result) == 3

    def test_returns_most_recent_when_over_limit(self, tmp_path):
        mem = CrossRunMemory(str(tmp_path / "mem"))
        for i in range(10):
            mem.add_failed_approach(f"fail-{i}", score=float(i), iteration=i, run_id="r1")
        result = mem.get_failed_approaches(limit=3)
        assert len(result) == 3
        # Should be the last 3 entries (most recent)
        assert result[0]["description"] == "fail-7"
        assert result[1]["description"] == "fail-8"
        assert result[2]["description"] == "fail-9"

    def test_default_limit_is_ten(self, tmp_path):
        mem = CrossRunMemory(str(tmp_path / "mem"))
        for i in range(15):
            mem.add_failed_approach(f"fail-{i}", score=0.0, iteration=i, run_id="r1")
        result = mem.get_failed_approaches()
        assert len(result) == 10

    def test_limit_one(self, tmp_path):
        mem = CrossRunMemory(str(tmp_path / "mem"))
        for i in range(5):
            mem.add_failed_approach(f"fail-{i}", score=0.0, iteration=i, run_id="r1")
        result = mem.get_failed_approaches(limit=1)
        assert len(result) == 1
        assert result[0]["description"] == "fail-4"


# ---------------------------------------------------------------------------
# get_successful_strategies with min_score filter
# ---------------------------------------------------------------------------

class TestGetSuccessfulStrategies:
    """Test get_successful_strategies with min_score filtering."""

    def test_empty_returns_empty(self, tmp_path):
        mem = CrossRunMemory(str(tmp_path / "mem"))
        assert mem.get_successful_strategies() == []

    def test_returns_all_above_zero(self, tmp_path):
        mem = CrossRunMemory(str(tmp_path / "mem"))
        mem.add_strategy("A", "Desc A", score=0.5, run_id="r1")
        mem.add_strategy("B", "Desc B", score=0.8, run_id="r1")
        result = mem.get_successful_strategies(min_score=0.0)
        assert len(result) == 2

    def test_filters_below_min_score(self, tmp_path):
        mem = CrossRunMemory(str(tmp_path / "mem"))
        mem.add_strategy("Low", "Low scorer", score=0.3, run_id="r1")
        mem.add_strategy("Mid", "Mid scorer", score=0.6, run_id="r1")
        mem.add_strategy("High", "High scorer", score=0.9, run_id="r1")
        result = mem.get_successful_strategies(min_score=0.5)
        assert len(result) == 2
        names = [s["name"] for s in result]
        assert "Low" not in names
        assert "Mid" in names
        assert "High" in names

    def test_exact_min_score_included(self, tmp_path):
        mem = CrossRunMemory(str(tmp_path / "mem"))
        mem.add_strategy("Exact", "Score at threshold", score=0.7, run_id="r1")
        result = mem.get_successful_strategies(min_score=0.7)
        assert len(result) == 1
        assert result[0]["name"] == "Exact"

    def test_high_min_score_returns_empty(self, tmp_path):
        mem = CrossRunMemory(str(tmp_path / "mem"))
        mem.add_strategy("A", "Desc", score=0.5, run_id="r1")
        result = mem.get_successful_strategies(min_score=0.99)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# get_learnings_by_category
# ---------------------------------------------------------------------------

class TestGetLearningsByCategory:
    """Test get_learnings_by_category filtering."""

    def test_empty_returns_empty(self, tmp_path):
        mem = CrossRunMemory(str(tmp_path / "mem"))
        assert mem.get_learnings_by_category("success") == []

    def test_filters_correct_category(self, tmp_path):
        mem = CrossRunMemory(str(tmp_path / "mem"))
        mem.add_learning(_make_learning(id="s1", category="success", description="Good"))
        mem.add_learning(_make_learning(id="f1", category="failure", description="Bad"))
        mem.add_learning(_make_learning(id="s2", category="success", description="Also good"))
        mem.add_learning(_make_learning(id="i1", category="insight", description="Interesting"))

        successes = mem.get_learnings_by_category("success")
        assert len(successes) == 2
        assert all(l.category == "success" for l in successes)

        failures = mem.get_learnings_by_category("failure")
        assert len(failures) == 1
        assert failures[0].id == "f1"

        insights = mem.get_learnings_by_category("insight")
        assert len(insights) == 1
        assert insights[0].id == "i1"

    def test_nonexistent_category_returns_empty(self, tmp_path):
        mem = CrossRunMemory(str(tmp_path / "mem"))
        mem.add_learning(_make_learning(id="x1", category="success"))
        assert mem.get_learnings_by_category("nonexistent") == []


# ---------------------------------------------------------------------------
# format_for_prompt
# ---------------------------------------------------------------------------

class TestFormatForPrompt:
    """Test format_for_prompt output."""

    def test_empty_memory_returns_empty_string(self, tmp_path):
        mem = CrossRunMemory(str(tmp_path / "mem"))
        assert mem.format_for_prompt() == ""

    def test_includes_failed_approaches_section(self, tmp_path):
        mem = CrossRunMemory(str(tmp_path / "mem"))
        mem.add_failed_approach("Greedy fails", score=0.1, iteration=2, run_id="run-1")
        output = mem.format_for_prompt()
        assert "## Failed Approaches (Avoid These)" in output
        assert "Greedy fails" in output
        assert "0.1" in output
        assert "run-1" in output

    def test_includes_strategies_section(self, tmp_path):
        mem = CrossRunMemory(str(tmp_path / "mem"))
        mem.add_strategy("DP", "Dynamic programming approach", score=0.95, run_id="run-2")
        output = mem.format_for_prompt()
        assert "## Successful Strategies" in output
        assert "DP" in output
        assert "Dynamic programming approach" in output
        assert "0.95" in output

    def test_includes_insights_section(self, tmp_path):
        mem = CrossRunMemory(str(tmp_path / "mem"))
        mem.add_learning(_make_learning(id="ins-1", category="insight",
                                        description="Small mutations work better"))
        output = mem.format_for_prompt()
        assert "## Key Insights from Previous Runs" in output
        assert "Small mutations work better" in output

    def test_does_not_include_non_insight_learnings(self, tmp_path):
        mem = CrossRunMemory(str(tmp_path / "mem"))
        mem.add_learning(_make_learning(id="s1", category="success",
                                        description="Success only learning"))
        output = mem.format_for_prompt()
        # No insights => no insights section, and success learnings are not shown
        assert "Key Insights" not in output
        # With no failed approaches or strategies either, should be empty
        assert output == ""

    def test_strategies_sorted_by_score_descending(self, tmp_path):
        mem = CrossRunMemory(str(tmp_path / "mem"))
        mem.add_strategy("Low", "Low score", score=0.3, run_id="r1")
        mem.add_strategy("High", "High score", score=0.9, run_id="r1")
        mem.add_strategy("Mid", "Mid score", score=0.6, run_id="r1")
        output = mem.format_for_prompt()
        high_pos = output.index("High")
        mid_pos = output.index("Mid")
        low_pos = output.index("Low")
        assert high_pos < mid_pos < low_pos

    def test_max_items_limits_failed_approaches(self, tmp_path):
        mem = CrossRunMemory(str(tmp_path / "mem"))
        for i in range(10):
            mem.add_failed_approach(f"fail-{i}", score=0.0, iteration=i, run_id="r1")
        output = mem.format_for_prompt(max_items=3)
        # Should only show last 3
        assert "fail-7" in output
        assert "fail-8" in output
        assert "fail-9" in output
        assert "fail-0" not in output

    def test_max_items_limits_strategies(self, tmp_path):
        mem = CrossRunMemory(str(tmp_path / "mem"))
        for i in range(10):
            mem.add_strategy(f"strat-{i}", f"desc-{i}", score=0.1 * i, run_id="r1")
        output = mem.format_for_prompt(max_items=2)
        # Sorted by score desc, top 2 should be strat-9 (0.9) and strat-8 (0.8)
        assert "strat-9" in output
        assert "strat-8" in output
        # strat-0 (0.0) should not appear
        assert "strat-0" not in output

    def test_max_items_limits_insights(self, tmp_path):
        mem = CrossRunMemory(str(tmp_path / "mem"))
        for i in range(8):
            mem.add_learning(_make_learning(id=f"ins-{i}", category="insight",
                                            description=f"Insight number {i}"))
        output = mem.format_for_prompt(max_items=3)
        # Should show last 3 insights
        assert "Insight number 5" in output
        assert "Insight number 6" in output
        assert "Insight number 7" in output
        assert "Insight number 0" not in output

    def test_all_sections_combined(self, tmp_path):
        mem = CrossRunMemory(str(tmp_path / "mem"))
        mem.add_failed_approach("Bad idea", score=0.05, iteration=1, run_id="r1")
        mem.add_strategy("Good idea", "Works great", score=0.95, run_id="r1")
        mem.add_learning(_make_learning(id="ins-1", category="insight",
                                        description="Combine approaches"))
        output = mem.format_for_prompt()
        assert "## Failed Approaches (Avoid These)" in output
        assert "## Successful Strategies" in output
        assert "## Key Insights from Previous Runs" in output
        # Sections should be separated by double newlines
        assert "\n\n" in output


# ---------------------------------------------------------------------------
# _enforce_limits
# ---------------------------------------------------------------------------

class TestEnforceLimits:
    """Test _enforce_limits trims correctly."""

    def test_learnings_trimmed_to_max(self, tmp_path):
        mem = CrossRunMemory(str(tmp_path / "mem"), max_learnings=5)
        for i in range(10):
            mem.add_learning(_make_learning(id=f"trim-{i}"))
        mem._enforce_limits()
        assert len(mem.learnings) == 5
        # Should keep the most recent (last 5)
        assert mem.learnings[0].id == "trim-5"
        assert mem.learnings[4].id == "trim-9"

    def test_failed_approaches_trimmed_to_max(self, tmp_path):
        mem = CrossRunMemory(str(tmp_path / "mem"), max_failed_approaches=3)
        for i in range(8):
            mem.add_failed_approach(f"fail-{i}", score=0.0, iteration=i, run_id="r1")
        mem._enforce_limits()
        assert len(mem.failed_approaches) == 3
        # Should keep the most recent (last 3)
        assert mem.failed_approaches[0]["description"] == "fail-5"
        assert mem.failed_approaches[2]["description"] == "fail-7"

    def test_no_trim_when_under_limit(self, tmp_path):
        mem = CrossRunMemory(str(tmp_path / "mem"), max_learnings=10, max_failed_approaches=10)
        for i in range(5):
            mem.add_learning(_make_learning(id=f"ok-{i}"))
            mem.add_failed_approach(f"fail-{i}", score=0.0, iteration=i, run_id="r1")
        mem._enforce_limits()
        assert len(mem.learnings) == 5
        assert len(mem.failed_approaches) == 5

    def test_exact_limit_no_trim(self, tmp_path):
        mem = CrossRunMemory(str(tmp_path / "mem"), max_learnings=3)
        for i in range(3):
            mem.add_learning(_make_learning(id=f"exact-{i}"))
        mem._enforce_limits()
        assert len(mem.learnings) == 3

    def test_save_enforces_limits(self, tmp_path):
        """save() should call _enforce_limits before writing."""
        mem_dir = str(tmp_path / "mem")
        mem = CrossRunMemory(mem_dir, max_learnings=3, max_failed_approaches=2)
        for i in range(10):
            mem.add_learning(_make_learning(id=f"save-{i}"))
        for i in range(7):
            mem.add_failed_approach(f"fail-{i}", score=0.0, iteration=i, run_id="r1")
        mem.save()

        mem2 = CrossRunMemory(mem_dir)
        mem2.load()
        assert len(mem2.learnings) == 3
        assert len(mem2.failed_approaches) == 2
        # Verify most recent are kept
        assert mem2.learnings[0].id == "save-7"
        assert mem2.failed_approaches[0]["description"] == "fail-5"

    def test_strategies_not_trimmed(self, tmp_path):
        """Strategies have no configured limit and should not be trimmed."""
        mem = CrossRunMemory(str(tmp_path / "mem"), max_learnings=2, max_failed_approaches=2)
        for i in range(20):
            mem.add_strategy(f"strat-{i}", f"desc-{i}", score=0.1 * i, run_id="r1")
        mem._enforce_limits()
        assert len(mem.strategies) == 20


# ---------------------------------------------------------------------------
# Edge cases and integration
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge cases and integration scenarios."""

    def test_save_to_nested_nonexistent_directory(self, tmp_path):
        mem_dir = str(tmp_path / "a" / "b" / "c" / "memory")
        mem = CrossRunMemory(mem_dir)
        mem.add_learning(_make_learning(id="nested-1"))
        mem.save()
        assert os.path.isdir(mem_dir)

        mem2 = CrossRunMemory(mem_dir)
        mem2.load()
        assert len(mem2.learnings) == 1

    def test_multiple_save_load_cycles(self, tmp_path):
        mem_dir = str(tmp_path / "mem")
        mem = CrossRunMemory(mem_dir)
        mem.add_learning(_make_learning(id="cycle-1"))
        mem.save()

        mem2 = CrossRunMemory(mem_dir)
        mem2.load()
        mem2.add_learning(_make_learning(id="cycle-2"))
        mem2.save()

        mem3 = CrossRunMemory(mem_dir)
        mem3.load()
        assert len(mem3.learnings) == 2
        assert mem3.learnings[0].id == "cycle-1"
        assert mem3.learnings[1].id == "cycle-2"

    def test_load_sets_loaded_flag(self, tmp_path):
        mem = CrossRunMemory(str(tmp_path / "mem"))
        assert mem._loaded is False
        mem.load()
        assert mem._loaded is True

    def test_json_files_are_valid_json(self, tmp_path):
        mem_dir = str(tmp_path / "mem")
        mem = CrossRunMemory(mem_dir)
        mem.add_learning(_make_learning())
        mem.add_failed_approach("test", score=0.0, iteration=1, run_id="r1")
        mem.add_strategy("test", "desc", score=0.5, run_id="r1")
        mem.save()

        for filename in ["learnings.json", "failed_approaches.json", "strategies.json"]:
            filepath = os.path.join(mem_dir, filename)
            with open(filepath, "r") as f:
                data = json.load(f)
            assert isinstance(data, list)

    def test_learning_metadata_survives_roundtrip(self, tmp_path):
        mem_dir = str(tmp_path / "mem")
        mem = CrossRunMemory(mem_dir)
        mem.add_learning(_make_learning(
            id="meta-1",
            metadata={"tags": ["important", "verified"], "count": 42, "nested": {"x": 1}},
        ))
        mem.save()

        mem2 = CrossRunMemory(mem_dir)
        mem2.load()
        meta = mem2.learnings[0].metadata
        assert meta["tags"] == ["important", "verified"]
        assert meta["count"] == 42
        assert meta["nested"]["x"] == 1
