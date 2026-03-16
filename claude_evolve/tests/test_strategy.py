"""
Tests for claude_evolve.core.strategy.

Covers Strategy dataclass creation, serialization (to_dict/from_dict),
computed properties (avg_score, best_score), DEFAULT_STRATEGIES validity,
StrategyManager load/save round-trip, select_strategy at different
stagnation levels, record_outcome, add_strategy, format_for_prompt,
empty strategy list handling, and novelty bonus behavior.
"""

import json
import os
import time

import pytest

from claude_evolve.core.strategy import (
    DEFAULT_STRATEGIES,
    Strategy,
    StrategyManager,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_strategy(
    id: str = "test-1",
    name: str = "Test Strategy",
    description: str = "A test strategy for unit tests.",
    generation_approach: str = "diff",
    research_focus: str = "",
    template_key: str = "diff_user",
    exploration_weight: float = 0.5,
    score_history: list = None,
    times_used: int = 0,
    metadata: dict = None,
) -> Strategy:
    """Create a Strategy with sensible defaults for testing."""
    return Strategy(
        id=id,
        name=name,
        description=description,
        generation_approach=generation_approach,
        research_focus=research_focus,
        template_key=template_key,
        exploration_weight=exploration_weight,
        score_history=score_history if score_history is not None else [],
        times_used=times_used,
        metadata=metadata or {},
    )


# ---------------------------------------------------------------------------
# Strategy dataclass creation
# ---------------------------------------------------------------------------

class TestStrategyCreation:
    """Test Strategy dataclass instantiation."""

    def test_create_with_required_fields(self):
        s = Strategy(id="s1", name="Basic", description="Basic strategy")
        assert s.id == "s1"
        assert s.name == "Basic"
        assert s.description == "Basic strategy"

    def test_default_field_values(self):
        s = Strategy(id="s2", name="Defaults", description="Uses defaults")
        assert s.generation_approach == "diff"
        assert s.research_focus == ""
        assert s.template_key == "diff_user"
        assert s.exploration_weight == 0.5
        assert s.score_history == []
        assert s.times_used == 0
        assert isinstance(s.created_at, float)
        assert s.metadata == {}

    def test_create_with_all_fields(self):
        s = Strategy(
            id="s3",
            name="Full",
            description="All fields set",
            generation_approach="full_rewrite",
            research_focus="algorithms",
            template_key="full_rewrite_user",
            exploration_weight=0.8,
            score_history=[0.5, 0.7],
            times_used=2,
            created_at=1000.0,
            metadata={"source": "test"},
        )
        assert s.generation_approach == "full_rewrite"
        assert s.research_focus == "algorithms"
        assert s.template_key == "full_rewrite_user"
        assert s.exploration_weight == 0.8
        assert s.score_history == [0.5, 0.7]
        assert s.times_used == 2
        assert s.created_at == 1000.0
        assert s.metadata["source"] == "test"

    def test_generate_id_returns_valid_uuid(self):
        id1 = Strategy.generate_id()
        id2 = Strategy.generate_id()
        assert isinstance(id1, str)
        assert len(id1) == 36  # UUID format: 8-4-4-4-12
        assert id1 != id2  # Should be unique


# ---------------------------------------------------------------------------
# Strategy computed properties
# ---------------------------------------------------------------------------

class TestStrategyProperties:
    """Test avg_score and best_score computed properties."""

    def test_avg_score_empty_history(self):
        s = _make_strategy(score_history=[])
        assert s.avg_score == 0.0

    def test_avg_score_single_value(self):
        s = _make_strategy(score_history=[0.8])
        assert s.avg_score == pytest.approx(0.8)

    def test_avg_score_multiple_values(self):
        s = _make_strategy(score_history=[0.2, 0.4, 0.6, 0.8])
        assert s.avg_score == pytest.approx(0.5)

    def test_avg_score_all_same(self):
        s = _make_strategy(score_history=[0.5, 0.5, 0.5])
        assert s.avg_score == pytest.approx(0.5)

    def test_best_score_empty_history(self):
        s = _make_strategy(score_history=[])
        assert s.best_score == 0.0

    def test_best_score_single_value(self):
        s = _make_strategy(score_history=[0.7])
        assert s.best_score == pytest.approx(0.7)

    def test_best_score_multiple_values(self):
        s = _make_strategy(score_history=[0.2, 0.9, 0.4, 0.6])
        assert s.best_score == pytest.approx(0.9)

    def test_best_score_all_same(self):
        s = _make_strategy(score_history=[0.3, 0.3, 0.3])
        assert s.best_score == pytest.approx(0.3)

    def test_best_score_negative_values(self):
        s = _make_strategy(score_history=[-0.5, -0.1, -0.3])
        assert s.best_score == pytest.approx(-0.1)


# ---------------------------------------------------------------------------
# Strategy serialization (to_dict / from_dict)
# ---------------------------------------------------------------------------

class TestStrategySerialization:
    """Test Strategy.to_dict() and Strategy.from_dict()."""

    def test_to_dict_returns_all_fields(self):
        s = _make_strategy(
            id="ser-1",
            name="Serialize Me",
            description="For serialization test",
            generation_approach="solver_hybrid",
            research_focus="constraints",
            template_key="solver_user",
            exploration_weight=0.7,
            score_history=[0.4, 0.6],
            times_used=2,
            metadata={"tag": "test"},
        )
        d = s.to_dict()
        assert d["id"] == "ser-1"
        assert d["name"] == "Serialize Me"
        assert d["description"] == "For serialization test"
        assert d["generation_approach"] == "solver_hybrid"
        assert d["research_focus"] == "constraints"
        assert d["template_key"] == "solver_user"
        assert d["exploration_weight"] == 0.7
        assert d["score_history"] == [0.4, 0.6]
        assert d["times_used"] == 2
        assert isinstance(d["created_at"], float)
        assert d["metadata"]["tag"] == "test"

    def test_from_dict_required_fields(self):
        d = {
            "id": "fd-1",
            "name": "From Dict",
            "description": "Created from dict",
        }
        s = Strategy.from_dict(d)
        assert s.id == "fd-1"
        assert s.name == "From Dict"
        assert s.description == "Created from dict"
        # Defaults applied
        assert s.generation_approach == "diff"
        assert s.score_history == []

    def test_from_dict_ignores_extra_keys(self):
        d = {
            "id": "fd-2",
            "name": "Extra Keys",
            "description": "Has extra keys",
            "unknown_field": "should be ignored",
            "another_extra": 42,
        }
        s = Strategy.from_dict(d)
        assert s.id == "fd-2"
        assert not hasattr(s, "unknown_field")

    def test_roundtrip_to_dict_from_dict(self):
        original = _make_strategy(
            id="rt-1",
            name="Roundtrip",
            description="Roundtrip test",
            generation_approach="full_rewrite",
            research_focus="search",
            template_key="full_rewrite_user",
            exploration_weight=0.65,
            score_history=[0.1, 0.5, 0.9],
            times_used=3,
            metadata={"key": "value", "nested": {"a": 1}},
        )
        d = original.to_dict()
        restored = Strategy.from_dict(d)
        assert restored.id == original.id
        assert restored.name == original.name
        assert restored.description == original.description
        assert restored.generation_approach == original.generation_approach
        assert restored.research_focus == original.research_focus
        assert restored.template_key == original.template_key
        assert restored.exploration_weight == original.exploration_weight
        assert restored.score_history == original.score_history
        assert restored.times_used == original.times_used
        assert restored.metadata == original.metadata
        assert abs(restored.created_at - original.created_at) < 0.001

    def test_to_dict_json_serializable(self):
        s = _make_strategy(metadata={"nested": {"a": 1}})
        d = s.to_dict()
        serialized = json.dumps(d)
        assert isinstance(serialized, str)
        deserialized = json.loads(serialized)
        assert deserialized["id"] == s.id


# ---------------------------------------------------------------------------
# DEFAULT_STRATEGIES validity
# ---------------------------------------------------------------------------

class TestDefaultStrategies:
    """Test that DEFAULT_STRATEGIES are well-formed."""

    def test_default_strategies_count(self):
        assert len(DEFAULT_STRATEGIES) == 7

    def test_all_have_unique_ids(self):
        ids = [s.id for s in DEFAULT_STRATEGIES]
        assert len(ids) == len(set(ids))

    def test_all_have_unique_names(self):
        names = [s.name for s in DEFAULT_STRATEGIES]
        assert len(names) == len(set(names))

    def test_all_have_nonempty_description(self):
        for s in DEFAULT_STRATEGIES:
            assert len(s.description) > 10, f"Strategy {s.id} has short description"

    def test_all_have_valid_generation_approach(self):
        valid_approaches = {"diff", "full_rewrite", "solver_hybrid", "from_scratch"}
        for s in DEFAULT_STRATEGIES:
            assert s.generation_approach in valid_approaches, (
                f"Strategy {s.id} has invalid approach: {s.generation_approach}"
            )

    def test_exploration_weights_in_range(self):
        for s in DEFAULT_STRATEGIES:
            assert 0.0 <= s.exploration_weight <= 1.0, (
                f"Strategy {s.id} weight out of range: {s.exploration_weight}"
            )

    def test_all_are_json_serializable(self):
        for s in DEFAULT_STRATEGIES:
            d = s.to_dict()
            serialized = json.dumps(d)
            assert isinstance(serialized, str)

    def test_includes_incremental_strategy(self):
        ids = [s.id for s in DEFAULT_STRATEGIES]
        assert "default-incremental" in ids

    def test_includes_creative_strategy(self):
        ids = [s.id for s in DEFAULT_STRATEGIES]
        assert "default-creative" in ids

    def test_includes_solver_strategy(self):
        ids = [s.id for s in DEFAULT_STRATEGIES]
        assert "default-solver" in ids

    def test_solver_strategy_has_research_focus(self):
        solver = None
        for s in DEFAULT_STRATEGIES:
            if s.id == "default-solver":
                solver = s
                break
        assert solver is not None
        assert len(solver.research_focus) > 0

    def test_research_driven_has_research_focus(self):
        research = None
        for s in DEFAULT_STRATEGIES:
            if s.id == "default-research-first":
                research = s
                break
        assert research is not None
        assert len(research.research_focus) > 0


# ---------------------------------------------------------------------------
# StrategyManager initialization
# ---------------------------------------------------------------------------

class TestStrategyManagerInit:
    """Test StrategyManager constructor."""

    def test_default_initialization(self, tmp_path):
        path = str(tmp_path / "strategies.json")
        mgr = StrategyManager(path)
        assert mgr.strategies_path == path
        assert mgr.strategies == []
        assert mgr._loaded is False


# ---------------------------------------------------------------------------
# StrategyManager load
# ---------------------------------------------------------------------------

class TestStrategyManagerLoad:
    """Test StrategyManager.load() from various states."""

    def test_load_nonexistent_file_initializes_defaults(self, tmp_path):
        path = str(tmp_path / "nonexistent.json")
        mgr = StrategyManager(path)
        mgr.load()
        assert len(mgr.strategies) == len(DEFAULT_STRATEGIES)
        assert mgr._loaded is True

    def test_load_existing_file(self, tmp_path):
        path = str(tmp_path / "strategies.json")
        # Write a custom strategies file
        custom = [
            {"id": "custom-1", "name": "Custom", "description": "Custom strategy"},
        ]
        with open(path, "w") as f:
            json.dump(custom, f)
        mgr = StrategyManager(path)
        mgr.load()
        assert len(mgr.strategies) == 1
        assert mgr.strategies[0].id == "custom-1"
        assert mgr._loaded is True

    def test_load_sets_loaded_flag(self, tmp_path):
        path = str(tmp_path / "strategies.json")
        mgr = StrategyManager(path)
        assert mgr._loaded is False
        mgr.load()
        assert mgr._loaded is True


# ---------------------------------------------------------------------------
# StrategyManager save
# ---------------------------------------------------------------------------

class TestStrategyManagerSave:
    """Test StrategyManager.save() creates valid JSON."""

    def test_save_creates_file(self, tmp_path):
        path = str(tmp_path / "strategies.json")
        mgr = StrategyManager(path)
        mgr.strategies = [_make_strategy(id="save-1")]
        mgr.save()
        assert os.path.exists(path)

    def test_save_creates_parent_directory(self, tmp_path):
        path = str(tmp_path / "nested" / "deep" / "strategies.json")
        mgr = StrategyManager(path)
        mgr.strategies = [_make_strategy(id="save-nested")]
        mgr.save()
        assert os.path.exists(path)

    def test_save_writes_valid_json(self, tmp_path):
        path = str(tmp_path / "strategies.json")
        mgr = StrategyManager(path)
        mgr.strategies = [_make_strategy(id="json-1"), _make_strategy(id="json-2")]
        mgr.save()
        with open(path, "r") as f:
            data = json.load(f)
        assert isinstance(data, list)
        assert len(data) == 2
        assert data[0]["id"] == "json-1"
        assert data[1]["id"] == "json-2"

    def test_save_empty_strategies(self, tmp_path):
        path = str(tmp_path / "strategies.json")
        mgr = StrategyManager(path)
        mgr.strategies = []
        mgr.save()
        with open(path, "r") as f:
            data = json.load(f)
        assert data == []


# ---------------------------------------------------------------------------
# StrategyManager load/save round-trip
# ---------------------------------------------------------------------------

class TestStrategyManagerRoundTrip:
    """Test that save() followed by load() preserves all data."""

    def test_roundtrip_single_strategy(self, tmp_path):
        path = str(tmp_path / "strategies.json")
        mgr1 = StrategyManager(path)
        s = _make_strategy(
            id="rt-1",
            name="Roundtrip",
            description="Roundtrip strategy",
            score_history=[0.3, 0.6, 0.9],
            times_used=3,
            metadata={"key": "value"},
        )
        mgr1.strategies = [s]
        mgr1.save()

        mgr2 = StrategyManager(path)
        mgr2.load()
        assert len(mgr2.strategies) == 1
        loaded = mgr2.strategies[0]
        assert loaded.id == "rt-1"
        assert loaded.name == "Roundtrip"
        assert loaded.description == "Roundtrip strategy"
        assert loaded.score_history == [0.3, 0.6, 0.9]
        assert loaded.times_used == 3
        assert loaded.metadata["key"] == "value"

    def test_roundtrip_multiple_strategies(self, tmp_path):
        path = str(tmp_path / "strategies.json")
        mgr1 = StrategyManager(path)
        mgr1.strategies = [
            _make_strategy(id="rt-a", name="Alpha"),
            _make_strategy(id="rt-b", name="Beta"),
            _make_strategy(id="rt-c", name="Gamma"),
        ]
        mgr1.save()

        mgr2 = StrategyManager(path)
        mgr2.load()
        assert len(mgr2.strategies) == 3
        assert mgr2.strategies[0].name == "Alpha"
        assert mgr2.strategies[1].name == "Beta"
        assert mgr2.strategies[2].name == "Gamma"

    def test_roundtrip_defaults(self, tmp_path):
        """Load defaults, save, reload -- should be identical."""
        path = str(tmp_path / "strategies.json")
        mgr1 = StrategyManager(path)
        mgr1.load()  # Loads defaults
        mgr1.save()

        mgr2 = StrategyManager(path)
        mgr2.load()
        assert len(mgr2.strategies) == len(DEFAULT_STRATEGIES)
        for i, s in enumerate(mgr2.strategies):
            assert s.id == DEFAULT_STRATEGIES[i].id
            assert s.name == DEFAULT_STRATEGIES[i].name

    def test_multiple_save_load_cycles(self, tmp_path):
        path = str(tmp_path / "strategies.json")
        mgr = StrategyManager(path)
        mgr.strategies = [_make_strategy(id="cycle-1")]
        mgr.save()

        mgr2 = StrategyManager(path)
        mgr2.load()
        mgr2.strategies.append(_make_strategy(id="cycle-2"))
        mgr2.save()

        mgr3 = StrategyManager(path)
        mgr3.load()
        assert len(mgr3.strategies) == 2
        assert mgr3.strategies[0].id == "cycle-1"
        assert mgr3.strategies[1].id == "cycle-2"


# ---------------------------------------------------------------------------
# StrategyManager.select_strategy
# ---------------------------------------------------------------------------

class TestSelectStrategy:
    """Test select_strategy at different stagnation levels."""

    def test_returns_a_strategy(self, tmp_path):
        path = str(tmp_path / "strategies.json")
        mgr = StrategyManager(path)
        mgr.load()
        selected = mgr.select_strategy(stagnation_level="none")
        assert isinstance(selected, Strategy)
        assert selected.id in [s.id for s in mgr.strategies]

    def test_returns_strategy_at_each_stagnation_level(self, tmp_path):
        path = str(tmp_path / "strategies.json")
        mgr = StrategyManager(path)
        mgr.load()
        for level in ["none", "mild", "moderate", "severe", "critical"]:
            selected = mgr.select_strategy(stagnation_level=level)
            assert isinstance(selected, Strategy)

    def test_unknown_stagnation_level_uses_default(self, tmp_path):
        path = str(tmp_path / "strategies.json")
        mgr = StrategyManager(path)
        mgr.load()
        # Should not crash, uses 0.5 as default explore_weight
        selected = mgr.select_strategy(stagnation_level="unknown_level")
        assert isinstance(selected, Strategy)

    def test_empty_strategies_initializes_defaults(self, tmp_path):
        path = str(tmp_path / "strategies.json")
        mgr = StrategyManager(path)
        mgr.strategies = []  # Empty
        selected = mgr.select_strategy(stagnation_level="none")
        assert isinstance(selected, Strategy)
        assert len(mgr.strategies) == len(DEFAULT_STRATEGIES)

    def test_single_strategy_always_selected(self, tmp_path):
        path = str(tmp_path / "strategies.json")
        mgr = StrategyManager(path)
        only = _make_strategy(id="only-one", name="Only One")
        mgr.strategies = [only]
        for _ in range(10):
            selected = mgr.select_strategy(stagnation_level="none")
            assert selected.id == "only-one"

    def test_high_stagnation_favors_exploratory(self, tmp_path):
        """At critical stagnation, exploratory strategies should be favored."""
        path = str(tmp_path / "strategies.json")
        mgr = StrategyManager(path)
        # Set up two strategies: one exploitative, one exploratory
        exploit = _make_strategy(
            id="exploit",
            name="Exploit",
            exploration_weight=0.1,
            score_history=[0.9, 0.9],
            times_used=2,
        )
        explore = _make_strategy(
            id="explore",
            name="Explore",
            exploration_weight=0.95,
            score_history=[0.3, 0.4],
            times_used=2,
        )
        mgr.strategies = [exploit, explore]

        # Run many selections at critical stagnation
        explore_count = 0
        trials = 200
        for _ in range(trials):
            selected = mgr.select_strategy(stagnation_level="critical")
            if selected.id == "explore":
                explore_count += 1

        # At critical stagnation (explore_weight=0.95), exploratory should dominate
        assert explore_count > trials * 0.4, (
            f"Expected exploratory strategy to be selected often at critical stagnation, "
            f"but only selected {explore_count}/{trials} times"
        )

    def test_no_stagnation_favors_high_performers(self, tmp_path):
        """At no stagnation, high-scoring strategies should be favored."""
        path = str(tmp_path / "strategies.json")
        mgr = StrategyManager(path)
        high = _make_strategy(
            id="high",
            name="High Scorer",
            exploration_weight=0.1,
            score_history=[0.9, 0.95, 0.92],
            times_used=3,
        )
        low = _make_strategy(
            id="low",
            name="Low Scorer",
            exploration_weight=0.9,
            score_history=[0.1, 0.15],
            times_used=2,
        )
        mgr.strategies = [high, low]

        high_count = 0
        trials = 200
        for _ in range(trials):
            selected = mgr.select_strategy(stagnation_level="none")
            if selected.id == "high":
                high_count += 1

        # At no stagnation (explore_weight=0.2), high performers should dominate
        assert high_count > trials * 0.4, (
            f"Expected high-scoring strategy to be selected often at no stagnation, "
            f"but only selected {high_count}/{trials} times"
        )


# ---------------------------------------------------------------------------
# StrategyManager.select_strategy -- novelty bonus
# ---------------------------------------------------------------------------

class TestSelectStrategyNoveltyBonus:
    """Test that untried strategies get a novelty bonus."""

    def test_untried_strategy_gets_selected(self, tmp_path):
        """An untried strategy should have a novelty bonus making it more likely."""
        path = str(tmp_path / "strategies.json")
        mgr = StrategyManager(path)
        tried = _make_strategy(
            id="tried",
            name="Tried",
            exploration_weight=0.5,
            score_history=[0.5],
            times_used=1,
        )
        untried = _make_strategy(
            id="untried",
            name="Untried",
            exploration_weight=0.5,
            score_history=[],
            times_used=0,
        )
        mgr.strategies = [tried, untried]

        untried_count = 0
        trials = 200
        for _ in range(trials):
            selected = mgr.select_strategy(stagnation_level="none")
            if selected.id == "untried":
                untried_count += 1

        # The untried strategy should be selected more due to novelty bonus
        assert untried_count > trials * 0.3, (
            f"Expected untried strategy to be selected often due to novelty bonus, "
            f"but only selected {untried_count}/{trials} times"
        )


# ---------------------------------------------------------------------------
# StrategyManager.record_outcome
# ---------------------------------------------------------------------------

class TestRecordOutcome:
    """Test record_outcome updates strategy correctly."""

    def test_record_appends_score(self, tmp_path):
        path = str(tmp_path / "strategies.json")
        mgr = StrategyManager(path)
        s = _make_strategy(id="rec-1", score_history=[], times_used=0)
        mgr.strategies = [s]
        mgr.record_outcome("rec-1", 0.75)
        assert mgr.strategies[0].score_history == [0.75]
        assert mgr.strategies[0].times_used == 1

    def test_record_multiple_outcomes(self, tmp_path):
        path = str(tmp_path / "strategies.json")
        mgr = StrategyManager(path)
        s = _make_strategy(id="rec-2", score_history=[], times_used=0)
        mgr.strategies = [s]
        mgr.record_outcome("rec-2", 0.3)
        mgr.record_outcome("rec-2", 0.6)
        mgr.record_outcome("rec-2", 0.9)
        assert mgr.strategies[0].score_history == [0.3, 0.6, 0.9]
        assert mgr.strategies[0].times_used == 3

    def test_record_nonexistent_id_is_noop(self, tmp_path):
        path = str(tmp_path / "strategies.json")
        mgr = StrategyManager(path)
        s = _make_strategy(id="rec-3", score_history=[], times_used=0)
        mgr.strategies = [s]
        mgr.record_outcome("nonexistent", 0.5)
        # Original strategy should be unchanged
        assert mgr.strategies[0].score_history == []
        assert mgr.strategies[0].times_used == 0

    def test_record_updates_correct_strategy(self, tmp_path):
        path = str(tmp_path / "strategies.json")
        mgr = StrategyManager(path)
        s1 = _make_strategy(id="rec-a", name="A", score_history=[], times_used=0)
        s2 = _make_strategy(id="rec-b", name="B", score_history=[], times_used=0)
        mgr.strategies = [s1, s2]
        mgr.record_outcome("rec-b", 0.8)
        assert mgr.strategies[0].score_history == []  # A unchanged
        assert mgr.strategies[0].times_used == 0
        assert mgr.strategies[1].score_history == [0.8]  # B updated
        assert mgr.strategies[1].times_used == 1

    def test_record_outcome_survives_save_load(self, tmp_path):
        path = str(tmp_path / "strategies.json")
        mgr = StrategyManager(path)
        mgr.strategies = [_make_strategy(id="persist-1", score_history=[], times_used=0)]
        mgr.record_outcome("persist-1", 0.7)
        mgr.save()

        mgr2 = StrategyManager(path)
        mgr2.load()
        assert mgr2.strategies[0].score_history == [0.7]
        assert mgr2.strategies[0].times_used == 1


# ---------------------------------------------------------------------------
# StrategyManager.add_strategy
# ---------------------------------------------------------------------------

class TestAddStrategy:
    """Test add_strategy appends correctly."""

    def test_add_single_strategy(self, tmp_path):
        path = str(tmp_path / "strategies.json")
        mgr = StrategyManager(path)
        s = _make_strategy(id="add-1", name="Added")
        mgr.add_strategy(s)
        assert len(mgr.strategies) == 1
        assert mgr.strategies[0].id == "add-1"

    def test_add_multiple_strategies(self, tmp_path):
        path = str(tmp_path / "strategies.json")
        mgr = StrategyManager(path)
        for i in range(5):
            mgr.add_strategy(_make_strategy(id=f"add-{i}"))
        assert len(mgr.strategies) == 5

    def test_add_to_existing_strategies(self, tmp_path):
        path = str(tmp_path / "strategies.json")
        mgr = StrategyManager(path)
        mgr.load()  # Loads defaults
        count_before = len(mgr.strategies)
        mgr.add_strategy(_make_strategy(id="new-custom"))
        assert len(mgr.strategies) == count_before + 1


# ---------------------------------------------------------------------------
# StrategyManager.get_strategy_by_id
# ---------------------------------------------------------------------------

class TestGetStrategyById:
    """Test get_strategy_by_id lookup."""

    def test_find_existing_strategy(self, tmp_path):
        path = str(tmp_path / "strategies.json")
        mgr = StrategyManager(path)
        mgr.strategies = [
            _make_strategy(id="find-1", name="Alpha"),
            _make_strategy(id="find-2", name="Beta"),
        ]
        found = mgr.get_strategy_by_id("find-2")
        assert found is not None
        assert found.name == "Beta"

    def test_returns_none_for_missing_id(self, tmp_path):
        path = str(tmp_path / "strategies.json")
        mgr = StrategyManager(path)
        mgr.strategies = [_make_strategy(id="find-1")]
        assert mgr.get_strategy_by_id("nonexistent") is None

    def test_returns_none_from_empty_list(self, tmp_path):
        path = str(tmp_path / "strategies.json")
        mgr = StrategyManager(path)
        assert mgr.get_strategy_by_id("any") is None


# ---------------------------------------------------------------------------
# StrategyManager.format_for_prompt
# ---------------------------------------------------------------------------

class TestFormatForPrompt:
    """Test format_for_prompt output."""

    def test_basic_format_includes_name(self, tmp_path):
        path = str(tmp_path / "strategies.json")
        mgr = StrategyManager(path)
        s = _make_strategy(name="Test Strategy")
        output = mgr.format_for_prompt(s)
        assert "Test Strategy" in output

    def test_format_includes_strategy_directive_header(self, tmp_path):
        path = str(tmp_path / "strategies.json")
        mgr = StrategyManager(path)
        s = _make_strategy()
        output = mgr.format_for_prompt(s)
        assert "## Strategy Directive" in output

    def test_format_includes_approach(self, tmp_path):
        path = str(tmp_path / "strategies.json")
        mgr = StrategyManager(path)
        s = _make_strategy(generation_approach="full_rewrite")
        output = mgr.format_for_prompt(s)
        assert "full_rewrite" in output

    def test_format_includes_exploration_level(self, tmp_path):
        path = str(tmp_path / "strategies.json")
        mgr = StrategyManager(path)
        s = _make_strategy(exploration_weight=0.7)
        output = mgr.format_for_prompt(s)
        assert "0.7" in output

    def test_format_includes_description(self, tmp_path):
        path = str(tmp_path / "strategies.json")
        mgr = StrategyManager(path)
        s = _make_strategy(description="A unique approach to solving the problem.")
        output = mgr.format_for_prompt(s)
        assert "A unique approach to solving the problem." in output

    def test_format_includes_research_focus_when_present(self, tmp_path):
        path = str(tmp_path / "strategies.json")
        mgr = StrategyManager(path)
        s = _make_strategy(research_focus="theoretical bounds, optimization")
        output = mgr.format_for_prompt(s)
        assert "Research focus" in output
        assert "theoretical bounds" in output

    def test_format_excludes_research_focus_when_empty(self, tmp_path):
        path = str(tmp_path / "strategies.json")
        mgr = StrategyManager(path)
        s = _make_strategy(research_focus="")
        output = mgr.format_for_prompt(s)
        assert "Research focus" not in output

    def test_format_includes_track_record_when_history_exists(self, tmp_path):
        path = str(tmp_path / "strategies.json")
        mgr = StrategyManager(path)
        s = _make_strategy(score_history=[0.5, 0.7, 0.9], times_used=3)
        output = mgr.format_for_prompt(s)
        assert "Track record" in output
        assert "Used 3 times" in output
        assert "avg score" in output
        assert "best" in output

    def test_format_excludes_track_record_when_no_history(self, tmp_path):
        path = str(tmp_path / "strategies.json")
        mgr = StrategyManager(path)
        s = _make_strategy(score_history=[], times_used=0)
        output = mgr.format_for_prompt(s)
        assert "Track record" not in output

    def test_format_for_each_default_strategy(self, tmp_path):
        """Every default strategy should produce valid format output."""
        path = str(tmp_path / "strategies.json")
        mgr = StrategyManager(path)
        for s in DEFAULT_STRATEGIES:
            output = mgr.format_for_prompt(s)
            assert "## Strategy Directive" in output
            assert s.name in output
            assert len(output) > 50


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge cases and integration scenarios."""

    def test_strategy_with_zero_exploration_weight(self, tmp_path):
        s = _make_strategy(exploration_weight=0.0)
        assert s.exploration_weight == 0.0
        d = s.to_dict()
        restored = Strategy.from_dict(d)
        assert restored.exploration_weight == 0.0

    def test_strategy_with_max_exploration_weight(self, tmp_path):
        s = _make_strategy(exploration_weight=1.0)
        assert s.exploration_weight == 1.0

    def test_select_from_all_zero_score_strategies(self, tmp_path):
        """All strategies with zero combined score should not crash."""
        path = str(tmp_path / "strategies.json")
        mgr = StrategyManager(path)
        mgr.strategies = [
            _make_strategy(id="z1", exploration_weight=0.0, score_history=[0.0], times_used=1),
            _make_strategy(id="z2", exploration_weight=0.0, score_history=[0.0], times_used=1),
        ]
        # When total is zero, should fall back to random.choice
        selected = mgr.select_strategy(stagnation_level="none")
        assert isinstance(selected, Strategy)

    def test_large_score_history(self, tmp_path):
        scores = [float(i) / 1000 for i in range(1000)]
        s = _make_strategy(score_history=scores)
        assert s.avg_score == pytest.approx(sum(scores) / len(scores))
        assert s.best_score == pytest.approx(0.999)

    def test_json_files_are_valid_json(self, tmp_path):
        path = str(tmp_path / "strategies.json")
        mgr = StrategyManager(path)
        mgr.load()
        mgr.save()
        with open(path, "r") as f:
            data = json.load(f)
        assert isinstance(data, list)
        assert len(data) == len(DEFAULT_STRATEGIES)

    def test_concurrent_managers_last_write_wins(self, tmp_path):
        """Two managers writing to same file -- last write wins."""
        path = str(tmp_path / "strategies.json")
        mgr1 = StrategyManager(path)
        mgr1.strategies = [_make_strategy(id="m1")]
        mgr1.save()

        mgr2 = StrategyManager(path)
        mgr2.strategies = [_make_strategy(id="m2-a"), _make_strategy(id="m2-b")]
        mgr2.save()

        mgr3 = StrategyManager(path)
        mgr3.load()
        assert len(mgr3.strategies) == 2
        assert mgr3.strategies[0].id == "m2-a"
