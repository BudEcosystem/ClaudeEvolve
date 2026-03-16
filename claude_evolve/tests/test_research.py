"""
Tests for the research log module.

Covers ResearchFinding creation and serialization, ResearchLog initialization,
load/save round-trips, finding management, trigger logic, and prompt formatting.
"""

import json
import os
import time

import pytest

from claude_evolve.core.research import ResearchFinding, ResearchLog


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_finding(
    id: str = "find-1",
    iteration: int = 5,
    approach_name: str = "Test Approach",
    description: str = "A test approach description",
    novelty: str = "medium",
    implementation_hint: str = "",
    source_url: str = "",
    was_tried: bool = False,
    outcome_score=None,
    metadata: dict = None,
) -> ResearchFinding:
    """Create a ResearchFinding with sensible defaults for testing."""
    return ResearchFinding(
        id=id,
        iteration=iteration,
        timestamp=time.time(),
        approach_name=approach_name,
        description=description,
        novelty=novelty,
        implementation_hint=implementation_hint,
        source_url=source_url,
        was_tried=was_tried,
        outcome_score=outcome_score,
        metadata=metadata or {},
    )


class _FakeResearchConfig:
    """Minimal config-like object for testing should_research."""

    def __init__(
        self,
        enabled: bool = True,
        trigger: str = "on_stagnation",
        periodic_interval: int = 10,
    ):
        self.enabled = enabled
        self.trigger = trigger
        self.periodic_interval = periodic_interval


# ---------------------------------------------------------------------------
# ResearchFinding creation and serialization
# ---------------------------------------------------------------------------

class TestResearchFindingCreation:
    """Test ResearchFinding dataclass instantiation."""

    def test_create_with_required_fields(self):
        f = ResearchFinding(
            id="rf-1",
            iteration=1,
            timestamp=1000.0,
            approach_name="Greedy",
            description="Use greedy algorithm",
            novelty="high",
        )
        assert f.id == "rf-1"
        assert f.iteration == 1
        assert f.timestamp == 1000.0
        assert f.approach_name == "Greedy"
        assert f.description == "Use greedy algorithm"
        assert f.novelty == "high"

    def test_optional_fields_defaults(self):
        f = ResearchFinding(
            id="rf-2",
            iteration=2,
            timestamp=1001.0,
            approach_name="DP",
            description="Dynamic programming",
            novelty="medium",
        )
        assert f.implementation_hint == ""
        assert f.source_url == ""
        assert f.was_tried is False
        assert f.outcome_score is None
        assert f.metadata == {}

    def test_create_with_all_fields(self):
        f = ResearchFinding(
            id="rf-3",
            iteration=10,
            timestamp=2000.0,
            approach_name="SA",
            description="Simulated annealing",
            novelty="low",
            implementation_hint="Use temperature schedule",
            source_url="https://example.com/sa",
            was_tried=True,
            outcome_score=0.85,
            metadata={"technique": "cooling"},
        )
        assert f.implementation_hint == "Use temperature schedule"
        assert f.source_url == "https://example.com/sa"
        assert f.was_tried is True
        assert f.outcome_score == 0.85
        assert f.metadata["technique"] == "cooling"


class TestResearchFindingSerialization:
    """Test ResearchFinding.to_dict() and ResearchFinding.from_dict()."""

    def test_to_dict_returns_all_fields(self):
        f = _make_finding()
        d = f.to_dict()
        assert d["id"] == "find-1"
        assert d["iteration"] == 5
        assert d["approach_name"] == "Test Approach"
        assert d["description"] == "A test approach description"
        assert d["novelty"] == "medium"
        assert isinstance(d["timestamp"], float)
        assert isinstance(d["metadata"], dict)

    def test_from_dict_required_fields(self):
        d = {
            "id": "x1",
            "iteration": 3,
            "timestamp": 999.0,
            "approach_name": "Monte Carlo",
            "description": "Random sampling",
            "novelty": "high",
        }
        f = ResearchFinding.from_dict(d)
        assert f.id == "x1"
        assert f.approach_name == "Monte Carlo"
        assert f.was_tried is False

    def test_from_dict_ignores_extra_keys(self):
        d = {
            "id": "x2",
            "iteration": 1,
            "timestamp": 100.0,
            "approach_name": "Test",
            "description": "Desc",
            "novelty": "low",
            "unknown_field": "should be ignored",
            "another_extra": 42,
        }
        f = ResearchFinding.from_dict(d)
        assert f.id == "x2"
        assert not hasattr(f, "unknown_field")

    def test_roundtrip_to_dict_from_dict(self):
        original = _make_finding(
            id="rt-1",
            iteration=7,
            approach_name="Genetic",
            description="Genetic algorithm approach",
            novelty="high",
            implementation_hint="Use crossover",
            source_url="https://example.com",
            was_tried=True,
            outcome_score=0.9,
            metadata={"key": "value"},
        )
        d = original.to_dict()
        restored = ResearchFinding.from_dict(d)
        assert restored.id == original.id
        assert restored.iteration == original.iteration
        assert restored.approach_name == original.approach_name
        assert restored.description == original.description
        assert restored.novelty == original.novelty
        assert restored.implementation_hint == original.implementation_hint
        assert restored.source_url == original.source_url
        assert restored.was_tried == original.was_tried
        assert restored.outcome_score == original.outcome_score
        assert restored.metadata == original.metadata
        assert abs(restored.timestamp - original.timestamp) < 0.001

    def test_to_dict_json_serializable(self):
        """to_dict output must be JSON-serializable."""
        f = _make_finding(metadata={"nested": {"a": 1}})
        d = f.to_dict()
        serialized = json.dumps(d)
        assert isinstance(serialized, str)
        deserialized = json.loads(serialized)
        assert deserialized["id"] == f.id


# ---------------------------------------------------------------------------
# ResearchLog initialization
# ---------------------------------------------------------------------------

class TestResearchLogInit:
    """Test ResearchLog constructor defaults."""

    def test_default_initialization(self, tmp_path):
        log = ResearchLog(str(tmp_path / "research.json"))
        assert log.log_path == str(tmp_path / "research.json")
        assert log.findings == []
        assert log.theoretical_bounds == {}
        assert log.key_papers == []
        assert log.approaches_to_avoid == []


# ---------------------------------------------------------------------------
# load() from nonexistent path
# ---------------------------------------------------------------------------

class TestResearchLogLoadEmpty:
    """Test loading when the log file does not exist."""

    def test_load_nonexistent_file(self, tmp_path):
        log = ResearchLog(str(tmp_path / "nonexistent.json"))
        log.load()
        assert log.findings == []
        assert log.theoretical_bounds == {}
        assert log.key_papers == []
        assert log.approaches_to_avoid == []


# ---------------------------------------------------------------------------
# save() and load() round-trip
# ---------------------------------------------------------------------------

class TestResearchLogRoundTrip:
    """Test that save() followed by load() preserves all data."""

    def test_save_creates_file(self, tmp_path):
        log_path = str(tmp_path / "research.json")
        log = ResearchLog(log_path)
        log.save()
        assert os.path.exists(log_path)

    def test_roundtrip_findings(self, tmp_path):
        log_path = str(tmp_path / "research.json")
        log = ResearchLog(log_path)
        f1 = _make_finding(id="rt-f1", approach_name="Approach A")
        f2 = _make_finding(id="rt-f2", approach_name="Approach B")
        log.add_finding(f1)
        log.add_finding(f2)
        log.save()

        log2 = ResearchLog(log_path)
        log2.load()
        assert len(log2.findings) == 2
        assert log2.findings[0].id == "rt-f1"
        assert log2.findings[1].id == "rt-f2"
        assert log2.findings[0].approach_name == "Approach A"
        assert log2.findings[1].approach_name == "Approach B"

    def test_roundtrip_theoretical_bounds(self, tmp_path):
        log_path = str(tmp_path / "research.json")
        log = ResearchLog(log_path)
        log.theoretical_bounds = {"max_score": "0.95 based on XYZ bound"}
        log.save()

        log2 = ResearchLog(log_path)
        log2.load()
        assert log2.theoretical_bounds["max_score"] == "0.95 based on XYZ bound"

    def test_roundtrip_key_papers(self, tmp_path):
        log_path = str(tmp_path / "research.json")
        log = ResearchLog(log_path)
        log.key_papers = [
            {"title": "Paper A", "url": "https://example.com/a", "relevance": "core"},
        ]
        log.save()

        log2 = ResearchLog(log_path)
        log2.load()
        assert len(log2.key_papers) == 1
        assert log2.key_papers[0]["title"] == "Paper A"

    def test_roundtrip_approaches_to_avoid(self, tmp_path):
        log_path = str(tmp_path / "research.json")
        log = ResearchLog(log_path)
        log.approaches_to_avoid = ["Brute force", "Naive O(n^3)"]
        log.save()

        log2 = ResearchLog(log_path)
        log2.load()
        assert log2.approaches_to_avoid == ["Brute force", "Naive O(n^3)"]

    def test_roundtrip_all_combined(self, tmp_path):
        """Verify all collections survive a full save/load cycle."""
        log_path = str(tmp_path / "research.json")
        log = ResearchLog(log_path)
        log.add_finding(_make_finding(id="combined-1"))
        log.theoretical_bounds = {"bound_a": "value_a"}
        log.key_papers = [{"title": "P1", "url": "#", "relevance": "yes"}]
        log.approaches_to_avoid = ["avoid this"]
        log.save()

        log2 = ResearchLog(log_path)
        log2.load()
        assert len(log2.findings) == 1
        assert len(log2.theoretical_bounds) == 1
        assert len(log2.key_papers) == 1
        assert len(log2.approaches_to_avoid) == 1

    def test_save_to_nested_nonexistent_directory(self, tmp_path):
        log_path = str(tmp_path / "a" / "b" / "c" / "research.json")
        log = ResearchLog(log_path)
        log.add_finding(_make_finding(id="nested-1"))
        log.save()
        assert os.path.exists(log_path)

        log2 = ResearchLog(log_path)
        log2.load()
        assert len(log2.findings) == 1

    def test_json_file_is_valid_json(self, tmp_path):
        log_path = str(tmp_path / "research.json")
        log = ResearchLog(log_path)
        log.add_finding(_make_finding())
        log.theoretical_bounds = {"x": "y"}
        log.save()

        with open(log_path, "r") as f:
            data = json.load(f)
        assert "findings" in data
        assert isinstance(data["findings"], list)


# ---------------------------------------------------------------------------
# add_finding and mark_finding_tried
# ---------------------------------------------------------------------------

class TestResearchLogAddAndMark:
    """Test adding findings and marking them as tried."""

    def test_add_finding(self, tmp_path):
        log = ResearchLog(str(tmp_path / "research.json"))
        f = _make_finding(id="add-1")
        log.add_finding(f)
        assert len(log.findings) == 1
        assert log.findings[0].id == "add-1"

    def test_add_multiple_findings(self, tmp_path):
        log = ResearchLog(str(tmp_path / "research.json"))
        for i in range(5):
            log.add_finding(_make_finding(id=f"multi-{i}"))
        assert len(log.findings) == 5

    def test_mark_finding_tried(self, tmp_path):
        log = ResearchLog(str(tmp_path / "research.json"))
        log.add_finding(_make_finding(id="mark-1"))
        log.add_finding(_make_finding(id="mark-2"))

        log.mark_finding_tried("mark-1", 0.85)

        assert log.findings[0].was_tried is True
        assert log.findings[0].outcome_score == 0.85
        assert log.findings[1].was_tried is False
        assert log.findings[1].outcome_score is None

    def test_mark_nonexistent_finding_no_error(self, tmp_path):
        log = ResearchLog(str(tmp_path / "research.json"))
        log.add_finding(_make_finding(id="exists"))
        # Should not raise
        log.mark_finding_tried("nonexistent", 0.5)
        assert log.findings[0].was_tried is False


# ---------------------------------------------------------------------------
# get_untried_findings ordering by novelty
# ---------------------------------------------------------------------------

class TestGetUntriedFindings:
    """Test get_untried_findings filtering and ordering."""

    def test_empty_returns_empty(self, tmp_path):
        log = ResearchLog(str(tmp_path / "research.json"))
        assert log.get_untried_findings() == []

    def test_excludes_tried_findings(self, tmp_path):
        log = ResearchLog(str(tmp_path / "research.json"))
        log.add_finding(_make_finding(id="tried", was_tried=True))
        log.add_finding(_make_finding(id="untried", was_tried=False))
        result = log.get_untried_findings()
        assert len(result) == 1
        assert result[0].id == "untried"

    def test_orders_by_novelty_high_first(self, tmp_path):
        log = ResearchLog(str(tmp_path / "research.json"))
        log.add_finding(_make_finding(id="low", novelty="low"))
        log.add_finding(_make_finding(id="high", novelty="high"))
        log.add_finding(_make_finding(id="medium", novelty="medium"))
        result = log.get_untried_findings()
        assert result[0].id == "high"
        assert result[1].id == "medium"
        assert result[2].id == "low"

    def test_respects_max_items(self, tmp_path):
        log = ResearchLog(str(tmp_path / "research.json"))
        for i in range(10):
            log.add_finding(_make_finding(id=f"f-{i}", novelty="medium"))
        result = log.get_untried_findings(max_items=3)
        assert len(result) == 3

    def test_all_tried_returns_empty(self, tmp_path):
        log = ResearchLog(str(tmp_path / "research.json"))
        for i in range(5):
            log.add_finding(_make_finding(id=f"tried-{i}", was_tried=True))
        assert log.get_untried_findings() == []

    def test_unknown_novelty_sorts_last(self, tmp_path):
        log = ResearchLog(str(tmp_path / "research.json"))
        log.add_finding(_make_finding(id="unknown", novelty="unknown"))
        log.add_finding(_make_finding(id="low", novelty="low"))
        log.add_finding(_make_finding(id="high", novelty="high"))
        result = log.get_untried_findings()
        assert result[0].id == "high"
        assert result[1].id == "low"
        assert result[2].id == "unknown"


# ---------------------------------------------------------------------------
# should_research logic for all trigger modes
# ---------------------------------------------------------------------------

class TestShouldResearch:
    """Test should_research trigger logic."""

    def test_disabled_returns_false(self, tmp_path):
        log = ResearchLog(str(tmp_path / "research.json"))
        config = _FakeResearchConfig(enabled=False, trigger="always")
        assert log.should_research(1, "mild", config) is False

    def test_trigger_never_returns_false(self, tmp_path):
        log = ResearchLog(str(tmp_path / "research.json"))
        config = _FakeResearchConfig(enabled=True, trigger="never")
        assert log.should_research(1, "mild", config) is False

    def test_trigger_always_returns_true(self, tmp_path):
        log = ResearchLog(str(tmp_path / "research.json"))
        config = _FakeResearchConfig(enabled=True, trigger="always")
        assert log.should_research(1, "none", config) is True
        assert log.should_research(50, "critical", config) is True

    def test_trigger_on_stagnation_with_stagnation(self, tmp_path):
        log = ResearchLog(str(tmp_path / "research.json"))
        config = _FakeResearchConfig(enabled=True, trigger="on_stagnation")
        assert log.should_research(5, "mild", config) is True
        assert log.should_research(10, "moderate", config) is True
        assert log.should_research(20, "severe", config) is True
        assert log.should_research(30, "critical", config) is True

    def test_trigger_on_stagnation_without_stagnation(self, tmp_path):
        log = ResearchLog(str(tmp_path / "research.json"))
        config = _FakeResearchConfig(enabled=True, trigger="on_stagnation")
        assert log.should_research(5, "none", config) is False

    def test_trigger_periodic_at_interval(self, tmp_path):
        log = ResearchLog(str(tmp_path / "research.json"))
        config = _FakeResearchConfig(enabled=True, trigger="periodic", periodic_interval=10)
        assert log.should_research(10, "none", config) is True
        assert log.should_research(20, "none", config) is True
        assert log.should_research(30, "none", config) is True

    def test_trigger_periodic_off_interval(self, tmp_path):
        log = ResearchLog(str(tmp_path / "research.json"))
        config = _FakeResearchConfig(enabled=True, trigger="periodic", periodic_interval=10)
        assert log.should_research(5, "none", config) is False
        assert log.should_research(7, "none", config) is False
        assert log.should_research(15, "none", config) is False

    def test_trigger_periodic_first_three_iterations(self, tmp_path):
        log = ResearchLog(str(tmp_path / "research.json"))
        config = _FakeResearchConfig(enabled=True, trigger="periodic", periodic_interval=10)
        # First 3 iterations always trigger for periodic
        assert log.should_research(1, "none", config) is True
        assert log.should_research(2, "none", config) is True
        assert log.should_research(3, "none", config) is True
        assert log.should_research(4, "none", config) is False

    def test_unknown_trigger_returns_false(self, tmp_path):
        log = ResearchLog(str(tmp_path / "research.json"))
        config = _FakeResearchConfig(enabled=True, trigger="unknown_mode")
        assert log.should_research(5, "mild", config) is False


# ---------------------------------------------------------------------------
# format_for_prompt output
# ---------------------------------------------------------------------------

class TestFormatForPrompt:
    """Test format_for_prompt output."""

    def test_empty_log_returns_empty_string(self, tmp_path):
        log = ResearchLog(str(tmp_path / "research.json"))
        assert log.format_for_prompt() == ""

    def test_includes_untried_findings(self, tmp_path):
        log = ResearchLog(str(tmp_path / "research.json"))
        log.add_finding(_make_finding(
            id="f1",
            approach_name="Greedy Search",
            description="Use greedy heuristic",
            novelty="high",
        ))
        output = log.format_for_prompt()
        assert "## Research Findings (Untried Approaches)" in output
        assert "Greedy Search" in output
        assert "Use greedy heuristic" in output
        assert "Novelty: high" in output

    def test_includes_implementation_hint(self, tmp_path):
        log = ResearchLog(str(tmp_path / "research.json"))
        log.add_finding(_make_finding(
            id="f1",
            implementation_hint="Start with sorted array",
        ))
        output = log.format_for_prompt()
        assert "**Implementation hint:** Start with sorted array" in output

    def test_includes_source_url(self, tmp_path):
        log = ResearchLog(str(tmp_path / "research.json"))
        log.add_finding(_make_finding(
            id="f1",
            source_url="https://arxiv.org/abs/1234.5678",
        ))
        output = log.format_for_prompt()
        assert "**Source:** https://arxiv.org/abs/1234.5678" in output

    def test_excludes_tried_findings(self, tmp_path):
        log = ResearchLog(str(tmp_path / "research.json"))
        log.add_finding(_make_finding(id="tried", approach_name="Tried Approach", was_tried=True))
        log.add_finding(_make_finding(id="untried", approach_name="Fresh Approach", was_tried=False))
        output = log.format_for_prompt()
        assert "Fresh Approach" in output
        assert "Tried Approach" not in output

    def test_includes_theoretical_bounds(self, tmp_path):
        log = ResearchLog(str(tmp_path / "research.json"))
        log.theoretical_bounds = {"max_accuracy": "0.98 per Shannon limit"}
        output = log.format_for_prompt()
        assert "## Theoretical Bounds" in output
        assert "**max_accuracy:**" in output
        assert "0.98 per Shannon limit" in output

    def test_includes_key_papers(self, tmp_path):
        log = ResearchLog(str(tmp_path / "research.json"))
        log.key_papers = [
            {"title": "Optimal Sorting", "url": "https://example.com", "relevance": "Direct"},
        ]
        output = log.format_for_prompt()
        assert "## Key Papers" in output
        assert "Optimal Sorting" in output
        assert "https://example.com" in output
        assert "Direct" in output

    def test_key_papers_limited_to_three(self, tmp_path):
        log = ResearchLog(str(tmp_path / "research.json"))
        log.key_papers = [
            {"title": f"Paper {i}", "url": "#", "relevance": "yes"}
            for i in range(10)
        ]
        output = log.format_for_prompt()
        assert "Paper 0" in output
        assert "Paper 1" in output
        assert "Paper 2" in output
        assert "Paper 3" not in output

    def test_includes_approaches_to_avoid(self, tmp_path):
        log = ResearchLog(str(tmp_path / "research.json"))
        log.approaches_to_avoid = ["Bubble sort", "Brute force enumeration"]
        output = log.format_for_prompt()
        assert "## Approaches to Avoid (Research-Based)" in output
        assert "Bubble sort" in output
        assert "Brute force enumeration" in output

    def test_all_sections_combined(self, tmp_path):
        log = ResearchLog(str(tmp_path / "research.json"))
        log.add_finding(_make_finding(id="f1", approach_name="Approach A"))
        log.theoretical_bounds = {"bound": "value"}
        log.key_papers = [{"title": "P1", "url": "#", "relevance": "high"}]
        log.approaches_to_avoid = ["Bad idea"]
        output = log.format_for_prompt()
        assert "## Research Findings (Untried Approaches)" in output
        assert "## Theoretical Bounds" in output
        assert "## Key Papers" in output
        assert "## Approaches to Avoid (Research-Based)" in output
        # Sections should be separated by double newlines
        assert "\n\n" in output

    def test_only_tried_findings_still_shows_other_sections(self, tmp_path):
        log = ResearchLog(str(tmp_path / "research.json"))
        log.add_finding(_make_finding(id="tried", was_tried=True))
        log.theoretical_bounds = {"bound": "value"}
        output = log.format_for_prompt()
        # No untried findings section
        assert "## Research Findings (Untried Approaches)" not in output
        # But theoretical bounds should still appear
        assert "## Theoretical Bounds" in output

    def test_max_findings_respected(self, tmp_path):
        log = ResearchLog(str(tmp_path / "research.json"))
        for i in range(10):
            log.add_finding(_make_finding(
                id=f"f-{i}",
                approach_name=f"Approach {i}",
                novelty="medium",
            ))
        output = log.format_for_prompt(max_findings=3)
        # Count how many approach names appear
        count = sum(1 for i in range(10) if f"Approach {i}" in output)
        assert count == 3


# ---------------------------------------------------------------------------
# Multiple save/load cycles
# ---------------------------------------------------------------------------

class TestMultipleCycles:
    """Test multiple save/load cycles accumulate data."""

    def test_findings_accumulate(self, tmp_path):
        log_path = str(tmp_path / "research.json")
        log = ResearchLog(log_path)
        log.add_finding(_make_finding(id="cycle-1"))
        log.save()

        log2 = ResearchLog(log_path)
        log2.load()
        log2.add_finding(_make_finding(id="cycle-2"))
        log2.save()

        log3 = ResearchLog(log_path)
        log3.load()
        assert len(log3.findings) == 2
        assert log3.findings[0].id == "cycle-1"
        assert log3.findings[1].id == "cycle-2"

    def test_mark_finding_persists(self, tmp_path):
        log_path = str(tmp_path / "research.json")
        log = ResearchLog(log_path)
        log.add_finding(_make_finding(id="mark-persist"))
        log.save()

        log2 = ResearchLog(log_path)
        log2.load()
        log2.mark_finding_tried("mark-persist", 0.75)
        log2.save()

        log3 = ResearchLog(log_path)
        log3.load()
        assert log3.findings[0].was_tried is True
        assert log3.findings[0].outcome_score == 0.75
