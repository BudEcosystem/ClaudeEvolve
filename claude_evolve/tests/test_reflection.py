"""Tests for ReflectionEngine -- pairwise verbal gradients and long-term synthesis."""

import json
import os
import tempfile

from claude_evolve.core.artifact import Artifact
from claude_evolve.core.reflection import ReflectionEngine, ShortReflection


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _art(content: str, artifact_type: str = "python",
         metrics: dict | None = None) -> Artifact:
    """Shorthand factory for test artifacts with optional metrics."""
    a = Artifact(
        id=Artifact.generate_id(),
        content=content,
        artifact_type=artifact_type,
    )
    if metrics is not None:
        a.metrics = metrics
    return a


# ---------------------------------------------------------------------------
# ShortReflection dataclass
# ---------------------------------------------------------------------------

def test_short_reflection_dataclass():
    """ShortReflection stores all required fields."""
    sr = ShortReflection(
        better_id="b1", worse_id="w1",
        better_score=0.9, worse_score=0.3,
        insight="Better adds: numpy", iteration=5,
    )
    assert sr.better_id == "b1"
    assert sr.worse_id == "w1"
    assert sr.better_score == 0.9
    assert sr.worse_score == 0.3
    assert sr.insight == "Better adds: numpy"
    assert sr.iteration == 5


# ---------------------------------------------------------------------------
# generate_short_reflection
# ---------------------------------------------------------------------------

def test_short_reflection_identifies_added_concepts():
    """Semantic fingerprint diff surfaces concepts added by the better artifact."""
    eng = ReflectionEngine.__new__(ReflectionEngine)
    eng.short_reflections = []
    eng.long_reflection = ""
    eng.max_short = 20

    better = _art(
        "import numpy\ndef solve(): return simulated_annealing()",
        "python",
        {"combined_score": 0.8},
    )
    worse = _art(
        "def solve(): return random_search()",
        "python",
        {"combined_score": 0.3},
    )
    ref = eng.generate_short_reflection(better, worse)
    # The better artifact adds numpy and simulated_annealing concepts
    assert "numpy" in ref.insight or "simulated_annealing" in ref.insight


def test_short_reflection_captures_dropped_concepts():
    """When the worse artifact has unique concepts, they appear as 'Drops'."""
    eng = ReflectionEngine.__new__(ReflectionEngine)
    eng.short_reflections = []
    eng.long_reflection = ""
    eng.max_short = 20

    better = _art("def solve(): pass", "python", {"combined_score": 0.7})
    worse = _art(
        "import scipy\ndef solve(): return scipy.optimize.minimize(f, x0)",
        "python",
        {"combined_score": 0.2},
    )
    ref = eng.generate_short_reflection(better, worse)
    assert "Drops" in ref.insight


def test_short_reflection_fallback_line_count():
    """When fingerprints are identical, fallback to line count comparison."""
    eng = ReflectionEngine.__new__(ReflectionEngine)
    eng.short_reflections = []
    eng.long_reflection = ""
    eng.max_short = 20

    # Two artifacts with identical semantic fingerprints (plain text, short,
    # no repeated words of length >= 5)
    better = _art("hi", "text", {"combined_score": 0.5})
    worse = _art("ho", "text", {"combined_score": 0.1})
    ref = eng.generate_short_reflection(better, worse)
    assert "lines vs" in ref.insight


def test_short_reflection_extracts_scores_from_metrics():
    """Scores are pulled from artifact.metrics with None safety."""
    eng = ReflectionEngine.__new__(ReflectionEngine)
    eng.short_reflections = []
    eng.long_reflection = ""
    eng.max_short = 20

    better = _art("import numpy\ndef f(): pass", "python", {"combined_score": 0.85})
    worse = _art("def g(): pass", "python", {"combined_score": 0.15})
    ref = eng.generate_short_reflection(better, worse)
    assert ref.better_score == 0.85
    assert ref.worse_score == 0.15


def test_short_reflection_none_metrics_defaults_to_zero():
    """If metrics is None, scores default to 0.0."""
    eng = ReflectionEngine.__new__(ReflectionEngine)
    eng.short_reflections = []
    eng.long_reflection = ""
    eng.max_short = 20

    better = _art("import torch\ndef train(): pass", "python")
    better.metrics = None
    worse = _art("def noop(): pass", "python")
    worse.metrics = None
    ref = eng.generate_short_reflection(better, worse)
    assert ref.better_score == 0.0
    assert ref.worse_score == 0.0


def test_short_reflections_capped_at_max_short():
    """The short_reflections list never exceeds max_short."""
    with tempfile.TemporaryDirectory() as d:
        eng = ReflectionEngine(d, max_short=3)
        for i in range(5):
            better = _art(f"import mod{i}\ndef f{i}(): pass", "python",
                          {"combined_score": float(i) / 10})
            worse = _art("def baseline(): pass", "python",
                         {"combined_score": 0.0})
            eng.generate_short_reflection(better, worse)
        assert len(eng.short_reflections) == 3


# ---------------------------------------------------------------------------
# accumulate_long_reflection
# ---------------------------------------------------------------------------

def test_long_reflection_synthesizes_short():
    """Long reflection is produced at synthesis_interval multiples."""
    with tempfile.TemporaryDirectory() as d:
        eng = ReflectionEngine(d, synthesis_interval=2)
        eng.short_reflections = [
            ShortReflection("a", "b", 0.8, 0.3,
                            "Better adds: numpy, simulated_annealing", 1),
            ShortReflection("c", "d", 0.9, 0.4,
                            "Better adds: numpy, gradient", 2),
            ShortReflection("e", "f", 0.7, 0.2,
                            "Better adds: scipy, linear_programming", 3),
        ]
        result = eng.accumulate_long_reflection(4)  # Multiple of 2
        assert result is not None
        assert "numpy" in result  # Most common concept


def test_long_reflection_skips_non_interval():
    """Long reflection returns None when not on a synthesis interval."""
    with tempfile.TemporaryDirectory() as d:
        eng = ReflectionEngine(d, synthesis_interval=5)
        eng.short_reflections = [
            ShortReflection("a", "b", 0.8, 0.3, "Better adds: numpy", 1),
        ]
        assert eng.accumulate_long_reflection(3) is None
        assert eng.accumulate_long_reflection(0) is None


def test_long_reflection_empty_short_returns_none():
    """If there are no short reflections, accumulation returns None."""
    with tempfile.TemporaryDirectory() as d:
        eng = ReflectionEngine(d, synthesis_interval=2)
        assert eng.accumulate_long_reflection(2) is None


# ---------------------------------------------------------------------------
# format_for_prompt
# ---------------------------------------------------------------------------

def test_format_for_prompt_with_both():
    """format_for_prompt includes both verbal gradient and accumulated wisdom."""
    with tempfile.TemporaryDirectory() as d:
        eng = ReflectionEngine(d)
        eng.short_reflections = [
            ShortReflection("a", "b", 0.8, 0.3, "Better adds: numpy", 1),
        ]
        eng.long_reflection = "Key insight: use numpy"
        text = eng.format_for_prompt()
        assert "numpy" in text
        assert "Key insight" in text
        assert "Verbal Gradient" in text
        assert "Accumulated Wisdom" in text


def test_format_for_prompt_empty():
    """An engine with no reflections returns an empty string."""
    with tempfile.TemporaryDirectory() as d:
        eng = ReflectionEngine(d)
        assert eng.format_for_prompt() == ""


def test_format_for_prompt_only_short():
    """When only short reflections exist, only verbal gradient section appears."""
    with tempfile.TemporaryDirectory() as d:
        eng = ReflectionEngine(d)
        eng.short_reflections = [
            ShortReflection("a", "b", 0.8, 0.3, "Better adds: torch", 1),
        ]
        text = eng.format_for_prompt()
        assert "Verbal Gradient" in text
        assert "torch" in text
        assert "Accumulated Wisdom" not in text


# ---------------------------------------------------------------------------
# save / load_from
# ---------------------------------------------------------------------------

def test_save_and_load():
    """Round-trip serialization preserves all data."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "reflections.json")
        eng = ReflectionEngine(d)
        eng.short_reflections = [
            ShortReflection("a", "b", 0.8, 0.3, "test insight", 1),
        ]
        eng.long_reflection = "accumulated wisdom"
        eng.save(path)

        loaded = ReflectionEngine.load_from(path)
        assert len(loaded.short_reflections) == 1
        assert loaded.short_reflections[0].insight == "test insight"
        assert loaded.short_reflections[0].better_score == 0.8
        assert loaded.long_reflection == "accumulated wisdom"


def test_load_from_missing_file():
    """Loading from a nonexistent path returns a fresh engine."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "does_not_exist.json")
        loaded = ReflectionEngine.load_from(path)
        assert len(loaded.short_reflections) == 0
        assert loaded.long_reflection == ""


def test_save_creates_parent_dirs():
    """save() creates intermediate directories if needed."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "sub", "dir", "reflections.json")
        eng = ReflectionEngine(d)
        eng.short_reflections = [
            ShortReflection("x", "y", 0.5, 0.1, "insight", 2),
        ]
        eng.save(path)
        assert os.path.exists(path)

        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert len(data["short_reflections"]) == 1


def test_save_load_multiple_reflections():
    """Multiple short reflections and a long reflection survive serialization."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "reflections.json")
        eng = ReflectionEngine(d, synthesis_interval=2)
        eng.short_reflections = [
            ShortReflection("a", "b", 0.8, 0.3, "Better adds: numpy", 1),
            ShortReflection("c", "d", 0.9, 0.4, "Better adds: scipy", 2),
            ShortReflection("e", "f", 0.7, 0.2, "Better adds: torch", 3),
        ]
        eng.long_reflection = "Key patterns: numpy, scipy"
        eng.save(path)

        loaded = ReflectionEngine.load_from(path)
        assert len(loaded.short_reflections) == 3
        assert loaded.short_reflections[1].better_id == "c"
        assert loaded.long_reflection == "Key patterns: numpy, scipy"
