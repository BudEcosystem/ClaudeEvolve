# Claude Evolve - OpenEvolve as a Claude Code Plugin

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a hybrid Claude Code plugin that implements OpenEvolve's evolutionary algorithm natively within Claude Code sessions, using Claude as both the LLM and an autonomous research agent, with a Ralph-style loop mechanism and a standalone Python package for deterministic evolution logic.

**Architecture:** Three-layer hybrid system. Layer 1: Shell-based loop (Ralph-style Stop hook manages iteration lifecycle). Layer 2: Standalone Python package (`claude_evolve`) extracted from OpenEvolve providing deterministic MAP-Elites database, evaluation, selection, and prompt construction. Layer 3: Claude Code skill empowering full autonomous exploration per iteration (web research, subagents, literature review, TDD, debugging).

**Tech Stack:** Python 3.10+ (standalone package with Click CLI), Bash (hooks/scripts), Markdown (skills/commands/agents), Claude Code Plugin system

---

## Phase 1: Python Package Core (`claude_evolve`)

### Task 1: Project Scaffold + Artifact Dataclass

**Files:**
- Create: `claude_evolve/pyproject.toml`
- Create: `claude_evolve/claude_evolve/__init__.py`
- Create: `claude_evolve/claude_evolve/core/__init__.py`
- Create: `claude_evolve/claude_evolve/core/artifact.py`
- Test: `claude_evolve/tests/__init__.py`
- Test: `claude_evolve/tests/test_artifact.py`

**Step 1: Write the failing test**

```python
# tests/test_artifact.py
import json
import time
import unittest
from claude_evolve.core.artifact import Artifact


class TestArtifact(unittest.TestCase):
    def test_create_artifact_with_defaults(self):
        a = Artifact(id="test-1", content="print('hello')")
        self.assertEqual(a.id, "test-1")
        self.assertEqual(a.content, "print('hello')")
        self.assertEqual(a.artifact_type, "python")
        self.assertIsNone(a.parent_id)
        self.assertEqual(a.generation, 0)
        self.assertEqual(a.metrics, {})
        self.assertAlmostEqual(a.timestamp, time.time(), delta=2)

    def test_create_artifact_with_all_fields(self):
        a = Artifact(
            id="test-2",
            content="# prompt text",
            artifact_type="markdown",
            parent_id="test-1",
            generation=3,
            iteration_found=7,
            metrics={"score": 0.85, "clarity": 0.9},
            complexity=42.0,
            diversity=0.7,
            metadata={"island": 2, "changes": "improved clarity"},
            changes_description="Rewrote introduction paragraph",
            eval_artifacts={"stderr": "warning: unused var"},
        )
        self.assertEqual(a.artifact_type, "markdown")
        self.assertEqual(a.parent_id, "test-1")
        self.assertEqual(a.generation, 3)
        self.assertEqual(a.metrics["score"], 0.85)

    def test_to_dict_roundtrip(self):
        a = Artifact(
            id="rt-1",
            content="code here",
            artifact_type="python",
            metrics={"combined_score": 0.5},
        )
        d = a.to_dict()
        self.assertIsInstance(d, dict)
        self.assertEqual(d["id"], "rt-1")
        self.assertEqual(d["content"], "code here")
        b = Artifact.from_dict(d)
        self.assertEqual(b.id, a.id)
        self.assertEqual(b.content, a.content)
        self.assertEqual(b.metrics, a.metrics)

    def test_from_dict_ignores_unknown_fields(self):
        d = {"id": "u-1", "content": "x", "unknown_field": 42}
        a = Artifact.from_dict(d)
        self.assertEqual(a.id, "u-1")

    def test_from_dict_backward_compat_code_field(self):
        """OpenEvolve uses 'code' field; we accept both 'code' and 'content'"""
        d = {"id": "bc-1", "code": "print(1)", "language": "python"}
        a = Artifact.from_dict(d)
        self.assertEqual(a.content, "print(1)")
        self.assertEqual(a.artifact_type, "python")

    def test_json_serialization(self):
        a = Artifact(id="j-1", content="test", metrics={"s": 1.0})
        j = json.dumps(a.to_dict())
        d = json.loads(j)
        b = Artifact.from_dict(d)
        self.assertEqual(b.id, "j-1")


if __name__ == "__main__":
    unittest.main()
```

**Step 2: Run test to verify it fails**

Run: `cd claude_evolve && python -m pytest tests/test_artifact.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'claude_evolve'"

**Step 3: Write pyproject.toml and minimal package**

```toml
# pyproject.toml
[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "claude-evolve"
version = "0.1.0"
description = "Evolutionary code/artifact optimization for Claude Code - extracted from OpenEvolve"
requires-python = ">=3.10"
dependencies = [
    "click>=8.0",
    "numpy>=1.24",
    "pyyaml>=6.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-asyncio>=0.21",
]

[project.scripts]
claude-evolve = "claude_evolve.cli:main"

[tool.pytest.ini_options]
testpaths = ["tests"]
```

**Step 4: Write Artifact dataclass**

```python
# claude_evolve/core/artifact.py
"""
Artifact dataclass for Claude Evolve.

Generalization of OpenEvolve's Program dataclass.
An artifact can be any text: code, prompts, configs, prose.
"""
import time
import uuid
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List, Optional


@dataclass
class Artifact:
    """Represents an evolvable artifact in the database."""

    # Identification
    id: str
    content: str
    artifact_type: str = "python"

    # Evolution lineage
    parent_id: Optional[str] = None
    generation: int = 0
    timestamp: float = field(default_factory=time.time)
    iteration_found: int = 0

    # Performance metrics
    metrics: Dict[str, float] = field(default_factory=dict)

    # MAP-Elites features
    complexity: float = 0.0
    diversity: float = 0.0

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    changes_description: str = ""

    # Evaluation artifacts (stderr, critic feedback, etc.)
    eval_artifacts: Optional[Dict[str, str]] = None

    # Prompt logging (optional)
    prompts: Optional[Dict[str, Any]] = None

    # Embedding for novelty (optional)
    embedding: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Artifact":
        # Backward compat: OpenEvolve uses 'code'/'language'
        if "code" in data and "content" not in data:
            data = {**data, "content": data.pop("code")}
        if "language" in data and "artifact_type" not in data:
            data = {**data, "artifact_type": data.pop("language")}

        # Handle missing changes_description
        if "changes_description" not in data:
            metadata = data.get("metadata") or {}
            if isinstance(metadata, dict):
                data = {
                    **data,
                    "changes_description": metadata.get("changes_description")
                    or metadata.get("changes", ""),
                }
            else:
                data = {**data, "changes_description": ""}

        valid_fields = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)

    @staticmethod
    def generate_id() -> str:
        return str(uuid.uuid4())
```

**Step 5: Create __init__.py files**

```python
# claude_evolve/__init__.py
"""Claude Evolve - Evolutionary artifact optimization for Claude Code."""
__version__ = "0.1.0"

# claude_evolve/core/__init__.py
# tests/__init__.py
```

**Step 6: Install in dev mode and run tests**

Run: `cd claude_evolve && pip install -e ".[dev]" && python -m pytest tests/test_artifact.py -v`
Expected: All 6 tests PASS

**Step 7: Commit**

```bash
git add claude_evolve/
git commit -m "feat: scaffold claude_evolve package with Artifact dataclass"
```

---

### Task 2: Utility Modules (code_utils, metrics_utils, format_utils)

**Files:**
- Create: `claude_evolve/claude_evolve/utils/__init__.py`
- Create: `claude_evolve/claude_evolve/utils/code_utils.py`
- Create: `claude_evolve/claude_evolve/utils/metrics_utils.py`
- Create: `claude_evolve/claude_evolve/utils/format_utils.py`
- Test: `claude_evolve/tests/test_code_utils.py`
- Test: `claude_evolve/tests/test_metrics_utils.py`

**Step 1: Write failing tests for code_utils**

```python
# tests/test_code_utils.py
import unittest
from claude_evolve.utils.code_utils import (
    apply_diff,
    extract_diffs,
    parse_full_rewrite,
    calculate_edit_distance,
    extract_code_language,
    format_diff_summary,
    apply_diff_blocks,
)


class TestExtractDiffs(unittest.TestCase):
    def test_single_diff_block(self):
        text = """Some text
<<<<<<< SEARCH
old code
=======
new code
>>>>>>> REPLACE
More text"""
        blocks = extract_diffs(text)
        self.assertEqual(len(blocks), 1)
        self.assertIn("old code", blocks[0]["search"])
        self.assertIn("new code", blocks[0]["replace"])

    def test_multiple_diff_blocks(self):
        text = """<<<<<<< SEARCH
a = 1
=======
a = 2
>>>>>>> REPLACE

<<<<<<< SEARCH
b = 3
=======
b = 4
>>>>>>> REPLACE"""
        blocks = extract_diffs(text)
        self.assertEqual(len(blocks), 2)

    def test_no_diffs(self):
        blocks = extract_diffs("just plain text")
        self.assertEqual(len(blocks), 0)


class TestApplyDiff(unittest.TestCase):
    def test_apply_single_diff(self):
        original = "x = 1\ny = 2\nz = 3"
        diff_text = """<<<<<<< SEARCH
y = 2
=======
y = 20
>>>>>>> REPLACE"""
        result = apply_diff(original, diff_text)
        self.assertIn("y = 20", result)
        self.assertIn("x = 1", result)
        self.assertIn("z = 3", result)

    def test_apply_diff_no_match(self):
        original = "a = 1"
        diff_text = """<<<<<<< SEARCH
nonexistent
=======
replacement
>>>>>>> REPLACE"""
        result = apply_diff(original, diff_text)
        self.assertEqual(result, original)


class TestApplyDiffBlocks(unittest.TestCase):
    def test_apply_multiple_blocks(self):
        original = "a = 1\nb = 2"
        blocks = [
            {"search": "a = 1", "replace": "a = 10"},
            {"search": "b = 2", "replace": "b = 20"},
        ]
        result, count = apply_diff_blocks(original, blocks)
        self.assertIn("a = 10", result)
        self.assertIn("b = 20", result)
        self.assertEqual(count, 2)


class TestParseFullRewrite(unittest.TestCase):
    def test_extract_python_code(self):
        text = """Here's the improved code:
```python
def hello():
    return "world"
```"""
        code = parse_full_rewrite(text, "python")
        self.assertIn("def hello():", code)

    def test_no_code_block(self):
        code = parse_full_rewrite("no code here", "python")
        self.assertIsNone(code)


class TestCalculateEditDistance(unittest.TestCase):
    def test_identical(self):
        self.assertEqual(calculate_edit_distance("abc", "abc"), 0)

    def test_different(self):
        dist = calculate_edit_distance("abc", "abd")
        self.assertGreater(dist, 0)

    def test_empty(self):
        dist = calculate_edit_distance("", "abc")
        self.assertEqual(dist, 3)


class TestExtractCodeLanguage(unittest.TestCase):
    def test_python(self):
        code = "import numpy\ndef func():\n    pass"
        self.assertEqual(extract_code_language(code), "python")

    def test_rust(self):
        code = "fn main() {\n    println!(\"hello\");\n}"
        self.assertEqual(extract_code_language(code), "rust")


class TestFormatDiffSummary(unittest.TestCase):
    def test_summary_format(self):
        blocks = [{"search": "old", "replace": "new"}]
        summary = format_diff_summary(blocks)
        self.assertIsInstance(summary, str)
        self.assertTrue(len(summary) > 0)


if __name__ == "__main__":
    unittest.main()
```

**Step 2: Write failing tests for metrics_utils**

```python
# tests/test_metrics_utils.py
import unittest
from claude_evolve.utils.metrics_utils import (
    safe_numeric_average,
    safe_numeric_sum,
    get_fitness_score,
    format_feature_coordinates,
)


class TestSafeNumericAverage(unittest.TestCase):
    def test_all_numeric(self):
        self.assertAlmostEqual(safe_numeric_average({"a": 1.0, "b": 3.0}), 2.0)

    def test_mixed_types(self):
        result = safe_numeric_average({"a": 1.0, "b": "text", "c": 3.0})
        self.assertAlmostEqual(result, 2.0)

    def test_empty(self):
        self.assertEqual(safe_numeric_average({}), 0.0)

    def test_ignores_booleans(self):
        result = safe_numeric_average({"a": 1.0, "b": True})
        self.assertAlmostEqual(result, 1.0)


class TestSafeNumericSum(unittest.TestCase):
    def test_all_numeric(self):
        self.assertAlmostEqual(safe_numeric_sum({"a": 1.0, "b": 2.0}), 3.0)

    def test_mixed(self):
        result = safe_numeric_sum({"a": 5.0, "b": "skip"})
        self.assertAlmostEqual(result, 5.0)


class TestGetFitnessScore(unittest.TestCase):
    def test_combined_score_preferred(self):
        metrics = {"combined_score": 0.9, "other": 0.5}
        self.assertAlmostEqual(get_fitness_score(metrics, []), 0.9)

    def test_excludes_feature_dimensions(self):
        metrics = {"combined_score": 0.8, "complexity": 100, "diversity": 0.5}
        score = get_fitness_score(metrics, ["complexity", "diversity"])
        self.assertAlmostEqual(score, 0.8)

    def test_fallback_to_average(self):
        metrics = {"accuracy": 0.8, "speed": 0.6}
        score = get_fitness_score(metrics, [])
        self.assertAlmostEqual(score, 0.7)

    def test_empty_metrics(self):
        self.assertEqual(get_fitness_score({}, []), 0.0)


class TestFormatFeatureCoordinates(unittest.TestCase):
    def test_with_features(self):
        metrics = {"complexity": 100, "diversity": 0.5, "score": 0.9}
        result = format_feature_coordinates(metrics, ["complexity", "diversity"])
        self.assertIn("complexity", result)
        self.assertIn("diversity", result)

    def test_no_features(self):
        result = format_feature_coordinates({"a": 1}, [])
        self.assertEqual(result, "")


if __name__ == "__main__":
    unittest.main()
```

**Step 3: Run tests to verify they fail**

Run: `cd claude_evolve && python -m pytest tests/test_code_utils.py tests/test_metrics_utils.py -v`
Expected: FAIL with ModuleNotFoundError

**Step 4: Implement code_utils.py**

Extract from `/home/bud/Desktop/claudeEvolve/openevolve/openevolve/utils/code_utils.py` (all 299 lines). Refactor to use the default diff pattern `<<<<<<< SEARCH` as constant. Keep all functions: `extract_diffs`, `apply_diff`, `apply_diff_blocks`, `parse_full_rewrite`, `calculate_edit_distance`, `extract_code_language`, `format_diff_summary`, `split_diffs_by_target`, `parse_evolve_blocks`.

**Step 5: Implement metrics_utils.py**

Extract from `/home/bud/Desktop/claudeEvolve/openevolve/openevolve/utils/metrics_utils.py` (all 145 lines). Keep all functions: `safe_numeric_average`, `safe_numeric_sum`, `get_fitness_score`, `format_feature_coordinates`.

**Step 6: Implement format_utils.py**

Extract `format_metrics_safe()` from `/home/bud/Desktop/claudeEvolve/openevolve/openevolve/utils/format_utils.py`.

**Step 7: Run tests**

Run: `cd claude_evolve && python -m pytest tests/test_code_utils.py tests/test_metrics_utils.py -v`
Expected: All tests PASS

**Step 8: Commit**

```bash
git add claude_evolve/claude_evolve/utils/ claude_evolve/tests/test_code_utils.py claude_evolve/tests/test_metrics_utils.py
git commit -m "feat: add code_utils, metrics_utils, format_utils extracted from OpenEvolve"
```

---

### Task 3: Configuration System

**Files:**
- Create: `claude_evolve/claude_evolve/config.py`
- Test: `claude_evolve/tests/test_config.py`

**Step 1: Write the failing test**

```python
# tests/test_config.py
import os
import tempfile
import unittest
import yaml
from claude_evolve.config import (
    Config,
    DatabaseConfig,
    EvaluatorConfig,
    PromptConfig,
    EvolutionConfig,
)


class TestDatabaseConfig(unittest.TestCase):
    def test_defaults(self):
        c = DatabaseConfig()
        self.assertEqual(c.population_size, 1000)
        self.assertEqual(c.num_islands, 5)
        self.assertAlmostEqual(c.elite_selection_ratio, 0.1)
        self.assertAlmostEqual(c.exploration_ratio, 0.2)
        self.assertAlmostEqual(c.exploitation_ratio, 0.7)
        self.assertEqual(c.feature_dimensions, ["complexity", "diversity"])
        self.assertEqual(c.feature_bins, 10)
        self.assertEqual(c.migration_interval, 50)


class TestEvaluatorConfig(unittest.TestCase):
    def test_defaults(self):
        c = EvaluatorConfig()
        self.assertEqual(c.timeout, 300)
        self.assertTrue(c.cascade_evaluation)
        self.assertEqual(c.mode, "script")

    def test_critic_mode(self):
        c = EvaluatorConfig(mode="critic")
        self.assertEqual(c.mode, "critic")


class TestConfig(unittest.TestCase):
    def test_defaults(self):
        c = Config()
        self.assertEqual(c.max_iterations, 50)
        self.assertIsInstance(c.database, DatabaseConfig)
        self.assertIsInstance(c.evaluator, EvaluatorConfig)

    def test_from_dict(self):
        d = {
            "max_iterations": 100,
            "target_score": 0.95,
            "database": {"num_islands": 3},
            "evaluator": {"timeout": 600, "mode": "critic"},
        }
        c = Config.from_dict(d)
        self.assertEqual(c.max_iterations, 100)
        self.assertAlmostEqual(c.target_score, 0.95)
        self.assertEqual(c.database.num_islands, 3)
        self.assertEqual(c.evaluator.mode, "critic")

    def test_from_yaml_file(self):
        config_data = {
            "max_iterations": 30,
            "artifact_type": "markdown",
            "database": {"population_size": 500},
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(config_data, f)
            f.flush()
            c = Config.from_yaml(f.name)
        os.unlink(f.name)
        self.assertEqual(c.max_iterations, 30)
        self.assertEqual(c.artifact_type, "markdown")
        self.assertEqual(c.database.population_size, 500)

    def test_to_dict_roundtrip(self):
        c = Config(max_iterations=77, target_score=0.8)
        d = c.to_dict()
        c2 = Config.from_dict(d)
        self.assertEqual(c2.max_iterations, 77)
        self.assertAlmostEqual(c2.target_score, 0.8)

    def test_env_var_resolution(self):
        os.environ["TEST_EVOLVE_VAR"] = "resolved_value"
        d = {"output_dir": "${TEST_EVOLVE_VAR}/output"}
        c = Config.from_dict(d)
        self.assertIn("resolved_value", c.output_dir)
        del os.environ["TEST_EVOLVE_VAR"]


class TestEvolutionConfig(unittest.TestCase):
    def test_diff_based_default(self):
        c = EvolutionConfig()
        self.assertTrue(c.diff_based)
        self.assertEqual(c.max_content_length, 50000)


if __name__ == "__main__":
    unittest.main()
```

**Step 2: Run test to verify failure**

Run: `cd claude_evolve && python -m pytest tests/test_config.py -v`
Expected: FAIL

**Step 3: Implement config.py**

Extract from `/home/bud/Desktop/claudeEvolve/openevolve/openevolve/config.py`. Remove all LLM-related config (`LLMConfig`, `LLMModelConfig`). Add:
- `EvolutionConfig` with `diff_based`, `max_content_length`, `diff_pattern`
- `EvaluatorConfig.mode` field: `"script"`, `"critic"`, `"hybrid"`
- `Config.target_score: Optional[float]` for early stopping
- `Config.artifact_type: str` for artifact type detection
- `_resolve_env_var()` helper from OpenEvolve
- `from_yaml()`, `from_dict()`, `to_dict()`, `to_yaml()` methods

**Step 4: Run tests**

Run: `cd claude_evolve && python -m pytest tests/test_config.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add claude_evolve/claude_evolve/config.py claude_evolve/tests/test_config.py
git commit -m "feat: add configuration system with database, evaluator, and evolution configs"
```

---

### Task 4: MAP-Elites Program Database

**Files:**
- Create: `claude_evolve/claude_evolve/core/database.py`
- Test: `claude_evolve/tests/test_database.py`

This is the largest and most critical module. Extract from OpenEvolve's `database.py` (~1500 lines of core logic).

**Step 1: Write failing tests**

```python
# tests/test_database.py
import json
import os
import tempfile
import unittest
from claude_evolve.core.artifact import Artifact
from claude_evolve.core.database import ArtifactDatabase
from claude_evolve.config import DatabaseConfig


class TestArtifactDatabaseInit(unittest.TestCase):
    def test_create_empty_database(self):
        db = ArtifactDatabase(DatabaseConfig())
        self.assertEqual(db.size(), 0)
        self.assertIsNone(db.get_best())

    def test_add_initial_artifact(self):
        db = ArtifactDatabase(DatabaseConfig())
        a = Artifact(id="init", content="print(1)", metrics={"combined_score": 0.5})
        db.add(a)
        self.assertEqual(db.size(), 1)
        best = db.get_best()
        self.assertEqual(best.id, "init")


class TestMAPElites(unittest.TestCase):
    def setUp(self):
        self.db = ArtifactDatabase(
            DatabaseConfig(num_islands=2, feature_bins=5, population_size=100)
        )
        # Add seed artifact
        self.seed = Artifact(
            id="seed",
            content="x = 1",
            metrics={"combined_score": 0.3},
            complexity=10.0,
            diversity=0.5,
        )
        self.db.add(self.seed)

    def test_add_better_program_updates_best(self):
        better = Artifact(
            id="better",
            content="x = optimized()",
            parent_id="seed",
            generation=1,
            metrics={"combined_score": 0.8},
        )
        self.db.add(better)
        self.assertEqual(self.db.get_best().id, "better")

    def test_feature_grid_placement(self):
        """Programs with different features go to different grid cells"""
        a1 = Artifact(id="a1", content="short", metrics={"combined_score": 0.5},
                      complexity=10.0, diversity=0.2)
        a2 = Artifact(id="a2", content="a very long program " * 100,
                      metrics={"combined_score": 0.5},
                      complexity=500.0, diversity=0.9)
        self.db.add(a1)
        self.db.add(a2)
        # Both should be in the database (different cells)
        self.assertGreaterEqual(self.db.size(), 2)

    def test_get_top_programs(self):
        for i in range(10):
            a = Artifact(
                id=f"p{i}",
                content=f"prog {i}",
                metrics={"combined_score": i / 10.0},
            )
            self.db.add(a)
        top = self.db.get_top_programs(3)
        self.assertEqual(len(top), 3)
        # Should be sorted by score descending
        self.assertGreaterEqual(
            top[0].metrics["combined_score"],
            top[1].metrics["combined_score"],
        )


class TestIslandEvolution(unittest.TestCase):
    def setUp(self):
        self.db = ArtifactDatabase(
            DatabaseConfig(num_islands=3, migration_interval=5, migration_rate=0.2)
        )
        for i in range(9):
            a = Artifact(
                id=f"p{i}",
                content=f"island prog {i}",
                metrics={"combined_score": (i + 1) / 10.0},
            )
            self.db.add(a)

    def test_island_rotation(self):
        island0 = self.db.current_island
        self.db.next_island()
        self.assertNotEqual(self.db.current_island, island0)

    def test_migration_check(self):
        self.assertFalse(self.db.should_migrate())
        for _ in range(5):
            self.db.increment_island_generation()
        self.assertTrue(self.db.should_migrate())


class TestSampling(unittest.TestCase):
    def setUp(self):
        self.db = ArtifactDatabase(
            DatabaseConfig(num_islands=2, population_size=100)
        )
        for i in range(20):
            a = Artifact(
                id=f"s{i}",
                content=f"sample prog {i}" * (i + 1),
                metrics={"combined_score": i / 20.0},
            )
            self.db.add(a)

    def test_sample_returns_parent_and_inspirations(self):
        parent, inspirations = self.db.sample(num_inspirations=3)
        self.assertIsInstance(parent, Artifact)
        self.assertIsInstance(inspirations, list)
        self.assertLessEqual(len(inspirations), 3)

    def test_sample_from_populated_db(self):
        for _ in range(10):
            parent, _ = self.db.sample()
            self.assertIsNotNone(parent)
            self.assertIsNotNone(parent.content)


class TestPersistence(unittest.TestCase):
    def test_save_and_load_json(self):
        db = ArtifactDatabase(DatabaseConfig(num_islands=2))
        for i in range(5):
            a = Artifact(
                id=f"persist-{i}",
                content=f"code {i}",
                metrics={"combined_score": i / 5.0},
            )
            db.add(a)

        with tempfile.TemporaryDirectory() as tmpdir:
            db.save(tmpdir)
            db2 = ArtifactDatabase(DatabaseConfig(num_islands=2))
            db2.load(tmpdir)
            self.assertEqual(db2.size(), db.size())
            self.assertEqual(
                db2.get_best().id, db.get_best().id
            )

    def test_load_nonexistent_is_safe(self):
        db = ArtifactDatabase(DatabaseConfig())
        db.load("/nonexistent/path")
        self.assertEqual(db.size(), 0)


class TestArtifacts(unittest.TestCase):
    def test_store_and_retrieve_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = DatabaseConfig(artifacts_base_path=tmpdir)
            db = ArtifactDatabase(config)
            a = Artifact(id="art-1", content="code", metrics={"combined_score": 0.5})
            db.add(a)
            db.store_artifacts("art-1", {"stderr": "warning: x unused", "profile": "fast"})
            retrieved = db.get_artifacts("art-1")
            self.assertEqual(retrieved["stderr"], "warning: x unused")

    def test_missing_artifacts_returns_none(self):
        db = ArtifactDatabase(DatabaseConfig())
        self.assertIsNone(db.get_artifacts("nonexistent"))


if __name__ == "__main__":
    unittest.main()
```

**Step 2: Run tests to verify failure**

Run: `cd claude_evolve && python -m pytest tests/test_database.py -v`
Expected: FAIL

**Step 3: Implement database.py**

Extract from `/home/bud/Desktop/claudeEvolve/openevolve/openevolve/database.py`. This is ~1500 lines. Key refactoring:
- Rename `Program` references to `Artifact`
- Rename `ProgramDatabase` to `ArtifactDatabase`
- Rename `code` field references to `content`
- Keep all MAP-Elites logic: feature grid, island populations, migration
- Keep all selection logic: elite, exploration, exploitation sampling
- Keep all persistence: JSON save/load (not pickle)
- Keep artifact storage system
- Remove LLM novelty judgment (`_llm_judge_novelty`, `_cosine_similarity`)
- Use JSON for checkpoints (not pickle) for transparency

**Step 4: Run tests**

Run: `cd claude_evolve && python -m pytest tests/test_database.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add claude_evolve/claude_evolve/core/database.py claude_evolve/tests/test_database.py
git commit -m "feat: add MAP-Elites ArtifactDatabase with island evolution"
```

---

### Task 5: Evaluator (Script Mode)

**Files:**
- Create: `claude_evolve/claude_evolve/core/evaluator.py`
- Test: `claude_evolve/tests/test_evaluator.py`

**Step 1: Write failing tests**

```python
# tests/test_evaluator.py
import asyncio
import os
import tempfile
import unittest
from claude_evolve.core.evaluator import Evaluator
from claude_evolve.config import EvaluatorConfig


class TestEvaluatorScriptMode(unittest.TestCase):
    def setUp(self):
        # Create a simple evaluator script
        self.tmpdir = tempfile.mkdtemp()
        self.eval_script = os.path.join(self.tmpdir, "evaluator.py")
        with open(self.eval_script, "w") as f:
            f.write('''
def evaluate(artifact_path):
    with open(artifact_path) as f:
        content = f.read()
    length = len(content)
    return {
        "combined_score": min(length / 100.0, 1.0),
        "length": float(length),
    }
''')
        self.candidate = os.path.join(self.tmpdir, "candidate.py")
        with open(self.candidate, "w") as f:
            f.write("x = 1\n" * 20)

    def test_evaluate_returns_metrics(self):
        config = EvaluatorConfig(mode="script", timeout=30)
        evaluator = Evaluator(config, self.eval_script)
        metrics = asyncio.run(evaluator.evaluate(self.candidate))
        self.assertIn("combined_score", metrics)
        self.assertIsInstance(metrics["combined_score"], float)

    def test_evaluate_bad_script_returns_zero(self):
        bad_script = os.path.join(self.tmpdir, "bad_eval.py")
        with open(bad_script, "w") as f:
            f.write("def evaluate(path): raise Exception('boom')")
        config = EvaluatorConfig(mode="script", timeout=10)
        evaluator = Evaluator(config, bad_script)
        metrics = asyncio.run(evaluator.evaluate(self.candidate))
        self.assertEqual(metrics.get("combined_score", 0), 0.0)

    def test_evaluate_timeout(self):
        slow_script = os.path.join(self.tmpdir, "slow_eval.py")
        with open(slow_script, "w") as f:
            f.write('''
import time
def evaluate(path):
    time.sleep(100)
    return {"combined_score": 1.0}
''')
        config = EvaluatorConfig(mode="script", timeout=2)
        evaluator = Evaluator(config, slow_script)
        metrics = asyncio.run(evaluator.evaluate(self.candidate))
        self.assertEqual(metrics.get("combined_score", 0), 0.0)


class TestEvaluatorMetricsPassthrough(unittest.TestCase):
    def test_passthrough_mode(self):
        """Critic mode: metrics are pre-computed by Claude, just passed through"""
        config = EvaluatorConfig(mode="critic")
        evaluator = Evaluator(config, None)
        metrics = {"combined_score": 0.85, "flaws": 2}
        result = asyncio.run(evaluator.evaluate_with_metrics(metrics))
        self.assertEqual(result["combined_score"], 0.85)


class TestCascadeEvaluation(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.eval_script = os.path.join(self.tmpdir, "cascade_eval.py")
        with open(self.eval_script, "w") as f:
            f.write('''
def evaluate_stage1(path):
    return {"combined_score": 0.6, "validity": 1.0}

def evaluate_stage2(path):
    return {"combined_score": 0.8, "validity": 1.0, "performance": 0.7}

def evaluate(path):
    return {"combined_score": 0.9, "validity": 1.0, "performance": 0.85}
''')
        self.candidate = os.path.join(self.tmpdir, "candidate.py")
        with open(self.candidate, "w") as f:
            f.write("x = 1")

    def test_cascade_passes_all_stages(self):
        config = EvaluatorConfig(
            mode="script",
            cascade_evaluation=True,
            cascade_thresholds=[0.5, 0.7],
        )
        evaluator = Evaluator(config, self.eval_script)
        metrics = asyncio.run(evaluator.evaluate(self.candidate))
        self.assertGreater(metrics["combined_score"], 0.0)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir)


if __name__ == "__main__":
    unittest.main()
```

**Step 2: Run tests to verify failure**

Run: `cd claude_evolve && python -m pytest tests/test_evaluator.py -v`
Expected: FAIL

**Step 3: Implement evaluator.py**

Extract from `/home/bud/Desktop/claudeEvolve/openevolve/openevolve/evaluator.py`. Key changes:
- Keep subprocess-based evaluation with timeout
- Keep cascade evaluation (`evaluate_stage1`, `evaluate_stage2`, `evaluate`)
- Add `evaluate_with_metrics()` passthrough for critic mode
- Remove `_llm_evaluate` (LLM feedback not needed - Claude IS the LLM)
- Keep artifact collection from evaluation
- Make evaluation async-compatible but also support sync calling

**Step 4: Run tests**

Run: `cd claude_evolve && python -m pytest tests/test_evaluator.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add claude_evolve/claude_evolve/core/evaluator.py claude_evolve/tests/test_evaluator.py
git commit -m "feat: add Evaluator with script, critic, and cascade modes"
```

---

### Task 6: Template Manager + Prompt Context Builder

**Files:**
- Create: `claude_evolve/claude_evolve/prompt/__init__.py`
- Create: `claude_evolve/claude_evolve/prompt/templates.py`
- Create: `claude_evolve/claude_evolve/prompt/context_builder.py`
- Create: `claude_evolve/claude_evolve/prompt/default_templates/` (directory with .txt and fragments.json)
- Test: `claude_evolve/tests/test_context_builder.py`

**Step 1: Write the failing test**

```python
# tests/test_context_builder.py
import unittest
from claude_evolve.core.artifact import Artifact
from claude_evolve.prompt.context_builder import ContextBuilder
from claude_evolve.config import PromptConfig


class TestContextBuilder(unittest.TestCase):
    def setUp(self):
        self.builder = ContextBuilder(PromptConfig())
        self.parent = Artifact(
            id="parent-1",
            content="def solve():\n    return 42",
            artifact_type="python",
            metrics={"combined_score": 0.6, "accuracy": 0.7},
            generation=2,
        )

    def test_build_context_returns_dict(self):
        ctx = self.builder.build_context(
            parent=self.parent,
            iteration=5,
            best_score=0.6,
            top_programs=[self.parent.to_dict()],
            inspirations=[],
            previous_programs=[],
        )
        self.assertIn("prompt", ctx)
        self.assertIn("system_message", ctx)
        self.assertIn("parent_id", ctx)
        self.assertIsInstance(ctx["prompt"], str)

    def test_context_includes_parent_code(self):
        ctx = self.builder.build_context(
            parent=self.parent,
            iteration=1,
            best_score=0.6,
            top_programs=[],
            inspirations=[],
            previous_programs=[],
        )
        self.assertIn("def solve():", ctx["prompt"])

    def test_context_includes_metrics(self):
        ctx = self.builder.build_context(
            parent=self.parent,
            iteration=1,
            best_score=0.6,
            top_programs=[],
            inspirations=[],
            previous_programs=[],
        )
        self.assertIn("combined_score", ctx["prompt"])

    def test_context_includes_evolution_history(self):
        prev = Artifact(
            id="prev-1",
            content="old code",
            metrics={"combined_score": 0.4},
            metadata={"changes": "initial attempt"},
        )
        ctx = self.builder.build_context(
            parent=self.parent,
            iteration=3,
            best_score=0.6,
            top_programs=[self.parent.to_dict()],
            inspirations=[],
            previous_programs=[prev.to_dict()],
        )
        self.assertIn("Previous Attempts", ctx["prompt"])

    def test_context_includes_improvement_areas(self):
        ctx = self.builder.build_context(
            parent=self.parent,
            iteration=1,
            best_score=0.6,
            top_programs=[],
            inspirations=[],
            previous_programs=[],
        )
        self.assertIn("improvement", ctx["prompt"].lower())

    def test_context_includes_artifacts_when_present(self):
        ctx = self.builder.build_context(
            parent=self.parent,
            iteration=1,
            best_score=0.6,
            top_programs=[],
            inspirations=[],
            previous_programs=[],
            parent_artifacts={"stderr": "TypeError: bad arg"},
        )
        self.assertIn("TypeError", ctx["prompt"])

    def test_context_diff_vs_rewrite_mode(self):
        ctx_diff = self.builder.build_context(
            parent=self.parent,
            iteration=1,
            best_score=0.6,
            top_programs=[],
            inspirations=[],
            previous_programs=[],
            diff_based=True,
        )
        self.assertIn("SEARCH", ctx_diff["prompt"])

        ctx_rewrite = self.builder.build_context(
            parent=self.parent,
            iteration=1,
            best_score=0.6,
            top_programs=[],
            inspirations=[],
            previous_programs=[],
            diff_based=False,
        )
        self.assertIn("Rewrite", ctx_rewrite["prompt"])

    def test_render_iteration_context_markdown(self):
        """The iteration context written to .claude/evolve-state/iteration_context.md"""
        ctx = self.builder.build_context(
            parent=self.parent,
            iteration=5,
            best_score=0.6,
            top_programs=[],
            inspirations=[],
            previous_programs=[],
        )
        md = self.builder.render_iteration_context(ctx, iteration=5, max_iterations=30)
        self.assertIn("Iteration 5", md)
        self.assertIn("Best Score", md)


if __name__ == "__main__":
    unittest.main()
```

**Step 2: Run tests to verify failure**

Run: `cd claude_evolve && python -m pytest tests/test_context_builder.py -v`
Expected: FAIL

**Step 3: Implement templates.py**

Extract `TemplateManager` class and all template constants from `/home/bud/Desktop/claudeEvolve/openevolve/openevolve/prompt/templates.py`. Copy default templates from `/home/bud/Desktop/claudeEvolve/openevolve/openevolve/prompts/defaults/` into `claude_evolve/claude_evolve/prompt/default_templates/`.

**Step 4: Implement context_builder.py**

Rewrite from `PromptSampler` in `/home/bud/Desktop/claudeEvolve/openevolve/openevolve/prompt/sampler.py`. Key differences from OpenEvolve:
- `build_context()` returns a dict with `prompt`, `system_message`, `parent_id`, `metadata`
- `render_iteration_context()` produces markdown for `.claude/evolve-state/iteration_context.md`
- Context is designed for a tool-using agent (Claude), not a raw LLM API call
- Include iteration metadata (iteration number, best score, target score)
- Keep all the OpenEvolve prompt construction logic: metrics formatting, improvement areas, evolution history, top programs, inspirations, artifacts

**Step 5: Run tests**

Run: `cd claude_evolve && python -m pytest tests/test_context_builder.py -v`
Expected: All PASS

**Step 6: Commit**

```bash
git add claude_evolve/claude_evolve/prompt/ claude_evolve/tests/test_context_builder.py
git commit -m "feat: add TemplateManager and ContextBuilder for evolution prompts"
```

---

### Task 7: State Manager + Loop State

**Files:**
- Create: `claude_evolve/claude_evolve/state/__init__.py`
- Create: `claude_evolve/claude_evolve/state/manager.py`
- Create: `claude_evolve/claude_evolve/state/loop_state.py`
- Create: `claude_evolve/claude_evolve/state/checkpoint.py`
- Test: `claude_evolve/tests/test_state.py`

**Step 1: Write the failing test**

```python
# tests/test_state.py
import json
import os
import tempfile
import unittest
import yaml
from claude_evolve.state.manager import StateManager
from claude_evolve.state.loop_state import LoopState
from claude_evolve.state.checkpoint import CheckpointManager
from claude_evolve.config import Config


class TestLoopState(unittest.TestCase):
    """Tests for .claude/evolve.local.md state file management"""

    def test_create_state_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = os.path.join(tmpdir, "evolve.local.md")
            ls = LoopState.create(
                path=state_path,
                session_id="test-session-123",
                max_iterations=30,
                target_score=0.95,
                completion_promise="EVOLUTION_TARGET_REACHED",
                prompt="Evolve circle packing algorithm",
            )
            self.assertTrue(os.path.exists(state_path))
            self.assertEqual(ls.iteration, 1)
            self.assertEqual(ls.max_iterations, 30)

    def test_read_state_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = os.path.join(tmpdir, "evolve.local.md")
            LoopState.create(
                path=state_path,
                session_id="s1",
                max_iterations=10,
                target_score=0.9,
                completion_promise="DONE",
                prompt="test prompt",
            )
            ls = LoopState.read(state_path)
            self.assertEqual(ls.iteration, 1)
            self.assertEqual(ls.max_iterations, 10)
            self.assertAlmostEqual(ls.target_score, 0.9)
            self.assertEqual(ls.prompt, "test prompt")
            self.assertEqual(ls.session_id, "s1")

    def test_increment_iteration(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = os.path.join(tmpdir, "evolve.local.md")
            LoopState.create(
                path=state_path,
                session_id="s1",
                max_iterations=10,
                prompt="test",
            )
            ls = LoopState.read(state_path)
            ls.increment_iteration()
            ls.write(state_path)
            ls2 = LoopState.read(state_path)
            self.assertEqual(ls2.iteration, 2)

    def test_update_prompt(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = os.path.join(tmpdir, "evolve.local.md")
            LoopState.create(
                path=state_path,
                session_id="s1",
                max_iterations=10,
                prompt="original prompt",
            )
            ls = LoopState.read(state_path)
            ls.prompt = "new iteration prompt with context"
            ls.write(state_path)
            ls2 = LoopState.read(state_path)
            self.assertEqual(ls2.prompt, "new iteration prompt with context")

    def test_is_complete_max_iterations(self):
        ls = LoopState(iteration=10, max_iterations=10, prompt="x")
        self.assertTrue(ls.is_max_iterations_reached())

    def test_is_complete_target_score(self):
        ls = LoopState(iteration=3, max_iterations=50, prompt="x", target_score=0.9)
        self.assertFalse(ls.is_target_reached(0.5))
        self.assertTrue(ls.is_target_reached(0.95))


class TestStateManager(unittest.TestCase):
    """Tests for evolution state directory management"""

    def test_initialize_state(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_dir = os.path.join(tmpdir, "evolve-state")
            sm = StateManager(state_dir)
            sm.initialize(
                config=Config(),
                initial_content="print('hello')",
                artifact_type="python",
            )
            self.assertTrue(os.path.exists(os.path.join(state_dir, "database.json")))
            self.assertTrue(os.path.exists(os.path.join(state_dir, "config.json")))

    def test_load_state(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_dir = os.path.join(tmpdir, "evolve-state")
            sm = StateManager(state_dir)
            sm.initialize(
                config=Config(),
                initial_content="print('hello')",
                artifact_type="python",
            )
            sm2 = StateManager(state_dir)
            sm2.load()
            self.assertIsNotNone(sm2.database)
            self.assertEqual(sm2.database.size(), 1)


class TestCheckpointManager(unittest.TestCase):
    def test_save_and_restore_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_dir = os.path.join(tmpdir, "evolve-state")
            sm = StateManager(state_dir)
            sm.initialize(
                config=Config(checkpoint_interval=5),
                initial_content="x = 1",
                artifact_type="python",
            )
            # Save checkpoint
            cp = CheckpointManager(state_dir)
            cp.save(sm.database, iteration=5)
            # Verify checkpoint exists
            checkpoints = cp.list_checkpoints()
            self.assertEqual(len(checkpoints), 1)
            self.assertEqual(checkpoints[0]["iteration"], 5)


if __name__ == "__main__":
    unittest.main()
```

**Step 2: Run tests to verify failure**

Run: `cd claude_evolve && python -m pytest tests/test_state.py -v`
Expected: FAIL

**Step 3: Implement state modules**

- `loop_state.py`: Manages `.claude/evolve.local.md` (markdown with YAML frontmatter, Ralph-style). Methods: `create()`, `read()`, `write()`, `increment_iteration()`, `is_max_iterations_reached()`, `is_target_reached()`.
- `manager.py`: Manages `.claude/evolve-state/` directory. Holds `ArtifactDatabase` + `Config`. Methods: `initialize()`, `load()`, `save()`, `get_database()`, `get_config()`.
- `checkpoint.py`: Manages checkpoints within state directory. Methods: `save()`, `restore()`, `list_checkpoints()`.

**Step 4: Run tests**

Run: `cd claude_evolve && python -m pytest tests/test_state.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add claude_evolve/claude_evolve/state/ claude_evolve/tests/test_state.py
git commit -m "feat: add state management with loop state, state manager, and checkpoints"
```

---

### Task 8: CLI Entry Points

**Files:**
- Create: `claude_evolve/claude_evolve/cli.py`
- Test: `claude_evolve/tests/test_cli.py`

**Step 1: Write failing tests**

```python
# tests/test_cli.py
import json
import os
import tempfile
import unittest
from click.testing import CliRunner
from claude_evolve.cli import main


class TestCliInit(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()
        self.tmpdir = tempfile.mkdtemp()
        # Create artifact
        self.artifact = os.path.join(self.tmpdir, "program.py")
        with open(self.artifact, "w") as f:
            f.write("def solve():\n    return 42\n")
        # Create evaluator
        self.evaluator = os.path.join(self.tmpdir, "evaluator.py")
        with open(self.evaluator, "w") as f:
            f.write('''
def evaluate(path):
    return {"combined_score": 0.5}
''')

    def test_init_creates_state(self):
        state_dir = os.path.join(self.tmpdir, ".claude", "evolve-state")
        result = self.runner.invoke(main, [
            "init",
            "--artifact", self.artifact,
            "--evaluator", self.evaluator,
            "--state-dir", state_dir,
            "--max-iterations", "10",
        ])
        self.assertEqual(result.exit_code, 0, msg=result.output)
        self.assertTrue(os.path.exists(os.path.join(state_dir, "database.json")))

    def test_init_missing_artifact_fails(self):
        result = self.runner.invoke(main, [
            "init",
            "--artifact", "/nonexistent/file.py",
            "--evaluator", self.evaluator,
        ])
        self.assertNotEqual(result.exit_code, 0)


class TestCliNext(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()
        self.tmpdir = tempfile.mkdtemp()
        self.artifact = os.path.join(self.tmpdir, "program.py")
        with open(self.artifact, "w") as f:
            f.write("x = 1\n")
        self.evaluator = os.path.join(self.tmpdir, "evaluator.py")
        with open(self.evaluator, "w") as f:
            f.write('def evaluate(p): return {"combined_score": 0.3}')
        self.state_dir = os.path.join(self.tmpdir, "state")
        # Initialize first
        self.runner.invoke(main, [
            "init",
            "--artifact", self.artifact,
            "--evaluator", self.evaluator,
            "--state-dir", self.state_dir,
            "--max-iterations", "10",
        ])

    def test_next_produces_context(self):
        result = self.runner.invoke(main, [
            "next",
            "--state-dir", self.state_dir,
        ])
        self.assertEqual(result.exit_code, 0, msg=result.output)
        # Should produce iteration context
        context_file = os.path.join(self.state_dir, "iteration_context.md")
        self.assertTrue(os.path.exists(context_file))


class TestCliSubmit(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()
        self.tmpdir = tempfile.mkdtemp()
        self.artifact = os.path.join(self.tmpdir, "program.py")
        with open(self.artifact, "w") as f:
            f.write("x = 1\n")
        self.evaluator = os.path.join(self.tmpdir, "evaluator.py")
        with open(self.evaluator, "w") as f:
            f.write('def evaluate(p): return {"combined_score": 0.5}')
        self.state_dir = os.path.join(self.tmpdir, "state")
        self.runner.invoke(main, [
            "init",
            "--artifact", self.artifact,
            "--evaluator", self.evaluator,
            "--state-dir", self.state_dir,
        ])
        # Create a candidate
        self.candidate = os.path.join(self.tmpdir, "candidate.py")
        with open(self.candidate, "w") as f:
            f.write("x = optimized_value\n")

    def test_submit_evaluates_and_stores(self):
        result = self.runner.invoke(main, [
            "submit",
            "--candidate", self.candidate,
            "--state-dir", self.state_dir,
        ])
        self.assertEqual(result.exit_code, 0, msg=result.output)
        output = json.loads(result.output)
        self.assertIn("combined_score", output)

    def test_submit_with_metrics_passthrough(self):
        result = self.runner.invoke(main, [
            "submit",
            "--candidate", self.candidate,
            "--state-dir", self.state_dir,
            "--metrics", '{"combined_score": 0.85, "clarity": 0.9}',
        ])
        self.assertEqual(result.exit_code, 0, msg=result.output)
        output = json.loads(result.output)
        self.assertAlmostEqual(output["combined_score"], 0.85)


class TestCliStatus(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()
        self.tmpdir = tempfile.mkdtemp()
        self.artifact = os.path.join(self.tmpdir, "program.py")
        with open(self.artifact, "w") as f:
            f.write("x = 1\n")
        self.evaluator = os.path.join(self.tmpdir, "evaluator.py")
        with open(self.evaluator, "w") as f:
            f.write('def evaluate(p): return {"combined_score": 0.5}')
        self.state_dir = os.path.join(self.tmpdir, "state")
        self.runner.invoke(main, [
            "init",
            "--artifact", self.artifact,
            "--evaluator", self.evaluator,
            "--state-dir", self.state_dir,
        ])

    def test_status_shows_info(self):
        result = self.runner.invoke(main, [
            "status",
            "--state-dir", self.state_dir,
        ])
        self.assertEqual(result.exit_code, 0, msg=result.output)
        output = json.loads(result.output)
        self.assertIn("iteration", output)
        self.assertIn("best_score", output)
        self.assertIn("population_size", output)


class TestCliExport(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()
        self.tmpdir = tempfile.mkdtemp()
        self.artifact = os.path.join(self.tmpdir, "program.py")
        with open(self.artifact, "w") as f:
            f.write("x = 1\n")
        self.evaluator = os.path.join(self.tmpdir, "evaluator.py")
        with open(self.evaluator, "w") as f:
            f.write('def evaluate(p): return {"combined_score": 0.5}')
        self.state_dir = os.path.join(self.tmpdir, "state")
        self.runner.invoke(main, [
            "init",
            "--artifact", self.artifact,
            "--evaluator", self.evaluator,
            "--state-dir", self.state_dir,
        ])

    def test_export_best(self):
        output_file = os.path.join(self.tmpdir, "best.py")
        result = self.runner.invoke(main, [
            "export",
            "--state-dir", self.state_dir,
            "--output", output_file,
        ])
        self.assertEqual(result.exit_code, 0, msg=result.output)
        self.assertTrue(os.path.exists(output_file))
        with open(output_file) as f:
            content = f.read()
        self.assertIn("x = 1", content)


if __name__ == "__main__":
    unittest.main()
```

**Step 2: Run tests to verify failure**

Run: `cd claude_evolve && python -m pytest tests/test_cli.py -v`
Expected: FAIL

**Step 3: Implement cli.py**

Use Click framework. Commands:
- `init`: Create state directory, initialize database with initial artifact, run baseline evaluation, create `.claude/evolve.local.md`
- `next`: Load state, call selector.sample(), call context_builder.build_context(), write iteration_context.md, update loop state with new prompt, print prompt to stdout
- `submit`: Load state, evaluate candidate (script or metrics passthrough), add to database, update best, handle migration, save checkpoint if interval reached, print results as JSON
- `status`: Load state, print JSON with iteration, best_score, population_size, island_stats
- `export`: Load state, write best artifact to output path, optionally export top N

**Step 4: Run tests**

Run: `cd claude_evolve && python -m pytest tests/test_cli.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add claude_evolve/claude_evolve/cli.py claude_evolve/tests/test_cli.py
git commit -m "feat: add CLI with init, next, submit, status, export commands"
```

---

### Task 9: Integration Test - Full Evolution Cycle

**Files:**
- Test: `claude_evolve/tests/test_integration.py`

**Step 1: Write integration test**

```python
# tests/test_integration.py
"""End-to-end test: init → next → submit → next → submit → export"""
import json
import os
import tempfile
import unittest
from click.testing import CliRunner
from claude_evolve.cli import main


class TestFullEvolutionCycle(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()
        self.tmpdir = tempfile.mkdtemp()

        # Initial program
        self.artifact = os.path.join(self.tmpdir, "program.py")
        with open(self.artifact, "w") as f:
            f.write("""
def pack_circles(n=10):
    import random
    circles = []
    for i in range(n):
        x = random.random()
        y = random.random()
        r = 0.01
        circles.append((x, y, r))
    return circles
""")

        # Evaluator
        self.evaluator = os.path.join(self.tmpdir, "evaluator.py")
        with open(self.evaluator, "w") as f:
            f.write('''
def evaluate(artifact_path):
    with open(artifact_path) as f:
        content = f.read()
    has_optimization = "optimize" in content.lower() or "numpy" in content.lower()
    base_score = 0.3
    if has_optimization:
        base_score = 0.7
    if "grid" in content.lower():
        base_score = 0.9
    return {
        "combined_score": base_score,
        "code_quality": min(len(content) / 500.0, 1.0),
    }
''')

        self.state_dir = os.path.join(self.tmpdir, "state")

    def test_full_cycle(self):
        # 1. Init
        result = self.runner.invoke(main, [
            "init",
            "--artifact", self.artifact,
            "--evaluator", self.evaluator,
            "--state-dir", self.state_dir,
            "--max-iterations", "5",
        ])
        self.assertEqual(result.exit_code, 0, msg=f"Init failed: {result.output}")

        # 2. First next (prepare iteration context)
        result = self.runner.invoke(main, ["next", "--state-dir", self.state_dir])
        self.assertEqual(result.exit_code, 0, msg=f"Next failed: {result.output}")
        context_file = os.path.join(self.state_dir, "iteration_context.md")
        self.assertTrue(os.path.exists(context_file))

        # 3. Simulate Claude producing a better candidate
        candidate1 = os.path.join(self.tmpdir, "candidate1.py")
        with open(candidate1, "w") as f:
            f.write("""
import numpy as np
def pack_circles(n=10):
    # Optimized version using grid placement
    circles = []
    grid_size = int(np.ceil(np.sqrt(n)))
    r = 0.4 / grid_size
    for i in range(n):
        x = (i % grid_size + 0.5) / grid_size
        y = (i // grid_size + 0.5) / grid_size
        circles.append((x, y, r))
    return circles
""")

        # 4. Submit candidate 1
        result = self.runner.invoke(main, [
            "submit", "--candidate", candidate1, "--state-dir", self.state_dir
        ])
        self.assertEqual(result.exit_code, 0, msg=f"Submit failed: {result.output}")
        metrics1 = json.loads(result.output)
        self.assertGreater(metrics1["combined_score"], 0.3)

        # 5. Second next
        result = self.runner.invoke(main, ["next", "--state-dir", self.state_dir])
        self.assertEqual(result.exit_code, 0)

        # 6. Submit candidate 2 with critic metrics
        candidate2 = os.path.join(self.tmpdir, "candidate2.py")
        with open(candidate2, "w") as f:
            f.write("# grid-based optimal solution\ndef pack_circles(): pass")
        result = self.runner.invoke(main, [
            "submit", "--candidate", candidate2,
            "--state-dir", self.state_dir,
            "--metrics", '{"combined_score": 0.95, "elegance": 0.8}',
        ])
        self.assertEqual(result.exit_code, 0)
        metrics2 = json.loads(result.output)
        self.assertAlmostEqual(metrics2["combined_score"], 0.95)

        # 7. Status check
        result = self.runner.invoke(main, ["status", "--state-dir", self.state_dir])
        self.assertEqual(result.exit_code, 0)
        status = json.loads(result.output)
        self.assertGreaterEqual(status["best_score"], 0.9)
        self.assertGreaterEqual(status["population_size"], 3)

        # 8. Export best
        best_output = os.path.join(self.tmpdir, "best.py")
        result = self.runner.invoke(main, [
            "export", "--state-dir", self.state_dir, "--output", best_output
        ])
        self.assertEqual(result.exit_code, 0)
        with open(best_output) as f:
            best_code = f.read()
        self.assertTrue(len(best_code) > 0)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
```

**Step 2: Run integration test**

Run: `cd claude_evolve && python -m pytest tests/test_integration.py -v`
Expected: All PASS (requires Tasks 1-8 complete)

**Step 3: Commit**

```bash
git add claude_evolve/tests/test_integration.py
git commit -m "test: add full evolution cycle integration test"
```

---

## Phase 2: Claude Code Plugin

### Task 10: Plugin Scaffold + Manifest

**Files:**
- Create: `plugin/.claude-plugin/plugin.json`
- Create: `plugin/hooks/hooks.json`
- Create: `plugin/README.md`

**Step 1: Create plugin manifest**

```json
// plugin/.claude-plugin/plugin.json
{
  "name": "claude-evolve",
  "description": "Evolutionary code and artifact optimization powered by OpenEvolve. Evolve programs, prompts, algorithms, and any text artifact using MAP-Elites quality-diversity search with Claude as the intelligent mutation engine.",
  "version": "0.1.0",
  "author": {
    "name": "Claude Evolve",
    "email": "noreply@example.com"
  }
}
```

**Step 2: Create hooks.json**

```json
// plugin/hooks/hooks.json
{
  "description": "Claude Evolve stop hook for evolution loop management",
  "hooks": {
    "Stop": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "${CLAUDE_PLUGIN_ROOT}/hooks/stop-hook.sh"
          }
        ]
      }
    ]
  }
}
```

**Step 3: Verify directory structure**

Run: `find plugin/ -type f | sort`
Expected: Plugin structure matches design

**Step 4: Commit**

```bash
git add plugin/
git commit -m "feat: scaffold claude-evolve plugin with manifest and hooks config"
```

---

### Task 11: Setup Script

**Files:**
- Create: `plugin/scripts/setup-evolve.sh`

**Step 1: Implement setup-evolve.sh**

Pattern: Follows Ralph Loop's `setup-ralph-loop.sh` exactly for argument parsing, then calls `claude-evolve init` for state initialization. Arguments:
- Positional: `<artifact> <evaluator>`
- `--mode script|critic|hybrid` (default: auto-detect from evaluator extension)
- `--max-iterations N` (default: 50)
- `--target-score F` (default: none)
- `--config path` (default: none)
- `--completion-promise TEXT` (default: "EVOLUTION_TARGET_REACHED")

The script:
1. Parses arguments (same robustness as Ralph)
2. Validates artifact and evaluator files exist
3. Calls `claude-evolve init --artifact <path> --evaluator <path> --state-dir .claude/evolve-state --max-iterations N`
4. Creates `.claude/evolve.local.md` with evolve-specific frontmatter
5. Outputs setup confirmation

**Step 2: Make executable and test manually**

Run: `chmod +x plugin/scripts/setup-evolve.sh && bash plugin/scripts/setup-evolve.sh --help`
Expected: Help text displayed

**Step 3: Commit**

```bash
git add plugin/scripts/setup-evolve.sh
git commit -m "feat: add setup-evolve.sh for evolution loop initialization"
```

---

### Task 12: Stop Hook (Evolution Loop)

**Files:**
- Create: `plugin/hooks/stop-hook.sh`

**Step 1: Implement stop-hook.sh**

Pattern: Based on Ralph Loop's stop hook, but with a critical difference — instead of re-feeding the same prompt, it calls `claude-evolve next` to generate a **dynamic per-iteration prompt**.

Flow:
1. Read hook input (JSON with session_id, transcript_path)
2. Check `.claude/evolve.local.md` exists
3. Session isolation check (same as Ralph)
4. Parse frontmatter: iteration, max_iterations, target_score, state_dir
5. Check max iterations
6. Check completion promise in transcript (same Perl regex as Ralph)
7. Check target score from latest status: `claude-evolve status --state-dir .claude/evolve-state | jq '.best_score'`
8. If target reached, allow exit
9. Otherwise: run `claude-evolve next --state-dir .claude/evolve-state` to prepare next iteration
10. Read updated prompt from `.claude/evolve-state/iteration_context.md`
11. Increment iteration in `.claude/evolve.local.md`
12. Output JSON: `{"decision": "block", "reason": <dynamic_prompt>, "systemMessage": <status>}`

**Step 2: Test manually**

Create a mock state file and test the hook with piped JSON input.

**Step 3: Commit**

```bash
git add plugin/hooks/stop-hook.sh
git commit -m "feat: add evolution stop hook with dynamic prompt generation"
```

---

### Task 13: Commands (/evolve, /evolve-status, /cancel-evolve)

**Files:**
- Create: `plugin/commands/evolve.md`
- Create: `plugin/commands/evolve-status.md`
- Create: `plugin/commands/cancel-evolve.md`

**Step 1: Create /evolve command**

```markdown
---
description: "Start evolutionary optimization loop"
argument-hint: "<artifact> <evaluator> [--mode script|critic|hybrid] [--max-iterations N] [--target-score F] [--config path]"
allowed-tools: ["Bash(${CLAUDE_PLUGIN_ROOT}/scripts/setup-evolve.sh:*)"]
hide-from-slash-command-tool: "true"
---

Execute the setup script to initialize the evolution loop:

```!
"${CLAUDE_PLUGIN_ROOT}/scripts/setup-evolve.sh" $ARGUMENTS
```
```

**Step 2: Create /evolve-status command**

```markdown
---
description: "Check evolution progress"
allowed-tools: ["Bash(claude-evolve status:*)", "Read(.claude/evolve.local.md)", "Read(.claude/evolve-state/*)"]
---

Check the current evolution status by running:

```!
claude-evolve status --state-dir .claude/evolve-state
```

Also read the loop state:

```!
head -15 .claude/evolve.local.md
```
```

**Step 3: Create /cancel-evolve command**

```markdown
---
description: "Cancel active evolution loop"
allowed-tools: ["Bash(test -f .claude/evolve.local.md:*)", "Bash(rm .claude/evolve.local.md)", "Read(.claude/evolve.local.md)", "Bash(claude-evolve status:*)", "Bash(claude-evolve export:*)"]
hide-from-slash-command-tool: "true"
---

Cancel the active evolution loop:

1. Check if evolution is active: `test -f .claude/evolve.local.md`
2. If not found: Report "No active evolution loop"
3. If found:
   - Read state to get iteration count
   - Export best artifact so far
   - Delete state file: `rm .claude/evolve.local.md`
   - Report: "Cancelled evolution at iteration N. Best artifact exported."
```

**Step 4: Commit**

```bash
git add plugin/commands/
git commit -m "feat: add /evolve, /evolve-status, /cancel-evolve commands"
```

---

### Task 14: Evolution Skill (SKILL.md)

**Files:**
- Create: `plugin/skills/evolve/SKILL.md`

**Step 1: Write the core evolution skill**

This is the most important file in the plugin. It teaches Claude HOW to behave during each evolution iteration. It must cover:

1. **Reading Context**: How to read `.claude/evolve-state/iteration_context.md`
2. **Research Phase**: WebSearch for literature, read evaluator code, understand fitness landscape
3. **Exploration Phase**: Use subagents for parallel approach exploration, brainstorm mutations
4. **Generation Phase**: Create candidate artifact (diff or full rewrite)
5. **Informal Testing**: Quick sanity checks before formal submission
6. **Submission**: Call `claude-evolve submit` with candidate
7. **Critic Mode**: When to spawn critic agent, how to collect scores
8. **Iteration Protocol**: What to do at the end of each iteration

```markdown
---
name: evolve
description: Core methodology for evolutionary artifact optimization. Use when working within a /evolve loop to generate, evaluate, and submit improved candidates using research, subagents, and autonomous exploration.
---

# Evolutionary Artifact Optimization

You are operating within an evolution loop. Each iteration, you receive context about the current best artifact, its performance metrics, evolution history, and inspiration programs. Your job is to produce the best possible improved candidate.

## Iteration Protocol

### Phase 1: Understand (10-20% of iteration time)

1. **Read the iteration context**: `Read .claude/evolve-state/iteration_context.md`
2. **Read the evaluator**: Understand what the fitness function rewards
3. **Read the parent artifact**: Understand what's working and what isn't
4. **Check artifacts**: Review stderr, warnings, profiling from previous evaluation

### Phase 2: Research (20-40% of iteration time)

Use ALL available tools aggressively:

- **WebSearch**: Find state-of-the-art approaches to the problem
  - Search for academic papers, blog posts, Stack Overflow solutions
  - Search for the specific algorithm/technique being optimized
  - Find competing implementations or theoretical bounds

- **Subagents**: Spawn parallel research agents
  - One agent researches algorithmic improvements
  - One agent analyzes why previous attempts failed
  - One agent explores entirely different approaches

- **Literature Review**: For each promising approach found:
  - Understand the core insight
  - Assess feasibility for this specific problem
  - Note key implementation details

- **Code Analysis**:
  - Grep for patterns in the codebase
  - Read related utility functions
  - Understand the evaluation pipeline

### Phase 3: Generate (30-40% of iteration time)

Based on research, generate the improved candidate:

- **For code artifacts**: Write production-quality code, not stubs
- **For prompt artifacts**: Apply prompt engineering best practices
- **For any artifact**: Focus on what the evaluator rewards

Strategies (choose based on context):
1. **Targeted diff**: Fix specific weakness identified in metrics
2. **Algorithmic redesign**: Replace core algorithm with better approach from research
3. **Hybrid**: Combine best elements from inspiration programs
4. **Creative leap**: Try fundamentally different approach

Write candidate to: `.claude/evolve-workspace/candidate.<ext>`

### Phase 4: Validate (10-20% of iteration time)

Before formal submission:
- Run quick sanity checks (syntax, imports, basic execution)
- If code: run it once to check for crashes
- If prompt: quick mental review for coherence
- Fix any obvious issues

### Phase 5: Submit

For **script** mode:
```bash
claude-evolve submit \
  --candidate .claude/evolve-workspace/candidate.<ext> \
  --state-dir .claude/evolve-state
```

For **critic** mode:
1. Spawn a critic subagent with the eval prompt
2. Collect structured metrics from critic
3. Submit with metrics:
```bash
claude-evolve submit \
  --candidate .claude/evolve-workspace/candidate.<ext> \
  --state-dir .claude/evolve-state \
  --metrics '{"combined_score": 0.85, "metric2": 0.9}'
```

### Phase 6: Report

After submission, briefly report:
- What approach you tried
- The evaluation score
- Key insight for next iteration

Then let the session end naturally. The stop hook will prepare the next iteration.

## When Target Score Is Reached

Output: `<promise>EVOLUTION_TARGET_REACHED</promise>`

ONLY when the evaluation score meets or exceeds the target. Never lie to exit.

## Advanced Techniques

### Using Ralph Loop Within Evolution
For deep refinement of a specific approach, you can start a Ralph Loop:
```
/ralph-loop "Refine this specific algorithm aspect..." --max-iterations 5 --completion-promise "REFINED"
```

### Parallel Exploration with Subagents
Spawn multiple agents to explore different mutation strategies simultaneously:
- Agent 1: Conservative improvement (small targeted changes)
- Agent 2: Aggressive redesign (new algorithm)
- Agent 3: Hybrid approach (combine best of top programs)
Pick the best candidate from the results.

### TDD Within Evolution
For code artifacts, use test-driven development:
1. Write a test that captures the improvement you want
2. Modify the artifact to pass the test
3. Verify it still passes the evaluator

## What NOT To Do

- Do NOT modify the evaluator
- Do NOT modify the evolution state files directly
- Do NOT fake metrics or lie about scores
- Do NOT skip the research phase on early iterations
- Do NOT submit without at least basic validation
```

**Step 2: Commit**

```bash
git add plugin/skills/
git commit -m "feat: add core evolution skill with full methodology"
```

---

### Task 15: Critic Agent Definition

**Files:**
- Create: `plugin/agents/critic.md`

**Step 1: Write critic agent definition**

```markdown
---
name: critic
description: |
  Adversarial evaluation agent for critic-mode evolution. Spawned to evaluate artifacts
  when the evaluator is a prompt (not a script). Tries to find flaws, rate quality,
  and return structured metrics.

  <example>
  Context: Evolution loop in critic mode needs to evaluate a prompt artifact
  assistant: "Let me spawn the critic agent to adversarially evaluate this candidate"
  </example>
model: inherit
---

You are a HARSH, ADVERSARIAL critic. Your job is to find every flaw in the artifact presented to you.

## Your Process

1. Read the evaluation prompt carefully - it defines what to evaluate
2. Read the candidate artifact thoroughly
3. Try to BREAK it:
   - Find logical flaws
   - Find edge cases that fail
   - Find inconsistencies
   - Find missing requirements
   - Find areas that could be better
4. Rate each dimension defined in the evaluation prompt on 0.0-1.0 scale
5. Return your evaluation as a JSON code block:

```json
{
  "combined_score": <float 0-1>,
  "<metric1>": <float 0-1>,
  "<metric2>": <float 0-1>,
  "flaws_found": <int>,
  "reasoning": "<brief explanation>"
}
```

## Rules

- Be HARSH. A score of 0.9+ should mean truly excellent with no significant flaws.
- Be SPECIFIC in your reasoning.
- Always try to disprove/break the artifact before scoring.
- The combined_score should reflect overall quality.
- Return ONLY the JSON block at the end.
```

**Step 2: Commit**

```bash
git add plugin/agents/
git commit -m "feat: add adversarial critic agent for prompt-based evaluation"
```

---

## Phase 3: Installation + End-to-End Testing

### Task 16: Installation Script + Integration

**Files:**
- Create: `plugin/scripts/install-deps.sh`
- Create: `install.sh` (root-level installer)

**Step 1: Create install-deps.sh**

```bash
#!/bin/bash
# Install claude_evolve Python package
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PLUGIN_ROOT="$(dirname "$SCRIPT_DIR")"
PACKAGE_DIR="$(dirname "$PLUGIN_ROOT")/claude_evolve"

if [[ -d "$PACKAGE_DIR" ]]; then
    pip install -e "$PACKAGE_DIR" --quiet
    echo "claude-evolve installed successfully"
else
    echo "Error: claude_evolve package not found at $PACKAGE_DIR" >&2
    exit 1
fi
```

**Step 2: Create root installer**

```bash
#!/bin/bash
# Install Claude Evolve: Python package + Claude Code plugin
set -euo pipefail

echo "Installing Claude Evolve..."

# 1. Install Python package
cd claude_evolve && pip install -e ".[dev]" && cd ..

# 2. Register plugin
PLUGIN_DIR="$HOME/.claude/plugins/cache/claude-evolve-local/claude-evolve/0.1.0"
mkdir -p "$PLUGIN_DIR"
cp -r plugin/* "$PLUGIN_DIR/"

echo "Claude Evolve installed. Use /evolve to start."
```

**Step 3: Commit**

```bash
git add plugin/scripts/install-deps.sh install.sh
git commit -m "feat: add installation scripts for Python package and plugin"
```

---

### Task 17: End-to-End Manual Test

**No new files. Verification only.**

**Step 1: Install everything**

Run: `bash install.sh`

**Step 2: Create test problem**

Create a simple optimization problem:
- `test_problem/program.py`: Initial circle packing attempt
- `test_problem/evaluator.py`: Score based on sum of radii

**Step 3: Run evolution**

```bash
cd test_problem
/evolve program.py evaluator.py --max-iterations 5 --target-score 0.8
```

**Step 4: Verify**

- [ ] Setup message appears with iteration info
- [ ] Claude reads iteration context
- [ ] Claude performs research (web search, reads evaluator)
- [ ] Claude generates candidate
- [ ] Claude submits with `claude-evolve submit`
- [ ] Stop hook intercepts exit
- [ ] New iteration starts with different prompt
- [ ] After max iterations or target score, loop ends
- [ ] `/evolve-status` works
- [ ] `/cancel-evolve` works
- [ ] Best artifact is exported

**Step 5: Commit test problem**

```bash
git add test_problem/
git commit -m "test: add end-to-end test problem for manual verification"
```

---

## Summary

| Phase | Tasks | Description |
|-------|-------|-------------|
| **Phase 1** | Tasks 1-9 | Python package (`claude_evolve`) - all deterministic logic |
| **Phase 2** | Tasks 10-15 | Claude Code plugin - hooks, commands, skills, agents |
| **Phase 3** | Tasks 16-17 | Installation and end-to-end verification |

**Total estimated scope:** ~4000 lines Python + ~500 lines Bash + ~300 lines Markdown

**Key files by importance:**
1. `claude_evolve/core/database.py` - MAP-Elites (extracted from OpenEvolve)
2. `plugin/hooks/stop-hook.sh` - Loop mechanism (adapted from Ralph)
3. `plugin/skills/evolve/SKILL.md` - Claude's evolution methodology
4. `claude_evolve/cli.py` - CLI bridge between plugin and Python
5. `claude_evolve/prompt/context_builder.py` - Per-iteration context generation
