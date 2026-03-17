# Claude Evolve v2 Phase 1: Stagnation Engine + Enhanced Prompts + Cross-Run Memory

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add stagnation detection, failure-aware prompts, novelty checking, and cross-run memory to Claude Evolve, forming the foundation for all v2 capabilities.

**Architecture:** StagnationEngine analyzes score history from ArtifactDatabase and produces StagnationReports. These reports flow through a new `diagnose` CLI command into ContextBuilder, which selects templates and injects guidance based on stagnation level. CrossRunMemory persists key learnings between evolution runs. The existing `_is_novel()` no-op gets a real implementation using code similarity.

**Tech Stack:** Python 3.10+, Click (CLI), numpy, existing test framework (pytest)

**Success Criterion:** When running on R(5,5), the system detects stagnation after 3+ iterations at the same score, adapts exploration ratio, injects "paradigm shift" guidance into prompts, and persists learnings for subsequent runs.

---

## File Structure

### New Files
| File | Responsibility |
|------|---------------|
| `claude_evolve/claude_evolve/core/stagnation.py` | StagnationEngine, StagnationReport, StagnationLevel |
| `claude_evolve/claude_evolve/core/memory.py` | CrossRunMemory for persisting learnings across runs |
| `claude_evolve/tests/test_stagnation.py` | Tests for stagnation detection |
| `claude_evolve/tests/test_memory.py` | Tests for cross-run memory |
| `plugin/agents/researcher.md` | Research agent definition |
| `plugin/agents/diagnostician.md` | Diagnostician agent definition |

### Modified Files
| File | What Changes |
|------|-------------|
| `claude_evolve/claude_evolve/config.py` | Add StagnationConfig, ResearchConfig, CrossRunMemoryConfig dataclasses |
| `claude_evolve/claude_evolve/core/database.py` | Add `get_score_history()`, `detect_stagnation()`, implement `_is_novel()`, add stagnation-aware sampling |
| `claude_evolve/claude_evolve/prompt/context_builder.py` | Inject stagnation report, research findings, failure patterns into prompts |
| `claude_evolve/claude_evolve/prompt/templates.py` | Add research_user, paradigm_shift_user, stagnation fragments |
| `claude_evolve/claude_evolve/cli.py` | Add `diagnose` command, enhance `next` command to read diagnostic data |
| `claude_evolve/claude_evolve/state/manager.py` | Add methods for diagnostic reports and research logs |
| `plugin/hooks/stop-hook.sh` | Call `claude-evolve diagnose` before `claude-evolve next` |
| `plugin/skills/evolve/SKILL.md` | Add Phase 1.5 (Diagnose) instructions, research log persistence |
| `claude_evolve/tests/test_database.py` | Add tests for score history, stagnation detection, novelty checking |
| `claude_evolve/tests/test_context_builder.py` | Add tests for stagnation-aware prompt generation |
| `claude_evolve/tests/test_cli.py` | Add tests for diagnose command |

---

## Chunk 1: Stagnation Engine

### Task 1: StagnationLevel and StagnationReport dataclasses

**Files:**
- Create: `claude_evolve/claude_evolve/core/stagnation.py`
- Test: `claude_evolve/tests/test_stagnation.py`

- [ ] **Step 1: Write the test for StagnationLevel and StagnationReport**

```python
# claude_evolve/tests/test_stagnation.py
import pytest
from claude_evolve.core.stagnation import StagnationLevel, StagnationReport


class TestStagnationLevel:
    def test_levels_exist(self):
        assert StagnationLevel.NONE.value == "none"
        assert StagnationLevel.MILD.value == "mild"
        assert StagnationLevel.MODERATE.value == "moderate"
        assert StagnationLevel.SEVERE.value == "severe"

    def test_level_ordering(self):
        levels = [StagnationLevel.NONE, StagnationLevel.MILD,
                  StagnationLevel.MODERATE, StagnationLevel.SEVERE]
        for i in range(len(levels) - 1):
            assert levels[i].severity < levels[i + 1].severity


class TestStagnationReport:
    def test_report_creation(self):
        report = StagnationReport(
            level=StagnationLevel.MILD,
            iterations_stuck=4,
            best_score=0.9857,
            score_history=[0.98, 0.9857, 0.9857, 0.9857, 0.9857],
            recommended_action="expand_research",
            underexplored_features=[],
            approaches_tried=["exoo_construction", "cyclic_flips"],
        )
        assert report.level == StagnationLevel.MILD
        assert report.iterations_stuck == 4

    def test_report_to_markdown(self):
        report = StagnationReport(
            level=StagnationLevel.SEVERE,
            iterations_stuck=10,
            best_score=0.9857,
            score_history=[0.9857] * 10,
            recommended_action="force_paradigm_shift",
            underexplored_features=["low_complexity"],
            approaches_tried=["approach_a", "approach_b"],
        )
        md = report.to_markdown()
        assert "SEVERE" in md
        assert "10 iterations" in md
        assert "force_paradigm_shift" in md
        assert "low_complexity" in md

    def test_report_to_dict(self):
        report = StagnationReport(
            level=StagnationLevel.NONE,
            iterations_stuck=0,
            best_score=0.5,
            score_history=[0.3, 0.4, 0.5],
            recommended_action="continue",
            underexplored_features=[],
            approaches_tried=[],
        )
        d = report.to_dict()
        assert d["level"] == "none"
        assert d["iterations_stuck"] == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd claude_evolve && python -m pytest tests/test_stagnation.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'claude_evolve.core.stagnation'`

- [ ] **Step 3: Implement StagnationLevel and StagnationReport**

```python
# claude_evolve/claude_evolve/core/stagnation.py
"""Stagnation detection engine for evolutionary optimization."""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional


class StagnationLevel(Enum):
    """Levels of stagnation severity."""
    NONE = "none"
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"

    @property
    def severity(self) -> int:
        return {"none": 0, "mild": 1, "moderate": 2, "severe": 3}[self.value]


@dataclass
class StagnationReport:
    """Report from stagnation analysis."""
    level: StagnationLevel
    iterations_stuck: int
    best_score: float
    score_history: List[float]
    recommended_action: str
    underexplored_features: List[str] = field(default_factory=list)
    approaches_tried: List[str] = field(default_factory=list)

    def to_markdown(self) -> str:
        lines = [
            f"## Stagnation Analysis",
            f"",
            f"**Level:** {self.level.value.upper()}",
            f"**Stuck for:** {self.iterations_stuck} iterations at score {self.best_score:.4f}",
            f"**Score history (last {len(self.score_history)}):** {', '.join(f'{s:.4f}' for s in self.score_history)}",
            f"**Recommended action:** {self.recommended_action}",
        ]
        if self.underexplored_features:
            lines.append(f"**Underexplored regions:** {', '.join(self.underexplored_features)}")
        if self.approaches_tried:
            lines.append(f"**Approaches already tried:** {', '.join(self.approaches_tried)}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "level": self.level.value,
            "iterations_stuck": self.iterations_stuck,
            "best_score": self.best_score,
            "score_history": self.score_history,
            "recommended_action": self.recommended_action,
            "underexplored_features": self.underexplored_features,
            "approaches_tried": self.approaches_tried,
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd claude_evolve && python -m pytest tests/test_stagnation.py -v`
Expected: PASS (all 5 tests)

- [ ] **Step 5: Commit**

```bash
git add claude_evolve/claude_evolve/core/stagnation.py claude_evolve/tests/test_stagnation.py
git commit -m "feat: add StagnationLevel and StagnationReport dataclasses"
```

---

### Task 2: StagnationEngine.analyze()

**Files:**
- Modify: `claude_evolve/claude_evolve/core/stagnation.py`
- Test: `claude_evolve/tests/test_stagnation.py`

- [ ] **Step 1: Write tests for StagnationEngine**

```python
# Append to claude_evolve/tests/test_stagnation.py

from claude_evolve.core.stagnation import StagnationEngine
from claude_evolve.config import DatabaseConfig


class TestStagnationEngine:
    def _make_engine(self, window=5, threshold=0.001):
        return StagnationEngine(window=window, threshold=threshold)

    def test_no_stagnation_with_improving_scores(self):
        engine = self._make_engine()
        history = [0.5, 0.6, 0.7, 0.8, 0.9]
        report = engine.analyze_from_history(history)
        assert report.level == StagnationLevel.NONE
        assert report.iterations_stuck == 0

    def test_mild_stagnation(self):
        engine = self._make_engine(window=5, threshold=0.001)
        history = [0.5, 0.6, 0.7, 0.98, 0.98, 0.98, 0.98]
        report = engine.analyze_from_history(history)
        assert report.level == StagnationLevel.MILD
        assert report.iterations_stuck >= 3

    def test_moderate_stagnation(self):
        engine = self._make_engine(window=5, threshold=0.001)
        history = [0.98] * 8
        report = engine.analyze_from_history(history)
        assert report.level == StagnationLevel.MODERATE

    def test_severe_stagnation(self):
        engine = self._make_engine(window=5, threshold=0.001)
        history = [0.9857] * 15
        report = engine.analyze_from_history(history)
        assert report.level == StagnationLevel.SEVERE

    def test_empty_history(self):
        engine = self._make_engine()
        report = engine.analyze_from_history([])
        assert report.level == StagnationLevel.NONE

    def test_single_entry(self):
        engine = self._make_engine()
        report = engine.analyze_from_history([0.5])
        assert report.level == StagnationLevel.NONE

    def test_recommended_action_for_severe(self):
        engine = self._make_engine()
        history = [0.9857] * 15
        report = engine.analyze_from_history(history)
        assert report.recommended_action == "force_paradigm_shift"

    def test_recommended_action_for_mild(self):
        engine = self._make_engine()
        history = [0.5, 0.6, 0.98, 0.98, 0.98, 0.98]
        report = engine.analyze_from_history(history)
        assert "research" in report.recommended_action or "explore" in report.recommended_action

    def test_exploration_ratio_override(self):
        engine = self._make_engine()
        history = [0.9857] * 8
        report = engine.analyze_from_history(history)
        override = engine.get_exploration_override(report)
        assert override > 0.2  # Default is 0.2, should be boosted
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd claude_evolve && python -m pytest tests/test_stagnation.py::TestStagnationEngine -v`
Expected: FAIL — `ImportError: cannot import name 'StagnationEngine'`

- [ ] **Step 3: Implement StagnationEngine**

```python
# Append to claude_evolve/claude_evolve/core/stagnation.py

class StagnationEngine:
    """Detects evolution stagnation and recommends adaptive responses."""

    # Thresholds for stagnation levels (iterations stuck)
    MILD_THRESHOLD = 3
    MODERATE_THRESHOLD = 6
    SEVERE_THRESHOLD = 10

    def __init__(self, window: int = 5, threshold: float = 0.001):
        self.window = window
        self.threshold = threshold

    def analyze_from_history(self, score_history: List[float]) -> StagnationReport:
        """Analyze a list of best scores (one per iteration) for stagnation."""
        if len(score_history) < 2:
            return StagnationReport(
                level=StagnationLevel.NONE,
                iterations_stuck=0,
                best_score=score_history[-1] if score_history else 0.0,
                score_history=score_history,
                recommended_action="continue",
            )

        best_score = max(score_history)
        iterations_stuck = self._count_stuck_iterations(score_history)
        level = self._classify_level(iterations_stuck)
        action = self._recommend_action(level)

        return StagnationReport(
            level=level,
            iterations_stuck=iterations_stuck,
            best_score=best_score,
            score_history=score_history[-self.window:],
            recommended_action=action,
        )

    def get_exploration_override(self, report: StagnationReport) -> float:
        """Return boosted exploration ratio based on stagnation level."""
        overrides = {
            StagnationLevel.NONE: 0.2,
            StagnationLevel.MILD: 0.4,
            StagnationLevel.MODERATE: 0.7,
            StagnationLevel.SEVERE: 0.9,
        }
        return overrides.get(report.level, 0.2)

    def _count_stuck_iterations(self, history: List[float]) -> int:
        """Count consecutive iterations at the end where score didn't improve."""
        if not history:
            return 0
        best = max(history)
        count = 0
        for score in reversed(history):
            if abs(score - best) <= self.threshold:
                count += 1
            else:
                break
        # Don't count the first iteration that reached the best as "stuck"
        return max(0, count - 1)

    def _classify_level(self, iterations_stuck: int) -> StagnationLevel:
        if iterations_stuck >= self.SEVERE_THRESHOLD:
            return StagnationLevel.SEVERE
        elif iterations_stuck >= self.MODERATE_THRESHOLD:
            return StagnationLevel.MODERATE
        elif iterations_stuck >= self.MILD_THRESHOLD:
            return StagnationLevel.MILD
        return StagnationLevel.NONE

    def _recommend_action(self, level: StagnationLevel) -> str:
        actions = {
            StagnationLevel.NONE: "continue",
            StagnationLevel.MILD: "expand_research_and_explore",
            StagnationLevel.MODERATE: "force_creative_leap",
            StagnationLevel.SEVERE: "force_paradigm_shift",
        }
        return actions.get(level, "continue")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd claude_evolve && python -m pytest tests/test_stagnation.py -v`
Expected: PASS (all 14 tests)

- [ ] **Step 5: Commit**

```bash
git add claude_evolve/claude_evolve/core/stagnation.py claude_evolve/tests/test_stagnation.py
git commit -m "feat: add StagnationEngine with level detection and exploration override"
```

---

### Task 3: Add StagnationConfig to config.py

**Files:**
- Modify: `claude_evolve/claude_evolve/config.py:232-270`
- Test: `claude_evolve/tests/test_config.py`

- [ ] **Step 1: Write test for new config**

```python
# Append to claude_evolve/tests/test_config.py

from claude_evolve.config import StagnationConfig, CrossRunMemoryConfig


class TestStagnationConfig:
    def test_defaults(self):
        cfg = StagnationConfig()
        assert cfg.enabled is True
        assert cfg.window == 5
        assert cfg.threshold == 0.001
        assert cfg.severe_window == 10
        assert cfg.auto_adapt_exploration is True

    def test_custom_values(self):
        cfg = StagnationConfig(window=10, threshold=0.01, enabled=False)
        assert cfg.window == 10
        assert cfg.enabled is False


class TestCrossRunMemoryConfig:
    def test_defaults(self):
        cfg = CrossRunMemoryConfig()
        assert cfg.enabled is True
        assert cfg.max_runs_remembered == 10
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd claude_evolve && python -m pytest tests/test_config.py::TestStagnationConfig -v`
Expected: FAIL — `ImportError: cannot import name 'StagnationConfig'`

- [ ] **Step 3: Add StagnationConfig and CrossRunMemoryConfig to config.py**

Add after line 230 (before the Config class):

```python
@dataclass
class StagnationConfig:
    """Configuration for stagnation detection."""
    enabled: bool = True
    window: int = 5
    threshold: float = 0.001
    severe_window: int = 10
    auto_adapt_exploration: bool = True


@dataclass
class CrossRunMemoryConfig:
    """Configuration for cross-run memory persistence."""
    enabled: bool = True
    memory_path: str = "cross_run_memory.json"
    max_runs_remembered: int = 10
```

Add to Config dataclass (after `evolution_trace` field):

```python
    stagnation: StagnationConfig = field(default_factory=StagnationConfig)
    cross_run_memory: CrossRunMemoryConfig = field(default_factory=CrossRunMemoryConfig)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd claude_evolve && python -m pytest tests/test_config.py -v`
Expected: PASS (all existing + 3 new tests)

- [ ] **Step 5: Commit**

```bash
git add claude_evolve/claude_evolve/config.py claude_evolve/tests/test_config.py
git commit -m "feat: add StagnationConfig and CrossRunMemoryConfig to config"
```

---

### Task 4: Add get_score_history() and detect_stagnation() to ArtifactDatabase

**Files:**
- Modify: `claude_evolve/claude_evolve/core/database.py`
- Test: `claude_evolve/tests/test_database.py`

- [ ] **Step 1: Write tests for score history and stagnation detection**

```python
# Append to claude_evolve/tests/test_database.py

from claude_evolve.core.stagnation import StagnationLevel


class TestScoreHistory:
    def test_empty_database(self, db):
        history = db.get_score_history()
        assert history == []

    def test_score_history_ordering(self, db):
        # Add artifacts with different iteration numbers
        for i, score in enumerate([0.5, 0.7, 0.9]):
            a = Artifact.create(f"code_{i}", {"combined_score": score})
            db.add(a, iteration=i + 1)
        history = db.get_score_history()
        assert len(history) >= 1  # At least tracks best per iteration

    def test_stagnation_detection_none(self, db):
        for i, score in enumerate([0.5, 0.6, 0.7, 0.8, 0.9]):
            a = Artifact.create(f"code_{i}", {"combined_score": score})
            db.add(a, iteration=i + 1)
        report = db.detect_stagnation()
        assert report.level == StagnationLevel.NONE
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd claude_evolve && python -m pytest tests/test_database.py::TestScoreHistory -v`
Expected: FAIL — `AttributeError: 'ArtifactDatabase' object has no attribute 'get_score_history'`

- [ ] **Step 3: Add get_score_history() and detect_stagnation() to database.py**

Add after `get_top_programs()` method (around line 450):

```python
    def get_score_history(self, window: int = 20) -> List[float]:
        """Return best combined_score per iteration, ordered by iteration."""
        iteration_best: Dict[int, float] = {}
        for artifact in self.artifacts.values():
            it = getattr(artifact, "iteration_found", 0)
            score = artifact.metrics.get("combined_score", 0.0)
            if it not in iteration_best or score > iteration_best[it]:
                iteration_best[it] = score

        if not iteration_best:
            return []

        sorted_iterations = sorted(iteration_best.keys())
        # Build cumulative best
        history = []
        running_best = 0.0
        for it in sorted_iterations:
            running_best = max(running_best, iteration_best[it])
            history.append(running_best)

        return history[-window:]

    def detect_stagnation(self, window: int = 5, threshold: float = 0.001) -> "StagnationReport":
        """Detect stagnation in the evolution process."""
        from claude_evolve.core.stagnation import StagnationEngine
        engine = StagnationEngine(window=window, threshold=threshold)
        history = self.get_score_history(window=window * 3)
        return engine.analyze_from_history(history)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd claude_evolve && python -m pytest tests/test_database.py::TestScoreHistory -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add claude_evolve/claude_evolve/core/database.py claude_evolve/tests/test_database.py
git commit -m "feat: add get_score_history() and detect_stagnation() to ArtifactDatabase"
```

---

### Task 5: Implement _is_novel() with code similarity

**Files:**
- Modify: `claude_evolve/claude_evolve/core/database.py:1366`
- Test: `claude_evolve/tests/test_database.py`

- [ ] **Step 1: Write test for novelty checking**

```python
# Append to claude_evolve/tests/test_database.py

class TestNoveltyChecking:
    def test_identical_code_not_novel(self, db):
        a1 = Artifact.create("def foo(): return 1", {"combined_score": 0.5})
        db.add(a1, iteration=1)
        a2 = Artifact.create("def foo(): return 1", {"combined_score": 0.6})
        # With novelty checking, identical code should be flagged
        assert not db._is_novel(a2.id, 0)

    def test_different_code_is_novel(self, db):
        a1 = Artifact.create("def foo(): return 1", {"combined_score": 0.5})
        db.add(a1, iteration=1)
        a2 = Artifact.create("def bar(): return 'completely_different'", {"combined_score": 0.6})
        db.artifacts[a2.id] = a2  # Add without full pipeline
        assert db._is_novel(a2.id, 0)
```

- [ ] **Step 2: Run test to verify behavior**

Run: `cd claude_evolve && python -m pytest tests/test_database.py::TestNoveltyChecking -v`
Expected: FAIL (currently `_is_novel` always returns True)

- [ ] **Step 3: Implement _is_novel() using SequenceMatcher**

Replace the existing `_is_novel()` method at line 1366:

```python
    def _is_novel(self, artifact_id: str, island_idx: int) -> bool:
        """Check if an artifact is sufficiently novel compared to existing population."""
        artifact = self.artifacts.get(artifact_id)
        if artifact is None:
            return True

        content = artifact.content
        if not content:
            return True

        similarity_threshold = getattr(self.config, "similarity_threshold", 0.95)

        # Check against island population
        island_artifacts = [
            a for a in self.artifacts.values()
            if a.id != artifact_id
            and getattr(a, "island", 0) == island_idx
        ]

        for existing in island_artifacts[-20:]:  # Check last 20 for efficiency
            if not existing.content:
                continue
            similarity = self._fast_code_diversity(content, existing.content)
            # _fast_code_diversity returns DIVERSITY (0=identical, 1=completely different)
            # So low diversity means high similarity
            if similarity < (1.0 - similarity_threshold):
                return False

        return True
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd claude_evolve && python -m pytest tests/test_database.py::TestNoveltyChecking -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add claude_evolve/claude_evolve/core/database.py claude_evolve/tests/test_database.py
git commit -m "feat: implement _is_novel() with code similarity checking"
```

---

### Task 6: Add stagnation-aware prompt templates

**Files:**
- Modify: `claude_evolve/claude_evolve/prompt/templates.py`
- Test: `claude_evolve/tests/test_context_builder.py`

- [ ] **Step 1: Write test for new templates**

```python
# Append to claude_evolve/tests/test_context_builder.py

class TestStagnationTemplates:
    def test_paradigm_shift_template_exists(self):
        from claude_evolve.prompt.templates import TemplateManager
        tm = TemplateManager()
        tpl = tm.get_template("paradigm_shift_guidance")
        assert tpl is not None
        assert "fundamentally different" in tpl.lower() or "paradigm" in tpl.lower()

    def test_stagnation_fragment_exists(self):
        from claude_evolve.prompt.templates import TemplateManager
        tm = TemplateManager()
        frag = tm.get_fragment("stagnation_severe")
        assert frag is not None
        assert "stuck" in frag.lower() or "stagnant" in frag.lower() or "paradigm" in frag.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd claude_evolve && python -m pytest tests/test_context_builder.py::TestStagnationTemplates -v`
Expected: FAIL — template not found

- [ ] **Step 3: Add new templates and fragments**

Add to `_INLINE_DEFAULTS` dict in templates.py:

```python
    "paradigm_shift_guidance": """## PARADIGM SHIFT REQUIRED

**The evolution is SEVERELY STAGNATED.** Score has not improved for {iterations_stuck} iterations.

You MUST try a FUNDAMENTALLY DIFFERENT approach. Do NOT:
- Make incremental diffs to the parent
- Use the same algorithmic family as previous attempts
- Repeat any approach listed in "Approaches Already Tried"

Instead:
- Research completely new algorithmic paradigms (web search MANDATORY)
- Consider reformulating the problem (e.g., constraint satisfaction, SAT encoding)
- Start from scratch with a novel construction method
- Look for approaches from adjacent fields (physics simulation, machine learning, etc.)

{stagnation_report}
""",

    "research_guidance": """## Research Phase (MANDATORY)

Before generating any code, you MUST:
1. Search the web for recent papers on this problem domain
2. Read the evaluator source to understand exactly what's being optimized
3. Analyze the top programs in the population for common patterns
4. Identify at least ONE approach not yet tried

Save your research findings for future iterations.

{research_context}
""",
```

Add to `_INLINE_FRAGMENTS` dict:

```python
    "stagnation_none": "",
    "stagnation_mild": "Note: Score improvement has slowed ({iterations_stuck} iterations without progress). Consider broadening your search to different algorithmic approaches.",
    "stagnation_moderate": "WARNING: Evolution appears stuck ({iterations_stuck} iterations at same score). You should try a SIGNIFICANTLY different approach. Web search for alternative methods is strongly recommended.",
    "stagnation_severe": "CRITICAL: Evolution is severely stagnated ({iterations_stuck} iterations without improvement). A complete paradigm shift is required. Do NOT continue with incremental modifications.",
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd claude_evolve && python -m pytest tests/test_context_builder.py::TestStagnationTemplates -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add claude_evolve/claude_evolve/prompt/templates.py claude_evolve/tests/test_context_builder.py
git commit -m "feat: add stagnation-aware templates and paradigm shift guidance"
```

---

### Task 7: Inject stagnation data into ContextBuilder

**Files:**
- Modify: `claude_evolve/claude_evolve/prompt/context_builder.py:70-200, 323-388`
- Test: `claude_evolve/tests/test_context_builder.py`

- [ ] **Step 1: Write test for stagnation-aware context building**

```python
# Append to claude_evolve/tests/test_context_builder.py

from claude_evolve.core.stagnation import StagnationReport, StagnationLevel


class TestStagnationAwareContext:
    def test_severe_stagnation_injects_paradigm_shift(self):
        from claude_evolve.prompt.context_builder import ContextBuilder
        from claude_evolve.config import PromptConfig

        cb = ContextBuilder(PromptConfig())
        report = StagnationReport(
            level=StagnationLevel.SEVERE,
            iterations_stuck=12,
            best_score=0.9857,
            score_history=[0.9857] * 12,
            recommended_action="force_paradigm_shift",
        )

        parent = {"id": "test", "content": "def foo(): pass",
                  "metrics": {"combined_score": 0.9857}, "artifact_type": "python"}

        ctx = cb.build_context(
            parent=parent, iteration=13, best_score=0.9857,
            top_programs=[], inspirations=[], previous_programs=[],
            stagnation_report=report,
        )
        # The improvement areas should mention stagnation
        assert "stagnation" in ctx.get("improvement_areas", "").lower() or \
               "paradigm" in ctx.get("improvement_areas", "").lower() or \
               "stuck" in ctx.get("improvement_areas", "").lower()

    def test_no_stagnation_normal_prompt(self):
        from claude_evolve.prompt.context_builder import ContextBuilder
        from claude_evolve.config import PromptConfig

        cb = ContextBuilder(PromptConfig())
        report = StagnationReport(
            level=StagnationLevel.NONE,
            iterations_stuck=0,
            best_score=0.5,
            score_history=[0.3, 0.4, 0.5],
            recommended_action="continue",
        )

        parent = {"id": "test", "content": "def foo(): pass",
                  "metrics": {"combined_score": 0.5}, "artifact_type": "python"}

        ctx = cb.build_context(
            parent=parent, iteration=4, best_score=0.5,
            top_programs=[], inspirations=[], previous_programs=[],
            stagnation_report=report,
        )
        # Should NOT mention paradigm shift
        assert "paradigm" not in ctx.get("improvement_areas", "").lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd claude_evolve && python -m pytest tests/test_context_builder.py::TestStagnationAwareContext -v`
Expected: FAIL — build_context doesn't accept stagnation_report

- [ ] **Step 3: Modify build_context() to accept and use stagnation_report**

In `context_builder.py`, modify `build_context()` signature (line 70) to add:

```python
    def build_context(
        self,
        parent: Any,
        iteration: int,
        best_score: float,
        top_programs: List[Dict[str, Any]],
        inspirations: List[Dict[str, Any]],
        previous_programs: List[Dict[str, Any]],
        language: str = "python",
        diff_based: bool = True,
        template_key: Optional[str] = None,
        parent_artifacts: Optional[Dict[str, Union[str, bytes]]] = None,
        feature_dimensions: Optional[List[str]] = None,
        current_changes_description: Optional[str] = None,
        stagnation_report: Optional[Any] = None,  # NEW
        research_context: Optional[str] = None,     # NEW
        **kwargs: Any,
    ) -> Dict[str, Any]:
```

In `_identify_improvement_areas()` (line 323), add stagnation-aware guidance at the end:

```python
        # After existing improvement area logic, add stagnation awareness
        stagnation_report = kwargs.get("stagnation_report")
        if stagnation_report and hasattr(stagnation_report, "level"):
            from claude_evolve.core.stagnation import StagnationLevel
            if stagnation_report.level == StagnationLevel.SEVERE:
                areas.append(f"- CRITICAL: Score stuck for {stagnation_report.iterations_stuck} iterations. A complete paradigm shift is required — try a fundamentally different approach.")
            elif stagnation_report.level == StagnationLevel.MODERATE:
                areas.append(f"- WARNING: Score stuck for {stagnation_report.iterations_stuck} iterations. Try a significantly different algorithmic approach. Web search recommended.")
            elif stagnation_report.level == StagnationLevel.MILD:
                areas.append(f"- Note: Score stagnant for {stagnation_report.iterations_stuck} iterations. Consider broader exploration.")
```

Pass `stagnation_report` through to `_identify_improvement_areas` via kwargs.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd claude_evolve && python -m pytest tests/test_context_builder.py::TestStagnationAwareContext -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add claude_evolve/claude_evolve/prompt/context_builder.py claude_evolve/tests/test_context_builder.py
git commit -m "feat: inject stagnation reports into iteration prompts"
```

---

### Task 8: Add `diagnose` CLI command

**Files:**
- Modify: `claude_evolve/claude_evolve/cli.py`
- Test: `claude_evolve/tests/test_cli.py`

- [ ] **Step 1: Write test for diagnose command**

```python
# Append to claude_evolve/tests/test_cli.py
from click.testing import CliRunner


class TestDiagnoseCommand:
    def test_diagnose_no_state(self, tmp_path):
        from claude_evolve.cli import main
        runner = CliRunner()
        result = runner.invoke(main, ["diagnose", "--state-dir", str(tmp_path / "nonexistent")])
        assert result.exit_code != 0

    def test_diagnose_with_state(self, initialized_state_dir):
        from claude_evolve.cli import main
        runner = CliRunner()
        result = runner.invoke(main, ["diagnose", "--state-dir", str(initialized_state_dir)])
        assert result.exit_code == 0
        # Output should be valid JSON with stagnation level
        import json
        data = json.loads(result.output)
        assert "level" in data
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd claude_evolve && python -m pytest tests/test_cli.py::TestDiagnoseCommand -v`
Expected: FAIL — no `diagnose` command

- [ ] **Step 3: Add diagnose command to cli.py**

Add after the `export` command (around line 496):

```python
@main.command()
@click.option("--state-dir", type=click.Path(), default=".claude/evolve-state",
              help="Path to the evolution state directory.")
def diagnose(state_dir):
    """Analyze evolution stagnation and write diagnostic report."""
    import json
    sm = StateManager(state_dir)
    try:
        sm.load()
    except FileNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)

    db = sm.get_database()
    config = sm.get_config()

    # Run stagnation analysis
    stagnation_cfg = getattr(config, "stagnation", None)
    window = stagnation_cfg.window if stagnation_cfg else 5
    threshold = stagnation_cfg.threshold if stagnation_cfg else 0.001

    report = db.detect_stagnation(window=window, threshold=threshold)

    # Write diagnostic report to state dir
    report_path = os.path.join(state_dir, "diagnostic_report.json")
    with open(report_path, "w") as f:
        json.dump(report.to_dict(), f, indent=2)

    # Output to stdout
    click.echo(json.dumps(report.to_dict(), indent=2))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd claude_evolve && python -m pytest tests/test_cli.py::TestDiagnoseCommand -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add claude_evolve/claude_evolve/cli.py claude_evolve/tests/test_cli.py
git commit -m "feat: add diagnose CLI command for stagnation analysis"
```

---

### Task 9: Integrate diagnose into stop-hook.sh

**Files:**
- Modify: `plugin/hooks/stop-hook.sh:164-177`

- [ ] **Step 1: Add diagnose call to stop hook**

After the target score check (line 162) and before preparing the next iteration (line 164), add:

```bash
# Run stagnation diagnosis before preparing next iteration
set +e
DIAG_OUTPUT=$(claude-evolve diagnose --state-dir "$STATE_DIR" 2>/dev/null)
DIAG_EXIT=$?
set -e

if [[ $DIAG_EXIT -eq 0 ]] && [[ -n "$DIAG_OUTPUT" ]]; then
  STAGNATION_LEVEL=$(echo "$DIAG_OUTPUT" | jq -r '.level // "none"')
else
  STAGNATION_LEVEL="none"
fi
```

Also include stagnation level in the system message (line 239-243):

```bash
if [[ "$STAGNATION_LEVEL" != "none" ]]; then
  SYSTEM_MSG="$SYSTEM_MSG | Stagnation: $STAGNATION_LEVEL"
fi
```

- [ ] **Step 2: Verify stop hook still parses correctly**

Run: `bash -n plugin/hooks/stop-hook.sh && echo "SYNTAX OK"`
Expected: "SYNTAX OK"

- [ ] **Step 3: Commit**

```bash
git add plugin/hooks/stop-hook.sh
git commit -m "feat: integrate stagnation diagnosis into stop hook iteration cycle"
```

---

### Task 10: Enhance `next` command to read diagnostic report

**Files:**
- Modify: `claude_evolve/claude_evolve/cli.py:194-275`

- [ ] **Step 1: Modify next command to read diagnostic report and pass to context builder**

In the `next()` command, after loading state and before building context, add:

```python
    # Read diagnostic report if available
    stagnation_report = None
    diag_path = os.path.join(state_dir, "diagnostic_report.json")
    if os.path.exists(diag_path):
        try:
            import json
            from claude_evolve.core.stagnation import StagnationReport, StagnationLevel
            with open(diag_path) as f:
                diag_data = json.load(f)
            stagnation_report = StagnationReport(
                level=StagnationLevel(diag_data.get("level", "none")),
                iterations_stuck=diag_data.get("iterations_stuck", 0),
                best_score=diag_data.get("best_score", 0.0),
                score_history=diag_data.get("score_history", []),
                recommended_action=diag_data.get("recommended_action", "continue"),
                underexplored_features=diag_data.get("underexplored_features", []),
                approaches_tried=diag_data.get("approaches_tried", []),
            )
        except Exception:
            pass
```

Then pass it to `build_context()`:

```python
    ctx = ctx_builder.build_context(
        parent=parent,
        iteration=db.last_iteration + 1,
        best_score=best_score,
        top_programs=top_programs_dicts,
        inspirations=inspiration_dicts,
        previous_programs=previous_programs,
        language=config.artifact_type,
        diff_based=config.evolution.diff_based,
        parent_artifacts=parent_artifacts,
        feature_dimensions=config.database.feature_dimensions,
        stagnation_report=stagnation_report,  # NEW
    )
```

- [ ] **Step 2: Run existing tests to verify no regressions**

Run: `cd claude_evolve && python -m pytest tests/ -v --timeout=60`
Expected: All existing tests PASS

- [ ] **Step 3: Commit**

```bash
git add claude_evolve/claude_evolve/cli.py
git commit -m "feat: pass stagnation report from diagnose into context builder"
```

---

### Task 11: Cross-Run Memory

**Files:**
- Create: `claude_evolve/claude_evolve/core/memory.py`
- Test: `claude_evolve/tests/test_memory.py`

- [ ] **Step 1: Write tests for CrossRunMemory**

```python
# claude_evolve/tests/test_memory.py
import pytest
import json
from pathlib import Path
from claude_evolve.core.memory import CrossRunMemory


class TestCrossRunMemory:
    def test_save_and_load(self, tmp_path):
        path = str(tmp_path / "memory.json")
        mem = CrossRunMemory(path)
        mem.save_run_summary(
            run_id="run1",
            iterations=17,
            best_score=0.9857,
            approaches_tried=["exoo", "hybrid_sa"],
            hard_barriers=["2 mono-K_5 interlocked"],
            recommendation="try non-Exoo construction",
        )

        mem2 = CrossRunMemory(path)
        runs = mem2.load_previous_runs()
        assert len(runs) == 1
        assert runs[0]["best_score"] == 0.9857

    def test_multiple_runs(self, tmp_path):
        path = str(tmp_path / "memory.json")
        mem = CrossRunMemory(path)
        for i in range(3):
            mem.save_run_summary(
                run_id=f"run{i}",
                iterations=i * 5,
                best_score=0.5 + i * 0.1,
                approaches_tried=[f"approach_{i}"],
            )

        runs = mem.load_previous_runs()
        assert len(runs) == 3

    def test_to_markdown(self, tmp_path):
        path = str(tmp_path / "memory.json")
        mem = CrossRunMemory(path)
        mem.save_run_summary(
            run_id="run1",
            iterations=17,
            best_score=0.9857,
            approaches_tried=["exoo", "hybrid_sa"],
            recommendation="try SAT solver",
        )
        md = mem.to_markdown()
        assert "run1" in md or "0.9857" in md
        assert "SAT solver" in md

    def test_max_runs_limit(self, tmp_path):
        path = str(tmp_path / "memory.json")
        mem = CrossRunMemory(path, max_runs=3)
        for i in range(5):
            mem.save_run_summary(run_id=f"run{i}", iterations=i, best_score=float(i))
        runs = mem.load_previous_runs()
        assert len(runs) == 3  # Only keeps last 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd claude_evolve && python -m pytest tests/test_memory.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement CrossRunMemory**

```python
# claude_evolve/claude_evolve/core/memory.py
"""Cross-run memory for persisting learnings between evolution runs."""

import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional


class CrossRunMemory:
    """Persists key learnings across evolution runs."""

    def __init__(self, path: str, max_runs: int = 10):
        self.path = path
        self.max_runs = max_runs

    def save_run_summary(
        self,
        run_id: str,
        iterations: int,
        best_score: float,
        approaches_tried: Optional[List[str]] = None,
        hard_barriers: Optional[List[str]] = None,
        recommendation: Optional[str] = None,
    ) -> None:
        """Append a run summary to memory."""
        runs = self._load_raw()
        summary = {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "iterations": iterations,
            "best_score": best_score,
            "approaches_tried": approaches_tried or [],
            "hard_barriers": hard_barriers or [],
            "recommendation": recommendation or "",
        }
        runs.append(summary)
        # Keep only last max_runs
        if len(runs) > self.max_runs:
            runs = runs[-self.max_runs:]
        self._save_raw(runs)

    def load_previous_runs(self) -> List[Dict[str, Any]]:
        """Load all previous run summaries."""
        return self._load_raw()

    def to_markdown(self) -> str:
        """Render previous runs as markdown for injection into prompts."""
        runs = self._load_raw()
        if not runs:
            return ""

        lines = ["## Previous Run Knowledge (from cross-run memory)", ""]
        for run in runs:
            lines.append(f"### {run.get('run_id', 'Unknown')} ({run.get('iterations', '?')} iterations, best: {run.get('best_score', '?')})")
            if run.get("approaches_tried"):
                lines.append(f"- Approaches tried: {', '.join(run['approaches_tried'])}")
            if run.get("hard_barriers"):
                lines.append(f"- Hard barriers: {', '.join(run['hard_barriers'])}")
            if run.get("recommendation"):
                lines.append(f"- Recommendation: {run['recommendation']}")
            lines.append("")

        return "\n".join(lines)

    def _load_raw(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.path):
            return []
        try:
            with open(self.path) as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
        except (json.JSONDecodeError, IOError):
            return []

    def _save_raw(self, runs: List[Dict[str, Any]]) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(runs, f, indent=2)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd claude_evolve && python -m pytest tests/test_memory.py -v`
Expected: PASS (all 4 tests)

- [ ] **Step 5: Commit**

```bash
git add claude_evolve/claude_evolve/core/memory.py claude_evolve/tests/test_memory.py
git commit -m "feat: add CrossRunMemory for persisting learnings across runs"
```

---

### Task 12: Create researcher.md and diagnostician.md agent definitions

**Files:**
- Create: `plugin/agents/researcher.md`
- Create: `plugin/agents/diagnostician.md`

- [ ] **Step 1: Create researcher agent**

```markdown
# plugin/agents/researcher.md
---
name: researcher
description: Research agent for evolutionary optimization. Performs literature review, web search, and approach discovery.
model: inherit
allowedTools: Read, Grep, Glob, Bash, WebSearch, WebFetch
---

You are a RESEARCH SPECIALIST for an evolutionary optimization system.

## Mission
Find novel approaches, relevant papers, and implementation insights that could improve the current best solution. Focus on ACTIONABLE findings.

## Protocol
1. **Read the evaluator** to understand exactly what's being optimized
2. **Read the current best program** to understand what's been tried
3. **Read the stagnation report** (if provided) to understand what's NOT working
4. **Search the web** for:
   - Recent papers on the specific problem domain
   - Alternative algorithmic approaches not yet tried
   - Known theoretical bounds and impossibility results
   - Related open-source implementations
5. **Analyze findings** for applicability to the current problem
6. **Output structured JSON** with research findings

## Output Format
Your final message MUST be raw JSON only:
```json
{
  "approaches": [
    {"name": "...", "description": "...", "novelty": "high",
     "implementation_hint": "...", "source_url": "..."}
  ],
  "theoretical_bounds": {"known_upper": "...", "known_lower": "..."},
  "key_papers": [{"title": "...", "url": "...", "relevance": "..."}],
  "approaches_to_avoid": ["..."],
  "recommended_next_step": "..."
}
```
```

- [ ] **Step 2: Create diagnostician agent**

```markdown
# plugin/agents/diagnostician.md
---
name: diagnostician
description: Failure analysis agent. Examines why candidates fail and identifies structural patterns.
model: inherit
allowedTools: Read, Grep, Glob, Bash
---

You are a DIAGNOSTIC SPECIALIST analyzing why evolutionary candidates fail.

## Protocol
1. **Read the evaluator source** completely
2. **Read the last 5-10 candidate results** from the database
3. **Identify patterns:**
   - Do failures cluster around specific test cases?
   - Are there shared structural features in failing candidates?
   - Is there a common "violation signature"?
4. **Classify the failure mode:**
   - LOCAL_MINIMUM: Small changes always make things worse
   - STRUCTURAL_BARRIER: A fundamental constraint prevents improvement
   - PARADIGM_LIMIT: The current approach class has been exhausted
   - EVALUATION_GAP: The evaluator has blind spots
5. **Recommend specific actions**

## Output Format
Your final message MUST be raw JSON only:
```json
{
  "failure_mode": "LOCAL_MINIMUM|STRUCTURAL_BARRIER|PARADIGM_LIMIT|EVALUATION_GAP",
  "evidence": ["..."],
  "violation_structure": {"shared_elements": [], "interlocking_constraints": []},
  "approaches_exhausted": ["..."],
  "recommended_pivot": "...",
  "confidence": 0.85
}
```
```

- [ ] **Step 3: Commit**

```bash
git add plugin/agents/researcher.md plugin/agents/diagnostician.md
git commit -m "feat: add researcher and diagnostician agent definitions"
```

---

### Task 13: Update SKILL.md with stagnation awareness

**Files:**
- Modify: `plugin/skills/evolve/SKILL.md`

- [ ] **Step 1: Add Phase 1.5 (Diagnose) to SKILL.md**

After Phase 1 (Understand) and before Phase 2 (Research), add:

```markdown
### Phase 1.5: Diagnose (if stagnation detected)

If the iteration context contains a **Stagnation Analysis** section:
1. Read the stagnation level (NONE, MILD, MODERATE, SEVERE)
2. Read the list of approaches already tried
3. Read any failure patterns identified
4. Plan your strategy accordingly:
   - **MILD**: Broaden your search, try a different variant
   - **MODERATE**: Use web search to find alternative algorithms. Spawn a research subagent.
   - **SEVERE**: Complete paradigm shift required. Ignore the parent program. Start from scratch with a fundamentally different approach. Web search is MANDATORY.
5. If cross-run memory is present, read it to avoid repeating past failures
```

- [ ] **Step 2: Commit**

```bash
git add plugin/skills/evolve/SKILL.md
git commit -m "feat: add stagnation-aware Phase 1.5 to evolution skill"
```

---

### Task 14: Run full test suite and verify no regressions

- [ ] **Step 1: Run complete test suite**

Run: `cd claude_evolve && python -m pytest tests/ -v --timeout=120 2>&1 | tail -30`
Expected: All tests PASS

- [ ] **Step 2: Run the evaluator manually to verify R(5,5) still works**

Run: `cd /home/bud/Desktop/claudeEvolve && timeout 20 python3 ramsey_R5_5/evaluator.py ramsey_R5_5/program.py 2>&1 | head -5`
Expected: Valid JSON output with combined_score

- [ ] **Step 3: Final commit with all Phase 1 changes**

```bash
git add -A
git commit -m "feat: Claude Evolve v2 Phase 1 complete — stagnation engine, enhanced prompts, cross-run memory, researcher/diagnostician agents"
```
