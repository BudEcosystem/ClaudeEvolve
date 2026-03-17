# Approach B: Evolutionary Leap — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make ClaudeEvolve competitive with AdaEvolve, ShinkaEvolve, ReEvo, and EoH by fixing production bugs, adding quick wins from literature, and implementing 5 critical gap closers.

**Architecture:** Three incremental phases. Phase 1 fixes existing bugs. Phase 2 adds 6 proven techniques requiring minimal architectural change. Phase 3 adds 5 new modules (improvement_signal, scratchpad, reflection, ucb_selector, orchestrator) that integrate via an IterationOrchestrator. All new modules are independently testable.

**Tech Stack:** Python 3.10+, pytest, click CLI, dataclasses, JSON persistence, numpy (for existing warm_cache)

**Spec:** `docs/superpowers/specs/2026-03-17-approach-b-evolutionary-leap-design.md`

---

## File Map

### New Files
| File | Responsibility |
|------|---------------|
| `claude_evolve/claude_evolve/core/improvement_signal.py` | G_t continuous improvement signal + exploration intensity |
| `claude_evolve/claude_evolve/core/scratchpad.py` | Meta-scratchpad: periodic pattern synthesis from evolution history |
| `claude_evolve/claude_evolve/core/reflection.py` | Short/long-term reflection engine using semantic fingerprint diffs |
| `claude_evolve/claude_evolve/core/ucb_selector.py` | UCB1 bandit for strategy selection |
| `claude_evolve/claude_evolve/core/orchestrator.py` | IterationOrchestrator: wires all modules for next/submit lifecycle |
| `claude_evolve/tests/test_improvement_signal.py` | Tests for G_t signal |
| `claude_evolve/tests/test_scratchpad.py` | Tests for meta-scratchpad |
| `claude_evolve/tests/test_reflection.py` | Tests for reflection engine |
| `claude_evolve/tests/test_ucb_selector.py` | Tests for UCB1 selector |
| `claude_evolve/tests/test_orchestrator.py` | Tests for orchestrator integration |

### Modified Files
| File | What Changes |
|------|-------------|
| `claude_evolve/claude_evolve/core/artifact.py:13-77` | Add `rationale`, `offspring_count` fields |
| `claude_evolve/claude_evolve/core/database.py:58-1798` | Thread safety, island-artifact sync, power-law selection |
| `claude_evolve/claude_evolve/core/warm_cache.py:30-247` | LRU eviction policy |
| `claude_evolve/claude_evolve/core/evaluator.py` | Error logging with program path, exponential backoff |
| `claude_evolve/claude_evolve/config.py:306-413` | Add 4 new config dataclasses to master Config |
| `claude_evolve/claude_evolve/cli.py:209-370,737-873` | Delegate next/submit to orchestrator, iteration manifest |
| `claude_evolve/claude_evolve/prompt/context_builder.py:70-265` | New params: scratchpad, reflection, evaluator_source, comparison, rationale |
| `claude_evolve/claude_evolve/prompt/templates.py:299-440` | New templates for all new context sections |
| `claude_evolve/tests/test_database.py` | New tests for thread safety, sync, power-law |
| `claude_evolve/tests/test_warm_cache.py` | New tests for LRU eviction |

---

## Chunk 1: Phase 1 — Production Fixes (Tasks 1-7)

### Task 1: Cross-Run Memory Population (Fix 1)

**Files:**
- Modify: `claude_evolve/claude_evolve/cli.py:737-873` (submit command)
- Modify: `claude_evolve/claude_evolve/cli.py:89-202` (init command — add run_id)
- Test: `claude_evolve/tests/test_cli.py`

- [ ] **Step 1: Write failing test for memory population on submit**

```python
# In test_cli.py (or new test_cli_memory.py)
import json, os, tempfile
from unittest.mock import patch, MagicMock
from click.testing import CliRunner
from claude_evolve.cli import main

def test_submit_populates_memory_on_failure():
    """When submit score <= best, failed approach should be recorded in memory."""
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmpdir:
        state_dir = os.path.join(tmpdir, '.claude', 'evolve-state')
        os.makedirs(state_dir, exist_ok=True)
        # Create minimal state: iteration manifest, database, memory
        manifest = {
            "iteration": 5, "run_id": "test1234",
            "selected_strategy_id": "default-incremental",
            "parent_artifact_id": "p1", "parent_score": 0.8,
            "parent_island_id": 0
        }
        with open(os.path.join(state_dir, 'current_iteration.json'), 'w') as f:
            json.dump(manifest, f)
        # Create candidate file
        candidate = os.path.join(tmpdir, 'candidate.py')
        with open(candidate, 'w') as f:
            f.write('print("hello")')
        # Mock evaluator to return score lower than parent
        with patch('claude_evolve.cli._run_evaluator', return_value={"combined_score": 0.5}):
            with patch('claude_evolve.cli._load_database') as mock_db:
                mock_db.return_value = MagicMock(get_best=MagicMock(return_value=MagicMock(metrics={"combined_score": 0.8})))
                result = runner.invoke(main, ['submit', '--candidate', candidate, '--state-dir', state_dir])
        # Check memory was populated
        memory_path = os.path.join(state_dir, 'cross_run_memory.json')
        assert os.path.exists(memory_path)
        with open(memory_path) as f:
            data = json.load(f)
        assert any(l.get('category') == 'failed_approach' for l in data.get('learnings', []))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd claude_evolve && python -m pytest tests/test_cli.py::test_submit_populates_memory_on_failure -v`
Expected: FAIL — no memory population logic exists yet

- [ ] **Step 3: Add run_id generation to init command**

In `cli.py` init command (~line 150), after creating state directory:
```python
import uuid
run_id = uuid.uuid4().hex[:8]
# Save to state metadata
metadata_path = os.path.join(state_dir, 'metadata.json')
metadata = {}
if os.path.exists(metadata_path):
    with open(metadata_path) as f:
        metadata = json.load(f)
metadata['run_id'] = run_id
with open(metadata_path, 'w') as f:
    json.dump(metadata, f)
```

- [ ] **Step 4: Add iteration manifest writing to next command**

In `cli.py` next command (~line 280), after selecting parent and strategy:
```python
manifest = {
    "iteration": iteration,
    "run_id": metadata.get('run_id', 'unknown'),
    "selected_strategy_id": selected_strategy.id if selected_strategy else "unknown",
    "parent_artifact_id": parent.id if parent else None,
    "parent_score": parent.metrics.get('combined_score', 0.0) if parent else 0.0,
    "parent_island_id": getattr(parent, 'island_id', 0) if parent else 0,
}
with open(os.path.join(state_dir, 'current_iteration.json'), 'w') as f:
    json.dump(manifest, f)
```

- [ ] **Step 5: Add memory population to submit command**

In `cli.py` submit command (~line 830), after evaluation:
```python
# Load iteration manifest
manifest_path = os.path.join(state_dir, 'current_iteration.json')
manifest = {}
if os.path.exists(manifest_path):
    with open(manifest_path) as f:
        manifest = json.load(f)

# Populate cross-run memory
memory_path = os.path.join(state_dir, 'cross_run_memory.json')
memory = CrossRunMemory(memory_path)
memory.load()
score = metrics.get('combined_score', 0.0)
best_score = manifest.get('parent_score', 0.0)
strategy_name = manifest.get('selected_strategy_id', 'unknown')
run_id = manifest.get('run_id', 'unknown')
iteration = manifest.get('iteration', 0)

if score <= best_score:
    memory.add_failed_approach(
        description=f"Strategy '{strategy_name}' on iteration {iteration}",
        score=score, iteration=iteration, run_id=run_id
    )
else:
    memory.add_strategy(
        name=strategy_name,
        description=f"Improved from {best_score:.4f} to {score:.4f}",
        score=score, run_id=run_id
    )
memory.save()
```

- [ ] **Step 6: Run tests**

Run: `cd claude_evolve && python -m pytest tests/test_cli.py::test_submit_populates_memory_on_failure -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add claude_evolve/claude_evolve/cli.py claude_evolve/tests/test_cli.py
git commit -m "fix: populate cross-run memory on submit with run_id and iteration"
```

---

### Task 2: Strategy Outcome Recording (Fix 2)

**Files:**
- Modify: `claude_evolve/claude_evolve/cli.py:737-873` (submit command)
- Test: `claude_evolve/tests/test_cli.py`

- [ ] **Step 1: Write failing test**

```python
def test_submit_records_strategy_outcome():
    """Strategy outcome should be recorded with score delta after evaluation."""
    # Similar setup to Task 1, mock strategy_mgr
    # Assert strategy_mgr.record_outcome() was called with correct strategy_id and delta
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd claude_evolve && python -m pytest tests/test_cli.py::test_submit_records_strategy_outcome -v`
Expected: FAIL

- [ ] **Step 3: Add strategy outcome recording to submit**

In `cli.py` submit command, after memory population (from Task 1):
```python
# Record strategy outcome
strategy_id = manifest.get('selected_strategy_id')
parent_score = manifest.get('parent_score', 0.0)
if strategy_id:
    strategy_path = os.path.join(state_dir, 'strategies.json')
    if os.path.exists(strategy_path):
        strategy_mgr = StrategyManager()
        strategy_mgr.load(strategy_path)
        strategy_mgr.record_outcome(strategy_id, score - parent_score)
        strategy_mgr.save(strategy_path)
```

- [ ] **Step 4: Run test**

Run: `cd claude_evolve && python -m pytest tests/test_cli.py::test_submit_records_strategy_outcome -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add claude_evolve/claude_evolve/cli.py claude_evolve/tests/test_cli.py
git commit -m "fix: record strategy outcomes on submit for feedback loop"
```

---

### Task 3: Research Trigger Logic (Fix 3)

**Files:**
- Modify: `claude_evolve/claude_evolve/cli.py:209-370` (next command, ~line 300)
- Test: `claude_evolve/tests/test_cli.py`

- [ ] **Step 1: Write failing test**

```python
def test_next_excludes_research_when_no_stagnation():
    """Research text should NOT be included when stagnation is NONE and trigger is on_stagnation."""
    # Setup: stagnation_level=NONE, research config trigger="on_stagnation"
    # Assert: research_text is empty/None in the context output
```

- [ ] **Step 2: Run test to verify it fails**

Expected: FAIL — research text is always included unconditionally

- [ ] **Step 3: Gate research text on should_research()**

In `cli.py` next command (~line 295-300), change from unconditional:
```python
# Before (always includes):
research_text = research_log.format_for_prompt()

# After (gated):
research_text = ""
if research_log.should_research(stagnation_level):
    research_text = research_log.format_for_prompt()
```

- [ ] **Step 4: Run test**

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add claude_evolve/claude_evolve/cli.py claude_evolve/tests/test_cli.py
git commit -m "fix: gate research text on should_research() trigger logic"
```

---

### Task 4: Thread Safety for ArtifactDatabase (Fix 4)

**Files:**
- Modify: `claude_evolve/claude_evolve/core/database.py:58-70` (init), `174` (add), `277` (sample), `1178` (migrate)
- Test: `claude_evolve/tests/test_database.py`

- [ ] **Step 1: Write failing test for concurrent adds**

```python
import threading
from claude_evolve.core.database import ArtifactDatabase
from claude_evolve.core.artifact import Artifact
from claude_evolve.config import DatabaseConfig

def test_concurrent_adds_produce_correct_count():
    """Multiple threads calling add() should produce correct total artifact count."""
    config = DatabaseConfig()
    db = ArtifactDatabase(config)
    errors = []

    def add_artifact(i):
        try:
            a = Artifact(content=f"program_{i}", artifact_type="python")
            db.add(a, {"combined_score": 0.5 + i * 0.001})
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=add_artifact, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0, f"Errors during concurrent add: {errors}"
    assert len(db.artifacts) == 20
```

- [ ] **Step 2: Run test to verify it fails (race condition)**

Run: `cd claude_evolve && python -m pytest tests/test_database.py::test_concurrent_adds_produce_correct_count -v`
Expected: May FAIL intermittently or produce wrong count

- [ ] **Step 3: Add RLock to ArtifactDatabase**

In `database.py` `__init__()` (~line 70), add:
```python
import threading
self._lock = threading.RLock()
```

Wrap `add()` (~line 174):
```python
def add(self, artifact, metrics, ...):
    with self._lock:
        # ... existing add logic ...
```

Wrap `sample()` (~line 277), `sample_from_island()` (~line 300), `migrate_programs()` (~line 1178) similarly.

- [ ] **Step 4: Run test**

Expected: PASS consistently

- [ ] **Step 5: Commit**

```bash
git add claude_evolve/claude_evolve/core/database.py claude_evolve/tests/test_database.py
git commit -m "fix: add thread safety to ArtifactDatabase with RLock"
```

---

### Task 5: Island-Artifact Sync Guard (Fix 5)

**Files:**
- Modify: `claude_evolve/claude_evolve/core/database.py` (remove method + sampling)
- Test: `claude_evolve/tests/test_database.py`

- [ ] **Step 1: Write failing test**

```python
def test_remove_cleans_island_sets():
    """After remove(), no island should contain the removed artifact's ID."""
    config = DatabaseConfig()
    db = ArtifactDatabase(config)
    a = Artifact(content="test", artifact_type="python")
    db.add(a, {"combined_score": 0.5})
    aid = a.id
    db.remove(aid)
    for island_id, members in db.islands.items():
        assert aid not in members, f"Orphaned ID {aid} in island {island_id}"

def test_sampling_with_orphaned_ids_does_not_crash():
    """If an island set has orphaned IDs, sampling should skip them gracefully."""
    config = DatabaseConfig()
    db = ArtifactDatabase(config)
    a = Artifact(content="test", artifact_type="python")
    db.add(a, {"combined_score": 0.5})
    # Artificially inject orphaned ID
    island_id = list(db.islands.keys())[0]
    db.islands[island_id].add("orphan_id_999")
    # Sampling should not crash
    result = db.sample()
    assert result is not None or result is None  # just verify no exception
```

- [ ] **Step 2: Run test to verify it fails**

Expected: FAIL if remove() doesn't clean island sets

- [ ] **Step 3: Fix remove() and add guard in sampling**

In `database.py`, ensure the `remove()` method cleans island sets:
```python
def remove(self, artifact_id):
    with self._lock:
        if artifact_id in self.artifacts:
            del self.artifacts[artifact_id]
        # Clean from all island sets
        for island_id in self.islands:
            self.islands[island_id].discard(artifact_id)
        # Clean from feature maps
        for island_id in self.island_feature_maps:
            cells_to_clean = []
            for cell, aid in self.island_feature_maps[island_id].items():
                if aid == artifact_id:
                    cells_to_clean.append(cell)
            for cell in cells_to_clean:
                del self.island_feature_maps[island_id][cell]
```

In `_sample_from_island_weighted()` (~line 919), add guard:
```python
# Filter to only valid IDs
valid_ids = [pid for pid in island_programs if pid in self.artifacts]
if not valid_ids:
    return None
```

- [ ] **Step 4: Run tests**

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add claude_evolve/claude_evolve/core/database.py claude_evolve/tests/test_database.py
git commit -m "fix: ensure island-artifact sync on remove, guard orphaned IDs in sampling"
```

---

### Task 6: Warm Cache LRU Eviction (Fix 6)

**Files:**
- Modify: `claude_evolve/claude_evolve/core/warm_cache.py:30-247`
- Test: `claude_evolve/tests/test_warm_cache.py`

- [ ] **Step 1: Write failing test**

```python
import time
from claude_evolve.core.warm_cache import WarmCache

def test_lru_eviction_removes_oldest():
    """Cache with max_items=3 should evict oldest item when 4th is added."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = WarmCache(tmpdir, max_items=3)
        cache.put_json("key1", {"v": 1}, description="first")
        time.sleep(0.01)
        cache.put_json("key2", {"v": 2}, description="second")
        time.sleep(0.01)
        cache.put_json("key3", {"v": 3}, description="third")
        time.sleep(0.01)
        cache.put_json("key4", {"v": 4}, description="fourth")
        # key1 should be evicted (oldest)
        assert not cache.has("key1")
        assert cache.has("key2")
        assert cache.has("key3")
        assert cache.has("key4")

def test_lru_get_updates_access_time():
    """Accessing an item via get should update its access time, preventing eviction."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = WarmCache(tmpdir, max_items=3)
        cache.put_json("key1", {"v": 1}, description="first")
        time.sleep(0.01)
        cache.put_json("key2", {"v": 2}, description="second")
        time.sleep(0.01)
        cache.put_json("key3", {"v": 3}, description="third")
        # Access key1 to refresh its access time
        cache.get_json("key1")
        time.sleep(0.01)
        cache.put_json("key4", {"v": 4}, description="fourth")
        # key2 should be evicted (oldest access), not key1
        assert cache.has("key1")
        assert not cache.has("key2")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd claude_evolve && python -m pytest tests/test_warm_cache.py::test_lru_eviction_removes_oldest -v`
Expected: FAIL — no max_items parameter or eviction logic exists

- [ ] **Step 3: Implement LRU eviction**

In `warm_cache.py` `__init__()` (~line 42), add `max_items` param:
```python
def __init__(self, state_dir: str, max_items: int = 50):
    self.state_dir = state_dir
    self.max_items = max_items
    self.items_dir = os.path.join(state_dir, 'items')
    self._access_times: dict[str, float] = {}
    os.makedirs(self.items_dir, exist_ok=True)
    self._load_access_times()
```

Add access time tracking:
```python
def _load_access_times(self):
    path = os.path.join(self.state_dir, 'access_times.json')
    if os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            self._access_times = json.load(f)

def _save_access_times(self):
    path = os.path.join(self.state_dir, 'access_times.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(self._access_times, f)

def _touch(self, key: str):
    self._access_times[key] = time.time()
    self._save_access_times()

def _evict_if_needed(self):
    if len(self._access_times) > self.max_items:
        # Sort by access time, evict oldest
        sorted_keys = sorted(self._access_times, key=lambda k: self._access_times[k])
        while len(self._access_times) > self.max_items:
            oldest = sorted_keys.pop(0)
            self._remove_item(oldest)

def _remove_item(self, key: str):
    self._access_times.pop(key, None)
    # Remove all files matching this key
    for ext in ['.npy', '.json', '.txt', '.meta.json']:
        path = os.path.join(self.items_dir, f"{key}{ext}")
        if os.path.exists(path):
            os.remove(path)
    self._save_access_times()
```

Add `_touch()` calls to all `put_*()` and `get_*()` methods. Add `_evict_if_needed()` call at end of all `put_*()` methods.

- [ ] **Step 4: Run tests**

Run: `cd claude_evolve && python -m pytest tests/test_warm_cache.py -v`
Expected: All pass including new LRU tests

- [ ] **Step 5: Commit**

```bash
git add claude_evolve/claude_evolve/core/warm_cache.py claude_evolve/tests/test_warm_cache.py
git commit -m "fix: add LRU eviction to warm cache with configurable max_items"
```

---

### Task 7: Evaluator Error Logging (Fix 7)

**Files:**
- Modify: `claude_evolve/claude_evolve/core/evaluator.py`
- Test: `claude_evolve/tests/test_evaluator.py`

- [ ] **Step 1: Write failing test**

```python
import logging
from claude_evolve.core.evaluator import Evaluator

def test_timeout_logs_program_path(caplog):
    """Timeout should log which program was being evaluated."""
    # Create evaluator with very short timeout, feed it a slow program
    # Assert log contains the program path
    with caplog.at_level(logging.WARNING):
        # ... trigger timeout ...
        pass
    assert "candidate.py" in caplog.text or "timed out" in caplog.text
```

- [ ] **Step 2: Run test to verify it fails**

Expected: FAIL — current timeout log doesn't include program path

- [ ] **Step 3: Fix timeout logging and add backoff**

In evaluator.py, find the timeout handler and update:
```python
except asyncio.TimeoutError:
    logger.warning(
        f"Evaluation of '{program_path}' timed out after {self.config.timeout}s"
    )
```

For retry backoff, replace fixed sleep:
```python
# Before:
await asyncio.sleep(0.5)

# After:
backoff = min(0.5 * (2 ** retry_count), 4.0)  # 0.5, 1.0, 2.0, 4.0 max
await asyncio.sleep(backoff)
```

- [ ] **Step 4: Run tests**

Run: `cd claude_evolve && python -m pytest tests/test_evaluator.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add claude_evolve/claude_evolve/core/evaluator.py claude_evolve/tests/test_evaluator.py
git commit -m "fix: log program path on timeout, add exponential backoff for retries"
```

---

## Chunk 2: Phase 2 — Quick Wins (Tasks 8-13)

### Task 8: Config Additions

**Files:**
- Modify: `claude_evolve/claude_evolve/config.py:306-413`
- Test: `claude_evolve/tests/test_config.py`

All subsequent tasks need the new config dataclasses. Add them first.

- [ ] **Step 1: Write failing test**

```python
from claude_evolve.config import Config, ImprovementSignalConfig, SelectionConfig, ScratchpadConfig, ReflectionConfig

def test_config_has_new_sections():
    """Config should include improvement_signal, selection, scratchpad, reflection."""
    config = Config()
    assert hasattr(config, 'improvement_signal')
    assert hasattr(config, 'selection')
    assert hasattr(config, 'scratchpad')
    assert hasattr(config, 'reflection')
    assert config.improvement_signal.rho == 0.95
    assert config.selection.strategy_selection == "ucb1"
    assert config.scratchpad.synthesis_interval == 10
    assert config.reflection.max_short_reflections == 20

def test_config_from_dict_with_new_sections():
    """Config.from_dict should deserialize new sections."""
    d = {"improvement_signal": {"rho": 0.9}, "selection": {"ucb_c": 2.0}}
    config = Config.from_dict(d)
    assert config.improvement_signal.rho == 0.9
    assert config.selection.ucb_c == 2.0
```

- [ ] **Step 2: Run test to verify it fails**

Expected: FAIL — new config classes don't exist

- [ ] **Step 3: Add new config dataclasses**

In `config.py`, before the `Config` class (~line 303), add:

```python
@dataclass
class ImprovementSignalConfig:
    enabled: bool = True
    rho: float = 0.95
    i_min: float = 0.1
    i_max: float = 0.7
    meta_threshold: float = 0.12

@dataclass
class SelectionConfig:
    strategy_selection: str = "ucb1"
    ucb_c: float = 1.414
    ucb_decay: float = 0.95
    parent_selection: str = "power_law"
    novelty_gate_min_novelty: float = 0.05

@dataclass
class ScratchpadConfig:
    enabled: bool = True
    synthesis_interval: int = 10

@dataclass
class ReflectionConfig:
    enabled: bool = True
    max_short_reflections: int = 20
    synthesis_interval: int = 5
```

In the `Config` dataclass (~line 306), add fields:
```python
improvement_signal: ImprovementSignalConfig = field(default_factory=ImprovementSignalConfig)
selection: SelectionConfig = field(default_factory=SelectionConfig)
scratchpad: ScratchpadConfig = field(default_factory=ScratchpadConfig)
reflection: ReflectionConfig = field(default_factory=ReflectionConfig)
```

- [ ] **Step 4: Run tests**

Run: `cd claude_evolve && python -m pytest tests/test_config.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add claude_evolve/claude_evolve/config.py claude_evolve/tests/test_config.py
git commit -m "feat: add config dataclasses for improvement signal, selection, scratchpad, reflection"
```

---

### Task 9: Meta-Scratchpad (QW1)

**Files:**
- Create: `claude_evolve/claude_evolve/core/scratchpad.py`
- Create: `claude_evolve/tests/test_scratchpad.py`

- [ ] **Step 1: Write failing tests**

```python
import os, json, tempfile
from claude_evolve.core.scratchpad import MetaScratchpad
from claude_evolve.core.artifact import Artifact

def test_should_synthesize_on_interval():
    with tempfile.TemporaryDirectory() as d:
        sp = MetaScratchpad(d, synthesis_interval=10)
        assert not sp.should_synthesize(5)
        assert sp.should_synthesize(10)
        assert sp.should_synthesize(20)
        assert not sp.should_synthesize(0)

def test_synthesize_extracts_patterns():
    with tempfile.TemporaryDirectory() as d:
        sp = MetaScratchpad(d)
        arts = [
            Artifact(content="import numpy\ndef solve(): return simulated_annealing()", artifact_type="python"),
            Artifact(content="import numpy\ndef solve(): return simulated_annealing(x)", artifact_type="python"),
            Artifact(content="import scipy\ndef solve(): return gradient_descent()", artifact_type="python"),
        ]
        result = sp.synthesize(arts, [0.8, 0.7, 0.3], ["gradient_descent approach failed"])
        assert "numpy" in result or "simulated_annealing" in result  # common pattern in top 2
        assert "gradient_descent" in result  # failed pattern

def test_save_and_load_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        sp = MetaScratchpad(d)
        sp.save("## Patterns\n- use SA\n")
        loaded = sp.load()
        assert "use SA" in loaded

def test_synthesize_with_empty_artifacts():
    with tempfile.TemporaryDirectory() as d:
        sp = MetaScratchpad(d)
        result = sp.synthesize([], [], [])
        assert result == "" or "No patterns" in result
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd claude_evolve && python -m pytest tests/test_scratchpad.py -v`
Expected: FAIL — module doesn't exist

- [ ] **Step 3: Implement MetaScratchpad**

Create `claude_evolve/claude_evolve/core/scratchpad.py`:

```python
"""Meta-scratchpad: periodic pattern synthesis from evolution history.

Inspired by ShinkaEvolve (ICLR 2026). Every N iterations, analyzes top artifacts
and recent failures to extract patterns that work, patterns that fail, and
recommended directions. Deterministic — no LLM calls.
"""

import json
import os
from typing import Optional

from claude_evolve.core.artifact import Artifact
from claude_evolve.core.novelty import semantic_fingerprint


class MetaScratchpad:
    """Generates and persists accumulated insights from evolution history."""

    def __init__(self, state_dir: str, synthesis_interval: int = 10):
        self.state_dir = state_dir
        self.synthesis_interval = synthesis_interval
        self._path = os.path.join(state_dir, 'meta_scratchpad.json')

    def should_synthesize(self, current_iteration: int) -> bool:
        """True if current_iteration is a positive multiple of synthesis_interval."""
        return current_iteration > 0 and current_iteration % self.synthesis_interval == 0

    def synthesize(
        self,
        top_artifacts: list,
        recent_scores: list,
        failed_approaches: list,
    ) -> str:
        """Generate structured scratchpad from recent evolution history.

        Uses semantic_fingerprint() to extract concepts from artifacts, then
        computes set intersections/differences to find patterns.
        """
        if not top_artifacts:
            return ""

        # Extract fingerprints from top artifacts
        top_fps = [semantic_fingerprint(a.content) for a in top_artifacts]
        all_top_concepts = set()
        for fp in top_fps:
            all_top_concepts.update(fp)

        # Patterns that work: concepts in 2+ top artifacts
        concept_counts: dict[str, int] = {}
        for fp in top_fps:
            for concept in fp:
                concept_counts[concept] = concept_counts.get(concept, 0) + 1
        working_patterns = sorted(
            [c for c, count in concept_counts.items() if count >= 2],
            key=lambda c: concept_counts[c],
            reverse=True,
        )[:10]

        # Patterns that fail: concepts from failed approaches
        failed_concepts: set[str] = set()
        for desc in failed_approaches:
            failed_concepts.update(semantic_fingerprint(desc))
        failing_patterns = sorted(failed_concepts - all_top_concepts)[:5]

        # Build scratchpad text
        sections = []
        if working_patterns:
            items = ", ".join(working_patterns[:5])
            sections.append(f"**Patterns That Work:** {items}")
        if failing_patterns:
            items = ", ".join(failing_patterns[:5])
            sections.append(f"**Patterns That Fail:** {items}")
        if recent_scores and len(recent_scores) >= 2:
            trend = "improving" if recent_scores[-1] > recent_scores[0] else "stagnating"
            sections.append(f"**Score Trend:** {trend} ({recent_scores[0]:.4f} -> {recent_scores[-1]:.4f})")

        return "\n".join(sections) if sections else ""

    def load(self) -> Optional[str]:
        """Load current scratchpad from state."""
        if os.path.exists(self._path):
            with open(self._path, encoding='utf-8') as f:
                data = json.load(f)
                return data.get('content', '')
        return None

    def save(self, content: str) -> None:
        """Persist scratchpad to state."""
        os.makedirs(os.path.dirname(self._path), exist_ok=True)
        with open(self._path, 'w', encoding='utf-8') as f:
            json.dump({'content': content}, f)
```

- [ ] **Step 4: Run tests**

Run: `cd claude_evolve && python -m pytest tests/test_scratchpad.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add claude_evolve/claude_evolve/core/scratchpad.py claude_evolve/tests/test_scratchpad.py
git commit -m "feat: add meta-scratchpad for periodic pattern synthesis (ShinkaEvolve)"
```

---

### Task 10: Reflexion on Failures (QW2)

**Files:**
- Modify: `claude_evolve/claude_evolve/cli.py` (submit + next)
- Test: `claude_evolve/tests/test_cli.py`

- [ ] **Step 1: Write failing test**

```python
def test_failure_captured_in_recent_failures():
    """When eval score < parent * 0.95, failure should be stored."""
    # Setup: submit with score=0.3 when parent_score=0.8
    # Assert: evolve-state/recent_failures.json exists and contains entry

def test_recent_failures_circular_buffer():
    """Recent failures should keep only last 5 entries."""
    # Write 6 failures, assert only last 5 remain
```

- [ ] **Step 2: Run to verify fails**

- [ ] **Step 3: Implement failure capture in submit and loading in next**

In submit (~line 835), after memory population:
```python
# Capture failure for reflexion
failures_path = os.path.join(state_dir, 'recent_failures.json')
failures = []
if os.path.exists(failures_path):
    with open(failures_path, encoding='utf-8') as f:
        failures = json.load(f)
if score < parent_score * 0.95 or metrics.get('error'):
    failures.append({
        "approach": strategy_name,
        "error": metrics.get('error'),
        "score": score,
        "parent_score": parent_score,
        "iteration": iteration,
    })
    failures = failures[-5:]  # Circular buffer
    with open(failures_path, 'w', encoding='utf-8') as f:
        json.dump(failures, f)
```

In next (~line 280), load and format:
```python
failures_path = os.path.join(state_dir, 'recent_failures.json')
failures_text = ""
if os.path.exists(failures_path):
    with open(failures_path, encoding='utf-8') as f:
        failures = json.load(f)
    if failures:
        lines = ["## Recent Failures (Avoid These)"]
        for f_entry in failures:
            err = f_entry.get('error') or f"score dropped to {f_entry.get('score', '?')}"
            lines.append(f"- Approach: {f_entry.get('approach', '?')}. Result: {err}")
        failures_text = "\n".join(lines)
```

- [ ] **Step 4: Run tests**

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add claude_evolve/claude_evolve/cli.py claude_evolve/tests/test_cli.py
git commit -m "feat: add failure reflexion with circular buffer (ShinkaEvolve)"
```

---

### Task 11: Ascending Score Sort + Evaluator Context (QW4 + QW5)

**Files:**
- Modify: `claude_evolve/claude_evolve/prompt/context_builder.py:525-732` (evolution history formatting)
- Modify: `claude_evolve/claude_evolve/cli.py:209-370` (next — load evaluator source)
- Test: `claude_evolve/tests/test_context_builder.py`

- [ ] **Step 1: Write failing tests**

```python
def test_top_programs_sorted_ascending_in_context():
    """Top programs should be rendered worst-first, best-last in context."""
    # Build context with 3 programs of known scores
    # Assert the rendered text has lowest score first, highest last

def test_evaluator_source_included_in_context():
    """When evaluator_source is provided, it should appear in the context."""
    # Build context with evaluator_source="def evaluate(x): return x"
    # Assert "def evaluate" appears in output
```

- [ ] **Step 2: Run to verify fails**

- [ ] **Step 3: Implement ascending sort**

In `context_builder.py` `_format_evolution_history()` (~line 525), where top programs are rendered, reverse the list before formatting:
```python
# Reverse to ascending order (worst first, best last) per OPRO
top_programs_ascending = list(reversed(top_programs))
```

Add score annotation to each program:
```python
# Prefix each program snippet with score
header = f"# Score: {prog.metrics.get('combined_score', 0.0):.6f}"
```

- [ ] **Step 4: Add evaluator_source parameter to context_builder**

In `build_context()` (~line 70), add parameter:
```python
evaluator_source: Optional[str] = None,
```

In the rendering logic, add section:
```python
if evaluator_source:
    sections.append(f"## Evaluator Source (How Candidates Are Scored)\n```python\n{evaluator_source}\n```")
```

In `cli.py` next command, load and pass:
```python
evaluator_path = _find_evaluator_path(state_dir)
evaluator_source = None
if evaluator_path and os.path.exists(evaluator_path):
    with open(evaluator_path, encoding='utf-8') as f:
        lines = f.readlines()[:200]
        evaluator_source = "".join(lines)
```

- [ ] **Step 5: Run tests**

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add claude_evolve/claude_evolve/prompt/context_builder.py claude_evolve/claude_evolve/cli.py claude_evolve/tests/test_context_builder.py
git commit -m "feat: ascending score sort (OPRO) + evaluator source as context (Eureka)"
```

---

### Task 12: Pairwise Comparison (QW3)

**Files:**
- Modify: `claude_evolve/claude_evolve/prompt/context_builder.py:70-265`
- Test: `claude_evolve/tests/test_context_builder.py`

- [ ] **Step 1: Write failing test**

```python
def test_comparison_artifact_in_context():
    """When comparison_artifact is provided, both parent and comparison rendered."""
    # Build context with parent (score 0.8) and comparison (score 0.5)
    # Assert both appear in output with scores
```

- [ ] **Step 2: Run to verify fails**

- [ ] **Step 3: Add comparison_artifact parameter to build_context**

In `build_context()` (~line 70), add:
```python
comparison_artifact: Optional[Artifact] = None,
comparison_score: float = 0.0,
```

In rendering:
```python
if comparison_artifact:
    parent_score = parent.metrics.get('combined_score', 0.0) if parent else 0.0
    sections.append(
        f"## Pairwise Comparison (Verbal Gradient)\n"
        f"The parent (score {parent_score:.4f}) and this comparison "
        f"(score {comparison_score:.4f}) differ in their approach. "
        f"Consider what makes the higher-scoring one better.\n\n"
        f"### Comparison Program\n```\n{comparison_artifact.content[:500]}\n```"
    )
```

- [ ] **Step 4: Run tests**

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add claude_evolve/claude_evolve/prompt/context_builder.py claude_evolve/tests/test_context_builder.py
git commit -m "feat: add pairwise comparison for verbal gradients (ReEvo)"
```

---

### Task 13: Pre-Evaluation Novelty Gate (QW6)

**Files:**
- Modify: `claude_evolve/claude_evolve/cli.py:737-873` (submit command)
- Test: `claude_evolve/tests/test_cli.py`

- [ ] **Step 1: Write failing test**

```python
def test_novelty_gate_rejects_similar_candidate():
    """Candidate with novelty < 0.05 vs archive should be rejected before eval."""
    # Setup: archive with one artifact, candidate nearly identical
    # Assert: submit returns rejection message, evaluator NOT called

def test_novelty_gate_passes_novel_candidate():
    """Candidate with novelty >= 0.05 should pass through to evaluation."""

def test_novelty_gate_skips_with_small_archive():
    """Gate should skip when archive has < 3 members."""
```

- [ ] **Step 2: Run to verify fails**

- [ ] **Step 3: Implement novelty gate in submit**

In `cli.py` submit command, before running evaluator:
```python
from claude_evolve.core.novelty import compute_novelty

# Pre-evaluation novelty gate
min_novelty = config.selection.novelty_gate_min_novelty
top_artifacts = db.get_top_programs(n=10)
if len(top_artifacts) >= 3:
    with open(candidate_path, encoding='utf-8') as f:
        candidate_content = f.read()
    for archive_art in top_artifacts:
        novelty = compute_novelty(candidate_content, archive_art.content,
                                  artifact_type=artifact_type)
        if novelty < min_novelty:
            click.echo(json.dumps({
                "rejected": True,
                "reason": f"Candidate too similar to existing solution {archive_art.id} "
                          f"(novelty: {novelty:.3f}, threshold: {min_novelty}). "
                          f"Try a more novel approach.",
            }))
            return
```

- [ ] **Step 4: Run tests**

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add claude_evolve/claude_evolve/cli.py claude_evolve/tests/test_cli.py
git commit -m "feat: add pre-evaluation novelty gate to reject near-duplicates (ShinkaEvolve)"
```

---

## Chunk 3: Phase 3 — Critical Gap Closers (Tasks 14-19)

### Task 14: Improvement Signal G_t (CG1)

**Files:**
- Create: `claude_evolve/claude_evolve/core/improvement_signal.py`
- Create: `claude_evolve/tests/test_improvement_signal.py`

- [ ] **Step 1: Write failing tests**

```python
import os, tempfile
from claude_evolve.core.improvement_signal import ImprovementSignal

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
    from claude_evolve.core.stagnation import StagnationLevel
    sig = ImprovementSignal()
    sig.g_t = 0.2
    assert sig.derive_stagnation_level() == StagnationLevel.NONE
    sig.g_t = 0.001
    assert sig.derive_stagnation_level() == StagnationLevel.CRITICAL

def test_save_and_load_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'signal.json')
        sig = ImprovementSignal(g_t=0.42, rho=0.9)
        sig.per_island_g_t = {0: 0.3, 1: 0.1}
        sig.save(path)
        loaded = ImprovementSignal.load(path)
        assert abs(loaded.g_t - 0.42) < 1e-10
        assert loaded.per_island_g_t == {0: 0.3, 1: 0.1}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd claude_evolve && python -m pytest tests/test_improvement_signal.py -v`
Expected: FAIL — module doesn't exist

- [ ] **Step 3: Implement ImprovementSignal**

Create `claude_evolve/claude_evolve/core/improvement_signal.py`:

```python
"""Continuous improvement signal G_t (inspired by AdaEvolve).

Replaces discrete stagnation levels with a continuous exponential moving average
of improvement magnitude. Drives exploration intensity, strategy selection
modulation, and meta-guidance triggering from a single unified signal.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Optional

from claude_evolve.core.stagnation import StagnationLevel


@dataclass
class ImprovementSignal:
    g_t: float = 0.0
    rho: float = 0.95
    i_min: float = 0.1
    i_max: float = 0.7
    meta_threshold: float = 0.12
    per_island_g_t: dict = field(default_factory=dict)

    def update(self, child_score: float, parent_score: float, island_id: int) -> None:
        """Update G_t after an evaluation.

        delta = max((child - parent) / max(parent, 1e-10), 0)
        g_t = rho * g_t + (1 - rho) * delta
        """
        delta = max((child_score - parent_score) / max(parent_score, 1e-10), 0.0)
        self.g_t = self.rho * self.g_t + (1 - self.rho) * delta

        # Per-island signal
        island_key = str(island_id)
        prev = self.per_island_g_t.get(island_key, 0.0)
        self.per_island_g_t[island_key] = self.rho * prev + (1 - self.rho) * delta

    @property
    def exploration_intensity(self) -> float:
        """Exploration intensity derived from G_t.

        I_t = I_min + (I_max - I_min) / (1 + G_t / 0.05)
        """
        return self.i_min + (self.i_max - self.i_min) / (1.0 + self.g_t / 0.05)

    def should_trigger_meta_guidance(self) -> bool:
        """True when ALL islands have g_t <= meta_threshold."""
        if not self.per_island_g_t:
            return self.g_t <= self.meta_threshold
        return all(v <= self.meta_threshold for v in self.per_island_g_t.values())

    def derive_stagnation_level(self) -> StagnationLevel:
        """Backward-compatible stagnation level from G_t."""
        if self.g_t > 0.1:
            return StagnationLevel.NONE
        elif self.g_t > 0.05:
            return StagnationLevel.MILD
        elif self.g_t > 0.02:
            return StagnationLevel.MODERATE
        elif self.g_t > 0.005:
            return StagnationLevel.SEVERE
        else:
            return StagnationLevel.CRITICAL

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                'g_t': self.g_t,
                'rho': self.rho,
                'i_min': self.i_min,
                'i_max': self.i_max,
                'meta_threshold': self.meta_threshold,
                'per_island_g_t': self.per_island_g_t,
            }, f)

    @classmethod
    def load(cls, path: str) -> 'ImprovementSignal':
        if not os.path.exists(path):
            return cls()
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        sig = cls(
            g_t=data.get('g_t', 0.0),
            rho=data.get('rho', 0.95),
            i_min=data.get('i_min', 0.1),
            i_max=data.get('i_max', 0.7),
            meta_threshold=data.get('meta_threshold', 0.12),
        )
        sig.per_island_g_t = data.get('per_island_g_t', {})
        return sig
```

- [ ] **Step 4: Run tests**

Run: `cd claude_evolve && python -m pytest tests/test_improvement_signal.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add claude_evolve/claude_evolve/core/improvement_signal.py claude_evolve/tests/test_improvement_signal.py
git commit -m "feat: add continuous improvement signal G_t (AdaEvolve)"
```

---

### Task 15: UCB1 Strategy Selector (CG4)

**Files:**
- Create: `claude_evolve/claude_evolve/core/ucb_selector.py`
- Create: `claude_evolve/tests/test_ucb_selector.py`

- [ ] **Step 1: Write failing tests**

```python
import os, math, tempfile
from claude_evolve.core.ucb_selector import UCBStrategySelector

def test_unvisited_arms_selected_first():
    sel = UCBStrategySelector(["a", "b", "c"])
    first = sel.select()
    assert first in ["a", "b", "c"]
    sel.record(first, 0.1)
    second = sel.select()
    assert second != first  # Should pick unvisited

def test_high_reward_arm_preferred_after_exploration():
    sel = UCBStrategySelector(["a", "b", "c"])
    # Visit all arms
    for sid in ["a", "b", "c"]:
        sel.select()
        sel.record(sid, 0.0)
    # Give "b" a big reward
    sel.record("b", 0.5)
    # After enough visits, b should be preferred
    counts = {"a": 0, "b": 0, "c": 0}
    for _ in range(100):
        s = sel.select(exploration_intensity=0.1)  # Low exploration
        counts[s] += 1
        sel.record(s, 0.0)
    assert counts["b"] > counts["a"]

def test_reward_is_capped_at_one():
    sel = UCBStrategySelector(["a"])
    sel.record("a", 5.0)  # Large delta
    assert sel.arms["a"].total_reward <= 1.0

def test_decay_reduces_old_rewards():
    sel = UCBStrategySelector(["a"], decay=0.5)
    sel.record("a", 0.8)
    r1 = sel.arms["a"].decayed_reward
    sel.record("a", 0.0)  # No new reward, but decay applied
    r2 = sel.arms["a"].decayed_reward
    assert r2 < r1

def test_save_and_load_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'ucb.json')
        sel = UCBStrategySelector(["a", "b"])
        sel.record("a", 0.3)
        sel.save(path)
        loaded = UCBStrategySelector.load(path)
        assert loaded.arms["a"].visit_count == 1
        assert loaded.arms["a"].total_reward > 0
```

- [ ] **Step 2: Run to verify they fail**

- [ ] **Step 3: Implement UCBStrategySelector**

Create `claude_evolve/claude_evolve/core/ucb_selector.py`:

```python
"""UCB1 bandit for adaptive strategy selection.

Inspired by ShinkaEvolve (UCB1 with exponential reward) and AdaEvolve
(exploration modulation). Uses capped linear reward to prevent single
lucky outcomes from dominating.
"""

import json
import math
import os
from dataclasses import dataclass, field


@dataclass
class StrategyArm:
    strategy_id: str
    total_reward: float = 0.0
    visit_count: int = 0
    decayed_reward: float = 0.0


class UCBStrategySelector:
    """UCB1-based strategy selection with exploration modulation."""

    def __init__(self, strategy_ids: list, c: float = 1.414, decay: float = 0.95):
        self.arms: dict[str, StrategyArm] = {
            sid: StrategyArm(strategy_id=sid) for sid in strategy_ids
        }
        self.c = c
        self.decay = decay
        self.total_selections = 0

    def select(self, exploration_intensity: float = 0.5) -> str:
        """Select strategy using UCB1 with exploration modulation."""
        self.total_selections += 1

        # Always select unvisited arms first
        unvisited = [a for a in self.arms.values() if a.visit_count == 0]
        if unvisited:
            return unvisited[0].strategy_id

        c_adj = self.c * (0.5 + exploration_intensity)
        n = self.total_selections

        best_score = -float('inf')
        best_id = list(self.arms.keys())[0]

        for arm in self.arms.values():
            if arm.visit_count == 0:
                return arm.strategy_id
            avg_reward = arm.decayed_reward / arm.visit_count
            ucb_bonus = c_adj * math.sqrt(math.log(n) / arm.visit_count)
            ucb_score = avg_reward + ucb_bonus
            if ucb_score > best_score:
                best_score = ucb_score
                best_id = arm.strategy_id

        return best_id

    def record(self, strategy_id: str, score_delta: float) -> None:
        """Record outcome with capped linear reward and global decay."""
        reward = min(max(score_delta, 0.0), 1.0)

        # Decay all arms
        for arm in self.arms.values():
            arm.decayed_reward *= self.decay

        # Update selected arm
        if strategy_id in self.arms:
            arm = self.arms[strategy_id]
            arm.total_reward = min(arm.total_reward + reward, arm.visit_count + 1)
            arm.visit_count += 1
            arm.decayed_reward += reward

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        data = {
            'c': self.c, 'decay': self.decay,
            'total_selections': self.total_selections,
            'arms': {
                sid: {'total_reward': a.total_reward, 'visit_count': a.visit_count,
                      'decayed_reward': a.decayed_reward}
                for sid, a in self.arms.items()
            },
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: str) -> 'UCBStrategySelector':
        if not os.path.exists(path):
            return cls([])
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        sel = cls(list(data.get('arms', {}).keys()),
                  c=data.get('c', 1.414), decay=data.get('decay', 0.95))
        sel.total_selections = data.get('total_selections', 0)
        for sid, arm_data in data.get('arms', {}).items():
            if sid in sel.arms:
                sel.arms[sid].total_reward = arm_data.get('total_reward', 0.0)
                sel.arms[sid].visit_count = arm_data.get('visit_count', 0)
                sel.arms[sid].decayed_reward = arm_data.get('decayed_reward', 0.0)
        return sel
```

- [ ] **Step 4: Run tests**

Run: `cd claude_evolve && python -m pytest tests/test_ucb_selector.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add claude_evolve/claude_evolve/core/ucb_selector.py claude_evolve/tests/test_ucb_selector.py
git commit -m "feat: add UCB1 strategy selector with capped reward (ShinkaEvolve + AdaEvolve)"
```

---

### Task 16: Reflection Engine (CG3)

**Files:**
- Create: `claude_evolve/claude_evolve/core/reflection.py`
- Create: `claude_evolve/tests/test_reflection.py`

- [ ] **Step 1: Write failing tests**

```python
import os, tempfile
from claude_evolve.core.reflection import ReflectionEngine, ShortReflection
from claude_evolve.core.artifact import Artifact

def test_short_reflection_identifies_added_concepts():
    eng = ReflectionEngine.__new__(ReflectionEngine)
    eng.short_reflections = []
    eng.long_reflection = ""
    better = Artifact(content="import numpy\ndef solve(): return simulated_annealing()", artifact_type="python")
    better.metrics = {"combined_score": 0.8}
    worse = Artifact(content="def solve(): return random_search()", artifact_type="python")
    worse.metrics = {"combined_score": 0.3}
    ref = eng.generate_short_reflection(better, worse)
    assert "numpy" in ref.insight or "simulated_annealing" in ref.insight

def test_long_reflection_synthesizes_short():
    with tempfile.TemporaryDirectory() as d:
        eng = ReflectionEngine(d, synthesis_interval=2)
        # Add 3 short reflections
        eng.short_reflections = [
            ShortReflection("a", "b", 0.8, 0.3, "Better adds: numpy, simulated_annealing", 1),
            ShortReflection("c", "d", 0.9, 0.4, "Better adds: numpy, gradient", 2),
            ShortReflection("e", "f", 0.7, 0.2, "Better adds: scipy, linear_programming", 3),
        ]
        result = eng.accumulate_long_reflection(4)  # Multiple of 2
        assert result is not None
        assert "numpy" in result  # Most common concept

def test_format_for_prompt():
    with tempfile.TemporaryDirectory() as d:
        eng = ReflectionEngine(d)
        eng.short_reflections = [
            ShortReflection("a", "b", 0.8, 0.3, "Better adds: numpy", 1),
        ]
        eng.long_reflection = "Key insight: use numpy"
        text = eng.format_for_prompt()
        assert "numpy" in text
        assert "Key insight" in text

def test_save_and_load():
    with tempfile.TemporaryDirectory() as d:
        eng = ReflectionEngine(d)
        eng.short_reflections = [
            ShortReflection("a", "b", 0.8, 0.3, "test insight", 1),
        ]
        eng.long_reflection = "accumulated wisdom"
        eng.save(os.path.join(d, 'reflections.json'))
        loaded = ReflectionEngine(d)
        loaded_data = ReflectionEngine.load_from(os.path.join(d, 'reflections.json'))
        assert len(loaded_data.short_reflections) == 1
```

- [ ] **Step 2: Run to verify they fail**

- [ ] **Step 3: Implement ReflectionEngine**

Create `claude_evolve/claude_evolve/core/reflection.py`:

```python
"""Pairwise reflection / verbal gradients engine (inspired by ReEvo, NeurIPS 2024).

Generates short-term reflections by comparing artifact pairs using semantic
fingerprint set differences. Accumulates into long-term reflections via
concept frequency counting. Fully deterministic — no LLM calls.
"""

import json
import os
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

from claude_evolve.core.artifact import Artifact
from claude_evolve.core.novelty import semantic_fingerprint


@dataclass
class ShortReflection:
    better_id: str
    worse_id: str
    better_score: float
    worse_score: float
    insight: str
    iteration: int


class ReflectionEngine:
    """Generates and accumulates pairwise reflections."""

    def __init__(self, state_dir: str, max_short: int = 20, synthesis_interval: int = 5):
        self.state_dir = state_dir
        self.max_short = max_short
        self.synthesis_interval = synthesis_interval
        self.short_reflections: list[ShortReflection] = []
        self.long_reflection: str = ""

    def generate_short_reflection(self, better: Artifact, worse: Artifact) -> ShortReflection:
        """Compare two artifacts using semantic fingerprint differences."""
        better_fp = semantic_fingerprint(better.content)
        worse_fp = semantic_fingerprint(worse.content)
        added = sorted(better_fp - worse_fp)[:3]
        removed = sorted(worse_fp - better_fp)[:3]

        parts = []
        if added:
            parts.append(f"Better adds: {', '.join(added)}")
        if removed:
            parts.append(f"Drops: {', '.join(removed)}")
        if not parts:
            # Fallback: line count comparison
            b_lines = better.content.count('\n')
            w_lines = worse.content.count('\n')
            parts.append(f"Better has {b_lines} lines vs {w_lines}")

        insight = ". ".join(parts)
        better_score = better.metrics.get('combined_score', 0.0) if hasattr(better, 'metrics') and better.metrics else 0.0
        worse_score = worse.metrics.get('combined_score', 0.0) if hasattr(worse, 'metrics') and worse.metrics else 0.0

        ref = ShortReflection(
            better_id=better.id, worse_id=worse.id,
            better_score=better_score, worse_score=worse_score,
            insight=insight, iteration=0,
        )
        self.short_reflections.append(ref)
        self.short_reflections = self.short_reflections[-self.max_short:]
        return ref

    def accumulate_long_reflection(self, current_iteration: int) -> Optional[str]:
        """Every synthesis_interval iterations, compress short reflections."""
        if current_iteration <= 0 or current_iteration % self.synthesis_interval != 0:
            return None
        if not self.short_reflections:
            return None

        # Count concept occurrences across all insights
        concept_counter: Counter = Counter()
        for ref in self.short_reflections:
            for word in ref.insight.split():
                word = word.strip(".,:")
                if len(word) > 3 and word.isalpha():
                    concept_counter[word] += 1

        top_concepts = [c for c, _ in concept_counter.most_common(5)]
        if top_concepts:
            self.long_reflection = f"Key patterns: {', '.join(top_concepts)}"
        return self.long_reflection

    def format_for_prompt(self) -> str:
        """Format for injection into iteration context."""
        parts = []
        if self.short_reflections:
            latest = self.short_reflections[-1]
            parts.append(f"## Verbal Gradient\n{latest.insight}")
        if self.long_reflection:
            parts.append(f"## Accumulated Wisdom\n{self.long_reflection}")
        return "\n\n".join(parts)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        data = {
            'short_reflections': [
                {'better_id': r.better_id, 'worse_id': r.worse_id,
                 'better_score': r.better_score, 'worse_score': r.worse_score,
                 'insight': r.insight, 'iteration': r.iteration}
                for r in self.short_reflections
            ],
            'long_reflection': self.long_reflection,
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f)

    @classmethod
    def load_from(cls, path: str) -> 'ReflectionEngine':
        eng = cls(os.path.dirname(path))
        if os.path.exists(path):
            with open(path, encoding='utf-8') as f:
                data = json.load(f)
            eng.short_reflections = [
                ShortReflection(**r) for r in data.get('short_reflections', [])
            ]
            eng.long_reflection = data.get('long_reflection', '')
        return eng
```

- [ ] **Step 4: Run tests**

Run: `cd claude_evolve && python -m pytest tests/test_reflection.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add claude_evolve/claude_evolve/core/reflection.py claude_evolve/tests/test_reflection.py
git commit -m "feat: add reflection engine with verbal gradients (ReEvo)"
```

---

### Task 17: Thought-Code Coevolution (CG2)

**Files:**
- Modify: `claude_evolve/claude_evolve/core/artifact.py:13-77`
- Modify: `claude_evolve/claude_evolve/prompt/templates.py`
- Modify: `claude_evolve/claude_evolve/prompt/context_builder.py`
- Test: `claude_evolve/tests/test_artifact.py`, `claude_evolve/tests/test_context_builder.py`

- [ ] **Step 1: Write failing tests**

```python
from claude_evolve.core.artifact import Artifact

def test_artifact_has_rationale_and_offspring():
    a = Artifact(content="test", artifact_type="python",
                 rationale="Use SA because local optima", offspring_count=3)
    assert a.rationale == "Use SA because local optima"
    assert a.offspring_count == 3

def test_artifact_default_rationale_is_none():
    a = Artifact(content="test", artifact_type="python")
    assert a.rationale is None
    assert a.offspring_count == 0

def test_artifact_roundtrip_with_rationale():
    a = Artifact(content="test", artifact_type="python", rationale="my reason")
    d = a.to_dict()
    loaded = Artifact.from_dict(d)
    assert loaded.rationale == "my reason"

def test_extract_rationale_from_content():
    content = '''RATIONALE-START
Use simulated annealing with adaptive cooling.
RATIONALE-END
def solve(): pass'''
    # Test extraction function
    import re
    match = re.search(r'RATIONALE-START\s*\n(.*?)\nRATIONALE-END', content, re.DOTALL)
    assert match
    assert "simulated annealing" in match.group(1)
```

- [ ] **Step 2: Run to verify they fail**

- [ ] **Step 3: Add fields to Artifact**

In `artifact.py` Artifact dataclass (~line 13), add:
```python
rationale: Optional[str] = None
offspring_count: int = 0
```

Ensure `to_dict()` and `from_dict()` handle the new fields (they should via `asdict()` and field filtering).

- [ ] **Step 4: Add RATIONALE_SECTION template**

In `templates.py`, add:
```python
RATIONALE_SECTION = """## Current Approach Rationale
{rationale}

## Instructions
1. First, write an updated rationale (2-3 sentences) between RATIONALE-START and RATIONALE-END markers
2. Then write the code implementing that approach
"""
```

- [ ] **Step 5: Add rationale to context_builder**

In `build_context()`, add `parent_rationale` parameter. Include rationale section when available.

- [ ] **Step 6: Run tests**

Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add claude_evolve/claude_evolve/core/artifact.py claude_evolve/claude_evolve/prompt/templates.py claude_evolve/claude_evolve/prompt/context_builder.py claude_evolve/tests/
git commit -m "feat: add thought-code coevolution with rationale markers (EoH)"
```

---

### Task 18: Power-Law Parent Selection (CG5)

**Files:**
- Modify: `claude_evolve/claude_evolve/core/database.py`
- Test: `claude_evolve/tests/test_database.py`

- [ ] **Step 1: Write failing tests**

```python
import numpy as np
from claude_evolve.core.database import ArtifactDatabase
from claude_evolve.core.artifact import Artifact
from claude_evolve.config import DatabaseConfig

def test_power_law_high_alpha_favors_best():
    """With high alpha (exploitation), top-scoring artifact selected most."""
    config = DatabaseConfig()
    db = ArtifactDatabase(config)
    for i in range(10):
        a = Artifact(content=f"prog_{i}", artifact_type="python")
        db.add(a, {"combined_score": 0.1 * i})
    counts = {}
    for _ in range(200):
        parent = db.select_parent_power_law(island_id=0, exploration_intensity=0.1)
        if parent:
            score = parent.metrics.get('combined_score', 0)
            bucket = round(score, 1)
            counts[bucket] = counts.get(bucket, 0) + 1
    # Highest score bucket should have most selections
    assert max(counts, key=counts.get) >= 0.7

def test_power_law_low_alpha_more_uniform():
    """With low alpha (exploration), selection should be more spread out."""
    config = DatabaseConfig()
    db = ArtifactDatabase(config)
    for i in range(10):
        a = Artifact(content=f"prog_{i}", artifact_type="python")
        db.add(a, {"combined_score": 0.1 * i})
    counts = {}
    for _ in range(200):
        parent = db.select_parent_power_law(island_id=0, exploration_intensity=0.7)
        if parent:
            score = parent.metrics.get('combined_score', 0)
            bucket = round(score, 1)
            counts[bucket] = counts.get(bucket, 0) + 1
    # Should have selections across multiple buckets
    assert len(counts) >= 4

def test_offspring_count_affects_selection():
    """Artifacts with many offspring should be selected less often."""
    config = DatabaseConfig()
    db = ArtifactDatabase(config)
    a1 = Artifact(content="over_exploited", artifact_type="python", offspring_count=50)
    a2 = Artifact(content="fresh_artifact", artifact_type="python", offspring_count=0)
    db.add(a1, {"combined_score": 0.9})
    db.add(a2, {"combined_score": 0.9})
    # With equal scores, fresh artifact should be selected more
    counts = {"a1": 0, "a2": 0}
    for _ in range(100):
        p = db.select_parent_power_law(0, 0.5)
        if p and p.content == "over_exploited":
            counts["a1"] += 1
        elif p:
            counts["a2"] += 1
    assert counts["a2"] > counts["a1"]
```

- [ ] **Step 2: Run to verify they fail**

- [ ] **Step 3: Implement select_parent_power_law**

In `database.py`, add method to `ArtifactDatabase`:

```python
def select_parent_power_law(self, island_id: int, exploration_intensity: float = 0.5,
                            i_max: float = 0.7) -> Optional[Artifact]:
    """Rank-based power-law parent selection with offspring novelty weighting."""
    with self._lock:
        if island_id not in self.islands or not self.islands[island_id]:
            return None
        valid_ids = [pid for pid in self.islands[island_id] if pid in self.artifacts]
        if not valid_ids:
            return None

        # Sort by fitness descending
        sorted_ids = sorted(valid_ids,
            key=lambda pid: self.artifacts[pid].metrics.get('combined_score', 0.0),
            reverse=True)

        # Compute alpha from exploration intensity
        normalized = min(exploration_intensity / max(i_max, 0.01), 1.0)
        alpha = 0.3 + 3.0 * (1.0 - normalized)

        # Power-law fitness weights
        n = len(sorted_ids)
        ranks = list(range(1, n + 1))
        fitness_weights = [r ** (-alpha) for r in ranks]

        # Offspring novelty weights
        novelty_weights = []
        for pid in sorted_ids:
            oc = getattr(self.artifacts[pid], 'offspring_count', 0)
            novelty_weights.append(1.0 / (1.0 + oc))

        # Combined weights
        combined = [fw * nw for fw, nw in zip(fitness_weights, novelty_weights)]
        total = sum(combined)
        if total <= 0:
            return self.artifacts[sorted_ids[0]]
        probs = [w / total for w in combined]

        # Weighted random selection
        import random
        chosen_id = random.choices(sorted_ids, weights=probs, k=1)[0]
        return self.artifacts[chosen_id]
```

- [ ] **Step 4: Run tests**

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add claude_evolve/claude_evolve/core/database.py claude_evolve/tests/test_database.py
git commit -m "feat: add power-law parent selection with offspring weighting (ShinkaEvolve)"
```

---

### Task 19: IterationOrchestrator (Integration)

**Files:**
- Create: `claude_evolve/claude_evolve/core/orchestrator.py`
- Create: `claude_evolve/tests/test_orchestrator.py`
- Modify: `claude_evolve/claude_evolve/cli.py:209-370,737-873`

This is the final integration task. The orchestrator wires all modules together and the CLI delegates to it.

- [ ] **Step 1: Write failing integration tests**

```python
import os, json, tempfile
from claude_evolve.core.orchestrator import IterationOrchestrator
from claude_evolve.core.artifact import Artifact
from claude_evolve.config import Config

def test_prepare_next_iteration_returns_context():
    """Orchestrator should produce a complete context dict for the builder."""
    with tempfile.TemporaryDirectory() as d:
        config = Config()
        orch = IterationOrchestrator(state_dir=d, config=config)
        # Add a seed artifact
        orch.db.add(Artifact(content="def solve(): pass", artifact_type="python"),
                     {"combined_score": 0.5})
        ctx = orch.prepare_next_iteration(iteration=1)
        assert 'parent' in ctx
        assert 'strategy_name' in ctx
        assert 'exploration_intensity' in ctx

def test_process_submission_updates_signal():
    """Orchestrator should update G_t signal after submission."""
    with tempfile.TemporaryDirectory() as d:
        config = Config()
        orch = IterationOrchestrator(state_dir=d, config=config)
        # Setup manifest
        orch._save_manifest({"parent_score": 0.5, "selected_strategy_id": "default-incremental",
                            "parent_island_id": 0, "iteration": 1, "run_id": "test"})
        result = orch.process_submission("def solve(): return 1", {"combined_score": 0.8})
        assert orch.signal.g_t > 0.0  # Improvement recorded
```

- [ ] **Step 2: Run to verify they fail**

- [ ] **Step 3: Implement IterationOrchestrator**

Create `claude_evolve/claude_evolve/core/orchestrator.py`:

```python
"""IterationOrchestrator: wires all feature modules for next/submit lifecycle.

Coordinates: ImprovementSignal, UCBStrategySelector, MetaScratchpad,
ReflectionEngine, CrossRunMemory, StrategyManager, ArtifactDatabase.
"""

import json
import os
from typing import Optional

from claude_evolve.config import Config
from claude_evolve.core.artifact import Artifact
from claude_evolve.core.database import ArtifactDatabase
from claude_evolve.core.improvement_signal import ImprovementSignal
from claude_evolve.core.memory import CrossRunMemory
from claude_evolve.core.novelty import compute_novelty
from claude_evolve.core.reflection import ReflectionEngine
from claude_evolve.core.scratchpad import MetaScratchpad
from claude_evolve.core.strategy import StrategyManager
from claude_evolve.core.ucb_selector import UCBStrategySelector


class IterationOrchestrator:
    """Coordinates all feature modules for the evolution lifecycle."""

    def __init__(self, state_dir: str, config: Config, db: Optional[ArtifactDatabase] = None):
        self.state_dir = state_dir
        self.config = config
        self.db = db or ArtifactDatabase(config.database)

        # Load all modules
        self.signal = ImprovementSignal.load(
            os.path.join(state_dir, 'improvement_signal.json'))
        self.scratchpad = MetaScratchpad(
            state_dir, config.scratchpad.synthesis_interval)
        self.reflection = ReflectionEngine.load_from(
            os.path.join(state_dir, 'reflections.json'))

        # UCB selector - initialize with strategy IDs
        ucb_path = os.path.join(state_dir, 'strategy_bandit.json')
        if os.path.exists(ucb_path):
            self.ucb = UCBStrategySelector.load(ucb_path)
        else:
            strategy_mgr = StrategyManager()
            strategy_path = os.path.join(state_dir, 'strategies.json')
            if os.path.exists(strategy_path):
                strategy_mgr.load(strategy_path)
            self.ucb = UCBStrategySelector(
                [s.id for s in strategy_mgr.strategies],
                c=config.selection.ucb_c,
                decay=config.selection.ucb_decay)

        self.memory = CrossRunMemory(
            os.path.join(state_dir, 'cross_run_memory.json'))
        self.memory.load()

    def prepare_next_iteration(self, iteration: int) -> dict:
        """Run all next-phase logic. Returns context dict for the builder."""
        ei = self.signal.exploration_intensity
        stag_level = self.signal.derive_stagnation_level()

        # Strategy selection
        if self.config.selection.strategy_selection == "ucb1":
            strategy_id = self.ucb.select(ei)
        else:
            strategy_id = "default-incremental"

        # Parent selection
        island_id = iteration % max(len(self.db.islands), 1)
        if self.config.selection.parent_selection == "power_law":
            parent = self.db.select_parent_power_law(island_id, ei, self.config.improvement_signal.i_max)
        else:
            result = self.db.sample()
            parent = result.get('parent') if result else None

        # Comparison artifact for pairwise reflection
        comparison = None
        if parent:
            top = self.db.get_top_programs(n=5)
            parent_score = parent.metrics.get('combined_score', 0) if parent.metrics else 0
            candidates = [a for a in top if a.id != parent.id]
            if candidates:
                candidates.sort(key=lambda a: abs(
                    a.metrics.get('combined_score', 0) - parent_score), reverse=True)
                comparison = candidates[0]

        # Generate short reflection
        if parent and comparison:
            self.reflection.generate_short_reflection(
                parent if (parent.metrics or {}).get('combined_score', 0) >= (comparison.metrics or {}).get('combined_score', 0) else comparison,
                comparison if (parent.metrics or {}).get('combined_score', 0) >= (comparison.metrics or {}).get('combined_score', 0) else parent,
            )
        self.reflection.accumulate_long_reflection(iteration)

        # Meta-scratchpad
        scratchpad_text = self.scratchpad.load() or ""
        if self.scratchpad.should_synthesize(iteration):
            top_arts = self.db.get_top_programs(n=5)
            scores = [a.metrics.get('combined_score', 0) for a in top_arts] if top_arts else []
            failed = []
            failures_path = os.path.join(self.state_dir, 'recent_failures.json')
            if os.path.exists(failures_path):
                with open(failures_path, encoding='utf-8') as f:
                    failed = [e.get('approach', '') for e in json.load(f)]
            scratchpad_text = self.scratchpad.synthesize(top_arts, scores, failed)
            self.scratchpad.save(scratchpad_text)

        # Recent failures
        failures_text = ""
        failures_path = os.path.join(self.state_dir, 'recent_failures.json')
        if os.path.exists(failures_path):
            with open(failures_path, encoding='utf-8') as f:
                failures = json.load(f)
            if failures:
                lines = ["## Recent Failures (Avoid These)"]
                for fe in failures:
                    err = fe.get('error') or f"score dropped to {fe.get('score', '?')}"
                    lines.append(f"- Approach: {fe.get('approach', '?')}. Result: {err}")
                failures_text = "\n".join(lines)

        # Meta-guidance check
        meta_guidance = ""
        if self.signal.should_trigger_meta_guidance():
            meta_guidance = (
                "## BREAKTHROUGH REQUIRED\n"
                "All search directions have stagnated. You MUST try a fundamentally "
                "different algorithmic approach. Analyze the evaluator source and current "
                "best to identify completely untried strategies. Do NOT make incremental changes."
            )

        # Save iteration manifest
        self._save_manifest({
            "iteration": iteration,
            "run_id": self._get_run_id(),
            "selected_strategy_id": strategy_id,
            "parent_artifact_id": parent.id if parent else None,
            "parent_score": (parent.metrics or {}).get('combined_score', 0.0) if parent else 0.0,
            "parent_island_id": island_id,
        })

        # Save module states
        self.signal.save(os.path.join(self.state_dir, 'improvement_signal.json'))
        self.ucb.save(os.path.join(self.state_dir, 'strategy_bandit.json'))
        self.reflection.save(os.path.join(self.state_dir, 'reflections.json'))

        return {
            'parent': parent,
            'comparison': comparison,
            'strategy_name': strategy_id,
            'exploration_intensity': ei,
            'stagnation_level': stag_level,
            'scratchpad_text': scratchpad_text,
            'failures_text': failures_text,
            'reflection_text': self.reflection.format_for_prompt(),
            'meta_guidance': meta_guidance,
        }

    def process_submission(self, candidate_content: str, metrics: dict) -> dict:
        """Run all submit-phase logic. Returns result dict."""
        manifest = self._load_manifest()
        score = metrics.get('combined_score', 0.0)
        parent_score = manifest.get('parent_score', 0.0)
        strategy_id = manifest.get('selected_strategy_id', 'unknown')
        island_id = manifest.get('parent_island_id', 0)
        run_id = manifest.get('run_id', 'unknown')
        iteration = manifest.get('iteration', 0)

        # Update G_t signal
        self.signal.update(score, parent_score, island_id)

        # Record UCB outcome
        self.ucb.record(strategy_id, score - parent_score)

        # Update cross-run memory
        if score <= parent_score:
            self.memory.add_failed_approach(
                description=f"Strategy '{strategy_id}' on iteration {iteration}",
                score=score, iteration=iteration, run_id=run_id)
        else:
            self.memory.add_strategy(
                name=strategy_id,
                description=f"Improved from {parent_score:.4f} to {score:.4f}",
                score=score, run_id=run_id)
        self.memory.save()

        # Capture failure for reflexion
        if score < parent_score * 0.95 or metrics.get('error'):
            self._capture_failure(strategy_id, metrics.get('error'), score, parent_score, iteration)

        # Increment parent offspring count
        parent_id = manifest.get('parent_artifact_id')
        if parent_id and parent_id in self.db.artifacts:
            self.db.artifacts[parent_id].offspring_count += 1

        # Save states
        self.signal.save(os.path.join(self.state_dir, 'improvement_signal.json'))
        self.ucb.save(os.path.join(self.state_dir, 'strategy_bandit.json'))

        return {'score': score, 'parent_score': parent_score, 'improved': score > parent_score}

    def _save_manifest(self, data: dict) -> None:
        path = os.path.join(self.state_dir, 'current_iteration.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f)

    def _load_manifest(self) -> dict:
        path = os.path.join(self.state_dir, 'current_iteration.json')
        if os.path.exists(path):
            with open(path, encoding='utf-8') as f:
                return json.load(f)
        return {}

    def _get_run_id(self) -> str:
        meta_path = os.path.join(self.state_dir, 'metadata.json')
        if os.path.exists(meta_path):
            with open(meta_path, encoding='utf-8') as f:
                return json.load(f).get('run_id', 'unknown')
        return 'unknown'

    def _capture_failure(self, strategy, error, score, parent_score, iteration):
        path = os.path.join(self.state_dir, 'recent_failures.json')
        failures = []
        if os.path.exists(path):
            with open(path, encoding='utf-8') as f:
                failures = json.load(f)
        failures.append({
            "approach": strategy, "error": error,
            "score": score, "parent_score": parent_score,
            "iteration": iteration,
        })
        failures = failures[-5:]
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(failures, f)
```

- [ ] **Step 4: Update cli.py to delegate to orchestrator**

In `cli.py` next command, replace the manual wiring with:
```python
from claude_evolve.core.orchestrator import IterationOrchestrator
orch = IterationOrchestrator(state_dir, config, db)
ctx = orch.prepare_next_iteration(iteration)
# Pass ctx values to context_builder.build_context(...)
```

In `cli.py` submit command, after evaluation:
```python
orch = IterationOrchestrator(state_dir, config, db)
orch.process_submission(candidate_content, metrics)
```

- [ ] **Step 5: Run all tests**

Run: `cd claude_evolve && python -m pytest tests/ -v`
Expected: All pass (848 existing + ~89 new)

- [ ] **Step 6: Commit**

```bash
git add claude_evolve/claude_evolve/core/orchestrator.py claude_evolve/tests/test_orchestrator.py claude_evolve/claude_evolve/cli.py
git commit -m "feat: add IterationOrchestrator integrating all Approach B features"
```

---

## Final Verification

- [ ] **Run full test suite**: `cd claude_evolve && python -m pytest tests/ -v --tb=short`
- [ ] **Verify test count**: Should be ~937 (848 existing + ~89 new)
- [ ] **Run a smoke test evolution**: `claude-evolve init --artifact circle_packing/program.py --evaluator circle_packing/evaluator.py && claude-evolve next --state-dir .claude/evolve-state`
- [ ] **Commit and push all changes**
