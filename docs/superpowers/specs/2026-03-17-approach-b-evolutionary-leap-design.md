# Approach B: Evolutionary Leap — Design Spec

**Date:** 2026-03-17
**Scope:** Fix production issues + implement 5 critical gaps + 6 quick wins from literature review
**Goal:** Make ClaudeEvolve competitive with AdaEvolve, ShinkaEvolve, LoongFlow, and ReEvo

---

## Overview

This spec covers 18 changes organized into three phases:
- **Phase 1** (7 items): Production bug fixes
- **Phase 2** (6 items): Quick wins from literature (1-2 days each)
- **Phase 3** (5 items): Critical gap closers (3-5 days each)

### Cross-Cutting: Iteration State Persistence

Several features require state to pass between `next` (which selects parent/strategy) and `submit` (which records outcomes). Since these are separate CLI invocations, we persist an **iteration manifest** at `evolve-state/current_iteration.json`:

```json
{
  "iteration": 12,
  "run_id": "a1b2c3",
  "selected_strategy_id": "default-research",
  "parent_artifact_id": "abc123",
  "parent_score": 0.78,
  "parent_island_id": 2,
  "comparison_artifact_id": "def456"
}
```

- `cli.py init`: Generate `run_id` (UUID4 truncated to 8 chars), persist in state metadata.
- `cli.py next`: Write iteration manifest after selecting parent/strategy.
- `cli.py submit`: Read iteration manifest to get parent_id, parent_score, strategy_id, island_id.

### Cross-Cutting: IterationOrchestrator

To manage the growing wiring complexity, extract an `IterationOrchestrator` class in a new file `claude_evolve/claude_evolve/core/orchestrator.py` (~150 lines):

```python
class IterationOrchestrator:
    """Coordinates all feature modules for next/submit lifecycle."""
    def __init__(self, state_dir, config, db, ...):
        self.signal = ImprovementSignal.load(...)
        self.ucb = UCBStrategySelector.load(...)
        self.scratchpad = MetaScratchpad(...)
        self.reflection = ReflectionEngine(...)
        # ...

    def prepare_next_iteration(self, iteration: int) -> dict:
        """Runs all next-phase logic. Returns context dict for builder."""

    def process_submission(self, candidate_content, metrics) -> dict:
        """Runs all submit-phase logic. Returns evaluation result."""
```

- `cli.py next` calls `orchestrator.prepare_next_iteration()`.
- `cli.py submit` calls `orchestrator.process_submission()`.
- Each module remains independently testable; orchestrator handles wiring only.

### Cross-Cutting: G_t Signal Authority

When `ImprovementSignalConfig.enabled == True`, the G_t signal is **authoritative** for:
- Stagnation level (via `signal.derive_stagnation_level()`)
- Exploration intensity
- Meta-guidance triggering
- Strategy selection modulation (CG4)
- Parent selection alpha (CG5)

The existing `StagnationEngine` iteration-count-based detection is the fallback ONLY when `improvement_signal.enabled == False`. They never run simultaneously.

All changes are additive and independently testable. Each phase can be landed and validated before starting the next.

---

## Phase 1: Production Fixes

### Fix 1: Cross-Run Memory Population

**File:** `claude_evolve/claude_evolve/cli.py` (submit command)
**Problem:** `memory.add_failed_approach()` and `memory.add_strategy()` exist but are never called.
**Fix:** After evaluation in submit, read iteration manifest (see Cross-Cutting) for `run_id`, `parent_score`, `selected_strategy_id`:
- If `score <= best_score`: call `memory.add_failed_approach(description=strategy_name, score=score, iteration=current_iteration, run_id=run_id)`
- If `score > best_score`: call `memory.add_strategy(name=strategy_name, description=approach_summary, score=score, run_id=run_id)`
- `approach_summary` = strategy name + "on iteration {N}"
- `run_id` read from `evolve-state/current_iteration.json` (persisted by `init`)
**Test:** Assert memory file grows after submissions. Assert `run_id` and `iteration` are populated.

### Fix 2: Strategy Outcome Recording

**File:** `claude_evolve/claude_evolve/cli.py` (submit command)
**Problem:** `strategy.record_outcome()` exists but never called. No feedback loop.
**Fix:** After evaluation, read `selected_strategy_id` and `parent_score` from iteration manifest (see Cross-Cutting). Call `strategy_mgr.record_outcome(strategy_id, score - parent_score)`.
**Test:** Assert strategy stats update after submission. Assert correct strategy_id is read from manifest.

### Fix 3: Research Trigger Logic

**File:** `claude_evolve/claude_evolve/cli.py` (next command)
**Problem:** `should_research()` defined but never called; research text always included.
**Fix:** Only include research text in context if `research_log.should_research(stagnation_level)` returns True.
**Test:** Assert research text absent when stagnation is NONE and trigger is "on_stagnation".

### Fix 4: Thread Safety

**File:** `claude_evolve/claude_evolve/core/database.py`
**Problem:** All data structures accessed without locks. Parallel evaluation corrupts state.
**Fix:** Add `threading.RLock` to `ArtifactDatabase.__init__()`. Wrap `add()`, `sample()`, `migrate()`, and `remove()` with `with self._lock:`. Use RLock to allow re-entrant calls.
**Test:** Concurrent add() calls from multiple threads produce correct artifact count.

### Fix 5: Island-Artifact Sync Guard

**File:** `claude_evolve/claude_evolve/core/database.py`
**Problem:** `self.islands[island_id]` can get out of sync with `self.artifacts` during `remove()` calls, leaving orphaned IDs in island sets.
**Fix:** In `remove()`, ensure the artifact ID is removed from all island sets. Add a defensive `if pid not in self.artifacts: continue` guard in `_sample_from_island_weighted()` and any remaining unguarded paths. Log warning for orphaned IDs detected.
**Test:** After remove(), island sets contain no orphaned IDs. Sampling with artificially injected orphaned IDs doesn't crash.

### Fix 6: Warm Cache Eviction

**File:** `claude_evolve/claude_evolve/core/warm_cache.py`
**Problem:** No eviction policy; cache grows indefinitely.
**Fix:** Add `max_items: int = 50` config. Track access times in metadata. On `put()`, if items exceed `max_items`, evict least-recently-accessed. Update `get()` to touch access time.
**Test:** Cache with max_items=3 evicts oldest after 4th put.

### Fix 7: Evaluator Error Logging

**File:** `claude_evolve/claude_evolve/core/evaluator.py`
**Problem:** Timeouts don't log which program failed; fixed 0.5s retry sleep.
**Fix:** Log program path and timeout duration on `asyncio.TimeoutError`. Add exponential backoff: 0.5s, 1s, 2s for retries.
**Test:** Timeout logs include program identifier.

---

## Phase 2: Quick Wins

### QW1: Meta-Scratchpad (ShinkaEvolve, ICLR 2026)

**New file:** `claude_evolve/claude_evolve/core/scratchpad.py` (~120 lines)

**Class: `MetaScratchpad`**
```python
class MetaScratchpad:
    def __init__(self, state_dir: str, synthesis_interval: int = 10):
        """synthesis_interval: generate new scratchpad every N iterations."""

    def should_synthesize(self, current_iteration: int) -> bool:
        """True if current_iteration is a multiple of synthesis_interval."""

    def synthesize(self, top_artifacts: list[Artifact], recent_scores: list[float],
                   failed_approaches: list[str]) -> str:
        """Generate structured scratchpad text from recent evolution history.
        Returns markdown with sections:
        - Patterns That Work (techniques in top artifacts)
        - Patterns That Fail (from recent failures)
        - Recommended Directions (actionable next steps)
        """

    def load(self) -> Optional[str]:
        """Load current scratchpad from evolve-state/meta_scratchpad.json."""

    def save(self, content: str) -> None:
        """Persist scratchpad to evolve-state/meta_scratchpad.json."""
```

**Integration:**
- `cli.py` next: Load scratchpad. If `should_synthesize()`, regenerate from top-K artifacts and recent scores. Inject into context via `context_builder`.
- `context_builder.py`: New `scratchpad_text` parameter. Template: `## Meta-Scratchpad (Accumulated Insights)\n{scratchpad_text}`
- `templates.py`: Add `SCRATCHPAD_TEMPLATE`.

**Scratchpad generation is deterministic** — it analyzes artifact contents and score patterns, NOT an LLM call. Uses `novelty.semantic_fingerprint()` to extract concepts (function names, algorithm keywords, imports) from each artifact. Then:
- **Patterns That Work**: Concepts appearing in 2+ of the top-K artifacts (intersection of fingerprints)
- **Patterns That Fail**: Concepts unique to failed approaches (difference of failed fingerprints vs successful)
- **Recommended Directions**: Concepts from evaluator source not yet present in any top artifact (gap analysis)

**Tests:** 8-10 tests covering synthesis, persistence, interval logic.

### QW2: Reflexion on Failures (ShinkaEvolve)

**Changes:**
- `cli.py` submit: When eval returns error OR `score < parent_score * 0.95` (significant regression):
  - Capture: `{"approach": strategy_description, "error": error_msg_or_null, "score": score, "parent_score": parent_score}`
  - Append to `evolve-state/recent_failures.json` (circular buffer, max 5 entries)
- `cli.py` next: Load recent failures, inject into context.
- `context_builder.py`: New `failures_text` parameter.
- Template: `## Recent Failures (Avoid These)\n{for f in failures: "- Approach: {f.approach}. Result: {f.error or 'score dropped to ' + f.score}"}`

**Tests:** 5 tests covering capture, circular buffer, injection.

### QW3: Pairwise Comparison Reflection (ReEvo, simplified)

**Changes:**
- `cli.py` next: After selecting parent, select comparison artifact using this algorithm:
  1. Get all artifacts NOT in parent's MAP-Elites cell
  2. Sort by absolute score difference from parent (largest first)
  3. Pick the first one (maximizes contrast for useful comparison)
  4. If no other cells populated, skip comparison (graceful degradation)
- Include both in context with prompt: "The parent (score {X}) and this comparison (score {Y}) differ in their approach. Consider what makes the higher-scoring one better."
- `context_builder.py`: New `comparison_artifact` parameter. Renders code snippets of both.

This is the simplified version — no separate LLM call. The comparison is embedded in the mutation prompt so Claude generates the reflection inline.

**Tests:** 3 tests covering selection and formatting.

### QW4: Ascending Score Sort (OPRO)

**Changes:**
- `context_builder.py` (NOT `database.py`): In `_format_evolution_history()` or equivalent method that renders the top programs section, reverse the received list to sort ascending (worst first, best last). Do NOT change `database.get_top_programs()` itself — that returns descending for other consumers like `export`.
- Add score annotation: each program prefixed with `# Score: {score}`

**Rationale:** OPRO showed ascending sort creates an improvement trajectory the LLM extrapolates upward from. The best example is always last — freshest in context.

**Tests:** 2 tests: assert sort order in rendered text is ascending, assert `get_top_programs()` still returns descending.

### QW5: Evaluator Code as Context (Eureka)

**Changes:**
- `cli.py` init: Read evaluator source file, store path in state metadata.
- `cli.py` next: Read evaluator source (truncated to first 200 lines if longer). Inject into context.
- `context_builder.py`: New `evaluator_source` parameter.
- Template: `## Evaluator Source (How Candidates Are Scored)\n\`\`\`python\n{evaluator_source}\n\`\`\``

**Tests:** 2 tests covering truncation and injection.

### QW6: Pre-Evaluation Novelty Gate (ShinkaEvolve)

**Changes:**
- `cli.py` submit: Before running evaluator, compute `novelty_score = novelty.compute_novelty(candidate_content, archive_member.content)` against top-10 archive members by score.
- Note: `compute_novelty()` returns **novelty** (1.0 = completely different, 0.0 = identical). Similarity = 1 - novelty.
- If ANY `novelty_score < novelty_gate_min_novelty` (default 0.05, i.e. >95% similar): reject submission with message "Candidate too similar to existing solution {id} (novelty: {novelty_score:.2f}, threshold: {threshold}). Try a more novel approach."
- Config: `novelty_gate_min_novelty: float = 0.05` in `SelectionConfig` (threshold on novelty, not similarity).
- Skip the gate if archive has < 3 members (not enough data).

**Tests:** 3 tests: pass when novel (novelty > 0.05), reject when too similar (novelty < 0.05), skip when archive small.

---

## Phase 3: Critical Gap Closers

### CG1: Continuous Improvement Signal G_t (AdaEvolve)

**New file:** `claude_evolve/claude_evolve/core/improvement_signal.py` (~80 lines)

```python
@dataclass
class ImprovementSignal:
    g_t: float = 0.0  # Accumulated improvement signal
    rho: float = 0.95  # Decay factor
    i_min: float = 0.1  # Min exploration intensity
    i_max: float = 0.7  # Max exploration intensity
    meta_threshold: float = 0.12  # Trigger meta-guidance when g_t <= this for all islands
    per_island_g_t: dict[int, float] = field(default_factory=dict)

    def update(self, child_score: float, parent_score: float, island_id: int) -> None:
        """Update G_t after an evaluation.
        delta = max((child_score - parent_score) / max(parent_score, 1e-10), 0)
        g_t = rho * g_t + (1 - rho) * delta   # Mean-tracking (NOT delta^2)
        Also updates per-island signal.
        Note: We use delta (not delta^2) to track the running mean of improvement
        magnitude. This matches AdaEvolve's operational intent: G_t directly
        measures how much improvement is happening per iteration.
        """

    @property
    def exploration_intensity(self) -> float:
        """I_t = I_min + (I_max - I_min) / (1 + G_t / 0.05)
        Returns value in [I_min, I_max]. High = explore, Low = exploit.

        Worked examples:
        - G_t = 0.0 (no improvement): I_t = 0.1 + 0.6/1.0 = 0.7 (max explore)
        - G_t = 0.05 (moderate): I_t = 0.1 + 0.6/2.0 = 0.4 (balanced)
        - G_t = 0.25 (strong): I_t = 0.1 + 0.6/6.0 = 0.2 (exploit)
        - G_t = 1.0 (massive): I_t = 0.1 + 0.6/21.0 = 0.129 (near min)
        """

    def should_trigger_meta_guidance(self) -> bool:
        """True when ALL islands have g_t <= meta_threshold (0.12)."""

    def derive_stagnation_level(self) -> StagnationLevel:
        """Backward-compatible: map G_t ranges to discrete levels.
        G_t > 0.1 → NONE, > 0.05 → MILD, > 0.02 → MODERATE, > 0.005 → SEVERE, else CRITICAL
        (Thresholds calibrated for mean-tracking, not variance-tracking.)
        """

    def save(self, path: str) -> None:
        """Persist to JSON."""

    @classmethod
    def load(cls, path: str) -> 'ImprovementSignal':
        """Load from JSON."""
```

**Integration:**
- `cli.py` submit: `signal.update(child_score, parent_score, island_id)` after evaluation.
- `cli.py` next: Load signal. Pass `exploration_intensity` to strategy selection (CG4), parent sampling (CG5), and context builder. If `should_trigger_meta_guidance()`, add breakthrough-tactics prompt section.
- `stagnation.py`: Keep existing `StagnationEngine` but prefer `signal.derive_stagnation_level()` when signal is available. Fall back to iteration-count-based detection for backward compatibility.
- `config.py`: Add `ImprovementSignalConfig(rho=0.95, i_min=0.1, i_max=0.7, meta_threshold=0.12)`.

**Meta-guidance prompt** (when triggered): `## BREAKTHROUGH REQUIRED\nAll search directions have stagnated. You MUST try a fundamentally different algorithmic approach. Analyze the evaluator source and current best to identify completely untried strategies. Do NOT make incremental changes.`

**Tests:** 10-12 tests covering signal update, exploration intensity curve, meta-guidance trigger, stagnation level derivation, persistence.

### CG2: Thought-Code Coevolution (EoH, ICML 2024 Oral)

**Changes to `core/artifact.py`:**
- Add field: `rationale: Optional[str] = None`

**Changes to `prompt/templates.py`:**
- New template `RATIONALE_SECTION`:
```
## Current Approach Rationale
{rationale}

## Instructions
1. First, write an updated rationale (2-3 sentences) describing your improved approach between RATIONALE-START and RATIONALE-END markers
2. Then write the code implementing that approach

Example:
RATIONALE-START
I will use simulated annealing with adaptive cooling because the search space has many local optima...
RATIONALE-END
```

**Changes to `context_builder.py`:**
- Include parent's rationale when available.
- Include top programs' rationales alongside their code snippets (truncated to 1 sentence each for space).

**Changes to `cli.py` submit:**
- After reading candidate file, extract text between `RATIONALE-START` and `RATIONALE-END` markers.
- Store in artifact's `rationale` field.
- If no markers found, set rationale to None (graceful degradation).

**Changes to `core/database.py`:**
- Serialize/deserialize `rationale` field in artifact JSON.

**Tests:** 6 tests covering extraction, storage, formatting, graceful degradation.

### CG3: Pairwise Reflection / Verbal Gradients (ReEvo, NeurIPS 2024)

**New file:** `claude_evolve/claude_evolve/core/reflection.py` (~100 lines)

```python
@dataclass
class ShortReflection:
    better_id: str
    worse_id: str
    better_score: float
    worse_score: float
    insight: str  # <20 words, generated by analyzing code diffs
    iteration: int

class ReflectionEngine:
    def __init__(self, state_dir: str, max_short: int = 20, synthesis_interval: int = 5):
        self.short_reflections: list[ShortReflection] = []
        self.long_reflection: str = ""

    def generate_short_reflection(self, better: Artifact, worse: Artifact) -> ShortReflection:
        """Deterministic analysis using novelty.semantic_fingerprint():
        1. Extract semantic_fingerprint(better.content) and semantic_fingerprint(worse.content)
        2. Compute added_concepts = better_fp - worse_fp (set difference)
        3. Compute removed_concepts = worse_fp - better_fp
        4. Format insight: "Better adds: {top-3 added}. Drops: {top-3 removed}."
        5. If no concept differences, fall back to line count / import diff.
        Returns concise insight string (<20 words).
        """

    def accumulate_long_reflection(self, current_iteration: int) -> Optional[str]:
        """Every synthesis_interval iterations, compress short reflections into
        a long-term reflection (<50 words). Returns new long reflection or None.
        Algorithm: count concept occurrences across all short reflections.
        Format the top-5 most-added and top-3 most-removed concepts as the
        long-term insight. No LLM call.
        """

    def format_for_prompt(self) -> str:
        """Format latest short reflection + long-term reflection for injection."""

    def save(self, path: str) -> None:
    def load(cls, path: str) -> 'ReflectionEngine':
```

**Integration:**
- `cli.py` next: After selecting parent + comparison, call `reflection.generate_short_reflection(parent, comparison)`. Add to short reflections. Check if `accumulate_long_reflection()` triggers. Inject via context builder.
- `context_builder.py`: New `reflection_text` parameter with both short-term and long-term sections.

**Reflection generation is deterministic** — uses `novelty.semantic_fingerprint()` for concept extraction, then set operations to identify what changed. No LLM call needed.

**Tests:** 8 tests covering short reflection generation, accumulation, persistence, formatting.

### CG4: UCB1 Strategy Selection (ShinkaEvolve + AdaEvolve)

**New file:** `claude_evolve/claude_evolve/core/ucb_selector.py` (~90 lines)

```python
@dataclass
class StrategyArm:
    strategy_id: str
    total_reward: float = 0.0
    visit_count: int = 0
    decayed_reward: float = 0.0

class UCBStrategySelector:
    def __init__(self, strategy_ids: list[str], c: float = 1.414, decay: float = 0.95):
        self.arms: dict[str, StrategyArm] = {sid: StrategyArm(sid) for sid in strategy_ids}
        self.c = c  # Exploration constant
        self.decay = decay  # Reward decay per iteration
        self.total_selections = 0

    def select(self, exploration_intensity: float = 0.5) -> str:
        """Select strategy using UCB1.
        UCB1_score = R_k/n_k + C_adj * sqrt(ln(N) / n_k)
        C_adj = c * (0.5 + exploration_intensity)  # More exploration when G_t is low
        Unvisited arms always selected first (infinite UCB).
        """

    def record(self, strategy_id: str, score_delta: float) -> None:
        """Record outcome. Reward = min(max(score_delta, 0), 1.0) (capped linear).
        Using capped linear instead of exponential to prevent single lucky outcomes
        from dominating future selections. Cap at 1.0 bounds the reward.
        Decay all arms' rewards by self.decay (recency bias).
        Note: decay is applied to ALL arms on every record() call, creating
        intentional recency bias — under-selected arms' rewards decay even when
        not used, encouraging re-exploration of stale arms via UCB1 bonus.
        """

    def save(self, path: str) -> None:
    @classmethod
    def load(cls, path: str) -> 'UCBStrategySelector':
```

**Integration:**
- `cli.py` next: Load UCB selector. Call `selector.select(exploration_intensity)` to choose strategy.
- `cli.py` submit: Call `selector.record(strategy_id, child_score - parent_score)`.
- `config.py`: Add `ucb_c: float = 1.414`, `ucb_decay: float = 0.95`, `strategy_selection: str = "ucb1"` to config.
- Backward compat: if `strategy_selection == "weighted"`, use existing `StrategyManager.select_strategy()`.

**Tests:** 8 tests covering selection, reward recording, decay, exploration modulation, unvisited arms.

### CG5: Power-Law Parent Selection with Adaptive Alpha (ShinkaEvolve + FunSearch)

**Changes to `core/database.py`:**

```python
def select_parent_power_law(self, island_id: int, exploration_intensity: float = 0.5) -> Optional[Artifact]:
    """Rank-based power-law parent selection.
    alpha = 0.3 + 3.0 * (1 - exploration_intensity / i_max)
      where i_max = 0.7 (from ImprovementSignal config)
    Normalize exploration_intensity to [0, 1] range before computing alpha.
    Worked examples (with i_max=0.7):
      - exploration_intensity=0.7 (max explore): alpha = 0.3 + 3.0*0.0 = 0.3 (nearly uniform)
      - exploration_intensity=0.4 (balanced):    alpha = 0.3 + 3.0*0.43 = 1.59
      - exploration_intensity=0.1 (min explore):  alpha = 0.3 + 3.0*0.86 = 2.87 (strong pressure)
    p_i = rank_i^(-alpha) / sum(rank_j^(-alpha))
    Additionally weight by novelty: h_i = 1 / (1 + offspring_count_i)
    Final: p_i proportional to fitness_weight_i * h_i
    """
```

**Changes to `core/artifact.py`:**
- Add field: `offspring_count: int = 0`

**Changes to `cli.py`:**
- `submit`: Increment parent's `offspring_count` after evaluation.
- `next`: Call `db.select_parent_power_law(island_id, exploration_intensity)` instead of current sampling.

**Changes to `core/database.py`:**
- Serialize/deserialize `offspring_count` in artifact JSON.

**Tests:** 6 tests covering alpha adaptation, probability distribution, offspring weighting, edge cases (single artifact, empty island).

---

## Configuration Additions

New fields in `config.py`:

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
    strategy_selection: str = "ucb1"  # "ucb1" or "weighted"
    ucb_c: float = 1.414
    ucb_decay: float = 0.95
    parent_selection: str = "power_law"  # "power_law" or "weighted"
    novelty_gate_min_novelty: float = 0.05  # Reject if novelty < this (0.05 = >95% similar)

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

**Master Config integration** — add to the existing `Config` dataclass:
```python
improvement_signal: ImprovementSignalConfig = field(default_factory=ImprovementSignalConfig)
selection: SelectionConfig = field(default_factory=SelectionConfig)
scratchpad: ScratchpadConfig = field(default_factory=ScratchpadConfig)
reflection: ReflectionConfig = field(default_factory=ReflectionConfig)
```
The existing `dacite.from_dict` with `Config(cast=[List, Union])` handles nested dataclass deserialization.

---

## New Files Summary

| File | Lines (est.) | Purpose |
|------|-------------|---------|
| `core/scratchpad.py` | ~120 | Meta-scratchpad generation and persistence |
| `core/reflection.py` | ~100 | Short/long-term reflection engine |
| `core/improvement_signal.py` | ~80 | G_t continuous signal + exploration intensity |
| `core/ucb_selector.py` | ~90 | UCB1 bandit strategy selection |
| `core/orchestrator.py` | ~150 | IterationOrchestrator wiring next/submit lifecycle |

## Modified Files Summary

| File | Changes |
|------|---------|
| `core/database.py` | Island-artifact sync, thread safety, power-law selection, offspring tracking |
| `core/artifact.py` | Add `rationale`, `offspring_count` fields |
| `core/stagnation.py` | G_t integration for backward-compatible level derivation |
| `core/strategy.py` | Outcome recording wiring |
| `core/memory.py` | Population calls from submit |
| `core/warm_cache.py` | LRU eviction policy |
| `core/evaluator.py` | Error logging, exponential backoff |
| `core/novelty.py` | Pre-eval gate logic |
| `cli.py` | Delegates to IterationOrchestrator for next/submit, iteration manifest |
| `prompt/context_builder.py` | New sections: scratchpad, reflection, evaluator, ascending sort, comparison, rationale |
| `prompt/templates.py` | New templates for rationale, reflection, meta-guidance, scratchpad, failures |
| `config.py` | New config dataclasses |

## Test Expectations

| Area | New Tests |
|------|-----------|
| Phase 1 fixes | ~15 |
| QW1 Meta-scratchpad | ~10 |
| QW2 Failure reflexion | ~5 |
| QW3 Pairwise comparison | ~4 |
| QW4 Ascending sort | ~2 |
| QW5 Evaluator context | ~2 |
| QW6 Novelty gate | ~3 |
| CG1 Improvement signal | ~12 |
| CG2 Thought-code coevolution | ~6 |
| CG3 Reflection engine | ~8 |
| CG4 UCB1 selector | ~8 |
| CG5 Power-law selection | ~6 |
| Orchestrator integration | ~8 |
| **Total** | **~89 new tests** |

---

## References

- [AdaEvolve: Adaptive LLM Driven Zeroth-Order Optimization](https://arxiv.org/abs/2602.20133) (Feb 2026)
- [ShinkaEvolve: Open-Ended and Sample-Efficient Program Evolution](https://arxiv.org/abs/2509.19349) (ICLR 2026)
- [Evolution of Heuristics (EoH)](https://arxiv.org/abs/2401.02051) (ICML 2024 Oral)
- [ReEvo: LLMs as Hyper-Heuristics with Reflective Evolution](https://arxiv.org/abs/2402.01145) (NeurIPS 2024)
- [OPRO: Large Language Models as Optimizers](https://arxiv.org/abs/2309.03409) (ICLR 2024)
- [AlphaEvolve](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/AlphaEvolve.pdf) (2025)
- [FunSearch](https://www.nature.com/articles/s41586-023-06924-6) (Nature 2023)
- [Eureka](https://arxiv.org/abs/2310.12931) (ICLR 2024)
- [CodeEvolve](https://arxiv.org/abs/2510.14150) (Oct 2025)
- [LoongFlow](https://arxiv.org/abs/2512.24077) (Dec 2025)
