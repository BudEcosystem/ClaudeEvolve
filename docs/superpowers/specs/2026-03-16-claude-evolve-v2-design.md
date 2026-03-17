# Claude Evolve v2: Research-Driven Evolutionary Discovery System

## Design Specification

**Date:** 2026-03-16
**Problem:** Claude Evolve currently optimizes code within known paradigms but cannot discover fundamentally new approaches to open problems. It lacks research capabilities, stagnation awareness, failure diagnosis, and multi-agent exploration.

**Goal:** Transform Claude Evolve from a "code mutator" into a "research-driven discovery system" capable of breaking novel open problems through deep literature review, structured ideation, multi-agent exploration, and adaptive search strategies.

---

## 1. Executive Summary

After running 17 iterations on the R(5,5) Ramsey number problem, we identified 10 architectural gaps that prevent Claude Evolve from tackling truly open problems. Drawing from extensive research into AlphaEvolve, FunSearch, AI Scientist v2, ShinkaEvolve, AFLOW, and 30+ other systems, we propose a layered enhancement architecture:

| Layer | What It Does | Inspired By |
|-------|-------------|-------------|
| **Stagnation Engine** | Detects plateaus, adapts strategy | AlphaEvolve ablation studies |
| **Research Agent** | Literature review, web search, approach discovery | AI Scientist v2, Agent Laboratory |
| **Diagnostician Agent** | Failure analysis, violation structure mapping | GVU framework ("strengthen the verifier") |
| **Strategy Evolver** | Evolves the research strategy itself | DSPy MIPROv2, EvoPrompt, CodeEvolve meta-prompting |
| **Multi-Agent Parallel Explorer** | Spawns diverse agents per iteration | CAMEL, EvoAgent, MAR |
| **Solver Integration** | SAT/SMT + LLM hybrid formulations | AutoModSAT, ConstraintLLM |
| **Cross-Run Memory** | Persists learnings across evolution runs | Reflexion, Agent Laboratory |

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    EVOLUTION ITERATION LOOP                      │
│                                                                  │
│  ┌──────────┐   ┌──────────────┐   ┌───────────────────────┐   │
│  │ DIAGNOSE │──>│   RESEARCH   │──>│    GENERATE + SUBMIT   │   │
│  │  Phase   │   │    Phase     │   │       Phase            │   │
│  └──────────┘   └──────────────┘   └───────────────────────┘   │
│       │               │                      │                   │
│       │               │                      │                   │
│  ┌────▼────┐   ┌──────▼───────┐   ┌─────────▼──────────┐      │
│  │Stagnation│  │Research Agent│   │ Parallel Explorers  │      │
│  │ Engine  │   │(web, papers) │   │ (3-4 agents with    │      │
│  │         │   │              │   │  different mandates) │      │
│  │Diagnostic│  │Strategy      │   │                     │      │
│  │ Agent   │   │ Evolver     │   │ Solver Integration   │      │
│  └─────────┘   └──────────────┘   └─────────────────────┘      │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              ENHANCED PROGRAM DATABASE                     │   │
│  │  MAP-Elites + Islands + Cross-Run Memory + Novelty Check  │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Component Designs

### 3.1 Stagnation Engine

**What:** Detects when evolution has plateaued and triggers adaptive responses.

**Why:** In our R(5,5) run, iterations 11-17 all produced identical score (0.9857). The system had no awareness it was stuck and kept trying the same approaches. AlphaEvolve's ablation studies confirm the evolutionary loop is critical, but only when it actually explores new territory.

**Design:**

```python
# New file: claude_evolve/claude_evolve/core/stagnation.py

class StagnationLevel(Enum):
    NONE = "none"           # Score improving normally
    MILD = "mild"           # 3-5 iterations without improvement
    MODERATE = "moderate"   # 6-10 iterations, same score plateau
    SEVERE = "severe"       # 10+ iterations, exhausted local strategies

@dataclass
class StagnationReport:
    level: StagnationLevel
    iterations_stuck: int
    best_score_history: List[float]    # Last N iteration best scores
    approaches_tried: List[str]         # Heuristic labels of attempted approaches
    recommended_action: str             # "force_paradigm_shift" | "expand_research" | "try_solver"
    underexplored_features: List[str]   # MAP-Elites cells with low occupancy

class StagnationEngine:
    def analyze(self, db: ArtifactDatabase, config: Config) -> StagnationReport:
        """Analyze population history and return stagnation assessment."""
        history = db.get_score_history(window=config.stagnation_window)

        # Compute improvement rate
        if len(history) < 3:
            return StagnationReport(level=StagnationLevel.NONE, ...)

        max_improvement = max(history) - min(history[-config.stagnation_window:])

        if max_improvement < config.stagnation_threshold:
            iterations_flat = self._count_flat_iterations(history, config.stagnation_threshold)
            level = self._classify_level(iterations_flat)

            # Identify underexplored regions in MAP-Elites
            underexplored = db.get_sparse_feature_cells(min_occupancy=2)

            # Recommend action based on level
            action = self._recommend_action(level, iterations_flat)

            return StagnationReport(
                level=level,
                iterations_stuck=iterations_flat,
                recommended_action=action,
                underexplored_features=underexplored,
                ...
            )
```

**Adaptive responses by level:**

| Level | exploration_ratio | Template | Research | Agents |
|-------|------------------|----------|----------|--------|
| NONE | 0.2 (default) | diff_user | Optional | 1 |
| MILD | 0.5 | full_rewrite_user | Encouraged | 1-2 |
| MODERATE | 0.7 | research_user | Mandatory | 2-3 |
| SEVERE | 0.9 | paradigm_shift_user | Mandatory + solver | 3-4 |

**Integration point:** `stop-hook.sh` calls `claude-evolve diagnose` before `claude-evolve next`. The diagnose command runs `StagnationEngine.analyze()` and writes `diagnostic_report.json` to the state directory.

---

### 3.2 Research Agent

**What:** A dedicated agent that performs literature review, web search, and approach discovery before each iteration (or when triggered by stagnation).

**Why:** In our R(5,5) run, the breakthrough from 43→2 mono-K_5 required: (1) reading Exoo's 1989 paper from arXiv, (2) understanding Cyclic(43) structure, (3) inventing the hybrid bottom-up+SA approach. All of this was done manually. A research agent could have discovered these insights autonomously.

**Inspired by:** AI Scientist v2 (progressive tree-search), Agent Laboratory (literature review phase), FunSearch (searching in function space).

**Design:**

```markdown
# New file: plugin/agents/researcher.md

---
name: researcher
description: Research agent for evolutionary optimization. Performs literature
  review, web search, and approach discovery for the current problem.
model: inherit
allowedTools: Read, Grep, Glob, Bash, WebSearch, WebFetch
---

You are a RESEARCH SPECIALIST for an evolutionary optimization system.

## Your Mission
Find novel approaches, relevant papers, and implementation insights that could
help improve the current best solution. Focus on ACTIONABLE findings.

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
Your final message MUST be raw JSON:
{
  "approaches": [
    {"name": "...", "description": "...", "novelty": "high|medium|low",
     "implementation_hint": "...", "source_url": "..."}
  ],
  "theoretical_bounds": {
    "known_upper": "...", "known_lower": "...",
    "impossibility_results": "..."
  },
  "key_papers": [{"title": "...", "url": "...", "relevance": "..."}],
  "approaches_to_avoid": ["..."],
  "recommended_next_step": "..."
}
```

**Trigger conditions:**
- Always on first 3 iterations (exploration phase)
- When stagnation level >= MILD
- Every N iterations (configurable, default 10)
- When user explicitly enables `research_mode: true`

**Persistence:** Research findings are appended to `.claude/evolve-state/research_log.md` and summarized in subsequent iteration prompts. This prevents re-researching the same things.

---

### 3.3 Diagnostician Agent

**What:** Analyzes WHY candidates fail, identifying specific structural patterns in the failures.

**Why:** The GVU framework's key insight is "strengthen the verifier, not the generator." In our R(5,5) run, understanding that the 2 remaining mono-K_5 shared vertex 42 and that flipping their shared edges always created new violations — this structural insight was never surfaced automatically.

**Inspired by:** GVU framework (variance inequality), Agent Laboratory (experiment analysis phase), Multi-Agent Reflexion (diagnostic persona).

**Design:**

```markdown
# New file: plugin/agents/diagnostician.md

---
name: diagnostician
description: Failure analysis agent. Examines why recent candidates fail and
  identifies structural patterns in the violations.
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
   - EVALUATION_GAP: The evaluator has blind spots or ambiguities
5. **Recommend specific actions** based on diagnosis

## Output Format
{
  "failure_mode": "LOCAL_MINIMUM|STRUCTURAL_BARRIER|PARADIGM_LIMIT|EVALUATION_GAP",
  "evidence": ["..."],
  "violation_structure": {"shared_elements": [...], "interlocking_constraints": [...]},
  "approaches_exhausted": ["..."],
  "recommended_pivot": "...",
  "confidence": 0.0-1.0
}
```

---

### 3.4 Strategy Evolver (Meta-Prompting)

**What:** Evolves the research/generation strategy itself, not just the candidate code.

**Why:** CodeEvolve's ablation studies show meta-prompting (LLMs reflecting on and rewriting their own instructions) unlocks diverse evolutionary pathways. DSPy's MIPROv2 demonstrates that any text-based strategy can be systematically optimized through Bayesian search.

**Inspired by:** CodeEvolve (meta-prompting), DSPy MIPROv2 (prompt optimization), EvoPrompt (evolutionary prompt optimization), ShinkaEvolve (adaptive LLM selection).

**Design:**

The strategy evolver maintains a `strategy_population` — a set of text descriptions of how to approach the problem. These strategies are themselves evolved using the same MAP-Elites framework:

```python
# New file: claude_evolve/claude_evolve/core/strategy_evolver.py

@dataclass
class Strategy:
    id: str
    description: str          # Natural language strategy description
    template_overrides: Dict   # Which template/fragments to use
    research_focus: str        # What to research
    generation_approach: str   # "diff" | "full_rewrite" | "solver_hybrid" | "from_scratch"
    score_history: List[float] # Scores of candidates generated with this strategy

class StrategyEvolver:
    def __init__(self, config):
        self.strategies = []  # Population of strategies
        self.best_strategy = None

    def evolve_strategy(self, db, stagnation_report, research_findings):
        """Generate a new strategy by mutating/crossing existing ones."""
        # If stagnated, generate a NOVEL strategy via LLM
        if stagnation_report.level >= StagnationLevel.MODERATE:
            prompt = f"""
            The evolution is stuck at score {stagnation_report.best_score_history[-1]}.
            Approaches tried: {stagnation_report.approaches_tried}
            Research findings: {research_findings}

            Propose a FUNDAMENTALLY DIFFERENT strategy that avoids all previously
            tried approaches. Be specific and actionable.
            """
            # Call LLM to generate new strategy
            new_strategy = self._llm_generate_strategy(prompt)
            return new_strategy

        # Otherwise, mutate the best-performing strategy
        return self._mutate_strategy(self.best_strategy)

    def update_scores(self, strategy_id, candidate_score):
        """Track which strategies produce the best candidates."""
        # ... update score_history, recompute best_strategy
```

**Key insight from EvoPrompt:** Strategies can be crossed over just like code. Two strategies can be combined by showing both to an LLM and asking it to synthesize the best elements of each.

---

### 3.5 Multi-Agent Parallel Explorer

**What:** Spawns 2-4 agents per iteration, each with a different mandate, when stagnation is detected.

**Why:** In our R(5,5) run, the biggest breakthroughs came from paradigm shifts (bottom-up vs. top-down, SA vs. deterministic). A single agent per iteration tends to stay in the same paradigm. Multiple agents with forced different perspectives discover more.

**Inspired by:** EvoAgent (evolutionary multi-agent generation), MAR (multi-agent reflexion with diverse personas), CAMEL (role-playing methodology), AlphaEvolve's dual-LLM approach (Flash for breadth, Pro for depth).

**Design:**

When stagnation level >= MODERATE, the stop hook spawns parallel agents:

```bash
# In stop-hook.sh, after stagnation detection:
if [[ "$STAGNATION_LEVEL" == "moderate" ]] || [[ "$STAGNATION_LEVEL" == "severe" ]]; then
    # Spawn parallel exploration agents in worktrees
    # Agent 1: Incremental (diff from best parent)
    # Agent 2: Creative leap (full rewrite, forced novelty)
    # Agent 3: Research-first (literature review -> implementation)
    # Agent 4: Solver hybrid (formulate as SAT/constraint problem)
fi
```

Each agent gets a different variant of the iteration context:

| Agent | Template | Directive | exploration_ratio |
|-------|----------|-----------|-------------------|
| Incrementalist | diff_user | "Improve the parent with targeted changes" | 0.2 |
| Creative Leaper | paradigm_shift_user | "Ignore the parent. Start from a completely different approach" | 1.0 |
| Researcher | research_user | "Spend 80% of effort on literature review, then implement" | 0.8 |
| Solver Hybrid | solver_user | "Formulate this as a constraint satisfaction problem" | 0.5 |

All agents submit independently. The best candidate wins.

---

### 3.6 Solver Integration Layer

**What:** Enables hybrid LLM + formal solver approaches (SAT, SMT, constraint programming).

**Why:** AutoModSAT achieved 50% improvement over baseline SAT solvers by using LLMs to design heuristics. For combinatorial problems like R(5,5), the search space is too large for pure LLM exploration but has rich structure that solvers can exploit. The LLM formulates; the solver searches.

**Inspired by:** AutoModSAT (LLM-designed SAT heuristics), ConstraintLLM (neuro-symbolic constraint solving), AlphaProof (RL + formal verification), Axiom AI (LLM → Lean proofs).

**Design:**

```python
# New template: solver_user
"""
## Solver-Hybrid Mode

Instead of directly generating a solution, formulate the problem as a
constraint satisfaction problem that a solver can tackle.

### Steps:
1. Read the evaluator to understand the constraints
2. Express the constraints formally (SAT clauses, LP constraints, or CSP)
3. Use Python's `pysat`, `z3-solver`, or `scipy.optimize` to solve
4. Convert the solver's output back to the evaluator's expected format

### Example for graph coloring:
```python
from pysat.solvers import Glucose3
# Each edge (i,j) has a boolean variable x_{ij} (True = red, False = blue)
# For each 5-subset, add clause preventing all-same-color
```

This approach works best when:
- The problem has clear combinatorial structure
- Pure heuristic search has stagnated
- The constraint set is expressible in CNF/SMT
"""
```

**Integration:** When stagnation is SEVERE or the diagnostician identifies STRUCTURAL_BARRIER, the solver template is injected into the iteration context, and the candidate is expected to use solver libraries.

---

### 3.7 Cross-Run Memory

**What:** Persists learnings, failed approaches, and research findings across evolution runs.

**Why:** FunSearch periodically resets stagnant islands by seeding from global top performers. But across runs, all knowledge is lost. After 17 iterations on R(5,5), the system knows exactly which approaches work (Exoo + hybrid SA → 2 mono-K_5) and which don't (QR circulants, random starts, distance-class flips alone). A new run should start with this knowledge.

**Inspired by:** Reflexion (verbal reinforcement learning with episodic memory), Agent Laboratory (persistent research context), FunSearch (island reset from global elites).

**Design:**

```python
# New file: claude_evolve/claude_evolve/core/memory.py

class CrossRunMemory:
    """Persists learnings across evolution runs for the same problem."""

    def __init__(self, memory_path: str):
        self.path = memory_path  # .claude/evolve-state/cross_run_memory.json

    def save_run_summary(self, db, stagnation_report, research_log):
        """Called at end of each run. Saves key learnings."""
        summary = {
            "run_id": str(uuid4()),
            "timestamp": datetime.now().isoformat(),
            "iterations": db.last_iteration,
            "best_score": db.get_best().metrics.get("combined_score", 0),
            "approaches_tried": self._extract_approach_labels(db),
            "hard_barriers": stagnation_report.approaches_exhausted,
            "research_findings": research_log.entries,
            "best_artifact_hash": hash(db.get_best().content),
            "recommendation_for_next_run": stagnation_report.recommended_pivot,
        }
        # Append to memory file

    def load_for_new_run(self) -> Dict:
        """Called at start of each run. Returns accumulated knowledge."""
        # Returns structured summary of all previous runs
        # Injected into the first iteration's prompt
```

**Injection into prompts:**

```markdown
## Previous Run Knowledge (from cross-run memory)

### Run 1 (17 iterations, best: 0.9857)
- Exoo(42) construction matches world record (0 mono-K_5 on K_42)
- Bottom-up vertex extension + targeted SA reaches 2 mono-K_5 on K_43
- HARD BARRIER: 2 mono-K_5 are structurally interlocked (exhaustive 1/2/3-edge flip search confirms)
- Approaches exhausted: circulant modifications, SA from Cyclic(43), multi-base vertex deletion
- RECOMMENDATION: Try non-Exoo-derived K_42 construction, or SAT solver formulation
```

---

## 4. New Iteration Flow

```
ITERATION N:
│
├─ 1. DIAGNOSE (new phase, ~2s)
│   ├─ StagnationEngine.analyze() → StagnationReport
│   ├─ If stagnation >= MILD: spawn DiagnosticianAgent
│   └─ Write diagnostic_report.json
│
├─ 2. RESEARCH (new phase, conditional)
│   ├─ If stagnation >= MILD OR iteration < 3 OR every 10th iteration:
│   │   ├─ Spawn ResearcherAgent (web search, paper analysis)
│   │   └─ Append findings to research_log.md
│   ├─ StrategyEvolver.evolve_strategy() → new Strategy
│   └─ Write research_context.json
│
├─ 3. CONTEXT GENERATION (enhanced claude-evolve next)
│   ├─ Select template based on stagnation level
│   ├─ Inject: stagnation report, research findings, failure patterns
│   ├─ Inject: cross-run memory (if exists)
│   ├─ Inject: strategy directives
│   ├─ Select parent/inspirations with stagnation-aware sampling
│   └─ Write iteration_context.md
│
├─ 4. GENERATE (existing, but enhanced)
│   ├─ If stagnation >= MODERATE: spawn parallel agents
│   │   ├─ Agent 1: Incrementalist
│   │   ├─ Agent 2: Creative Leaper
│   │   ├─ Agent 3: Researcher-Implementer
│   │   └─ Agent 4: Solver Hybrid (if SEVERE)
│   ├─ Each agent: Understand → Research → Generate → Validate → Submit
│   └─ Best submission wins
│
├─ 5. EVALUATE (existing)
│   └─ Score candidate(s)
│
└─ 6. UPDATE (enhanced)
    ├─ Update program database
    ├─ Update strategy scores
    ├─ Update cross-run memory (if run ending)
    └─ Trigger stop-hook for next iteration
```

---

## 5. Configuration Schema

```yaml
# New config options for Claude Evolve v2

stagnation:
  enabled: true
  window: 5                    # Iterations to look back
  threshold: 0.001             # Min improvement to not be stagnant
  severe_window: 10            # Iterations for severe stagnation
  auto_adapt_exploration: true # Automatically boost exploration when stuck

research:
  enabled: false               # Enable research agent
  trigger: "on_stagnation"     # "always" | "on_stagnation" | "periodic" | "never"
  periodic_interval: 10        # If periodic, every N iterations
  max_web_searches: 5          # Per research phase
  persist_findings: true       # Save to research_log.md

diagnostics:
  enabled: true
  trigger: "on_stagnation"     # When to run diagnostician agent
  failure_pattern_window: 10   # Iterations of history to analyze

strategy_evolution:
  enabled: false               # Enable meta-strategy evolution
  population_size: 5           # Number of strategies to maintain
  mutate_on_stagnation: true

parallel_agents:
  enabled: false               # Enable multi-agent exploration
  trigger: "on_moderate_stagnation"
  max_agents: 4
  agent_types: ["incrementalist", "creative_leaper", "researcher", "solver_hybrid"]

solver_integration:
  enabled: false               # Enable solver-hybrid mode
  trigger: "on_severe_stagnation"
  solver_libraries: ["pysat", "z3-solver", "scipy"]

cross_run_memory:
  enabled: true
  memory_path: ".claude/evolve-state/cross_run_memory.json"
  max_runs_remembered: 10
```

---

## 6. Implementation Priority

### Phase 1: Foundation (High Impact, Low Effort)
1. **Stagnation Engine** — Wire up existing dead `early_stopping_patience` config + new `detect_stagnation()` method on database
2. **Enhanced prompts** — Inject stagnation reports and improvement guidance into iteration context
3. **Novelty checking** — Implement `_is_novel()` (currently a no-op) using code embedding similarity
4. **Cross-run memory** — Simple JSON append/load between runs

### Phase 2: Research Capabilities (High Impact, Medium Effort)
5. **Research agent** — New agent definition + trigger logic in stop hook
6. **Diagnostician agent** — New agent definition + failure pattern analysis
7. **Research-oriented templates** — `research_user`, `paradigm_shift_user`, `solver_user`
8. **Persistent research log** — Append findings, inject into prompts

### Phase 3: Advanced Search (Medium Impact, High Effort)
9. **Strategy evolver** — Meta-prompting to evolve research strategies themselves
10. **Multi-agent parallel exploration** — Worktree-based parallel agents with different mandates
11. **Solver integration** — SAT/SMT template + library detection

### Phase 4: Polish (Lower Impact, Medium Effort)
12. **Phase-based iteration budgets** — Exploration → Exploitation → Deep Dive → Paradigm Shift
13. **Evaluator-in-the-loop feedback** — Structured `__feedback__` field in evaluator output
14. **Adaptive LLM selection** — Bandit-based model selection (ShinkaEvolve approach)

---

## 7. Key Research Sources

| System | Key Insight | Source |
|--------|-------------|--------|
| AlphaEvolve | Evolutionary loop + evaluation cascade + island migration | [arxiv.org/abs/2506.13131](https://arxiv.org/abs/2506.13131) |
| FunSearch | Search in function space + island reset of stagnant populations | [Nature 2023](https://www.nature.com/articles/s41586-023-06924-6) |
| AI Scientist v2 | Progressive agentic tree-search for scientific discovery | [arxiv.org/abs/2504.08066](https://arxiv.org/abs/2504.08066) |
| ShinkaEvolve | Novelty-based rejection + adaptive LLM selection | [sakana.ai/shinka-evolve](https://sakana.ai/shinka-evolve/) |
| CodeEvolve | Meta-prompting + inspiration-based crossover | [arxiv.org/html/2510.14150v1](https://arxiv.org/html/2510.14150v1) |
| GVU Framework | "Strengthen the verifier, not the generator" | [arxiv.org/abs/2512.02731](https://arxiv.org/abs/2512.02731) |
| DSPy MIPROv2 | Bayesian optimization of prompt strategies | [dspy.ai](https://dspy.ai/) |
| EvoPrompt | Evolutionary prompt optimization | [ICLR 2024](https://arxiv.org/abs/2309.08532) |
| AutoModSAT | LLM-designed SAT heuristics (50% improvement) | [arxiv.org/html/2507.22876v1](https://arxiv.org/html/2507.22876v1) |
| AFLOW | MCTS for agentic workflow optimization | [arxiv.org/abs/2410.10762](https://arxiv.org/abs/2410.10762) |
| Agent Laboratory | PhD/Postdoc agents for literature review + experimentation | [arxiv.org/abs/2501.04227](https://arxiv.org/abs/2501.04227) |
| QDAIF | Quality-Diversity through AI Feedback | [arxiv.org/abs/2310.13032](https://arxiv.org/abs/2310.13032) |
| Darwin Godel Machine | Self-improving coding agent | [sakana.ai/dgm](https://sakana.ai/dgm/) |
| AxiProver | LLM → Lean proofs for open math problems | [axiommath.ai](https://axiommath.ai/) |
| Reflexion | Verbal reinforcement learning with episodic memory | [arxiv.org/abs/2303.11366](https://arxiv.org/abs/2303.11366) |

---

## 8. Expected Impact

With these enhancements, Claude Evolve v2 would have:

**On R(5,5) specifically:**
- Automatically discovered Exoo's construction from web search (iteration 1 instead of manual)
- Detected stagnation at 13 mono-K_5 and forced paradigm shift to bottom-up (saving 3 iterations)
- Diagnosed the 2-violation structural barrier and recommended SAT solver formulation
- Persisted findings for the next run, starting from 2 mono-K_5 instead of scratch

**On open math problems generally:**
- Literature review before first iteration (know what's been tried)
- Automatic detection of known impossibility results (avoid wasting iterations)
- Multi-paradigm exploration (algebraic, computational, solver-based) in parallel
- Cross-run memory accumulating knowledge across attempts

**On broader optimization problems:**
- Strategy evolution finding the best meta-approach for each problem class
- Solver integration for combinatorial/constraint problems
- Research agent discovering domain-specific libraries and techniques
- Stagnation-aware adaptation preventing wasted compute
