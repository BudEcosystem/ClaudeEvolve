# Claude Evolve

**Evolutionary artifact optimization for Claude Code** — an open-source implementation of [AlphaEvolve](https://deepmind.google/discover/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/)-style evolutionary search that runs natively inside [Claude Code](https://docs.anthropic.com/en/docs/claude-code/overview) sessions.

Claude Evolve uses **MAP-Elites quality-diversity search** with island-based populations to evolve programs, prompts, algorithms, configurations, and any text artifact — with Claude acting as both the intelligent mutation engine and an autonomous research agent.

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-1039%20passing-brightgreen.svg)]()

---

## Headline Result: Circle Packing

Claude Evolve achieved **sum of radii = 2.635983** for packing 26 unequal circles in a unit square with strict non-overlap constraints, numerically exceeding published results from:

| System | Sum of Radii | Source |
|--------|-------------|--------|
| **Claude Evolve** | **2.635983** | **This project** |
| OpenEvolve community | 2.635977 | [GitHub #156](https://github.com/algorithmicsuperintelligence/openevolve/issues/156) |
| FICO Xpress (ZIB/MODAL) | 2.635916 | [FICO blog](https://www.fico.com/blogs/best-global-optimization-solver) |
| AlphaEvolve (DeepMind) | 2.635863 | [ArXiv 2511.02864](https://arxiv.org/abs/2511.02864) |

The solution uses strict zero-tolerance constraints (all pairwise gaps > 0, no evaluator tolerance exploitation). Full coordinates and an independent verification script are provided.

**[Read the full paper](docs/circle_packing_paper.md)** | **[Verify the result](docs/verify_circle_packing.py)** | **[Solution code](evolve_output/best_circle_packing_strict.py)**

---

## Table of Contents

- [How It Works](#how-it-works)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [v2/v3 Research-Driven Features](#v2v3-research-driven-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Evaluation Modes](#evaluation-modes)
- [Configuration](#configuration)
- [CLI Reference](#cli-reference)
- [Plugin Commands](#plugin-commands)
- [MAP-Elites & Island Evolution](#map-elites--island-evolution)
- [Writing Evaluators](#writing-evaluators)
- [Advanced Usage](#advanced-usage)
- [Project Structure](#project-structure)
- [License](#license)

---

## How It Works

Claude Evolve turns Claude Code into an evolutionary optimization engine. You provide:

1. **An artifact** — the file you want to improve (a Python program, a prompt, a config, an algorithm)
2. **An evaluator** — a script that scores candidates on 0.0-1.0 metrics, or a prompt for Claude-as-judge

Then Claude Evolve runs an evolution loop:

<img width="1440" height="1736" alt="image" src="https://github.com/user-attachments/assets/5ba83b44-ac12-416f-ad9e-ef4d2a56a96d" />


Each iteration, Claude receives **different context** — a new parent artifact selected from the population, inspiration from diverse high-performing solutions, stagnation diagnostics, strategy directives, and guidance on unexplored regions of the solution space.

---

## Key Features

### Evolutionary Algorithm
- **MAP-Elites quality-diversity search** — maintains diverse elite solutions across configurable feature dimensions
- **Island-based evolution** — multiple isolated populations with periodic migration
- **Universal novelty system** — 3-layer similarity (structural + behavioral + semantic) working across all artifact types, not just code
- **Stepping stones archive** — preserves diverse intermediate solutions that open new search space regions
- **7 built-in strategies** — Incremental, Creative Leap, Hybrid Synthesis, Research-Driven, Solver Hybrid, Multi-Iteration Accumulation, Problem Decomposition

### Research-Driven Discovery (v2)
- **Stagnation Engine** — detects plateaus (5 levels: NONE to CRITICAL) and adapts exploration
- **Continuous G_t Signal** — AdaEvolve-inspired exponential moving average replacing discrete stagnation levels, driving all adaptation from a single continuous signal
- **Research Agent** — literature search and approach discovery via web search
- **Diagnostician Agent** — root cause analysis of why evolution is stuck
- **UCB1 Strategy Selection** — bandit-based strategy selection replacing weighted-random, with capped reward and exploration modulation
- **Cross-Run Memory** — persists learnings, failed approaches, and successful strategies across runs
- **Meta-Scratchpad** — periodic pattern synthesis from evolution history (ShinkaEvolve-inspired)
- **Verbal Gradients** — pairwise reflection comparing artifacts to generate directional mutation guidance (ReEvo-inspired)
- **Thought-Code Coevolution** — evolves natural-language rationale alongside code for better LLM reasoning (EoH-inspired)

### Warm-Start & Accumulation (v3)
- **Warm-Start Cache** — persists intermediate computation (numpy arrays, JSON, text) between iterations with LRU eviction
- **Multi-Iteration Accumulation** — each iteration continues from where the last left off, enabling sustained search across hundreds of iterations
- **Evaluation Caching** — skip re-evaluation of deterministic results
- **Solution Seeding** — inject known-good solutions into the population
- **Power-Law Parent Selection** — rank-based selection with adaptive alpha from G_t signal and offspring novelty weighting (ShinkaEvolve/FunSearch-inspired)
- **Failure Reflexion** — captures recent failures with reasons, injecting "avoid these" guidance into future iterations
- **Pre-Evaluation Novelty Gate** — rejects near-duplicate candidates before wasting evaluation budget
- **IterationOrchestrator** — unified coordination of all feature modules for next/submit lifecycle

### Universal Novelty & Diversity (v3)
- **Structural similarity** — token n-gram overlap analysis working across Python, JS, YAML, JSON, SQL, markdown, and prose
- **Behavioral similarity** — metric fingerprint comparison (normalized score vectors across evaluation dimensions)
- **Semantic fingerprints** — concept extraction identifying algorithmic ideas, data structures, and approaches
- **Stepping stones** — archive of diverse intermediate solutions injected into iteration context for crossover-inspired evolution
- **Artifact-agnostic** — same novelty pipeline handles code, prompts, configs, and any text artifact

### Claude Code Integration
- **Native plugin** — runs inside Claude Code sessions via `/evolve` command
- **Dynamic per-iteration prompts** — each iteration gets fresh context with population insights and strategy directives
- **Full autonomy per iteration** — Claude can use web search, spawn subagents, run code, and apply any available skill
- **Critic mode** — Claude acts as adversarial evaluator for non-code artifacts

### Problem-Type Guidance
- **Quantitative problems** (math, optimization) — warm cache, multi-iteration accumulation, constraint propagation
- **Qualitative problems** (business, writing) — research agents, section-by-section iteration, style consistency
- **Hybrid problems** (data science, ML) — model checkpoints, hyperparameter search, problem decomposition

### Production Quality
- **1039 tests** covering unit, integration, and end-to-end flows
- **Subprocess isolation** — evaluator runs in isolated subprocess with timeout protection
- **Checkpoint/resume** — periodic snapshots with seamless resume
- **Session isolation** — multiple sessions don't interfere
- **Fresh init** — `init` clears stale state by default (cross-run memory preserved)

---

## Architecture

Claude Evolve is a three-layer hybrid system:


<img width="1440" height="1524" alt="image" src="https://github.com/user-attachments/assets/0719899d-2d6d-4407-a1d4-3c032e329d10" />



**Layer 1 (Shell)** manages the iteration lifecycle. The stop hook calls `diagnose` (stagnation detection) before `next` (context generation).

**Layer 2 (Python)** provides all deterministic logic — MAP-Elites database, evaluation, stagnation detection, strategy selection, warm-start caching, cross-run memory, research log management, and state persistence. This is a standalone `pip`-installable package with a `claude-evolve` CLI.

**Layer 3 (Skill + Agents)** teaches Claude how to behave during each iteration, with problem-type-specific guidance and specialized agents for research and diagnosis.

---

## v2/v3 Research-Driven Features

### Stagnation Engine
Detects when evolution has plateaued and adapts the search strategy:

| Level | Iterations Stuck | Response |
|-------|-----------------|----------|
| NONE | 0-2 | Continue normally |
| MILD | 3-5 | Increase exploration, try new approaches |
| MODERATE | 6-10 | Paradigm shift, spawn research agent |
| SEVERE | 11-20 | Radical departure, spawn diagnostician |
| CRITICAL | 20+ | Full restart, problem reformulation |

### Strategy Evolver
7 built-in strategies, selected based on stagnation level and past performance:

1. **Incremental Improvement** — small targeted changes (low exploration)
2. **Creative Leap** — ignore current approach, try something novel (high exploration)
3. **Hybrid Synthesis** — combine best elements from top solutions
4. **Research-Driven** — 80% effort on literature review, then implement
5. **Solver Hybrid** — formulate as constraint satisfaction, use solvers
6. **Multi-Iteration Accumulation** — continue from warm-cached state
7. **Problem Decomposition** — break into independent sub-problems

### Warm-Start Cache
Persists intermediate computation between iterations:

```python
# In your candidate code:
import os, numpy as np

# Load from previous iteration
cache_file = '.claude/evolve-state/warm_cache/items/best_matrix.npy'
if os.path.exists(cache_file):
    prev_best = np.load(cache_file)
    # Continue optimizing from prev_best

# Save for next iteration
os.makedirs('.claude/evolve-state/warm_cache/items', exist_ok=True)
np.save(cache_file, my_result)
```

### Cross-Run Memory
Learnings persist across evolution runs:
- Failed approaches (avoid repeating)
- Successful strategies (build on)
- Key insights from prior runs

### Universal Novelty System

Traditional code-evolution systems use code-specific similarity (AST diff, token overlap). Claude Evolve's novelty system works across **all artifact types** via three complementary layers:

| Layer | Method | What It Captures |
|-------|--------|-----------------|
| **Structural** | Token n-gram overlap (bigrams + trigrams) | Surface-level textual similarity |
| **Behavioral** | Metric fingerprint cosine similarity | Functional equivalence (same scores = same behavior) |
| **Semantic** | Concept extraction + Jaccard overlap | Algorithmic ideas, data structures, approaches |

Combined similarity = weighted average (structural 0.4, behavioral 0.3, semantic 0.3). Candidates above the novelty threshold are rejected as duplicates, preserving population diversity.

### Stepping Stones Archive

Inspired by FunSearch's best-shot prompting and ShinkaEvolve's novelty rejection sampling, the stepping stones archive preserves **diverse intermediate solutions** — not just the best. These are injected into iteration context to enable crossover-style evolution:

1. Each submission is checked against the archive for novelty
2. Sufficiently novel solutions are preserved regardless of fitness
3. During context generation, stepping stones from different search space regions are selected
4. Claude can combine ideas from stepping stones with the current best (semantic crossover)

This prevents the population from collapsing to a single approach and enables discovering solutions that require traversing low-fitness intermediates.

---

## Installation

### Prerequisites

- Python 3.10+
- [Claude Code](https://docs.anthropic.com/en/docs/claude-code/overview) installed
- `jq` (for stop hook JSON processing): `apt install jq` or `brew install jq`

### Install

```bash
git clone https://github.com/BudEcosystem/ClaudeEvolve.git
cd ClaudeEvolve

# Install with a virtual environment (recommended)
bash install.sh --venv

# Or install directly
bash install.sh
```

### Verify

```bash
claude-evolve --help
cd claude_evolve && python -m pytest tests/ -q  # 1039 tests
```

---

## Quick Start

### Example: Circle Packing (Benchmark)

```bash
cd ClaudeEvolve
claude  # Start Claude Code

# In Claude Code:
/evolve circle_packing/program.py circle_packing/evaluator.py --max-iterations 50 --target-score 1.0
```

### Example: Ramsey Number R(5,5)

```bash
/evolve ramsey_R5_5/program.py ramsey_R5_5/evaluator.py --max-iterations 500 --target-score 1.0
```

### Example: Evolve a Prompt (Critic Mode)

```bash
/evolve my_prompt.md eval_criteria.md --mode critic --max-iterations 20
```

---

## Evaluation Modes

### Script Mode (Default)

The evaluator is a Python script that scores candidates:

```python
# evaluator.py
import json, sys

def evaluate(candidate_path):
    return {
        "combined_score": 0.85,
        "accuracy": 0.9,
        "efficiency": 0.8,
    }

if __name__ == "__main__":
    result = evaluate(sys.argv[1])
    print(json.dumps(result))
```

### Critic Mode

Claude spawns an adversarial critic agent:

```bash
/evolve prompt.md eval_criteria.md --mode critic
```

### Hybrid Mode

Combines script evaluation with critic feedback:

```bash
/evolve algorithm.py benchmark.py --mode hybrid
```

---

## Configuration

Claude Evolve uses YAML configuration with sensible defaults. Key options:

| Category | Option | Default | Description |
|----------|--------|---------|-------------|
| General | `max_iterations` | 50 | Maximum evolution iterations |
| General | `target_score` | null | Early stop threshold |
| Database | `num_islands` | 5 | Isolated populations |
| Database | `population_size` | 1000 | Max population per island |
| Database | `feature_dimensions` | `["complexity", "diversity"]` | MAP-Elites grid |
| Database | `similarity_threshold` | 0.99 | Novelty rejection threshold |
| Evaluator | `timeout` | 300 | Evaluation timeout (seconds) |
| Stagnation | `enabled` | true | Enable stagnation detection |
| Stagnation | `mild_threshold` | 3 | Iterations for MILD level |
| Cross-Run | `enabled` | true | Persist learnings across runs |
| Research | `enabled` | false | Enable research agent |
| Research | `trigger` | `"on_stagnation"` | When to research |

See [Configuration docs](docs/) for full reference.

---

## CLI Reference

```bash
# Initialize a new evolution run (clears stale state)
claude-evolve init --artifact solution.py --evaluator benchmark.py

# Initialize preserving previous state
claude-evolve init --artifact solution.py --evaluator benchmark.py --keep-state

# Generate next iteration context (called by stop hook)
claude-evolve next --state-dir .claude/evolve-state

# Submit a candidate for evaluation
claude-evolve submit --candidate candidate.py --state-dir .claude/evolve-state

# Run stagnation diagnostics
claude-evolve diagnose --state-dir .claude/evolve-state

# Seed a known-good solution into the population
claude-evolve seed --artifact known_good.py --state-dir .claude/evolve-state

# Save/inspect warm cache
claude-evolve cache-put --key best_matrix --file result.npy --type numpy
claude-evolve cache --key best_matrix --state-dir .claude/evolve-state

# Cache evaluation results
claude-evolve cache-eval --n 42 --result '{"valid": true}'

# Append research findings
claude-evolve research-log --findings '{"approaches": [...]}'

# Show evolution progress
claude-evolve status --state-dir .claude/evolve-state

# Export best artifact
claude-evolve export --state-dir .claude/evolve-state --output best.py
```

---

## Plugin Commands

| Command | Description |
|---------|-------------|
| `/evolve <artifact> <evaluator> [OPTIONS]` | Start evolution loop |
| `/evolve-status` | Check progress |
| `/cancel-evolve` | Cancel and export best |

---

## MAP-Elites & Island Evolution

MAP-Elites maintains a **map of the best solution in each region** of a feature space, producing a diverse collection of high-performing solutions. Solutions are placed on a configurable grid:

```yaml
database:
  feature_dimensions: ["complexity", "diversity"]
  feature_bins: 10  # 10x10 = 100 cells
```

Islands evolve independently with periodic migration (ring topology), preventing premature convergence. Selection balances elite exploitation (70%), exploration (20%), and elite sampling (10%).

---

## Project Structure

```
ClaudeEvolve/
├── claude_evolve/                    # Python package (pip installable)
│   ├── claude_evolve/
│   │   ├── core/
│   │   │   ├── artifact.py           # Artifact dataclass
│   │   │   ├── database.py           # MAP-Elites + islands + novelty
│   │   │   ├── evaluator.py          # Subprocess-isolated evaluation
│   │   │   ├── stagnation.py         # Stagnation detection engine (v2)
│   │   │   ├── memory.py             # Cross-run memory (v2)
│   │   │   ├── research.py           # Research log management (v2)
│   │   │   ├── strategy.py           # Strategy evolver (v2)
│   │   │   ├── warm_cache.py         # Warm-start cache with LRU eviction (v3)
│   │   │   ├── novelty.py            # Universal novelty system (v3)
│   │   │   ├── improvement_signal.py # Continuous G_t signal (v4, AdaEvolve)
│   │   │   ├── ucb_selector.py       # UCB1 strategy selection (v4, ShinkaEvolve)
│   │   │   ├── reflection.py         # Verbal gradients engine (v4, ReEvo)
│   │   │   ├── scratchpad.py         # Meta-scratchpad synthesis (v4, ShinkaEvolve)
│   │   │   └── orchestrator.py       # IterationOrchestrator (v4)
│   │   ├── prompt/
│   │   │   ├── context_builder.py    # Per-iteration context generation
│   │   │   └── templates.py          # Template management
│   │   ├── state/
│   │   │   ├── manager.py            # State persistence (fresh init)
│   │   │   └── checkpoint.py         # Checkpoint management
│   │   ├── cli.py                    # CLI (init/next/submit/diagnose/seed/cache)
│   │   └── config.py                 # 6 sub-configs + master config
│   └── tests/                        # 1039 tests
├── plugin/                           # Claude Code plugin
│   ├── hooks/stop-hook.sh            # Evolution loop + stagnation
│   ├── skills/evolve/SKILL.md        # Iteration protocol + problem guidance
│   ├── agents/
│   │   ├── critic.md                 # Adversarial evaluation
│   │   ├── researcher.md             # Literature search (v2)
│   │   └── diagnostician.md          # Root cause analysis (v2)
│   └── commands/                     # /evolve, /evolve-status, /cancel-evolve
├── circle_packing/                   # Circle packing problem (n=26)
│   ├── program.py                    # Seed program
│   └── evaluator.py                  # Evaluator (target: sum_radii >= 2.636)
├── ramsey_R5_5/                      # Ramsey R(5,5) problem
│   ├── program.py                    # Seed program
│   └── evaluator.py                  # Evaluator (target: 0 mono-K_5 at n=43)
├── evolve_output/                    # Best artifacts from evolution runs
│   ├── best_circle_packing_strict.py # Circle packing result (2.635983)
│   └── best_circle_packing_final.py  # Final 15dp result (2.635982928557747)
├── docs/
│   ├── circle_packing_paper.md       # Paper: circle packing result
│   └── verify_circle_packing.py      # Independent verification script
├── install.sh
├── LICENSE                           # Apache 2.0
└── README.md
```

---

## Running Tests

```bash
cd claude_evolve
python -m pytest tests/ -v             # All 1039 tests
python -m pytest tests/ -q             # Quick summary
python -m pytest tests/test_database.py # Specific module
```

---

## Acknowledgments

Claude Evolve is inspired by [AlphaEvolve](https://deepmind.google/discover/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/) (Google DeepMind) and built upon the open-source [OpenEvolve](https://github.com/algorithmicsuperintelligence/openevolve) implementation. The loop mechanism is adapted from the [Ralph Loop](https://github.com/anthropics/claude-code-plugins/tree/main/ralph-loop) Claude Code plugin pattern.

The universal novelty system and stepping stones archive draw from research in automated discovery systems including [FunSearch](https://www.nature.com/articles/s41586-023-06924-6) (best-shot prompting, stepping stones), [ShinkaEvolve](https://github.com/XinmingTu/auto-discovery) (code novelty rejection sampling), [CodeEvolve](https://arxiv.org/abs/2410.12553) (inspiration-based crossover), and [DiscoPOP](https://arxiv.org/abs/2401.13385) (quality-diversity for prompt optimization).

---

## License

This project is licensed under the Apache License 2.0 — see the [LICENSE](LICENSE) file for details.
