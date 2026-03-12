# Claude Evolve

**Evolutionary artifact optimization for Claude Code** — an open-source implementation of [AlphaEvolve](https://deepmind.google/discover/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/)-style evolutionary search that runs natively inside [Claude Code](https://docs.anthropic.com/en/docs/claude-code/overview) sessions.

Claude Evolve uses **MAP-Elites quality-diversity search** with island-based populations to evolve programs, prompts, algorithms, configurations, and any text artifact — with Claude acting as both the intelligent mutation engine and an autonomous research agent.

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-387%20passing-brightgreen.svg)]()

---

## Table of Contents

- [How It Works](#how-it-works)
- [Key Features](#key-features)
- [Architecture](#architecture)
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
2. **An evaluator** — a script that scores candidates on 0.0–1.0 metrics, or a prompt for Claude-as-judge

Then Claude Evolve runs an evolution loop:

```
                     ┌─────────────────────────────────────────────┐
                     │            Evolution Loop                   │
                     │                                             │
  /evolve ──────►    │  1. Claude reads iteration context          │
                     │     (parent, inspirations, metrics, guidance)│
                     │                                             │
                     │  2. Claude researches the problem            │
                     │     (web search, literature, code analysis)  │
                     │                                             │
                     │  3. Claude generates improved candidate      │
                     │     (targeted diff or full rewrite)          │
                     │                                             │
                     │  4. Candidate is evaluated & scored          │
                     │                                             │
                     │  5. Result stored in MAP-Elites database     │
                     │     with diversity-aware population mgmt     │
                     │                                             │
                     │  6. Stop hook intercepts exit, generates     │
                     │     NEW dynamic prompt for next iteration    │
                     │                                             │
                     └────────────────┬────────────────────────────┘
                                      │
                                      ▼
                          Stops when target score met,
                          max iterations reached, or
                          manually cancelled
```

Each iteration, Claude receives **different context** — a new parent artifact selected from the population, inspiration from diverse high-performing solutions, and guidance on unexplored regions of the solution space. This is fundamentally different from simple re-prompting: the evolutionary algorithm guides exploration through quality-diversity search.

---

## Key Features

### Evolutionary Algorithm
- **MAP-Elites quality-diversity search** — maintains a population of diverse elite solutions across configurable feature dimensions (complexity, diversity, custom metrics)
- **Island-based evolution** — multiple isolated populations evolve independently with periodic migration, preventing premature convergence
- **Cascade evaluation** — multi-stage evaluation pipeline with threshold gates for efficient candidate filtering
- **Configurable selection** — tunable elite/exploration/exploitation ratios for parent sampling

### Claude Code Integration
- **Native plugin** — runs inside Claude Code sessions via `/evolve` command
- **Dynamic per-iteration prompts** — each iteration, Claude receives fresh context with population insights, parent selection, and diversity guidance (not the same prompt repeated)
- **Full autonomy per iteration** — Claude can use web search, spawn subagents, run code, review literature, and apply any available skill during each iteration
- **Critic mode** — Claude can act as an adversarial evaluator for non-code artifacts (prompts, configs, prose)

### Generalized Artifact Evolution
- **Not just code** — evolve Python, JavaScript, Rust, Go, prompts, YAML configs, SQL queries, Markdown documents, or any text artifact
- **Auto-detection** — artifact type automatically detected from file extension
- **Language-agnostic** — evaluator defines what "better" means; the system is agnostic to the artifact's content

### Production Quality
- **387 tests** covering unit, integration, and end-to-end flows
- **Subprocess isolation** — evaluator runs in isolated subprocess with timeout protection
- **Checkpoint/resume** — periodic snapshots with seamless resume capability
- **Session isolation** — multiple Claude Code sessions in the same project don't interfere
- **JSON state** — all state is human-readable JSON (no pickle)

---

## Architecture

Claude Evolve is a three-layer hybrid system:

```
┌──────────────────────────────────────────────────────────────────┐
│  Layer 3: Claude Code Skill                                      │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  SKILL.md — 6-phase iteration protocol                     │  │
│  │  Research → Generate → Validate → Submit → Report          │  │
│  │  + WebSearch, subagents, TDD, literature review            │  │
│  └────────────────────────────────────────────────────────────┘  │
├──────────────────────────────────────────────────────────────────┤
│  Layer 2: Python Package (claude_evolve)                         │
│  ┌──────────┐  ┌──────────┐  ┌───────────┐  ┌───────────────┐  │
│  │ Database  │  │Evaluator │  │  Context  │  │     CLI       │  │
│  │ MAP-Elites│  │Subprocess│  │  Builder  │  │ init/next/    │  │
│  │ + Islands │  │Isolation │  │  Prompts  │  │ submit/status │  │
│  └──────────┘  └──────────┘  └───────────┘  └───────────────┘  │
├──────────────────────────────────────────────────────────────────┤
│  Layer 1: Shell Loop (Stop Hook)                                 │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  stop-hook.sh — intercepts session exit                    │  │
│  │  Calls 'claude-evolve next' for dynamic prompt generation  │  │
│  │  Checks completion: max iterations / target score / promise│  │
│  └────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

**Layer 1 (Shell)** manages the iteration lifecycle via a Stop hook. When Claude tries to end a session, the hook intercepts, checks completion conditions, and if the loop should continue, calls `claude-evolve next` to generate a fresh prompt with new population context.

**Layer 2 (Python)** provides all deterministic logic — MAP-Elites database, evaluation, parent/inspiration selection, context building, and state management. This is a standalone `pip`-installable package with a `claude-evolve` CLI.

**Layer 3 (Skill)** teaches Claude *how* to behave during each iteration — research the problem, explore approaches, generate candidates, validate before submitting, and report results.

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

# Or install directly (if you manage your own environment)
bash install.sh
```

This installs the `claude-evolve` CLI and registers the Claude Code plugin.

### Verify Installation

```bash
claude-evolve --help
```

---

## Quick Start

### Example: Optimize a Circle Packing Algorithm

The repository includes a test problem — packing non-overlapping circles into a unit square to maximize the sum of radii.

```bash
cd ClaudeEvolve

# Start Claude Code in this directory
claude

# In Claude Code, run:
/evolve test_problem/program.py test_problem/evaluator.py --max-iterations 10 --target-score 0.9
```

Claude will:
1. Read the initial program and evaluator
2. Run a baseline evaluation (score ~0.75)
3. Research circle packing algorithms
4. Generate improved candidates each iteration
5. Submit candidates for evaluation
6. Continue until target score is met or max iterations reached

### Example: Evolve a Prompt with Critic Mode

For non-code artifacts, use critic mode where Claude evaluates candidates:

```bash
/evolve my_prompt.md eval_criteria.md --mode critic --max-iterations 20
```

The evaluator file (`eval_criteria.md`) defines what to evaluate:

```markdown
Evaluate this prompt for:
- clarity (0.0-1.0): Is it unambiguous?
- effectiveness (0.0-1.0): Will it reliably produce the desired output?
- conciseness (0.0-1.0): Is it appropriately brief?
```

### Example: Evolve with Custom Config

```bash
/evolve solution.py benchmark.py --config evolve_config.yaml --max-iterations 100
```

```yaml
# evolve_config.yaml
database:
  num_islands: 8
  feature_dimensions: ["complexity", "performance"]
  migration_interval: 25

evaluator:
  timeout: 600
  cascade_evaluation: true

evolution:
  diff_based: true
```

---

## Evaluation Modes

### Script Mode (Default)

The evaluator is a Python script that scores candidates via subprocess execution:

```python
# evaluator.py
import json, sys

def evaluate(candidate_path):
    # Load and test the candidate
    # Return metrics dict with scores in 0.0-1.0 range
    return {
        "combined_score": 0.85,
        "accuracy": 0.9,
        "efficiency": 0.8,
    }

if __name__ == "__main__":
    result = evaluate(sys.argv[1])
    print(json.dumps(result))
```

```bash
/evolve solution.py evaluator.py
```

### Critic Mode

The evaluator is a prompt file. Claude spawns an adversarial critic agent to evaluate candidates:

```bash
/evolve prompt.md eval_criteria.md --mode critic
```

The critic agent applies harsh, systematic analysis — finding flaws, testing edge cases, checking for gaming, and returning structured scores. This is ideal for evolving prompts, documentation, configurations, and other non-executable artifacts.

### Hybrid Mode

Combines script evaluation with critic feedback:

```bash
/evolve algorithm.py benchmark.py --mode hybrid
```

The script evaluator provides quantitative metrics, while the critic agent adds qualitative analysis.

### Auto-Detection

When `--mode` is not specified, the mode is auto-detected from the evaluator file extension:
- `.py`, `.sh`, `.js` → **script** mode
- `.md`, `.txt` → **critic** mode

---

## Configuration

Claude Evolve uses YAML configuration with sensible defaults. Pass a config file via `--config`:

### Database (MAP-Elites)

| Option | Default | Description |
|--------|---------|-------------|
| `population_size` | 1000 | Maximum population per island |
| `num_islands` | 5 | Number of isolated populations |
| `feature_dimensions` | `["complexity", "diversity"]` | Feature grid dimensions |
| `feature_bins` | 10 | Grid resolution per dimension |
| `migration_interval` | 50 | Iterations between migrations |
| `migration_rate` | 0.1 | Fraction of population that migrates |
| `elite_selection_ratio` | 0.1 | Fraction of parents from elite |
| `exploration_ratio` | 0.2 | Fraction of parents from unexplored regions |
| `exploitation_ratio` | 0.7 | Fraction of parents from high-fitness regions |
| `similarity_threshold` | 0.99 | Reject near-duplicate artifacts |

### Evaluator

| Option | Default | Description |
|--------|---------|-------------|
| `mode` | `"script"` | Evaluation mode: `script`, `critic`, `hybrid` |
| `timeout` | 300 | Evaluation timeout in seconds |
| `max_retries` | 3 | Retry count on evaluation failure |
| `cascade_evaluation` | `true` | Enable multi-stage evaluation |
| `cascade_thresholds` | `[0.5, 0.75, 0.9]` | Stage pass thresholds |
| `parallel_evaluations` | 1 | Concurrent evaluations |

### Evolution

| Option | Default | Description |
|--------|---------|-------------|
| `diff_based` | `true` | Use diff-based mutation (vs full rewrite) |
| `max_content_length` | 50000 | Maximum artifact size in bytes |

### Prompt

| Option | Default | Description |
|--------|---------|-------------|
| `num_top_programs` | 3 | Top performers shown in context |
| `num_diverse_programs` | 2 | Diverse inspirations shown |
| `use_template_stochasticity` | `true` | Randomize prompt templates |
| `include_artifacts` | `true` | Include eval artifacts in context |
| `suggest_simplification_after_chars` | 500 | Suggest simplification threshold |

### General

| Option | Default | Description |
|--------|---------|-------------|
| `max_iterations` | 50 | Maximum evolution iterations |
| `checkpoint_interval` | 100 | Iterations between checkpoints |
| `target_score` | `null` | Early stop threshold |
| `random_seed` | 42 | Reproducibility seed |

### Full Example Config

```yaml
max_iterations: 100
target_score: 0.95
checkpoint_interval: 20
artifact_type: python

database:
  num_islands: 8
  population_size: 500
  feature_dimensions: ["complexity", "accuracy", "efficiency"]
  feature_bins:
    complexity: 5
    accuracy: 10
    efficiency: 10
  migration_interval: 25
  migration_rate: 0.15
  elite_selection_ratio: 0.15
  exploration_ratio: 0.25
  exploitation_ratio: 0.60

evaluator:
  mode: script
  timeout: 600
  max_retries: 2
  cascade_evaluation: true
  cascade_thresholds: [0.3, 0.6, 0.85]

evolution:
  diff_based: true
  max_content_length: 100000

prompt:
  num_top_programs: 5
  num_diverse_programs: 3
  use_template_stochasticity: true
```

---

## CLI Reference

The `claude-evolve` CLI is the bridge between the plugin layer and the Python package.

### `claude-evolve init`

Initialize a new evolution run.

```bash
claude-evolve init \
  --artifact path/to/artifact.py \
  --evaluator path/to/evaluator.py \
  --mode script \
  --max-iterations 50 \
  --target-score 0.95 \
  --config config.yaml \
  --state-dir .claude/evolve-state
```

Outputs JSON: `{"status": "initialized", "population_size": 1, "baseline_score": 0.75}`

### `claude-evolve next`

Generate the next iteration context. Called by the stop hook.

```bash
claude-evolve next --state-dir .claude/evolve-state
```

Outputs the iteration context (Markdown) to stdout and writes it to `iteration_context.md` in the state directory.

### `claude-evolve submit`

Submit a candidate artifact for evaluation.

```bash
# Script mode (evaluator runs automatically)
claude-evolve submit \
  --candidate path/to/candidate.py \
  --state-dir .claude/evolve-state

# Critic mode (pass pre-computed metrics)
claude-evolve submit \
  --candidate path/to/candidate.md \
  --state-dir .claude/evolve-state \
  --metrics '{"combined_score": 0.85, "clarity": 0.9, "effectiveness": 0.8}'
```

Outputs JSON: `{"combined_score": 0.85, ..., "is_new_best": true}`

### `claude-evolve status`

Show evolution progress.

```bash
claude-evolve status --state-dir .claude/evolve-state
```

Outputs JSON with iteration count, best score, population size, target, and per-island statistics.

### `claude-evolve export`

Export the best artifact(s).

```bash
# Export single best
claude-evolve export --state-dir .claude/evolve-state --output best_solution.py

# Export top 5
claude-evolve export --state-dir .claude/evolve-state --output top_solutions.py --top-n 5
```

---

## Plugin Commands

### `/evolve`

Start an evolutionary optimization loop in your current Claude Code session.

```
/evolve <artifact> <evaluator> [OPTIONS]

Options:
  --mode <script|critic|hybrid>    Evaluation mode (default: auto-detect)
  --max-iterations <N>             Max iterations (default: 50)
  --target-score <F>               Stop when score >= F
  --completion-promise <text>      Custom completion tag
  --config <path>                  YAML config file
```

### `/evolve-status`

Check current evolution progress — iteration count, best score, population statistics.

### `/cancel-evolve`

Cancel the active evolution loop. Exports the best artifact found before stopping.

---

## MAP-Elites & Island Evolution

### What is MAP-Elites?

MAP-Elites (Multi-dimensional Archive of Phenotypic Elites) is a quality-diversity algorithm. Instead of searching for a single best solution, it maintains a **map of the best solution in each region** of a feature space. This produces a diverse collection of high-performing solutions.

In Claude Evolve, the feature space is defined by configurable dimensions:

```yaml
database:
  feature_dimensions: ["complexity", "diversity"]  # Default
  feature_bins: 10  # 10x10 = 100 cells in the grid
```

Each cell in the grid holds the best artifact for that combination of features. A simple but effective solution occupies a different cell than a complex but highly optimized one — both are preserved.

### Island-Based Evolution

To prevent premature convergence, Claude Evolve maintains multiple isolated populations (islands):

```
Island 0 ──────► Island 1 ──────► Island 2
    ▲                                  │
    │                                  ▼
Island 4 ◄────── Island 3 ◄──────────┘
```

- Each island evolves independently with its own feature grid
- Periodically, top performers **migrate** between islands (ring topology)
- Migration introduces genetic diversity without disrupting each island's evolutionary trajectory
- The migration interval and rate are configurable

### Selection Strategy

When selecting a parent for the next iteration, the system uses a configurable mix:

- **Elite selection** (10% default): Choose from the absolute best performers
- **Exploration** (20% default): Choose from underrepresented regions of the feature grid
- **Exploitation** (70% default): Choose from high-fitness regions near the current best

This balance ensures the system both improves on known good solutions and discovers novel approaches.

---

## Writing Evaluators

### Script Evaluator Template

```python
"""evaluator.py — Score candidates on your custom metrics."""
import json
import sys

def evaluate(candidate_path):
    """Evaluate a candidate artifact.

    Args:
        candidate_path: Path to the candidate file.

    Returns:
        dict with metric names mapped to float scores (0.0-1.0).
        Must include 'combined_score' (or it will be auto-computed
        as the average of all metrics).
    """
    # Load the candidate
    with open(candidate_path) as f:
        content = f.read()

    # Your evaluation logic here
    accuracy = test_accuracy(content)
    efficiency = test_efficiency(content)
    robustness = test_robustness(content)

    return {
        "combined_score": 0.5 * accuracy + 0.3 * efficiency + 0.2 * robustness,
        "accuracy": accuracy,
        "efficiency": efficiency,
        "robustness": robustness,
    }

if __name__ == "__main__":
    result = evaluate(sys.argv[1])
    print(json.dumps(result))
```

### Cascade Evaluation

For expensive evaluators, use cascade evaluation to filter bad candidates early:

```python
def evaluate_stage1(candidate_path):
    """Quick syntax and basic correctness check."""
    # Fast check — return low score to reject obviously bad candidates
    return {"combined_score": 0.0}  # or a real quick score

def evaluate_stage2(candidate_path):
    """Moderate test — run basic test cases."""
    return {"combined_score": 0.5}

def evaluate(candidate_path):
    """Full evaluation — comprehensive testing."""
    return {"combined_score": 0.85, "accuracy": 0.9, "speed": 0.8}
```

The system calls `evaluate_stage1` first. If the score exceeds the threshold (default 0.5), it proceeds to `evaluate_stage2`, then `evaluate`. This saves time by not fully evaluating clearly poor candidates.

### Critic Evaluator Template

```markdown
<!-- eval_criteria.md -->
Evaluate this candidate on the following dimensions:

- **correctness** (weight 0.35): Does it produce correct results?
- **efficiency** (weight 0.25): Time and space complexity
- **readability** (weight 0.20): Code clarity and documentation
- **robustness** (weight 0.20): Error handling and edge cases

Score each dimension 0.0 to 1.0. Be harsh — 0.9+ means exceptional.
```

---

## Advanced Usage

### Custom Feature Dimensions

Map solutions to a feature grid based on any evaluator metric:

```yaml
database:
  feature_dimensions: ["accuracy", "latency", "model_size"]
  feature_bins:
    accuracy: 20     # Fine-grained accuracy tracking
    latency: 10      # Coarser latency buckets
    model_size: 5    # Few model size categories
```

The evaluator must return these as numeric metrics. The system automatically bins continuous values into the grid.

### Resume from Checkpoint

Evolution state is automatically checkpointed. To resume:

```bash
# State persists in .claude/evolve-state/
# Just run /evolve again — it picks up where it left off
/evolve solution.py evaluator.py --max-iterations 200
```

### Environment Variable Support in Config

Config files support `${VAR}` environment variable expansion:

```yaml
evaluator:
  timeout: ${EVAL_TIMEOUT:-300}

output_dir: ${HOME}/evolve_results
```

### Custom Prompt Templates

Override the default prompt templates by providing a custom template directory:

```yaml
prompt:
  template_dir: ./my_templates/
```

Place `.txt` template files and a `fragments.json` in the directory. Templates cascade: custom templates override defaults, unspecified templates fall back to built-in defaults.

---

## Project Structure

```
ClaudeEvolve/
├── claude_evolve/                    # Python package
│   ├── claude_evolve/
│   │   ├── core/
│   │   │   ├── artifact.py           # Generalized artifact dataclass
│   │   │   ├── database.py           # MAP-Elites with island evolution
│   │   │   └── evaluator.py          # Subprocess-isolated evaluation
│   │   ├── prompt/
│   │   │   ├── context_builder.py    # Per-iteration context generation
│   │   │   ├── templates.py          # Template management
│   │   │   └── default_templates/    # Built-in prompt templates
│   │   ├── state/
│   │   │   ├── manager.py            # State persistence
│   │   │   ├── checkpoint.py         # Checkpoint management
│   │   │   └── loop_state.py         # Loop lifecycle state
│   │   ├── utils/
│   │   │   ├── code_utils.py         # Diff parsing and application
│   │   │   ├── metrics_utils.py      # Fitness computation
│   │   │   └── format_utils.py       # Formatting helpers
│   │   ├── cli.py                    # CLI entry points
│   │   └── config.py                 # Configuration system
│   ├── tests/                        # 387 tests
│   └── pyproject.toml
├── plugin/                           # Claude Code plugin
│   ├── .claude-plugin/
│   │   └── plugin.json               # Plugin manifest
│   ├── hooks/
│   │   ├── hooks.json                # Hook registration
│   │   └── stop-hook.sh              # Evolution loop mechanism
│   ├── scripts/
│   │   ├── setup-evolve.sh           # Loop initialization
│   │   └── install-deps.sh           # Dependency installer
│   ├── commands/
│   │   ├── evolve.md                 # /evolve command
│   │   ├── evolve-status.md          # /evolve-status command
│   │   └── cancel-evolve.md          # /cancel-evolve command
│   ├── skills/
│   │   └── evolve/
│   │       └── SKILL.md              # Evolution iteration protocol
│   └── agents/
│       └── critic.md                 # Adversarial evaluation agent
├── test_problem/                     # Example: circle packing
│   ├── program.py                    # Initial solution
│   └── evaluator.py                  # Fitness function
├── install.sh                        # One-command installer
├── LICENSE                           # Apache 2.0
└── README.md
```

---

## Supported Artifact Types

| Extension | Type | Example Use Case |
|-----------|------|-----------------|
| `.py` | Python | Algorithm optimization, ML models |
| `.js` / `.ts` | JavaScript/TypeScript | Frontend components, Node.js services |
| `.rs` | Rust | Systems programming, performance-critical code |
| `.go` | Go | Network services, concurrent systems |
| `.java` | Java | Enterprise applications |
| `.c` / `.cpp` | C/C++ | Low-level optimization |
| `.rb` | Ruby | Scripts, DSLs |
| `.sh` | Shell | Build scripts, automation |
| `.md` | Markdown | Prompts, documentation |
| `.yaml` / `.yml` | YAML | Configurations |
| `.json` | JSON | Data schemas, configs |
| `.sql` | SQL | Query optimization |
| `.txt` | Text | General text artifacts |

---

## Running Tests

```bash
cd claude_evolve

# Run all tests
python -m pytest tests/ -v

# Run specific test module
python -m pytest tests/test_database.py -v

# Run with coverage
python -m pytest tests/ --cov=claude_evolve
```

---

## Acknowledgments

Claude Evolve is inspired by [AlphaEvolve](https://deepmind.google/discover/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/) (Google DeepMind) and built upon the open-source [OpenEvolve](https://github.com/algorithmicsuperintelligence/openevolve) implementation. The loop mechanism is adapted from the [Ralph Loop](https://github.com/anthropics/claude-code-plugins/tree/main/ralph-loop) Claude Code plugin pattern.

---

## License

This project is licensed under the Apache License 2.0 — see the [LICENSE](LICENSE) file for details.
