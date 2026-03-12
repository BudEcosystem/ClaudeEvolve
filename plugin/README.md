# Claude Evolve Plugin

Evolutionary code and artifact optimization for Claude Code, powered by MAP-Elites quality-diversity search.

## Prerequisites

Install the `claude_evolve` Python package:

```bash
cd claude_evolve
pip install -e .
```

Verify the CLI is available:

```bash
claude-evolve --help
```

## Installation

Add this plugin to your Claude Code configuration. From your project root:

```bash
claude plugins add /path/to/plugin
```

Or symlink into your plugins directory:

```bash
ln -s /path/to/plugin ~/.claude/plugins/claude-evolve
```

## Usage

### Start an Evolution Loop

```bash
/evolve solution.py evaluator.py --max-iterations 100 --target-score 0.95
```

Arguments:
- `<artifact>` -- The file to evolve (program, prompt, config, etc.)
- `<evaluator>` -- The evaluator script (.py) or prompt (.md)
- `--mode script|critic|hybrid` -- Evaluation mode (default: script)
- `--max-iterations N` -- Maximum iterations (default: 50)
- `--target-score F` -- Stop early when this score is reached
- `--config path` -- Optional YAML config for advanced settings

### Check Progress

```bash
/evolve-status
```

### Cancel a Running Loop

```bash
/cancel-evolve
```

## How It Works

1. `/evolve` initializes a MAP-Elites population database and runs a baseline evaluation
2. Each iteration, Claude receives dynamic context about the population: parent to improve, top performers, diverse inspirations, and improvement guidance
3. Claude researches, generates an improved candidate, and submits it for evaluation
4. The stop hook intercepts session exit, calls `claude-evolve next` to prepare the next iteration context, and feeds it back as a new prompt
5. The loop continues until max iterations, target score, or manual cancellation

Unlike Ralph Loop (same prompt every iteration), Claude Evolve generates a NEW prompt each iteration based on the evolving population state.

## Evaluation Modes

- **script**: Evaluator is an executable script that outputs JSON metrics
- **critic**: Evaluator is a prompt; the critic agent evaluates candidates
- **hybrid**: Combines script evaluation with critic agent analysis

## File Structure

```
plugin/
  .claude-plugin/plugin.json   Plugin manifest
  hooks/hooks.json              Hook configuration
  hooks/stop-hook.sh            Stop hook (manages iteration lifecycle)
  scripts/setup-evolve.sh       Setup script (initializes evolution run)
  commands/evolve.md            /evolve command
  commands/evolve-status.md     /evolve-status command
  commands/cancel-evolve.md     /cancel-evolve command
  skills/evolve/SKILL.md        Evolution methodology skill
  agents/critic.md              Adversarial critic evaluation agent
```
