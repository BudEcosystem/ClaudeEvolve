---
description: "Start evolutionary optimization loop"
argument-hint: "<artifact> <evaluator> [--mode script|critic|hybrid] [--max-iterations N] [--target-score F] [--config path]"
allowed-tools: ["Bash(${CLAUDE_PLUGIN_ROOT}/scripts/setup-evolve.sh:*)"]
hide-from-slash-command-tool: "true"
---

# Evolve Command

Execute the setup script to initialize the evolution loop:

```!
"${CLAUDE_PLUGIN_ROOT}/scripts/setup-evolve.sh" $ARGUMENTS
```

You are now in an evolutionary optimization loop. Each iteration you will:

1. **Read** the iteration context from `.claude/evolve-state/iteration_context.md`
2. **Research** the problem space - study the evaluator, search for approaches, read prior results
3. **Generate** an improved candidate artifact
4. **Submit** via `claude-evolve submit --candidate <path> --state-dir .claude/evolve-state`
5. **Report** your approach and score

The stop hook generates a NEW dynamic prompt each iteration based on the evolving population. You will see different parent programs, inspiration from diverse solutions, and guidance on unexplored regions of the solution space.

CRITICAL RULES:
- NEVER modify the evaluator script
- NEVER modify files in `.claude/evolve-state/` directly
- ALWAYS submit candidates through `claude-evolve submit`
- ALWAYS validate your candidate before submitting
- Output `<promise>EVOLUTION_TARGET_REACHED</promise>` ONLY when the target score is genuinely met
- Do NOT output false promises to escape the loop
