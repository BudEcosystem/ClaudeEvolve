---
description: "Check evolution progress"
allowed-tools: ["Bash(claude-evolve status:*)", "Read(.claude/evolve.local.md)", "Read(.claude/evolve-state/*)"]
---

# Evolution Status

Check evolution progress:

1. Run `claude-evolve status --state-dir .claude/evolve-state` to get JSON status
2. Read `.claude/evolve.local.md` for loop state (iteration count, target)
3. Present a clear summary to the user including:
   - Current iteration / max iterations
   - Best score achieved and target score
   - Population size and island statistics
   - Mode (script/critic/hybrid)
   - Time running (from started_at field)
