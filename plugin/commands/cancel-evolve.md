---
description: "Cancel active evolution loop"
allowed-tools: ["Bash(test -f .claude/evolve.local.md:*)", "Bash(rm .claude/evolve.local.md)", "Read(.claude/evolve.local.md)", "Bash(claude-evolve status:*)", "Bash(claude-evolve export:*)"]
hide-from-slash-command-tool: "true"
---

# Cancel Evolution

Cancel the active evolution loop:

1. Check if `.claude/evolve.local.md` exists using Bash: `test -f .claude/evolve.local.md && echo "EXISTS" || echo "NOT_FOUND"`

2. **If NOT_FOUND**: Say "No active evolution loop found."

3. **If EXISTS**:
   - Read `.claude/evolve.local.md` to get the current iteration number and state
   - Run `claude-evolve status --state-dir .claude/evolve-state` for final statistics
   - Run `claude-evolve export --state-dir .claude/evolve-state --output evolve_output/best_artifact` to save the best artifact found so far
   - Delete state file: `rm .claude/evolve.local.md`
   - Report: "Cancelled evolution at iteration N. Best artifact (score: X) exported to evolve_output/"
