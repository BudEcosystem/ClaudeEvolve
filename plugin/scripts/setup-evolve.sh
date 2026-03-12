#!/bin/bash

# Claude Evolve Setup Script
# Initializes evolutionary optimization loop via claude-evolve CLI
# Creates state file for the stop hook to manage iteration lifecycle

set -euo pipefail

# Parse arguments
ARTIFACT=""
EVALUATOR=""
MODE=""
MAX_ITERATIONS=50
TARGET_SCORE=""
CONFIG_PATH=""
COMPLETION_PROMISE="EVOLUTION_TARGET_REACHED"
POSITIONAL_COUNT=0

while [[ $# -gt 0 ]]; do
  case $1 in
    -h|--help)
      cat << 'HELP_EOF'
Claude Evolve - Evolutionary artifact optimization loop

USAGE:
  /evolve <artifact> <evaluator> [OPTIONS]

ARGUMENTS:
  <artifact>     Path to the artifact file to evolve (program, prompt, config, etc.)
  <evaluator>    Path to the evaluator script (evaluator.py) or prompt (eval.md)

OPTIONS:
  --mode <script|critic|hybrid>  Evaluation mode (default: script)
                                  script: Evaluator is an executable script
                                  critic: Evaluator is a prompt for Claude-as-judge
                                  hybrid: Script + critic combined
  --max-iterations <n>           Maximum iterations before auto-stop (default: 50)
  --target-score <f>             Stop when best score >= this value (default: none)
  --completion-promise <text>    Promise tag text for completion detection
                                  (default: EVOLUTION_TARGET_REACHED)
  --config <path>                Optional YAML config file for advanced settings
  -h, --help                     Show this help message

DESCRIPTION:
  Starts an evolutionary optimization loop in your CURRENT session.
  Each iteration, Claude researches, generates an improved candidate,
  submits it for evaluation, and the stop hook prepares the next
  iteration context with MAP-Elites quality-diversity guidance.

  Unlike Ralph Loop (same prompt each iteration), Claude Evolve generates
  a DYNAMIC prompt per iteration based on the evolving population, parent
  selection, and diversity metrics.

EXAMPLES:
  /evolve solution.py evaluator.py --max-iterations 100 --target-score 0.95
  /evolve prompt.md eval_prompt.md --mode critic --max-iterations 30
  /evolve algorithm.py benchmark.py --config evolve_config.yaml

STOPPING:
  The loop stops when:
  - --max-iterations is reached
  - --target-score is met or exceeded
  - Claude outputs <promise>EVOLUTION_TARGET_REACHED</promise> (or custom --completion-promise)
  - You run /cancel-evolve

MONITORING:
  /evolve-status          View current evolution progress
  head -10 .claude/evolve.local.md    View loop state
HELP_EOF
      exit 0
      ;;
    --mode)
      if [[ -z "${2:-}" ]]; then
        echo "Error: --mode requires an argument (script, critic, or hybrid)" >&2
        exit 1
      fi
      case "$2" in
        script|critic|hybrid)
          MODE="$2"
          ;;
        *)
          echo "Error: --mode must be one of: script, critic, hybrid (got: $2)" >&2
          exit 1
          ;;
      esac
      shift 2
      ;;
    --max-iterations)
      if [[ -z "${2:-}" ]]; then
        echo "Error: --max-iterations requires a number argument" >&2
        exit 1
      fi
      if ! [[ "$2" =~ ^[0-9]+$ ]]; then
        echo "Error: --max-iterations must be a positive integer, got: $2" >&2
        exit 1
      fi
      MAX_ITERATIONS="$2"
      shift 2
      ;;
    --target-score)
      if [[ -z "${2:-}" ]]; then
        echo "Error: --target-score requires a numeric argument" >&2
        exit 1
      fi
      if ! [[ "$2" =~ ^[0-9]*\.?[0-9]+$ ]]; then
        echo "Error: --target-score must be a number, got: $2" >&2
        exit 1
      fi
      TARGET_SCORE="$2"
      shift 2
      ;;
    --config)
      if [[ -z "${2:-}" ]]; then
        echo "Error: --config requires a path argument" >&2
        exit 1
      fi
      CONFIG_PATH="$2"
      shift 2
      ;;
    --completion-promise)
      if [[ -z "${2:-}" ]]; then
        echo "Error: --completion-promise requires a text argument" >&2
        exit 1
      fi
      COMPLETION_PROMISE="$2"
      shift 2
      ;;
    *)
      # Positional arguments: first is artifact, second is evaluator
      if [[ $POSITIONAL_COUNT -eq 0 ]]; then
        ARTIFACT="$1"
        POSITIONAL_COUNT=1
      elif [[ $POSITIONAL_COUNT -eq 1 ]]; then
        EVALUATOR="$1"
        POSITIONAL_COUNT=2
      else
        echo "Error: Unexpected argument: $1" >&2
        echo "  Usage: /evolve <artifact> <evaluator> [OPTIONS]" >&2
        echo "  Run /evolve --help for details" >&2
        exit 1
      fi
      shift
      ;;
  esac
done

# Validate required arguments
if [[ -z "$ARTIFACT" ]]; then
  echo "Error: No artifact file provided" >&2
  echo "" >&2
  echo "  Usage: /evolve <artifact> <evaluator> [OPTIONS]" >&2
  echo "  Run /evolve --help for details" >&2
  exit 1
fi

if [[ -z "$EVALUATOR" ]]; then
  echo "Error: No evaluator file provided" >&2
  echo "" >&2
  echo "  Usage: /evolve <artifact> <evaluator> [OPTIONS]" >&2
  echo "  Run /evolve --help for details" >&2
  exit 1
fi

# Validate artifact file exists
if [[ ! -f "$ARTIFACT" ]]; then
  echo "Error: Artifact file not found: $ARTIFACT" >&2
  exit 1
fi

# Validate evaluator file exists
# In critic mode with .md evaluator, it is a prompt file (still must exist)
if [[ ! -f "$EVALUATOR" ]]; then
  echo "Error: Evaluator file not found: $EVALUATOR" >&2
  exit 1
fi

# Validate config file if provided
if [[ -n "$CONFIG_PATH" ]] && [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Error: Config file not found: $CONFIG_PATH" >&2
  exit 1
fi

# Auto-detect mode from evaluator extension if not explicitly set
if [[ -z "$MODE" ]]; then
  case "$EVALUATOR" in
    *.md|*.txt)
      MODE="critic"
      ;;
    *)
      MODE="script"
      ;;
  esac
fi

# Build claude-evolve init command
INIT_CMD=(claude-evolve init
  --artifact "$ARTIFACT"
  --evaluator "$EVALUATOR"
  --mode "$MODE"
  --state-dir .claude/evolve-state
  --max-iterations "$MAX_ITERATIONS"
)

if [[ -n "$TARGET_SCORE" ]]; then
  INIT_CMD+=(--target-score "$TARGET_SCORE")
fi

if [[ -n "$CONFIG_PATH" ]]; then
  INIT_CMD+=(--config "$CONFIG_PATH")
fi

# Run initialization
echo "Initializing evolution run..."
INIT_OUTPUT=$("${INIT_CMD[@]}" 2>&1)
INIT_EXIT=$?

if [[ $INIT_EXIT -ne 0 ]]; then
  echo "Error: claude-evolve init failed (exit code $INIT_EXIT):" >&2
  echo "$INIT_OUTPUT" >&2
  exit 1
fi

echo "$INIT_OUTPUT"

# Extract baseline score from init output if present; validate it is numeric
BASELINE_SCORE=$(echo "$INIT_OUTPUT" | jq -r '.baseline_score // 0.0' 2>/dev/null || echo "0.0")
if ! [[ "$BASELINE_SCORE" =~ ^[0-9]*\.?[0-9]+$ ]]; then
  BASELINE_SCORE="0.0"
fi

# Pre-create candidate workspace directory
mkdir -p .claude/evolve-workspace

# Generate the first iteration context
NEXT_OUTPUT=$(claude-evolve next --state-dir .claude/evolve-state 2>&1)
NEXT_EXIT=$?

CONTEXT_WARNING=""
if [[ $NEXT_EXIT -ne 0 ]]; then
  echo "Warning: Failed to generate initial iteration context:" >&2
  echo "$NEXT_OUTPUT" >&2
  CONTEXT_WARNING="WARNING: Initial iteration context generation failed. Using fallback prompt."
  NEXT_OUTPUT="Read .claude/evolve-state/iteration_context.md for your evolution task. Study the evaluator, understand the fitness landscape, and generate an improved candidate. Submit using: claude-evolve submit --candidate <path> --state-dir .claude/evolve-state"
fi

# Read the iteration context file (this is what Claude will work with)
ITERATION_CONTEXT=""
if [[ -f ".claude/evolve-state/iteration_context.md" ]]; then
  ITERATION_CONTEXT=$(cat ".claude/evolve-state/iteration_context.md")
fi

# Format target score for YAML
if [[ -n "$TARGET_SCORE" ]]; then
  TARGET_SCORE_YAML="$TARGET_SCORE"
else
  TARGET_SCORE_YAML="null"
fi

# Create state file for stop hook (markdown with YAML frontmatter)
mkdir -p .claude
cat > .claude/evolve.local.md <<EOF
---
active: true
iteration: 1
session_id: ${CLAUDE_CODE_SESSION_ID:-}
max_iterations: $MAX_ITERATIONS
target_score: $TARGET_SCORE_YAML
completion_promise: "$COMPLETION_PROMISE"
state_dir: .claude/evolve-state
mode: $MODE
started_at: "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
best_score: $BASELINE_SCORE
---

$ITERATION_CONTEXT
EOF

# Output setup confirmation
if [[ -n "$CONTEXT_WARNING" ]]; then
  echo ""
  echo "$CONTEXT_WARNING"
fi
cat <<EOF

Evolution loop activated!

  Artifact:        $ARTIFACT
  Evaluator:       $EVALUATOR
  Mode:            $MODE
  Iteration:       1 of $MAX_ITERATIONS
  Target score:    $(if [[ -n "$TARGET_SCORE" ]]; then echo "$TARGET_SCORE"; else echo "none (run until max iterations)"; fi)
  Baseline score:  $BASELINE_SCORE
  State directory: .claude/evolve-state

The stop hook is now active. After each iteration:
  1. You generate and submit an improved candidate
  2. The stop hook calls 'claude-evolve next' to prepare dynamic context
  3. You receive a NEW prompt with population insights, parent selection,
     and diversity guidance for the next iteration

Stopping conditions:
  - Max iterations ($MAX_ITERATIONS) reached
  $(if [[ -n "$TARGET_SCORE" ]]; then echo "- Best score >= $TARGET_SCORE"; fi)
  - Output <promise>$COMPLETION_PROMISE</promise> when target is met
  - Run /cancel-evolve to stop manually

To monitor: /evolve-status

Begin your first iteration now. Read the evolution context below and start
researching, generating, and submitting your first improved candidate.

EOF

# Output the iteration context as the initial prompt
if [[ -n "$ITERATION_CONTEXT" ]]; then
  echo "$ITERATION_CONTEXT"
fi
