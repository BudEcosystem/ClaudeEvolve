#!/bin/bash

# Claude Evolve Stop Hook
# Prevents session exit when an evolution loop is active.
# KEY DIFFERENCE from Ralph Loop: calls 'claude-evolve next' to generate
# a DYNAMIC per-iteration prompt instead of re-feeding the same prompt.

set -euo pipefail

# Dependency check: jq is required for JSON parsing
if ! command -v jq &>/dev/null; then
  echo "Error: 'jq' is required but not found. Install with: apt install jq / brew install jq" >&2
  exit 0  # Allow exit rather than blocking the session
fi

# Read hook input from stdin (advanced stop hook API)
HOOK_INPUT=$(cat)

# Check if evolve loop is active
EVOLVE_STATE_FILE=".claude/evolve.local.md"

if [[ ! -f "$EVOLVE_STATE_FILE" ]]; then
  # No active loop - allow exit
  exit 0
fi

# Parse markdown frontmatter (YAML between ---) and extract values
FRONTMATTER=$(sed -n '/^---$/,/^---$/{ /^---$/d; p; }' "$EVOLVE_STATE_FILE")
ITERATION=$(echo "$FRONTMATTER" | grep '^iteration:' | sed 's/iteration: *//')
MAX_ITERATIONS=$(echo "$FRONTMATTER" | grep '^max_iterations:' | sed 's/max_iterations: *//')
TARGET_SCORE=$(echo "$FRONTMATTER" | grep '^target_score:' | sed 's/target_score: *//')
STATE_DIR=$(echo "$FRONTMATTER" | grep '^state_dir:' | sed 's/state_dir: *//')
MODE=$(echo "$FRONTMATTER" | grep '^mode:' | sed 's/mode: *//')
BEST_SCORE=$(echo "$FRONTMATTER" | grep '^best_score:' | sed 's/best_score: *//')
# Extract completion_promise and strip surrounding quotes if present
COMPLETION_PROMISE=$(echo "$FRONTMATTER" | grep '^completion_promise:' | sed 's/completion_promise: *//' | sed 's/^"\(.*\)"$/\1/')

# Default state_dir if not set
STATE_DIR="${STATE_DIR:-.claude/evolve-state}"

# Session isolation: the state file is project-scoped, but the Stop hook
# fires in every Claude Code session in that project. If another session
# started the loop, this session must not block (or touch the state file).
# Legacy state files without session_id fall through (preserves old behavior).
STATE_SESSION=$(echo "$FRONTMATTER" | grep '^session_id:' | sed 's/session_id: *//' || true)
HOOK_SESSION=$(echo "$HOOK_INPUT" | jq -r '.session_id // ""')
if [[ -n "$STATE_SESSION" ]] && [[ "$STATE_SESSION" != "$HOOK_SESSION" ]]; then
  exit 0
fi

# Validate numeric fields before arithmetic operations
if [[ ! "$ITERATION" =~ ^[0-9]+$ ]]; then
  echo "Warning: Evolution loop state file corrupted" >&2
  echo "  File: $EVOLVE_STATE_FILE" >&2
  echo "  Problem: 'iteration' field is not a valid number (got: '$ITERATION')" >&2
  echo "  Evolution loop is stopping. Run /evolve again to start fresh." >&2
  rm -f "$EVOLVE_STATE_FILE"
  exit 0
fi

if [[ ! "$MAX_ITERATIONS" =~ ^[0-9]+$ ]]; then
  echo "Warning: Evolution loop state file corrupted" >&2
  echo "  File: $EVOLVE_STATE_FILE" >&2
  echo "  Problem: 'max_iterations' field is not a valid number (got: '$MAX_ITERATIONS')" >&2
  echo "  Evolution loop is stopping. Run /evolve again to start fresh." >&2
  rm -f "$EVOLVE_STATE_FILE"
  exit 0
fi

# Check if max iterations reached
if [[ $MAX_ITERATIONS -gt 0 ]] && [[ $ITERATION -ge $MAX_ITERATIONS ]]; then
  echo "Evolution complete: Max iterations ($MAX_ITERATIONS) reached."
  echo "Run 'claude-evolve export --state-dir $STATE_DIR --output evolve_output/best_artifact' to save the best."
  rm -f "$EVOLVE_STATE_FILE"
  exit 0
fi

# Get transcript path from hook input
TRANSCRIPT_PATH=$(echo "$HOOK_INPUT" | jq -r '.transcript_path')

if [[ ! -f "$TRANSCRIPT_PATH" ]]; then
  echo "Warning: Evolution loop: Transcript file not found" >&2
  echo "  Expected: $TRANSCRIPT_PATH" >&2
  echo "  This is unusual and may indicate a Claude Code internal issue." >&2
  echo "  Evolution loop is stopping." >&2
  rm -f "$EVOLVE_STATE_FILE"
  exit 0
fi

# Read last assistant message from transcript (JSONL format - one JSON per line)
# First check if there are any assistant messages
if ! grep -q '"role":"assistant"' "$TRANSCRIPT_PATH"; then
  echo "Warning: Evolution loop: No assistant messages found in transcript" >&2
  echo "  Transcript: $TRANSCRIPT_PATH" >&2
  echo "  Evolution loop is stopping." >&2
  rm -f "$EVOLVE_STATE_FILE"
  exit 0
fi

# Extract the most recent assistant text block.
# Claude Code writes each content block as its own JSONL line.
# Capped at the last 100 assistant lines to keep jq bounded.
LAST_LINES=$(grep '"role":"assistant"' "$TRANSCRIPT_PATH" | tail -n 100)
if [[ -z "$LAST_LINES" ]]; then
  echo "Warning: Evolution loop: Failed to extract assistant messages" >&2
  echo "  Evolution loop is stopping." >&2
  rm -f "$EVOLVE_STATE_FILE"
  exit 0
fi

# Parse the recent lines and pull out the final text block.
set +e
LAST_OUTPUT=$(echo "$LAST_LINES" | jq -rs '
  map(.message.content[]? | select(.type == "text") | .text) | last // ""
' 2>&1)
JQ_EXIT=$?
set -e

if [[ $JQ_EXIT -ne 0 ]]; then
  echo "Warning: Evolution loop: Failed to parse assistant message JSON" >&2
  echo "  Error: $LAST_OUTPUT" >&2
  echo "  Evolution loop is stopping." >&2
  rm -f "$EVOLVE_STATE_FILE"
  exit 0
fi

# Check for completion promise
if [[ "$COMPLETION_PROMISE" != "null" ]] && [[ -n "$COMPLETION_PROMISE" ]]; then
  # Extract text from <promise> tags using Perl for multiline support
  PROMISE_TEXT=$(echo "$LAST_OUTPUT" | perl -0777 -pe 's/.*?<promise>(.*?)<\/promise>.*/$1/s; s/^\s+|\s+$//g; s/\s+/ /g' 2>/dev/null || echo "")

  # Use = for literal string comparison (not pattern matching)
  if [[ -n "$PROMISE_TEXT" ]] && [[ "$PROMISE_TEXT" = "$COMPLETION_PROMISE" ]]; then
    echo "Evolution complete: Target reached! Detected <promise>$COMPLETION_PROMISE</promise>"
    echo "Run 'claude-evolve export --state-dir $STATE_DIR --output evolve_output/best_artifact' to save the best."
    rm -f "$EVOLVE_STATE_FILE"
    exit 0
  fi
fi

# Check target score: query claude-evolve status for current best
if [[ -n "$TARGET_SCORE" ]] && [[ "$TARGET_SCORE" != "null" ]]; then
  set +e
  STATUS_OUTPUT=$(claude-evolve status --state-dir "$STATE_DIR" 2>/dev/null)
  STATUS_EXIT=$?
  set -e

  if [[ $STATUS_EXIT -eq 0 ]] && [[ -n "$STATUS_OUTPUT" ]]; then
    CURRENT_BEST=$(echo "$STATUS_OUTPUT" | jq -r '.best_score // 0.0' 2>/dev/null || echo "0.0")

    # Compare using awk for float comparison
    TARGET_MET=$(awk -v best="$CURRENT_BEST" -v target="$TARGET_SCORE" 'BEGIN { print (best >= target) ? "1" : "0" }')
    if [[ "$TARGET_MET" == "1" ]]; then
      echo "Evolution complete: Target score $TARGET_SCORE met! Best: $CURRENT_BEST"
      echo "Run 'claude-evolve export --state-dir $STATE_DIR --output evolve_output/best_artifact' to save the best."
      rm -f "$EVOLVE_STATE_FILE"
      exit 0
    fi

    BEST_SCORE="$CURRENT_BEST"
  fi
fi

# Not complete - prepare next iteration
# KEY DIFFERENCE from Ralph Loop: generate a dynamic prompt via 'claude-evolve next'
NEXT_ITERATION=$((ITERATION + 1))

# Run claude-evolve next to prepare the next iteration context
set +e
NEXT_OUTPUT=$(claude-evolve next --state-dir "$STATE_DIR" 2>&1)
NEXT_EXIT=$?
set -e

# Read the iteration context from file (more reliable than stdout capture
# since the CLI might emit diagnostic text to stderr mixed in)
NEXT_PROMPT=""
if [[ -f "$STATE_DIR/iteration_context.md" ]]; then
  NEXT_PROMPT=$(cat "$STATE_DIR/iteration_context.md")
fi

# Fallback if iteration context is empty
if [[ -z "$NEXT_PROMPT" ]]; then
  if [[ $NEXT_EXIT -ne 0 ]]; then
    echo "Warning: claude-evolve next failed (exit $NEXT_EXIT): $NEXT_OUTPUT" >&2
  fi
  # Use a minimal fallback prompt
  NEXT_PROMPT="Evolution iteration $NEXT_ITERATION. Read the evaluator and previous results in $STATE_DIR/. Generate an improved candidate and submit with: claude-evolve submit --candidate <path> --state-dir $STATE_DIR"
fi

# Single atomic update: update frontmatter values AND replace body in one pass.
# We extract frontmatter line-by-line (between first and second ---) to avoid
# issues if the body contains a bare "---" line that would confuse sed ranges.
# NOTE: The YAML parsing (grep + sed above) is adequate for the controlled set
# of values written by setup-evolve.sh. Values with colons or special chars
# could break parsing if the state file format evolves.
TEMP_FILE="${EVOLVE_STATE_FILE}.tmp.$$"
{
  echo "---"
  # Read frontmatter lines (between first and second ---), updating values inline
  IN_FRONTMATTER=0
  while IFS= read -r line; do
    if [[ "$line" == "---" ]]; then
      IN_FRONTMATTER=$((IN_FRONTMATTER + 1))
      if [[ $IN_FRONTMATTER -ge 2 ]]; then
        break
      fi
      continue
    fi
    if [[ $IN_FRONTMATTER -eq 1 ]]; then
      # Update iteration
      if [[ "$line" =~ ^iteration:\ .* ]]; then
        echo "iteration: $NEXT_ITERATION"
      elif [[ "$line" =~ ^best_score:\ .* ]] && [[ -n "$BEST_SCORE" ]] && [[ "$BEST_SCORE" != "null" ]]; then
        echo "best_score: $BEST_SCORE"
      else
        echo "$line"
      fi
    fi
  done < "$EVOLVE_STATE_FILE"
  echo "---"
  echo ""
  echo "$NEXT_PROMPT"
} > "$TEMP_FILE"
mv "$TEMP_FILE" "$EVOLVE_STATE_FILE"

echo "Evolution: iteration $((NEXT_ITERATION - 1)) -> $NEXT_ITERATION (best: ${BEST_SCORE:-0.0})" >&2

# Build system message with iteration count and score info
if [[ -n "$TARGET_SCORE" ]] && [[ "$TARGET_SCORE" != "null" ]]; then
  SYSTEM_MSG="Evolution iteration $NEXT_ITERATION of $MAX_ITERATIONS | Best: ${BEST_SCORE:-0.0} | Target: $TARGET_SCORE | Submit improved candidates via: claude-evolve submit --candidate <path> --state-dir $STATE_DIR"
else
  SYSTEM_MSG="Evolution iteration $NEXT_ITERATION of $MAX_ITERATIONS | Best: ${BEST_SCORE:-0.0} | Submit improved candidates via: claude-evolve submit --candidate <path> --state-dir $STATE_DIR"
fi

# Output JSON to block the stop and feed the dynamic prompt
jq -n \
  --arg prompt "$NEXT_PROMPT" \
  --arg msg "$SYSTEM_MSG" \
  '{
    "decision": "block",
    "reason": $prompt,
    "systemMessage": $msg
  }'

# Exit 0 for successful hook execution
exit 0
