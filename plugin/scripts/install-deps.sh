#!/bin/bash
# Install claude_evolve Python package
# Called during plugin installation to ensure the CLI is available.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PLUGIN_ROOT="$(dirname "$SCRIPT_DIR")"
PACKAGE_DIR="$(dirname "$PLUGIN_ROOT")/claude_evolve"

# Check for jq (required by stop hook)
if ! command -v jq &>/dev/null; then
  echo "Warning: 'jq' is not installed. The evolution stop hook requires jq." >&2
  echo "  Install with: apt install jq  (Linux) / brew install jq  (macOS)" >&2
fi

if [[ -d "$PACKAGE_DIR" ]]; then
    echo "Installing claude-evolve from $PACKAGE_DIR..."
    pip install -e "$PACKAGE_DIR" --quiet
    echo "claude-evolve installed successfully"
else
    echo "Error: claude_evolve package not found at $PACKAGE_DIR" >&2
    echo "  Expected directory structure:" >&2
    echo "    <root>/claude_evolve/   (Python package)" >&2
    echo "    <root>/plugin/          (Claude Code plugin)" >&2
    exit 1
fi
