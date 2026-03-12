#!/bin/bash
# Install Claude Evolve: Python package + Claude Code plugin
#
# Usage:
#   bash install.sh          # Install with pip in current environment
#   bash install.sh --venv   # Create a venv first, then install
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Installing Claude Evolve..."
echo ""

# Check for jq dependency
if ! command -v jq &>/dev/null; then
  echo "Warning: 'jq' is not installed but is required by the evolution stop hook."
  echo "  Install with: apt install jq  (Linux) / brew install jq  (macOS)"
  echo ""
fi

# Optionally create a virtual environment
if [[ "${1:-}" == "--venv" ]]; then
  VENV_DIR="$SCRIPT_DIR/claude_evolve/.venv"
  if [[ ! -d "$VENV_DIR" ]]; then
    echo "Creating virtual environment at $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
  fi
  echo "Activating virtual environment..."
  source "$VENV_DIR/bin/activate"
fi

# 1. Install Python package
echo "Step 1: Installing claude_evolve Python package..."
cd "$SCRIPT_DIR/claude_evolve" && pip install -e ".[dev]" --quiet && cd "$SCRIPT_DIR"

# Verify CLI is available
if ! command -v claude-evolve &>/dev/null; then
  echo "Warning: 'claude-evolve' command not found on PATH after installation." >&2
  echo "  You may need to activate the virtual environment or add the install" >&2
  echo "  location to your PATH." >&2
fi

# 2. Register plugin with Claude Code
echo "Step 2: Registering Claude Code plugin..."
PLUGIN_DIR="$HOME/.claude/plugins/cache/claude-evolve-local/claude-evolve/0.1.0"
mkdir -p "$PLUGIN_DIR"
cp -r "$SCRIPT_DIR/plugin/"* "$PLUGIN_DIR/"

# Ensure shell scripts are executable
chmod +x "$PLUGIN_DIR/hooks/stop-hook.sh"
chmod +x "$PLUGIN_DIR/scripts/setup-evolve.sh"
chmod +x "$PLUGIN_DIR/scripts/install-deps.sh"

echo ""
echo "Claude Evolve installed successfully!"
echo ""
echo "  Python package: claude-evolve ($(claude-evolve --help 2>/dev/null | head -1 || echo 'installed'))"
echo "  Plugin location: $PLUGIN_DIR"
echo ""
echo "Usage:"
echo "  /evolve <artifact> <evaluator> [--max-iterations N] [--target-score F]"
echo ""
echo "See plugin/README.md for full documentation."
