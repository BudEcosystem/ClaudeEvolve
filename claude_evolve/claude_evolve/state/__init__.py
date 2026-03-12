"""
State management layer for claude_evolve.

Bridges the Ralph-style loop mechanism with the Python package.
Provides LoopState (evolve.local.md), StateManager (evolve-state/),
and CheckpointManager (save/restore snapshots).
"""

from claude_evolve.state.checkpoint import CheckpointManager
from claude_evolve.state.loop_state import LoopState
from claude_evolve.state.manager import StateManager

__all__ = ["CheckpointManager", "LoopState", "StateManager"]
