"""
Cross-run memory for Claude Evolve.

Persists learnings, failed approaches, and successful strategies across
evolution runs so that future runs can avoid repeating mistakes and
build on past successes.
"""

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Learning:
    """A single learning from an evolution run."""
    id: str
    run_id: str
    iteration: int
    timestamp: float
    category: str  # "success", "failure", "insight", "approach"
    description: str
    score_before: Optional[float] = None
    score_after: Optional[float] = None
    score_delta: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Learning":
        valid_fields = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)


class CrossRunMemory:
    """Persists learnings across evolution runs.

    Storage layout:
        memory_dir/
        ├── learnings.json    # All learnings across runs
        ├── failed_approaches.json  # Approaches that didn't work
        └── strategies.json   # Successful strategies and their scores
    """

    def __init__(self, memory_dir: str, max_learnings: int = 100, max_failed_approaches: int = 50):
        self.memory_dir = memory_dir
        self.max_learnings = max_learnings
        self.max_failed_approaches = max_failed_approaches
        self.learnings: List[Learning] = []
        self.failed_approaches: List[Dict[str, Any]] = []
        self.strategies: List[Dict[str, Any]] = []
        self._loaded = False

    def load(self) -> None:
        """Load memory from disk."""
        os.makedirs(self.memory_dir, exist_ok=True)

        learnings_path = os.path.join(self.memory_dir, "learnings.json")
        if os.path.exists(learnings_path):
            with open(learnings_path, "r") as f:
                data = json.load(f)
                self.learnings = [Learning.from_dict(d) for d in data]

        failed_path = os.path.join(self.memory_dir, "failed_approaches.json")
        if os.path.exists(failed_path):
            with open(failed_path, "r") as f:
                self.failed_approaches = json.load(f)

        strategies_path = os.path.join(self.memory_dir, "strategies.json")
        if os.path.exists(strategies_path):
            with open(strategies_path, "r") as f:
                self.strategies = json.load(f)

        self._loaded = True
        logger.info("Loaded cross-run memory from %s (%d learnings, %d failed approaches, %d strategies)",
                     self.memory_dir, len(self.learnings), len(self.failed_approaches), len(self.strategies))

    def save(self) -> None:
        """Save memory to disk."""
        os.makedirs(self.memory_dir, exist_ok=True)

        # Enforce limits before saving
        self._enforce_limits()

        learnings_path = os.path.join(self.memory_dir, "learnings.json")
        with open(learnings_path, "w") as f:
            json.dump([l.to_dict() for l in self.learnings], f, indent=2)

        failed_path = os.path.join(self.memory_dir, "failed_approaches.json")
        with open(failed_path, "w") as f:
            json.dump(self.failed_approaches, f, indent=2)

        strategies_path = os.path.join(self.memory_dir, "strategies.json")
        with open(strategies_path, "w") as f:
            json.dump(self.strategies, f, indent=2)

        logger.info("Saved cross-run memory to %s", self.memory_dir)

    def add_learning(self, learning: Learning) -> None:
        """Add a learning entry."""
        self.learnings.append(learning)

    def add_failed_approach(self, description: str, score: float, iteration: int,
                            run_id: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Record a failed approach."""
        self.failed_approaches.append({
            "description": description,
            "score": score,
            "iteration": iteration,
            "run_id": run_id,
            "timestamp": time.time(),
            "metadata": metadata or {},
        })

    def add_strategy(self, name: str, description: str, score: float,
                     run_id: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Record a successful strategy."""
        self.strategies.append({
            "name": name,
            "description": description,
            "score": score,
            "run_id": run_id,
            "timestamp": time.time(),
            "metadata": metadata or {},
        })

    def get_failed_approaches(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get the most recent failed approaches."""
        return self.failed_approaches[-limit:]

    def get_successful_strategies(self, min_score: float = 0.0) -> List[Dict[str, Any]]:
        """Get strategies that achieved at least min_score."""
        return [s for s in self.strategies if s.get("score", 0.0) >= min_score]

    def get_learnings_by_category(self, category: str) -> List[Learning]:
        """Get learnings filtered by category."""
        return [l for l in self.learnings if l.category == category]

    def format_for_prompt(self, max_items: int = 5) -> str:
        """Format memory contents for inclusion in evolution prompts.

        Returns a markdown-formatted string with relevant learnings,
        failed approaches, and successful strategies.
        """
        sections = []

        # Failed approaches (most relevant for avoiding repeats)
        if self.failed_approaches:
            items = self.failed_approaches[-max_items:]
            lines = ["## Failed Approaches (Avoid These)", ""]
            for fa in items:
                lines.append(f"- **{fa['description']}** (score: {fa.get('score', 'N/A')}, run: {fa.get('run_id', 'unknown')})")
            sections.append("\n".join(lines))

        # Successful strategies
        if self.strategies:
            items = sorted(self.strategies, key=lambda s: s.get("score", 0.0), reverse=True)[:max_items]
            lines = ["## Successful Strategies", ""]
            for s in items:
                lines.append(f"- **{s['name']}**: {s['description']} (score: {s.get('score', 'N/A')})")
            sections.append("\n".join(lines))

        # Key insights
        insights = self.get_learnings_by_category("insight")
        if insights:
            items = insights[-max_items:]
            lines = ["## Key Insights from Previous Runs", ""]
            for ins in items:
                lines.append(f"- {ins.description}")
            sections.append("\n".join(lines))

        if not sections:
            return ""

        return "\n\n".join(sections)

    def _enforce_limits(self) -> None:
        """Trim collections to configured maximums, keeping most recent."""
        if len(self.learnings) > self.max_learnings:
            self.learnings = self.learnings[-self.max_learnings:]
        if len(self.failed_approaches) > self.max_failed_approaches:
            self.failed_approaches = self.failed_approaches[-self.max_failed_approaches:]
