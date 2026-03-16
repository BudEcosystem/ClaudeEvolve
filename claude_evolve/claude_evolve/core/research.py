"""
Research log management for Claude Evolve.

Persists research findings from the researcher agent across iterations
and provides formatted output for injection into evolution prompts.
"""

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field, fields as dc_fields
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ResearchFinding:
    """A single research finding from the researcher agent."""

    id: str
    iteration: int
    timestamp: float
    approach_name: str
    description: str
    novelty: str  # "high", "medium", "low"
    implementation_hint: str = ""
    source_url: str = ""
    was_tried: bool = False
    outcome_score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResearchFinding":
        valid = {f.name for f in dc_fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in valid})


class ResearchLog:
    """Manages research findings across iterations.

    Storage: JSON file in the state directory.  The log persists
    findings, theoretical bounds, key papers, and approaches to avoid
    so that future iterations can leverage past research.
    """

    def __init__(self, log_path: str):
        self.log_path = log_path
        self.findings: List[ResearchFinding] = []
        self.theoretical_bounds: Dict[str, str] = {}
        self.key_papers: List[Dict[str, str]] = []
        self.approaches_to_avoid: List[str] = []

    def load(self) -> None:
        """Load research log from disk."""
        if not os.path.exists(self.log_path):
            return
        with open(self.log_path, "r") as f:
            data = json.load(f)
        self.findings = [ResearchFinding.from_dict(d) for d in data.get("findings", [])]
        self.theoretical_bounds = data.get("theoretical_bounds", {})
        self.key_papers = data.get("key_papers", [])
        self.approaches_to_avoid = data.get("approaches_to_avoid", [])

    def save(self) -> None:
        """Save research log to disk."""
        os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
        data = {
            "findings": [f.to_dict() for f in self.findings],
            "theoretical_bounds": self.theoretical_bounds,
            "key_papers": self.key_papers,
            "approaches_to_avoid": self.approaches_to_avoid,
        }
        with open(self.log_path, "w") as f:
            json.dump(data, f, indent=2)

    def add_finding(self, finding: ResearchFinding) -> None:
        """Add a research finding."""
        self.findings.append(finding)

    def mark_finding_tried(self, finding_id: str, score: float) -> None:
        """Mark a finding as tried with its outcome score."""
        for f in self.findings:
            if f.id == finding_id:
                f.was_tried = True
                f.outcome_score = score
                break

    def get_untried_findings(self, max_items: int = 5) -> List[ResearchFinding]:
        """Get findings that haven't been tried yet, sorted by novelty.

        Higher novelty findings are returned first (high > medium > low).
        """
        novelty_order = {"high": 0, "medium": 1, "low": 2}
        untried = [f for f in self.findings if not f.was_tried]
        untried.sort(key=lambda f: novelty_order.get(f.novelty, 3))
        return untried[:max_items]

    def should_research(self, iteration: int, stagnation_level: str, config) -> bool:
        """Determine if research should be triggered this iteration.

        Args:
            iteration: Current iteration number.
            stagnation_level: Current stagnation level string ("none", "mild", etc.)
            config: ResearchConfig instance.
        """
        if not config.enabled:
            return False

        if config.trigger == "never":
            return False
        elif config.trigger == "always":
            return True
        elif config.trigger == "periodic":
            return iteration % config.periodic_interval == 0 or iteration <= 3
        elif config.trigger == "on_stagnation":
            return stagnation_level != "none"

        return False

    def format_for_prompt(self, max_findings: int = 5) -> str:
        """Format research findings for inclusion in evolution prompts.

        Returns a Markdown-formatted string with untried findings,
        theoretical bounds, key papers, and approaches to avoid.
        Returns empty string if there is nothing to include.
        """
        sections = []

        # Untried findings (most actionable)
        untried = self.get_untried_findings(max_findings)
        if untried:
            lines = ["## Research Findings (Untried Approaches)", ""]
            for f in untried:
                lines.append(f"### {f.approach_name} (Novelty: {f.novelty})")
                lines.append(f"{f.description}")
                if f.implementation_hint:
                    lines.append(f"**Implementation hint:** {f.implementation_hint}")
                if f.source_url:
                    lines.append(f"**Source:** {f.source_url}")
                lines.append("")
            sections.append("\n".join(lines))

        # Theoretical bounds
        if self.theoretical_bounds:
            lines = ["## Theoretical Bounds", ""]
            for key, value in self.theoretical_bounds.items():
                lines.append(f"- **{key}:** {value}")
            sections.append("\n".join(lines))

        # Key papers
        if self.key_papers:
            lines = ["## Key Papers", ""]
            for paper in self.key_papers[:3]:
                lines.append(
                    f"- [{paper.get('title', 'Untitled')}]({paper.get('url', '#')})"
                    f" -- {paper.get('relevance', '')}"
                )
            sections.append("\n".join(lines))

        # Approaches to avoid
        if self.approaches_to_avoid:
            lines = ["## Approaches to Avoid (Research-Based)", ""]
            for approach in self.approaches_to_avoid:
                lines.append(f"- {approach}")
            sections.append("\n".join(lines))

        return "\n\n".join(sections) if sections else ""
