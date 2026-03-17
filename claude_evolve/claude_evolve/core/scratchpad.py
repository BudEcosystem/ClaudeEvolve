"""Meta-scratchpad: periodic pattern synthesis from evolution history.

Inspired by ShinkaEvolve (ICLR 2026). Every N iterations, analyzes top artifacts
and recent failures to extract patterns that work, patterns that fail, and
recommended directions. Deterministic -- no LLM calls.
"""

import json
import logging
import os
from typing import Dict, List, Optional

from claude_evolve.core.artifact import Artifact
from claude_evolve.core.novelty import semantic_fingerprint

logger = logging.getLogger(__name__)


class MetaScratchpad:
    """Generates and persists accumulated insights from evolution history.

    The scratchpad is synthesized periodically (every ``synthesis_interval``
    iterations) from the current top artifacts and recent failure descriptions.
    Pattern extraction works as follows:

    * **Patterns That Work** -- concepts appearing in 2+ top-scoring artifacts.
    * **Patterns That Fail** -- concepts present only in failure descriptions
      and absent from all top artifacts.
    * **Score Trend** -- whether recent scores are improving or stagnating.

    All analysis is deterministic: it uses ``semantic_fingerprint()`` from the
    novelty module for concept extraction and plain set arithmetic for pattern
    classification.
    """

    def __init__(self, state_dir: str, synthesis_interval: int = 10) -> None:
        self.state_dir = state_dir
        self.synthesis_interval = synthesis_interval
        self._path = os.path.join(state_dir, "meta_scratchpad.json")

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------

    def should_synthesize(self, current_iteration: int) -> bool:
        """Return True if *current_iteration* is a positive multiple of the interval."""
        return current_iteration > 0 and current_iteration % self.synthesis_interval == 0

    # ------------------------------------------------------------------
    # Core synthesis
    # ------------------------------------------------------------------

    def synthesize(
        self,
        top_artifacts: List[Artifact],
        recent_scores: List[float],
        failed_approaches: List[str],
    ) -> str:
        """Generate structured scratchpad text from recent evolution history.

        Uses ``semantic_fingerprint()`` to extract concepts from each artifact,
        then computes set intersections / differences to surface patterns.

        Args:
            top_artifacts: Best-scoring artifacts from the current population.
            recent_scores: Ordered list of recent combined scores (oldest first).
            failed_approaches: Free-text descriptions of approaches that failed.

        Returns:
            A multi-line markdown-ish string suitable for prompt injection, or
            the empty string when no patterns can be extracted.
        """
        if not top_artifacts:
            return ""

        # Extract fingerprints from top artifacts (pass artifact_type for
        # type-aware concept extraction)
        top_fps: List[set] = [
            semantic_fingerprint(a.content, a.artifact_type) for a in top_artifacts
        ]
        all_top_concepts: set = set()
        for fp in top_fps:
            all_top_concepts.update(fp)

        # Patterns that work: concepts appearing in 2+ top artifacts
        concept_counts: Dict[str, int] = {}
        for fp in top_fps:
            for concept in fp:
                concept_counts[concept] = concept_counts.get(concept, 0) + 1
        working_patterns = sorted(
            [c for c, count in concept_counts.items() if count >= 2],
            key=lambda c: concept_counts[c],
            reverse=True,
        )[:10]

        # Patterns that fail: concepts from failure descriptions that are
        # *absent* from all top artifacts
        failed_concepts: set = set()
        for desc in failed_approaches:
            failed_concepts.update(semantic_fingerprint(desc))
        failing_patterns = sorted(failed_concepts - all_top_concepts)[:5]

        # Build scratchpad text
        sections: List[str] = []
        if working_patterns:
            items = ", ".join(working_patterns[:5])
            sections.append(f"**Patterns That Work:** {items}")
        if failing_patterns:
            items = ", ".join(failing_patterns[:5])
            sections.append(f"**Patterns That Fail:** {items}")
        if recent_scores and len(recent_scores) >= 2:
            trend = (
                "improving"
                if recent_scores[-1] > recent_scores[0]
                else "stagnating"
            )
            sections.append(
                f"**Score Trend:** {trend} "
                f"({recent_scores[0]:.4f} -> {recent_scores[-1]:.4f})"
            )

        result = "\n".join(sections) if sections else ""
        logger.debug("Synthesized scratchpad (%d chars)", len(result))
        return result

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def load(self) -> Optional[str]:
        """Load the current scratchpad content from persistent state.

        Returns:
            The stored scratchpad text, or ``None`` if nothing has been saved.
        """
        if os.path.exists(self._path):
            with open(self._path, encoding="utf-8") as f:
                data = json.load(f)
                return data.get("content", "")
        return None

    def save(self, content: str) -> None:
        """Persist *content* as the current scratchpad."""
        os.makedirs(os.path.dirname(self._path), exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump({"content": content}, f)
        logger.debug("Saved scratchpad to %s", self._path)
