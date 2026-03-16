"""
Stagnation detection and strategy recommendation for Claude Evolve.

Analyzes score history to detect when evolution has stalled and recommends
increasingly aggressive exploration strategies to escape local optima.

Stagnation levels:
  - NONE:     Score improving normally.
  - MILD:     3-5 iterations without improvement.
  - MODERATE: 6-10 iterations without improvement.
  - SEVERE:   11-20 iterations without improvement.
  - CRITICAL: 20+ iterations without improvement.

Each level maps to an exploration boost and a suggested prompt strategy
that the prompt builder can use to diversify candidate generation.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# StagnationLevel
# ---------------------------------------------------------------------------


class StagnationLevel(Enum):
    """Classification of how stagnant the evolutionary search has become."""

    NONE = "none"           # Score improving normally
    MILD = "mild"           # 3-5 iterations at same score
    MODERATE = "moderate"   # 6-10 iterations at same score
    SEVERE = "severe"       # 11-20 iterations at same score
    CRITICAL = "critical"   # 20+ iterations at same score


# ---------------------------------------------------------------------------
# StagnationReport
# ---------------------------------------------------------------------------


@dataclass
class StagnationReport:
    """Complete diagnostic report on the current stagnation state.

    Produced by ``StagnationEngine.analyze()`` and consumed by the prompt
    builder and evolution controller to adjust exploration parameters.
    """

    level: StagnationLevel
    iterations_stagnant: int
    best_score: float
    score_history: List[float]
    exploration_ratio_boost: float  # How much to increase exploration (0.0-0.5)
    suggested_strategy: str         # Template key for the suggested prompt strategy
    failed_approaches: List[str]    # List of approaches that didn't improve score
    diagnosis: str                  # Human-readable diagnosis text
    recommendations: List[str]      # Specific recommendations for next iteration


# ---------------------------------------------------------------------------
# StagnationConfig (lightweight, no dependency on main Config)
# ---------------------------------------------------------------------------


@dataclass
class StagnationConfig:
    """Configuration for the stagnation engine.

    All thresholds define the *minimum* number of stagnant iterations
    required to reach a given level.
    """

    mild_threshold: int = 3
    moderate_threshold: int = 6
    severe_threshold: int = 11
    critical_threshold: int = 20
    score_tolerance: float = 0.001
    exploration_boost_mild: float = 0.1
    exploration_boost_moderate: float = 0.2
    exploration_boost_severe: float = 0.3
    exploration_boost_critical: float = 0.5


# ---------------------------------------------------------------------------
# Strategy and recommendation mappings
# ---------------------------------------------------------------------------

_STRATEGY_MAP: Dict[StagnationLevel, str] = {
    StagnationLevel.NONE: "standard",
    StagnationLevel.MILD: "diversify",
    StagnationLevel.MODERATE: "paradigm_shift",
    StagnationLevel.SEVERE: "radical_departure",
    StagnationLevel.CRITICAL: "full_restart",
}

_BASE_RECOMMENDATIONS: Dict[StagnationLevel, List[str]] = {
    StagnationLevel.NONE: [
        "Continue with current approach.",
        "Consider minor variations on the best solution.",
    ],
    StagnationLevel.MILD: [
        "Try different variable names or code structure.",
        "Explore alternative algorithms for the same sub-problem.",
        "Increase randomness in template selection.",
    ],
    StagnationLevel.MODERATE: [
        "Shift to a fundamentally different algorithmic paradigm.",
        "Re-read the problem statement for missed constraints.",
        "Try decomposing the problem differently.",
        "Consider entirely different data structures.",
    ],
    StagnationLevel.SEVERE: [
        "Attempt a radical reimplementation from scratch.",
        "Combine ideas from the top two distinct solutions.",
        "Relax a constraint temporarily to explore new regions.",
        "Try a meta-heuristic approach (simulated annealing, genetic operators).",
        "Review failed approaches to identify common failure modes.",
    ],
    StagnationLevel.CRITICAL: [
        "Perform a full restart with a blank-slate solution.",
        "Re-examine the evaluation function for possible issues.",
        "Consider whether the problem formulation itself needs revision.",
        "Try an entirely different programming paradigm or language subset.",
        "Consult failed approaches log -- every prior strategy has been exhausted.",
        "Increase population diversity by injecting random seeds.",
    ],
}


# ---------------------------------------------------------------------------
# StagnationEngine
# ---------------------------------------------------------------------------


class StagnationEngine:
    """Analyzes score history to detect stagnation and recommend strategies.

    The engine counts how many iterations have elapsed since the last
    improvement (within a configurable tolerance) and maps that count to
    a ``StagnationLevel``.  Each level carries an exploration boost factor
    and a suggested prompt strategy key that downstream components can use
    to adjust their behavior.
    """

    def __init__(self, config: Optional[StagnationConfig] = None):
        """Initialize the stagnation engine.

        Args:
            config: Optional configuration overriding default thresholds
                    and boost values.  When ``None``, ``StagnationConfig()``
                    defaults are used.
        """
        cfg = config or StagnationConfig()
        self.mild_threshold: int = cfg.mild_threshold
        self.moderate_threshold: int = cfg.moderate_threshold
        self.severe_threshold: int = cfg.severe_threshold
        self.critical_threshold: int = cfg.critical_threshold
        self.score_tolerance: float = cfg.score_tolerance
        self.exploration_boost_mild: float = cfg.exploration_boost_mild
        self.exploration_boost_moderate: float = cfg.exploration_boost_moderate
        self.exploration_boost_severe: float = cfg.exploration_boost_severe
        self.exploration_boost_critical: float = cfg.exploration_boost_critical

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(
        self,
        score_history: List[float],
        failed_approaches: Optional[List[str]] = None,
    ) -> StagnationReport:
        """Analyze score history and produce a stagnation report.

        Args:
            score_history:     List of combined_score values in chronological
                               order (oldest first).
            failed_approaches: Optional list of approach descriptions that
                               did not improve the score.

        Returns:
            A ``StagnationReport`` with level, recommendations, and strategy.
        """
        if failed_approaches is None:
            failed_approaches = []

        if not score_history:
            return self._build_report(
                level=StagnationLevel.NONE,
                iterations_stagnant=0,
                best_score=0.0,
                score_history=[],
                failed_approaches=failed_approaches,
            )

        best_score = max(score_history)
        iterations_stagnant = self._count_stagnant_iterations(score_history)
        level = self._classify_level(iterations_stagnant)

        return self._build_report(
            level=level,
            iterations_stagnant=iterations_stagnant,
            best_score=best_score,
            score_history=list(score_history),
            failed_approaches=failed_approaches,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _count_stagnant_iterations(self, score_history: List[float]) -> int:
        """Count iterations since the last improvement.

        Walks backward from the end of the history to find the last index
        where the best score was achieved (within tolerance).  The stagnant
        count is the number of iterations *after* that index.
        """
        if len(score_history) < 2:
            return 0

        best = max(score_history)

        # Find the last index where best was achieved
        last_best_idx = 0
        for i in range(len(score_history) - 1, -1, -1):
            if abs(score_history[i] - best) < self.score_tolerance:
                last_best_idx = i
                break

        return len(score_history) - 1 - last_best_idx

    def _classify_level(self, stagnant_count: int) -> StagnationLevel:
        """Map a stagnant iteration count to a ``StagnationLevel``."""
        if stagnant_count >= self.critical_threshold:
            return StagnationLevel.CRITICAL
        if stagnant_count >= self.severe_threshold:
            return StagnationLevel.SEVERE
        if stagnant_count >= self.moderate_threshold:
            return StagnationLevel.MODERATE
        if stagnant_count >= self.mild_threshold:
            return StagnationLevel.MILD
        return StagnationLevel.NONE

    def _get_exploration_boost(self, level: StagnationLevel) -> float:
        """Get exploration ratio boost for a given stagnation level."""
        boost_map = {
            StagnationLevel.NONE: 0.0,
            StagnationLevel.MILD: self.exploration_boost_mild,
            StagnationLevel.MODERATE: self.exploration_boost_moderate,
            StagnationLevel.SEVERE: self.exploration_boost_severe,
            StagnationLevel.CRITICAL: self.exploration_boost_critical,
        }
        return boost_map[level]

    def _suggest_strategy(self, level: StagnationLevel) -> str:
        """Suggest a template strategy key based on stagnation level."""
        return _STRATEGY_MAP[level]

    def _generate_diagnosis(
        self,
        level: StagnationLevel,
        iterations_stagnant: int,
        best_score: float,
    ) -> str:
        """Generate a human-readable diagnosis text."""
        if level == StagnationLevel.NONE:
            return (
                f"Evolution is progressing normally. "
                f"Best score: {best_score:.4f}."
            )
        if level == StagnationLevel.MILD:
            return (
                f"Mild stagnation detected: no improvement in "
                f"{iterations_stagnant} iterations. Best score: {best_score:.4f}. "
                f"Consider diversifying the search."
            )
        if level == StagnationLevel.MODERATE:
            return (
                f"Moderate stagnation detected: no improvement in "
                f"{iterations_stagnant} iterations. Best score: {best_score:.4f}. "
                f"A paradigm shift in approach is recommended."
            )
        if level == StagnationLevel.SEVERE:
            return (
                f"Severe stagnation detected: no improvement in "
                f"{iterations_stagnant} iterations. Best score: {best_score:.4f}. "
                f"Radical departure from current approaches is needed."
            )
        # CRITICAL
        return (
            f"Critical stagnation detected: no improvement in "
            f"{iterations_stagnant} iterations. Best score: {best_score:.4f}. "
            f"All conventional strategies appear exhausted. "
            f"A full restart or problem reformulation is strongly recommended."
        )

    def _generate_recommendations(
        self,
        level: StagnationLevel,
        failed_approaches: List[str],
    ) -> List[str]:
        """Generate specific recommendations based on stagnation level and
        failed approaches.

        The base recommendations for the level are always included.  When
        failed approaches are provided, additional advice about avoiding
        those approaches is appended.
        """
        recommendations = list(_BASE_RECOMMENDATIONS[level])

        if failed_approaches:
            # Add a recommendation to explicitly avoid failed approaches
            if len(failed_approaches) <= 3:
                avoided = ", ".join(f'"{a}"' for a in failed_approaches)
                recommendations.append(
                    f"Avoid previously failed approaches: {avoided}."
                )
            else:
                recommendations.append(
                    f"Avoid the {len(failed_approaches)} previously failed "
                    f"approaches (see failed_approaches list for details)."
                )

            # At higher severity, call out the pattern
            if level in (StagnationLevel.SEVERE, StagnationLevel.CRITICAL):
                recommendations.append(
                    "Analyze common patterns across failed approaches to "
                    "identify systematic issues."
                )

        return recommendations

    def _build_report(
        self,
        level: StagnationLevel,
        iterations_stagnant: int,
        best_score: float,
        score_history: List[float],
        failed_approaches: List[str],
    ) -> StagnationReport:
        """Assemble a complete ``StagnationReport`` from computed values."""
        return StagnationReport(
            level=level,
            iterations_stagnant=iterations_stagnant,
            best_score=best_score,
            score_history=score_history,
            exploration_ratio_boost=self._get_exploration_boost(level),
            suggested_strategy=self._suggest_strategy(level),
            failed_approaches=list(failed_approaches),
            diagnosis=self._generate_diagnosis(level, iterations_stagnant, best_score),
            recommendations=self._generate_recommendations(level, failed_approaches),
        )
