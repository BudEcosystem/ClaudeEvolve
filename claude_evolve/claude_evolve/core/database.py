"""
MAP-Elites ArtifactDatabase for Claude Evolve.

Extracted and adapted from OpenEvolve's ProgramDatabase. Implements
the MAP-Elites algorithm with island-based population evolution,
feature-grid placement, diversity caching, elite archive maintenance,
fitness-weighted sampling, and JSON-based persistence.

Terminology mapping from OpenEvolve:
    Program   -> Artifact
    code      -> content
    language  -> artifact_type
"""

import base64
import json
import logging
import os
import random
import threading
import time
import uuid
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from claude_evolve.core.artifact import Artifact
from claude_evolve.config import DatabaseConfig
from claude_evolve.utils.metrics_utils import (
    get_fitness_score,
    safe_numeric_average,
)

logger = logging.getLogger(__name__)


def _safe_sum_metrics(metrics: Dict[str, Any]) -> float:
    """Safely sum only numeric metric values, ignoring strings and other types."""
    numeric_values = [
        v
        for v in metrics.values()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    ]
    return sum(numeric_values) if numeric_values else 0.0


def _safe_avg_metrics(metrics: Dict[str, Any]) -> float:
    """Safely calculate average of only numeric metric values."""
    numeric_values = [
        v
        for v in metrics.values()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    ]
    return (
        sum(numeric_values) / max(1, len(numeric_values))
        if numeric_values
        else 0.0
    )


class ArtifactDatabase:
    """
    Database for storing and sampling artifacts during evolution.

    Combines the MAP-Elites algorithm with an island-based population
    model to maintain diversity.  Tracks the absolute best artifact
    separately so it is never lost.
    """

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def __init__(self, config: DatabaseConfig) -> None:
        self.config = config

        # In-memory artifact storage
        self.artifacts: Dict[str, Artifact] = {}

        # Per-island feature grids for MAP-Elites
        self.island_feature_maps: List[Dict[str, str]] = [
            {} for _ in range(config.num_islands)
        ]

        # Handle both int and dict types for feature_bins
        if isinstance(config.feature_bins, int):
            self.feature_bins = max(
                config.feature_bins,
                int(
                    pow(
                        config.archive_size,
                        1 / len(config.feature_dimensions),
                    )
                    + 0.99
                ),
            )
        else:
            self.feature_bins = 10  # Default fallback for backward compat

        # Island populations
        self.islands: List[Set[str]] = [
            set() for _ in range(config.num_islands)
        ]

        # Island management
        self.current_island: int = 0
        self.island_generations: List[int] = [0] * config.num_islands
        self.last_migration_generation: int = 0
        self.migration_interval: int = getattr(config, "migration_interval", 10)
        self.migration_rate: float = getattr(config, "migration_rate", 0.1)

        # Archive of elite artifacts
        self.archive: Set[str] = set()

        # Absolute best
        self.best_program_id: Optional[str] = None

        # Per-island best
        self.island_best_programs: List[Optional[str]] = [
            None
        ] * config.num_islands

        # Last iteration number (for resuming)
        self.last_iteration: int = 0

        # Prompt log
        self.prompts_by_program: Optional[
            Dict[str, Dict[str, Dict[str, str]]]
        ] = None

        # Reproducible sampling
        if config.random_seed is not None:
            random.seed(config.random_seed)
            logger.debug("Database: Set random seed to %s", config.random_seed)

        # Diversity caching infrastructure
        self.diversity_cache: Dict[int, Dict[str, Union[float, float]]] = {}
        self.diversity_cache_size: int = 1000
        self.diversity_reference_set: List[str] = []
        self.diversity_reference_size: int = getattr(
            config, "diversity_reference_size", 20
        )

        # Feature scaling infrastructure
        self.feature_stats: Dict[
            str, Dict[str, Union[float, float, List[float]]]
        ] = {}
        self.feature_scaling_method: str = "minmax"

        # Per-dimension bins
        if hasattr(config, "feature_bins") and isinstance(
            config.feature_bins, dict
        ):
            self.feature_bins_per_dim = config.feature_bins
        else:
            self.feature_bins_per_dim = {
                dim: self.feature_bins
                for dim in config.feature_dimensions
            }

        # Similarity threshold for novelty (embedding-based novelty is
        # skipped in claude_evolve; this controls the simple check)
        self.similarity_threshold = getattr(config, "similarity_threshold", 0.99)

        # Thread safety: RLock because migrate_programs() -> add() and
        # sample_from_island() -> sample() create reentrant call chains.
        self._lock = threading.RLock()

        # Load from disk if path provided
        if config.db_path and os.path.exists(config.db_path):
            self.load(config.db_path)

        logger.info(
            "Initialized artifact database with %d artifacts",
            len(self.artifacts),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(
        self,
        artifact: Artifact,
        iteration: Optional[int] = None,
        target_island: Optional[int] = None,
    ) -> str:
        """
        Add an artifact to the database.

        Args:
            artifact: Artifact to add.
            iteration: Current iteration (defaults to last_iteration).
            target_island: Specific island to place into.

        Returns:
            Artifact ID.
        """
        with self._lock:
            if iteration is not None:
                artifact.iteration_found = iteration
                self.last_iteration = max(self.last_iteration, iteration)

            self.artifacts[artifact.id] = artifact

            # MAP-Elites feature coordinates
            feature_coords = self._calculate_feature_coords(artifact)

            # Determine target island
            if target_island is None and artifact.parent_id:
                parent = self.artifacts.get(artifact.parent_id)
                if parent and "island" in parent.metadata:
                    island_idx = parent.metadata["island"]
                else:
                    island_idx = self.current_island
            elif target_island is not None:
                island_idx = target_island
            else:
                island_idx = self.current_island

            island_idx = island_idx % len(self.islands)

            # Novelty gate
            if not self._is_novel(artifact.id, island_idx):
                logger.debug(
                    "Artifact %s failed novelty check for island %d",
                    artifact.id,
                    island_idx,
                )
                return artifact.id

            # Feature-map placement (replace if better)
            feature_key = self._feature_coords_to_key(feature_coords)
            island_feature_map = self.island_feature_maps[island_idx]
            should_replace = feature_key not in island_feature_map

            if not should_replace:
                existing_id = island_feature_map[feature_key]
                if existing_id not in self.artifacts:
                    should_replace = True
                else:
                    should_replace = self._is_better(
                        artifact, self.artifacts[existing_id]
                    )

            if should_replace:
                if feature_key in island_feature_map:
                    existing_id = island_feature_map[feature_key]
                    if existing_id in self.artifacts:
                        # Maintain archive consistency
                        if existing_id in self.archive:
                            self.archive.discard(existing_id)
                            self.archive.add(artifact.id)
                    # Remove replaced artifact from island set
                    self.islands[island_idx].discard(existing_id)

                island_feature_map[feature_key] = artifact.id

            # Add to island
            self.islands[island_idx].add(artifact.id)

            # Track island membership
            artifact.metadata["island"] = island_idx

            # Update archive
            self._update_archive(artifact)

            # Enforce population size limit (protect newly added)
            self._enforce_population_limit(exclude_program_id=artifact.id)

            # Update best tracking
            self._update_best(artifact)
            self._update_island_best(artifact, island_idx)

            # Persist if configured
            if self.config.db_path:
                self._save_artifact(artifact)

            logger.debug("Added artifact %s to island %d", artifact.id, island_idx)
            return artifact.id

    def get(self, artifact_id: str) -> Optional[Artifact]:
        """Retrieve an artifact by ID, or ``None``."""
        return self.artifacts.get(artifact_id)

    def remove(self, artifact_id: str) -> bool:
        """Remove an artifact and clean all internal references.

        Removes the artifact from the main store, all island sets,
        island feature maps, and the archive.  Updates island-best
        and global-best tracking when the removed artifact held those
        positions.

        Args:
            artifact_id: ID of the artifact to remove.

        Returns:
            ``True`` if the artifact was found and removed, ``False``
            if the ID was not present.
        """
        with self._lock:
            if artifact_id not in self.artifacts:
                return False

            del self.artifacts[artifact_id]

            # Clean from all island sets
            for island in self.islands:
                island.discard(artifact_id)

            # Clean from island feature maps
            for island_map in self.island_feature_maps:
                cells_to_clean = [
                    cell for cell, aid in island_map.items()
                    if aid == artifact_id
                ]
                for cell in cells_to_clean:
                    del island_map[cell]

            # Clean from archive
            self.archive.discard(artifact_id)

            # Repair global best if it pointed to the removed artifact
            if self.best_program_id == artifact_id:
                self.best_program_id = None
                if self.artifacts:
                    best = max(
                        self.artifacts.values(),
                        key=lambda a: get_fitness_score(
                            a.metrics, self.config.feature_dimensions
                        ),
                    )
                    self.best_program_id = best.id

            # Repair per-island bests
            for idx, best_pid in enumerate(self.island_best_programs):
                if best_pid == artifact_id:
                    self.island_best_programs[idx] = None

            return True

    def sample(
        self, num_inspirations: Optional[int] = None
    ) -> Tuple[Artifact, List[Artifact]]:
        """
        Sample a parent and inspiration artifacts.

        Args:
            num_inspirations: How many inspirations (default 5).

        Returns:
            Tuple of (parent, inspirations).
        """
        with self._lock:
            parent = self._sample_parent()
            if num_inspirations is None:
                num_inspirations = 5
            inspirations = self._sample_inspirations(parent, n=num_inspirations)
            logger.debug(
                "Sampled parent %s and %d inspirations",
                parent.id,
                len(inspirations),
            )
            return parent, inspirations

    def sample_from_island(
        self,
        island_id: int,
        num_inspirations: Optional[int] = None,
    ) -> Tuple[Artifact, List[Artifact]]:
        """
        Sample from a specific island without modifying ``current_island``.

        Args:
            island_id: Island index.
            num_inspirations: How many inspirations (default 5).

        Returns:
            Tuple of (parent, inspirations).
        """
        with self._lock:
            island_id = island_id % len(self.islands)
            island_programs = list(self.islands[island_id])

            if not island_programs:
                logger.debug(
                    "Island %d is empty, falling back to global sample",
                    island_id,
                )
                return self.sample(num_inspirations)

            rand_val = random.random()

            if rand_val < self.config.exploration_ratio:
                parent = self._sample_from_island_random(island_id)
            elif rand_val < (
                self.config.exploration_ratio + self.config.exploitation_ratio
            ):
                parent = self._sample_from_archive_for_island(island_id)
            else:
                parent = self._sample_from_island_weighted(island_id)

            if num_inspirations is None:
                num_inspirations = 5

            other = [pid for pid in island_programs if pid != parent.id]
            if len(other) < num_inspirations:
                inspiration_ids = other
            else:
                inspiration_ids = random.sample(other, num_inspirations)

            inspirations = [
                self.artifacts[pid]
                for pid in inspiration_ids
                if pid in self.artifacts
            ]
            return parent, inspirations

    def get_best(self, metric: Optional[str] = None) -> Optional[Artifact]:
        """
        Get the best artifact.

        Args:
            metric: If given, rank by that specific metric key.

        Returns:
            Best artifact or ``None``.
        """
        if not self.artifacts:
            return None

        if metric is None and self.best_program_id:
            if self.best_program_id in self.artifacts:
                return self.artifacts[self.best_program_id]
            else:
                self.best_program_id = None

        if metric:
            sorted_artifacts = sorted(
                [a for a in self.artifacts.values() if metric in a.metrics],
                key=lambda a: a.metrics[metric],
                reverse=True,
            )
        else:
            sorted_artifacts = sorted(
                self.artifacts.values(),
                key=lambda a: get_fitness_score(
                    a.metrics, self.config.feature_dimensions
                ),
                reverse=True,
            )

        if sorted_artifacts:
            if (
                self.best_program_id is None
                or sorted_artifacts[0].id != self.best_program_id
            ):
                self.best_program_id = sorted_artifacts[0].id
            return sorted_artifacts[0]
        return None

    def get_top_programs(
        self,
        n: int = 10,
        metric: Optional[str] = None,
        island_idx: Optional[int] = None,
    ) -> List[Artifact]:
        """
        Get the top *n* artifacts ranked by fitness.

        Args:
            n: Number of artifacts.
            metric: Specific metric to rank by.
            island_idx: Restrict to a specific island.

        Returns:
            Sorted list of artifacts (best first).
        """
        if island_idx is not None and (
            island_idx < 0 or island_idx >= len(self.islands)
        ):
            raise IndexError(
                f"Island index {island_idx} out of range "
                f"(0-{len(self.islands) - 1})"
            )

        if not self.artifacts:
            return []

        if island_idx is not None:
            candidates = [
                self.artifacts[pid]
                for pid in self.islands[island_idx]
                if pid in self.artifacts
            ]
        else:
            candidates = list(self.artifacts.values())

        if not candidates:
            return []

        if metric:
            sorted_artifacts = sorted(
                [a for a in candidates if metric in a.metrics],
                key=lambda a: a.metrics[metric],
                reverse=True,
            )
        else:
            sorted_artifacts = sorted(
                candidates,
                key=lambda a: get_fitness_score(
                    a.metrics, self.config.feature_dimensions
                ),
                reverse=True,
            )

        return sorted_artifacts[:n]

    def size(self) -> int:
        """Return total number of stored artifacts."""
        return len(self.artifacts)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Optional[str] = None, iteration: int = 0) -> None:
        """
        Save the database to *path* (JSON, no pickle).

        Args:
            path: Directory to save into (falls back to ``config.db_path``).
            iteration: Current iteration number.
        """
        save_path = path or self.config.db_path
        if not save_path:
            logger.warning("No database path specified, skipping save")
            return

        os.makedirs(save_path, exist_ok=True)

        # Save each artifact
        for artifact in self.artifacts.values():
            prompts = None
            if (
                self.config.log_prompts
                and self.prompts_by_program
                and artifact.id in self.prompts_by_program
            ):
                prompts = self.prompts_by_program[artifact.id]
            self._save_artifact(artifact, save_path, prompts=prompts)

        # Save metadata
        metadata = {
            "island_feature_maps": self.island_feature_maps,
            "islands": [list(island) for island in self.islands],
            "archive": list(self.archive),
            "best_program_id": self.best_program_id,
            "island_best_programs": self.island_best_programs,
            "last_iteration": iteration or self.last_iteration,
            "current_island": self.current_island,
            "island_generations": self.island_generations,
            "last_migration_generation": self.last_migration_generation,
            "feature_stats": self._serialize_feature_stats(),
        }

        with open(os.path.join(save_path, "metadata.json"), "w") as f:
            json.dump(metadata, f)

        logger.info(
            "Saved database with %d artifacts to %s",
            len(self.artifacts),
            save_path,
        )

    def load(self, path: str) -> None:
        """
        Load the database from *path*.

        Args:
            path: Directory to load from.
        """
        if not os.path.exists(path):
            logger.warning("Database path %s does not exist, skipping load", path)
            return

        # Load metadata
        metadata_path = os.path.join(path, "metadata.json")
        saved_islands: List[List[str]] = []
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            self.island_feature_maps = metadata.get(
                "island_feature_maps",
                [{} for _ in range(self.config.num_islands)],
            )
            saved_islands = metadata.get("islands", [])
            self.archive = set(metadata.get("archive", []))
            self.best_program_id = metadata.get("best_program_id")
            self.island_best_programs = metadata.get(
                "island_best_programs", [None] * len(saved_islands)
            )
            self.last_iteration = metadata.get("last_iteration", 0)
            self.current_island = metadata.get("current_island", 0)
            self.island_generations = metadata.get(
                "island_generations", [0] * len(saved_islands)
            )
            self.last_migration_generation = metadata.get(
                "last_migration_generation", 0
            )
            self.feature_stats = self._deserialize_feature_stats(
                metadata.get("feature_stats", {})
            )

        # Load artifacts
        programs_dir = os.path.join(path, "programs")
        if os.path.exists(programs_dir):
            for filename in os.listdir(programs_dir):
                if filename.endswith(".json"):
                    filepath = os.path.join(programs_dir, filename)
                    try:
                        with open(filepath, "r") as f:
                            data = json.load(f)
                        artifact = Artifact.from_dict(data)
                        self.artifacts[artifact.id] = artifact
                    except Exception as e:
                        logger.warning(
                            "Error loading artifact %s: %s", filename, e
                        )

        self._reconstruct_islands(saved_islands)

        # Ensure list lengths match
        if len(self.island_generations) != len(self.islands):
            self.island_generations = [0] * len(self.islands)
        if len(self.island_best_programs) != len(self.islands):
            self.island_best_programs = [None] * len(self.islands)

        logger.info(
            "Loaded database with %d artifacts from %s",
            len(self.artifacts),
            path,
        )

    # ------------------------------------------------------------------
    # Feature mapping & MAP-Elites
    # ------------------------------------------------------------------

    def _calculate_feature_coords(self, artifact: Artifact) -> List[int]:
        """Map an artifact onto the MAP-Elites feature grid."""
        coords: List[int] = []

        for dim in self.config.feature_dimensions:
            # Priority 1: custom metric from evaluator
            if dim in artifact.metrics:
                score = artifact.metrics[dim]
                self._update_feature_stats(dim, score)
                scaled = self._scale_feature_value(dim, score)
                num_bins = self.feature_bins_per_dim.get(dim, self.feature_bins)
                bin_idx = int(scaled * num_bins)
                bin_idx = max(0, min(num_bins - 1, bin_idx))
                coords.append(bin_idx)
            elif dim == "complexity":
                complexity = len(artifact.content)
                bin_idx = self._calculate_complexity_bin(complexity)
                coords.append(bin_idx)
            elif dim == "diversity":
                if len(self.artifacts) < 2:
                    bin_idx = 0
                else:
                    diversity = self._get_cached_diversity(artifact)
                    bin_idx = self._calculate_diversity_bin(diversity)
                coords.append(bin_idx)
            elif dim == "score":
                if not artifact.metrics:
                    bin_idx = 0
                else:
                    avg_score = get_fitness_score(
                        artifact.metrics, self.config.feature_dimensions
                    )
                    self._update_feature_stats("score", avg_score)
                    scaled = self._scale_feature_value("score", avg_score)
                    num_bins = self.feature_bins_per_dim.get(
                        "score", self.feature_bins
                    )
                    bin_idx = int(scaled * num_bins)
                    bin_idx = max(0, min(num_bins - 1, bin_idx))
                coords.append(bin_idx)
            else:
                raise ValueError(
                    f"Feature dimension '{dim}' not found in artifact metrics. "
                    f"Available metrics: {list(artifact.metrics.keys())}. "
                    f"Built-in features: 'complexity', 'diversity', 'score'."
                )

        return coords

    def _calculate_complexity_bin(self, complexity: int) -> int:
        """Bin a complexity value into ``[0, bins-1]``."""
        self._update_feature_stats("complexity", float(complexity))
        scaled = self._scale_feature_value("complexity", float(complexity))
        num_bins = self.feature_bins_per_dim.get("complexity", self.feature_bins)
        bin_idx = int(scaled * num_bins)
        return max(0, min(num_bins - 1, bin_idx))

    def _calculate_diversity_bin(self, diversity: float) -> int:
        """Bin a diversity value into ``[0, bins-1]``."""
        self._update_feature_stats("diversity", diversity)
        scaled = self._scale_feature_value("diversity", diversity)
        num_bins = self.feature_bins_per_dim.get("diversity", self.feature_bins)
        bin_idx = int(scaled * num_bins)
        return max(0, min(num_bins - 1, bin_idx))

    def _feature_coords_to_key(self, coords: List[int]) -> str:
        """Convert feature coordinates to a hashable string key."""
        return "-".join(str(c) for c in coords)

    # ------------------------------------------------------------------
    # Feature scaling
    # ------------------------------------------------------------------

    def _update_feature_stats(self, feature_name: str, value: float) -> None:
        """Incorporate *value* into running min/max/values for *feature_name*."""
        if feature_name not in self.feature_stats:
            self.feature_stats[feature_name] = {
                "min": value,
                "max": value,
                "values": [],
            }

        stats = self.feature_stats[feature_name]
        stats["min"] = min(stats["min"], value)
        stats["max"] = max(stats["max"], value)
        stats["values"].append(value)
        if len(stats["values"]) > 1000:
            stats["values"] = stats["values"][-1000:]

    def _scale_feature_value(self, feature_name: str, value: float) -> float:
        """Scale *value* to ``[0, 1]`` using configured method."""
        if feature_name not in self.feature_stats:
            return min(1.0, max(0.0, value))

        stats = self.feature_stats[feature_name]

        if self.feature_scaling_method == "minmax":
            min_val = stats["min"]
            max_val = stats["max"]
            if max_val == min_val:
                return 0.5
            scaled = (value - min_val) / (max_val - min_val)
            return min(1.0, max(0.0, scaled))

        elif self.feature_scaling_method == "percentile":
            values = stats["values"]
            if not values:
                return 0.5
            count = sum(1 for v in values if v <= value)
            return count / len(values)

        else:
            # Fallback to minmax
            return self._scale_feature_value_minmax(feature_name, value)

    def _scale_feature_value_minmax(
        self, feature_name: str, value: float
    ) -> float:
        """Helper for explicit min-max scaling."""
        if feature_name not in self.feature_stats:
            return min(1.0, max(0.0, value))
        stats = self.feature_stats[feature_name]
        min_val = stats["min"]
        max_val = stats["max"]
        if max_val == min_val:
            return 0.5
        scaled = (value - min_val) / (max_val - min_val)
        return min(1.0, max(0.0, scaled))

    def _serialize_feature_stats(self) -> Dict[str, Any]:
        """Prepare feature_stats for JSON serialization."""
        serialized: Dict[str, Any] = {}
        for name, stats in self.feature_stats.items():
            s: Dict[str, Any] = {}
            for key, value in stats.items():
                if key == "values":
                    if isinstance(value, list) and len(value) > 100:
                        s[key] = value[-100:]
                    else:
                        s[key] = value
                else:
                    if hasattr(value, "item"):  # numpy scalar
                        s[key] = value.item()
                    else:
                        s[key] = value
            serialized[name] = s
        return serialized

    def _deserialize_feature_stats(
        self, stats_dict: Dict[str, Any]
    ) -> Dict[str, Dict[str, Union[float, List[float]]]]:
        """Restore feature_stats from JSON."""
        if not stats_dict:
            return {}
        deserialized: Dict[str, Dict[str, Union[float, List[float]]]] = {}
        for name, stats in stats_dict.items():
            if isinstance(stats, dict):
                deserialized[name] = {
                    "min": float(stats.get("min", 0.0)),
                    "max": float(stats.get("max", 1.0)),
                    "values": list(stats.get("values", [])),
                }
            else:
                logger.warning(
                    "Skipping malformed feature_stats for '%s'", name
                )
        return deserialized

    # ------------------------------------------------------------------
    # Fitness comparison
    # ------------------------------------------------------------------

    def _is_better(self, a1: Artifact, a2: Artifact) -> bool:
        """Return ``True`` if *a1* has strictly better fitness than *a2*."""
        if not a1.metrics and not a2.metrics:
            return a1.timestamp > a2.timestamp
        if a1.metrics and not a2.metrics:
            return True
        if not a1.metrics and a2.metrics:
            return False

        f1 = get_fitness_score(a1.metrics, self.config.feature_dimensions)
        f2 = get_fitness_score(a2.metrics, self.config.feature_dimensions)
        return f1 > f2

    # ------------------------------------------------------------------
    # Archive maintenance
    # ------------------------------------------------------------------

    def _update_archive(self, artifact: Artifact) -> None:
        """Consider *artifact* for the elite archive."""
        if len(self.archive) < self.config.archive_size:
            self.archive.add(artifact.id)
            return

        # Clean stale refs
        valid = []
        stale = []
        for pid in self.archive:
            if pid in self.artifacts:
                valid.append(self.artifacts[pid])
            else:
                stale.append(pid)
        for sid in stale:
            self.archive.discard(sid)

        if len(self.archive) < self.config.archive_size:
            self.archive.add(artifact.id)
            return

        if valid:
            worst = min(
                valid,
                key=lambda a: get_fitness_score(
                    a.metrics, self.config.feature_dimensions
                ),
            )
            if self._is_better(artifact, worst):
                self.archive.remove(worst.id)
                self.archive.add(artifact.id)
        else:
            self.archive.add(artifact.id)

    # ------------------------------------------------------------------
    # Best tracking
    # ------------------------------------------------------------------

    def _update_best(self, artifact: Artifact) -> None:
        """Update the absolute-best artifact reference."""
        if self.best_program_id is None:
            self.best_program_id = artifact.id
            return

        if self.best_program_id not in self.artifacts:
            self.best_program_id = artifact.id
            return

        current_best = self.artifacts[self.best_program_id]
        if self._is_better(artifact, current_best):
            old_id = self.best_program_id
            self.best_program_id = artifact.id
            logger.info(
                "New best artifact %s replaces %s", artifact.id, old_id
            )

    def _update_island_best(
        self, artifact: Artifact, island_idx: int
    ) -> None:
        """Update the per-island best artifact reference."""
        if island_idx >= len(self.island_best_programs):
            return

        current_id = self.island_best_programs[island_idx]
        if current_id is None:
            self.island_best_programs[island_idx] = artifact.id
            return

        if current_id not in self.artifacts:
            self.island_best_programs[island_idx] = artifact.id
            return

        current_best = self.artifacts[current_id]
        if self._is_better(artifact, current_best):
            self.island_best_programs[island_idx] = artifact.id

    # ------------------------------------------------------------------
    # Selection / Sampling
    # ------------------------------------------------------------------

    def _sample_parent(self) -> Artifact:
        """Sample a parent from the current island using exploration/exploitation balance."""
        rand_val = random.random()

        if rand_val < self.config.exploration_ratio:
            return self._sample_exploration_parent()
        elif rand_val < (
            self.config.exploration_ratio + self.config.exploitation_ratio
        ):
            return self._sample_exploitation_parent()
        else:
            return self._sample_random_parent()

    def _sample_exploration_parent(self) -> Artifact:
        """Random sampling from the current island."""
        current_island_programs = self.islands[self.current_island]

        if not current_island_programs:
            return self._initialize_empty_island(self.current_island)

        valid = [
            pid for pid in current_island_programs if pid in self.artifacts
        ]

        # Remove stale
        if len(valid) < len(current_island_programs):
            stale = current_island_programs - set(valid)
            for sid in stale:
                self.islands[self.current_island].discard(sid)

        if not valid:
            return self._initialize_empty_island(self.current_island)

        parent_id = random.choice(valid)
        return self.artifacts[parent_id]

    def _sample_exploitation_parent(self) -> Artifact:
        """Sample from the elite archive, preferring current island."""
        if not self.archive:
            return self._sample_exploration_parent()

        valid_archive = [pid for pid in self.archive if pid in self.artifacts]
        if len(valid_archive) < len(self.archive):
            for sid in self.archive - set(valid_archive):
                self.archive.discard(sid)

        if not valid_archive:
            return self._sample_exploration_parent()

        # Prefer current island
        in_island = [
            pid
            for pid in valid_archive
            if self.artifacts[pid].metadata.get("island") == self.current_island
        ]

        if in_island:
            return self.artifacts[random.choice(in_island)]
        return self.artifacts[random.choice(valid_archive)]

    def _sample_random_parent(self) -> Artifact:
        """Uniform random from the entire population."""
        if not self.artifacts:
            raise ValueError("No artifacts available for sampling")
        pid = random.choice(list(self.artifacts.keys()))
        return self.artifacts[pid]

    def _sample_from_island_weighted(self, island_id: int) -> Artifact:
        """Fitness-weighted sampling from a specific island."""
        island_id = island_id % len(self.islands)
        island_programs = list(self.islands[island_id])

        if not island_programs:
            return self._sample_random_parent()

        # Guard: filter to only valid (non-orphaned) IDs and clean stale refs
        valid_ids = [pid for pid in island_programs if pid in self.artifacts]
        orphaned = set(island_programs) - set(valid_ids)
        if orphaned:
            logger.warning(
                "Removed %d orphaned IDs from island %d during sampling",
                len(orphaned),
                island_id,
            )
            for oid in orphaned:
                self.islands[island_id].discard(oid)

        if not valid_ids:
            return self._sample_random_parent()

        if len(valid_ids) == 1:
            return self.artifacts[valid_ids[0]]

        objs = [self.artifacts[pid] for pid in valid_ids]
        if not objs:
            return self._sample_random_parent()

        weights = []
        for a in objs:
            fitness = get_fitness_score(
                a.metrics, self.config.feature_dimensions
            )
            weights.append(max(fitness, 0.001))

        total = sum(weights)
        if total > 0:
            weights = [w / total for w in weights]
        else:
            weights = [1.0 / len(objs)] * len(objs)

        parent = random.choices(objs, weights=weights, k=1)[0]
        return parent

    def _sample_from_island_random(self, island_id: int) -> Artifact:
        """Uniform random from a specific island."""
        island_id = island_id % len(self.islands)
        island_programs = list(self.islands[island_id])

        if not island_programs:
            return self._sample_random_parent()

        valid = [pid for pid in island_programs if pid in self.artifacts]
        if not valid:
            return self._sample_random_parent()

        return self.artifacts[random.choice(valid)]

    def _sample_from_archive_for_island(self, island_id: int) -> Artifact:
        """Sample from the archive, preferring *island_id*."""
        if not self.archive:
            return self._sample_from_island_weighted(island_id)

        valid_archive = [pid for pid in self.archive if pid in self.artifacts]
        if not valid_archive:
            return self._sample_from_island_weighted(island_id)

        island_id = island_id % len(self.islands)
        in_island = [
            pid
            for pid in valid_archive
            if self.artifacts[pid].metadata.get("island") == island_id
        ]

        if in_island:
            return self.artifacts[random.choice(in_island)]
        return self.artifacts[random.choice(valid_archive)]

    def _sample_inspirations(
        self, parent: Artifact, n: int = 5
    ) -> List[Artifact]:
        """
        Sample inspiration artifacts from the parent's island.

        Includes the island best (if different from parent), top programs,
        nearby feature-map cells, and random island members.
        """
        inspirations: List[Artifact] = []
        parent_island = parent.metadata.get("island", self.current_island)

        island_program_ids = list(self.islands[parent_island])
        island_artifacts = [
            self.artifacts[pid]
            for pid in island_program_ids
            if pid in self.artifacts
        ]

        if not island_artifacts:
            return []

        # Include island best if available
        island_best_id = self.island_best_programs[parent_island]
        if (
            island_best_id is not None
            and island_best_id != parent.id
            and island_best_id in self.artifacts
        ):
            inspirations.append(self.artifacts[island_best_id])
        elif island_best_id is not None and island_best_id not in self.artifacts:
            self.island_best_programs[parent_island] = None

        # Top programs from the island
        top_n = max(1, int(n * self.config.elite_selection_ratio))
        top_island = self.get_top_programs(n=top_n, island_idx=parent_island)
        for a in top_island:
            if a.id not in {p.id for p in inspirations} and a.id != parent.id:
                inspirations.append(a)

        # Nearby feature-map cells + random fill
        if len(island_artifacts) > n and len(inspirations) < n:
            remaining_slots = n - len(inspirations)
            feature_coords = self._calculate_feature_coords(parent)
            nearby: List[Artifact] = []

            island_fm: Dict[str, str] = {}
            for pid in island_program_ids:
                if pid in self.artifacts:
                    prog = self.artifacts[pid]
                    prog_coords = self._calculate_feature_coords(prog)
                    cell_key = self._feature_coords_to_key(prog_coords)
                    island_fm[cell_key] = pid

            for _ in range(remaining_slots * 3):
                perturbed = [
                    max(0, min(self.feature_bins - 1, c + random.randint(-2, 2)))
                    for c in feature_coords
                ]
                cell_key = self._feature_coords_to_key(perturbed)
                if cell_key in island_fm:
                    pid = island_fm[cell_key]
                    if (
                        pid != parent.id
                        and pid not in {p.id for p in inspirations}
                        and pid not in {p.id for p in nearby}
                        and pid in self.artifacts
                    ):
                        nearby.append(self.artifacts[pid])
                        if len(nearby) >= remaining_slots:
                            break

            # Random fill from island
            if len(inspirations) + len(nearby) < n:
                remaining = n - len(inspirations) - len(nearby)
                excluded = (
                    {parent.id}
                    | {p.id for p in inspirations}
                    | {p.id for p in nearby}
                )
                available = [
                    pid
                    for pid in island_program_ids
                    if pid not in excluded and pid in self.artifacts
                ]
                if available:
                    chosen = random.sample(
                        available, min(remaining, len(available))
                    )
                    nearby.extend(self.artifacts[pid] for pid in chosen)

            inspirations.extend(nearby)

        return inspirations[:n]

    def _initialize_empty_island(self, island_idx: int) -> Artifact:
        """Seed an empty island with a copy of the best artifact (or any)."""
        if self.best_program_id and self.best_program_id in self.artifacts:
            best = self.artifacts[self.best_program_id]
            copy_artifact = Artifact(
                id=str(uuid.uuid4()),
                content=best.content,
                changes_description=best.changes_description,
                artifact_type=best.artifact_type,
                parent_id=best.id,
                generation=best.generation,
                timestamp=time.time(),
                iteration_found=self.last_iteration,
                metrics=best.metrics.copy(),
                complexity=best.complexity,
                diversity=best.diversity,
                metadata={"island": island_idx},
            )
            self.artifacts[copy_artifact.id] = copy_artifact
            self.islands[island_idx].add(copy_artifact.id)
            return copy_artifact
        else:
            return next(iter(self.artifacts.values()))

    # ------------------------------------------------------------------
    # Population limit
    # ------------------------------------------------------------------

    def _enforce_population_limit(
        self, exclude_program_id: Optional[str] = None
    ) -> None:
        """Remove worst artifacts when population exceeds the limit."""
        if len(self.artifacts) <= self.config.population_size:
            return

        num_to_remove = len(self.artifacts) - self.config.population_size

        sorted_artifacts = sorted(
            self.artifacts.values(),
            key=lambda a: get_fitness_score(
                a.metrics, self.config.feature_dimensions
            ),
        )

        protected = {self.best_program_id, exclude_program_id} - {None}
        to_remove: List[Artifact] = []

        for a in sorted_artifacts:
            if len(to_remove) >= num_to_remove:
                break
            if a.id not in protected:
                to_remove.append(a)

        for a in to_remove:
            aid = a.id
            if aid in self.artifacts:
                del self.artifacts[aid]
            for island_map in self.island_feature_maps:
                keys_del = [k for k, v in island_map.items() if v == aid]
                for k in keys_del:
                    del island_map[k]
            for island in self.islands:
                island.discard(aid)
            self.archive.discard(aid)

        self._cleanup_stale_island_bests()

    # ------------------------------------------------------------------
    # Island management
    # ------------------------------------------------------------------

    def set_current_island(self, island_idx: int) -> None:
        """Switch the active island."""
        self.current_island = island_idx % len(self.islands)

    def next_island(self) -> int:
        """Advance to the next island (round-robin)."""
        self.current_island = (self.current_island + 1) % len(self.islands)
        return self.current_island

    def increment_island_generation(
        self, island_idx: Optional[int] = None
    ) -> None:
        """Increment the generation counter for an island."""
        idx = island_idx if island_idx is not None else self.current_island
        self.island_generations[idx] += 1

    def should_migrate(self) -> bool:
        """Return ``True`` if migration is due."""
        max_gen = max(self.island_generations)
        return (max_gen - self.last_migration_generation) >= self.migration_interval

    def migrate_programs(self) -> None:
        """Copy top artifacts from each island to adjacent islands (ring topology)."""
        with self._lock:
            if len(self.islands) < 2:
                return

            logger.info("Performing migration between islands")

            for i, island in enumerate(self.islands):
                if not island:
                    continue

                island_artifacts = [
                    self.artifacts[pid]
                    for pid in island
                    if pid in self.artifacts
                ]
                if not island_artifacts:
                    continue

                island_artifacts.sort(
                    key=lambda a: get_fitness_score(
                        a.metrics, self.config.feature_dimensions
                    ),
                    reverse=True,
                )

                num_migrate = max(1, int(len(island_artifacts) * self.migration_rate))
                migrants = island_artifacts[:num_migrate]

                targets = [
                    (i + 1) % len(self.islands),
                    (i - 1) % len(self.islands),
                ]

                for migrant in migrants:
                    if migrant.metadata.get("migrant", False):
                        continue

                    for target in targets:
                        # Skip duplicates
                        target_artifacts = [
                            self.artifacts[pid]
                            for pid in self.islands[target]
                            if pid in self.artifacts
                        ]
                        if any(a.content == migrant.content for a in target_artifacts):
                            continue

                        copy = Artifact(
                            id=str(uuid.uuid4()),
                            content=migrant.content,
                            changes_description=migrant.changes_description,
                            artifact_type=migrant.artifact_type,
                            parent_id=migrant.id,
                            generation=migrant.generation,
                            metrics=migrant.metrics.copy(),
                            metadata={
                                **migrant.metadata,
                                "island": target,
                                "migrant": True,
                            },
                        )
                        self.add(copy, target_island=target)

            self.last_migration_generation = max(self.island_generations)
            logger.info(
                "Migration completed at generation %d",
                self.last_migration_generation,
            )

    def _reconstruct_islands(
        self, saved_islands: List[List[str]]
    ) -> None:
        """Rebuild island sets from saved metadata."""
        num_islands = max(len(saved_islands), self.config.num_islands)
        self.islands = [set() for _ in range(num_islands)]

        for island_idx, program_ids in enumerate(saved_islands):
            if island_idx >= len(self.islands):
                continue
            for pid in program_ids:
                if pid in self.artifacts:
                    self.islands[island_idx].add(pid)
                    self.artifacts[pid].metadata["island"] = island_idx

        # Clean archive
        self.archive = {pid for pid in self.archive if pid in self.artifacts}

        # Clean feature maps
        for island_map in self.island_feature_maps:
            stale_keys = [
                k for k, v in island_map.items() if v not in self.artifacts
            ]
            for k in stale_keys:
                del island_map[k]

        self._cleanup_stale_island_bests()

        if self.best_program_id and self.best_program_id not in self.artifacts:
            self.best_program_id = None

        # Distribute if no assignments
        if self.artifacts and sum(len(isl) for isl in self.islands) == 0:
            self._distribute_programs_to_islands()

    def _distribute_programs_to_islands(self) -> None:
        """Round-robin distribute artifacts when no island metadata exists."""
        for i, pid in enumerate(self.artifacts):
            island_idx = i % len(self.islands)
            self.islands[island_idx].add(pid)
            self.artifacts[pid].metadata["island"] = island_idx

    def get_island_stats(self) -> List[dict]:
        """Return per-island statistics."""
        stats: List[dict] = []
        for i, island in enumerate(self.islands):
            island_artifacts = [
                self.artifacts[pid]
                for pid in island
                if pid in self.artifacts
            ]
            if island_artifacts:
                scores = [
                    get_fitness_score(
                        a.metrics, self.config.feature_dimensions
                    )
                    for a in island_artifacts
                ]
                best_score = max(scores) if scores else 0.0
                avg_score = (
                    sum(scores) / len(scores) if scores else 0.0
                )
                diversity = self._calculate_island_diversity(island_artifacts)
            else:
                best_score = avg_score = diversity = 0.0

            stats.append(
                {
                    "island": i,
                    "population_size": len(island_artifacts),
                    "best_score": best_score,
                    "average_score": avg_score,
                    "diversity": diversity,
                    "generation": self.island_generations[i]
                    if i < len(self.island_generations)
                    else 0,
                    "is_current": i == self.current_island,
                }
            )
        return stats

    def _cleanup_stale_island_bests(self) -> None:
        """Remove stale island best references and recalculate."""
        cleaned = 0
        for i, best_id in enumerate(self.island_best_programs):
            if best_id is not None:
                should_clear = False
                if best_id not in self.artifacts:
                    should_clear = True
                elif i < len(self.islands) and best_id not in self.islands[i]:
                    should_clear = True

                if should_clear:
                    self.island_best_programs[i] = None
                    cleaned += 1

        if cleaned > 0:
            for i, best_id in enumerate(self.island_best_programs):
                if best_id is None and i < len(self.islands) and self.islands[i]:
                    island_artifacts = [
                        self.artifacts[pid]
                        for pid in self.islands[i]
                        if pid in self.artifacts
                    ]
                    if island_artifacts:
                        best_a = max(
                            island_artifacts,
                            key=lambda a: a.metrics.get(
                                "combined_score",
                                safe_numeric_average(a.metrics),
                            ),
                        )
                        self.island_best_programs[i] = best_a.id

    # ------------------------------------------------------------------
    # Stagnation
    # ------------------------------------------------------------------

    def get_score_history(self, min_iteration: int = 0) -> List[float]:
        """Return combined_score values in chronological order (by iteration_found).

        Iterates all artifacts, sorts by iteration_found, and extracts
        the combined_score metric. Useful for stagnation detection.

        Args:
            min_iteration: Only include artifacts from this iteration onward.
                Defaults to 0 (includes all). Use this to scope score history
                to the current run and avoid contamination from previous runs.

        Returns:
            List of floats representing the score at each iteration.
        """
        # Get all artifacts sorted by iteration_found
        sorted_artifacts = sorted(
            self.artifacts.values(),
            key=lambda a: (a.iteration_found, a.timestamp),
        )

        scores = []
        for artifact in sorted_artifacts:
            if artifact.iteration_found < min_iteration:
                continue
            score = artifact.metrics.get("combined_score")
            if score is not None and isinstance(score, (int, float)):
                scores.append(float(score))

        return scores

    def detect_stagnation(self, config=None) -> "StagnationReport":
        """Run stagnation detection on the current score history.

        Args:
            config: Optional StagnationConfig (from config.py or stagnation.py).
                    If None, uses default thresholds.

        Returns:
            StagnationReport from the StagnationEngine.
        """
        from claude_evolve.core.stagnation import StagnationEngine

        engine = StagnationEngine(config)
        score_history = self.get_score_history()
        return engine.analyze(score_history)

    # ------------------------------------------------------------------
    # Novelty
    # ------------------------------------------------------------------

    def _is_novel(self, artifact_id: str, island_idx: int) -> bool:
        """Check if an artifact is sufficiently novel compared to island population.

        Uses the universal novelty system (structural + behavioral + semantic)
        that works across ALL artifact types — code, prose, configs, SQL, etc.

        Returns True if novel (should be added), False if too similar.
        """
        if artifact_id not in self.artifacts:
            return True

        artifact = self.artifacts[artifact_id]
        island_ids = self.islands[island_idx]

        if not island_ids:
            return True

        artifact_content = artifact.content
        if not artifact_content:
            return True

        from claude_evolve.core.novelty import compute_novelty

        for existing_id in island_ids:
            if existing_id == artifact_id:
                continue
            existing = self.artifacts.get(existing_id)
            if existing is None or not existing.content:
                continue

            # Quick exact-match check
            if artifact_content == existing.content:
                logger.debug("Artifact %s is identical to %s", artifact_id, existing_id)
                return False

            # Quick length-based pre-filter
            len_ratio = min(len(artifact_content), len(existing.content)) / max(len(artifact_content), len(existing.content), 1)
            if len_ratio < 0.5:
                continue  # Very different lengths = likely novel, skip expensive check

            # Universal novelty check (works for any artifact type)
            novelty = compute_novelty(
                artifact_content,
                existing.content,
                artifact_type=artifact.artifact_type,
                metrics_a=artifact.metrics,
                metrics_b=existing.metrics,
            )

            # Novelty < (1 - threshold) means too similar
            if novelty < (1.0 - self.config.similarity_threshold):
                logger.debug(
                    "Artifact %s too similar to %s (novelty=%.3f < %.3f)",
                    artifact_id, existing_id, novelty,
                    1.0 - self.config.similarity_threshold,
                )
                return False

        return True

    @staticmethod
    def _line_similarity(content_a: str, content_b: str) -> float:
        """Compute line-based Jaccard similarity between two strings.

        Kept for backward compatibility. For new code, use
        ``novelty.structural_similarity()`` which is artifact-type-aware.
        """
        lines_a = set(line.strip() for line in content_a.splitlines() if line.strip())
        lines_b = set(line.strip() for line in content_b.splitlines() if line.strip())

        if not lines_a and not lines_b:
            return 1.0
        if not lines_a or not lines_b:
            return 0.0

        intersection = len(lines_a & lines_b)
        union = len(lines_a | lines_b)

        return intersection / union if union > 0 else 0.0

    # ------------------------------------------------------------------
    # Diversity
    # ------------------------------------------------------------------

    def _get_cached_diversity(self, artifact: Artifact) -> float:
        """Diversity score against a reference set, with caching."""
        code_hash = hash(artifact.content)

        if code_hash in self.diversity_cache:
            return self.diversity_cache[code_hash]["value"]

        if (
            not self.diversity_reference_set
            or len(self.diversity_reference_set) < self.diversity_reference_size
        ):
            self._update_diversity_reference_set()

        scores: List[float] = []
        for ref_content in self.diversity_reference_set:
            if ref_content != artifact.content:
                scores.append(
                    self._fast_code_diversity(artifact.content, ref_content)
                )

        diversity = (
            sum(scores) / max(1, len(scores)) if scores else 0.0
        )

        self._cache_diversity_value(code_hash, diversity)
        return diversity

    def _update_diversity_reference_set(self) -> None:
        """Build a diverse reference set for diversity computation."""
        if not self.artifacts:
            return

        all_artifacts = list(self.artifacts.values())

        if len(all_artifacts) <= self.diversity_reference_size:
            self.diversity_reference_set = [a.content for a in all_artifacts]
        else:
            selected: List[Artifact] = []
            remaining = all_artifacts.copy()

            first_idx = random.randint(0, len(remaining) - 1)
            selected.append(remaining.pop(first_idx))

            while (
                len(selected) < self.diversity_reference_size and remaining
            ):
                max_div = -1.0
                best_idx = -1
                for i, candidate in enumerate(remaining):
                    min_d = float("inf")
                    for sel in selected:
                        d = self._fast_code_diversity(
                            candidate.content, sel.content
                        )
                        min_d = min(min_d, d)
                    if min_d > max_div:
                        max_div = min_d
                        best_idx = i
                if best_idx >= 0:
                    selected.append(remaining.pop(best_idx))

            self.diversity_reference_set = [a.content for a in selected]

    def _cache_diversity_value(
        self, code_hash: int, diversity: float
    ) -> None:
        """LRU cache for diversity scores."""
        if len(self.diversity_cache) >= self.diversity_cache_size:
            oldest = min(
                self.diversity_cache.items(),
                key=lambda x: x[1]["timestamp"],
            )[0]
            del self.diversity_cache[oldest]

        self.diversity_cache[code_hash] = {
            "value": diversity,
            "timestamp": time.time(),
        }

    def _calculate_island_diversity(
        self, artifacts: List[Artifact]
    ) -> float:
        """Deterministic diversity metric within a set of artifacts."""
        if len(artifacts) < 2:
            return 0.0

        sample_size = min(5, len(artifacts))
        sorted_arts = sorted(artifacts, key=lambda a: a.id)
        sample = sorted_arts[:sample_size]

        total = 0.0
        comparisons = 0
        max_comparisons = 6

        for i, a1 in enumerate(sample):
            for a2 in sample[i + 1 :]:
                if comparisons >= max_comparisons:
                    break
                total += self._fast_code_diversity(a1.content, a2.content)
                comparisons += 1
            if comparisons >= max_comparisons:
                break

        return total / max(1, comparisons)

    def _fast_code_diversity(self, content1: str, content2: str) -> float:
        """Fast approximation of content diversity.

        Uses the universal novelty system when artifact types are available,
        falls back to character-level heuristics for speed.
        """
        if content1 == content2:
            return 0.0

        # Use universal structural similarity for better cross-type support
        from claude_evolve.core.novelty import structural_similarity
        sim = structural_similarity(content1, content2)
        # Convert similarity [0,1] to diversity score (higher = more diverse)
        # Scale to match the old range (roughly 0-1000)
        return (1.0 - sim) * 1000.0

    # ------------------------------------------------------------------
    # Artifact storage (small -> JSON, large -> disk)
    # ------------------------------------------------------------------

    def store_artifacts(
        self,
        program_id: str,
        artifacts: Dict[str, Union[str, bytes]],
    ) -> None:
        """
        Store evaluation artifacts for a program.

        Small artifacts are serialized into the Artifact's ``eval_artifacts``
        field; large ones are written to disk.
        """
        if not artifacts:
            return

        artifact = self.get(program_id)
        if not artifact:
            logger.warning(
                "Cannot store artifacts: artifact %s not found", program_id
            )
            return

        size_threshold = getattr(
            self.config, "artifact_size_threshold", 32 * 1024
        )

        small: Dict[str, str] = {}
        large: Dict[str, Union[str, bytes]] = {}

        for key, value in artifacts.items():
            sz = self._get_artifact_size(value)
            if sz <= size_threshold:
                small[key] = value  # type: ignore[assignment]
            else:
                large[key] = value

        if small:
            artifact.eval_artifacts = artifact.eval_artifacts or {}
            artifact.eval_artifacts.update(
                {k: v if isinstance(v, str) else str(v) for k, v in small.items()}
            )

        if large:
            artifact_dir = self._create_artifact_dir(program_id)
            for key, value in large.items():
                self._write_artifact_file(artifact_dir, key, value)
            # Track the directory path in metadata
            artifact.metadata["artifact_dir"] = artifact_dir

    def get_artifacts(
        self, program_id: str
    ) -> Optional[Dict[str, Union[str, bytes]]]:
        """
        Retrieve all stored artifacts for a program.

        Returns ``None`` if the program does not exist.
        """
        artifact = self.get(program_id)
        if not artifact:
            return None

        result: Dict[str, Union[str, bytes]] = {}

        if artifact.eval_artifacts:
            result.update(artifact.eval_artifacts)

        artifact_dir = artifact.metadata.get("artifact_dir")
        if artifact_dir and os.path.exists(artifact_dir):
            result.update(self._load_artifact_dir(artifact_dir))

        return result

    def _get_artifact_size(self, value: Union[str, bytes]) -> int:
        """Size of an artifact in bytes."""
        if isinstance(value, str):
            return len(value.encode("utf-8"))
        elif isinstance(value, bytes):
            return len(value)
        return len(str(value).encode("utf-8"))

    def _artifact_serializer(self, obj: Any) -> Any:
        """JSON serializer that handles bytes."""
        if isinstance(obj, bytes):
            return {"__bytes__": base64.b64encode(obj).decode("utf-8")}
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    def _create_artifact_dir(self, program_id: str) -> str:
        """Create (and return) the on-disk artifact directory for *program_id*."""
        base_path = getattr(self.config, "artifacts_base_path", None)
        if not base_path:
            base_path = (
                os.path.join(self.config.db_path or ".", "artifacts")
                if self.config.db_path
                else "./artifacts"
            )

        artifact_dir = os.path.join(base_path, program_id)
        os.makedirs(artifact_dir, exist_ok=True)
        return artifact_dir

    def _write_artifact_file(
        self, artifact_dir: str, key: str, value: Union[str, bytes]
    ) -> None:
        """Write a single artifact to disk."""
        safe_key = "".join(c for c in key if c.isalnum() or c in "._-")
        if not safe_key:
            safe_key = "artifact"

        file_path = os.path.join(artifact_dir, safe_key)

        try:
            if isinstance(value, str):
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(value)
            elif isinstance(value, bytes):
                with open(file_path, "wb") as f:
                    f.write(value)
            else:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(str(value))
        except Exception as e:
            logger.warning(
                "Failed to write artifact %s to %s: %s", key, file_path, e
            )

    def _load_artifact_dir(
        self, artifact_dir: str
    ) -> Dict[str, Union[str, bytes]]:
        """Load all artifacts from an on-disk directory."""
        artifacts: Dict[str, Union[str, bytes]] = {}
        try:
            for filename in os.listdir(artifact_dir):
                filepath = os.path.join(artifact_dir, filename)
                if os.path.isfile(filepath):
                    try:
                        with open(filepath, "r", encoding="utf-8") as f:
                            artifacts[filename] = f.read()
                    except UnicodeDecodeError:
                        with open(filepath, "rb") as f:
                            artifacts[filename] = f.read()
                    except Exception as e:
                        logger.warning(
                            "Failed to read artifact %s: %s", filepath, e
                        )
        except Exception as e:
            logger.warning(
                "Failed to list artifact dir %s: %s", artifact_dir, e
            )
        return artifacts

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _save_artifact(
        self,
        artifact: Artifact,
        base_path: Optional[str] = None,
        prompts: Optional[Dict[str, Dict[str, str]]] = None,
    ) -> None:
        """Write a single artifact to its JSON file."""
        save_path = base_path or self.config.db_path
        if not save_path:
            return

        programs_dir = os.path.join(save_path, "programs")
        os.makedirs(programs_dir, exist_ok=True)

        data = artifact.to_dict()
        if prompts:
            data["prompts"] = prompts
        filepath = os.path.join(programs_dir, f"{artifact.id}.json")

        with open(filepath, "w") as f:
            json.dump(data, f)
