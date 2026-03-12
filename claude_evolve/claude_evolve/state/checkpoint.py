"""
CheckpointManager -- saves and restores evolution checkpoints.

Checkpoints are stored under ``<state_dir>/checkpoints/iter_NNN/`` where
*NNN* is the zero-padded iteration number.  Each checkpoint is a full
snapshot of the ArtifactDatabase at that point in time, plus a small
``meta.json`` file with timestamp and summary statistics.

Directory layout::

    checkpoints/
    +-- iter_005/
    |   +-- meta.json          # {iteration, timestamp, num_artifacts, best_id}
    |   +-- metadata.json      # ArtifactDatabase metadata
    |   +-- programs/          # Per-artifact JSON files
    +-- iter_010/
    |   +-- ...
"""

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Union

from claude_evolve.config import Config
from claude_evolve.core.database import ArtifactDatabase

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages periodic snapshots of evolution state.

    Usage::

        cp = CheckpointManager("/project/.claude/evolve-state")
        cp.save(database, iteration=5)
        cp.save(database, iteration=10)
        checkpoints = cp.list_checkpoints()
        db = cp.restore(iteration=5, config=cfg)
    """

    def __init__(self, state_dir: str) -> None:
        self.state_dir = state_dir
        self.checkpoints_dir = os.path.join(state_dir, "checkpoints")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    def save(self, database: ArtifactDatabase, iteration: int) -> str:
        """Save a checkpoint of the database at the given iteration.

        Args:
            database: The ArtifactDatabase to snapshot.
            iteration: The iteration number for this checkpoint.

        Returns:
            The path to the newly created checkpoint directory.
        """
        checkpoint_name = f"iter_{iteration:03d}"
        checkpoint_dir = os.path.join(self.checkpoints_dir, checkpoint_name)
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Save the database into the checkpoint directory
        database.save(checkpoint_dir, iteration=iteration)

        # Write checkpoint metadata
        meta = {
            "iteration": iteration,
            "timestamp": time.time(),
            "timestamp_iso": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
            ),
            "num_artifacts": database.size(),
            "best_id": database.best_program_id,
        }
        best = database.get_best()
        if best is not None:
            meta["best_score"] = best.metrics.get("combined_score", 0.0)

        meta_path = os.path.join(checkpoint_dir, "meta.json")
        with open(meta_path, "w") as fh:
            json.dump(meta, fh, indent=2)

        logger.info(
            "Saved checkpoint at iteration %d (%d artifacts) to %s",
            iteration,
            database.size(),
            checkpoint_dir,
        )
        return checkpoint_dir

    # ------------------------------------------------------------------
    # Restore
    # ------------------------------------------------------------------
    def restore(
        self,
        iteration: int,
        config: Union[Config, None] = None,
    ) -> ArtifactDatabase:
        """Restore a database from a checkpoint at the given iteration.

        Args:
            iteration: The iteration number to restore.
            config: The Config to use when reconstructing the database.
                If ``None``, a default ``Config`` is used.

        Returns:
            A new ``ArtifactDatabase`` populated from the checkpoint.

        Raises:
            FileNotFoundError: If no checkpoint exists for *iteration*.
        """
        checkpoint_name = f"iter_{iteration:03d}"
        checkpoint_dir = os.path.join(self.checkpoints_dir, checkpoint_name)

        if not os.path.exists(checkpoint_dir):
            raise FileNotFoundError(
                f"No checkpoint found for iteration {iteration} "
                f"(expected {checkpoint_dir})"
            )

        if config is None:
            config = Config()

        db = ArtifactDatabase(config.database)
        db.load(checkpoint_dir)

        logger.info(
            "Restored checkpoint at iteration %d (%d artifacts) from %s",
            iteration,
            db.size(),
            checkpoint_dir,
        )
        return db

    # ------------------------------------------------------------------
    # List
    # ------------------------------------------------------------------
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all available checkpoints sorted by iteration number.

        Returns:
            A list of metadata dicts, each containing at least
            ``iteration``, ``timestamp``, and ``num_artifacts``.
        """
        if not os.path.exists(self.checkpoints_dir):
            return []

        results: List[Dict[str, Any]] = []
        for entry in os.listdir(self.checkpoints_dir):
            entry_path = os.path.join(self.checkpoints_dir, entry)
            if not os.path.isdir(entry_path):
                continue
            if not entry.startswith("iter_"):
                continue

            meta_path = os.path.join(entry_path, "meta.json")
            if os.path.exists(meta_path):
                with open(meta_path, "r") as fh:
                    meta = json.load(fh)
                results.append(meta)
            else:
                # Reconstruct minimal metadata from directory name
                try:
                    iter_num = int(entry.split("_", 1)[1])
                except (IndexError, ValueError):
                    continue
                results.append(
                    {
                        "iteration": iter_num,
                        "timestamp": os.path.getmtime(entry_path),
                        "num_artifacts": 0,
                    }
                )

        # Sort by iteration number
        results.sort(key=lambda m: m["iteration"])
        return results
