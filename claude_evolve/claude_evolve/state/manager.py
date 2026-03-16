"""
StateManager -- manages the ``.claude/evolve-state/`` directory.

Directory layout::

    evolve-state/
    +-- database.json          # Serialized ArtifactDatabase metadata
    +-- database/              # ArtifactDatabase on-disk storage
    |   +-- metadata.json
    |   +-- programs/
    +-- config.json            # Resolved Config (full snapshot)
    +-- best_artifact.*        # Current best artifact content
    +-- iteration_context.md   # Latest context for Claude
    +-- checkpoints/           # Periodic snapshots (managed by CheckpointManager)
"""

import json
import logging
import os
from typing import Optional

from claude_evolve.config import Config
from claude_evolve.core.artifact import Artifact
from claude_evolve.core.database import ArtifactDatabase

logger = logging.getLogger(__name__)

# Map artifact_type -> file extension
_EXTENSION_MAP = {
    "python": ".py",
    "javascript": ".js",
    "typescript": ".ts",
    "rust": ".rs",
    "go": ".go",
    "java": ".java",
    "c": ".c",
    "cpp": ".cpp",
    "ruby": ".rb",
    "shell": ".sh",
    "bash": ".sh",
    "html": ".html",
    "css": ".css",
    "json": ".json",
    "yaml": ".yaml",
    "toml": ".toml",
    "markdown": ".md",
    "sql": ".sql",
}


class StateManager:
    """Manages the ``evolve-state/`` directory for a single evolution run.

    Responsibilities:
    - Initialize directory structure with seed artifact and config.
    - Save/load the ArtifactDatabase and Config between iterations.
    - Write/read the iteration context file used by Claude.
    - Write the current best artifact to disk.

    Usage::

        sm = StateManager("/project/.claude/evolve-state")
        sm.initialize(config=cfg, initial_content="x = 1", artifact_type="python")
        # ... evolution iterations ...
        sm.save()
        # Later:
        sm2 = StateManager("/project/.claude/evolve-state")
        sm2.load()
    """

    def __init__(self, state_dir: str) -> None:
        self.state_dir = state_dir
        self.database: Optional[ArtifactDatabase] = None
        self.config: Optional[Config] = None

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def initialize(
        self,
        config: Config,
        initial_content: str,
        artifact_type: str = "python",
        evaluator_path: Optional[str] = None,
        fresh: bool = True,
    ) -> None:
        """Create the state directory and seed the database.

        Args:
            config: Resolved configuration for this evolution run.
            initial_content: Content of the seed artifact.
            artifact_type: Type of artifact (``python``, ``javascript``, etc.).
            evaluator_path: Optional path to the evaluator script (stored
                in config metadata but not used directly here).
            fresh: If True (default), clear the entire state directory before
                initializing. This prevents contamination from previous runs
                (stale database, warm cache, strategies, eval cache).
        """
        if fresh and os.path.exists(self.state_dir):
            import shutil
            # Preserve cross_run_memory if it exists (it's meant to persist)
            crm_dir = os.path.join(self.state_dir, "cross_run_memory")
            crm_backup = None
            if os.path.exists(crm_dir):
                crm_backup = crm_dir + ".bak"
                shutil.move(crm_dir, crm_backup)

            # Clear everything else
            shutil.rmtree(self.state_dir)

            # Restore cross_run_memory
            os.makedirs(self.state_dir, exist_ok=True)
            if crm_backup and os.path.exists(crm_backup):
                shutil.move(crm_backup, crm_dir)

        os.makedirs(self.state_dir, exist_ok=True)
        os.makedirs(os.path.join(self.state_dir, "checkpoints"), exist_ok=True)

        self.config = config

        # Create and seed the database
        db_dir = os.path.join(self.state_dir, "database")
        db_config = config.database
        self.database = ArtifactDatabase(db_config)

        seed_artifact = Artifact(
            id=Artifact.generate_id(),
            content=initial_content,
            artifact_type=artifact_type,
            generation=0,
            metrics={},
            metadata={"is_seed": True},
        )
        self.database.add(seed_artifact)

        # Persist everything
        self._save_database()
        self._save_config()

        logger.info(
            "Initialized evolve-state at %s with seed artifact (type=%s, %d bytes)",
            self.state_dir,
            artifact_type,
            len(initial_content),
        )

    # ------------------------------------------------------------------
    # Load / Save
    # ------------------------------------------------------------------
    def load(self) -> None:
        """Load database and config from the state directory.

        Raises:
            FileNotFoundError: If the state directory or required files
                do not exist.
        """
        config_path = os.path.join(self.state_dir, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"State directory not initialized: {config_path} not found"
            )

        self._load_config()
        self._load_database()
        logger.info(
            "Loaded evolve-state from %s (%d artifacts)",
            self.state_dir,
            self.database.size() if self.database else 0,
        )

    def save(self) -> None:
        """Persist the current database and config to disk."""
        if self.database is None:
            logger.warning("No database to save -- call initialize() or load() first")
            return
        self._save_database()
        self._save_config()
        logger.info("Saved evolve-state to %s", self.state_dir)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------
    def get_database(self) -> ArtifactDatabase:
        """Return the in-memory ArtifactDatabase.

        Raises:
            RuntimeError: If the state has not been initialized or loaded.
        """
        if self.database is None:
            raise RuntimeError(
                "StateManager not initialized -- call initialize() or load()"
            )
        return self.database

    def get_config(self) -> Config:
        """Return the resolved Config.

        Raises:
            RuntimeError: If the state has not been initialized or loaded.
        """
        if self.config is None:
            raise RuntimeError(
                "StateManager not initialized -- call initialize() or load()"
            )
        return self.config

    # ------------------------------------------------------------------
    # Iteration context
    # ------------------------------------------------------------------
    def write_iteration_context(self, content: str) -> None:
        """Write the iteration context markdown file.

        This file is what Claude reads at the start of each iteration to
        understand what to do next.

        Args:
            content: Markdown text for the current iteration context.
        """
        path = os.path.join(self.state_dir, "iteration_context.md")
        with open(path, "w") as fh:
            fh.write(content)

    def read_iteration_context(self) -> str:
        """Read the iteration context markdown file.

        Returns:
            The context text, or an empty string if the file does not exist.
        """
        path = os.path.join(self.state_dir, "iteration_context.md")
        if not os.path.exists(path):
            return ""
        with open(path, "r") as fh:
            return fh.read()

    # ------------------------------------------------------------------
    # Best artifact
    # ------------------------------------------------------------------
    def write_best_artifact(self, content: str, artifact_type: str) -> None:
        """Write the best artifact content to ``best_artifact.<ext>``.

        Any previous ``best_artifact.*`` files are removed first so that
        only one best artifact exists at a time.

        Args:
            content: The artifact content.
            artifact_type: Artifact type string used to determine extension.
        """
        ext = _EXTENSION_MAP.get(artifact_type, ".txt")
        filename = f"best_artifact{ext}"
        target_path = os.path.join(self.state_dir, filename)

        # Remove stale best_artifact.* files
        for entry in os.listdir(self.state_dir):
            if entry.startswith("best_artifact."):
                old_path = os.path.join(self.state_dir, entry)
                if old_path != target_path:
                    os.remove(old_path)

        with open(target_path, "w") as fh:
            fh.write(content)

    # ------------------------------------------------------------------
    # Internal persistence helpers
    # ------------------------------------------------------------------
    def _save_database(self) -> None:
        """Save the ArtifactDatabase to ``database.json`` and ``database/``."""
        if self.database is None:
            return
        db_dir = os.path.join(self.state_dir, "database")
        self.database.save(db_dir)

        # Also write a lightweight top-level database.json with summary info
        summary = {
            "num_artifacts": self.database.size(),
            "best_id": self.database.best_program_id,
            "db_dir": "database",
        }
        best = self.database.get_best()
        if best is not None:
            summary["best_score"] = best.metrics.get("combined_score", 0.0)

        summary_path = os.path.join(self.state_dir, "database.json")
        with open(summary_path, "w") as fh:
            json.dump(summary, fh, indent=2)

    def _load_database(self) -> None:
        """Load the ArtifactDatabase from ``database/``."""
        db_dir = os.path.join(self.state_dir, "database")
        if self.config is None:
            raise RuntimeError("Config must be loaded before database")
        self.database = ArtifactDatabase(self.config.database)
        if os.path.exists(db_dir):
            self.database.load(db_dir)

    def _save_config(self) -> None:
        """Save the Config to ``config.json``."""
        if self.config is None:
            return
        config_path = os.path.join(self.state_dir, "config.json")
        with open(config_path, "w") as fh:
            json.dump(self.config.to_dict(), fh, indent=2)

    def _load_config(self) -> None:
        """Load the Config from ``config.json``."""
        config_path = os.path.join(self.state_dir, "config.json")
        with open(config_path, "r") as fh:
            data = json.load(fh)
        self.config = Config.from_dict(data)
