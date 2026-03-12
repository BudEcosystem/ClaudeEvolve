"""
LoopState -- manages the ``.claude/evolve.local.md`` file.

The file uses markdown with YAML frontmatter, compatible with the Ralph
Loop mechanism.  Frontmatter stores evolution metadata (iteration counter,
session id, target score, etc.) and the markdown body carries the dynamic
prompt text that is updated each iteration by ``claude-evolve next``.

File format::

    ---
    active: true
    iteration: 1
    session_id: <session-id>
    max_iterations: 50
    target_score: 0.95
    completion_promise: "EVOLUTION_TARGET_REACHED"
    state_dir: .claude/evolve-state
    evaluator_path: evaluator.py
    artifact_path: program.py
    mode: script
    started_at: "2026-03-12T14:30:45Z"
    best_score: 0.0
    ---

    <dynamic prompt text>
"""

import logging
from datetime import datetime, timezone
from typing import Optional

import yaml

logger = logging.getLogger(__name__)


class LoopState:
    """In-memory representation of ``evolve.local.md``.

    Attributes are the YAML frontmatter keys.  The ``prompt`` attribute
    holds the markdown body (everything after the closing ``---``).
    """

    def __init__(
        self,
        *,
        iteration: int = 1,
        max_iterations: int = 50,
        prompt: str = "",
        session_id: str = "",
        active: bool = True,
        target_score: Optional[float] = None,
        completion_promise: str = "EVOLUTION_TARGET_REACHED",
        state_dir: str = ".claude/evolve-state",
        evaluator_path: str = "",
        artifact_path: str = "",
        mode: str = "script",
        started_at: Optional[str] = None,
        best_score: float = 0.0,
    ) -> None:
        self.active = active
        self.iteration = iteration
        self.session_id = session_id
        self.max_iterations = max_iterations
        self.target_score = target_score
        self.completion_promise = completion_promise
        self.state_dir = state_dir
        self.evaluator_path = evaluator_path
        self.artifact_path = artifact_path
        self.mode = mode
        self.started_at = started_at or datetime.now(timezone.utc).isoformat()
        self.best_score = best_score
        self.prompt = prompt

    # ------------------------------------------------------------------
    # Factory: create a brand-new state file
    # ------------------------------------------------------------------
    @classmethod
    def create(
        cls,
        *,
        path: str,
        session_id: str,
        max_iterations: int,
        prompt: str,
        target_score: Optional[float] = None,
        completion_promise: str = "EVOLUTION_TARGET_REACHED",
        state_dir: str = ".claude/evolve-state",
        evaluator_path: str = "",
        artifact_path: str = "",
        mode: str = "script",
    ) -> "LoopState":
        """Create a new ``evolve.local.md`` file and return the LoopState.

        Args:
            path: Filesystem path where the file will be written.
            session_id: Unique session identifier.
            max_iterations: Hard cap on iteration count.
            prompt: Initial prompt text (becomes the markdown body).
            target_score: Optional score at which evolution stops.
            completion_promise: Sentinel string written on completion.
            state_dir: Relative path to the evolve-state directory.
            evaluator_path: Path to the evaluator script.
            artifact_path: Path to the artifact being evolved.
            mode: Evaluation mode (``script``, ``critic``, ``hybrid``).

        Returns:
            The newly created ``LoopState`` instance.
        """
        instance = cls(
            iteration=1,
            max_iterations=max_iterations,
            prompt=prompt,
            session_id=session_id,
            active=True,
            target_score=target_score,
            completion_promise=completion_promise,
            state_dir=state_dir,
            evaluator_path=evaluator_path,
            artifact_path=artifact_path,
            mode=mode,
            best_score=0.0,
        )
        instance.write(path)
        logger.info(
            "Created evolve.local.md at %s (session=%s, max_iter=%d)",
            path,
            session_id,
            max_iterations,
        )
        return instance

    # ------------------------------------------------------------------
    # Factory: read an existing state file
    # ------------------------------------------------------------------
    @classmethod
    def read(cls, path: str) -> "LoopState":
        """Parse an existing ``evolve.local.md`` file.

        Args:
            path: Filesystem path to the file.

        Returns:
            Populated ``LoopState`` instance.

        Raises:
            FileNotFoundError: If *path* does not exist.
            ValueError: If the file is malformed (missing frontmatter).
        """
        with open(path, "r") as fh:
            raw = fh.read()

        frontmatter, prompt = _parse_frontmatter(raw)

        return cls(
            active=frontmatter.get("active", True),
            iteration=frontmatter.get("iteration", 1),
            session_id=frontmatter.get("session_id", ""),
            max_iterations=frontmatter.get("max_iterations", 50),
            target_score=frontmatter.get("target_score"),
            completion_promise=frontmatter.get(
                "completion_promise", "EVOLUTION_TARGET_REACHED"
            ),
            state_dir=frontmatter.get("state_dir", ".claude/evolve-state"),
            evaluator_path=frontmatter.get("evaluator_path", ""),
            artifact_path=frontmatter.get("artifact_path", ""),
            mode=frontmatter.get("mode", "script"),
            started_at=frontmatter.get("started_at"),
            best_score=float(frontmatter.get("best_score", 0.0)),
            prompt=prompt,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def write(self, path: str) -> None:
        """Write the current state back to *path*.

        The file is written atomically-ish: we build the full content
        in memory and then write in one call so a partial crash does
        not leave a half-written file.
        """
        frontmatter = self._build_frontmatter()
        yaml_text = yaml.dump(
            frontmatter,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )
        content = f"---\n{yaml_text}---\n\n{self.prompt}\n"

        with open(path, "w") as fh:
            fh.write(content)

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------
    def increment_iteration(self) -> None:
        """Bump the iteration counter by one."""
        self.iteration += 1

    def update_prompt(self, prompt: str) -> None:
        """Replace the current prompt text."""
        self.prompt = prompt

    def update_best_score(self, score: float) -> None:
        """Update best_score if *score* is strictly higher."""
        if score > self.best_score:
            self.best_score = score

    # ------------------------------------------------------------------
    # Completion checks
    # ------------------------------------------------------------------
    def is_max_iterations_reached(self) -> bool:
        """Return True if the iteration counter has reached or exceeded max."""
        return self.iteration >= self.max_iterations

    def is_target_reached(self, current_score: float) -> bool:
        """Return True if *current_score* meets or exceeds the target.

        If no ``target_score`` was configured, this always returns False.
        """
        if self.target_score is None:
            return False
        return current_score >= self.target_score

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_frontmatter(self) -> dict:
        """Build the YAML frontmatter dictionary."""
        fm: dict = {
            "active": self.active,
            "iteration": self.iteration,
            "session_id": self.session_id,
            "max_iterations": self.max_iterations,
        }
        if self.target_score is not None:
            fm["target_score"] = self.target_score
        fm["completion_promise"] = self.completion_promise
        fm["state_dir"] = self.state_dir
        fm["evaluator_path"] = self.evaluator_path
        fm["artifact_path"] = self.artifact_path
        fm["mode"] = self.mode
        fm["started_at"] = self.started_at
        fm["best_score"] = self.best_score
        return fm


# ---------------------------------------------------------------------------
# Module-private helpers
# ---------------------------------------------------------------------------

def _parse_frontmatter(raw: str) -> tuple:
    """Split a markdown-with-YAML-frontmatter string.

    Returns:
        A 2-tuple ``(frontmatter_dict, body_text)``.

    Raises:
        ValueError: If the file does not start with ``---``.
    """
    stripped = raw.strip()
    if not stripped.startswith("---"):
        raise ValueError("File does not contain YAML frontmatter (missing leading ---)")

    # Split on the *second* occurrence of "---"
    # The content is: "---\n<yaml>\n---\n<body>"
    # After stripping the leading "---\n", find the next "---".
    after_first = stripped[3:]  # skip leading "---"
    if after_first.startswith("\n"):
        after_first = after_first[1:]

    closing_idx = after_first.find("\n---")
    if closing_idx == -1:
        raise ValueError("File does not contain closing frontmatter delimiter (---)")

    yaml_text = after_first[:closing_idx]
    body = after_first[closing_idx + 4:]  # skip "\n---"

    # Strip leading newlines from body but preserve content
    body = body.lstrip("\n")
    # Strip single trailing newline if present (we add one on write)
    if body.endswith("\n"):
        body = body[:-1]

    frontmatter = yaml.safe_load(yaml_text) or {}
    return frontmatter, body
