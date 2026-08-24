"""The scored lineage: an append-only archive of committed versions.

This is the population ``P`` of agentic evolutionary search
(AVO, arXiv:2603.24517) in its single-lineage form: a sequence of
``(candidate, score)`` pairs where a new entry is admitted only through the
commit gate — the candidate passed correctness AND matched or improved on
the best committed score. Failed attempts stay in the caller's trajectory;
they never enter the lineage.

The lineage is deliberately not a :class:`~ai_functions.memory.base.MemoryBackend`:
a backend stores named parameters that optimizers rewrite, while a lineage is
an immutable history that only grows. The two compose — a TextGrad optimizer
can tune the *prompts* of the variation function across runs while the
lineage records what those runs produced.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from .types import CommittedVersion, RejectedCommit, Score


class Lineage:
    """Append-only, score-gated archive of committed candidates.

    Thread-safe: ``commit`` holds a lock across the gate check and the
    append, so concurrent variation steps cannot both commit against the
    same stale best.

    Args:
        path: When given, every commit is also appended to this JSONL file
            (one ``CommittedVersion`` per line), and an existing file is
            loaded on construction — the paper persists each committed
            version as a git commit; a JSONL file is the library-neutral
            equivalent.
        minimum_improvement: How much a candidate must beat the best committed
            value by. The default ``0.0`` admits ties, matching the paper's
            "matches or improves" rule.
    """

    def __init__(self, path: str | Path | None = None, minimum_improvement: float = 0.0) -> None:
        self._versions: list[CommittedVersion] = []
        self._lock = threading.Lock()
        self._path = Path(path) if path is not None else None
        self._minimum_improvement = minimum_improvement
        if self._path is not None and self._path.exists():
            for line in self._path.read_text().splitlines():
                if line.strip():
                    self._versions.append(CommittedVersion.model_validate_json(line))

    # -- Reading ---------------------------------------------------------------

    def __len__(self) -> int:
        """Number of committed versions."""
        return len(self._versions)

    @property
    def versions(self) -> tuple[CommittedVersion, ...]:
        """Every committed version, oldest first."""
        return tuple(self._versions)

    @property
    def best(self) -> CommittedVersion | None:
        """The highest-scoring committed version (latest wins ties), or ``None`` when empty."""
        if not self._versions:
            return None
        return max(reversed(self._versions), key=lambda v: v.score.value)

    def as_context(self, k: int | None = None) -> str:
        """Render the lineage for a variation prompt.

        Returns the last ``k`` committed versions (all when ``None``) as a
        compact text block — version number, score, metrics, and notes — the
        piece of ``P`` a variation agent consults to decide what to try next.
        Candidates themselves are not inlined (they may be large objects);
        callers who want the best candidate's content pass it separately.
        """
        versions = self._versions if k is None else self._versions[-k:]
        if not versions:
            return "The lineage is empty: produce the seed version."
        lines = ["Committed lineage (oldest first):"]
        for v in versions:
            metrics = f" metrics={v.score.metrics}" if v.score.metrics else ""
            notes = f" — {v.score.notes}" if v.score.notes else ""
            lines.append(f"  v{v.version}: score={v.score.value:g}{metrics}{notes}")
        return "\n".join(lines)

    # -- The commit gate ---------------------------------------------------------

    def admits(self, score: Score) -> bool:
        """Whether the gate would admit a candidate with this score right now."""
        if not score.correct:
            return False
        best = self.best
        return best is None or score.value >= best.score.value + self._minimum_improvement

    def commit(self, candidate: Any, score: Score, parent_version: int | None = None) -> CommittedVersion:  # pyright: ignore[reportExplicitAny]
        """Admit a candidate through the gate and append it.

        Args:
            candidate: The result to commit.
            score: The score that justifies the commit.
            parent_version: Version the candidate was derived from; defaults
                to the current best's version.

        Returns:
            The appended ``CommittedVersion``.

        Raises:
            RejectedCommit: The candidate failed correctness or scored below
                the best committed value (plus ``minimum_improvement``).
        """
        with self._lock:
            if not score.correct:
                raise RejectedCommit(f"candidate failed correctness: {score.notes or 'no notes'}")
            best = self.best
            if best is not None and score.value < best.score.value + self._minimum_improvement:
                raise RejectedCommit(
                    f"score {score.value:g} does not improve on committed best {best.score.value:g} (v{best.version})"
                )
            version = CommittedVersion(
                version=len(self._versions) + 1,
                candidate=candidate,
                score=score,
                parent_version=parent_version if parent_version is not None else (best.version if best else None),
            )
            self._versions.append(version)
            if self._path is not None:
                with self._path.open("a") as f:
                    _ = f.write(json.dumps(version.model_dump(mode="json")) + "\n")
            return version
