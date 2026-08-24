"""Data types for the evolution module: scores and committed versions.

Post-conditions accept or reject a result; a :class:`Score` additionally
*ranks* it. This is the shape of the scoring function ``f`` in agentic
evolutionary search (AVO, arXiv:2603.24517): a correctness gate joined to a
fitness value, where a candidate that fails correctness scores zero
regardless of its measured fitness.

Invariants:
    V1 — correctness dominates fitness. ``Score.correct is False`` forces
    ``Score.value`` to ``0.0`` at construction, so no consumer can rank an
    incorrect candidate above a correct one.

    V2 — a committed version is immutable. The lineage appends; it never
    rewrites history.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ai_functions.ai_thread.errors import AIFunctionError


class Score(BaseModel):
    """Outcome of scoring one candidate: a correctness gate plus a fitness value.

    ``metrics`` carries the per-configuration vector behind the scalar (the
    paper scores a kernel per benchmark configuration and gates on the
    aggregate); ``value`` is the scalar the lineage ranks by, higher is
    better.
    """

    model_config = ConfigDict(frozen=True)

    correct: bool
    """Whether the candidate passed the correctness check. Gates the commit."""
    value: float = 0.0
    """Scalar fitness, higher is better. Forced to 0.0 when ``correct`` is false (V1)."""
    metrics: dict[str, float] = Field(default_factory=dict)
    """Optional per-configuration breakdown behind ``value``."""
    notes: str = ""
    """Free-form diagnostics (profiler output, failing case), fed back to the agent."""

    @model_validator(mode="after")
    def _zero_incorrect(self) -> Score:
        if not self.correct and self.value != 0.0:
            object.__setattr__(self, "value", 0.0)
        return self


ScoreFn = Callable[..., "Score | Awaitable[Score]"]
"""Callable scoring a candidate result.

Receives the candidate as the first positional argument. May be sync or
async; the loop awaits awaitables. This is the caller-owned ``f``: the
evolution module never grades candidates itself, exactly as post-conditions
are caller-owned validators.
"""


class CommittedVersion(BaseModel):
    """One committed entry in a lineage: a candidate and the score that admitted it.

    Immutable (V2). ``version`` is 1-based and dense: the lineage assigns it
    at commit time.
    """

    model_config = ConfigDict(frozen=True)

    version: int
    candidate: Any = None  # pyright: ignore[reportExplicitAny]
    """The committed result, exactly as the variation step produced it."""
    score: Score
    parent_version: int | None = None
    """Version this candidate was derived from; ``None`` for the seed."""


class EvolutionError(AIFunctionError):
    """Base of the evolution module's failure modes."""


class RejectedCommit(EvolutionError):
    """A candidate failed the commit gate (incorrect, or below the best committed score)."""


@runtime_checkable
class LineageStore(Protocol):
    """The storage contract a variation loop evolves against.

    This is the interoperability boundary: :func:`~.loop.evolve` depends only
    on this protocol, and what two implementations must agree on is exactly
    these five members — the gate semantics (``admits``: correct AND at least
    the best committed value), the append (``commit``, which must raise
    :class:`RejectedCommit` on a gate failure), and the three reads a
    variation operator needs. *Where* versions live — memory, JSONL, a git
    repository, an artifact/revision store — is an implementation decision
    that stays outside this library; :class:`~.lineage.Lineage` is the
    batteries-included reference.
    """

    def __len__(self) -> int:
        """Number of committed versions."""
        ...

    @property
    def best(self) -> CommittedVersion | None:
        """The highest-scoring committed version, or ``None`` when empty."""
        ...

    def as_context(self, k: int | None = None) -> str:
        """Render the last ``k`` committed versions (all when ``None``) for a variation prompt."""
        ...

    def admits(self, score: Score) -> bool:
        """Whether the commit gate would admit a candidate with this score right now."""
        ...

    def commit(self, candidate: Any, score: Score, parent_version: int | None = None) -> CommittedVersion:  # pyright: ignore[reportExplicitAny]
        """Admit a candidate through the gate and append it; raise :class:`RejectedCommit` otherwise."""
        ...
