"""The variation loop: drive a variation operator against a score function and a lineage.

One :func:`evolve` call is the outer loop of agentic evolutionary search
(AVO, arXiv:2603.24517): each step hands the current lineage to a
caller-supplied variation operator (any async callable — an ``@ai_function``,
a ``ClaudeAgent``-backed closure, plain Python), scores the candidate it
returns with the caller's ``score_fn``, and commits it when the gate admits.
The agent stays generic; the domain enters only through the variation
prompt and ``score_fn`` — the paper's ``Vary(P) = Agent(P, K, f)`` with
``K`` folded into the operator's own prompt and tools.

Failed and non-improving candidates are recorded in the returned
:class:`EvolutionReport` but never enter the lineage.
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from .lineage import Lineage
from .types import LineageStore, Score, ScoreFn

VaryFn = Callable[[LineageStore], Awaitable[Any]]
"""Variation operator: consult the lineage, return the next candidate."""


@dataclass(frozen=True)
class AttemptOutcome:
    """One variation step's result: the score, and the committed version if admitted."""

    step: int
    score: Score
    committed_version: int | None
    """Version number when the gate admitted the candidate; ``None`` otherwise."""


@dataclass
class EvolutionReport:
    """What an :func:`evolve` run tried and what it committed."""

    attempts: list[AttemptOutcome] = field(default_factory=list)

    @property
    def committed(self) -> int:
        """Number of attempts the gate admitted."""
        return sum(1 for a in self.attempts if a.committed_version is not None)

    @property
    def consecutive_rejections(self) -> int:
        """Length of the trailing run of non-committed attempts (the stall signal)."""
        n = 0
        for attempt in reversed(self.attempts):
            if attempt.committed_version is not None:
                break
            n += 1
        return n


async def _score(score_fn: ScoreFn, candidate: Any) -> Score:  # pyright: ignore[reportExplicitAny]
    """Invoke a sync-or-async score function and normalize exceptions to a failing score.

    A raising ``score_fn`` is an incorrect candidate, not a crashed loop —
    the exception text lands in ``Score.notes`` so the next variation step
    can read what went wrong, mirroring how a failing post-condition's
    message is fed back to the agent.
    """
    try:
        result = score_fn(candidate)
        if inspect.isawaitable(result):
            result = await result
    except Exception as exc:  # noqa: BLE001 — the score IS the error channel
        return Score(correct=False, notes=f"score_fn raised {type(exc).__name__}: {exc}")
    return result


async def evolve(
    vary: VaryFn,
    score_fn: ScoreFn,
    lineage: LineageStore | None = None,
    *,
    steps: int = 10,
    stall_after: int | None = None,
    on_stall: Callable[[LineageStore, EvolutionReport], Awaitable[None]] | None = None,
) -> tuple[LineageStore, EvolutionReport]:
    """Run ``steps`` variation steps, committing improvements to the lineage.

    Args:
        vary: The variation operator. Receives the lineage (use
            ``lineage.as_context()`` in prompts, ``lineage.best`` for the
            current champion) and returns the next candidate.
        score_fn: The scoring function ``f``. A raised exception is treated
            as a failed correctness check, with the message in ``Score.notes``.
        lineage: Any :class:`~.types.LineageStore` implementation; a fresh
            in-memory :class:`Lineage` when ``None``.
        steps: Variation steps to run.
        stall_after: When this many consecutive attempts fail to commit,
            invoke ``on_stall`` (the paper's conditional supervisor seam) —
            or, with no ``on_stall``, stop early.
        on_stall: Intervention called at the stall threshold, typically an
            agent that reviews the report and injects a redirection (e.g. via
            ``ThreadHandle.notify`` on the variation thread). After it runs,
            the rejection counter is considered reset.

    Returns:
        The lineage and a report of every attempt.
    """
    lineage = lineage if lineage is not None else Lineage()
    report = EvolutionReport()
    rejections_at_last_intervention = 0
    for step in range(1, steps + 1):
        candidate = await vary(lineage)
        score = await _score(score_fn, candidate)
        version: int | None = None
        if lineage.admits(score):
            version = lineage.commit(candidate, score).version
        report.attempts.append(AttemptOutcome(step=step, score=score, committed_version=version))
        if stall_after is not None:
            stalled_for = report.consecutive_rejections - rejections_at_last_intervention
            if stalled_for >= stall_after:
                if on_stall is None:
                    break
                await on_stall(lineage, report)
                rejections_at_last_intervention = report.consecutive_rejections
    return lineage, report
