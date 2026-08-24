"""Agentic evolutionary search primitives: score functions, lineages, and the variation loop.

Post-conditions give an AI Function correctness semantics; this module adds
*fitness*. A :class:`Score` is a correctness gate joined to a rankable value,
a :class:`LineageStore` is the contract for the append-only archive of
candidates that passed the commit gate (correct AND at least as good as the
best committed), and :func:`evolve` drives any variation operator — an
``@ai_function``, a ``ClaudeAgent`` session, plain Python — against a
caller-owned score function, following the Agentic Variation Operators
formulation (``Vary(P) = Agent(P, K, f)``, arXiv:2603.24517).

The module exports contracts, not policy. It does not grade candidates
(``score_fn`` is the caller's ``f``), does not prescribe the variation agent
(``K`` enters through the operator's own prompt and tools), and does not own
storage: :class:`Lineage` is the batteries-included in-memory/JSONL reference
implementation of :class:`LineageStore`, and opinionated backends (git,
artifact/revision stores) belong in downstream libraries.
"""

from __future__ import annotations

from .lineage import Lineage
from .loop import AttemptOutcome, EvolutionReport, VaryFn, evolve
from .types import CommittedVersion, EvolutionError, LineageStore, RejectedCommit, Score, ScoreFn

__all__ = [
    "AttemptOutcome",
    "CommittedVersion",
    "EvolutionError",
    "EvolutionReport",
    "Lineage",
    "LineageStore",
    "RejectedCommit",
    "Score",
    "ScoreFn",
    "VaryFn",
    "evolve",
]
