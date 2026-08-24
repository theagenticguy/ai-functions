"""Agentic evolutionary search primitives: score functions, lineages, and the variation loop.

Post-conditions give an AI Function correctness semantics; this module adds
*fitness*. A :class:`Score` is a correctness gate joined to a rankable value,
a :class:`Lineage` is the append-only archive of candidates that passed the
commit gate (correct AND at least as good as the best committed), and
:func:`evolve` drives any variation operator — an ``@ai_function``, a
``ClaudeAgent`` session, plain Python — against a caller-owned score
function, following the Agentic Variation Operators formulation
(``Vary(P) = Agent(P, K, f)``, arXiv:2603.24517).

The module deliberately does not grade candidates and does not prescribe the
variation agent: the score function ``f`` and the knowledge base ``K`` are
the caller's, which is the paper's central design property (the agent stays
generic; the domain enters as data).
"""

from __future__ import annotations

from .lineage import Lineage
from .loop import AttemptOutcome, EvolutionReport, VaryFn, evolve
from .types import CommittedVersion, EvolutionError, RejectedCommit, Score, ScoreFn

__all__ = [
    "AttemptOutcome",
    "CommittedVersion",
    "EvolutionError",
    "EvolutionReport",
    "Lineage",
    "RejectedCommit",
    "Score",
    "ScoreFn",
    "VaryFn",
    "evolve",
]
