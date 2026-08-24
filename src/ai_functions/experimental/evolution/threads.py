"""Thread-backed variation: stateful operators and the notify-based supervisor.

:func:`~.loop.evolve` accepts any async callable as the variation operator. A
one-shot ``@ai_function`` call works, but it forgets everything between steps
— each proposal starts cold. The AVO paper's operator is the opposite: one
long-lived agent whose accumulated context (prior edits, profiler output,
failed directions) is what makes late-stage variation smart, and whose
supervisor injects course corrections into that same context.

This module is the two-adapter bridge between :func:`~.loop.evolve` and AI
Threads:

- :func:`thread_vary` turns a spawned :class:`~ai_functions.handle.ThreadHandle`
  into a ``VaryFn``. Every variation step runs on the *same* thread, so the
  operator's history accumulates across steps exactly like the paper's
  single-lineage agent (and summarization compacts it when it grows long).
- :func:`notify_on_stall` builds an ``on_stall`` callback that delivers a
  redirection into the variation thread via ``handle.notify`` — the paper's
  conditional supervisor as a side-channel message, surfaced to the agent on
  its next cycle without interrupting the loop.

Both are thin: the contracts stay in :mod:`.loop` and :mod:`.types`; this
module only binds them to the thread runtime.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from .loop import EvolutionReport, VaryFn
from .types import LineageStore

if TYPE_CHECKING:
    from ..handle import ThreadHandle  # noqa: F401 — sphinx cross-reference

# Maps the lineage to the keyword arguments of one thread run. The default
# passes ``lineage_context=lineage.as_context(k)``.
RenderFn = Callable[[LineageStore], dict[str, Any]]


def thread_vary(
    handle: Any,  # pyright: ignore[reportExplicitAny] — ThreadHandle[P, T]; typed loosely to avoid a runtime import cycle
    render: RenderFn | None = None,
    *,
    context_window: int | None = 5,
) -> VaryFn:
    """Adapt a live thread handle into a stateful variation operator.

    Each call runs one cycle on ``handle``'s thread with the rendered
    lineage as arguments; the thread keeps its full conversation history
    between steps, so proposal N sees what happened to proposals 1..N-1
    beyond what the lineage alone records.

    Args:
        handle: A spawned ``ThreadHandle`` for the proposing function. The
            caller owns its lifecycle (``spawn`` before, ``terminate`` after).
        render: Maps the lineage to the run's keyword arguments. Defaults to
            ``{"lineage_context": lineage.as_context(context_window)}`` — the
            proposing function then declares a ``lineage_context: str``
            parameter and interpolates it in its docstring prompt.
        context_window: ``k`` passed to ``as_context`` by the default
            renderer; ignored when ``render`` is given.

    Returns:
        A ``VaryFn`` for :func:`~.loop.evolve`.
    """

    async def vary(lineage: LineageStore) -> Any:  # pyright: ignore[reportExplicitAny]
        kwargs = render(lineage) if render is not None else {"lineage_context": lineage.as_context(context_window)}
        return await handle.run(**kwargs)

    return vary


def notify_on_stall(
    handle: Any,  # pyright: ignore[reportExplicitAny] — ThreadHandle[P, T]
    compose: Callable[[LineageStore, EvolutionReport], str] | None = None,
) -> Callable[[LineageStore, EvolutionReport], Any]:  # pyright: ignore[reportExplicitAny]
    """Build an ``on_stall`` intervention that notifies the variation thread.

    The returned callback composes a redirection message and delivers it via
    ``handle.notify`` — a side-channel the thread surfaces on its next cycle,
    so the intervention lands inside the operator's own context rather than
    resetting it. This is the smallest faithful version of the paper's
    supervisor; a richer one (an agent that reads the trajectory and writes
    the redirection itself) is a ``compose`` implementation away.

    Args:
        handle: The variation thread to notify.
        compose: Builds the message from the lineage and report. The default
            names the stall length, the best committed score, and the recent
            failure notes, and asks the operator to change strategy.

    Returns:
        An async callback suitable for ``evolve(..., on_stall=...)``.
    """

    def _default_compose(lineage: LineageStore, report: EvolutionReport) -> str:
        best = lineage.best
        best_line = (
            f"best committed score is {best.score.value:g} (v{best.version})" if best else "nothing committed yet"
        )
        recent_notes = [a.score.notes for a in report.attempts[-3:] if a.score.notes]
        notes_block = ("\nRecent failures:\n" + "\n".join(f"- {n}" for n in recent_notes)) if recent_notes else ""
        return (
            f"[SUPERVISOR] {report.consecutive_rejections} consecutive proposals failed to commit; "
            f"{best_line}.{notes_block}\n"
            "Your current line of exploration has plateaued. Review what the failures have in "
            "common and try a structurally different approach on the next proposal."
        )

    async def on_stall(lineage: LineageStore, report: EvolutionReport) -> None:
        message = (compose or _default_compose)(lineage, report)
        await handle.notify(message)

    return on_stall
