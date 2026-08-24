"""Tests for the thread-backed variation adapters: thread_vary and notify_on_stall."""

from __future__ import annotations

from ai_functions import ai_function
from ai_functions.experimental.evolution import (
    Score,
    evolve,
    notify_on_stall,
    thread_vary,
)
from ai_functions.testing import RuntimeHarness, ScriptedModel, Turn
from ai_functions.types import EventKind
from ai_functions.types.events import MessageUserEvent


async def _user_texts(h: RuntimeHarness, thread_id: object) -> list[str]:
    """All MESSAGE_USER texts in a thread's log, in order."""
    events = await h.events(thread_id, kinds=[EventKind.MESSAGE_USER])  # type: ignore[arg-type]
    return [e.text for e in events if isinstance(e, MessageUserEvent)]


@ai_function[str](structured_output=False)
def _propose(lineage_context: str) -> str:
    """Propose the next candidate.\n\n{lineage_context}"""


async def test_thread_vary_runs_every_step_on_one_thread() -> None:
    """Three variation steps land on the same thread, accumulating history."""
    async with RuntimeHarness() as h:
        model = ScriptedModel([Turn(text="alpha"), Turn(text="beta"), Turn(text="gamma")])
        handle = await h.spawn(_propose.replace(model=model))

        scores = iter([1.0, 2.0, 3.0])

        def score_fn(candidate: str) -> Score:
            return Score(correct=True, value=next(scores), notes=candidate)

        lineage, report = await evolve(thread_vary(handle), score_fn, steps=3)

        assert report.committed == 3
        assert [v.candidate.strip() for v in lineage.versions] == ["alpha", "beta", "gamma"]
        # All three runs are cycles of the SAME thread — one shared event log.
        user_events = [
            e for e in await h.events(handle.id, kinds=[EventKind.MESSAGE_USER]) if isinstance(e, MessageUserEvent)
        ]
        assert len(user_events) == 3


async def test_thread_vary_renders_lineage_into_the_prompt() -> None:
    """Step 2's prompt carries step 1's committed score via as_context."""
    async with RuntimeHarness() as h:
        model = ScriptedModel([Turn(text="first"), Turn(text="second")])
        handle = await h.spawn(_propose.replace(model=model))

        def score_fn(candidate: str) -> Score:
            return Score(correct=True, value=float(len(candidate)), notes=candidate.strip())

        _ = await evolve(thread_vary(handle), score_fn, steps=2)

        user_events = [
            e for e in await h.events(handle.id, kinds=[EventKind.MESSAGE_USER]) if isinstance(e, MessageUserEvent)
        ]
        assert "lineage is empty" in user_events[0].text
        assert "v1" in user_events[1].text and "first" in user_events[1].text


async def test_thread_vary_custom_render() -> None:
    """A custom render maps the lineage to arbitrary run kwargs."""
    async with RuntimeHarness() as h:
        model = ScriptedModel([Turn(text="candidate")])
        handle = await h.spawn(_propose.replace(model=model))

        def render(lineage: object) -> dict[str, str]:
            return {"lineage_context": "CUSTOM RENDER"}

        def score_fn(candidate: str) -> Score:
            return Score(correct=True, value=1.0)

        _ = await evolve(thread_vary(handle, render=render), score_fn, steps=1)

        user_events = [
            e for e in await h.events(handle.id, kinds=[EventKind.MESSAGE_USER]) if isinstance(e, MessageUserEvent)
        ]
        assert "CUSTOM RENDER" in user_events[0].text


async def test_notify_on_stall_surfaces_in_the_next_cycle() -> None:
    """The supervisor message is buffered by notify and observed by the NEXT variation step.

    ``ThreadHandle.notify`` appends to the thread's inject buffer; the runtime
    flushes that buffer at the next cycle's model-call boundary as its own
    MESSAGE_USER turn, ahead of the cycle's prompt. So the redirection
    composed at the stall threshold (after step 4 here) must appear in the
    log between step 4's prompt and step 5's prompt.
    """
    async with RuntimeHarness() as h:
        model = ScriptedModel([Turn(text="a"), Turn(text="b"), Turn(text="c"), Turn(text="d"), Turn(text="e")])
        handle = await h.spawn(_propose.replace(model=model))

        # Seed commits, then three consecutive rejections trigger the supervisor.
        outcomes = iter(
            [
                Score(correct=True, value=1.0),
                Score(correct=False, notes="regression on cfg-32k"),
                Score(correct=False, notes="regression on cfg-32k"),
                Score(correct=False, notes="regression on cfg-32k"),
                Score(correct=True, value=2.0),
            ]
        )

        def score_fn(candidate: str) -> Score:
            return next(outcomes)

        _, report = await evolve(
            thread_vary(handle),
            score_fn,
            steps=5,
            stall_after=3,
            on_stall=notify_on_stall(handle),
        )

        assert report.committed == 2
        texts = await _user_texts(h, handle.id)
        # 5 variation prompts + 1 injected supervisor turn.
        assert len(texts) == 6
        supervisor_text = texts[4]
        assert "[SUPERVISOR]" in supervisor_text
        assert "3 consecutive proposals" in supervisor_text
        assert "regression on cfg-32k" in supervisor_text
        assert "best committed score is 1" in supervisor_text
        # It surfaces BEFORE step 5's prompt, and no other turn carries it.
        assert "Propose the next candidate" in texts[5]
        assert all("[SUPERVISOR]" not in t for t in texts[:4] + texts[5:])


async def test_notify_on_stall_custom_compose() -> None:
    """A caller-supplied compose overrides the default supervisor message."""
    async with RuntimeHarness() as h:
        model = ScriptedModel([Turn(text="x"), Turn(text="y"), Turn(text="z")])
        handle = await h.spawn(_propose.replace(model=model))

        def score_fn(candidate: str) -> Score:
            return Score(correct=False, notes="nope")

        def compose(lineage: object, report: object) -> str:
            return "try simulated annealing"

        _ = await evolve(
            thread_vary(handle),
            score_fn,
            steps=3,
            stall_after=2,
            on_stall=notify_on_stall(handle, compose=compose),
        )

        texts = await _user_texts(h, handle.id)
        assert any("simulated annealing" in t for t in texts)
