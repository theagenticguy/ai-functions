"""``CustomEvent`` routing survives serialization and a real WebSocket hop.

A custom event declares the routing fields every other event declares, so the
three places routing is read all agree on it: ``EVENT_ADAPTER`` keeps
``thread_id`` in the wire dict, a filtered subscription over a
``CoordinatorClient`` matches it, and ``get_events(thread_id)`` returns it under
that thread. A client-side ``append_event`` of a routed custom event is accepted
by the remote coordinator; an unrouted one is rejected and the rejection shows
up on ``client.append_errors``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable

import pytest

from ai_functions.network import CoordinatorClient, CoordinatorEndpoint
from ai_functions.network.channel import EVENT_ADAPTER
from ai_functions.testing import RuntimeHarness
from ai_functions.types import CustomEvent, Event, ThreadId
from ai_functions.types.ids import MessageId

_TID = ThreadId("thr-custom-wire")


class _Annotated(CustomEvent):
    """A user subclass that declares its own field next to the routing ones."""

    step: str = ""


async def _until(predicate: Callable[[], bool], *, timeout: float = 2.0) -> None:
    """Poll ``predicate`` until true, or fail the test after ``timeout``."""
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition not reached within timeout")


# ── a: the adapter keeps routing on a stamped custom event ────────────────


def test_a1_dump_includes_thread_id() -> None:
    """The wire dict for a stamped custom event carries ``thread_id``."""
    stamped = CustomEvent(kind="my_kind", payload={"a": 1}).model_copy(update={"thread_id": _TID})
    dumped = EVENT_ADAPTER.dump_python(stamped)
    assert dumped["thread_id"] == _TID


def test_a2_round_trip_preserves_routing() -> None:
    """Every routing field survives dump → validate through the union adapter."""
    original = CustomEvent(
        kind="my_kind",
        thread_id=_TID,
        thread_name="worker-a",
        message_id=MessageId("msg-1"),
        payload={"a": 1},
    )
    reparsed = EVENT_ADAPTER.validate_python(EVENT_ADAPTER.dump_python(original))
    assert isinstance(reparsed, CustomEvent)
    assert reparsed.thread_id == _TID
    assert reparsed.thread_name == "worker-a"
    assert reparsed.message_id == MessageId("msg-1")
    assert reparsed.id == original.id
    assert reparsed.payload == {"a": 1}


def test_a3_subclass_declared_fields_round_trip() -> None:
    """A subclass's own field stays declared; routing rides alongside it."""
    original = _Annotated(kind="my_kind", thread_id=_TID, step="validate", payload={"a": 1})
    dumped = _Annotated.model_validate(original.model_dump())
    assert dumped.step == "validate"
    assert dumped.thread_id == _TID
    assert dumped.payload == {"a": 1}
    # Through the ``Event`` union the subclass degrades to ``CustomEvent``;
    # ``step`` is an undeclared key there, so it lands in the payload.
    via_union = EVENT_ADAPTER.validate_python(original.model_dump())
    assert isinstance(via_union, CustomEvent)
    assert via_union.thread_id == _TID
    assert via_union.payload == {"a": 1, "step": "validate"}


# ── b: the runtime gate stamps the declared field ─────────────────────────


async def test_b1_route_event_stamps_custom_event() -> None:
    """An unrouted custom event leaves the runtime gate carrying the thread id."""
    async with RuntimeHarness() as h:
        h.worker._route_event(  # noqa: SLF001 -- the gate is the unit under test
            CustomEvent(kind="my_kind", payload={"a": 1}),
            thread_id=_TID,
            source="thread",
        )
        stored = await h.coordinator.get_events(_TID)
        assert [e.thread_id for e in stored] == [_TID]


# ── c: a real endpoint + client agree on the routing ─────────────────────


async def test_c1_filtered_subscription_receives_custom_event() -> None:
    """``client.on(cb, thread_id=...)`` fires for a custom event from the endpoint."""
    async with CoordinatorEndpoint() as endpoint:
        await endpoint.start(host="127.0.0.1", port=0)
        client = await CoordinatorClient.connect(endpoint.url)
        async with client:
            received: list[Event] = []
            with client.on(received.append, thread_id=_TID):
                endpoint.coordinator.append_event(
                    CustomEvent(kind="my_kind", thread_id=_TID, payload={"a": 1}),
                )
                await _until(lambda: bool(received))
            event = received[0]
            assert isinstance(event, CustomEvent)
            assert event.thread_id == _TID
            assert event.payload == {"a": 1}


async def test_c2_get_events_returns_routed_custom_event() -> None:
    """``client.get_events(thread_id)`` returns the custom event with its routing."""
    async with CoordinatorEndpoint() as endpoint:
        await endpoint.start(host="127.0.0.1", port=0)
        client = await CoordinatorClient.connect(endpoint.url)
        async with client:
            endpoint.coordinator.append_event(
                CustomEvent(kind="my_kind", thread_id=_TID, payload={"a": 1}),
            )
            replayed = await client.get_events(_TID)
            assert [e.thread_id for e in replayed] == [_TID]
            assert [e.kind for e in replayed] == ["my_kind"]


async def test_c3_client_append_event_lands_on_the_endpoint() -> None:
    """A routed custom event appended from the client reaches the server log."""
    async with CoordinatorEndpoint() as endpoint:
        await endpoint.start(host="127.0.0.1", port=0)
        client = await CoordinatorClient.connect(endpoint.url)
        async with client:
            client.append_event(CustomEvent(kind="from_client", thread_id=_TID, payload={"a": 1}))
            await _until(lambda: bool(endpoint.coordinator._events.get(_TID)))  # noqa: SLF001
            stored = await endpoint.coordinator.get_events(_TID)
            assert [(e.kind, e.thread_id) for e in stored] == [("from_client", _TID)]
            assert client.append_errors == ()


async def test_c4_unrouted_append_is_recorded_on_append_errors(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A rejected append is observable without watching the log by hand."""
    async with CoordinatorEndpoint() as endpoint:
        await endpoint.start(host="127.0.0.1", port=0)
        client = await CoordinatorClient.connect(endpoint.url)
        async with client:
            with caplog.at_level("ERROR", logger="ai_functions.network.client"):
                client.append_event(CustomEvent(kind="unrouted", payload={"a": 1}))
                await _until(lambda: bool(client.append_errors))
            assert "thread_id" in str(client.append_errors[0])
            assert "append_event RPC failed" in caplog.text
            assert await endpoint.coordinator.get_events(_TID) == []
