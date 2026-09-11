"""Remote failures carry a classification, so no caller matches exception names.

``ErrorFrame.error_kind`` and ``RemoteError.kind`` let a caller decide retry,
escalate or abort from a closed vocabulary. ``classify`` seeds that vocabulary
with the runtime's own errors and ``register_error_kind`` lets a host add the
model-provider and storage exceptions the wire layer cannot know about.

The field is defaulted on the frame, so a peer that sends no classification
still produces a valid frame and reads as ``"internal"``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator

import pytest
from pydantic import TypeAdapter, ValidationError

from ai_functions.network import (
    ConnectionClosedError,
    CoordinatorClient,
    CoordinatorEndpoint,
    ErrorFrame,
    RemoteError,
    classify,
    register_error_kind,
)
from ai_functions.network.channel import _error_frame, _rehydrate_error  # noqa: PLC2701
from ai_functions.network.error_kinds import _REGISTRY, coerce_error_kind  # noqa: PLC2701
from ai_functions.network.wire import Frame
from ai_functions.runtime.coordinator import InMemoryCoordinator
from ai_functions.runtime.errors import (
    ConnectionLostError,
    EventEmissionError,
    SerializationError,
    ThreadIdMismatchError,
    ThreadNotFoundError,
    WorkerLostError,
)
from ai_functions.types import EventKind, ThreadId, ThreadInfo, WorkerId

_FRAME_ADAPTER: TypeAdapter[Frame] = TypeAdapter(Frame)  # pyright: ignore[reportInvalidTypeForm]
_TID = ThreadId("thr-error-kinds")


@pytest.fixture(autouse=True)
def _pristine_registry() -> Iterator[None]:
    """Registrations are process-wide; undo this test's before the next one."""
    saved = dict(_REGISTRY)
    yield
    _REGISTRY.clear()
    _REGISTRY.update(saved)


class _ProviderThrottled(Exception):
    """Stands in for a model provider's throttling exception."""


class _ProviderThrottledSubclass(_ProviderThrottled):
    """A provider exception the host never registered by name."""


class _DeclaresItsKind(Exception):
    """Stands in for an exception this codebase owns, declaring its own kind."""

    error_kind = "worker_lost"


class _InheritsDeclaredKind(_DeclaresItsKind):
    """A subclass with no declaration of its own."""


# ── classify: seeded defaults ─────────────────────────────────────────────


def test_unregistered_exception_is_internal() -> None:
    """``classify`` is total; anything unclassified is ``internal``."""
    assert classify(Exception("boom")) == "internal"


@pytest.mark.parametrize(
    "exc, expected",
    [
        (asyncio.CancelledError(), "cancelled"),
        (ThreadNotFoundError(_TID), "not_found"),
        (WorkerLostError(WorkerId("w-1"), [_TID]), "worker_lost"),
        (ConnectionLostError("ws://host/rpc", 3), "connection_lost"),
        (ConnectionClosedError("channel closed"), "connection_lost"),
        (ValueError("bad argument"), "invalid_input"),
        (TypeError("bad argument"), "invalid_input"),
        (ThreadIdMismatchError(_TID, ThreadId("thr-other")), "invalid_input"),
        (EventEmissionError(EventKind.STARTED, _TID, "thread"), "invalid_input"),
        (SerializationError("my_fn", "not picklable"), "invalid_input"),
    ],
)
def test_runtime_errors_are_seeded(exc: BaseException, expected: str) -> None:
    """The runtime's own errors classify without any host registration."""
    assert classify(exc) == expected


def test_pydantic_validation_error_is_invalid_input() -> None:
    """A rejected params model is a caller-input failure, not an internal one."""
    with pytest.raises(ValidationError) as excinfo:
        _ = ThreadInfo.model_validate({"thread_id": None})
    assert classify(excinfo.value) == "invalid_input"


def test_declared_error_kind_classifies_without_registration() -> None:
    """A class this codebase owns declares ``error_kind`` on itself."""
    assert classify(_DeclaresItsKind("gone")) == "worker_lost"
    assert classify(_InheritsDeclaredKind("gone")) == "worker_lost"


def test_registration_overrides_a_declaration_on_the_same_class() -> None:
    """The registry wins over the class's own declaration at the same level."""
    register_error_kind(_DeclaresItsKind, "model_unavailable")
    assert classify(_DeclaresItsKind("gone")) == "model_unavailable"


# ── register_error_kind: host extension and the MRO walk ──────────────────


def test_registration_classifies_a_host_exception() -> None:
    """A host registers a provider exception instead of matching its name."""
    register_error_kind(_ProviderThrottled, "model_unavailable")
    assert classify(_ProviderThrottled("slow down")) == "model_unavailable"


def test_registration_reaches_subclasses_through_the_mro() -> None:
    """Registering a base class classifies every subclass of it."""
    register_error_kind(_ProviderThrottled, "model_unavailable")
    assert classify(_ProviderThrottledSubclass("slow down")) == "model_unavailable"


def test_most_derived_registration_wins() -> None:
    """A subclass's own registration is not shadowed by its base's."""
    register_error_kind(_ProviderThrottled, "model_unavailable")
    register_error_kind(_ProviderThrottledSubclass, "connection_lost")
    assert classify(_ProviderThrottledSubclass("dropped")) == "connection_lost"
    assert classify(_ProviderThrottled("slow down")) == "model_unavailable"


# ── ErrorFrame: the field is additive ─────────────────────────────────────


def test_error_frame_round_trips_with_kind() -> None:
    """A classified frame keeps its classification through JSON."""
    frame = ErrorFrame(id="c-1", type="_ProviderThrottled", message="slow down", error_kind="model_unavailable")
    decoded = _FRAME_ADAPTER.validate_json(frame.model_dump_json())
    assert isinstance(decoded, ErrorFrame)
    assert decoded.error_kind == "model_unavailable"


def test_error_frame_from_a_peer_that_sends_no_kind() -> None:
    """A frame without ``error_kind`` parses; the caller reads it as internal."""
    decoded = _FRAME_ADAPTER.validate_json('{"kind":"error","id":"c-1","type":"Whatever","message":"boom"}')
    assert isinstance(decoded, ErrorFrame)
    assert decoded.error_kind is None
    rehydrated = _rehydrate_error(decoded.type, decoded.message, decoded.error_kind)
    assert isinstance(rehydrated, RemoteError)
    assert rehydrated.kind == "internal"


def test_error_frame_with_a_kind_from_a_newer_vocabulary_still_parses() -> None:
    """An unknown kind must not reject the frame — that would hang the call."""
    decoded = _FRAME_ADAPTER.validate_json(
        '{"kind":"error","id":"c-1","type":"Whatever","message":"boom","error_kind":"brand_new_kind"}'
    )
    assert isinstance(decoded, ErrorFrame)
    assert decoded.error_kind == "brand_new_kind"
    rehydrated = _rehydrate_error(decoded.type, decoded.message, decoded.error_kind)
    assert isinstance(rehydrated, RemoteError)
    assert rehydrated.kind == "internal"


def test_coerce_error_kind_passes_known_kinds_through() -> None:
    """A kind in this build's vocabulary survives coercion unchanged."""
    assert coerce_error_kind("not_found") == "not_found"
    assert coerce_error_kind(None) == "internal"
    assert coerce_error_kind("brand_new_kind") == "internal"


def test_relayed_remote_error_keeps_its_type_and_kind() -> None:
    """A middle peer re-encodes a ``RemoteError`` without flattening it."""
    relayed = _error_frame("c-2", RemoteError("_ProviderThrottled", "slow down", "model_unavailable"))
    assert relayed.type == "_ProviderThrottled"
    assert relayed.message == "slow down"
    assert relayed.error_kind == "model_unavailable"


def test_known_exception_still_rehydrates_to_its_class() -> None:
    """A name this process knows becomes that class, not a ``RemoteError``."""
    rehydrated = _rehydrate_error("ThreadNotFoundError", "thr-1", "not_found")
    assert isinstance(rehydrated, ThreadNotFoundError)


def test_remote_connection_closed_error_stays_remote() -> None:
    """A peer's ``ConnectionClosedError`` arrives as a classified ``RemoteError``, not the local class."""
    rehydrated = _rehydrate_error("ConnectionClosedError", "channel closed", "connection_lost")
    assert isinstance(rehydrated, RemoteError)
    assert not isinstance(rehydrated, ConnectionClosedError)
    assert rehydrated.kind == "connection_lost"


# ── Over a real endpoint ──────────────────────────────────────────────────


async def test_remote_error_carries_kind_over_a_real_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unknown exception type raised on the endpoint arrives classified."""
    register_error_kind(_ProviderThrottled, "model_unavailable")

    async def _throttle(self: InMemoryCoordinator) -> list[ThreadInfo]:
        del self
        raise _ProviderThrottled("slow down")

    monkeypatch.setattr(InMemoryCoordinator, "list_threads", _throttle)

    async with CoordinatorEndpoint() as endpoint:
        await endpoint.start(host="127.0.0.1", port=0)
        client = await CoordinatorClient.connect(endpoint.url)
        async with client:
            with pytest.raises(RemoteError) as excinfo:
                _ = await client.list_threads()
            assert excinfo.value.remote_type == "_ProviderThrottled"
            assert excinfo.value.kind == "model_unavailable"


async def test_missing_thread_over_a_real_endpoint_is_not_found(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A seeded runtime error classifies without the host doing anything.

    ``ThreadNotFoundError`` is a name the client knows, so it rehydrates to its
    own class; the frame that carried it is still classified, which is what a
    peer that does not know the name reads.
    """
    frames: list[ErrorFrame] = []
    original = _error_frame

    def _capture(call_id: str, exc: BaseException) -> ErrorFrame:
        frame = original(call_id, exc)
        frames.append(frame)
        return frame

    monkeypatch.setattr("ai_functions.network.channel._error_frame", _capture)

    async with CoordinatorEndpoint() as endpoint:
        await endpoint.start(host="127.0.0.1", port=0)
        client = await CoordinatorClient.connect(endpoint.url)
        async with client:
            with pytest.raises(ThreadNotFoundError):
                _ = await client.get_thread_info(_TID)
    assert [f.error_kind for f in frames] == ["not_found"]
