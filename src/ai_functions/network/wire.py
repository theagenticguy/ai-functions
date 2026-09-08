"""Wire protocol — pydantic frames for the Coordinator/Worker network layer.

The network layer is a symmetric, bidirectional RPC channel over a single
WebSocket. Either peer (client or endpoint) may initiate a call; the
other answers. Events broadcast from endpoint to client are the same
frame shape with no correlation expected.

Frames are discriminated on ``kind``:

- ``call``  — request: method name + JSON params. Awaits a ``result``
  or ``error`` frame with matching ``id``.
- ``result`` — success response; carries a JSON value.
- ``error``  — failure response; carries an error type, a message and an
  :data:`ErrorKind` classification the caller can branch on without
  matching the peer's exception class name.
- ``event``  — server-initiated event broadcast; no response expected.

A single ``Binary`` field is used for cloudpickle payloads (Spawnables,
run-cycle args/kwargs/results) which cannot be JSON-encoded.

Encoding: ``frame.model_dump_json()``
Decoding: ``TypeAdapter(Frame).validate_json(raw)``

The wire layer is transport-agnostic: any async duplex string channel
satisfying :class:`Transport` is acceptable. The default implementation
uses ``websockets``; FastAPI / Starlette WebSockets can be plugged in
via a thin adapter.
"""

from __future__ import annotations

import base64
from typing import Annotated, Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, BeforeValidator, Field, PlainSerializer

DEFAULT_HOST: str = "127.0.0.1"
DEFAULT_PORT: int = 9900


# ── Binary escape type ───────────────────────────────────────────────────────


def _decode_binary(value: object) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, str):
        return base64.b64decode(value.encode("ascii"), validate=True)
    raise TypeError(f"Binary field expects bytes or base64 str, got {type(value).__name__}")


def _encode_binary(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


Binary = Annotated[
    bytes,
    BeforeValidator(_decode_binary),
    PlainSerializer(_encode_binary, return_type=str, when_used="json"),
]
"""Binary field encoded as base64 in JSON; accepts raw bytes on construction."""


# ── Error classification ─────────────────────────────────────────────────────

ErrorKind = Literal[
    "cancelled",
    "not_found",
    "terminated",
    "invalid_input",
    "model_unavailable",
    "worker_lost",
    "connection_lost",
    "internal",
]
"""Transport-independent class of a remote failure, carried on every ``ErrorFrame``.

A caller decides retry, escalate or abort from this value, so a peer's exception
class name never has to be string-matched. The names an
:class:`ai_functions.network.error_kinds` registry maps to each kind are the
host's to extend; the vocabulary itself is closed:

- ``cancelled`` — the remote work was cancelled cooperatively.
- ``not_found`` — the named thread, worker or record does not exist there.
- ``terminated`` — the thread existed and is gone; retrying the same id cannot work.
- ``invalid_input`` — the call's arguments are wrong; retrying them cannot work.
- ``model_unavailable`` — a model provider refused or throttled the call; retry later.
- ``worker_lost`` — the worker hosting the thread died.
- ``connection_lost`` — the transport dropped; the call's outcome is unknown.
- ``internal`` — anything unclassified, which is the default a peer that sends no
  ``kind`` at all is read as.
"""


# ── Frames ────────────────────────────────────────────────────────────────────


class CallFrame(BaseModel):
    """Outbound RPC request.

    Fields:
        kind: Discriminator (always ``"call"``).
        id: Correlation id chosen by the caller; echoed in the matching
            ``result`` or ``error`` frame.
        method: Dotted method name the peer should dispatch
            (e.g. ``"coordinator.spawn"``, ``"worker.submit"``).
        params: JSON-compatible keyword arguments. Non-JSON values
            (cloudpickle blobs) are carried as ``Binary`` base64 fields
            inside this dict at method-specific keys.
    """

    kind: Literal["call"] = "call"
    id: str
    method: str
    params: dict[str, Any]  # pyright: ignore[reportExplicitAny]


class ResultFrame(BaseModel):
    """Successful RPC response.

    Fields:
        kind: Discriminator (always ``"result"``).
        id: Matches the ``id`` of the originating ``CallFrame``.
        value: JSON-compatible return value; may contain ``Binary``
            blobs at method-specific keys.
    """

    kind: Literal["result"] = "result"
    id: str
    value: Any = None  # pyright: ignore[reportExplicitAny]


class ErrorFrame(BaseModel):
    """Failed RPC response.

    Fields:
        kind: Discriminator (always ``"error"``).
        id: Matches the ``id`` of the originating ``CallFrame``.
        type: Exception class name (e.g. ``"ThreadNotFoundError"``,
            ``"ValueError"``). Used on the caller side to reconstruct a
            typed exception when possible; otherwise surfaced as
            :class:`RemoteError`.
        message: Human-readable error description.
        error_kind: Classification of the failure (see :data:`ErrorKind`).
            Defaults to ``None``, which a reader treats as ``"internal"``, so a
            peer that sends no classification still produces a valid frame.
    """

    kind: Literal["error"] = "error"
    id: str
    type: str
    message: str
    error_kind: ErrorKind | None = None


class EventFrame(BaseModel):
    """Broadcast from endpoint to client.

    Fields:
        kind: Discriminator (always ``"event"``).
        event: One :class:`ai_functions.types.Event` pydantic model, serialized
            in place (the wire layer calls ``model_dump`` / ``model_validate``
            on the tagged union).

    No correlation id — events are fire-and-forget from the sender's POV.
    """

    kind: Literal["event"] = "event"
    event: Any  # pyright: ignore[reportExplicitAny]  # ai_functions.types.Event tagged union


Frame = Annotated[
    CallFrame | ResultFrame | ErrorFrame | EventFrame,
    Field(discriminator="kind"),
]


# ── Transport protocol ───────────────────────────────────────────────────────


@runtime_checkable
class Transport(Protocol):
    """Thin async duplex string channel.

    Both sides of the wire layer talk to a ``Transport``. The default
    implementation wraps a ``websockets`` client / server socket; a
    FastAPI / Starlette WebSocket can be adapted to this protocol with a
    trivial wrapper (both expose ``send`` / ``recv`` with different
    method names).

    Implementations must be safe to call concurrently for ``send`` and
    ``recv`` from different tasks (full-duplex).
    """

    async def send(self, text: str) -> None:
        """Send one frame's serialized JSON over the wire.

        Args:
            text: The ``frame.model_dump_json()`` output.
        """
        ...

    async def recv(self) -> str:
        """Receive one frame's serialized JSON from the wire.

        Returns:
            The next frame's JSON text, as sent by the peer.

        Raises:
            ConnectionClosedError: The peer closed the connection.
        """
        ...

    async def close(self) -> None:
        """Close the underlying connection idempotently."""
        ...


# ── Error hierarchy ──────────────────────────────────────────────────────────


class WireError(Exception):
    """Base class for wire-layer errors."""


class RemoteError(WireError):
    """A peer returned an ``ErrorFrame`` whose ``type`` we cannot rehydrate.

    ``kind`` is what a caller branches on: the local process has no class for
    ``remote_type``, so classifying the failure by name is the only alternative
    and it breaks the moment the peer renames or wraps an exception.

    Attributes:
        remote_type (str): The ``type`` field from the ErrorFrame — the
            peer's exception class name.
        message (str): The ``message`` field from the ErrorFrame.
        kind (ErrorKind): Classification of the failure; ``"internal"`` for a
            peer that sent no ``error_kind``.

    Args:
        remote_type: The ``type`` field from the ErrorFrame.
        message: The ``message`` field from the ErrorFrame.
        kind: The frame's ``error_kind``, or ``"internal"`` when it carried none.
    """

    remote_type: str
    message: str
    kind: ErrorKind

    def __init__(self, remote_type: str, message: str, kind: ErrorKind = "internal") -> None:
        self.remote_type = remote_type
        self.message = message
        self.kind = kind
        super().__init__(f"{remote_type}: {message}")


class ConnectionClosedError(WireError):
    """The peer connection closed before a pending call resolved."""
