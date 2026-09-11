"""Network layer — WebSocket RPC between coordinators and workers.

This package provides the transport so :class:`~ai_functions.protocols.Coordinator`
clients and remote :class:`~ai_functions.runtime.worker.LocalWorker` s can talk
to a server-side :class:`~ai_functions.runtime.coordinator.InMemoryCoordinator`.

- :class:`CoordinatorEndpoint` — server; fronts an in-memory coordinator
  and accepts WebSocket connections.
- :class:`CoordinatorClient` — client; implements :class:`Coordinator`
  by proxying every method over the wire.
- :class:`WireChannel` — symmetric bidirectional RPC machinery used by
  both sides.
- Frame models live in :mod:`ai_functions.network.wire`; the
  :data:`ErrorKind` classification a host extends with
  :func:`register_error_kind` lives in
  :mod:`ai_functions.network.error_kinds`.

See the module-level docstrings for design details.
"""

from __future__ import annotations

from .channel import WireChannel
from .client import CoordinatorClient
from .endpoint import CoordinatorEndpoint
from .error_kinds import classify, register_error_kind
from .wire import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    CallFrame,
    ConnectionClosedError,
    ErrorFrame,
    ErrorKind,
    EventFrame,
    Frame,
    RemoteError,
    ResultFrame,
    Transport,
    WireError,
)

__all__ = [
    "CallFrame",
    "ConnectionClosedError",
    "CoordinatorClient",
    "CoordinatorEndpoint",
    "DEFAULT_HOST",
    "DEFAULT_PORT",
    "ErrorFrame",
    "ErrorKind",
    "EventFrame",
    "Frame",
    "RemoteError",
    "ResultFrame",
    "Transport",
    "WireChannel",
    "WireError",
    "classify",
    "register_error_kind",
]
