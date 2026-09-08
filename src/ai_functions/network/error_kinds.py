"""Exception-to-:data:`~ai_functions.network.wire.ErrorKind` classification registry.

An ``ErrorFrame`` carries the peer's exception class *name*, which is all a
different process can send. A caller that must decide retry-versus-abort from
that name ends up matching strings — ``"ThrottlingException"``, a ``"botocore."``
prefix, ``"ValidationException" in name`` — and every such match breaks when a
provider renames or wraps an exception. :func:`classify` answers the same
question from the exception's type, and :func:`register_error_kind` lets the host
that owns a model provider or storage client declare that provider's exceptions
once, at the process that raises them, where the real class is in scope.

Registration is process-wide and applies to the whole MRO of the registered
type, so registering a provider's base exception classifies every subclass.

Usage::

    from botocore.exceptions import ClientError
    from ai_functions.network import register_error_kind

    register_error_kind(ClientError, "model_unavailable")

This module keeps :mod:`ai_functions.network.wire` free of runtime imports: the
frame schemas stay transport- and runtime-agnostic while the seed registry here
names the runtime's own error classes.

Invariants:
    A. ``classify`` is total — every ``BaseException`` maps to some
       ``ErrorKind``, defaulting to ``"internal"``, so a caller never has to
       handle "unclassified".
    B. The most derived registered ancestor wins, so registering a base class
       never overrides a more specific registration for a subclass.
"""

from __future__ import annotations

import asyncio

from pydantic import ValidationError

from ..runtime.errors import (
    ConnectionLostError,
    EventEmissionError,
    SerializationError,
    ThreadIdMismatchError,
    ThreadNotFoundError,
    WorkerLostError,
)
from .wire import ConnectionClosedError, ErrorKind

_REGISTRY: dict[type[BaseException], ErrorKind] = {
    asyncio.CancelledError: "cancelled",
    ThreadNotFoundError: "not_found",
    WorkerLostError: "worker_lost",
    ConnectionLostError: "connection_lost",
    ConnectionClosedError: "connection_lost",
    # Argument-shaped failures: the same call with the same arguments fails again.
    ValidationError: "invalid_input",
    ValueError: "invalid_input",
    TypeError: "invalid_input",
    ThreadIdMismatchError: "invalid_input",
    EventEmissionError: "invalid_input",
    SerializationError: "invalid_input",
}
"""Seeded with the runtime's own errors; hosts extend it via :func:`register_error_kind`."""


def classify(exc: BaseException) -> ErrorKind:
    """Classify ``exc`` by walking its MRO against the registry.

    Args:
        exc: The exception a peer is about to report, or reported.

    Returns:
        The kind registered for the most derived class in ``type(exc).__mro__``
        that has one; ``"internal"`` when none does.
    """
    for base in type(exc).__mro__:
        kind = _REGISTRY.get(base)
        if kind is not None:
            return kind
    return "internal"


def register_error_kind(exc_type: type[BaseException], kind: ErrorKind) -> None:
    """Declare that ``exc_type`` and its subclasses classify as ``kind``.

    Call this once at process start for every third-party exception whose class
    the wire layer cannot know: a model provider's throttling and access errors
    (``model_unavailable``), a storage client's missing-key error
    (``not_found``). The registration replaces any previous one for the same
    type and shadows registrations for its base classes.

    Args:
        exc_type: Exception class to classify.
        kind: The classification a caller should see for it.
    """
    _REGISTRY[exc_type] = kind
