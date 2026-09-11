"""Exception-to-:data:`~ai_functions.network.wire.ErrorKind` classification.

:func:`classify` maps an exception to the :data:`ErrorKind` a peer reads on the
``ErrorFrame`` that reports it, so callers branch on the kind instead of
matching the peer's exception class name.

Two sources feed the classification, checked per class along the MRO:

- An ``error_kind`` class attribute, declared by exception classes this
  codebase owns (the runtime errors in :mod:`ai_functions.runtime.errors`,
  :class:`~ai_functions.network.wire.ConnectionClosedError`).
- A process-wide registry for third-party classes: :func:`register_error_kind`
  lets the host that owns a model provider or storage client declare that
  provider's exceptions once. Registration is last-write-wins across the whole
  process.

Both apply to the whole MRO, so declaring or registering a base exception
classifies every subclass.

Usage::

    from botocore.exceptions import ClientError
    from ai_functions.network import register_error_kind

    register_error_kind(ClientError, "model_unavailable")

Invariants:
    A. ``classify`` is total — every ``BaseException`` maps to some
       ``ErrorKind``, defaulting to ``"internal"``, so a caller never has to
       handle "unclassified".
    B. The most derived classified ancestor wins, so registering or declaring
       a base class never overrides a more specific classification for a
       subclass. At the same MRO level a registration wins over the class's
       own declaration.
"""

from __future__ import annotations

import asyncio
from typing import cast, get_args

from pydantic import ValidationError

from .wire import ErrorKind

_VALID_KINDS: frozenset[str] = frozenset(get_args(ErrorKind))
"""The vocabulary this build of the library knows how to branch on."""

_REGISTRY: dict[type[BaseException], ErrorKind] = {
    asyncio.CancelledError: "cancelled",
    # Argument-shaped failures: the same call with the same arguments fails again.
    ValidationError: "invalid_input",
    ValueError: "invalid_input",
    TypeError: "invalid_input",
}
"""Seeded with stdlib/pydantic classes; hosts extend it via
:func:`register_error_kind`. Classes this codebase owns declare their own
``error_kind`` attribute instead."""


def coerce_error_kind(value: str | None) -> ErrorKind:
    """Read a wire ``error_kind`` value into this build's vocabulary.

    Args:
        value: The frame's ``error_kind``; ``None`` from a peer that sends none.

    Returns:
        ``value`` when it is one of this build's :data:`ErrorKind` values,
        ``"internal"`` otherwise.
    """
    if value in _VALID_KINDS:
        return cast("ErrorKind", value)
    return "internal"


def classify(exc: BaseException) -> ErrorKind:
    """Classify ``exc`` by walking its MRO against the registry and declarations.

    Args:
        exc: The exception a peer is about to report, or reported.

    Returns:
        The kind registered or declared (``error_kind`` class attribute) for
        the most derived class in ``type(exc).__mro__`` that has one;
        ``"internal"`` when none does.
    """
    for base in type(exc).__mro__:
        kind = _REGISTRY.get(base)
        if kind is not None:
            return kind
        declared = vars(base).get("error_kind")
        if isinstance(declared, str) and declared in _VALID_KINDS:
            return cast("ErrorKind", declared)
    return "internal"


def register_error_kind(exc_type: type[BaseException], kind: ErrorKind) -> None:
    """Declare that ``exc_type`` and its subclasses classify as ``kind``.

    Call this once at process start for every third-party exception whose class
    the wire layer cannot know: a model provider's throttling and access errors
    (``model_unavailable``), a storage client's missing-key error
    (``not_found``). The registry is process-wide: the registration replaces any
    previous one for the same type — last write wins — and shadows
    registrations or declarations on its base classes.

    Args:
        exc_type: Exception class to classify.
        kind: The classification a caller should see for it.
    """
    _REGISTRY[exc_type] = kind
