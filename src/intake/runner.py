from __future__ import annotations

from collections.abc import Iterator
from typing import Any


def iter_run(payload: dict[str, Any], app: Any = None) -> Iterator[tuple[str, Any]]:
    """Run the intake graph and yield progress as it happens.

    Yields ('event', dict) for every pipeline event emitted by the nodes and the
    tool calls inside them, then a final ('result', state) with the same state that
    ``app.invoke(payload)`` would return. Nothing else about the run changes.
    """
    if app is None:
        from .graph import APP as app  # imported lazily so tests can pass their own graph

    state: dict[str, Any] = dict(payload)
    for mode, data in app.stream(payload, stream_mode=['updates', 'custom']):
        if mode == 'custom':
            if isinstance(data, dict):
                yield 'event', data
        elif mode == 'updates':
            for delta in data.values():
                if delta:  # a node that changes nothing is reported as None
                    state.update(delta)
    yield 'result', state
