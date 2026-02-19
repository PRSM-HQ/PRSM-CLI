"""Async event bus bridging orchestrator callbacks to TUI consumers.

The engine runs in a background worker and fires events via callback.
The EventBus queues them for the TUI's event consumer loop.
"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Any

from prsm.adapters.events import OrchestratorEvent, dict_to_event

logger = logging.getLogger(__name__)


class EventBus:
    """Async queue bridging engine callbacks to TUI event consumers."""

    def __init__(self, maxsize: int = 5000) -> None:
        self._queue: asyncio.Queue[OrchestratorEvent] = asyncio.Queue(
            maxsize=maxsize
        )
        self._closed = False
        self._drained = asyncio.Event()
        self._drained.set()  # Initially drained (empty queue)

    async def _callback(self, data: dict[str, Any]) -> None:
        """Callback to pass to EngineConfig.event_callback."""
        if self._closed:
            return
        event = dict_to_event(data)
        try:
            # Use await put() with timeout to add backpressure instead of dropping
            await asyncio.wait_for(self._queue.put(event), timeout=30.0)
        except asyncio.TimeoutError:
            logger.error(
                "EventBus queue blocked for 30s, dropping: %s (queue size: %d)",
                event.event_type,
                self._queue.qsize(),
            )
        except Exception as e:
            logger.error("EventBus callback error: %s", e)

    def make_callback(self):
        """Return the async callback for EngineConfig.event_callback."""
        return self._callback

    async def emit(self, event: OrchestratorEvent) -> None:
        """Manually emit an event (for TUI-generated events)."""
        if self._closed:
            return
        try:
            # Use await put() with timeout to add backpressure instead of dropping
            await asyncio.wait_for(self._queue.put(event), timeout=30.0)
        except asyncio.TimeoutError:
            logger.error(
                "EventBus queue blocked for 30s, dropping: %s (queue size: %d)",
                event.event_type,
                self._queue.qsize(),
            )
        except Exception as e:
            logger.error("EventBus emit error: %s", e)

    async def consume(self) -> AsyncIterator[OrchestratorEvent]:
        """Yield events as they arrive. Stops on close()."""
        while not self._closed:
            try:
                event = await asyncio.wait_for(
                    self._queue.get(), timeout=0.5
                )
                yield event
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

    def close(self) -> None:
        """Stop the consumer loop permanently."""
        self._closed = True

    def reset(self) -> None:
        """Reset the bus for a new orchestration run.

        Drains any leftover events and re-opens the bus.
        """
        # Drain the queue
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except Exception:
                break
        self._closed = False
