"""In-process pub/sub event bus feeding the SSE endpoint.

Pipeline stages run in a worker thread; FastAPI endpoints run on the asyncio
loop. `publish()` is thread-safe and forwards events to every subscriber
queue for the given project.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger("spatialforge.events")


class EventBus:
    def __init__(self) -> None:
        self._subs: dict[str, set[asyncio.Queue]] = {}
        self._loop: asyncio.AbstractEventLoop | None = None

    def attach_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    async def subscribe(self, project_id: str) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=2000)
        self._subs.setdefault(project_id, set()).add(q)
        return q

    def unsubscribe(self, project_id: str, q: asyncio.Queue) -> None:
        self._subs.get(project_id, set()).discard(q)
        if not self._subs.get(project_id):
            self._subs.pop(project_id, None)

    def publish(self, project_id: str, event_type: str, data: dict[str, Any]) -> None:
        event = {"type": event_type, "projectId": project_id, "data": data}
        queues = list(self._subs.get(project_id, ()))
        if not queues:
            return
        loop = self._loop

        def _put() -> None:
            for q in queues:
                try:
                    q.put_nowait(event)
                except asyncio.QueueFull:
                    logger.warning("SSE queue overflow for project %s", project_id)

        if loop is not None and loop.is_running():
            loop.call_soon_threadsafe(_put)
        # no loop attached yet (e.g. unit tests): drop silently

    def subscriber_count(self, project_id: str) -> int:
        return len(self._subs.get(project_id, ()))


bus = EventBus()
