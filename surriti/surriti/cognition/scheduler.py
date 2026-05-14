"""Per-group debounced background scheduler for the cognitive layer.

The scheduler is owned by a ``Surriti`` instance. ``Surriti.connect()``
calls ``start()``; ``Surriti.close()`` calls ``shutdown()``. After
every successful ``Surriti.add_episode()`` call, the engine invokes
``notify(group_id, episode_uuid)``. The scheduler then either:

- Force-fires immediately if the per-group buffer has reached
  ``CognitionConfig.batch_threshold``, or
- Waits ``CognitionConfig.idle_seconds`` after the most recent notify
  and then fires.

A single fire = one ``run_cognition_pass``. All failures are caught
inside the runner; the scheduler itself never raises.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from surriti.cognition.config import CognitionConfig
from surriti.cognition.runner import run_cognition_pass
from surriti.cognition.state import GroupState

logger = logging.getLogger("surriti.cognition")


class CognitionScheduler:
    def __init__(
        self,
        *,
        driver: Any,
        llm: Any,
        embedder: Any,
        config: CognitionConfig,
    ) -> None:
        self._driver = driver
        self._llm = llm
        self._embedder = embedder
        self._config = config
        self._states: dict[str, GroupState] = {}
        self._tasks: dict[str, asyncio.Task] = {}
        self._lock = asyncio.Lock()
        self._stopped = False

    @property
    def enabled(self) -> bool:
        return self._config.enabled

    def start(self) -> None:
        """Mark the scheduler as accepting notifications. No background
        task is created until the first ``notify`` arrives."""

        self._stopped = False

    def notify(self, group_id: str, episode_uuid: str | None = None) -> None:
        """Record a new episode for ``group_id`` and arm a debounced
        cognition pass. Safe to call from inside ``add_episode``; never
        raises."""

        if not self._config.enabled or self._stopped:
            return
        try:
            state = self._states.setdefault(group_id, GroupState(group_id=group_id))
            state.mark_dirty(episode_uuid)
            # Spawn or refresh the debounce task.
            existing = self._tasks.get(group_id)
            if existing and not existing.done():
                # Cancel and restart so the idle window resets.
                existing.cancel()
            self._tasks[group_id] = asyncio.create_task(
                self._debounced_run(group_id), name=f"cognition:{group_id}"
            )
        except RuntimeError:
            # No running event loop (e.g. add_episode invoked outside an
            # async context in tests). Silently skip.
            logger.debug("cognition notify skipped: no running loop")
        except Exception:
            logger.exception("cognition notify failed for group=%s", group_id)

    async def _debounced_run(self, group_id: str) -> None:
        try:
            state = self._states.get(group_id)
            if state is None:
                return
            # If batch threshold already reached, fire immediately.
            if len(state.pending_episode_uuids) < self._config.batch_threshold:
                try:
                    await asyncio.sleep(self._config.idle_seconds)
                except asyncio.CancelledError:
                    return
            await self._run_once_locked(group_id)
        except asyncio.CancelledError:
            return
        except Exception:
            logger.exception("cognition debounce run failed for group=%s", group_id)

    async def run_once(self, group_id: str) -> dict[str, Any] | None:
        """Force a synchronous cognition pass for ``group_id`` (used by
        tests and explicit caller-driven cadence)."""

        if not self._config.enabled:
            return None
        # Cancel any pending debounce task so it cannot race with us
        # for the same buffer of pending episode UUIDs.
        existing = self._tasks.pop(group_id, None)
        if existing and not existing.done():
            existing.cancel()
            try:
                await existing
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        return await self._run_once_locked(group_id)

    async def _run_once_locked(self, group_id: str) -> dict[str, Any]:
        async with self._lock:
            state = self._states.setdefault(
                group_id, GroupState(group_id=group_id)
            )
            episode_uuids = state.take()
            metrics = await run_cognition_pass(
                driver=self._driver,
                llm=self._llm,
                embedder=self._embedder,
                group_id=group_id,
                episode_uuids=episode_uuids,
                config=self._config,
                pass_count=state.pass_count,
            )
            return metrics

    async def shutdown(self) -> None:
        """Cancel any in-flight tasks. Idempotent."""

        self._stopped = True
        tasks = list(self._tasks.values())
        for t in tasks:
            if not t.done():
                t.cancel()
        for t in tasks:
            try:
                await t
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        self._tasks.clear()
