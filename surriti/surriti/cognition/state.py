"""Per-group dirty-tracking state for the cognition scheduler.

A tiny in-memory ledger -- one entry per ``group_id`` -- recording how
many episodes have been queued since the last cognition pass and the
timestamp of the most recent ``notify``. This is intentionally not
persisted: on process restart cognition simply runs once at the next
ingest for any group with new episodes.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class GroupState:
    """Mutable per-group dirty marker."""

    group_id: str
    pending_episode_uuids: list[str] = field(default_factory=list)
    last_notify_at: float = 0.0
    pass_count: int = 0
    """Total cognition passes that have completed for this group; used by
    the periodic domain-labelling cadence."""

    def mark_dirty(self, episode_uuid: str | None) -> None:
        if episode_uuid:
            self.pending_episode_uuids.append(episode_uuid)
        self.last_notify_at = time.monotonic()

    def take(self) -> list[str]:
        """Atomically claim the pending episodes and reset the buffer."""

        pending = self.pending_episode_uuids
        self.pending_episode_uuids = []
        self.pass_count += 1
        return pending
