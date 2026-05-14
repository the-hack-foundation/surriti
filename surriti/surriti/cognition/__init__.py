"""Cognitive abstraction layer for Surriti.

This package implements the second-pass "cognitive summarizer" that turns
literal, episodic transcript extractions into a continuously-synthesized
semantic self-model: traits, goals, beliefs, emotional valence,
procedural interaction patterns, decay-weighted associative memory,
domain-labelled clusters, consolidated abstractions, and a small
predictive bundle.

Everything here is **internal**: the public ``Surriti`` surface
(``add_episode``, ``recall``, ``search``, ``build_communities``,
``upsert_user``, ``retrieve_episodes``) is unchanged. Callers that
already use Surriti pick up the richer behaviour transparently.

Entry points
------------

- :class:`CognitionScheduler` -- per-``group_id`` debounced background
  task launched by ``Surriti.connect()`` and cancelled by ``close()``.
- :class:`CognitionConfig` -- aggregated tunables passed to ``Surriti``
  via the ``cognition=`` constructor parameter (or simply
  ``cognition=False`` to opt out entirely).
- :func:`run_cognition_pass` -- the thing the scheduler invokes (and
  what tests call directly via ``Surriti._cognition.run_once(...)``).

The package deliberately keeps a small public API for the rest of
surriti to call, but exports nothing from the top-level
``surriti`` package -- this layer is implementation detail.
"""

from __future__ import annotations

from surriti.cognition.config import CognitionConfig
from surriti.cognition.scheduler import CognitionScheduler
from surriti.cognition.runner import run_cognition_pass

__all__ = ["CognitionConfig", "CognitionScheduler", "run_cognition_pass"]
