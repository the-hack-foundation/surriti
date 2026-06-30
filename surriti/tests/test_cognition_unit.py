"""Pure unit tests for the cognitive abstraction layer.

These exercise the standalone helpers (decay math, affect lexicon,
procedural classifier, belief regex, jsonio, scheduler lifecycle) that
do not require a SurrealDB driver. Driver-coupled passes
(reinforcement, traits, goals, consolidation) are covered indirectly
through `test_pipeline_fake_driver` once the cognition layer is wired
end-to-end with a real backing store; we only verify here that the
in-process pieces obey the contracts the runner relies on.
"""

from __future__ import annotations

import asyncio
import math
from datetime import datetime, timedelta, timezone

import pytest

from surriti.cognition import CognitionConfig, CognitionScheduler
from surriti.cognition._jsonio import parse_json_loose, snake_case
from surriti.cognition.affect import score_affect
from surriti.cognition.self_awareness import _render_episodes_for_llm
from surriti.cognition.decay import effective_confidence, half_life_for
from surriti.cognition.perspective import looks_like_belief
from surriti.cognition.procedural import classify_episode
from surriti.cognition.state import GroupState
from surriti.edges import EntityEdge


# ---------------------------------------------------------------- decay math
def _edge(**kw) -> EntityEdge:
    base = dict(
        uuid="e1",
        group_id="g",
        source_node_uuid="s",
        target_node_uuid="o",
        name="likes",
        fact="s likes o",
        confidence=1.0,
        reinforcement_count=1,
        stability="episodic",
        valid_at=datetime.now(timezone.utc),
        created_at=datetime.now(timezone.utc),
    )
    base.update(kw)
    return EntityEdge(**base)


def test_half_life_defaults_and_overrides():
    assert half_life_for("episodic") == 30.0
    assert half_life_for("reinforced") == 90.0
    assert half_life_for("persistent") == 365.0
    assert math.isinf(half_life_for("consolidated"))
    assert half_life_for("episodic", {"episodic": 7.0}) == 7.0
    # Unknown stability falls through to episodic.
    assert half_life_for("nonsense") == 30.0


def test_effective_confidence_consolidated_never_decays():
    now = datetime(2026, 5, 1, tzinfo=timezone.utc)
    long_ago = now - timedelta(days=900)
    edge = _edge(stability="consolidated", last_reinforced_at=long_ago, confidence=0.8)
    assert effective_confidence(edge, now=now) == pytest.approx(0.8, abs=1e-6)


def test_effective_confidence_decays_over_time():
    now = datetime(2026, 5, 1, tzinfo=timezone.utc)
    fresh = _edge(last_reinforced_at=now, confidence=1.0)
    stale = _edge(last_reinforced_at=now - timedelta(days=120), confidence=1.0)
    assert effective_confidence(fresh, now=now) > effective_confidence(stale, now=now)


def test_effective_confidence_reinforcement_boost():
    now = datetime(2026, 5, 1, tzinfo=timezone.utc)
    once = _edge(last_reinforced_at=now, reinforcement_count=1, confidence=0.5)
    many = _edge(last_reinforced_at=now, reinforcement_count=20, confidence=0.5)
    assert effective_confidence(many, now=now) > effective_confidence(once, now=now)


def test_effective_confidence_uses_bounded_recall_boost():
    now = datetime(2026, 5, 1, tzinfo=timezone.utc)
    stale = _edge(
        last_reinforced_at=now - timedelta(days=120),
        confidence=0.5,
        recall_count=0,
    )
    recalled = _edge(
        last_reinforced_at=now - timedelta(days=120),
        last_recalled_at=now,
        confidence=0.5,
        recall_count=1000,
    )
    assert effective_confidence(recalled, now=now) > effective_confidence(stale, now=now)
    assert effective_confidence(recalled, now=now) <= 1.0


def test_effective_confidence_clipped_to_unit_interval():
    now = datetime(2026, 5, 1, tzinfo=timezone.utc)
    edge = _edge(last_reinforced_at=now, reinforcement_count=10_000, confidence=1.0)
    val = effective_confidence(edge, now=now)
    assert 0.0 <= val <= 1.0


# ---------------------------------------------------------------- affect
def test_score_affect_neutral_for_empty():
    assert score_affect("") == {}
    assert score_affect("   ") == {}


def test_score_affect_detects_excitement_and_frustration():
    pos = score_affect("This is awesome, I love it -- so excited!")
    neg = score_affect("Ugh, this is frustrating and broken, I hate it.")
    # Both must produce some signal
    assert pos and neg
    assert pos.get("polarity", 0) > 0
    assert neg.get("polarity", 0) < 0
    assert pos.get("intensity", 0) > 0
    assert neg.get("intensity", 0) > 0


# ---------------------------------------------------------------- perspective
@pytest.mark.parametrize(
    "text",
    [
        "I think this is a good idea",
        "I believe we should ship",
        "It feels like the team is stuck",
        "It seems like rain",
        "Probably we should reconsider",
    ],
)
def test_looks_like_belief_positive(text):
    assert looks_like_belief(text)


@pytest.mark.parametrize(
    "text",
    [
        "Alice works at Acme",
        "The meeting is at 3pm",
        "Bob bought a car",
    ],
)
def test_looks_like_belief_negative(text):
    assert not looks_like_belief(text)


# ---------------------------------------------------------------- procedural
def test_classify_episode_returns_known_label():
    label = classify_episode("can you help me make this faster and more efficient?")
    assert isinstance(label, str)
    assert label  # non-empty


def test_classify_episode_iterative():
    label = classify_episode("actually let's tweak that, change the wording again")
    assert label  # any of iterative_refinement / clarification etc.


# ---------------------------------------------------------------- jsonio
def test_parse_json_loose_handles_fences_and_prose():
    raw = "Sure, here it is:\n```json\n{\"a\": 1, \"b\": [2,3]}\n```\nthanks!"
    assert parse_json_loose(raw) == {"a": 1, "b": [2, 3]}


def test_parse_json_loose_returns_none_on_garbage():
    assert parse_json_loose("not json at all") is None
    assert parse_json_loose(None) is None


def test_snake_case_normalises():
    assert snake_case("Distance Running") == "distance_running"
    assert snake_case("FortNite_BR") == "fortnite_br"
    assert snake_case("  multi   space\tword ") == "multi_space_word"


def test_render_structured_self_episode_accepts_string_confidence():
    text = _render_episodes_for_llm([
        {
            "source_description": "legba/post_turn_reflection",
            "content": (
                '{"kind":"reflective_self_observation",'
                '"lesson_candidates":[{"lesson":"Lead with implementation.",'
                '"confidence":"0.86","evidence":"User asked for code."}],'
                '"future_behavior_adjustments":["Start with the patch."]}'
            ),
        }
    ])

    assert "lesson (conf=0.86): Lead with implementation." in text
    assert "adjustment: Start with the patch." in text


# ---------------------------------------------------------------- state
def test_group_state_mark_dirty_and_take():
    s = GroupState(group_id="g")
    s.mark_dirty("u1")
    s.mark_dirty("u2")
    taken = s.take()
    assert sorted(taken) == ["u1", "u2"]
    assert s.take() == []  # cleared
    assert s.pass_count == 2


# ---------------------------------------------------------------- scheduler
class _StubDriver:
    embedding_dim = 8

    async def query(self, surql, variables=None):
        return []


@pytest.mark.asyncio
async def test_scheduler_disabled_is_noop():
    sch = CognitionScheduler(
        driver=_StubDriver(), llm=None, embedder=None,
        config=CognitionConfig(enabled=False),
    )
    sch.start()
    sch.notify("g", "u1")  # must not raise
    out = await sch.run_once("g")
    assert out is None
    await sch.shutdown()


@pytest.mark.asyncio
async def test_scheduler_force_run_returns_metrics():
    from surriti.embedder import DummyEmbedder
    from surriti.llm import DummyLLMClient

    sch = CognitionScheduler(
        driver=_StubDriver(),
        llm=DummyLLMClient(),
        embedder=DummyEmbedder(embedding_dim=8),
        config=CognitionConfig(enabled=True, idle_seconds=0.01, batch_threshold=1),
    )
    sch.start()
    metrics = await sch.run_once("g")
    assert isinstance(metrics, dict)
    assert metrics["group_id"] == "g"
    assert "duration_ms" in metrics
    await sch.shutdown()


@pytest.mark.asyncio
async def test_scheduler_marks_recovered_episodes_processed():
    from surriti.embedder import DummyEmbedder
    from surriti.llm import DummyLLMClient
    from surriti.testing import InMemoryDriver

    driver = InMemoryDriver()
    driver.records["episode"].append(
        {
            "uuid": "ep-1",
            "group_id": "g",
            "name": "ep",
            "content": "Alice met Bob.",
        }
    )
    sch = CognitionScheduler(
        driver=driver,
        llm=DummyLLMClient(),
        embedder=DummyEmbedder(embedding_dim=8),
        config=CognitionConfig(
            enabled=True,
            idle_seconds=60,
            batch_threshold=99,
            self_awareness=False,
            trait_synthesis=False,
            goal_synthesis=False,
            procedural_synthesis=False,
            consolidation=False,
            prediction=False,
        ),
    )
    sch.start()
    recovered = await sch.recover_pending_episodes()
    metrics = await sch.run_once("g")

    assert recovered == 1
    assert metrics["episodes"] == 1
    assert driver.records["episode"][0]["cognition_processed_at"] is not None
    assert driver.records["episode"][0]["cognition_version"]
    await sch.shutdown()


@pytest.mark.asyncio
async def test_scheduler_notify_debounces_and_fires():
    from surriti.embedder import DummyEmbedder
    from surriti.llm import DummyLLMClient

    sch = CognitionScheduler(
        driver=_StubDriver(),
        llm=DummyLLMClient(),
        embedder=DummyEmbedder(embedding_dim=8),
        config=CognitionConfig(enabled=True, idle_seconds=0.05, batch_threshold=99),
    )
    sch.start()
    sch.notify("g", "u1")
    sch.notify("g", "u2")
    # Wait past the debounce window.
    await asyncio.sleep(0.25)
    await sch.shutdown()
    # No exceptions and the in-flight task drained cleanly.


# ---------------------------------------------------------------- config
def test_cognition_config_defaults():
    cfg = CognitionConfig()
    assert cfg.enabled is True
    assert cfg.batch_threshold >= 1
    assert cfg.idle_seconds > 0
    assert cfg.consolidation_threshold >= 2


def test_relation_frame_is_managed_table():
    from surriti.schema import ALL_TABLES

    assert "relation_frame" in ALL_TABLES


@pytest.mark.asyncio
async def test_reinforce_edges_on_recall_updates_recall_side_fields():
    from surriti.cognition.reinforcement import reinforce_edges_on_recall
    from surriti.testing import InMemoryDriver

    driver = InMemoryDriver()
    driver.records["relates_to"].append(
        {"uuid": "edge-1", "group_id": "g", "recall_count": 2}
    )

    updated = await reinforce_edges_on_recall(
        driver,
        group_id="g",
        edge_uuids=["edge-1", "edge-1"],
    )

    assert updated == 1
    row = driver.records["relates_to"][0]
    assert row["recall_count"] == 3
    assert row["last_recalled_at"] is not None


# ---------------------------------------------------------------- engine wiring
@pytest.mark.asyncio
async def test_surriti_cognition_disabled_skips_scheduler():
    from surriti.embedder import DummyEmbedder
    from surriti.graphiti import Surriti
    from surriti.testing import InMemoryDriver

    s = Surriti(
        InMemoryDriver(),
        embedder=DummyEmbedder(embedding_dim=8),
        cognition=False,
    )
    await s.connect()
    assert s._cognition is not None
    assert s._cognition.enabled is False
    # add_episode must not raise even with cognition disabled.
    res = await s.add_episode(
        name="ep1",
        episode_body="Alice met Bob.",
        group_id="g1",
    )
    assert res.episode is not None
    await s.close()


@pytest.mark.asyncio
async def test_surriti_cognition_enabled_default():
    from surriti.embedder import DummyEmbedder
    from surriti.graphiti import Surriti
    from surriti.testing import InMemoryDriver

    s = Surriti(InMemoryDriver(), embedder=DummyEmbedder(embedding_dim=8))
    await s.connect()
    assert s._cognition is not None
    assert s._cognition.enabled is True
    await s.close()


@pytest.mark.asyncio
async def test_recall_uses_cognition_decay_config(monkeypatch):
    from surriti.cognition import CognitionConfig
    from surriti.embedder import DummyEmbedder
    from surriti.graphiti import Surriti
    from surriti.search import SearchResults
    from surriti.testing import InMemoryDriver

    seen = {}

    async def fake_hybrid_search(driver, **kwargs):
        del driver
        cfg = kwargs["config"]
        seen["decay_aware"] = cfg.decay_aware
        seen["overrides"] = cfg.decay_half_life_overrides
        return SearchResults()

    monkeypatch.setattr("surriti.graphiti.hybrid_search", fake_hybrid_search)
    s = Surriti(
        InMemoryDriver(),
        embedder=DummyEmbedder(embedding_dim=8),
        cognition=CognitionConfig(
            enabled=True,
            decay_aware_recall=False,
            decay_half_life_days={"episodic": 7.0},
        ),
    )

    await s.recall("anything", group_id="g")

    assert seen["decay_aware"] is False
    assert seen["overrides"] == {"episodic": 7.0}


@pytest.mark.asyncio
async def test_self_awareness_uses_synthesize_hook():
    from surriti.cognition.self_awareness import _extract_self_traits
    from surriti.testing import InMemoryDriver

    class SynthOnlyLLM:
        async def synthesize(self, system, user):
            assert "self-observations" in system.lower()
            assert "I am concise" in user
            return '{"traits":[{"trait":"concise","confidence":0.9}],"beliefs":[]}'

    driver = InMemoryDriver()
    driver.records["entity"].append(
        {
            "uuid": "self-1",
            "group_id": "g",
            "name": "assistant_g",
            "summary": "",
            "labels": ["SelfEntity", "Assistant"],
        }
    )

    traits, beliefs = await _extract_self_traits(
        driver,
        SynthOnlyLLM(),
        "g",
        [{"content": "I am concise.", "source_description": "note"}],
        CognitionConfig(),
    )

    assert traits == 1
    assert beliefs == 0


@pytest.mark.asyncio
async def test_self_awareness_traits_are_visible_in_get_self_model():
    from surriti.cognition.self_awareness import _extract_self_traits
    from surriti.embedder import DummyEmbedder
    from surriti.graphiti import Surriti
    from surriti.testing import InMemoryDriver

    class SynthOnlyLLM:
        async def synthesize(self, system, user):
            del system, user
            return (
                '{"traits":[{"trait":"concise","confidence":0.9}],'
                '"beliefs":[{"belief":"I value brevity","confidence":0.7}]}'
            )

    driver = InMemoryDriver(enforce_entity_name_uniq=True)
    s = Surriti(driver, embedder=DummyEmbedder(embedding_dim=8))
    driver.records["entity"].append(
        {
            "uuid": "self-1",
            "group_id": "g",
            "name": "assistant_g",
            "summary": "",
            "labels": ["SelfEntity", "Assistant"],
        }
    )

    first = await _extract_self_traits(
        driver,
        SynthOnlyLLM(),
        "g",
        [{"content": "I am concise.", "source_description": "note"}],
        CognitionConfig(),
    )
    second = await _extract_self_traits(
        driver,
        SynthOnlyLLM(),
        "g",
        [{"content": "I am concise.", "source_description": "note"}],
        CognitionConfig(),
    )
    model = await s.get_self_model(group_id="g")

    assert first == (1, 1)
    assert second == (1, 1)
    assert model["traits"]
    assert model["beliefs"]
    assert len([r for r in driver.records["relates_to"] if r["name"] == "has_trait"]) == 1
    assert len([r for r in driver.records["relates_to"] if r["name"] == "has_belief"]) == 1
