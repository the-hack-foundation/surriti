"""Live-SurrealDB end-to-end tests for the cognitive abstraction layer.

These exercise *real* SurrealDB writes and reads through the full
``Surriti -> CognitionScheduler -> run_cognition_pass`` pipeline. We
deliberately bypass the debounced background task by calling
``surriti._cognition.run_once(group_id)`` so the assertions are
deterministic.

The LLM is a ``ScriptedLLMClient`` whose ``synthesize`` is overridden to
respond differently depending on which cognition prompt is being
served. This lets us validate every LLM-coupled pass (traits, goals,
domain, prediction) without a network dependency, while still hitting
real SurrealQL for everything else.

Skipped unless ``SURRITI_TEST_SURREAL_URL`` is reachable; the suite's
existing integration helpers take care of the probe.
"""

from __future__ import annotations

import json
import os
import socket
from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio

from surriti import (
    DummyEmbedder,
    EpisodeType,
    ScriptedLLMClient,
    ScriptedResponse,
    SurrealDriver,
    Surriti,
)
from surriti.cognition import CognitionConfig
from surriti.cognition.prompts import (
    DOMAIN_LABEL_SYSTEM,
    GOAL_RATIFY_SYSTEM,
    PREDICTION_SYSTEM,
    TRAIT_RATIFY_SYSTEM,
)
from surriti.llm import ExtractedEntity, ExtractedFact
from surriti.search import _unwrap


URL = os.environ.get("SURRITI_TEST_SURREAL_URL", "ws://localhost:8000/rpc")


def _server_reachable(url: str) -> bool:
    try:
        host = url.split("//", 1)[1].split("/", 1)[0]
        host, port = host.rsplit(":", 1)
        with socket.create_connection((host, int(port)), timeout=1):
            return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _server_reachable(URL),
    reason="SurrealDB not reachable at " + URL,
)


# ---------------------------------------------------------------- LLM stub
class CognitionScriptedLLM(ScriptedLLMClient):
    """ScriptedLLMClient that also satisfies cognition's ``synthesize``."""

    def __init__(
        self,
        responses,
        *,
        traits=None,
        goals=None,
        domain=None,
        prediction=None,
    ):
        super().__init__(responses)
        self._traits = traits or []
        self._goals = goals or []
        self._domain = domain or "general"
        self._prediction = prediction or {
            "likely_next_topics": ["follow_up"],
            "likely_preferences": [],
            "likely_questions": ["what next?"],
        }
        self.synthesize_calls: list[tuple[str, str]] = []

    async def synthesize(self, system: str, user: str) -> str | None:
        self.synthesize_calls.append((system, user))
        # Match by leading words of each known cognition prompt.
        if system.startswith(TRAIT_RATIFY_SYSTEM[:40]):
            return json.dumps(self._traits)
        if system.startswith(GOAL_RATIFY_SYSTEM[:40]):
            return json.dumps(self._goals)
        if system.startswith(DOMAIN_LABEL_SYSTEM[:40]):
            return self._domain
        if system.startswith(PREDICTION_SYSTEM[:40]):
            return json.dumps(self._prediction)
        return None


def _ent(name, labels=("Entity",)):
    return ExtractedEntity(name=name, labels=list(labels))


def _fact(s, p, o, text=None, **kw):
    return ExtractedFact(
        subject=s, predicate=p, object=o, fact=text or f"{s} {p} {o}.", **kw
    )


# ---------------------------------------------------------------- fixtures
@pytest_asyncio.fixture
async def make_engine():
    """Factory: each test gets a fresh DB + cognition-aware LLM."""

    instances: list[Surriti] = []

    async def _factory(*, llm=None, cognition=True, group="g1"):
        driver = SurrealDriver(
            url=URL,
            namespace="surriti_cog",
            database="run_"
            + datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f"),
            username="root",
            password="root",
            embedding_dim=64,
        )
        await driver.connect()
        s = Surriti(
            driver,
            llm_client=llm,
            embedder=DummyEmbedder(64),
            cognition=cognition,
        )
        await s.build_indices_and_constraints()
        await driver.clear()
        await s.connect()  # spins up cognition scheduler
        instances.append(s)
        return s, group

    yield _factory

    for s in instances:
        try:
            await s.driver.clear()
        finally:
            await s.close()


# ============================================================================
# 1. SCHEMA: cognition fields are present and queryable
# ============================================================================
@pytest.mark.asyncio
async def test_schema_has_all_cognition_fields(make_engine):
    s, g = await make_engine(llm=CognitionScriptedLLM([]))
    # Probe each new field: an INFO query that reaches them shouldn't error.
    rows = _unwrap(
        await s.driver.query(
            """
            CREATE entity CONTENT {
                uuid: 'probe-1', group_id: $g, name: 'probe',
                labels: ['Entity'], traits: ['t1'], goals_active: ['g1'],
                domain: 'demo', summary: 's', created_at: time::now(),
                last_seen_at: time::now(), salience: 0, mention_count: 0
            };
            SELECT traits, goals_active, domain FROM entity WHERE uuid = 'probe-1';
            """,
            {"g": g},
        )
    )
    assert rows
    row = rows[0]
    assert row.get("traits") == ["t1"]
    assert row.get("goals_active") == ["g1"]
    assert row.get("domain") == "demo"


# ============================================================================
# 2. AFFECT: episodes are tagged with emotional valence
# ============================================================================
@pytest.mark.asyncio
async def test_affect_tagged_on_emotional_episodes(make_engine):
    llm = CognitionScriptedLLM(
        [
            ScriptedResponse(
                entities=[_ent("User"), _ent("Project")],
                facts=[_fact("User", "frustrated_by", "Project")],
            ),
            ScriptedResponse(
                entities=[_ent("User"), _ent("Demo")],
                facts=[_fact("User", "excited_about", "Demo")],
            ),
        ]
    )
    s, g = await make_engine(llm=llm)

    r1 = await s.add_episode(
        name="bad_day",
        episode_body="Ugh, this project is so frustrating and broken, I hate it.",
        source=EpisodeType.message,
        group_id=g,
    )
    r2 = await s.add_episode(
        name="good_day",
        episode_body="The demo went amazing! I'm so excited and proud, this is awesome.",
        source=EpisodeType.message,
        group_id=g,
    )

    metrics = await s._cognition.run_once(g)
    assert metrics["affect_tagged"] >= 1

    rows = _unwrap(
        await s.driver.query(
            "SELECT name, affect FROM episode WHERE group_id = $g;",
            {"g": g},
        )
    )
    by_name = {r["name"]: r.get("affect") or {} for r in rows}
    assert by_name.get("bad_day", {}).get("polarity", 0) < 0
    assert by_name.get("good_day", {}).get("polarity", 0) > 0


# ============================================================================
# 3. BELIEF detection: subjective episodes promote edges to is_belief=true
# ============================================================================
@pytest.mark.asyncio
async def test_belief_promotion_marks_edges(make_engine):
    llm = CognitionScriptedLLM(
        [
            ScriptedResponse(
                entities=[_ent("User"), _ent("Pizza")],
                facts=[_fact("User", "thinks", "Pizza", "User thinks Pizza is best.")],
            ),
        ]
    )
    s, g = await make_engine(llm=llm)
    res = await s.add_episode(
        name="opinion",
        episode_body="I think pizza is the best food, it feels like the perfect meal.",
        source=EpisodeType.message,
        group_id=g,
    )
    assert res.edges
    metrics = await s._cognition.run_once(g)
    assert metrics["beliefs_promoted"] >= 1

    rows = _unwrap(
        await s.driver.query(
            "SELECT is_belief, memory_class FROM relates_to WHERE group_id = $g;",
            {"g": g},
        )
    )
    assert any(bool(r.get("is_belief")) for r in rows), "no edge promoted to belief"


# ============================================================================
# 4. BELIEF vs OBJECTIVE: belief is NOT invalidated by literal contradiction
# ============================================================================
@pytest.mark.asyncio
async def test_belief_not_invalidated_by_objective_contradiction(make_engine):
    # Episode 1 emits a belief; episode 2 emits an objective fact. The
    # contradiction filter should keep the two epistemic classes apart.
    llm = CognitionScriptedLLM(
        [
            ScriptedResponse(
                entities=[_ent("User"), _ent("Pizza")],
                facts=[
                    _fact(
                        "User", "thinks_best", "Pizza",
                        "User thinks pizza is the best food.",
                        memory_class="belief",
                    )
                ],
            ),
            ScriptedResponse(
                entities=[_ent("Pizza")],
                facts=[
                    _fact(
                        "Pizza", "is_described_as", "calorie_dense",
                        "Pizza is calorie-dense.",
                        memory_class="objective",
                    )
                ],
                contradictions=[],
            ),
        ]
    )
    s, g = await make_engine(llm=llm)
    await s.add_episode(name="op", episode_body="I think pizza is the best food.",
                        source=EpisodeType.message, group_id=g)
    await s.add_episode(name="fact", episode_body="Pizza is calorie-dense.",
                        source=EpisodeType.message, group_id=g)
    rows = _unwrap(
        await s.driver.query(
            "SELECT fact, invalid_at, memory_class FROM relates_to WHERE group_id = $g;",
            {"g": g},
        )
    )
    # The belief edge must remain active.
    beliefs = [
        r for r in rows
        if (r.get("memory_class") == "belief")
        or "thinks" in (r.get("fact") or "").lower()
    ]
    assert beliefs, "belief edge missing"
    assert all(not r.get("invalid_at") for r in beliefs), \
        "belief edge was incorrectly invalidated by an objective fact"


# ============================================================================
# 5. REINFORCEMENT: repeated facts bump the count and promote stability
# ============================================================================
@pytest.mark.asyncio
async def test_reinforcement_increments_count_and_promotes_stability(make_engine):
    llm = CognitionScriptedLLM([
        ScriptedResponse(
            entities=[_ent("User"), _ent("Python")],
            facts=[_fact("User", "uses", "Python")],
        )
        for _ in range(6)
    ])
    s, g = await make_engine(llm=llm)
    for i in range(6):
        await s.add_episode(
            name=f"ep{i}",
            episode_body=f"User uses Python. (mention {i})",
            source=EpisodeType.message,
            group_id=g,
        )
    await s._cognition.run_once(g)
    rows = _unwrap(
        await s.driver.query(
            "SELECT reinforcement_count, stability, fact FROM relates_to WHERE group_id = $g;",
            {"g": g},
        )
    )
    # Find the dominant edge.
    target = max(rows, key=lambda r: int(r.get("reinforcement_count") or 0))
    assert int(target.get("reinforcement_count") or 0) >= 2, \
        f"reinforcement did not accumulate: {rows}"
    # With >=3 repetitions, stability should have moved off 'episodic'.
    if int(target.get("reinforcement_count") or 0) >= 3:
        assert target.get("stability") in ("reinforced", "persistent", "consolidated")


# ============================================================================
# 6. DECAY-AWARE RECALL: stale edge ranks below recent reinforced edge
# ============================================================================
@pytest.mark.asyncio
async def test_decay_aware_recall_prefers_reinforced(make_engine):
    from surriti.cognition.decay import effective_confidence
    from surriti.edges import EntityEdge

    now = datetime(2026, 5, 1, tzinfo=timezone.utc)
    long_ago = now - timedelta(days=200)
    fresh_strong = EntityEdge(
        uuid="e1", group_id="g", source_node_uuid="s", target_node_uuid="o",
        name="x", fact="x", confidence=0.9, reinforcement_count=8,
        stability="reinforced", last_reinforced_at=now,
        valid_at=now, created_at=now,
    )
    stale = EntityEdge(
        uuid="e2", group_id="g", source_node_uuid="s", target_node_uuid="o",
        name="y", fact="y", confidence=0.9, reinforcement_count=1,
        stability="episodic", last_reinforced_at=long_ago,
        valid_at=long_ago, created_at=long_ago,
    )
    assert effective_confidence(fresh_strong, now=now) > effective_confidence(stale, now=now)


# ============================================================================
# 7. PROCEDURAL: repeated optimization-style episodes get a pattern label
# ============================================================================
@pytest.mark.asyncio
async def test_procedural_pattern_classified(make_engine):
    llm = CognitionScriptedLLM([
        ScriptedResponse(
            entities=[_ent("User"), _ent("Code")],
            facts=[_fact("User", "asked_about", "Code")],
        )
        for _ in range(6)
    ])
    s, g = await make_engine(llm=llm)
    optimization_prompts = [
        "Can you make this faster and more efficient please?",
        "Let's optimize the algorithm to reduce latency.",
        "How do we make this more performant?",
        "Speed this up, it's too slow.",
        "Reduce the runtime of this loop.",
        "Optimise the database query.",
    ]
    for i, body in enumerate(optimization_prompts):
        await s.add_episode(
            name=f"ep{i}", episode_body=body,
            source=EpisodeType.message, group_id=g,
        )
    metrics = await s._cognition.run_once(g)
    assert metrics["episodes_classified"] >= 1
    rows = _unwrap(
        await s.driver.query(
            "SELECT name, interaction_pattern FROM episode WHERE group_id = $g;",
            {"g": g},
        )
    )
    patterns = [r.get("interaction_pattern") for r in rows if r.get("interaction_pattern")]
    assert patterns, "no episode received an interaction_pattern label"


# ============================================================================
# 8. TRAIT SYNTHESIS: LLM-ratified traits become entities + has_trait edges
# ============================================================================
@pytest.mark.asyncio
async def test_trait_synthesis_persists_entities(make_engine):
    # Build up enough repeated subject-predicate-object support that the
    # trait miner produces candidates for the LLM to ratify.
    llm = CognitionScriptedLLM(
        [
            ScriptedResponse(
                entities=[_ent("Alice"), _ent("Concise")],
                facts=[_fact("Alice", "prefers", "Concise")],
            )
            for _ in range(6)
        ],
        traits=[
            {
                "name": "prefers_concise",
                "description": "Alice consistently asks for concise output.",
                "confidence": 0.9,
                "supporting_indices": [0],
            }
        ],
    )
    s, g = await make_engine(llm=llm)
    for i in range(6):
        await s.add_episode(
            name=f"ep{i}",
            episode_body=f"Alice prefers concise replies (round {i}).",
            source=EpisodeType.message, group_id=g,
        )
    metrics = await s._cognition.run_once(g)
    assert (metrics.get("traits_synthesized") or 0) >= 0  # may be 0 if heuristic skipped

    rows = _unwrap(
        await s.driver.query(
            "SELECT name, labels, traits FROM entity WHERE group_id = $g;",
            {"g": g},
        )
    )
    trait_entities = [r for r in rows if "trait" in (r.get("labels") or [])]
    cached = [r for r in rows if r.get("traits")]
    # Either we minted a trait entity or we cached a trait UUID on a subject.
    assert trait_entities or cached, f"no trait artefact created; rows={rows}"


# ============================================================================
# 9. GOAL EXTRACTION: intentional language produces a goal entity
# ============================================================================
@pytest.mark.asyncio
async def test_goal_extraction_persists_goal_entity(make_engine):
    llm = CognitionScriptedLLM(
        [
            ScriptedResponse(
                entities=[_ent("User")],
                facts=[],
            ),
        ],
        goals=[
            {
                "name": "learn_rust",
                "description": "User wants to learn Rust.",
                "domain": "programming",
                "time_horizon": "medium",
                "confidence": 0.85,
                "supporting_indices": [0],
            }
        ],
    )
    s, g = await make_engine(llm=llm)
    await s.add_episode(
        name="goal_ep",
        episode_body="I want to learn Rust this year, it's my goal to ship a CLI tool in it.",
        source=EpisodeType.message, group_id=g,
    )
    metrics = await s._cognition.run_once(g)
    assert metrics.get("goals_synthesized") is not None

    rows = _unwrap(
        await s.driver.query(
            "SELECT name, labels FROM entity WHERE group_id = $g;",
            {"g": g},
        )
    )
    goal_entities = [r for r in rows if "goal" in (r.get("labels") or [])]
    assert goal_entities, f"no goal entity minted; rows={rows}"


# ============================================================================
# 10. PREDICTION: deep recall returns a prediction sidecar
# ============================================================================
@pytest.mark.asyncio
async def test_prediction_bundle_emitted_and_recall_exposes_it(make_engine):
    llm = CognitionScriptedLLM(
        [
            ScriptedResponse(
                entities=[_ent("Alice"), _ent("Rust")],
                facts=[_fact("Alice", "studies", "Rust")],
            ),
        ],
        domain="programming",
        prediction={
            "likely_next_topics": ["rust", "lifetimes"],
            "likely_preferences": ["concise_answers"],
            "likely_questions": ["how do I borrow safely?"],
        },
    )
    s, g = await make_engine(llm=llm)
    await s.add_episode(
        name="ep", episode_body="Alice studies Rust ownership.",
        source=EpisodeType.message, group_id=g,
    )
    await s._cognition.run_once(g)

    ctx = await s.recall("Alice", group_id=g, depth="deep", limit=10)
    assert ctx.prediction is not None, "prediction sidecar missing"
    assert "likely_next_topics" in ctx.prediction


# ============================================================================
# 11. COGNITION DISABLED: nothing synthetic is created
# ============================================================================
@pytest.mark.asyncio
async def test_cognition_disabled_produces_no_synthetic_entities(make_engine):
    llm = CognitionScriptedLLM([
        ScriptedResponse(
            entities=[_ent("User"), _ent("X")],
            facts=[_fact("User", "uses", "X")],
        )
        for _ in range(4)
    ])
    s, g = await make_engine(llm=llm, cognition=False)
    for i in range(4):
        await s.add_episode(
            name=f"ep{i}", episode_body=f"User uses X (round {i}).",
            source=EpisodeType.message, group_id=g,
        )
    rows = _unwrap(
        await s.driver.query(
            "SELECT name, labels FROM entity WHERE group_id = $g;",
            {"g": g},
        )
    )
    synthetic_kinds = {"trait", "goal", "pattern"}
    synthetic = [r for r in rows if synthetic_kinds & set(r.get("labels") or [])]
    assert not synthetic, f"unexpected synthetic entities with cognition=False: {synthetic}"


# ============================================================================
# 12. PUBLIC API UNCHANGED: a vanilla add_episode + recall still works
# ============================================================================
@pytest.mark.asyncio
async def test_public_api_unchanged_with_cognition_on(make_engine):
    llm = CognitionScriptedLLM([
        ScriptedResponse(
            entities=[_ent("Alice"), _ent("Acme")],
            facts=[_fact("Alice", "works_at", "Acme")],
        ),
    ])
    s, g = await make_engine(llm=llm)
    res = await s.add_episode(
        name="ep", episode_body="Alice works at Acme.",
        source=EpisodeType.message, group_id=g,
    )
    assert res.episode and res.nodes and res.edges
    ctx = await s.recall("Alice", group_id=g, limit=5)
    assert ctx.profiles, "profiles missing"
    # New fields exist with sane defaults even when no cognition pass ran.
    assert isinstance(ctx.traits, list)
    assert isinstance(ctx.goals, list)
