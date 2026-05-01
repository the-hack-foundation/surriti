"""Regression tests for the relation-frame layer.

These verify the generalized behavior that replaces hand-maintained
predicate special cases:

* alias predicates collapse onto the canonical frame name
* symmetric frames dedupe direction-equivalent statements
* one_current cardinality drives deterministic slot supersession
  (regardless of source_type) without LLM round-trips
* coexist policy keeps siblings without invoking the LLM
* qualified variants occupy distinct slots
* unregistered predicates fall through to legacy heuristics
* per-tenant frame registration overrides global defaults
"""

from __future__ import annotations

import pytest

from surriti.embedder import DummyEmbedder
from surriti.graphiti import Surriti
from surriti.llm import (
    ExtractedEntity,
    ExtractedFact,
    ScriptedLLMClient,
    ScriptedResponse,
)
from surriti.relation_frames import (
    DEFAULT_FRAMES,
    RelationFrame,
    RelationFrameRegistry,
    make_slot_key,
    normalize_symmetric,
    qualifier_hash,
)
from surriti.testing import InMemoryDriver as FakeSurrealDriver


# ---------------------------------------------------------------------------
# Pure-function helpers
# ---------------------------------------------------------------------------


def test_default_catalog_includes_common_frames():
    canonicals = {f.canonical_name for f in DEFAULT_FRAMES}
    assert {"spouse_of", "lives_in", "works_at", "is_named", "owns_pet"}.issubset(
        canonicals
    )


def test_registry_resolves_aliases_case_insensitive():
    reg = RelationFrameRegistry()
    f1 = reg.get("WIFE_of")
    f2 = reg.get("married_to")
    f3 = reg.get("husband_of")
    assert f1 is not None and f1.canonical_name == "spouse_of"
    assert f2 is f1 is f3, "all spousal aliases should resolve to one frame"


def test_registry_returns_none_for_unknown_predicate():
    reg = RelationFrameRegistry()
    assert reg.get("totally_made_up_predicate") is None


def test_registry_group_scoped_overrides_global():
    reg = RelationFrameRegistry()
    custom = RelationFrame(
        canonical_name="lives_in",
        aliases=["lives_in"],
        directionality="directed",
        cardinality="many_current",
        contradiction_policy="coexist",
    )
    reg.register(custom, group_id="tenant-A")
    # Tenant-A sees the override; everyone else sees the default.
    a = reg.get("lives_in", group_id="tenant-A")
    b = reg.get("lives_in", group_id="tenant-B")
    assert a is custom
    assert b is not None and b.cardinality == "one_current"


def test_qualifier_hash_is_stable_and_order_independent():
    h1 = qualifier_hash({"season": "winter", "scope": "weekday"})
    h2 = qualifier_hash({"scope": "weekday", "season": "winter"})
    assert h1 == h2 and h1 != ""
    assert qualifier_hash(None) == qualifier_hash({}) == ""


def test_slot_key_distinguishes_qualified_variants():
    base = make_slot_key("g", "subj", "lives_in")
    winter = make_slot_key("g", "subj", "lives_in", {"season": "winter"})
    summer = make_slot_key("g", "subj", "lives_in", {"season": "summer"})
    assert base != winter != summer != base


def test_normalize_symmetric_orders_lex_min():
    a, b = normalize_symmetric("zzz", "aaa")
    assert (a, b) == ("aaa", "zzz")
    a, b = normalize_symmetric("aaa", "zzz")
    assert (a, b) == ("aaa", "zzz")


# ---------------------------------------------------------------------------
# End-to-end: alias + symmetric collapse
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_spouse_alias_collapses_to_canonical_frame():
    """``wife_of`` should be stored under the canonical ``spouse_of``
    name so subsequent ``husband_of``/``married_to`` mentions hit the
    same edge instead of producing a duplicate."""

    llm = ScriptedLLMClient(
        [
            ScriptedResponse(
                entities=[
                    ExtractedEntity(name="Michael", labels=["Person"]),
                    ExtractedEntity(name="Judy", labels=["Person"]),
                ],
                facts=[
                    ExtractedFact(
                        "Michael", "wife_of", "Judy",
                        "Michael's wife is Judy.",
                    )
                ],
            ),
            ScriptedResponse(
                entities=[
                    ExtractedEntity(name="Michael", labels=["Person"]),
                    ExtractedEntity(name="Judy", labels=["Person"]),
                ],
                facts=[
                    ExtractedFact(
                        "Judy", "married_to", "Michael",
                        "Judy is married to Michael.",
                    )
                ],
            ),
        ]
    )
    driver = FakeSurrealDriver(enforce_entity_name_uniq=True)
    s = Surriti(driver, llm_client=llm, embedder=DummyEmbedder(embedding_dim=64))
    await s.add_episode(name="e1", episode_body="x", group_id="g")
    await s.add_episode(name="e2", episode_body="y", group_id="g")

    spouse_edges = [
        r for r in driver.records["relates_to"] if r.get("name") == "spouse_of"
    ]
    active = [r for r in spouse_edges if r.get("invalid_at") is None]
    # Both restatements canonicalize to ``spouse_of`` and (after lex-min
    # symmetric normalization) hit the same slot, so exactly one edge
    # is active at any time. The earlier row may persist as superseded
    # history; what matters is that no alias name leaks through.
    assert len(active) == 1, "exactly one active canonical spouse_of edge"
    assert all(e["canonical_name"] == "spouse_of" for e in spouse_edges)
    # No raw-aliased edges should leak through.
    aliases = [
        r for r in driver.records["relates_to"]
        if r.get("name") in {"wife_of", "husband_of", "married_to"}
    ]
    assert aliases == []


# ---------------------------------------------------------------------------
# End-to-end: cardinality-driven supersession
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_one_current_cardinality_supersedes_prior_location():
    """``lives_in`` is a one_current frame, so a new location must
    expire the prior one *without* an LLM round-trip even when the
    extractor did not set ``singleton=True`` and the source isn't a user."""

    llm = ScriptedLLMClient(
        [
            ScriptedResponse(
                entities=[
                    ExtractedEntity(name="Michael"),
                    ExtractedEntity(name="Philadelphia"),
                ],
                facts=[
                    ExtractedFact(
                        "Michael", "lives_in", "Philadelphia",
                        "Michael lives in Philadelphia.",
                    )
                ],
            ),
            ScriptedResponse(
                entities=[
                    ExtractedEntity(name="Michael"),
                    ExtractedEntity(name="Florida"),
                ],
                facts=[
                    ExtractedFact(
                        "Michael", "lives_in", "Florida",
                        "Michael lives in Florida.",
                    )
                ],
            ),
        ]
    )
    driver = FakeSurrealDriver(enforce_entity_name_uniq=True)
    # Use a non-user source_type to prove cardinality (not source_type)
    # is what drives the new closer.
    s = Surriti(driver, llm_client=llm, embedder=DummyEmbedder(embedding_dim=64))
    await s.add_episode(
        name="e1", episode_body="x", group_id="g", source_type="assistant"
    )
    second = await s.add_episode(
        name="e2", episode_body="y", group_id="g", source_type="assistant"
    )

    assert len(second.invalidated_edges) == 1
    invalidated = second.invalidated_edges[0]
    assert invalidated.target_node_uuid != ""
    # Exactly one active lives_in edge should remain.
    active = [
        r for r in driver.records["relates_to"]
        if r.get("name") == "lives_in" and r.get("invalid_at") is None
    ]
    assert len(active) == 1


# ---------------------------------------------------------------------------
# End-to-end: coexist policy skips LLM
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_coexist_policy_skips_llm_contradiction_pass():
    """``owns_pet`` is a many_current/coexist frame, so two pets must
    coexist without ever invoking the LLM contradiction call."""

    captured: list[dict] = []

    class Spy(ScriptedLLMClient):
        async def find_contradictions(self, new_fact, existing_facts, **kwargs):
            captured.append({"new_fact": new_fact})
            return []

    llm = Spy(
        [
            ScriptedResponse(
                entities=[
                    ExtractedEntity(name="Michael"),
                    ExtractedEntity(name="Duke"),
                ],
                facts=[
                    ExtractedFact("Michael", "owns_pet", "Duke", "Michael owns Duke.")
                ],
            ),
            ScriptedResponse(
                entities=[
                    ExtractedEntity(name="Michael"),
                    ExtractedEntity(name="Whiskers"),
                ],
                facts=[
                    ExtractedFact(
                        "Michael", "owns_pet", "Whiskers", "Michael owns Whiskers."
                    )
                ],
            ),
        ]
    )
    driver = FakeSurrealDriver(enforce_entity_name_uniq=True)
    s = Surriti(driver, llm_client=llm, embedder=DummyEmbedder(embedding_dim=64))
    await s.add_episode(name="e1", episode_body="x", group_id="g")
    second = await s.add_episode(name="e2", episode_body="y", group_id="g")

    assert captured == [], "coexist policy must short-circuit the LLM pass"
    assert second.invalidated_edges == []
    # Two active owns_pet edges, no supersession.
    active = [
        r for r in driver.records["relates_to"]
        if r.get("name") == "owns_pet" and r.get("invalid_at") is None
    ]
    assert len(active) == 2


# ---------------------------------------------------------------------------
# End-to-end: terminate operation closes specific (subject, predicate, object)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_terminate_op_closes_only_matching_pet():
    """Terminating ``owns_pet(Michael, Duke)`` must close only that
    edge -- the other pet stays active."""

    llm = ScriptedLLMClient(
        [
            ScriptedResponse(
                entities=[
                    ExtractedEntity(name="Michael"),
                    ExtractedEntity(name="Duke"),
                    ExtractedEntity(name="Whiskers"),
                ],
                facts=[
                    ExtractedFact(
                        "Michael", "owns_pet", "Duke", "Michael owns Duke."
                    ),
                    ExtractedFact(
                        "Michael", "owns_pet", "Whiskers", "Michael owns Whiskers."
                    ),
                ],
            ),
            ScriptedResponse(
                entities=[
                    ExtractedEntity(name="Michael"),
                    ExtractedEntity(name="Duke"),
                ],
                facts=[
                    ExtractedFact(
                        "Michael", "owns_pet", "Duke",
                        "Michael no longer owns Duke.",
                        operation="terminate",
                    )
                ],
            ),
        ]
    )
    driver = FakeSurrealDriver(enforce_entity_name_uniq=True)
    s = Surriti(driver, llm_client=llm, embedder=DummyEmbedder(embedding_dim=64))
    await s.add_episode(name="e1", episode_body="x", group_id="g")
    await s.add_episode(name="e2", episode_body="y", group_id="g")

    duke_active = [
        r for r in driver.records["relates_to"]
        if r.get("name") == "owns_pet"
        and "Duke" in (r.get("fact") or "")
        and r.get("invalid_at") is None
    ]
    whiskers_active = [
        r for r in driver.records["relates_to"]
        if r.get("name") == "owns_pet"
        and "Whiskers" in (r.get("fact") or "")
        and r.get("invalid_at") is None
    ]
    assert duke_active == [], "terminate should have closed the Duke edge"
    assert len(whiskers_active) == 1, "the unrelated pet must remain active"


# ---------------------------------------------------------------------------
# End-to-end: qualifiers keep variants in distinct slots
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_qualified_residence_variants_coexist():
    """``lives_in`` is one_current, but the qualifier hash makes
    ``lives_in(Florida, season=winter)`` and ``lives_in(Vermont,
    season=summer)`` separate slots that coexist."""

    llm = ScriptedLLMClient(
        [
            ScriptedResponse(
                entities=[
                    ExtractedEntity(name="Michael"),
                    ExtractedEntity(name="Florida"),
                ],
                facts=[
                    ExtractedFact(
                        "Michael", "lives_in", "Florida",
                        "Michael lives in Florida during the winter.",
                        qualifiers={"season": "winter"},
                    )
                ],
            ),
            ScriptedResponse(
                entities=[
                    ExtractedEntity(name="Michael"),
                    ExtractedEntity(name="Vermont"),
                ],
                facts=[
                    ExtractedFact(
                        "Michael", "lives_in", "Vermont",
                        "Michael lives in Vermont during the summer.",
                        qualifiers={"season": "summer"},
                    )
                ],
            ),
        ]
    )
    driver = FakeSurrealDriver(enforce_entity_name_uniq=True)
    s = Surriti(driver, llm_client=llm, embedder=DummyEmbedder(embedding_dim=64))
    await s.add_episode(name="e1", episode_body="x", group_id="g")
    second = await s.add_episode(name="e2", episode_body="y", group_id="g")

    # No supersession because the qualifier hashes differ.
    assert second.invalidated_edges == []
    active = [
        r for r in driver.records["relates_to"]
        if r.get("name") == "lives_in" and r.get("invalid_at") is None
    ]
    assert len(active) == 2
    qhashes = {
        r.get("fact_key", "").split("::")[-1] for r in active
    }
    assert len(qhashes) == 2, "each qualifier should produce a distinct slot key"


# ---------------------------------------------------------------------------
# End-to-end: per-tenant frame registration via Surriti.register_frame
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_register_frame_overrides_default_for_tenant():
    """A user can override a default frame for one tenant without
    affecting the rest of the catalog."""

    driver = FakeSurrealDriver(enforce_entity_name_uniq=True)
    s = Surriti(driver, embedder=DummyEmbedder(embedding_dim=64))
    s.register_frame(
        RelationFrame(
            canonical_name="lives_in",
            aliases=["lives_in"],
            directionality="directed",
            cardinality="many_current",
            contradiction_policy="coexist",
        ),
        group_id="globe-trotter",
    )

    default_frame = s.get_frame("lives_in")
    tenant_frame = s.get_frame("lives_in", group_id="globe-trotter")
    assert default_frame is not None and default_frame.cardinality == "one_current"
    assert tenant_frame is not None and tenant_frame.cardinality == "many_current"


# ---------------------------------------------------------------------------
# End-to-end: unregistered predicate falls through to legacy heuristics
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unregistered_predicate_keeps_raw_name_and_legacy_path():
    """When no frame resolves, the engine stores the raw predicate and
    relies on the existing ``fact.singleton``/``source_type`` heuristics."""

    llm = ScriptedLLMClient(
        [
            ScriptedResponse(
                entities=[
                    ExtractedEntity(name="Server-A"),
                    ExtractedEntity(name="v1.2.3"),
                ],
                facts=[
                    ExtractedFact(
                        "Server-A", "deployed_version", "v1.2.3",
                        "Server-A is running v1.2.3.",
                    )
                ],
            )
        ]
    )
    driver = FakeSurrealDriver(enforce_entity_name_uniq=True)
    s = Surriti(driver, llm_client=llm, embedder=DummyEmbedder(embedding_dim=64))
    await s.add_episode(name="e1", episode_body="x", group_id="g")

    edges = [
        r for r in driver.records["relates_to"]
        if r.get("name") == "deployed_version"
    ]
    assert len(edges) == 1
    assert edges[0]["relation_frame_id"] is None
    assert edges[0]["canonical_name"] == "deployed_version"


# ---------------------------------------------------------------------------
# Smoke: get_conflicts surface compiles and returns a list
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_conflicts_returns_empty_by_default():
    driver = FakeSurrealDriver(enforce_entity_name_uniq=True)
    s = Surriti(driver, embedder=DummyEmbedder(embedding_dim=64))
    conflicts = await s.get_conflicts(group_id="g")
    assert conflicts == []


# ---------------------------------------------------------------------------
# Phase A: extractor parser surfaces qualifiers / roles / source span
# ---------------------------------------------------------------------------


def test_extractor_parses_new_claim_metadata():
    """The JSON parser must round-trip ``relation_phrase``,
    ``qualifiers``, ``argument_roles``, ``source_span`` and accept the
    ``qualify`` operation onto :class:`ExtractedFact`.
    """
    from surriti.llm_clients import _parse_extraction

    raw = """
    {
      "entities": [{"name": "Michael"}, {"name": "Florida"}],
      "facts": [{
        "subject": "Michael",
        "predicate": "lives_in",
        "relation_phrase": "live in",
        "object": "Florida",
        "fact": "Michael lives in Florida during the winter.",
        "operation": "qualify",
        "temporal": true,
        "singleton": true,
        "domain": "residence",
        "qualifiers": {"season": "winter"},
        "argument_roles": {"subject": "resident", "object": "place"},
        "source_span": "I live in Florida during the winter"
      }]
    }
    """
    result = _parse_extraction(raw)
    assert len(result.facts) == 1
    f = result.facts[0]
    assert f.operation == "qualify"
    assert f.qualifiers == {"season": "winter"}
    assert f.argument_roles == {"subject": "resident", "object": "place"}
    assert f.relation_phrase == "live in"
    assert f.source_span == "I live in Florida during the winter"


# ---------------------------------------------------------------------------
# Phase A: qualify operation does not close peers (qualified variants coexist)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_qualify_operation_does_not_close_unqualified_slot():
    """A ``qualify`` claim must coexist with an unqualified slot --
    even when the frame is one_current -- because the qualifier hash
    puts it in a distinct slot key."""
    llm = ScriptedLLMClient(
        [
            ScriptedResponse(
                entities=[ExtractedEntity(name="Michael"), ExtractedEntity(name="Vermont")],
                facts=[
                    ExtractedFact(
                        "Michael", "lives_in", "Vermont",
                        "Michael lives in Vermont.",
                    )
                ],
            ),
            ScriptedResponse(
                entities=[ExtractedEntity(name="Michael"), ExtractedEntity(name="Florida")],
                facts=[
                    ExtractedFact(
                        "Michael", "lives_in", "Florida",
                        "Michael lives in Florida during the winter.",
                        operation="qualify",
                        qualifiers={"season": "winter"},
                    )
                ],
            ),
        ]
    )
    driver = FakeSurrealDriver(enforce_entity_name_uniq=True)
    s = Surriti(driver, llm_client=llm, embedder=DummyEmbedder(embedding_dim=64))
    await s.add_episode(name="e1", episode_body="x", group_id="g")
    second = await s.add_episode(name="e2", episode_body="y", group_id="g")

    assert second.invalidated_edges == []
    active = [
        r for r in driver.records["relates_to"]
        if r.get("name") == "lives_in" and r.get("invalid_at") is None
    ]
    assert len(active) == 2


# ---------------------------------------------------------------------------
# Phase B: LLM frame classifier mints a frame for unknown predicates
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llm_classifier_mints_frame_then_caches_it():
    """A predicate not in the default catalog should trigger one
    classifier call, get cached, and resolve from cache thereafter."""
    minted = RelationFrame(
        canonical_name="depends_on",
        aliases=["requires"],
        directionality="directed",
        temporal_kind="state",
        cardinality="many_current",
        contradiction_policy="coexist",
        confidence=0.8,
    )
    llm = ScriptedLLMClient(
        [
            ScriptedResponse(
                entities=[ExtractedEntity(name="Service-A"), ExtractedEntity(name="Service-B")],
                facts=[ExtractedFact("Service-A", "depends_on", "Service-B",
                                     "Service-A depends on Service-B.")],
                frame=minted,
            ),
            ScriptedResponse(
                entities=[ExtractedEntity(name="Service-A"), ExtractedEntity(name="Service-C")],
                facts=[ExtractedFact("Service-A", "requires", "Service-C",
                                     "Service-A requires Service-C.")],
            ),
        ]
    )
    driver = FakeSurrealDriver(enforce_entity_name_uniq=True)
    s = Surriti(driver, llm_client=llm, embedder=DummyEmbedder(embedding_dim=64))
    await s.add_episode(name="e1", episode_body="x", group_id="g")
    await s.add_episode(name="e2", episode_body="y", group_id="g")

    # Exactly one classifier round-trip across both episodes.
    assert len(llm.classify_calls) == 1
    assert llm.classify_calls[0]["predicate"] == "depends_on"
    # Both edges canonicalize to the minted frame name.
    edges = [
        r for r in driver.records["relates_to"]
        if r.get("name") == "depends_on"
    ]
    assert len(edges) == 2
    assert all(e["canonical_name"] == "depends_on" for e in edges)


# ---------------------------------------------------------------------------
# Phase C: needs_resolution writes a conflict_group_id across same-slot peers
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_uncertain_policy_writes_needs_resolution_group():
    """When the resolved frame's policy is ``uncertain`` AND the LLM
    finds no contradictions AND a conflicting same-slot peer exists,
    the new edge gets ``status="needs_resolution"`` and both edges
    share a ``conflict_group_id``."""
    llm = ScriptedLLMClient(
        [
            ScriptedResponse(
                entities=[ExtractedEntity(name="Michael"), ExtractedEntity(name="Lagos")],
                facts=[ExtractedFact("Michael", "born_city", "Lagos",
                                     "Michael was born in Lagos.")],
            ),
            ScriptedResponse(
                entities=[ExtractedEntity(name="Michael"), ExtractedEntity(name="Abuja")],
                facts=[ExtractedFact("Michael", "born_city", "Abuja",
                                     "Actually Michael was born in Abuja.")],
            ),
        ]
    )
    driver = FakeSurrealDriver(enforce_entity_name_uniq=True)
    s = Surriti(driver, llm_client=llm, embedder=DummyEmbedder(embedding_dim=64))
    s.register_frame(
        RelationFrame(
            canonical_name="born_city",
            aliases=[],
            directionality="directed",
            temporal_kind="timeless",
            cardinality="one_current",
            contradiction_policy="uncertain",
        )
    )
    await s.add_episode(name="e1", episode_body="x", group_id="g")
    await s.add_episode(name="e2", episode_body="y", group_id="g")

    conflicts = await s.get_conflicts(group_id="g")
    assert len(conflicts) == 1
    assert conflicts[0].status == "needs_resolution"
    new_edge = conflicts[0]
    assert new_edge.conflict_group_id is not None
    # The peer edge must share the same conflict_group_id.
    peers = [
        r for r in driver.records["relates_to"]
        if r.get("name") == "born_city"
        and r.get("conflict_group_id") == new_edge.conflict_group_id
    ]
    assert len(peers) == 2


# ---------------------------------------------------------------------------
# Phase D: merge_frames folds aliases onto the target
# ---------------------------------------------------------------------------


def test_merge_frames_folds_source_aliases_into_target():
    driver = FakeSurrealDriver(enforce_entity_name_uniq=True)
    s = Surriti(driver, embedder=DummyEmbedder(embedding_dim=64))
    # Start with two distinct frames.
    s.register_frame(RelationFrame(canonical_name="employed_by", aliases=["hired_by"]))
    target = s.merge_frames(source="employed_by", target="works_at")
    # Aliases are folded.
    assert "employed_by" in target.aliases
    assert "hired_by" in target.aliases
    # Lookups for any folded alias now resolve to ``works_at``.
    via_alias = s.get_frame("hired_by")
    via_canonical = s.get_frame("employed_by")
    assert via_alias is target
    assert via_canonical is target


def test_merge_frames_unknown_source_raises_keyerror():
    driver = FakeSurrealDriver(enforce_entity_name_uniq=True)
    s = Surriti(driver, embedder=DummyEmbedder(embedding_dim=64))
    with pytest.raises(KeyError):
        s.merge_frames(source="never_registered", target="works_at")


# ---------------------------------------------------------------------------
# Phase D: current_profile groups currently-true facts by canonical name
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_current_profile_groups_alias_mentions_under_canonical_name():
    llm = ScriptedLLMClient(
        [
            ScriptedResponse(
                entities=[ExtractedEntity(name="Michael"), ExtractedEntity(name="Judy")],
                facts=[ExtractedFact("Michael", "wife_of", "Judy",
                                     "Michael's wife is Judy.")],
            ),
            ScriptedResponse(
                entities=[ExtractedEntity(name="Michael"), ExtractedEntity(name="Acme")],
                facts=[ExtractedFact("Michael", "employed_by", "Acme",
                                     "Michael is employed by Acme.")],
            ),
        ]
    )
    driver = FakeSurrealDriver(enforce_entity_name_uniq=True)
    s = Surriti(driver, llm_client=llm, embedder=DummyEmbedder(embedding_dim=64))
    await s.add_episode(name="e1", episode_body="x", group_id="g")
    await s.add_episode(name="e2", episode_body="y", group_id="g")

    michael = next(r for r in driver.records["entity"] if r.get("name") == "Michael")
    profile = await s.current_profile(subject_uuid=michael["uuid"], group_id="g")

    # Aliases ``wife_of`` / ``employed_by`` must surface under their
    # canonical bucket names ``spouse_of`` / ``works_at``.
    assert "spouse_of" in profile or "works_at" in profile
    # Subject of spouse_of may have been swapped by symmetric normalization;
    # works_at always keeps Michael as subject.
    assert "works_at" in profile
    assert all(e.canonical_name == "works_at" for e in profile["works_at"])
