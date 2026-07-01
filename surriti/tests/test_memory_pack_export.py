"""Tests for Surriti Memory Pack export.

Uses the in-memory driver from ``surriti.testing`` — no SurrealDB needed.
"""

from __future__ import annotations

import json
import tempfile
import zipfile
from pathlib import Path

import pytest

from surriti.embedder import DummyEmbedder
from surriti.graphiti import Surriti
from surriti.memory_pack import (
    export_group_to_dir,
    export_group_to_zip,
    import_group_from_zip,
    validate_pack_dir,
    validate_pack_zip,
)
from surriti.testing import InMemoryDriver as FakeSurrealDriver


# ------------------------------------------------------------------ helpers


def _read_zip_text(zf: zipfile.ZipFile, name: str) -> str:
    return zf.read(name).decode("utf-8")


def _read_zip_jsonl(zf: zipfile.ZipFile, name: str) -> list[dict]:
    text = _read_zip_text(zf, name)
    return [json.loads(line) for line in text.strip().split("\n") if line.strip()]


def _make_driver():
    return FakeSurrealDriver(embedding_dim=64)


async def _add_basic_episodes(surriti: Surriti, group_id: str = "g1"):
    """Add a handful of episodes so the graph has entities, edges, and aliases."""
    await surriti.add_episode(
        name="onboarding",
        episode_body="Alice works at Acme Corp as a staff engineer.",
        group_id=group_id,
    )
    await surriti.add_episode(
        name="update",
        episode_body="Alice moved to Globex Inc. She now leads the platform team.",
        group_id=group_id,
    )
    await surriti.add_episode(
        name="prefs",
        episode_body="Alice prefers async communication. She dislikes early morning meetings.",
        group_id=group_id,
    )
    await surriti.add_episode(
        name="more",
        episode_body="Bob is Alice's manager. Bob works at Globex too.",
        group_id=group_id,
    )


# ------------------------------------------------------------------ tests


async def test_export_pack_has_expected_files():
    """After adding episodes, the ZIP must contain manifest, checksums, and all JSONL files."""

    driver = _make_driver()
    surriti = Surriti(driver, embedder=DummyEmbedder(embedding_dim=64))
    await _add_basic_episodes(surriti)

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "test.surriti-pack.zip"
        result = await surriti.export_memory_pack("g1", str(out))
        assert "output_path" in result
        assert "manifest" in result

        assert zipfile.is_zipfile(out)
        with zipfile.ZipFile(out, "r") as zf:
            names = set(zf.namelist())
            required = {
                "manifest.json",
                "checksums.json",
                "entities.jsonl",
                "entity_aliases.jsonl",
                "relation_frames.jsonl",
                "edges.jsonl",
            }
            assert required.issubset(names), f"Missing files: {required - names}"

            manifest = json.loads(_read_zip_text(zf, "manifest.json"))
            assert manifest["format"] == "surriti.memory-pack"
            assert manifest["version"] == 1
            assert manifest["counts"]["entities"] > 0
            assert manifest["counts"]["edges"] > 0


async def test_export_omits_raw_episodes_by_default():
    """Raw episode records must not be exported. Fact edges may contain extracted text."""

    driver = _make_driver()
    surriti = Surriti(driver, embedder=DummyEmbedder(embedding_dim=64))
    secret = "XYZZY_SECRET_PHRASE_42"
    await surriti.add_episode(
        name="secret_ep",
        episode_body=f"Alice knows the secret: {secret}",
        group_id="g1",
    )

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "test.surriti-pack.zip"
        await surriti.export_memory_pack("g1", str(out))

        with zipfile.ZipFile(out, "r") as zf:
            names = set(zf.namelist())
            # Raw episode files must not exist.
            assert "episodes.jsonl" not in names, "episodes.jsonl should not be present"

            # Manifest must declare no raw transcripts.
            manifest = json.loads(_read_zip_text(zf, "manifest.json"))
            assert manifest["privacy"]["contains_raw_transcripts"] is False


async def test_export_omits_embeddings_by_default():
    """Export with default settings must not include any embedding vectors."""

    driver = _make_driver()
    surriti = Surriti(driver, embedder=DummyEmbedder(embedding_dim=64))
    await _add_basic_episodes(surriti)

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "test.surriti-pack.zip"
        await surriti.export_memory_pack("g1", str(out))

        with zipfile.ZipFile(out, "r") as zf:
            entities = _read_zip_jsonl(zf, "entities.jsonl")
            edges = _read_zip_jsonl(zf, "edges.jsonl")

            for e in entities:
                assert "name_embedding" not in e, f"name_embedding leaked in entity {e.get('uuid')}"
                assert "profile_embedding" not in e, f"profile_embedding leaked in entity {e.get('uuid')}"

            for e in edges:
                assert "fact_embedding" not in e, f"fact_embedding leaked in edge {e.get('uuid')}"


async def test_export_preserves_temporal_edge_fields():
    """Active and invalidated facts must preserve valid_at, invalid_at, status, supersedes, superseded_by."""

    driver = _make_driver()
    from surriti.llm import ScriptedLLMClient, ScriptedResponse, ExtractedEntity, ExtractedFact

    llm = ScriptedLLMClient(
        responses=[
            # Episode 1: Alice works at Acme.
            ScriptedResponse(
                entities=[ExtractedEntity(name="Alice", labels=["Person"]),
                          ExtractedEntity(name="Acme Corp", labels=["Organization"])],
                facts=[ExtractedFact(subject="Alice", predicate="works_at",
                                     object="Acme Corp", singleton=True)],
            ),
            # Episode 2: Alice no longer works at Acme; moved to Globex.
            ScriptedResponse(
                entities=[ExtractedEntity(name="Alice", labels=["Person"]),
                          ExtractedEntity(name="Acme Corp", labels=["Organization"]),
                          ExtractedEntity(name="Globex", labels=["Organization"])],
                facts=[
                    ExtractedFact(subject="Alice", predicate="works_at",
                                  object="Acme Corp", operation="terminate"),
                    ExtractedFact(subject="Alice", predicate="works_at",
                                  object="Globex", singleton=True),
                ],
            ),
        ]
    )
    surriti = Surriti(driver, llm_client=llm, embedder=DummyEmbedder(embedding_dim=64))

    await surriti.add_episode(
        name="e1",
        episode_body="Alice works at Acme Corp.",
        group_id="g1",
    )
    await surriti.add_episode(
        name="e2",
        episode_body="Alice no longer works at Acme Corp. Alice works at Globex.",
        group_id="g1",
    )

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "test.surriti-pack.zip"
        await surriti.export_memory_pack("g1", str(out))

        with zipfile.ZipFile(out, "r") as zf:
            edges = _read_zip_jsonl(zf, "edges.jsonl")

            invalidated = [e for e in edges if e.get("status") == "superseded" or e.get("invalid_at")]
            active = [e for e in edges if e.get("status") == "active"]

            assert invalidated, "Expected at least one superseded/invalidated edge"
            assert active, "Expected at least one active edge"

            for e in edges:
                assert "valid_at" in e
                assert "invalid_at" in e
                assert "status" in e
                assert "supersedes" in e
                # superseded_by is only present on superseded edges.
                assert "source_node_uuid" in e
                assert "target_node_uuid" in e

# At least one edge must be superseded with invalid_at set.
                superseded = [e for e in edges if e.get("status") == "superseded" and e.get("invalid_at")]
                assert superseded, "Expected at least one superseded edge with invalid_at"


async def test_export_preserves_memory_class_fields():
    """Preference/style/constraint edge metadata must survive export."""

    driver = _make_driver()
    surriti = Surriti(driver, embedder=DummyEmbedder(embedding_dim=64))

    await surriti.add_episode(
        name="prefs",
        episode_body="Alice prefers async communication. Alice likes Python. Alice has a style of being concise.",
        group_id="g1",
    )

    # Manually tag edges with memory_class so we can assert it survives.
    for r in driver.records["relates_to"]:
        if r.get("group_id") == "g1":
            r.setdefault("attributes", {})
            r["attributes"]["memory_class"] = "preference"

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "test.surriti-pack.zip"
        await surriti.export_memory_pack("g1", str(out))

        with zipfile.ZipFile(out, "r") as zf:
            edges = _read_zip_jsonl(zf, "edges.jsonl")
            assert len(edges) > 0

            for e in edges:
                attrs = e.get("attributes", {})
                assert "memory_class" in e or "memory_class" in attrs, (
                    f"memory_class missing from edge {e.get('uuid')}"
                )


async def test_export_aliases_without_dangling_episode_references():
    """Alias rows must not preserve source_episode_uuid by default."""

    driver = _make_driver()
    surriti = Surriti(driver, embedder=DummyEmbedder(embedding_dim=64))

    await surriti.add_episode(
        name="e1",
        episode_body="Alice (also known as Ally) works at Acme.",
        group_id="g1",
    )

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "test.surriti-pack.zip"
        await surriti.export_memory_pack("g1", str(out))

        with zipfile.ZipFile(out, "r") as zf:
            aliases = _read_zip_jsonl(zf, "entity_aliases.jsonl")
            for a in aliases:
                assert "alias" in a
                assert "entity_uuid" in a
                assert "source_episode_uuid" not in a, (
                    f"source_episode_uuid leaked in alias {a.get('uuid')}"
                )


async def test_export_checksums_match():
    """Checksums in checksums.json must match actual file contents."""

    driver = _make_driver()
    surriti = Surriti(driver, embedder=DummyEmbedder(embedding_dim=64))
    await _add_basic_episodes(surriti)

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "test.surriti-pack.zip"
        await surriti.export_memory_pack("g1", str(out))

        result = validate_pack_zip(str(out))
        assert result.ok, f"Validation failed: {result.errors}"
        assert result.counts.get("entities", 0) > 0


async def test_large_export_streaming_smoke():
    """Export with small page_size matches counts."""

    driver = _make_driver()
    surriti = Surriti(driver, embedder=DummyEmbedder(embedding_dim=64))
    await _add_basic_episodes(surriti)

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "test.surriti-pack.zip"
        result = await surriti.export_memory_pack("g1", str(out), page_size=3)

        with zipfile.ZipFile(out, "r") as zf:
            entities = _read_zip_jsonl(zf, "entities.jsonl")
            edges = _read_zip_jsonl(zf, "edges.jsonl")
            aliases = _read_zip_jsonl(zf, "entity_aliases.jsonl")
            frames = _read_zip_jsonl(zf, "relation_frames.jsonl")

            assert result["counts"]["entities"] == len(entities)
            assert result["counts"]["edges"] == len(edges)
            assert result["counts"]["entity_aliases"] == len(aliases)
            assert result["counts"]["relation_frames"] == len(frames)


async def test_export_dir_validation():
    """validate_pack_dir must accept a valid export directory."""

    driver = _make_driver()
    surriti = Surriti(driver, embedder=DummyEmbedder(embedding_dim=64))
    await _add_basic_episodes(surriti)

    with tempfile.TemporaryDirectory() as tmp:
        result = await export_group_to_dir(driver, "g1", tmp)
        assert result.output_path == tmp

        vr = validate_pack_dir(tmp)
        assert vr.ok, f"Directory validation failed: {vr.errors}"
        assert vr.counts["entities"] > 0


async def test_validate_rejects_missing_files():
    """validate_pack_zip must reject a ZIP missing required files."""

    with tempfile.TemporaryDirectory() as tmp:
        bad_zip = Path(tmp) / "bad.zip"
        with zipfile.ZipFile(bad_zip, "w") as zf:
            zf.writestr("manifest.json", "{}")

        vr = validate_pack_zip(str(bad_zip))
        assert not vr.ok
        assert any("Missing" in e or "missing" in e for e in vr.errors)


async def test_validate_rejects_bad_checksum():
    """validate_pack_zip must detect checksum mismatch."""

    driver = _make_driver()
    surriti = Surriti(driver, embedder=DummyEmbedder(embedding_dim=64))
    await _add_basic_episodes(surriti)

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "test.surriti-pack.zip"
        await surriti.export_memory_pack("g1", str(out))

        with zipfile.ZipFile(out, "r") as zf:
            data = {n: zf.read(n) for n in zf.namelist()}

        data["entities.jsonl"] = b"tampered content\n"

        tampered = Path(tmp) / "tampered.zip"
        with zipfile.ZipFile(tampered, "w", zipfile.ZIP_DEFLATED) as zf:
            for name, content in data.items():
                zf.writestr(name, content)

        vr = validate_pack_zip(str(tampered))
        assert not vr.ok
        assert any(
            "Checksum mismatch" in e or "checksum" in e.lower() for e in vr.errors
        ), f"Expected checksum error, got: {vr.errors}"


async def test_export_empty_group():
    """Exporting a group with no data should still produce a valid pack."""

    driver = _make_driver()

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "empty.surriti-pack.zip"
        result = await export_group_to_zip(driver, "nonexistent", str(out))

        assert result.counts["entities"] == 0
        assert result.counts["edges"] == 0

        vr = validate_pack_zip(str(out))
        assert vr.ok, f"Empty pack validation failed: {vr.errors}"


async def test_export_jsonl_is_valid_json():
    """Each line in each JSONL file must be valid JSON."""

    driver = _make_driver()
    surriti = Surriti(driver, embedder=DummyEmbedder(embedding_dim=64))
    await _add_basic_episodes(surriti)

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "test.surriti-pack.zip"
        await surriti.export_memory_pack("g1", str(out))

        with zipfile.ZipFile(out, "r") as zf:
            for jsonl_name in [
                "entities.jsonl",
                "entity_aliases.jsonl",
                "relation_frames.jsonl",
                "edges.jsonl",
            ]:
                text = _read_zip_text(zf, jsonl_name)
                for i, line in enumerate(text.strip().split("\n"), 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        json.loads(line)
                    except json.JSONDecodeError as exc:
                        pytest.fail(f"Invalid JSON in {jsonl_name} line {i}: {exc}")


async def test_export_manifest_has_all_keys():
    """Manifest must contain all required top-level keys."""

    driver = _make_driver()
    surriti = Surriti(driver, embedder=DummyEmbedder(embedding_dim=64))
    await _add_basic_episodes(surriti)

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "test.surriti-pack.zip"
        await surriti.export_memory_pack("g1", str(out))

        with zipfile.ZipFile(out, "r") as zf:
            manifest = json.loads(_read_zip_text(zf, "manifest.json"))

            required_keys = {
                "format", "version", "created_at", "export_started_at",
                "export_finished_at", "consistency", "source", "policy",
                "counts", "privacy", "files",
            }
            missing = required_keys - set(manifest.keys())
            assert not missing, f"Manifest missing keys: {missing}"

            assert manifest["policy"]["include_embeddings"] == "never"
            assert manifest["privacy"]["contains_embeddings"] is False
            assert manifest["privacy"]["contains_raw_transcripts"] is False


async def test_export_edge_source_target_uuids():
    """Every exported edge must have source_node_uuid and target_node_uuid."""

    driver = _make_driver()
    surriti = Surriti(driver, embedder=DummyEmbedder(embedding_dim=64))
    await _add_basic_episodes(surriti)

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "test.surriti-pack.zip"
        await surriti.export_memory_pack("g1", str(out))

        with zipfile.ZipFile(out, "r") as zf:
            edges = _read_zip_jsonl(zf, "edges.jsonl")
            assert len(edges) > 0

            for e in edges:
                assert "source_node_uuid" in e, f"Missing source_node_uuid in {e.get('uuid')}"
                assert "target_node_uuid" in e, f"Missing target_node_uuid in {e.get('uuid')}"
                assert e["source_node_uuid"], f"Empty source_node_uuid in {e.get('uuid')}"
                assert e["target_node_uuid"], f"Empty target_node_uuid in {e.get('uuid')}"


async def test_export_no_mentions_communities():
    """Export must not include mentions or community tables."""

    driver = _make_driver()
    surriti = Surriti(driver, embedder=DummyEmbedder(embedding_dim=64))
    await _add_basic_episodes(surriti)

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "test.surriti-pack.zip"
        await surriti.export_memory_pack("g1", str(out))

        with zipfile.ZipFile(out, "r") as zf:
            names = set(zf.namelist())
            assert "mentions.jsonl" not in names
            assert "communities.jsonl" not in names
            assert "episodes.jsonl" not in names


async def test_import_pack_round_trips_portable_graph_without_episodes_or_embeddings():
    """A pack can move useful graph memory into another group without raw episodes."""

    driver = _make_driver()
    surriti = Surriti(driver, embedder=DummyEmbedder(embedding_dim=64))
    await _add_basic_episodes(surriti, group_id="source")

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "test.surriti-pack.zip"
        exported = await surriti.export_memory_pack("source", str(out))
        imported = await surriti.import_memory_pack(str(out), "target")

    assert imported["validation"]["ok"] is True
    assert imported["counts"]["entities"] == exported["counts"]["entities"]
    assert imported["counts"]["edges"] == exported["counts"]["edges"]

    target_entities = [r for r in driver.records["entity"] if r.get("group_id") == "target"]
    target_edges = [r for r in driver.records["relates_to"] if r.get("group_id") == "target"]
    target_aliases = [r for r in driver.records["entity_alias"] if r.get("group_id") == "target"]
    target_episodes = [r for r in driver.records["episode"] if r.get("group_id") == "target"]

    assert target_entities
    assert target_edges
    assert len(target_aliases) == exported["counts"]["entity_aliases"]
    assert target_episodes == []
    assert all("name_embedding" not in e or e["name_embedding"] is None for e in target_entities)
    assert all("fact_embedding" not in e or e["fact_embedding"] is None for e in target_edges)
    assert all(e.get("episodes") == [] for e in target_edges)
    assert all(e.get("source_node_uuid") != e.get("uuid") for e in target_edges)


async def test_import_pack_is_idempotent_in_merge_mode():
    """Importing the same pack twice should update the same target rows."""

    driver = _make_driver()
    surriti = Surriti(driver, embedder=DummyEmbedder(embedding_dim=64))
    await _add_basic_episodes(surriti, group_id="source")

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "test.surriti-pack.zip"
        await surriti.export_memory_pack("source", str(out))
        first = await import_group_from_zip(driver, str(out), "target")
        second = await import_group_from_zip(driver, str(out), "target")

    assert first.validation.ok
    assert second.validation.ok
    assert len([r for r in driver.records["entity"] if r.get("group_id") == "target"]) == first.counts["entities"]
    assert len([r for r in driver.records["relates_to"] if r.get("group_id") == "target"]) == first.counts["edges"]


async def test_import_pack_preserves_superseded_history_and_relation_frames():
    """Current and invalidated fact state must survive import remapping."""

    driver = _make_driver()
    from surriti.llm import ScriptedLLMClient, ScriptedResponse, ExtractedEntity, ExtractedFact

    llm = ScriptedLLMClient(
        responses=[
            ScriptedResponse(
                entities=[
                    ExtractedEntity(name="Alice", labels=["Person"]),
                    ExtractedEntity(name="Acme Corp", labels=["Organization"]),
                ],
                facts=[
                    ExtractedFact(
                        subject="Alice",
                        predicate="works_at",
                        object="Acme Corp",
                        singleton=True,
                    )
                ],
            ),
            ScriptedResponse(
                entities=[
                    ExtractedEntity(name="Alice", labels=["Person"]),
                    ExtractedEntity(name="Globex", labels=["Organization"]),
                ],
                facts=[
                    ExtractedFact(
                        subject="Alice",
                        predicate="works_at",
                        object="Globex",
                        singleton=True,
                    )
                ],
            ),
        ]
    )
    surriti = Surriti(driver, llm_client=llm, embedder=DummyEmbedder(embedding_dim=64))
    await surriti.add_episode(
        name="e1",
        episode_body="Alice works at Acme Corp.",
        group_id="source",
    )
    await surriti.add_episode(
        name="e2",
        episode_body="Alice works at Globex.",
        group_id="source",
    )

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "test.surriti-pack.zip"
        await surriti.export_memory_pack("source", str(out))
        imported = await surriti.import_memory_pack(str(out), "target")

    assert imported["validation"]["ok"] is True
    target_edges = [r for r in driver.records["relates_to"] if r.get("group_id") == "target"]
    assert any(e.get("status") == "superseded" and e.get("invalid_at") for e in target_edges)
    assert any(e.get("status") == "active" for e in target_edges)
    assert all(str(e.get("fact_key", "")).startswith("target::") for e in target_edges)
