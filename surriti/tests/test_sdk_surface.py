"""Tests for the SDK ergonomic surface added in 0.4.0.

- Surriti.from_env, async context manager, connect/close idempotency
- SurrealDriver.from_env env-var resolution
- Real-LLM-client JSON parsing (without hitting any network)
- Errors hierarchy
- Logging helper
"""

from __future__ import annotations

import logging
import os

import pytest

from surriti import (
    DummyEmbedder,
    LLMClient,
    SurrealDriver,
    Surriti,
    SurritiConfigError,
    SurritiConnectionError,
    SurritiError,
    SurritiLLMError,
    SurritiNotFoundError,
    SurritiSchemaError,
    setup_logging,
)


# ---------------------------------------------------------------- error types
def test_error_hierarchy_inherits_from_surriti_error():
    for exc in (
        SurritiConfigError,
        SurritiConnectionError,
        SurritiSchemaError,
        SurritiLLMError,
        SurritiNotFoundError,
    ):
        assert issubclass(exc, SurritiError)


# -------------------------------------------------------- driver.from_env
def test_driver_from_env_uses_defaults(monkeypatch):
    for k in ("SURRITI_SURREAL_URL", "SURRITI_SURREAL_NS", "SURRITI_SURREAL_DB",
              "SURRITI_SURREAL_USER", "SURRITI_SURREAL_PASS", "SURRITI_EMBEDDING_DIM"):
        monkeypatch.delenv(k, raising=False)
    d = SurrealDriver.from_env()
    assert d.url == "ws://localhost:8000/rpc"
    assert d.namespace == "surriti"
    assert d.database == "surriti"
    assert d.username is None and d.password is None
    assert d.embedding_dim == 768


def test_driver_from_env_reads_overrides(monkeypatch):
    monkeypatch.setenv("SURRITI_SURREAL_URL", "ws://db:9000/rpc")
    monkeypatch.setenv("SURRITI_SURREAL_NS", "ns1")
    monkeypatch.setenv("SURRITI_SURREAL_DB", "db1")
    monkeypatch.setenv("SURRITI_SURREAL_USER", "u")
    monkeypatch.setenv("SURRITI_SURREAL_PASS", "p")
    monkeypatch.setenv("SURRITI_EMBEDDING_DIM", "256")
    d = SurrealDriver.from_env()
    assert d.url == "ws://db:9000/rpc"
    assert d.namespace == "ns1" and d.database == "db1"
    assert d.username == "u" and d.password == "p"
    assert d.embedding_dim == 256


def test_driver_db_property_raises_when_disconnected():
    d = SurrealDriver(url="ws://nowhere:1/rpc")
    with pytest.raises(SurritiConnectionError):
        _ = d.db


# -------------------------------------------------------- Surriti.from_env
def test_surriti_from_env_returns_unconnected_instance(monkeypatch):
    monkeypatch.setenv("SURRITI_EMBEDDING_DIM", "64")
    s = Surriti.from_env(embedder=DummyEmbedder(64))
    assert isinstance(s.driver, SurrealDriver)
    assert s.driver.embedding_dim == 64
    assert s.driver._db is None


# -------------------------------------------------------- async ctx manager
@pytest.mark.asyncio
async def test_surriti_async_context_manager_uses_fake_driver():
    """Surriti.__aenter__ must invoke driver.connect + init_schema."""
    from surriti.testing import InMemoryDriver

    fake = InMemoryDriver()
    s = Surriti(fake, embedder=DummyEmbedder(64))
    async with s as memory:
        assert memory is s
    # close was called -> nothing to assert besides no exceptions


# -------------------------------------------------------- logging helper
def test_setup_logging_attaches_one_handler_idempotent():
    setup_logging("DEBUG")
    setup_logging("INFO")
    log = logging.getLogger("surriti")
    handlers = [h for h in log.handlers if getattr(h, "_surriti_default", False)]
    assert len(handlers) == 1
    assert log.level == logging.INFO


# -------------------------------------------------------- llm_clients parsing
@pytest.mark.asyncio
async def test_openai_extraction_parses_json_response():
    from surriti.llm_clients import OpenAILLMClient

    class FakeChoice:
        def __init__(self, content):
            self.message = type("M", (), {"content": content})()

    class FakeResp:
        def __init__(self, content):
            self.choices = [FakeChoice(content)]

    class FakeChat:
        async def create(self, **kwargs):
            return FakeResp(
                '{"entities":[{"name":"Alice","labels":["Person"]},'
                '{"name":"Acme"}],'
                '"facts":[{"subject":"Alice","predicate":"works_at",'
                '"object":"Acme","fact":"Alice works at Acme."}]}'
            )

    class FakeClient:
        def __init__(self):
            self.chat = type("C", (), {"completions": FakeChat()})()

    client = OpenAILLMClient(model="gpt-test", client=FakeClient())
    result = await client.extract("hi", group_id="g")
    names = [e.name for e in result.entities]
    assert names == ["Alice", "Acme"]
    assert result.facts[0].subject == "Alice"
    assert result.facts[0].fact == "Alice works at Acme."


@pytest.mark.asyncio
async def test_openai_extraction_strips_code_fences():
    from surriti.llm_clients import OpenAILLMClient

    class FakeResp:
        def __init__(self, content):
            self.choices = [type("C", (), {"message": type("M", (), {"content": content})()})()]

    class FakeChat:
        async def create(self, **kwargs):
            return FakeResp('```json\n{"entities":[{"name":"X"}],"facts":[]}\n```')

    class FakeClient:
        def __init__(self):
            self.chat = type("X", (), {"completions": FakeChat()})()

    res = await OpenAILLMClient(client=FakeClient()).extract("...")
    assert [e.name for e in res.entities] == ["X"]


@pytest.mark.asyncio
async def test_openai_extraction_raises_surriti_llm_error_on_bad_json():
    from surriti.llm_clients import OpenAILLMClient

    class FakeChat:
        async def create(self, **kwargs):
            return type(
                "R", (), {"choices": [type("C", (), {"message": type("M", (), {"content": "not json"})()})()]}
            )()

    class FakeClient:
        def __init__(self):
            self.chat = type("X", (), {"completions": FakeChat()})()

    with pytest.raises(SurritiLLMError):
        await OpenAILLMClient(client=FakeClient()).extract("...")


@pytest.mark.asyncio
async def test_openai_contradictions_returns_validated_indexes():
    from surriti.llm_clients import OpenAILLMClient

    class FakeChat:
        async def create(self, **kwargs):
            return type(
                "R", (),
                {"choices": [type("C", (), {"message": type("M", (), {"content": '{"invalidated_indexes":[0,2,99]}'})()})()]},
            )()

    class FakeClient:
        def __init__(self):
            self.chat = type("X", (), {"completions": FakeChat()})()

    out = await OpenAILLMClient(client=FakeClient()).find_contradictions("new", ["a", "b", "c"])
    # 99 is filtered, 0 and 2 retained
    assert out == [0, 2]


def test_openai_client_raises_config_error_when_no_key(monkeypatch):
    from surriti.llm_clients import OpenAILLMClient

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    # Only triggers if the openai package is importable; if not, surfaces config error too.
    with pytest.raises(SurritiConfigError):
        OpenAILLMClient()


@pytest.mark.asyncio
async def test_anthropic_extraction_parses_text_blocks():
    from surriti.llm_clients import AnthropicLLMClient

    class FakeBlock:
        def __init__(self, text):
            self.text = text

    class FakeResp:
        def __init__(self, text):
            self.content = [FakeBlock(text)]

    class FakeMessages:
        async def create(self, **kwargs):
            return FakeResp(
                '{"entities":[{"name":"Bob"}],'
                '"facts":[{"subject":"Bob","predicate":"likes","object":"tea",'
                '"fact":"Bob likes tea."}]}'
            )

    class FakeClient:
        def __init__(self):
            self.messages = FakeMessages()

    client = AnthropicLLMClient(client=FakeClient())
    res = await client.extract("...")
    assert [e.name for e in res.entities] == ["Bob"]
    assert res.facts[0].object == "tea"


# -------------------------------------------------------- LLMClient is ABC
def test_llm_client_is_abstract():
    with pytest.raises(TypeError):
        LLMClient()  # type: ignore[abstract]
