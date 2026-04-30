"""Surriti — Temporal knowledge-graph memory for AI agents, on SurrealDB.

This is the public SDK surface. A typical app needs only the top-level
:class:`Surriti` class plus an LLM client and an embedder::

    import asyncio
    from surriti import Surriti, OpenAIEmbedder
    from surriti.llm_clients import OpenAILLMClient

    async def main():
        async with Surriti.from_env(
            llm_client=OpenAILLMClient(model="gpt-4o-mini"),
            embedder=OpenAIEmbedder(model="text-embedding-3-small"),
        ) as memory:
            await memory.add_episode(
                name="onboarding",
                episode_body="Alice joined Acme as a staff engineer.",
                group_id="user-42",
            )
            results = await memory.search("where does Alice work?", group_id="user-42")
            for edge in results.edges:
                print(edge.fact)

    asyncio.run(main())

See ``examples/`` for fuller scenarios.
"""

from surriti._logging import setup_logging
from surriti.driver import SurrealDriver
from surriti.edges import CommunityEdge, EntityEdge, EpisodicEdge
from surriti.embedder import DummyEmbedder, EmbedderClient, OpenAIEmbedder
from surriti.errors import (
    SurritiConfigError,
    SurritiConnectionError,
    SurritiError,
    SurritiLLMError,
    SurritiNotFoundError,
    SurritiSchemaError,
)
from surriti.graphiti import (
    AddBulkEpisodeResults,
    AddEpisodeResults,
    AddTripletResults,
    RawEpisode,
    Surriti,
)
from surriti.llm import (
    DummyLLMClient,
    ExtractedEntity,
    ExtractedFact,
    ExtractionResult,
    LLMClient,
    ScriptedLLMClient,
    ScriptedResponse,
)
from surriti.nodes import CommunityNode, EntityNode, EpisodeType, EpisodicNode
from surriti.rerankers import CrossEncoderClient, DummyCrossEncoder
from surriti.search import Reranker, SearchConfig, SearchResults
from surriti.search_filters import (
    ComparisonOperator,
    DateFilter,
    PropertyFilter,
    SearchFilters,
)

__all__ = [
    # Core facade
    "Surriti",
    # Result dataclasses
    "AddBulkEpisodeResults",
    "AddEpisodeResults",
    "AddTripletResults",
    "RawEpisode",
    # Driver / schema
    "SurrealDriver",
    # Node & edge models
    "CommunityEdge",
    "CommunityNode",
    "EntityEdge",
    "EntityNode",
    "EpisodeType",
    "EpisodicEdge",
    "EpisodicNode",
    # LLM
    "DummyLLMClient",
    "ExtractedEntity",
    "ExtractedFact",
    "ExtractionResult",
    "LLMClient",
    "ScriptedLLMClient",
    "ScriptedResponse",
    # Embeddings
    "DummyEmbedder",
    "EmbedderClient",
    "OpenAIEmbedder",
    # Reranking
    "CrossEncoderClient",
    "DummyCrossEncoder",
    # Search
    "Reranker",
    "SearchConfig",
    "SearchResults",
    # Filters
    "ComparisonOperator",
    "DateFilter",
    "PropertyFilter",
    "SearchFilters",
    # Errors
    "SurritiConfigError",
    "SurritiConnectionError",
    "SurritiError",
    "SurritiLLMError",
    "SurritiNotFoundError",
    "SurritiSchemaError",
    # Helpers
    "setup_logging",
]

__version__ = "0.5.0"

