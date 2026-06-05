"""Embedder interfaces.

Surriti is intentionally agnostic about which embedding model you use. The
``EmbedderClient`` ABC mirrors Graphiti's interface so existing Graphiti
embedders can be reused with a thin adapter.
"""

from __future__ import annotations

import hashlib
import math
from abc import ABC, abstractmethod
from collections.abc import Iterable


class EmbedderClient(ABC):
    """Abstract embedder. Returns L2-normalised float vectors."""

    embedding_dim: int

    @abstractmethod
    async def create(self, input_data: str) -> list[float]:
        ...

    async def create_batch(self, input_data: list[str]) -> list[list[float]]:
        return [await self.create(text) for text in input_data]


class DummyEmbedder(EmbedderClient):
    """Deterministic, offline embedder useful for tests and demos.

    Produces a hashed bag-of-tokens vector so semantically related strings
    share at least some dimensions. Not suitable for production retrieval
    but lets every other layer of Surriti be exercised without API keys.
    """

    def __init__(self, embedding_dim: int = 768) -> None:
        self.embedding_dim = embedding_dim

    async def create(self, input_data: str) -> list[float]:
        vec = [0.0] * self.embedding_dim
        tokens = (input_data or "").lower().split()
        if not tokens:
            return vec
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            # Spread the token across two coordinates with opposite signs so
            # that the resulting vector has both positive and negative entries.
            idx = int.from_bytes(digest[:4], "big") % self.embedding_dim
            vec[idx] += 1.0
            idx2 = int.from_bytes(digest[4:8], "big") % self.embedding_dim
            vec[idx2] -= 1.0
        norm = math.sqrt(sum(x * x for x in vec))
        if norm < 1e-10:
            return vec
        return [x / norm for x in vec]


class OpenAIEmbedder(EmbedderClient):
    """OpenAI ``text-embedding-3-*`` adapter. Optional dependency.
    
    Works with any OpenAI-compatible endpoint (vLLM, Ollama, etc.).
    Pass ``api_key="EMPTY"`` when the server doesn't require auth.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        embedding_dim: int = 1536,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        try:
            from openai import AsyncOpenAI  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - import guard
            raise ImportError(
                "Install surriti[openai] to use OpenAIEmbedder."
            ) from exc

        self.model = model
        self.embedding_dim = embedding_dim
        kwargs: dict = {"api_key": api_key or "EMPTY"}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = AsyncOpenAI(**kwargs)

    async def create(self, input_data: str) -> list[float]:
        response = await self._client.embeddings.create(
            model=self.model, input=input_data, dimensions=self.embedding_dim
        )
        return response.data[0].embedding

    async def create_batch(self, input_data: list[str]) -> list[list[float]]:
        if not input_data:
            return []
        response = await self._client.embeddings.create(
            model=self.model, input=input_data, dimensions=self.embedding_dim
        )
        return [item.embedding for item in response.data]


def cosine_similarity(a: Iterable[float], b: Iterable[float]) -> float:
    a_list = list(a)
    b_list = list(b)
    if not a_list or not b_list:
        return 0.0
    dot = sum(x * y for x, y in zip(a_list, b_list, strict=False))
    na = math.sqrt(sum(x * x for x in a_list))
    nb = math.sqrt(sum(y * y for y in b_list))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)
