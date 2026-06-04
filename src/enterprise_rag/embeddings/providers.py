# src/enterprise_rag/embeddings/providers.py

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from enterprise_rag.embeddings.models import (
    EmbeddingProviderInfo,
    EmbeddingRequest,
    EmbeddingResult,
)


class EmbeddingProvider(Protocol):
    @property
    def info(self) -> EmbeddingProviderInfo:
        raise NotImplementedError

    def embed_texts(
        self,
        requests: Sequence[EmbeddingRequest],
    ) -> list[EmbeddingResult]:
        raise NotImplementedError