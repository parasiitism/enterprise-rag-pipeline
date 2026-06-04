# src/enterprise_rag/embeddings/models.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


@dataclass(frozen=True)
class EmbeddingRequest:
    record_id: str
    text: str
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class EmbeddingResult:
    record_id: str
    vector: list[float]
    provider: str
    model: str
    dimensions: int
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class EmbeddingProviderInfo:
    provider: str
    model: str
    dimensions: int
    max_batch_size: int