from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from enterprise_rag.embeddings import EmbeddingProvider, EmbeddingRequest
from enterprise_rag.retrieval.bm25 import BM25ChunkIndex
from enterprise_rag.retrieval.models import SearchResult
from enterprise_rag.vector_store.models import VectorSearchResult


@dataclass(frozen=True)
class HybridSearchResult:
    chunk_id: str
    document_id: str
    title: str
    text: str
    metadata: dict[str, object]
    fused_score: float
    bm25_score: float | None
    semantic_score: float | None
    bm25_rank: int | None
    semantic_rank: int | None
    matched_by: tuple[str, ...]


class VectorSearcher(Protocol):
    def search(
        self,
        query_vector: Sequence[float],
        top_k: int,
    ) -> list[VectorSearchResult]:
        raise NotImplementedError


@dataclass
class _HybridCandidate:
    chunk_id: str
    document_id: str
    title: str
    text: str
    metadata: dict[str, object]
    fused_score: float = 0.0
    bm25_score: float | None = None
    semantic_score: float | None = None
    bm25_rank: int | None = None
    semantic_rank: int | None = None

    @property
    def matched_by(self) -> tuple[str, ...]:
        matches: list[str] = []

        if self.bm25_rank is not None:
            matches.append("bm25")

        if self.semantic_rank is not None:
            matches.append("semantic")

        return tuple(matches)


class HybridRetriever:
    def __init__(
        self,
        chunk_records: Sequence[dict[str, object]],
        bm25_index: BM25ChunkIndex,
        vector_index: VectorSearcher,
        embedding_provider: EmbeddingProvider,
        rrf_k: int = 60,
        bm25_weight: float = 1.0,
        semantic_weight: float = 1.0,
    ) -> None:
        if not chunk_records:
            raise ValueError("chunk_records cannot be empty")

        if rrf_k <= 0:
            raise ValueError("rrf_k must be greater than 0")

        if bm25_weight <= 0:
            raise ValueError("bm25_weight must be greater than 0")

        if semantic_weight <= 0:
            raise ValueError("semantic_weight must be greater than 0")

        self._chunk_records_by_id = {
            str(record["chunk_id"]): record
            for record in chunk_records
        }
        self._bm25_index = bm25_index
        self._vector_index = vector_index
        self._embedding_provider = embedding_provider
        self._rrf_k = rrf_k
        self._bm25_weight = bm25_weight
        self._semantic_weight = semantic_weight

    def search(
        self,
        query: str,
        top_k: int = 5,
        bm25_top_k: int = 50,
        semantic_top_k: int = 50,
    ) -> list[HybridSearchResult]:
        query = query.strip()

        if not query:
            raise ValueError("query must be a non-empty string")

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        bm25_results = self._bm25_index.search(query, top_k=bm25_top_k)
        semantic_results = self._semantic_search(query, top_k=semantic_top_k)

        candidates: dict[str, _HybridCandidate] = {}

        self._merge_bm25_results(candidates, bm25_results)
        self._merge_semantic_results(candidates, semantic_results)

        ranked = sorted(
            candidates.values(),
            key=lambda candidate: candidate.fused_score,
            reverse=True,
        )

        return [
            HybridSearchResult(
                chunk_id=candidate.chunk_id,
                document_id=candidate.document_id,
                title=candidate.title,
                text=candidate.text,
                metadata=candidate.metadata,
                fused_score=candidate.fused_score,
                bm25_score=candidate.bm25_score,
                semantic_score=candidate.semantic_score,
                bm25_rank=candidate.bm25_rank,
                semantic_rank=candidate.semantic_rank,
                matched_by=candidate.matched_by,
            )
            for candidate in ranked[:top_k]
        ]

    def _semantic_search(
        self,
        query: str,
        top_k: int,
    ) -> list[VectorSearchResult]:
        embedding_results = self._embedding_provider.embed_texts(
            [EmbeddingRequest(record_id="query", text=query)]
        )

        if len(embedding_results) != 1:
            raise ValueError("embedding provider must return exactly one query vector")

        query_vector = embedding_results[0].vector
        return self._vector_index.search(query_vector=query_vector, top_k=top_k)

    def _merge_bm25_results(
        self,
        candidates: dict[str, _HybridCandidate],
        results: Sequence[SearchResult],
    ) -> None:
        for rank, result in enumerate(results, start=1):
            candidate = self._get_or_create_candidate(candidates, result.chunk_id)
            candidate.bm25_score = result.score
            candidate.bm25_rank = rank
            candidate.fused_score += self._bm25_weight / (self._rrf_k + rank)

    def _merge_semantic_results(
        self,
        candidates: dict[str, _HybridCandidate],
        results: Sequence[VectorSearchResult],
    ) -> None:
        for rank, result in enumerate(results, start=1):
            chunk_id = result.record.chunk_id

            if chunk_id not in self._chunk_records_by_id:
                continue

            candidate = self._get_or_create_candidate(candidates, chunk_id)
            candidate.semantic_score = result.score
            candidate.semantic_rank = rank
            candidate.fused_score += self._semantic_weight / (self._rrf_k + rank)

    def _get_or_create_candidate(
        self,
        candidates: dict[str, _HybridCandidate],
        chunk_id: str,
    ) -> _HybridCandidate:
        if chunk_id in candidates:
            return candidates[chunk_id]

        record = self._chunk_records_by_id[chunk_id]

        candidate = _HybridCandidate(
            chunk_id=chunk_id,
            document_id=str(record["document_id"]),
            title=str(record["title"]),
            text=str(record["text"]),
            metadata=dict(record["metadata"]),
        )

        candidates[chunk_id] = candidate
        return candidate