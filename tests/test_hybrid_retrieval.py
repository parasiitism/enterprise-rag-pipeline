from __future__ import annotations

from collections.abc import Sequence

import pytest

from enterprise_rag.embeddings.models import (
    EmbeddingProviderInfo,
    EmbeddingRequest,
    EmbeddingResult,
)
from enterprise_rag.retrieval.bm25 import BM25ChunkIndex
from enterprise_rag.retrieval.hybrid import HybridRetriever
from enterprise_rag.vector_store.models import VectorRecord, VectorSearchResult


class FakeEmbeddingProvider:
    @property
    def info(self) -> EmbeddingProviderInfo:
        return EmbeddingProviderInfo(
            provider="fake",
            model="fake-embedding-model",
            dimensions=2,
            max_batch_size=8,
        )

    def embed_texts(
        self,
        requests: Sequence[EmbeddingRequest],
    ) -> list[EmbeddingResult]:
        return [
            EmbeddingResult(
                record_id=request.record_id,
                vector=[1.0, 0.0],
                provider="fake",
                model="fake-embedding-model",
                dimensions=2,
                metadata=request.metadata,
            )
            for request in requests
        ]


class FakeVectorIndex:
    def __init__(self, results: list[VectorSearchResult]) -> None:
        self.results = results
        self.last_query_vector: Sequence[float] | None = None

    def search(
        self,
        query_vector: Sequence[float],
        top_k: int,
    ) -> list[VectorSearchResult]:
        self.last_query_vector = query_vector
        return self.results[:top_k]


def test_hybrid_retriever_fuses_bm25_and_semantic_results() -> None:
    chunk_records = [
        _chunk_record(
            chunk_id="chunk_python",
            document_id="doc_python",
            title="Python",
            text="Python is a programming language with readable syntax.",
        ),
        _chunk_record(
            chunk_id="chunk_guido",
            document_id="doc_guido",
            title="Guido",
            text="Guido created a language focused on readability.",
        ),
    ]

    vector_index = FakeVectorIndex(
        [
            _vector_result(
                chunk_id="chunk_guido",
                document_id="doc_guido",
                title="Guido",
                score=0.91,
                rank=1,
            ),
            _vector_result(
                chunk_id="chunk_python",
                document_id="doc_python",
                title="Python",
                score=0.89,
                rank=2,
            ),
        ]
    )

    retriever = HybridRetriever(
        chunk_records=chunk_records,
        bm25_index=BM25ChunkIndex(chunk_records),
        vector_index=vector_index,
        embedding_provider=FakeEmbeddingProvider(),
    )

    results = retriever.search(
        "python programming",
        top_k=2,
        bm25_top_k=5,
        semantic_top_k=5,
    )

    assert len(results) == 2
    assert results[0].chunk_id == "chunk_python"
    assert results[0].matched_by == ("bm25", "semantic")
    assert results[0].bm25_rank == 1
    assert results[0].semantic_rank == 2
    assert results[1].chunk_id == "chunk_guido"
    assert results[1].matched_by == ("semantic",)
    assert vector_index.last_query_vector == [1.0, 0.0]


def test_hybrid_retriever_ignores_semantic_results_without_chunk_record() -> None:
    chunk_records = [
        _chunk_record(
            chunk_id="chunk_python",
            document_id="doc_python",
            title="Python",
            text="Python is a programming language.",
        )
    ]

    vector_index = FakeVectorIndex(
        [
            _vector_result(
                chunk_id="missing_chunk",
                document_id="missing_doc",
                title="Missing",
                score=0.99,
                rank=1,
            )
        ]
    )

    retriever = HybridRetriever(
        chunk_records=chunk_records,
        bm25_index=BM25ChunkIndex(chunk_records),
        vector_index=vector_index,
        embedding_provider=FakeEmbeddingProvider(),
    )

    results = retriever.search("python", top_k=5)

    assert [result.chunk_id for result in results] == ["chunk_python"]
    assert results[0].matched_by == ("bm25",)


def test_hybrid_retriever_rejects_empty_query() -> None:
    chunk_records = [
        _chunk_record(
            chunk_id="chunk_python",
            document_id="doc_python",
            title="Python",
            text="Python is a programming language.",
        )
    ]

    retriever = HybridRetriever(
        chunk_records=chunk_records,
        bm25_index=BM25ChunkIndex(chunk_records),
        vector_index=FakeVectorIndex([]),
        embedding_provider=FakeEmbeddingProvider(),
    )

    with pytest.raises(ValueError, match="query must be a non-empty string"):
        retriever.search("   ")


def _chunk_record(
    chunk_id: str,
    document_id: str,
    title: str,
    text: str,
) -> dict[str, object]:
    return {
        "chunk_id": chunk_id,
        "document_id": document_id,
        "source": "test",
        "source_id": document_id,
        "title": title,
        "chunk_index": 0,
        "text": text,
        "metadata": {"document_id": document_id},
    }


def _vector_result(
    chunk_id: str,
    document_id: str,
    title: str,
    score: float,
    rank: int,
) -> VectorSearchResult:
    return VectorSearchResult(
        record=VectorRecord(
            record_id=f"vector_{chunk_id}",
            chunk_id=chunk_id,
            document_id=document_id,
            title=title,
            vector=[1.0, 0.0],
            metadata={},
        ),
        score=score,
        rank=rank,
    )
