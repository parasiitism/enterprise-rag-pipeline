from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from enterprise_rag.embeddings import (
    DEFAULT_HUGGINGFACE_EMBEDDING_DIMENSIONS,
    DEFAULT_HUGGINGFACE_EMBEDDING_MODEL,
    EmbeddingProvider,
    EmbeddingRequest,
    HuggingFaceEmbeddingProvider,
)
from enterprise_rag.vector_store import (
    HNSWVectorIndex,
    VectorMetric,
    VectorRecord,
    VectorSearchResult,
    load_vector_records,
)


def embed_query(query: str, provider: EmbeddingProvider) -> list[float]:
    query = query.strip()

    if not query:
        raise ValueError("query must be a non-empty string")

    results = provider.embed_texts(
        [EmbeddingRequest(record_id="query", text=query)]
    )

    if len(results) != 1:
        raise ValueError("query embedding provider must return exactly one result")

    return results[0].vector


def build_hnsw_index(
    records: Sequence[VectorRecord],
    metric: VectorMetric = "cosine",
) -> HNSWVectorIndex:
    if not records:
        raise ValueError("records cannot be empty")

    index = HNSWVectorIndex(
        dimensions=len(records[0].vector),
        metric=metric,
        max_elements=len(records),
    )
    index.add(records)

    return index


def semantic_search(
    records: Sequence[VectorRecord],
    query: str,
    provider: EmbeddingProvider,
    top_k: int,
    metric: VectorMetric = "cosine",
) -> list[VectorSearchResult]:
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")

    query_vector = embed_query(query=query, provider=provider)
    index = build_hnsw_index(records=records, metric=metric)

    return index.search(query_vector=query_vector, top_k=top_k)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run semantic search over saved embeddings")

    parser.add_argument("embeddings_dir", type=Path)
    parser.add_argument("query", type=str)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--metric", choices=["cosine", "l2", "ip"], default="cosine")
    parser.add_argument("--model-name", type=str, default=DEFAULT_HUGGINGFACE_EMBEDDING_MODEL)
    parser.add_argument("--dimensions", type=int, default=DEFAULT_HUGGINGFACE_EMBEDDING_DIMENSIONS)
    parser.add_argument("--device", type=str, default=None)

    args = parser.parse_args()

    records = load_vector_records(args.embeddings_dir, limit=args.limit)

    provider = HuggingFaceEmbeddingProvider(
        model_name=args.model_name,
        dimensions=args.dimensions,
        device=args.device,
    )

    results = semantic_search(
        records=records,
        query=args.query,
        provider=provider,
        top_k=args.top_k,
        metric=args.metric,
    )

    print(f"Query: {args.query}")
    print(f"Records indexed: {len(records)}")
    print(f"Results: {len(results)}")

    for result in results:
        print("\n" + "=" * 80)
        print(f"Rank: {result.rank}")
        print(f"Score: {result.score:.4f}")
        print(f"Chunk ID: {result.record.chunk_id}")
        print(f"Document ID: {result.record.document_id}")
        print(f"Title: {result.record.title}")


if __name__ == "__main__":
    main()