from __future__ import annotations

import argparse
from pathlib import Path

from enterprise_rag.chunk_store import load_chunk_records
from enterprise_rag.embeddings import (
    DEFAULT_HUGGINGFACE_EMBEDDING_DIMENSIONS,
    DEFAULT_HUGGINGFACE_EMBEDDING_MODEL,
    HuggingFaceEmbeddingProvider,
)
from enterprise_rag.retrieval.bm25 import BM25ChunkIndex
from enterprise_rag.retrieval.hybrid import HybridRetriever
from enterprise_rag.search_vectors import build_hnsw_index
from enterprise_rag.vector_store import load_vector_records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run hybrid retrieval with BM25 + HNSW semantic search",
    )

    parser.add_argument("chunks_dir", type=Path)
    parser.add_argument("embeddings_dir", type=Path)
    parser.add_argument("query", type=str)

    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--bm25-top-k", type=int, default=50)
    parser.add_argument("--semantic-top-k", type=int, default=50)
    parser.add_argument("--limit", type=int, default=None)

    parser.add_argument("--metric", choices=["cosine", "l2", "ip"], default="cosine")
    parser.add_argument("--model-name", type=str, default=DEFAULT_HUGGINGFACE_EMBEDDING_MODEL)
    parser.add_argument("--dimensions", type=int, default=DEFAULT_HUGGINGFACE_EMBEDDING_DIMENSIONS)
    parser.add_argument("--device", type=str, default=None)

    args = parser.parse_args()

    chunk_records = load_chunk_records(args.chunks_dir, limit=args.limit)
    vector_records = load_vector_records(args.embeddings_dir, limit=args.limit)

    bm25_index = BM25ChunkIndex(chunk_records)
    vector_index = build_hnsw_index(
        records=vector_records,
        metric=args.metric,
    )

    embedding_provider = HuggingFaceEmbeddingProvider(
        model_name=args.model_name,
        dimensions=args.dimensions,
        device=args.device,
    )

    retriever = HybridRetriever(
        chunk_records=chunk_records,
        bm25_index=bm25_index,
        vector_index=vector_index,
        embedding_provider=embedding_provider,
    )

    results = retriever.search(
        query=args.query,
        top_k=args.top_k,
        bm25_top_k=args.bm25_top_k,
        semantic_top_k=args.semantic_top_k,
    )

    print(f"Query: {args.query}")
    print(f"Chunks loaded: {len(chunk_records)}")
    print(f"Vectors loaded: {len(vector_records)}")
    print(f"Results: {len(results)}")

    for result in results:
        print("\n" + "=" * 80)
        print(f"Fused score: {result.fused_score:.6f}")
        print(f"BM25 score: {result.bm25_score}")
        print(f"Semantic score: {result.semantic_score}")
        print(f"BM25 rank: {result.bm25_rank}")
        print(f"Semantic rank: {result.semantic_rank}")
        print(f"Matched by: {', '.join(result.matched_by)}")
        print(f"Chunk ID: {result.chunk_id}")
        print(f"Document ID: {result.document_id}")
        print(f"Title: {result.title}")
        print("-" * 80)
        print(result.text[:700])


if __name__ == "__main__":
    main()
