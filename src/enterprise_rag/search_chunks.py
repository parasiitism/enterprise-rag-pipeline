from __future__ import annotations

import argparse
from pathlib import Path

from enterprise_rag.chunk_store import load_chunk_records
from enterprise_rag.retrieval.bm25 import BM25ChunkIndex


def main() -> None:
    parser = argparse.ArgumentParser(description="Search chunk JSON files with BM25")

    parser.add_argument(
        "chunks_dir",
        type=Path,
        help="Directory containing chunk JSON files",
    )

    parser.add_argument(
        "query",
        type=str,
        help="Search query",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of chunk files to load",
    )

    args = parser.parse_args()

    chunk_records = load_chunk_records(args.chunks_dir, limit=args.limit)
    index = BM25ChunkIndex(chunk_records)
    results = index.search(args.query, top_k=args.top_k)

    print(f"Query: {args.query}")
    print(f"Results: {len(results)}")

    for result in results:
        print("\n" + "=" * 80)
        print(f"Score: {result.score:.4f}")
        print(f"Chunk ID: {result.chunk_id}")
        print(f"Document ID: {result.document_id}")
        print(f"Title: {result.title}")
        print("-" * 80)
        print(result.text[:700])


if __name__ == "__main__":
    main()