from __future__ import annotations

import argparse
from pathlib import Path

from enterprise_rag.chunk_store import load_chunk_records
from enterprise_rag.evaluation.dataset import load_retrieval_eval_cases
from enterprise_rag.evaluation.models import RetrievalEvalResult
from enterprise_rag.evaluation.retrieval_eval import evaluate_retrieval_cases
from enterprise_rag.retrieval.bm25 import BM25ChunkIndex


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate BM25 retrieval quality")

    parser.add_argument(
        "chunks_dir",
        type=Path,
        help="Directory containing chunk JSON files",
    )

    parser.add_argument(
        "eval_path",
        type=Path,
        help="JSONL file containing retrieval evaluation cases",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of search results to evaluate per query",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of chunk files to load",
    )

    parser.add_argument(
        "--show-misses",
        action="store_true",
        help="Print failed retrieval cases",
    )

    args = parser.parse_args()

    cases = load_retrieval_eval_cases(args.eval_path)
    chunk_records = load_chunk_records(args.chunks_dir, limit=args.limit)

    index = BM25ChunkIndex(chunk_records)

    results, summary = evaluate_retrieval_cases(
        cases=cases,
        search=index.search,
        top_k=args.top_k,
    )

    print("Retrieval Evaluation Summary")
    print("=" * 80)
    print(f"Eval cases: {summary.total_cases}")
    print(f"Hits: {summary.hits}")
    print(f"Misses: {summary.misses}")
    print(f"Hit@{args.top_k}: {summary.hit_rate:.2%}")

    if args.show_misses:
        _print_misses(results)


def _print_misses(results: list[RetrievalEvalResult]) -> None:
    misses = [result for result in results if not result.hit]

    if not misses:
        print("\nNo misses found.")
        return

    print("\nMisses")
    print("=" * 80)

    for result in misses:
        print(f"\nQuery: {result.case.query}")
        print(f"Expected document: {result.case.expected_document_id}")
        print(f"Expected chunk: {result.case.expected_chunk_id}")

        if not result.retrieved_results:
            print("Retrieved: no results")
            continue

        print("Retrieved top results:")

        for search_result in result.retrieved_results[:3]:
            print(
                f"- {search_result.document_id} | "
                f"{search_result.chunk_id} | "
                f"{search_result.title} | "
                f"score={search_result.score:.4f}"
            )


if __name__ == "__main__":
    main()