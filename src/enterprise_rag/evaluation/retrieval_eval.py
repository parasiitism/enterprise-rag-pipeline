from __future__ import annotations

from collections.abc import Callable

from enterprise_rag.evaluation.models import (
    RetrievalEvalCase,
    RetrievalEvalResult,
    RetrievalEvalSummary,
)
from enterprise_rag.retrieval.models import SearchResult


SearchFunction = Callable[[str, int], list[SearchResult]]


def evaluate_retrieval_case(
    case: RetrievalEvalCase,
    search: SearchFunction,
    top_k: int,
) -> RetrievalEvalResult:
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")

    retrieved_results = search(case.query, top_k)

    matched_document_id: str | None = None
    matched_chunk_id: str | None = None

    for result in retrieved_results:
        document_matches = result.document_id == case.expected_document_id
        chunk_matches = (
            case.expected_chunk_id is None
            or result.chunk_id == case.expected_chunk_id
        )

        if document_matches and chunk_matches:
            matched_document_id = result.document_id
            matched_chunk_id = result.chunk_id
            break

    hit = matched_document_id is not None

    return RetrievalEvalResult(
        case=case,
        retrieved_results=retrieved_results,
        top_k=top_k,
        hit=hit,
        matched_document_id=matched_document_id,
        matched_chunk_id=matched_chunk_id,
    )


def summarize_retrieval_results(
    results: list[RetrievalEvalResult],
) -> RetrievalEvalSummary:
    total_cases = len(results)
    hits = sum(1 for result in results if result.hit)
    misses = total_cases - hits
    hit_rate = hits / total_cases if total_cases else 0.0

    return RetrievalEvalSummary(
        total_cases=total_cases,
        hits=hits,
        misses=misses,
        hit_rate=hit_rate,
    )


def evaluate_retrieval_cases(
    cases: list[RetrievalEvalCase],
    search: SearchFunction,
    top_k: int,
) -> tuple[list[RetrievalEvalResult], RetrievalEvalSummary]:
    results = [
        evaluate_retrieval_case(
            case=case,
            search=search,
            top_k=top_k,
        )
        for case in cases
    ]

    summary = summarize_retrieval_results(results)

    return results, summary