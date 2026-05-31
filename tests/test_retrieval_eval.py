import pytest

from enterprise_rag.evaluation.models import RetrievalEvalCase
from enterprise_rag.evaluation.retrieval_eval import (
    evaluate_retrieval_case,
    evaluate_retrieval_cases,
)
from enterprise_rag.retrieval.models import SearchResult


def test_evaluate_retrieval_case_hits_expected_document() -> None:
    case = RetrievalEvalCase(
        query="python programming",
        expected_document_id="doc_python",
    )

    def search(query: str, top_k: int) -> list[SearchResult]:
        return [
            SearchResult(
                chunk_id="chunk_python_0",
                document_id="doc_python",
                title="Python",
                score=1.2,
                text="Python is a programming language.",
                metadata={},
            )
        ]

    result = evaluate_retrieval_case(
        case=case,
        search=search,
        top_k=5,
    )

    assert result.hit is True
    assert result.matched_document_id == "doc_python"
    assert result.matched_chunk_id == "chunk_python_0"


def test_evaluate_retrieval_case_misses_wrong_document() -> None:
    case = RetrievalEvalCase(
        query="python programming",
        expected_document_id="doc_python",
    )

    def search(query: str, top_k: int) -> list[SearchResult]:
        return [
            SearchResult(
                chunk_id="chunk_cooking_0",
                document_id="doc_cooking",
                title="Cooking",
                score=0.8,
                text="Cooking pasta requires boiling water.",
                metadata={},
            )
        ]

    result = evaluate_retrieval_case(
        case=case,
        search=search,
        top_k=5,
    )

    assert result.hit is False
    assert result.matched_document_id is None
    assert result.matched_chunk_id is None


def test_evaluate_retrieval_case_can_match_expected_chunk() -> None:
    case = RetrievalEvalCase(
        query="python programming",
        expected_document_id="doc_python",
        expected_chunk_id="chunk_python_1",
    )

    def search(query: str, top_k: int) -> list[SearchResult]:
        return [
            SearchResult(
                chunk_id="chunk_python_0",
                document_id="doc_python",
                title="Python",
                score=1.0,
                text="Intro chunk.",
                metadata={},
            ),
            SearchResult(
                chunk_id="chunk_python_1",
                document_id="doc_python",
                title="Python",
                score=0.9,
                text="Expected chunk.",
                metadata={},
            ),
        ]

    result = evaluate_retrieval_case(
        case=case,
        search=search,
        top_k=5,
    )

    assert result.hit is True
    assert result.matched_chunk_id == "chunk_python_1"


def test_summarize_retrieval_results_calculates_hit_rate() -> None:
    cases = [
        RetrievalEvalCase(query="python", expected_document_id="doc_python"),
        RetrievalEvalCase(query="cooking", expected_document_id="doc_cooking"),
    ]

    def search(query: str, top_k: int) -> list[SearchResult]:
        if query == "python":
            return [
                SearchResult(
                    chunk_id="chunk_python_0",
                    document_id="doc_python",
                    title="Python",
                    score=1.0,
                    text="Python text.",
                    metadata={},
                )
            ]

        return []

    results, summary = evaluate_retrieval_cases(
        cases=cases,
        search=search,
        top_k=5,
    )

    assert len(results) == 2
    assert summary.total_cases == 2
    assert summary.hits == 1
    assert summary.misses == 1
    assert summary.hit_rate == 0.5


def test_evaluate_retrieval_case_rejects_invalid_top_k() -> None:
    case = RetrievalEvalCase(
        query="python",
        expected_document_id="doc_python",
    )

    def search(query: str, top_k: int) -> list[SearchResult]:
        return []

    with pytest.raises(ValueError, match="top_k must be greater than 0"):
        evaluate_retrieval_case(
            case=case,
            search=search,
            top_k=0,
        )