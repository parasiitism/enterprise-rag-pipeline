import json

import pytest

from enterprise_rag.evaluation.dataset import load_retrieval_eval_cases


def test_load_retrieval_eval_cases_reads_jsonl(tmp_path) -> None:
    eval_path = tmp_path / "retrieval_eval.jsonl"

    eval_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "query": "python programming language",
                        "expected_document_id": "doc_python",
                        "expected_chunk_id": "chunk_python_0",
                    }
                ),
                json.dumps(
                    {
                        "query": "travel reimbursement deadline",
                        "expected_document_id": "doc_travel_policy",
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )

    cases = load_retrieval_eval_cases(eval_path)

    assert len(cases) == 2
    assert cases[0].query == "python programming language"
    assert cases[0].expected_document_id == "doc_python"
    assert cases[0].expected_chunk_id == "chunk_python_0"
    assert cases[1].expected_chunk_id is None


def test_load_retrieval_eval_cases_rejects_missing_query(tmp_path) -> None:
    eval_path = tmp_path / "retrieval_eval.jsonl"

    eval_path.write_text(
        json.dumps({"expected_document_id": "doc_python"}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="query must be a non-empty string"):
        load_retrieval_eval_cases(eval_path)


def test_load_retrieval_eval_cases_rejects_empty_file(tmp_path) -> None:
    eval_path = tmp_path / "retrieval_eval.jsonl"
    eval_path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="No evaluation cases found"):
        load_retrieval_eval_cases(eval_path)