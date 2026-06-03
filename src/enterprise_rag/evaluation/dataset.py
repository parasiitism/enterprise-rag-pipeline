from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from enterprise_rag.evaluation.models import RetrievalEvalCase


def load_retrieval_eval_cases(path: Path) -> list[RetrievalEvalCase]:
    if not path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found: {path}")

    cases: list[RetrievalEvalCase] = []

    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()

        if not line:
            continue

        payload = json.loads(line)
        cases.append(_parse_eval_case(payload, line_number))

    if not cases:
        raise ValueError(f"No evaluation cases found in: {path}")

    return cases


def _parse_eval_case(payload: dict[str, Any], line_number: int) -> RetrievalEvalCase:
    query = payload.get("query")
    expected_document_id = payload.get("expected_document_id")
    expected_chunk_id = payload.get("expected_chunk_id")

    if not isinstance(query, str) or not query.strip():
        raise ValueError(f"Line {line_number}: query must be a non-empty string")

    if not isinstance(expected_document_id, str) or not expected_document_id.strip():
        raise ValueError(
            f"Line {line_number}: expected_document_id must be a non-empty string"
        )

    if expected_chunk_id is not None and not isinstance(expected_chunk_id, str):
        raise ValueError(f"Line {line_number}: expected_chunk_id must be a string or null")

    return RetrievalEvalCase(
        query=query,
        expected_document_id=expected_document_id,
        expected_chunk_id=expected_chunk_id,
    )