from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from enterprise_rag.vector_store.models import VectorRecord


def embedding_record_to_vector_record(record: dict[str, Any]) -> VectorRecord:
    vector = record.get("vector")

    if not isinstance(vector, list) or not vector:
        raise ValueError("embedding record vector must be a non-empty list")

    metadata = record.get("metadata", {})

    if not isinstance(metadata, dict):
        raise ValueError("embedding record metadata must be an object")

    return VectorRecord(
        record_id=_required_string(record, "record_id"),
        chunk_id=_required_string(record, "chunk_id"),
        document_id=_required_string(record, "document_id"),
        title=str(record.get("title") or ""),
        vector=[float(value) for value in vector],
        metadata=metadata,
    )


def iter_vector_records(input_dir: Path, limit: int | None = None) -> Iterator[VectorRecord]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Embedding directory not found: {input_dir}")

    count = 0

    for path in sorted(input_dir.rglob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        yield embedding_record_to_vector_record(payload)

        count += 1

        if limit is not None and count >= limit:
            break


def load_vector_records(input_dir: Path, limit: int | None = None) -> list[VectorRecord]:
    return list(iter_vector_records(input_dir=input_dir, limit=limit))


def _required_string(record: dict[str, Any], key: str) -> str:
    value = record.get(key)

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"embedding record {key} must be a non-empty string")

    return value