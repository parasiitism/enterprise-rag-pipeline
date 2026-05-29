from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from enterprise_rag.documents import CanonicalDocument


def load_canonical_document(path: Path) -> CanonicalDocument:
    payload = json.loads(path.read_text(encoding="utf-8"))

    return CanonicalDocument(
        document_id=payload["document_id"],
        source=payload["source"],
        source_id=payload["source_id"],
        title=payload["title"],
        version=payload["version"],
        updated_at=payload["updated_at"],
        text=payload["text"],
        metadata=payload["metadata"],
    )


def iter_canonical_documents(
    input_dir: Path,
    limit: int | None = None,
) -> Iterator[CanonicalDocument]:
    count = 0

    for path in sorted(input_dir.rglob("*.json")):
        yield load_canonical_document(path)
        count += 1

        if limit is not None and count >= limit:
            break