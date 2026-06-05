from __future__ import annotations

import argparse
from collections.abc import Iterable, Iterator
from pathlib import Path

from enterprise_rag.chunk_store import iter_chunk_records
from enterprise_rag.embedding_store import (
    append_embedding_manifest_record,
    embedding_result_to_record,
    save_embedding_record,
)
from enterprise_rag.embeddings import (
    DEFAULT_HUGGINGFACE_EMBEDDING_DIMENSIONS,
    DEFAULT_HUGGINGFACE_EMBEDDING_MODEL,
    EmbeddingProvider,
    EmbeddingRequest,
    HuggingFaceEmbeddingProvider,
)


def chunk_record_to_embedding_request(record: dict[str, object]) -> EmbeddingRequest:
    metadata = dict(record["metadata"])

    metadata.update(
        {
            "chunk_id": record["chunk_id"],
            "document_id": record["document_id"],
            "source": record["source"],
            "source_id": record["source_id"],
            "title": record["title"],
            "chunk_index": record["chunk_index"],
        }
    )

    return EmbeddingRequest(
        record_id=str(record["chunk_id"]),
        text=str(record["text"]),
        metadata=metadata,
    )


def iter_batches(
    items: Iterable[dict[str, object]],
    batch_size: int,
) -> Iterator[list[dict[str, object]]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0")

    batch: list[dict[str, object]] = []

    for item in items:
        batch.append(item)

        if len(batch) >= batch_size:
            yield batch
            batch = []

    if batch:
        yield batch


def embed_chunk_records(
    records: Iterable[dict[str, object]],
    provider: EmbeddingProvider,
    output_dir: Path,
    manifest_path: Path,
    batch_size: int,
) -> int:
    total = 0

    for batch in iter_batches(records, batch_size=batch_size):
        requests = [chunk_record_to_embedding_request(record) for record in batch]
        results = provider.embed_texts(requests)

        for result in results:
            embedding_record = embedding_result_to_record(result)
            output_path = save_embedding_record(embedding_record, output_dir)
            append_embedding_manifest_record(
                record=embedding_record,
                output_path=output_path,
                manifest_path=manifest_path,
            )
            total += 1

        print(f"Embedded {total} chunks")

    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed chunk JSON files")

    parser.add_argument("chunks_dir", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/embeddings/wikipedia"),
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=Path("data/manifests/embedding_manifest.jsonl"),
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--model-name", type=str, default=DEFAULT_HUGGINGFACE_EMBEDDING_MODEL)
    parser.add_argument("--dimensions", type=int, default=DEFAULT_HUGGINGFACE_EMBEDDING_DIMENSIONS)
    parser.add_argument("--device", type=str, default=None)

    args = parser.parse_args()

    provider = HuggingFaceEmbeddingProvider(
        model_name=args.model_name,
        dimensions=args.dimensions,
        max_batch_size=args.batch_size,
        device=args.device,
    )

    records = iter_chunk_records(args.chunks_dir, limit=args.limit)

    total = embed_chunk_records(
        records=records,
        provider=provider,
        output_dir=args.output_dir,
        manifest_path=args.manifest_path,
        batch_size=args.batch_size,
    )

    print(f"Done. Embedded {total} chunks.")


if __name__ == "__main__":
    main()
