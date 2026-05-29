from pathlib import Path

from enterprise_rag.chunk_documents import chunk_document_directory
from enterprise_rag.documents import CanonicalDocument


def test_chunk_document_directory_saves_chunks_and_manifest(tmp_path: Path) -> None:
    input_dir = tmp_path / "canonical"
    output_dir = tmp_path / "chunks"
    manifest_path = tmp_path / "manifests" / "chunk_manifest.jsonl"

    document = CanonicalDocument(
        document_id="wikipedia_1",
        source="wikipedia",
        source_id="1",
        title="Test Page",
        version="100",
        updated_at="2026-01-01T00:00:00Z",
        text=" ".join(f"word{i}" for i in range(500)),
        metadata={"page_id": "1", "revision_id": "100"},
    )

    document.save_json(input_dir)

    summary = chunk_document_directory(
        input_dir=input_dir,
        output_dir=output_dir,
        manifest_path=manifest_path,
        limit=1,
        chunk_size=300,
        chunk_overlap=50,
    )

    assert summary.documents_processed == 1
    assert summary.chunks_saved > 1
    assert len(list(output_dir.glob("*.json"))) == summary.chunks_saved
    assert manifest_path.exists()
    assert "wikipedia_1_chunk_0" in manifest_path.read_text(encoding="utf-8")