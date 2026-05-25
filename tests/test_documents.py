from pathlib import Path

from enterprise_rag.documents import CanonicalDocument


def test_canonical_document_can_save_json(tmp_path: Path) -> None:
    document = CanonicalDocument(
        document_id="wikipedia_10",
        source="wikipedia",
        source_id="10",
        title="AccessibleComputing",
        version="1219062925",
        updated_at="2024-04-15T14:38:04Z",
        text="Example text",
        metadata={
            "page_id": "10",
            "revision_id": "1219062925",
        },
    )

    output_path = document.save_json(tmp_path)

    assert output_path.exists()
    assert output_path.name == "wikipedia_10.json"
    assert '"title": "AccessibleComputing"' in output_path.read_text(encoding="utf-8")