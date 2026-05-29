from langchain_core.documents import Document

from enterprise_rag.documents import CanonicalDocument
from enterprise_rag.langchain_adapters import canonical_document_to_langchain


def test_canonical_document_to_langchain_preserves_text_and_metadata() -> None:
    canonical_document = CanonicalDocument(
        document_id="wikipedia_10",
        source="wikipedia",
        source_id="10",
        title="AccessibleComputing",
        version="1219062925",
        updated_at="2024-04-15T14:38:04Z",
        text="#REDIRECT [[Computer accessibility]]",
        metadata={
            "page_id": "10",
            "revision_id": "1219062925",
        },
    )

    langchain_document = canonical_document_to_langchain(canonical_document)

    assert isinstance(langchain_document, Document)
    assert langchain_document.page_content == "#REDIRECT [[Computer accessibility]]"
    assert langchain_document.metadata == {
        "document_id": "wikipedia_10",
        "source": "wikipedia",
        "source_id": "10",
        "title": "AccessibleComputing",
        "version": "1219062925",
        "updated_at": "2024-04-15T14:38:04Z",
        "page_id": "10",
        "revision_id": "1219062925",
    }
