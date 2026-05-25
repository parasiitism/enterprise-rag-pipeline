from enterprise_rag.wikipedia.converter import wikipedia_page_to_document
from enterprise_rag.wikipedia.dump_reader import WikipediaPage


def test_wikipedia_page_to_document_maps_fields() -> None:
    page = WikipediaPage(
        title="AccessibleComputing",
        page_id="10",
        revision_id="1219062925",
        timestamp="2024-04-15T14:38:04Z",
        text="#REDIRECT [[Computer accessibility]]",
    )

    document = wikipedia_page_to_document(page)

    assert document.document_id == "wikipedia_10"
    assert document.source == "wikipedia"
    assert document.source_id == "10"
    assert document.title == "AccessibleComputing"
    assert document.version == "1219062925"
    assert document.updated_at == "2024-04-15T14:38:04Z"
    assert document.text == "#REDIRECT [[Computer accessibility]]"
    assert document.metadata == {
        "page_id": "10",
        "revision_id": "1219062925",
    }
