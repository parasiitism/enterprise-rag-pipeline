from pathlib import Path

from enterprise_rag.wikipedia.dump_reader import iter_wikipedia_pages


def test_iter_wikipedia_pages_reads_small_sample(tmp_path: Path) -> None:
    sample = tmp_path / "sample.xml.bz2"

    xml = """<mediawiki>
      <page>
        <title>Python</title>
        <id>1</id>
        <revision>
          <id>10</id>
          <timestamp>2026-01-01T00:00:00Z</timestamp>
          <text>Python is a programming language.</text>
        </revision>
      </page>
    </mediawiki>"""

    import bz2

    sample.write_bytes(bz2.compress(xml.encode("utf-8")))

    pages = list(iter_wikipedia_pages(sample))

    assert len(pages) == 1
    assert pages[0].title == "Python"
    assert pages[0].page_id == "1"
    assert pages[0].revision_id == "10"
    assert pages[0].text == "Python is a programming language."