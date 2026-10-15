"""Statute parsing and chunking, against trimmed copies of real 2026 flsenate.gov pages."""

from pathlib import Path

import pytest

from bailey_b.corpus.statutes import (
    _join_label,
    chunk_section,
    parse_chapter,
    read_chunks,
    write_chunks,
)

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def ch83():
    return {
        s.number: s
        for s in parse_chapter((FIXTURES / "chapter_83_excerpt.html").read_bytes(), "83")
    }


@pytest.fixture(scope="module")
def s39301():
    (section,) = parse_chapter((FIXTURES / "chapter_39_excerpt.html").read_bytes(), "39")
    return section


def test_parse_section_fields(ch83):
    s = ch83["83.56"]
    assert s.citation_id == "fl-stat:83.56"
    assert s.display_citation == "Fla. Stat. § 83.56"
    assert s.catchline == "Termination of rental agreement."
    assert s.chapter_title == "CHAPTER 83 LANDLORD AND TENANT"
    assert s.part == "PART II RESIDENTIAL TENANCIES"
    assert "ch. 73-330" in s.history
    assert [b.label for b in s.blocks][:3] == ["(1)", "(2)", "(3)"]


def test_footnote_marker_dropped_and_note_kept(ch83):
    s = ch83["83.5615"]
    assert s.number == "83.5615"
    assert len(s.notes) == 1
    assert "Pub. L. No. 111-22" in s.notes[0]


def test_three_day_notice_text_survives(ch83):
    (chunk,) = chunk_section(ch83["83.56"], year=2026, max_words=10_000)
    assert (
        "default continues for 3 days, excluding Saturday, Sunday, and legal holidays" in chunk.text
    )
    assert chunk.text.startswith("Fla. Stat. § 83.56 Termination of rental agreement.\n")
    assert chunk.source_url == "https://www.flsenate.gov/Laws/Statutes/2026/83.56"


def test_note_goes_in_last_piece_only(ch83):
    chunks = chunk_section(ch83["83.5615"], year=2026, max_words=150)
    assert len(chunks) > 1
    assert "Note:" in chunks[-1].text
    assert all("Note:" not in c.text for c in chunks[:-1])


def test_join_label():
    assert _join_label("(1)", "(a)") == "(1)(a)"
    assert _join_label("(1)(a)", "1.") == "(1)(a)1."
    # Unnumbered subsection whose first paragraph is numbered "(1)(a)".
    assert _join_label("(1)", "(1)(a)") == "(1)(a)"


def test_nested_labels(s39301):
    first = s39301.blocks[0]
    assert first.label == "(1)"
    assert first.children[0].label == "(1)(a)"
    nine = next(b for b in s39301.blocks if b.label == "(9)")
    assert nine.children[0].children[6].label == "(9)(a)7."


def test_long_section_split_preserves_citation_and_text(s39301):
    chunks = chunk_section(s39301, year=2026, max_words=1200)
    assert len(chunks) > 1
    assert {c.citation_id for c in chunks} == {"fl-stat:39.301"}
    assert [c.chunk_id for c in chunks] == [f"fl-stat:39.301#{i}" for i in range(len(chunks))]
    assert all(len(c.text.split()) <= 1200 + 40 for c in chunks)  # header allowance
    assert chunks[0].metadata["piece"] == f"1/{len(chunks)}"
    assert chunks[0].metadata["subsections"].startswith("(1)")

    def body(text: str) -> list[str]:
        return text.split("\n\n", 1)[1].split()

    whole = body(chunk_section(s39301, year=2026, max_words=100_000)[0].text)
    pieces = [w for c in chunks for w in body(c.text)]
    # Splitting only adds repeated subsection labels; no statute text is lost.
    assert set(whole) <= set(pieces)
    assert len(pieces) >= len(whole)


def test_write_chunks_roundtrip_and_rejects_duplicates(ch83, tmp_path):
    chunks = chunk_section(ch83["83.56"], year=2026)
    path = tmp_path / "c.jsonl"
    assert write_chunks(chunks, path) == len(chunks)
    assert read_chunks(path) == chunks
    with pytest.raises(ValueError, match="duplicate"):
        write_chunks(chunks + chunks[:1], path)
