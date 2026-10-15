"""Citation verification against a small corpus built from real 2026 statute text."""

from pathlib import Path

import pytest

from bailey_b.corpus.statutes import chunk_section, parse_chapter
from bailey_b.verify import CorpusIndex, Status, _label_paths, verify

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def corpus():
    chunks = []
    for name, chapter in (("chapter_83_excerpt.html", "83"), ("chapter_39_excerpt.html", "39")):
        for section in parse_chapter((FIXTURES / name).read_bytes(), chapter):
            chunks += chunk_section(section, year=2026, max_words=1200)
    return CorpusIndex.from_chunks(chunks, year=2026)


def statuses(report):
    return [(c.raw, c.status) for c in report.checks]


def test_verified_section_and_subsection(corpus):
    r = verify("See Fla. Stat. § 83.56(3) and § 83.5615.", corpus)
    assert statuses(r) == [("83.56(3)", Status.VERIFIED), ("83.5615", Status.VERIFIED)]
    assert r.checks[0].url == "https://www.flsenate.gov/Laws/Statutes/2026/83.56"
    assert r.ok


def test_confabulated_section_is_flagged(corpus):
    r = verify("Retaliation is barred by § 83.66.", corpus)
    (c,) = r.checks
    assert c.status == Status.NOT_FOUND
    assert not r.ok
    assert "does not exist" in c.message


def test_missing_subsection_is_flagged(corpus):
    (c,) = verify("§ 83.56(9)", corpus).checks
    assert c.status == Status.SUBSECTION_MISSING
    (c,) = verify("§ 83.56(3)(z)", corpus).checks
    assert c.status == Status.SUBSECTION_MISSING
    assert "§ 83.56(3) does exist" in c.message


def test_deep_subsection_paths_verify(corpus):
    # (9)(a)7. sits three levels down in § 39.301, and (1) is unnumbered with a "(1)(a)" child.
    r = verify("s. 39.301(9)(a)7. and s. 39.301(1)(a)", corpus)
    assert [c.status for c in r.checks] == [Status.VERIFIED, Status.VERIFIED]


def test_unindexed_chapter_is_not_called_fake(corpus):
    (c,) = verify("Service is governed by § 48.031(1).", corpus).checks
    assert c.status == Status.OUTSIDE_CORPUS
    assert verify("§ 48.031", corpus).ok  # reported, not flagged


def test_citation_forms(corpus):
    text = (
        "ss. 83.56 and 83.5615; §§ 83.56, 39.301; section 83.56, Florida Statutes; "
        "83.5615, Fla. Stat.; F.S. 83.56(1)"
    )
    raws = [c.raw for c in verify(text, corpus).checks]
    assert raws == ["83.56", "83.5615", "83.56", "39.301", "83.56", "83.5615", "83.56(1)"]


def test_no_false_positives_on_ordinary_numbers(corpus):
    text = "Pay $1.50 per day; version 2.0; s. 2, ch. 2020-99; 3.5 percent; Section 4 of the lease."
    assert verify(text, corpus).checks == []


def test_case_law_rules_and_federal_are_unverifiable(corpus):
    text = (
        "Doe v. Roe, 123 So. 3d 456 (Fla. 2d DCA 2015) (placeholder citation); "
        "Fla. R. Civ. P. 1.140; 15 U.S.C. § 1692c(b)."
    )
    r = verify(text, corpus)
    assert [(c.kind, c.status) for c in r.checks] == [
        ("case", Status.UNVERIFIABLE),
        ("rule", Status.UNVERIFIABLE),
        ("federal-statute", Status.UNVERIFIABLE),
    ]
    assert r.ok  # unverifiable is reported for human checking, not flagged as fake


def test_citations_outside_retrieved_sources_are_flagged(corpus):
    r = verify("§ 83.56(3) and § 83.5615", corpus, sources=["fl-stat:83.56"])
    assert [c.in_sources for c in r.checks] == [True, False]
    assert [c.raw for c in r.flagged] == ["83.5615"]


def test_label_paths_hierarchy():
    text = "\n".join(
        ["(1) a", "(a) b", "1. c", "a. d", "(I) e", "(A) f", "(c)1. g", "2. h", "(2)(b) i"]
    )
    paths = _label_paths(text)
    assert {"(1)", "(1)(a)", "(1)(a)1.", "(1)(a)1.a.", "(1)(a)1.a.(I)", "(1)(a)1.a.(I)(A)"} <= paths
    assert {"(1)(c)1.", "(1)(c)2.", "(2)(b)"} <= paths
