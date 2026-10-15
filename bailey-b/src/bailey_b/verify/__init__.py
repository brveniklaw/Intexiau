"""Build step 4: citation verification.

Every citation in model-written text is checked against the corpus before the
text reaches anyone. Legal citation is a closed-world problem *inside the
corpus*: a Florida Statutes section in an indexed chapter either exists or it
does not. Outside that world, the verifier says it cannot check rather than
calling a citation fake:

==================  =====================================================
VERIFIED            Section (and subsection, if cited) found in the corpus.
SUBSECTION_MISSING  Section exists, but the cited subsection does not.
NOT_FOUND           Section is in an indexed chapter but does not exist:
                    the confabulation case.
OUTSIDE_CORPUS      Florida statute in a chapter not indexed yet; cannot check.
UNVERIFIABLE        Case law, court rules, federal statutes: not in the
                    corpus yet; cannot check.
==================  =====================================================

This catches invented and mistyped citations. It does not catch a real section
cited for a proposition it does not support; that needs a human reader.
When the retrieved sources are passed in, a verified citation that was not
among them is also flagged, since the model should cite only what it was given.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum

from bailey_b.models import Chunk, SourceKind

STATUTE_URL = "https://www.flsenate.gov/Laws/Statutes/{year}/{section}"

# A Florida Statutes section number: chapter, dot, section (61.13001, 83.5615, 744.3201).
_SECTION = r"\d{1,3}[A-Z]?\.\d{1,5}[A-Za-z]?"
# Optional subsection path right after it: (3), (1)(b), (1)(b)1., (2)(a)3.b.
_SUBSECTION = r"(?:\([0-9A-Za-z]{1,4}\))+(?:\d{1,2}\.(?=[a-z\s,;)]|$)(?:[a-z]\.(?=[\s,;)]|$))?)?"
_CITE = rf"(?P<sec>{_SECTION})(?P<sub>{_SUBSECTION})?"

# Anchors that introduce statute citations, possibly a list: "§§ 83.56 and 83.60".
_ANCHOR_RE = re.compile(
    r"(?:Fla\.\s*Stat\.\s*(?:Ann\.\s*)?§*|F\.\s?S\.\s*§*|§§?|\bss?\.|\b[Ss]ections?)\s*"
)
_LIST_ITEM_RE = re.compile(rf"\s*(?:,|and|or|&|through|to|[-–—])?\s*{_CITE}")
# "83.56, Florida Statutes" / "83.56, Fla. Stat." (no leading anchor).
_SUFFIX_RE = re.compile(
    rf"(?<![\d.]){_CITE},?\s*(?:Florida Statutes|Fla\.\s*Stat\.|F\.\s?S\.)(?:\s*\(\d{{4}}\))?"
)
_CASE_RE = re.compile(
    r"\b\d{1,4}\s+(?:So\.\s?(?:2d|3d)?|F\.\s?(?:2d|3d|4th)?|F\.\s?Supp\.(?:\s?[23]d)?|"
    r"F\.\s?App'x|U\.\s?S\.|S\.\s?Ct\.|L\.\s?Ed\.(?:\s?2d)?|Fla\.\s?L\.\s?Weekly(?:\s?(?:D|Supp\.))?)"
    r"\s+\d{1,5}(?:\s*\([^)]{2,40}\d{4}\))?"
)
_RULE_RE = re.compile(
    r"\bFla\.\s?R\.\s?(?:[A-Z][a-z]*\.\s?){1,5}(?:&\s?(?:[A-Z][a-z]*\.\s?){1,3})?\d+\.\d+"
)
_USC_RE = re.compile(r"\b\d{1,2}\s+U\.\s?S\.\s?C\.\s+(?:§+\s*)?\d+[a-z]?(?:\([0-9a-z]+\))*")

# Label at the start of a statute text line: "(3)", "(9)(a)", "(c)1.", "1.", "b.", "(II)".
_LINE_LABEL_RE = re.compile(
    r"^((?:\([0-9A-Za-z]{1,5}\))+(?:\d{1,2}\.(?!\d))?(?:[a-z]\.(?=\s))?|\d{1,2}\.(?!\d)|[a-z]\.(?=\s))"
)
_LABEL_PART_RE = re.compile(r"\([0-9A-Za-z]{1,5}\)|\d{1,2}\.|[a-z]\.")
_PAREN_RE = re.compile(r"\([0-9A-Za-z]{1,5}\)")
_ROMAN_RE = re.compile(r"^\((?:[IVX]+)\)$")


def _level(part: str) -> int:
    """Florida statute hierarchy: (1) → (a) → 1. → a. → (I) → (A)."""
    if part.startswith("("):
        inner = part[1:-1]
        if inner.isdigit():
            return 0
        if inner.islower():
            return 1
        if _ROMAN_RE.match(part):
            return 4
        return 5
    return 2 if part[0].isdigit() else 3


class Status(StrEnum):
    VERIFIED = "verified"
    SUBSECTION_MISSING = "subsection_missing"
    NOT_FOUND = "not_found"
    OUTSIDE_CORPUS = "outside_corpus"
    UNVERIFIABLE = "unverifiable"


# Statuses that should stop text from reaching a user unreviewed.
FLAGGED = {Status.SUBSECTION_MISSING, Status.NOT_FOUND}


@dataclass
class CitationCheck:
    raw: str
    start: int
    end: int
    status: Status
    message: str
    kind: str = "fl-statute"  # fl-statute | case | rule | federal-statute
    section: str | None = None
    subsection: str | None = None
    citation_id: str | None = None
    url: str | None = None
    # None when no sources were given; False = cited but not among the retrieved sources.
    in_sources: bool | None = None


@dataclass
class VerificationReport:
    checks: list[CitationCheck] = field(default_factory=list)

    @property
    def flagged(self) -> list[CitationCheck]:
        return [c for c in self.checks if c.status in FLAGGED or c.in_sources is False]

    @property
    def ok(self) -> bool:
        """True when nothing is flagged. Unverifiable citations are reported, not flagged."""
        return not self.flagged

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for c in self.checks:
            out[c.status.value] = out.get(c.status.value, 0) + 1
        return out


def _label_paths(text: str) -> set[str]:
    """Citable subsection paths in a section's text, e.g. {"(1)", "(1)(a)", "(1)(a)1."}.

    Statute text has one subsection, paragraph or subparagraph per line (see
    corpus.statutes). A skipped level contributes nothing to the path, matching
    how the statutes cite it.
    """
    paths: set[str] = set()
    stack: list[str] = [""] * 6
    for line in text.splitlines():
        m = _LINE_LABEL_RE.match(line.strip())
        if not m:
            continue
        for part in _LABEL_PART_RE.findall(m.group(1)):
            level = _level(part)
            stack[level] = part
            for deeper in range(level + 1, 6):
                stack[deeper] = ""
            paths.add("".join(stack[: level + 1]))
    return paths


class CorpusIndex:
    """The closed world: which sections exist and which subsections each one has."""

    def __init__(self, section_texts: dict[str, str], *, year: int):
        self.year = year
        self._texts = section_texts
        self._paths: dict[str, set[str]] = {}
        self.chapters = {cid.split(":", 1)[1].split(".")[0] for cid in section_texts}

    @classmethod
    def from_chunks(cls, chunks: Iterable[Chunk], *, year: int) -> CorpusIndex:
        pieces: dict[str, list[tuple[int, str]]] = {}
        for c in chunks:
            if not c.citation_id.startswith(f"{SourceKind.FL_STATUTE.value}:"):
                continue
            n = int(c.chunk_id.rsplit("#", 1)[1]) if "#" in c.chunk_id else 0
            pieces.setdefault(c.citation_id, []).append((n, c.text))
        texts = {cid: "\n".join(t for _, t in sorted(ps)) for cid, ps in pieces.items()}
        return cls(texts, year=year)

    def has(self, citation_id: str) -> bool:
        return citation_id in self._texts

    def paths(self, citation_id: str) -> set[str]:
        if citation_id not in self._paths:
            self._paths[citation_id] = _label_paths(self._texts.get(citation_id, ""))
        return self._paths[citation_id]


def _statute_spans(text: str) -> list[tuple[int, int, str, str]]:
    """(start, end, section, subsection) for every statute citation in ``text``."""
    found: list[tuple[int, int, str, str]] = []
    for anchor in _ANCHOR_RE.finditer(text):
        pos = anchor.end()
        first = True
        while True:
            m = _LIST_ITEM_RE.match(text, pos) if not first else re.compile(_CITE).match(text, pos)
            if not m:
                break
            found.append((m.start("sec"), m.end(), m.group("sec"), m.group("sub") or ""))
            pos, first = m.end(), False
    for m in _SUFFIX_RE.finditer(text):
        found.append(
            (
                m.start("sec"),
                m.end("sub") if m.group("sub") else m.end("sec"),
                m.group("sec"),
                m.group("sub") or "",
            )
        )
    # De-duplicate citations matched by both an anchor and a suffix.
    unique: dict[int, tuple[int, int, str, str]] = {}
    for item in found:
        unique.setdefault(item[0], item)
    return sorted(unique.values())


def verify(
    text: str,
    corpus: CorpusIndex,
    *,
    sources: Iterable[str] | None = None,
) -> VerificationReport:
    """Check every citation in ``text``. ``sources`` = citation IDs the model was given."""
    source_ids = set(sources) if sources is not None else None
    report = VerificationReport()

    for start, end, section, sub in _statute_spans(text):
        cid = f"{SourceKind.FL_STATUTE.value}:{section}"
        chapter = section.split(".")[0]
        check = CitationCheck(
            raw=text[start:end],
            start=start,
            end=end,
            status=Status.VERIFIED,
            message="",
            section=section,
            subsection=sub or None,
            citation_id=cid,
        )
        if chapter not in corpus.chapters:
            check.status = Status.OUTSIDE_CORPUS
            check.message = f"Chapter {chapter} is not indexed yet; § {section} was not checked."
        elif not corpus.has(cid):
            check.status = Status.NOT_FOUND
            check.message = f"§ {section} does not exist in the {corpus.year} Florida Statutes."
        else:
            check.url = STATUTE_URL.format(year=corpus.year, section=section)
            paths = corpus.paths(cid)
            if sub and sub not in paths:
                top = _PAREN_RE.match(sub)
                has_top = bool(top) and top.group() in paths
                check.status = Status.SUBSECTION_MISSING
                check.message = f"§ {section} exists, but subsection {sub} was not found" + (
                    f" (§ {section}{top.group()} does exist)." if has_top else "."
                )
            else:
                check.message = f"§ {section}{sub} found in the {corpus.year} Florida Statutes."
            if source_ids is not None:
                check.in_sources = cid in source_ids
                if not check.in_sources:
                    check.message += " Not among the sources retrieved for this answer."
        report.checks.append(check)

    taken = [(c.start, c.end) for c in report.checks]
    for regex, kind, what in (
        (_CASE_RE, "case", "Case law is not in the corpus yet"),
        (_RULE_RE, "rule", "Court rules are not in the corpus"),
        (_USC_RE, "federal-statute", "Federal statutes are not in the corpus"),
    ):
        for m in regex.finditer(text):
            if any(s < m.end() and m.start() < e for s, e in taken):
                continue
            report.checks.append(
                CitationCheck(
                    raw=m.group().strip(),
                    start=m.start(),
                    end=m.end(),
                    status=Status.UNVERIFIABLE,
                    kind=kind,
                    message=f"{what}; this citation could not be checked. Verify it by hand.",
                )
            )
            taken.append((m.start(), m.end()))

    report.checks.sort(key=lambda c: c.start)
    return report
