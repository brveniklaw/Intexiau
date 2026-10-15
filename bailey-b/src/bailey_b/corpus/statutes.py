"""Florida Statutes: download from flsenate.gov and chunk one section per chunk.

Source: ``https://www.flsenate.gov/Laws/Statutes/<year>/Chapter<N>/All``, one
page per chapter. (The build doc names the Online Sunshine zip; its host,
leg.state.fl.us, does not resolve from our environment, and the Senate site
publishes the same official text with clean per-section markup.)

Each ``div.Section`` becomes one chunk. Sections longer than ``max_words`` are
split at subsection boundaries into several chunks that share the section's
citation ID, so verification still sees one citation. Notes stay in the text
because they can change what a section means (e.g. s. 83.5615 only takes
effect if a federal act is repealed); legislative history goes to metadata.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

from bs4 import BeautifulSoup, NavigableString, Tag

from bailey_b.config import CorpusSettings
from bailey_b.corpus.download import fetch_all, load_manifest
from bailey_b.models import Chunk, SourceKind

BASE_URL = "https://www.flsenate.gov/Laws/Statutes"
# Default split threshold. bge-m3 accepts ~8k tokens, but retrieval is sharper on
# smaller units; ~1,200 words keeps most sections whole (about 7% get split).
MAX_WORDS = 1200

_SECTION_NUMBER_RE = re.compile(r"\d+[A-Z]?\.\d+[A-Za-z]?")
# Block-level elements inside a section body, outermost first.
_BLOCK_CLASSES = ("Subsection", "Paragraph", "SubParagraph", "SubSubParagraph")
_WS_RE = re.compile(r"[ \t   ]+")


def chapter_url(year: int, chapter: str) -> str:
    return f"{BASE_URL}/{year}/Chapter{chapter}/All"


def section_url(year: int, section: str) -> str:
    return f"{BASE_URL}/{year}/{section}"


def raw_dir(settings: CorpusSettings) -> Path:
    return settings.data_dir / "raw" / "statutes" / str(settings.statutes_year)


def chunks_path(settings: CorpusSettings) -> Path:
    return settings.data_dir / "chunks" / f"fl-statutes-{settings.statutes_year}.jsonl"


def download(settings: CorpusSettings, *, force: bool = False) -> dict[str, dict]:
    urls = {
        f"chapter_{ch}.html": chapter_url(settings.statutes_year, ch)
        for ch in settings.statute_chapters
    }
    return fetch_all(urls, raw_dir(settings), force=force)


# --- Parsing -------------------------------------------------------------------


@dataclass
class Block:
    """A piece of a section body: intro text, a subsection, a paragraph, a table.

    ``label`` is the full citable path, e.g. ``(1)(b)1.``. ``lead`` is the block's
    own text before its children; ``text`` includes the children.
    """

    label: str
    text: str
    lead: str = ""
    children: list[Block] = field(default_factory=list)

    @property
    def words(self) -> int:
        return len(self.text.split())


@dataclass
class StatuteSection:
    number: str
    catchline: str
    chapter: str
    chapter_title: str
    part: str  # "PART II RESIDENTIAL TENANCIES", or "" for chapters without parts
    blocks: list[Block]
    notes: list[str]
    history: str

    @property
    def citation_id(self) -> str:
        return f"{SourceKind.FL_STATUTE.value}:{self.number}"

    @property
    def display_citation(self) -> str:
        return f"Fla. Stat. § {self.number}"


def _clean(text: str) -> str:
    lines = (_WS_RE.sub(" ", line).strip() for line in text.splitlines())
    return "\n".join(line for line in lines if line)


def _inline_text(el: Tag) -> str:
    """Text of an element with fractions as a/b and footnote markers dropped."""
    parts: list[str] = []
    for node in el.descendants:
        if isinstance(node, NavigableString):
            parent = node.parent
            if parent is not None and parent.find_parent("sup") is not None:
                continue
            if parent is not None and parent.name == "sup":
                continue
            if parent is not None and "Denominator" in (parent.get("class") or []):
                parts.append("/")
            parts.append(str(node))
    return _WS_RE.sub(" ", "".join(parts)).strip()


def _table_text(table: Tag) -> str:
    rows = []
    for tr in table.select("tr"):
        cells = [_inline_text(c) for c in tr.find_all(["td", "th"])]
        if any(cells):
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def _has_class(el: Tag, *names: str) -> bool:
    return any(c in names for c in (el.get("class") or []))


def _is_block(el: object) -> bool:
    return isinstance(el, Tag) and el.name == "div" and _has_class(el, *_BLOCK_CLASSES)


def _render(el: Tag, *, skip_blocks: bool = False) -> str:
    """Render an element as text, one line per subsection/paragraph/table row."""
    lines: list[str] = []
    buf: list[str] = []

    def flush() -> None:
        if buf:
            lines.append(_WS_RE.sub(" ", " ".join(buf)).strip())
            buf.clear()

    for child in el.children:
        if isinstance(child, NavigableString):
            if child.strip():
                buf.append(str(child))
        elif not isinstance(child, Tag) or child.name == "sup":
            continue
        elif skip_blocks and _is_block(child):
            continue
        elif child.name == "table":
            flush()
            lines.append(_table_text(child))
        elif child.name == "div" or child.find(["div", "table"]) is not None:
            flush()
            lines.append(_render(child))
        else:
            buf.append(_inline_text(child))
    flush()
    return _clean("\n".join(lines))


# Last component of a label: "(b)" in "(1)(b)", "1." in "(1)(b)1.".
_LAST_LABEL_RE = re.compile(r"(\([^()]+\)|[0-9a-z]+\.)$")
_FIRST_LABEL_RE = re.compile(r"^\([^()]+\)")


def _join_label(parent: str, own: str) -> str:
    """Append a block's number to its parent's label without repeating a shared prefix.

    Some subsections carry no number and their first paragraph is numbered
    "(1)(a)"; that paragraph's label must come out as "(1)(a)", not "(1)(1)(a)".
    """
    last = _LAST_LABEL_RE.search(parent)
    if last and own.startswith(last.group()):
        return parent[: last.start()] + own
    return parent + own


def _block(el: Tag, parent_label: str) -> Block:
    number = el.find("span", class_="Number", recursive=False)
    own = _inline_text(number) if number else ""
    child_divs = [c for c in el.find_all("div", recursive=False) if _is_block(c)]
    if not own and child_divs:
        first = child_divs[0].find("span", class_="Number", recursive=False)
        m = _FIRST_LABEL_RE.match(_inline_text(first)) if first else None
        own = m.group() if m else ""
    label = _join_label(parent_label, own)
    return Block(
        label=label,
        text=_render(el),
        lead=_render(el, skip_blocks=True),
        children=[_block(c, label) for c in child_divs],
    )


def _blocks(body: Tag) -> list[Block]:
    """Top-level blocks of a section body; loose text between subsections becomes its own block."""
    blocks: list[Block] = []
    loose: list[str] = []

    def flush() -> None:
        if text := _clean("\n".join(loose)):
            blocks.append(Block("", text, text))
        loose.clear()

    for child in body.children:
        if _is_block(child):
            flush()
            blocks.append(_block(child, ""))
        elif isinstance(child, NavigableString):
            loose.append(str(child))
        elif isinstance(child, Tag) and child.name != "sup":
            if child.name == "table":
                loose.append(_table_text(child))
            elif child.name == "div":
                loose.append(_render(child))
            else:
                loose.append(_inline_text(child))
    flush()
    return [b for b in blocks if b.text]


def _title(el: Tag | None) -> str:
    return _WS_RE.sub(" ", el.get_text(" ")).strip() if el else ""


def parse_chapter(html: str | bytes, chapter: str) -> list[StatuteSection]:
    soup = BeautifulSoup(html, "lxml")
    title_el = soup.select_one("div.ChapterTitle")
    chapter_title = _title(title_el) or f"CHAPTER {chapter}"
    sections: list[StatuteSection] = []
    for sec in soup.select("div.Section"):
        num_el = sec.select_one(".SectionNumber")
        m = _SECTION_NUMBER_RE.search(num_el.get_text() if num_el else "")
        if not m:
            raise ValueError(f"chapter {chapter}: section without a number: {str(sec)[:200]}")
        part_el = sec.find_parent("div", class_="Part")
        part_title = part_el.find("div", class_="PartTitle") if part_el else None
        catch = sec.select_one(".CatchlineText")
        body = sec.find("span", class_="SectionBody")
        history = sec.find("div", class_="History")
        history_text = history.find("span", class_="HistoryText") if history else None
        notes = []
        for note in sec.find_all("div", class_="Note", recursive=False):
            text = note.find("span", class_="Text")
            notes.append(_inline_text(text) if text else _inline_text(note))
        sections.append(
            StatuteSection(
                number=m.group(),
                catchline=_inline_text(catch) if catch else "",
                chapter=chapter,
                chapter_title=chapter_title,
                part=_title(part_title),
                blocks=_blocks(body) if body else [],
                notes=[n for n in notes if n],
                history=_inline_text(history_text) if history_text else "",
            )
        )
    return sections


# --- Chunking ------------------------------------------------------------------


def _header(section: StatuteSection) -> str:
    where = section.chapter_title
    if section.part:
        where += f", {section.part}"
    return f"{section.display_citation} {section.catchline}\n{where}"


def _split_leaf(block: Block, max_words: int) -> list[Block]:
    """Split an oversized block with no sub-blocks (a long form or table) by line."""
    pieces: list[Block] = []
    lines: list[str] = []
    size = 0
    for line in block.text.splitlines():
        n = len(line.split())
        if lines and size + n > max_words:
            pieces.append(Block(block.label, "\n".join(lines)))
            lines, size = [], 0
        lines.append(line)
        size += n
    if lines:
        pieces.append(Block(block.label, "\n".join(lines)))
    return pieces


def _pack(blocks: list[Block], max_words: int) -> list[list[Block]]:
    """Group consecutive blocks into runs under max_words, descending into oversized ones."""
    groups: list[list[Block]] = []
    current: list[Block] = []
    size = 0

    def close() -> None:
        nonlocal current, size
        if current:
            groups.append(current)
        current, size = [], 0

    for block in blocks:
        if block.words > max_words:
            close()
            if block.children:
                # The block's own opening text (which can hold a whole table), then its children.
                lead = _pack([Block(block.label, block.lead)], max_words) if block.lead else []
                inner = _pack(block.children, max_words)
                # A short lead rides along with the first group of children.
                first_size = sum(b.words for b in inner[0])
                if lead and len(lead[-1]) == 1 and lead[-1][0].words + first_size <= max_words:
                    inner[0].insert(0, lead.pop()[0])
                groups.extend(lead + inner)
            else:
                groups.extend([p] for p in _split_leaf(block, max_words))
            continue
        if current and size + block.words > max_words:
            close()
        current.append(block)
        size += block.words
    close()
    return groups


def _label_range(group: list[Block]) -> str:
    labels = [b.label for b in group if b.label]
    if not labels:
        return ""
    return labels[0] if len(labels) == 1 else f"{labels[0]}–{labels[-1]}"


def chunk_section(
    section: StatuteSection,
    *,
    year: int,
    source_url: str | None = None,
    retrieved_at: date | None = None,
    max_words: int = MAX_WORDS,
) -> list[Chunk]:
    header = _header(section)
    notes = "\n".join(f"Note: {n}" for n in section.notes)
    groups = _pack(section.blocks, max_words) or [[]]
    chunks = []
    for i, group in enumerate(groups):
        body = "\n".join(b.text for b in group)
        parts = [header, body]
        if notes and i == len(groups) - 1:
            parts.append(notes)
        metadata = {
            "chapter": section.chapter,
            "part": section.part,
            "edition": str(year),
            "history": section.history,
        }
        if len(groups) > 1:
            metadata["piece"] = f"{i + 1}/{len(groups)}"
            if span := _label_range(group):
                metadata["subsections"] = span
        chunks.append(
            Chunk(
                chunk_id=f"{section.citation_id}#{i}",
                citation_id=section.citation_id,
                display_citation=section.display_citation,
                title=section.catchline,
                text="\n\n".join(p for p in parts if p),
                source_url=source_url or section_url(year, section.number),
                retrieved_at=retrieved_at,
                metadata=metadata,
            )
        )
    return chunks


def iter_chunks(settings: CorpusSettings, *, max_words: int = MAX_WORDS) -> Iterator[Chunk]:
    directory = raw_dir(settings)
    manifest = load_manifest(directory)
    for chapter in settings.statute_chapters:
        name = f"chapter_{chapter}.html"
        path = directory / name
        if not path.exists():
            raise FileNotFoundError(f"{path} missing; run the download first")
        info = manifest.get(name, {})
        retrieved = (
            datetime.fromisoformat(info["retrieved_at"]).date() if "retrieved_at" in info else None
        )
        for section in parse_chapter(path.read_bytes(), chapter):
            yield from chunk_section(
                section,
                year=settings.statutes_year,
                retrieved_at=retrieved,
                max_words=max_words,
            )


def write_chunks(chunks: Iterable[Chunk], path: Path) -> int:
    """Write chunks as JSONL; raises on duplicate chunk IDs. Returns the count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    tmp = path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            if chunk.chunk_id in seen:
                raise ValueError(f"duplicate chunk id {chunk.chunk_id}")
            seen.add(chunk.chunk_id)
            f.write(chunk.model_dump_json() + "\n")
    tmp.replace(path)
    return len(seen)


def read_chunks(path: Path) -> list[Chunk]:
    with path.open(encoding="utf-8") as f:
        return [Chunk.model_validate(json.loads(line)) for line in f if line.strip()]
