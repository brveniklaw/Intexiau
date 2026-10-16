"""Build step 5: the attorney-facing interface, starting with intake triage.

One page (``static/index.html``) and a small JSON API around the same pieces
every other consumer uses: ``Retriever.retrieve`` (step 3), ``verify`` (step 4),
and an optional ``Answerer`` that drafts from the retrieved sources.

There is no login yet. The server binds to localhost by default and must sit
behind BALS single sign-on before anyone else can reach it.
"""

from __future__ import annotations

import re
import threading
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from bailey_b.answer import Answerer, DisabledAnswerer
from bailey_b.feedback import Feedback, MemoryFeedbackStore
from bailey_b.verify import CorpusIndex, Status, verify

STATIC = Path(__file__).parent / "static"

_WORD_RE = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")
_STOPWORDS = set(
    "a an and are as at be but by can do does for from has have her his how i if in into is "
    "it its me my no not of on or our she so that the their them then there they this to was "
    "we what when where which who why will with would you your".split()
)


def excerpt(text: str, question: str, max_chars: int = 650) -> str:
    """The passage of a chunk that best matches the question, for display.

    Scores each line of the section body by overlap with the question's content
    words, then grows a window around the best line up to ``max_chars``.
    """
    body = text.split("\n\n", 1)[1] if "\n\n" in text else text
    lines = [line for line in body.splitlines() if line.strip()]
    if not lines:
        return ""
    words = {w for w in _WORD_RE.findall(question.lower()) if w not in _STOPWORDS}
    scores = [len(words & set(_WORD_RE.findall(line.lower()))) for line in lines]
    best = max(range(len(lines)), key=lambda i: (scores[i], -i))
    lo = hi = best
    size = len(lines[best])
    while size < max_chars and (lo > 0 or hi < len(lines) - 1):
        if hi < len(lines) - 1:
            hi += 1
            size += len(lines[hi])
        if size < max_chars and lo > 0:
            lo -= 1
            size += len(lines[lo])
    out = "\n".join(lines[lo : hi + 1])
    if len(out) > max_chars:
        out = out[:max_chars].rsplit(" ", 1)[0] + " …"
    return ("… " if lo > 0 else "") + out


_PART_RE = re.compile(r"^PART ([IVXLC]+)\s+(.+)$")


def _where(meta: dict) -> str:
    parts = []
    if meta.get("chapter"):
        parts.append(f"Ch. {meta['chapter']}")
    if meta.get("part"):
        m = _PART_RE.match(meta["part"])
        parts.append(f"Part {m[1]}, {m[2].title()}" if m else meta["part"].title())
    if meta.get("subsections"):
        parts.append(f"subsections {meta['subsections']}")
    return " · ".join(parts)


class TriageRequest(BaseModel):
    question: str = Field(min_length=3, max_length=4000)
    rerank: bool = True
    draft: bool = True


class FeedbackRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    chunk_id: str
    citation_id: str
    rank: int
    score: float
    useful: bool


def create_app(
    retriever,
    corpus: CorpusIndex,
    *,
    answerer: Answerer | None = None,
    feedback=None,
    warm_up: bool = False,
) -> FastAPI:
    app = FastAPI(title="Bailey B", docs_url=None, redoc_url=None)
    answerer = answerer or DisabledAnswerer()
    feedback = feedback or MemoryFeedbackStore()
    # One database connection and one model instance: serialize requests.
    lock = threading.Lock()
    status = {"ready": not warm_up}

    if warm_up:
        # Load the models in the background at startup so the first attorney's
        # search does not pay for it (minutes on CPU).
        def _warm() -> None:
            with lock:
                retriever.retrieve("warm up", rerank=True)
            status["ready"] = True

        threading.Thread(target=_warm, daemon=True).start()

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @app.get("/api/health")
    def health() -> dict:
        return {
            "sections": len(corpus),
            "chapters": sorted(corpus.chapters, key=lambda c: (len(c), c)),
            "edition": corpus.year,
            "draft_answers": not isinstance(answerer, DisabledAnswerer),
            "ready": status["ready"],
        }

    @app.post("/api/triage")
    def triage(req: TriageRequest) -> dict:
        started = time.monotonic()
        with lock:
            results = retriever.retrieve(req.question, rerank=req.rerank)
            draft = answerer.answer(req.question, results) if req.draft else None
        out_results = [
            {
                "rank": r.rank,
                "chunk_id": r.chunk.chunk_id,
                "citation_id": r.chunk.citation_id,
                "citation": r.chunk.display_citation,
                "title": r.chunk.title,
                "where": _where(r.chunk.metadata),
                "score": round(r.score, 3),
                "url": r.chunk.source_url,
                "excerpt": excerpt(r.chunk.text, req.question),
                # Retrieval only returns sections that exist in the corpus.
                "verified": corpus.has(r.chunk.citation_id),
            }
            for r in results
        ]
        answer = None
        if draft is not None:
            answer = {
                "status": draft.status,
                "text": draft.text,
                "model": draft.model,
                "detail": draft.detail,
                "checks": [],
                "flagged": 0,
            }
            if draft.status == "ok":
                report = verify(draft.text, corpus, sources=draft.sources)
                answer["checks"] = [
                    {
                        "raw": c.raw,
                        "status": c.status.value,
                        "message": c.message,
                        "url": c.url,
                        "flagged": c.status in (Status.NOT_FOUND, Status.SUBSECTION_MISSING)
                        or c.in_sources is False,
                    }
                    for c in report.checks
                ]
                answer["flagged"] = len(report.flagged)
        return {
            "question": req.question,
            "results": out_results,
            "answer": answer,
            "elapsed_s": round(time.monotonic() - started, 1),
        }

    @app.post("/api/feedback")
    def add_feedback(req: FeedbackRequest) -> dict:
        if not corpus.has(req.citation_id):
            raise HTTPException(400, "unknown citation")
        feedback.add(Feedback(**req.model_dump()))
        return {"ok": True, "total": feedback.count()}

    return app
