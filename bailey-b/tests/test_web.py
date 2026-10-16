"""The web API with stand-in retrieval and model; corpus built from real statute text."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from bailey_b.answer import DraftAnswer
from bailey_b.corpus.statutes import chunk_section, parse_chapter
from bailey_b.feedback import MemoryFeedbackStore
from bailey_b.models import RetrievedChunk
from bailey_b.verify import CorpusIndex
from bailey_b.web import create_app, excerpt

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def chunks():
    out = []
    for section in parse_chapter((FIXTURES / "chapter_83_excerpt.html").read_bytes(), "83"):
        out += chunk_section(section, year=2026)
    return {c.chunk_id: c for c in out}


class FakeRetriever:
    def __init__(self, chunks):
        self.chunks = chunks
        self.calls = []

    def retrieve(self, question, *, rerank=True):
        self.calls.append((question, rerank))
        c = self.chunks["fl-stat:83.56#0"]
        return [RetrievedChunk(chunk=c, rank=1, score=0.91)]


class FakeAnswerer:
    def __init__(self, text):
        self.text = text

    def answer(self, question, results):
        return DraftAnswer(
            status="ok",
            text=self.text,
            model="fake",
            sources=[r.chunk.citation_id for r in results],
        )


@pytest.fixture
def make_client(chunks):
    corpus = CorpusIndex.from_chunks(chunks.values(), year=2026)

    def make(answer_text=None):
        fb = MemoryFeedbackStore()
        answerer = FakeAnswerer(answer_text) if answer_text else None
        make.retriever = FakeRetriever(chunks)
        app = create_app(make.retriever, corpus, answerer=answerer, feedback=fb)
        return TestClient(app), fb

    return make


def test_page_and_health(make_client):
    client, _ = make_client()
    assert "Bailey B" in client.get("/").text
    h = client.get("/api/health").json()
    assert h == {
        "sections": 2,
        "chapters": ["83"],
        "edition": 2026,
        "draft_answers": False,
        "ready": True,
    }


def test_triage_returns_ranked_sources_without_a_model(make_client):
    client, _ = make_client()
    data = client.post(
        "/api/triage", json={"question": "Does the weekend count for a 3-day notice?"}
    ).json()
    (r,) = data["results"]
    assert r["citation"] == "Fla. Stat. § 83.56"
    assert r["verified"] is True
    assert r["url"] == "https://www.flsenate.gov/Laws/Statutes/2026/83.56"
    assert "excluding Saturday, Sunday" in r["excerpt"]
    assert r["where"].startswith("Ch. 83 · Part II, Residential Tenancies")
    assert data["answer"]["status"] == "disabled"


def test_draft_answer_citations_are_verified_against_sources(make_client):
    text = "1. Weekends are excluded. § 83.56(3)\n2. See also § 83.5615.\n3. And § 83.66."
    client, _ = make_client(text)
    a = client.post("/api/triage", json={"question": "weekend?"}).json()["answer"]
    assert a["status"] == "ok"
    by_raw = {c["raw"]: c for c in a["checks"]}
    assert by_raw["83.56(3)"]["flagged"] is False
    assert by_raw["83.5615"]["flagged"] is True  # real section, but not a retrieved source
    assert by_raw["83.66"]["status"] == "not_found"
    assert a["flagged"] == 2


def test_rerank_flag_is_passed_through(make_client):
    client, _ = make_client()
    client.post("/api/triage", json={"question": "abc", "rerank": False})
    assert make_client.retriever.calls == [("abc", False)]


def test_feedback_is_stored_and_validated(make_client):
    client, fb = make_client()
    body = {
        "question": "q",
        "chunk_id": "fl-stat:83.56#0",
        "citation_id": "fl-stat:83.56",
        "rank": 1,
        "score": 0.9,
        "useful": True,
    }
    assert client.post("/api/feedback", json=body).json() == {"ok": True, "total": 1}
    assert fb.items[0].useful is True
    bad = dict(body, citation_id="fl-stat:99.99")
    assert client.post("/api/feedback", json=bad).status_code == 400


def test_question_validation(make_client):
    client, _ = make_client()
    assert client.post("/api/triage", json={"question": ""}).status_code == 422


def test_excerpt_picks_the_matching_passage(chunks):
    text = chunks["fl-stat:83.56#0"].text
    out = excerpt(text, "3 days excluding Saturday Sunday legal holidays rent", max_chars=300)
    assert "3 days, excluding Saturday, Sunday, and legal holidays" in out
    assert len(out) <= 310


def test_warm_up_runs_one_search_and_reports_ready(chunks):
    import time

    corpus = CorpusIndex.from_chunks(chunks.values(), year=2026)
    retriever = FakeRetriever(chunks)
    client = TestClient(create_app(retriever, corpus, warm_up=True))
    for _ in range(50):
        if client.get("/api/health").json()["ready"]:
            break
        time.sleep(0.02)
    assert client.get("/api/health").json()["ready"] is True
    assert retriever.calls == [("warm up", True)]
