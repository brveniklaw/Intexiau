import json
from pathlib import Path

import pytest

from bailey_b.evals import load_eval_set, recall_at_k, reciprocal_rank, run_eval
from bailey_b.models import Chunk, RetrievedChunk, parse_citation_id

EXAMPLE = Path(__file__).parent.parent / "evals" / "questions.example.jsonl"


def _rc(citation_id: str, rank: int) -> RetrievedChunk:
    chunk = Chunk(
        chunk_id=f"{citation_id}#0", citation_id=citation_id, display_citation="x", text="x"
    )
    return RetrievedChunk(chunk=chunk, rank=rank, score=1.0 / rank)


def test_parse_citation_id():
    assert parse_citation_id("fl-stat:83.56")[1] == "83.56"
    with pytest.raises(ValueError):
        parse_citation_id("Fla. Stat. 83.56")


def test_metrics():
    assert recall_at_k(["a", "b"], ["a", "c", "b"], k=2) == 0.5
    assert recall_at_k(["a", "b"], ["a", "c", "b"], k=3) == 1.0
    assert reciprocal_rank(["b"], ["a", "c", "b"]) == pytest.approx(1 / 3)
    assert reciprocal_rank(["z"], ["a"]) == 0.0


def test_example_eval_file_is_valid():
    assert load_eval_set(EXAMPLE)


def test_duplicate_ids_rejected(tmp_path):
    line = EXAMPLE.read_text().splitlines()[0]
    p = tmp_path / "dup.jsonl"
    p.write_text(line + "\n" + line + "\n")
    with pytest.raises(ValueError, match="duplicate"):
        load_eval_set(p)


def test_bad_citation_rejected(tmp_path):
    row = json.loads(EXAMPLE.read_text().splitlines()[0])
    row["expected_citations"] = ["83.56"]
    p = tmp_path / "bad.jsonl"
    p.write_text(json.dumps(row))
    with pytest.raises(ValueError):
        load_eval_set(p)


def test_run_eval_dedupes_citations():
    questions = load_eval_set(EXAMPLE)
    expected = questions[0].expected_citations[0]

    def fake_retrieve(_: str) -> list[RetrievedChunk]:
        # Two chunks of the same wrong opinion, then the right one.
        return [_rc("cl-opinion:1", 1), _rc("cl-opinion:1", 2), _rc(expected, 3)]

    report = run_eval(questions[:1], fake_retrieve, k=2)
    assert report.results[0].retrieved == ["cl-opinion:1", expected]
    assert report.mean_recall_at_k == 1.0
    assert report.mrr == pytest.approx(0.5)
