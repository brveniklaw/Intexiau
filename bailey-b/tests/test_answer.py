"""Draft answers: prompt shape and handling of the Claude response, with a fake client."""

from types import SimpleNamespace

from bailey_b.answer import ClaudeAnswerer, DisabledAnswerer, build_user_message
from bailey_b.config import AnswerSettings
from bailey_b.models import Chunk, RetrievedChunk


def result(section: str, text: str, rank: int = 1, **meta) -> RetrievedChunk:
    chunk = Chunk(
        chunk_id=f"fl-stat:{section}#0",
        citation_id=f"fl-stat:{section}",
        display_citation=f"Fla. Stat. § {section}",
        text=text,
        metadata=meta,
    )
    return RetrievedChunk(chunk=chunk, rank=rank, score=0.9)


class FakeMessages:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def fake_client(response):
    messages = FakeMessages(response)
    return SimpleNamespace(beta=SimpleNamespace(messages=messages)), messages


def message(text="", stop_reason="end_turn", stop_details=None):
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)] if text else [],
        stop_reason=stop_reason,
        stop_details=stop_details,
        model="claude-opus-5-5",
    )


def test_user_message_puts_tagged_sources_before_the_question():
    msg = build_user_message(
        "Does the weekend count?",
        [result("83.56", "Fla. Stat. § 83.56 ...\n\n(3) ... 3 days ...", subsections="(1)–(4)")],
    )
    assert msg.index("<sources>") < msg.index("<question>")
    assert 'citation="Fla. Stat. § 83.56" id="fl-stat:83.56" piece="(1)–(4)"' in msg
    assert msg.rstrip().endswith("</question>")


def test_request_shape_and_ok_answer():
    client, messages = fake_client(message("1. Pay rent into the registry. § 83.60(2)"))
    out = ClaudeAnswerer(AnswerSettings(), client=client).answer("q", [result("83.60", "t")])
    assert out.status == "ok"
    assert out.text.endswith("§ 83.60(2)")
    assert out.sources == ["fl-stat:83.60"]
    (call,) = messages.calls
    assert call["model"] == "claude-opus-5-5"
    assert call["output_config"] == {"effort": "high"}
    assert call["fallbacks"] == "default"
    assert call["betas"] == ["server-side-fallback-2026-07-01"]
    assert "temperature" not in call and "thinking" not in call


def test_refusal_is_reported_not_shown_as_an_answer():
    details = SimpleNamespace(category="general_harms", explanation="")
    client, _ = fake_client(message(stop_reason="refusal", stop_details=details))
    out = ClaudeAnswerer(AnswerSettings(), client=client).answer("q", [result("83.60", "t")])
    assert out.status == "refused"
    assert out.text == ""
    assert "general_harms" in out.detail


def test_cut_off_answer_is_marked():
    client, _ = fake_client(message("1. partial", stop_reason="max_tokens"))
    out = ClaudeAnswerer(AnswerSettings(), client=client).answer("q", [result("83.60", "t")])
    assert out.status == "ok" and "cut off" in out.detail


def test_no_sources_means_no_model_call():
    client, messages = fake_client(message("x"))
    out = ClaudeAnswerer(AnswerSettings(), client=client).answer("q", [])
    assert out.status == "error" and messages.calls == []


def test_disabled_by_default():
    assert AnswerSettings().enabled is False
    assert DisabledAnswerer().answer("q", []).status == "disabled"
