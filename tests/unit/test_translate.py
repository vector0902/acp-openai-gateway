"""Unit tests for the OpenAI payload-shaping helpers (pure, no I/O)."""

import pytest

from acp_openai_gateway import translate
from acp_openai_gateway.sessions import content_text

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "content, expected",
    [
        ("hello", "hello"),
        ([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}], "ab"),
        ([{"type": "image_url", "image_url": {"url": "x"}}], ""),  # non-text parts dropped
        ([{"type": "text", "text": "x"}, {"type": "image_url"}], "x"),
        (None, ""),
        (123, "123"),
    ],
)
def test_content_text_flattens_all_forms(content, expected):
    assert content_text(content) == expected


def test_last_user_text_returns_final_user_turn():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "reply"},
        {"role": "user", "content": "second"},
    ]
    assert translate.last_user_text(messages) == "second"


def test_last_user_text_falls_back_when_no_user_role():
    assert translate.last_user_text([{"role": "assistant", "content": "only"}]) == "only"


def test_flatten_transcript_labels_each_role():
    out = translate.flatten_transcript(
        [
            {"role": "system", "content": "be nice"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "yo"},
        ]
    )
    assert "System: be nice" in out
    assert "User: hi" in out
    assert "Assistant: yo" in out


def test_new_completion_id_is_unique_and_prefixed():
    a, b = translate.new_completion_id(), translate.new_completion_id()
    assert a.startswith("chatcmpl-") and b.startswith("chatcmpl-")
    assert a != b


def test_stream_chunk_shape():
    c = translate.stream_chunk("id1", "m", delta={"content": "x"})
    assert c["object"] == "chat.completion.chunk"
    assert c["choices"][0]["delta"] == {"content": "x"}
    assert c["choices"][0]["finish_reason"] is None


def test_stream_chunk_carries_finish_reason():
    c = translate.stream_chunk("id1", "m", delta={}, finish_reason="stop")
    assert c["choices"][0]["finish_reason"] == "stop"


def test_full_completion_shape():
    c = translate.full_completion("id1", "m", "answer")
    assert c["object"] == "chat.completion"
    assert c["choices"][0]["message"] == {"role": "assistant", "content": "answer"}
    assert c["choices"][0]["finish_reason"] == "stop"
    assert set(c["usage"]) == {"prompt_tokens", "completion_tokens", "total_tokens"}


def test_models_list_shape():
    d = translate.models_list("goose-1.39.0")
    assert d["object"] == "list"
    assert d["data"][0]["id"] == "goose-1.39.0"
    assert d["data"][0]["object"] == "model"
