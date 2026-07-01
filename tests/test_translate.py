from acp_openai_gateway import translate
from acp_openai_gateway.sessions import content_text


def test_content_text_handles_str_and_parts():
    assert content_text("hello") == "hello"
    assert content_text([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]) == "ab"
    assert content_text([{"type": "image_url", "image_url": {"url": "x"}}]) == ""
    assert content_text(None) == ""


def test_last_user_text_picks_last_user_turn():
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "reply"},
        {"role": "user", "content": "second"},
    ]
    assert translate.last_user_text(msgs) == "second"


def test_flatten_transcript_labels_roles():
    out = translate.flatten_transcript(
        [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]
    )
    assert "User: hi" in out and "Assistant: yo" in out


def test_stream_chunk_shape():
    c = translate.stream_chunk("id1", "m", delta={"content": "x"})
    assert c["object"] == "chat.completion.chunk"
    assert c["choices"][0]["delta"] == {"content": "x"}
    assert c["choices"][0]["finish_reason"] is None


def test_full_completion_shape():
    c = translate.full_completion("id1", "m", "answer")
    assert c["object"] == "chat.completion"
    assert c["choices"][0]["message"] == {"role": "assistant", "content": "answer"}
    assert c["choices"][0]["finish_reason"] == "stop"


def test_models_list_shape():
    d = translate.models_list("goose-1.39.0")
    assert d["object"] == "list"
    assert d["data"][0]["id"] == "goose-1.39.0"
