"""Integration tests for the OpenAI-compatible HTTP surface, via FastAPI's
TestClient wired to a StubAcp (no real agent)."""

import json

import pytest

pytestmark = pytest.mark.integration


def _sse_content(text: str) -> str:
    """Reassemble streamed `delta.content` from a raw SSE body."""
    out = []
    for line in text.splitlines():
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            break
        delta = json.loads(payload)["choices"][0]["delta"]
        out.append(delta.get("content", ""))
    return "".join(out)


def test_health(make_app):
    client, _, _ = make_app()
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_models_derives_agent_name(make_app, stub_acp_cls):
    client, _, _ = make_app(stub_acp_cls(info={"name": "goose", "version": "1.39.0"}))
    r = client.get("/v1/models")
    assert r.status_code == 200
    assert r.json()["data"][0]["id"] == "goose-1.39.0"


def test_models_honors_explicit_model_id(make_app):
    client, _, _ = make_app(model_id="my-model")
    assert client.get("/v1/models").json()["data"][0]["id"] == "my-model"


def test_auth_open_when_no_key(make_app):
    client, _, _ = make_app()
    assert client.get("/v1/models").status_code == 200


def test_auth_rejects_missing_or_wrong_key(make_app):
    client, _, _ = make_app(gateway_api_key="secret")
    assert client.get("/v1/models").status_code == 401
    assert client.get("/v1/models", headers={"Authorization": "Bearer nope"}).status_code == 401


def test_auth_accepts_correct_key(make_app):
    client, _, _ = make_app(gateway_api_key="secret")
    r = client.get("/v1/models", headers={"Authorization": "Bearer secret"})
    assert r.status_code == 200


def test_chat_non_streaming_returns_completion(make_app, stub_acp_cls):
    client, _, _ = make_app(stub_acp_cls(reply="pong"))
    r = client.post(
        "/v1/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": "ping"}]},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["content"] == "pong"
    assert body["model"] == "m"


def test_chat_streaming_reassembles_and_terminates(make_app, stub_acp_cls):
    client, _, _ = make_app(stub_acp_cls(reply="streamed reply"))
    r = client.post(
        "/v1/chat/completions",
        json={"model": "m", "stream": True, "messages": [{"role": "user", "content": "go"}]},
    )
    assert r.status_code == 200
    assert _sse_content(r.text) == "streamed reply"
    assert "data: [DONE]" in r.text


def test_empty_messages_is_400(make_app):
    client, _, _ = make_app()
    r = client.post("/v1/chat/completions", json={"model": "m", "messages": []})
    assert r.status_code == 400


def test_fresh_chat_creates_one_session_and_sends_only_user_text(make_app, stub_acp_cls):
    client, stub, store = make_app(stub_acp_cls(reply="ok"))
    client.post(
        "/v1/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": "hello"}]},
    )
    assert stub.new_sessions == 1
    assert stub.prompt_texts == ["hello"]


def test_second_turn_reuses_session(make_app, stub_acp_cls):
    stub = stub_acp_cls(reply="R")
    client, stub, store = make_app(stub)
    # turn 1
    client.post(
        "/v1/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
    )
    # turn 2 resends history (with the stub's reply) + a new user message
    client.post(
        "/v1/chat/completions",
        json={
            "model": "m",
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "R"},
                {"role": "user", "content": "more"},
            ],
        },
    )
    assert stub.new_sessions == 1  # reused, not recreated
    assert stub.prompt_texts == ["hi", "more"]  # only the new message on turn 2


def test_cold_unmatched_history_seeds_full_transcript(make_app, stub_acp_cls):
    # a conversation with prior history the gateway has never seen (e.g. edited):
    client, stub, store = make_app(stub_acp_cls(reply="ok"))
    client.post(
        "/v1/chat/completions",
        json={
            "model": "m",
            "messages": [
                {"role": "user", "content": "old q"},
                {"role": "assistant", "content": "old a"},
                {"role": "user", "content": "new q"},
            ],
        },
    )
    assert stub.new_sessions == 1
    seeded = stub.prompt_texts[0]
    assert "User: old q" in seeded and "Assistant: old a" in seeded and "User: new q" in seeded


def test_thoughts_and_tool_toggles_control_output(make_app, stub_acp_cls):
    rich_events = [
        {"type": "thought", "text": "thinking"},
        {"type": "tool", "name": "do_thing"},
        {"type": "message", "text": "answer"},
        {"type": "done"},
    ]

    # off (defaults): only the answer surfaces
    client, _, _ = make_app(stub_acp_cls(events=rich_events))
    r = client.post(
        "/v1/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": "x"}]},
    )
    assert r.json()["choices"][0]["message"]["content"] == "answer"

    # on: reasoning wrapped in <think>, tool status shown
    client2, _, _ = make_app(
        stub_acp_cls(events=rich_events), emit_thoughts=True, emit_tool_status=True
    )
    r2 = client2.post(
        "/v1/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": "x"}]},
    )
    content = r2.json()["choices"][0]["message"]["content"]
    assert "<think>" in content and "thinking" in content and "do_thing" in content
