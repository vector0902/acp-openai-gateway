"""Integration tests: the real AcpClient driven against the in-memory FakeACP
(httpx.MockTransport). No network, no live agent."""

import pytest

from acp_openai_gateway.acp import AcpClient

pytestmark = pytest.mark.integration


def _methods(fake) -> list[str]:
    return [b.get("method") for b in fake.posted]


async def test_agent_info_from_initialize(acp_client, fake_acp):
    info = await acp_client.agent_info()
    assert info == {"name": "fake-agent", "version": "9.9.9"}


async def test_connection_is_established_once(acp_client, fake_acp):
    await acp_client.new_session()
    sid = await acp_client.new_session()
    async for _ in acp_client.prompt(sid, "hi"):
        pass
    assert _methods(fake_acp).count("initialize") == 1  # cached across calls


async def test_new_session_returns_id(acp_client, fake_acp):
    sid = await acp_client.new_session()
    assert sid == "sess-1"
    assert "session/new" in _methods(fake_acp)


async def test_prompt_streams_message_then_done(fake_acp_cls):
    fake = fake_acp_cls(reply="hello world")
    client = AcpClient("http://fake", client=fake.client())
    sid = await client.new_session()
    events = [ev async for ev in client.prompt(sid, "hi")]

    text = "".join(e["text"] for e in events if e["type"] == "message")
    assert text == "hello world"
    assert events[-1] == {"type": "done"}
    # the prompt carried exactly the user's text
    prompt_post = next(b for b in fake.posted if b.get("method") == "session/prompt")
    assert prompt_post["params"]["prompt"][0]["text"] == "hi"


async def test_prompt_surfaces_thoughts_and_tools_when_present(fake_acp_cls):
    fake = fake_acp_cls(reply="ok", emit_thought=True, emit_tool=True)
    client = AcpClient("http://fake", client=fake.client())
    sid = await client.new_session()
    kinds = {ev["type"] async for ev in client.prompt(sid, "go")}
    assert {"thought", "tool", "message", "done"} <= kinds


async def test_permission_request_is_auto_answered(fake_acp_cls):
    fake = fake_acp_cls(reply="done", emit_permission=True)
    client = AcpClient("http://fake", client=fake.client())
    sid = await client.new_session()
    async for _ in client.prompt(sid, "please act"):
        pass
    assert fake.permission_answered is True


async def test_ensure_attached_creates_no_load_for_fresh_session(acp_client, fake_acp):
    sid = await acp_client.new_session()
    assert await acp_client.ensure_attached(sid) is True
    assert "session/load" not in _methods(fake_acp)


async def test_ensure_attached_loads_unknown_session(acp_client, fake_acp):
    # a session id the client never created must be re-attached via session/load
    assert await acp_client.ensure_attached("sess-external") is True
    assert "session/load" in _methods(fake_acp)


async def test_set_mode_issued_only_for_non_auto_mode(fake_acp_cls):
    fake = fake_acp_cls()
    client = AcpClient("http://fake", mode="smart_approve", client=fake.client())
    await client.new_session()
    assert "session/set_mode" in _methods(fake)


async def test_auto_mode_does_not_set_mode(acp_client, fake_acp):
    await acp_client.new_session()
    assert "session/set_mode" not in _methods(fake_acp)


async def test_prompt_rejection_yields_error_and_resets_connection(fake_acp_cls):
    fake = fake_acp_cls(prompt_status=400)
    client = AcpClient("http://fake", client=fake.client())
    sid = await client.new_session()
    events = [ev async for ev in client.prompt(sid, "hi")]
    assert events[-1]["type"] == "error"
    assert client._cid is None  # forced re-init on the next turn
