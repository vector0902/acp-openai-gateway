"""Shared fixtures.

The centerpiece is `FakeACP`: an in-memory ACP agent implemented as an
``httpx.MockTransport`` handler. It reproduces the real streamable-HTTP routing
(POST is fire-and-accept; results/notifications arrive on a GET SSE channel,
correlated by JSON-RPC id) so `AcpClient` can be exercised end-to-end without a
network or a real agent. The SSE body is generated lazily, so it can echo the id
of the request the client POSTs *after* opening the channel — exactly as a real
agent does.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx
import pytest

from acp_openai_gateway.acp import AcpClient
from acp_openai_gateway.config import Settings
from acp_openai_gateway.server import create_app
from acp_openai_gateway.sessions import SessionStore


def _sse(obj: dict) -> bytes:
    return ("data: " + json.dumps(obj) + "\n\n").encode("utf-8")


def _update(sid: str, disc: str, *, text: str | None = None, title: str | None = None) -> dict:
    update: dict = {"sessionUpdate": disc}
    if text is not None:
        update["content"] = {"type": "text", "text": text}
    if title is not None:
        update["title"] = title
    return {
        "jsonrpc": "2.0",
        "method": "session/update",
        "params": {"sessionId": sid, "update": update},
    }


class _LazyStream(httpx.AsyncByteStream):
    """Wrap an async byte generator so httpx streams it (and closes it)."""

    def __init__(self, agen: AsyncIterator[bytes]) -> None:
        self._agen = agen

    async def __aiter__(self) -> AsyncIterator[bytes]:
        async for chunk in self._agen:
            yield chunk

    async def aclose(self) -> None:
        aclose = getattr(self._agen, "aclose", None)
        if aclose is not None:
            await aclose()


class FakeACP:
    """Configurable in-memory ACP agent for MockTransport."""

    def __init__(
        self,
        *,
        reply: str = "hello world",
        agent: tuple[str, str] = ("fake-agent", "9.9.9"),
        emit_thought: bool = False,
        emit_tool: bool = False,
        emit_permission: bool = False,
        prompt_status: int = 202,
    ) -> None:
        self.reply = reply
        self.agent = agent
        self.emit_thought = emit_thought
        self.emit_tool = emit_tool
        self.emit_permission = emit_permission
        self.prompt_status = prompt_status
        # observable state for assertions
        self.posted: list[dict] = []
        self.permission_answered = False
        self.session_count = 0
        self._pending: tuple[str, int, str | None] | None = None

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self._handle)

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=self.transport())

    # ── request routing ──────────────────────────────────────────────────────
    def _handle(self, request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return self._handle_post(request)
        has_session = "acp-session-id" in request.headers
        return httpx.Response(200, stream=_LazyStream(self._emit(has_session)))

    def _handle_post(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.posted.append(body)
        method = body.get("method")
        if method == "initialize":
            return httpx.Response(
                200,
                headers={"acp-connection-id": "conn-1"},
                json={
                    "jsonrpc": "2.0",
                    "id": body["id"],
                    "result": {
                        "protocolVersion": 1,
                        "agentInfo": {"name": self.agent[0], "version": self.agent[1]},
                        "agentCapabilities": {"loadSession": True},
                    },
                },
            )
        if method == "session/new":
            self.session_count += 1
            sid = f"sess-{self.session_count}"
            self._pending = ("new", body["id"], sid)
            return httpx.Response(202)
        if method == "session/prompt":
            if self.prompt_status >= 400:
                return httpx.Response(self.prompt_status, text="rejected")
            self._pending = ("prompt", body["id"], body["params"]["sessionId"])
            return httpx.Response(202)
        if method == "session/load":
            self._pending = ("load", body["id"], body["params"]["sessionId"])
            return httpx.Response(202)
        if method == "session/set_mode":
            return httpx.Response(202)
        if "result" in body:  # a client→agent response (permission answer)
            self.permission_answered = True
            return httpx.Response(202)
        return httpx.Response(202)

    async def _emit(self, has_session: bool) -> AsyncIterator[bytes]:
        op, rid, sid = self._pending or (None, None, None)
        if op == "new":
            yield _sse(
                {
                    "jsonrpc": "2.0",
                    "id": rid,
                    "result": {"sessionId": sid, "modes": {"currentModeId": "auto"}},
                }
            )
            return
        if op == "load":
            # replayed history — the client must drain (not surface) these
            yield _sse(_update(sid, "user_message_chunk", text="earlier question"))
            yield _sse(_update(sid, "agent_message_chunk", text="earlier answer"))
            yield _sse({"jsonrpc": "2.0", "id": rid, "result": {"modes": {}}})
            return
        if op == "prompt":
            if self.emit_thought:
                yield _sse(_update(sid, "agent_thought_chunk", text="let me think"))
            if self.emit_permission and not self.permission_answered:
                yield _sse(
                    {
                        "jsonrpc": "2.0",
                        "id": 9001,
                        "method": "session/request_permission",
                        "params": {
                            "sessionId": sid,
                            "options": [{"optionId": "allow", "kind": "allow_once"}],
                        },
                    }
                )
            if self.emit_tool:
                yield _sse(_update(sid, "tool_call", title="do_thing"))
            for i in range(0, len(self.reply), 4):
                yield _sse(_update(sid, "agent_message_chunk", text=self.reply[i : i + 4]))
            yield _sse({"jsonrpc": "2.0", "id": rid, "result": {"stopReason": "end_turn"}})
            return


# ── fixtures ─────────────────────────────────────────────────────────────────
@pytest.fixture
def fake_acp() -> FakeACP:
    return FakeACP()


@pytest.fixture
def acp_client(fake_acp: FakeACP) -> AcpClient:
    return AcpClient("http://fake", client=fake_acp.client())


class StubAcp:
    """Minimal AcpClient stand-in for server tests: deterministic, no transport.

    Yields a plain message reply by default, or an explicit `events` sequence
    (to exercise thought/tool/error handling in the server)."""

    def __init__(
        self,
        reply: str = "stub reply",
        info: dict | None = None,
        events: list[dict] | None = None,
    ) -> None:
        self.reply = reply
        self._info = info or {"name": "stub", "version": "1.0"}
        self._events = events
        self.new_sessions = 0
        self.attach_ok = True
        self.prompt_texts: list[str] = []

    async def agent_info(self) -> dict:
        return dict(self._info)

    async def new_session(self) -> str:
        self.new_sessions += 1
        return f"stub-sess-{self.new_sessions}"

    async def ensure_attached(self, sid: str) -> bool:
        return self.attach_ok

    async def prompt(self, sid: str, text: str):
        self.prompt_texts.append(text)
        if self._events is not None:
            for ev in self._events:
                yield ev
            return
        for i in range(0, len(self.reply), 5):
            yield {"type": "message", "text": self.reply[i : i + 5]}
        yield {"type": "done"}

    async def aclose(self) -> None:
        pass


@pytest.fixture
def fake_acp_cls() -> type[FakeACP]:
    """The FakeACP class, for tests that build custom-configured agents."""
    return FakeACP


@pytest.fixture
def stub_acp_cls() -> type[StubAcp]:
    return StubAcp


@pytest.fixture
def make_app(tmp_path):
    """Factory for a TestClient wired to a StubAcp + a real (temp) store.

    The client is entered as a context manager so the app lifespan runs (that's
    what populates app.state); all clients are cleanly exited at teardown.
    """
    from fastapi.testclient import TestClient

    created: list[TestClient] = []

    def _factory(stub: StubAcp | None = None, **setting_overrides):
        stub = stub or StubAcp()
        settings = Settings(session_state_path=str(tmp_path / "state.json"), **setting_overrides)
        store = SessionStore(settings.session_state_path)
        app = create_app(settings, acp=stub, store=store)
        client = TestClient(app)
        client.__enter__()  # run startup lifespan
        created.append(client)
        return client, stub, store

    yield _factory
    for client in created:
        client.__exit__(None, None, None)
