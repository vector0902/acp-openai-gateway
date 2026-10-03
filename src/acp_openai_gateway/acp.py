"""Async client for the Agent Client Protocol (ACP) over streamable HTTP.

ACP is JSON-RPC 2.0. The streamable-HTTP transport (as implemented by goose's
`goose serve`, and shared by other ACP agents) has a specific, undocumented
header-routing model that this client encodes. It was pinned empirically by
driving full turns against goose 1.39.0:

  1. POST /acp  initialize                     (no acp headers)
       → 200 inline JSON; response header `acp-connection-id: <CID>`
  2. GET  /acp  (SSE)  header acp-connection-id
       → CONNECTION-scoped channel: `session/new` results arrive here
  3. POST /acp  session/new     header acp-connection-id
       → 202 (empty body); the {sessionId} result shows up on the step-2 channel
  4. GET  /acp  (SSE)  headers acp-connection-id + acp-session-id
       → SESSION-scoped channel: `session/update` notifications + the terminal
         prompt result land here
  5. POST /acp  session/prompt  headers acp-connection-id + acp-session-id
       → 202; a missing session header is a hard 400

Session-scoped methods (prompt, load, set_mode, cancel) require BOTH headers.
`session/load` replays the whole conversation as *_message_chunk frames — the
caller drains those. Default mode `auto` means `session/request_permission`
normally never fires; we auto-approve it anyway so a headless turn can't stall.
"""

from __future__ import annotations

import asyncio
import itertools
import json
from collections.abc import AsyncGenerator

import httpx


class AcpError(RuntimeError):
    pass


class AcpClient:
    def __init__(
        self,
        base_url: str,
        *,
        cwd: str = "/workspace",
        mode: str = "auto",
        provider: str = "goose",
        connect_timeout: float = 10.0,
        read_timeout: float = 1800.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._provider = provider
        if provider == "cbwb":
            # cbwb's ACP gateway: POST /api/v1/acp/connect returns JSON
            # {connectionId, sessionToken}; requests are request-scoped SSE
            # (no separate GET notification channel needed for prompt results).
            self._connect_url = base_url.rstrip("/") + "/api/v1/acp/connect"
            self._acp_url = base_url.rstrip("/") + "/api/v1/acp"
        else:
            # goose-style streamable HTTP: {base}/acp + connection-id header.
            self._connect_url = base_url.rstrip("/") + "/acp"
            self._acp_url = base_url.rstrip("/") + "/acp"
        self._cwd = cwd
        self._mode = mode
        self._ids = itertools.count(1)
        self._lock = asyncio.Lock()
        self._cid: str | None = None
        self._token: str | None = None
        self._attached: set[str] = set()
        self._agent_info: dict = {}
        # An injected client (e.g. with an httpx.MockTransport) enables testing
        # the transport without a live agent.
        if client is not None:
            self._client = client
        else:
            timeout = httpx.Timeout(read_timeout, connect=connect_timeout)
            self._client = httpx.AsyncClient(timeout=timeout)

    async def aclose(self) -> None:
        await self._client.aclose()

    # ── low-level helpers ────────────────────────────────────────────────────
    def _next_id(self) -> int:
        return next(self._ids)

    def _base_headers(self) -> dict[str, str]:
        """Headers common to every cbwb ACP request (CSRF guard + SSE)."""
        if self._provider == "cbwb":
            return {
                "X-CodeBuddy-Request": "1",
                "Accept": "application/json, text/event-stream",
            }
        return {}

    def _session_headers(self, sid: str | None = None) -> dict[str, str]:
        """Headers that tie a request to the ACP connection/session.

        goose: acp-connection-id + acp-session-id (session id).
        cbwb: acp-connection-id + acp-session-token (connect-time token).
        """
        base = self._base_headers()
        if not self._cid:
            return base
        if self._provider == "cbwb":
            return {
                **base,
                "acp-connection-id": self._cid,
                "acp-session-token": self._token or "",
            }
        h = {**base, "acp-connection-id": self._cid}
        if sid is not None:
            h["acp-session-id"] = sid
        return h

    async def _post(self, payload: dict, headers: dict) -> httpx.Response:
        return await self._client.post(
            self._acp_url,
            headers={
                "Content-Type": "application/json",
                **self._base_headers(),
                **headers,
            },
            content=json.dumps(payload),
        )

    def _post_sse(self, payload: dict, headers: dict):
        """POST and stream the response as SSE (cbwb request-scoped mode)."""
        return self._client.stream(
            "POST",
            self._acp_url,
            headers={
                "Content-Type": "application/json",
                **self._base_headers(),
                **headers,
            },
            content=json.dumps(payload),
        )

    def _sse(self, headers: dict):
        return self._client.stream(
            "GET", self._acp_url, headers={"Accept": "text/event-stream", **headers}
        )

    @staticmethod
    async def _frames(resp: httpx.Response) -> AsyncGenerator[dict, None]:
        """Yield JSON objects from an SSE `data:` stream.

        Buffer raw bytes and split on newlines, decoding each line as UTF-8
        explicitly — the ACP stream carries no charset, and guessing latin-1
        mojibakes multibyte characters.
        """
        buffer = b""
        async for chunk in resp.aiter_bytes():
            buffer += chunk
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                line = line.rstrip(b"\r")
                if not line.startswith(b"data:"):
                    continue
                payload = line[5:].strip()
                if not payload:
                    continue
                try:
                    yield json.loads(payload.decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue

    # ── connection / session lifecycle ───────────────────────────────────────
    async def ensure_connection(self) -> str:
        async with self._lock:
            if self._cid:
                return self._cid
            if self._provider == "cbwb":
                return await self._connect_cbwb()
            r = await self._client.post(
                self._acp_url,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json, text/event-stream",
                },
                content=json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": self._next_id(),
                        "method": "initialize",
                        "params": {"protocolVersion": 1, "clientCapabilities": {}},
                    }
                ),
            )
            r.raise_for_status()
            cid = r.headers.get("acp-connection-id")
            if not cid:
                raise AcpError("ACP initialize returned no acp-connection-id header")
            try:
                self._agent_info = r.json().get("result", {}).get("agentInfo", {})
            except (ValueError, json.JSONDecodeError):
                self._agent_info = {}
            self._cid = cid
            self._attached.clear()  # a new connection has no attached sessions
            return cid

    async def _connect_cbwb(self) -> str:
        """cbwb: POST /api/v1/acp/connect → {connectionId, sessionToken}"""
        r = await self._client.post(
            self._connect_url,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
                "X-CodeBuddy-Request": "1",
            },
            content=json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": self._next_id(),
                    "method": "initialize",
                    "params": {"protocolVersion": 1, "clientCapabilities": {}},
                }
            ),
        )
        r.raise_for_status()
        try:
            body = r.json()
        except (ValueError, json.JSONDecodeError) as e:
            raise AcpError(f"cbwb connect returned non-JSON: {r.text[:200]}") from e
        cid = body.get("connectionId")
        token = body.get("sessionToken")
        if not cid or not token:
            raise AcpError(f"cbwb connect missing connectionId/sessionToken: {body}")
        self._cid = cid
        self._token = token
        self._attached.clear()
        # cbwb requires an explicit initialize before session/new methods work.
        await self._initialize_cbwb()
        return cid

    async def _initialize_cbwb(self) -> None:
        req_id = self._next_id()
        payload = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": "initialize",
            "params": {"protocolVersion": 1, "clientCapabilities": {}},
        }
        async with self._post_sse(payload, self._session_headers()) as resp:
            if resp.status_code >= 400:
                raise AcpError(f"cbwb initialize rejected ({resp.status_code})")
            async for frame in self._frames(resp):
                if frame.get("id") == req_id and "result" in frame:
                    self._agent_info = frame["result"].get("agentInfo", {})
                    break

    async def agent_info(self) -> dict:
        await self.ensure_connection()
        return dict(self._agent_info)

    async def new_session(self) -> str:
        cid = await self.ensure_connection()
        req_id = self._next_id()
        sid: str | None = None
        payload = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": "session/new",
            "params": {"cwd": self._cwd, "mcpServers": []},
        }
        if self._provider == "cbwb":
            # cbwb: request-scoped SSE — result comes back on this POST.
            async with self._post_sse(payload, self._session_headers()) as resp:
                if resp.status_code >= 400:
                    raise AcpError(f"session/new rejected ({resp.status_code})")
                async for frame in self._frames(resp):
                    if frame.get("id") == req_id and "result" in frame:
                        sid = frame["result"].get("sessionId")
                        break
        else:
            conn_headers = {"acp-connection-id": cid}
            async with self._sse(conn_headers) as resp:
                r = await self._post(payload, conn_headers)
                r.raise_for_status()
                async for frame in self._frames(resp):
                    if frame.get("id") == req_id and "result" in frame:
                        sid = frame["result"].get("sessionId")
                        break
        if not sid:
            raise AcpError("ACP session/new produced no sessionId")
        self._attached.add(sid)
        await self._maybe_set_mode(cid, sid)
        return sid

    async def _load_session(self, cid: str, sid: str) -> bool:
        """Re-attach a persisted session via session/load; drain the replayed
        history. True once the agent confirms with a result frame."""
        load_id = self._next_id()
        payload = {
            "jsonrpc": "2.0",
            "id": load_id,
            "method": "session/load",
            "params": {"sessionId": sid, "cwd": self._cwd, "mcpServers": []},
        }
        try:
            if self._provider == "cbwb":
                async with self._post_sse(payload, self._session_headers()) as resp:
                    if resp.status_code >= 400:
                        return False
                    async for frame in self._frames(resp):
                        if frame.get("id") == load_id:
                            return "result" in frame
                return False
            headers = {"acp-connection-id": cid, "acp-session-id": sid}
            async with self._sse(headers) as resp:
                r = await self._post(payload, headers)
                if r.status_code >= 400:
                    return False
                async for frame in self._frames(resp):
                    if frame.get("id") == load_id:
                        return "result" in frame
        except httpx.HTTPError:
            return False
        return False

    async def ensure_attached(self, sid: str) -> bool:
        """Make `sid` usable on the current connection. Returns False if the
        session no longer exists upstream (caller should create a new one)."""
        cid = await self.ensure_connection()
        if sid in self._attached:
            return True
        if await self._load_session(cid, sid):
            self._attached.add(sid)
            return True
        return False

    async def _maybe_set_mode(self, cid: str, sid: str) -> None:
        if not self._mode or self._mode == "auto":
            return
        await self._post(
            {
                "jsonrpc": "2.0",
                "id": self._next_id(),
                "method": "session/set_mode",
                "params": {"sessionId": sid, "modeId": self._mode},
            },
            self._session_headers(sid),
        )

    async def _answer_permission(self, frame: dict, headers: dict) -> None:
        opts = frame.get("params", {}).get("options", []) or []
        allow = next(
            (o for o in opts if str(o.get("kind", "")).startswith("allow")),
            opts[0] if opts else {"optionId": "allow"},
        )
        await self._post(
            {
                "jsonrpc": "2.0",
                "id": frame.get("id"),
                "result": {"outcome": {"outcome": "selected", "optionId": allow.get("optionId")}},
            },
            headers,
        )

    # ── the turn ─────────────────────────────────────────────────────────────
    async def prompt(
        self,
        sid: str,
        text: str = "",
        blocks: list[dict] | None = None,
    ) -> AsyncGenerator[dict, None]:
        """Drive one turn. Yields semantic events until the turn completes:

          {"type": "message", "text": ...}   assistant output (stream this)
          {"type": "thought", "text": ...}   reasoning
          {"type": "tool",    "name": ...}   tool invocation
          {"type": "error",   "text": ...}   upstream error
          {"type": "done"}                   terminal (stopReason reached)

        `blocks` (ACP content blocks) overrides plain `text` when given.
        """
        await self.ensure_connection()
        prompt_id = self._next_id()
        headers = self._session_headers(sid)
        prompt_blocks = blocks if blocks is not None else [{"type": "text", "text": text}]
        payload = {
            "jsonrpc": "2.0",
            "id": prompt_id,
            "method": "session/prompt",
            "params": {"sessionId": sid, "prompt": prompt_blocks},
        }
        if self._provider == "cbwb":
            # cbwb: request-scoped SSE — events come back on this POST.
            async with self._post_sse(payload, headers) as resp:
                if resp.status_code >= 400:
                    async with self._lock:
                        self._cid = None
                        self._attached.discard(sid)
                    yield {"type": "error", "text": f"prompt rejected ({resp.status_code})"}
                    return
                async for frame in self._frames(resp):
                    async for ev in self._handle_frame(frame, prompt_id, headers):
                        yield ev
            return
        async with self._sse(headers) as resp:
            pr = await self._post(payload, headers)
            if pr.status_code >= 400:
                # Likely a stale connection. Force re-init + re-attach next turn.
                async with self._lock:
                    self._cid = None
                    self._attached.discard(sid)
                yield {"type": "error", "text": f"prompt rejected ({pr.status_code})"}
                return
            async for frame in self._frames(resp):
                async for ev in self._handle_frame(frame, prompt_id, headers):
                    yield ev

    async def _handle_frame(
        self, frame: dict, prompt_id: int, headers: dict
    ) -> AsyncGenerator[dict, None]:
        method = frame.get("method")
        if method == "session/update":
            upd = frame.get("params", {}).get("update", {})
            disc = upd.get("sessionUpdate")
            if disc == "agent_message_chunk":
                text_ = (upd.get("content") or {}).get("text")
                if text_:
                    yield {"type": "message", "text": text_}
            elif disc == "agent_thought_chunk":
                text_ = (upd.get("content") or {}).get("text")
                if text_:
                    yield {"type": "thought", "text": text_}
            elif disc == "tool_call":
                name = upd.get("title") or upd.get("name") or "tool"
                yield {"type": "tool", "name": name}
        elif method == "session/request_permission":
            await self._answer_permission(frame, headers)
        elif frame.get("id") == prompt_id and ("result" in frame or "error" in frame):
            if "error" in frame:
                yield {"type": "error", "text": json.dumps(frame["error"])[:300]}
            yield {"type": "done"}
