"""OpenAI-compatible HTTP surface over an ACP agent."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from . import __version__, translate
from .acp import AcpClient
from .config import Settings, load_settings
from .sessions import SessionStore


def create_app(
    settings: Settings | None = None,
    *,
    acp: AcpClient | None = None,
    store: SessionStore | None = None,
) -> FastAPI:
    settings = settings or load_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = settings
        # `acp`/`store` can be injected (tests pass fakes); otherwise build real
        # ones and take ownership of closing what we created.
        owns_acp = acp is None
        app.state.acp = acp or AcpClient(
            settings.acp_url,
            cwd=settings.acp_cwd,
            mode=settings.acp_mode,
            connect_timeout=settings.connect_timeout,
            read_timeout=settings.read_timeout,
        )
        app.state.store = store or SessionStore(settings.session_state_path)
        try:
            yield
        finally:
            if owns_acp:
                await app.state.acp.aclose()

    app = FastAPI(title="acp-openai-gateway", version=__version__, lifespan=lifespan)

    def require_auth(request: Request) -> None:
        key = settings.gateway_api_key
        if not key:
            return
        auth = request.headers.get("authorization", "")
        token = auth[7:].strip() if auth.lower().startswith("bearer ") else auth.strip()
        if token != key:
            raise HTTPException(status_code=401, detail="Invalid API key")

    async def model_name() -> str:
        if settings.model_id:
            return settings.model_id
        info = await app.state.acp.agent_info()
        name = info.get("name") or "acp"
        version = info.get("version")
        return f"{name}-{version}" if version else name

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok", "version": __version__}

    @app.get("/v1/models")
    async def list_models(_: None = Depends(require_auth)) -> dict:
        try:
            mid = await model_name()
        except Exception:
            mid = settings.model_id or "acp"
        return translate.models_list(mid)

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request, _: None = Depends(require_auth)):
        try:
            body = await request.json()
        except (ValueError, json.JSONDecodeError):
            raise HTTPException(status_code=400, detail="Body must be valid JSON") from None
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            raise HTTPException(status_code=400, detail="`messages` must be a non-empty array")
        stream = bool(body.get("stream", False))
        model = body.get("model") or (settings.model_id or "acp")

        acp: AcpClient = app.state.acp
        store: SessionStore = app.state.store

        # ── resolve the ACP session for this conversation ───────────────────
        prior = messages[:-1]
        sid = await store.lookup(prior)
        if sid is not None and not await acp.ensure_attached(sid):
            await store.forget(sid)
            sid = None
        if sid is None:
            sid = await acp.new_session()
            # cold branch: seed with full transcript so context isn't lost
            if prior and settings.cold_seed_history:
                send_text = translate.flatten_transcript(messages)
            else:
                send_text = translate.last_user_text(messages)
        else:
            send_text = translate.last_user_text(messages)

        cid = translate.new_completion_id()

        async def run():
            """Drive the turn once, yielding (kind, text) events and finally
            persisting the continuation mapping. `kind` is 'message' or 'error'."""
            reply_parts: list[str] = []
            thinking = False
            async for ev in acp.prompt(sid, send_text):
                kind = ev["type"]
                if kind == "message":
                    if thinking:
                        yield ("raw", "</think>\n")
                        thinking = False
                    reply_parts.append(ev["text"])
                    yield ("message", ev["text"])
                elif kind == "thought" and settings.emit_thoughts:
                    if not thinking:
                        yield ("raw", "<think>")
                        thinking = True
                    yield ("raw", ev["text"])
                elif kind == "tool" and settings.emit_tool_status:
                    yield ("raw", f"\n\n_⚙️ {ev['name']}…_\n\n")
                elif kind == "error":
                    yield ("error", ev["text"])
                elif kind == "done":
                    if thinking:
                        yield ("raw", "</think>\n")
                    break
            reply = "".join(reply_parts)
            # Record continuation so the next turn reuses this session.
            await store.remember(messages + [{"role": "assistant", "content": reply}], sid)

        if stream:
            async def sse():
                yield _sse(translate.stream_chunk(cid, model, delta={"role": "assistant"}))
                async for kind, text in run():
                    if kind in ("message", "raw"):
                        yield _sse(translate.stream_chunk(cid, model, delta={"content": text}))
                    elif kind == "error":
                        yield _sse(
                            translate.stream_chunk(cid, model, delta={"content": f"\n\n⚠️ {text}"})
                        )
                yield _sse(translate.stream_chunk(cid, model, delta={}, finish_reason="stop"))
                yield "data: [DONE]\n\n"

            return StreamingResponse(sse(), media_type="text/event-stream")

        # non-streaming: collect the whole reply
        parts: list[str] = []
        async for kind, text in run():
            if kind in ("message", "raw"):
                parts.append(text)
            elif kind == "error":
                parts.append(f"\n\n⚠️ {text}")
        return JSONResponse(translate.full_completion(cid, model, "".join(parts)))

    return app


def _sse(obj: dict) -> str:
    return "data: " + json.dumps(obj, ensure_ascii=False) + "\n\n"


app = create_app()
