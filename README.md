# acp-openai-gateway

Expose any agent that speaks the **[Agent Client Protocol](https://agentclientprotocol.com) (ACP)** behind an **OpenAI-compatible** `/v1/chat/completions` + `/v1/models` API.

ACP agents (e.g. [goose](https://github.com/aaif-goose/goose)'s `goose serve`) talk JSON-RPC over a streamable-HTTP `/acp` endpoint — which OpenAI-native tools like Open WebUI, the OpenAI SDKs, LibreChat, or `curl` can't consume. This gateway is the thin translator in between, so those agents show up as ordinary "models" wherever an OpenAI base URL is accepted.

```
OpenAI client ──► /v1/chat/completions ──►  acp-openai-gateway  ──► POST /acp (ACP) ──► agent
 (Open WebUI,        (this service)                                  (goose, etc.)
  SDK, curl)
```

- ✅ Streaming (SSE) and non-streaming responses
- ✅ `/v1/models` advertises the agent (name/version from the ACP handshake)
- ✅ **Stateful conversations** over the stateless OpenAI protocol — maps each chat to a persistent ACP session (survives restarts)
- ✅ Optional bearer-key auth; agent-agnostic; no code changes to the agent
- ✅ Auto-approves ACP permission prompts so headless turns never stall

> **Status:** beta. Verified end-to-end against **goose 1.39.0**. ACP's HTTP transport is still evolving and partly undocumented — see [Compatibility](#compatibility).

## Quickstart

### Run the agent (example: goose)

```bash
goose serve --host 0.0.0.0 --port 3000   # exposes ACP at http://localhost:3000/acp
```

### Run the gateway

```bash
pip install acp-openai-gateway          # or: pip install -e . from a clone
ACP_URL=http://localhost:3000 acp-openai-gateway
# serving OpenAI API at http://localhost:8000/v1
```

### Use it

```bash
# list models — reports the agent's own name/version
curl -s http://localhost:8000/v1/models

# chat (streaming)
curl -N http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"goose","stream":true,"messages":[{"role":"user","content":"Say hi in three words"}]}'
```

Point any OpenAI client at `http://localhost:8000/v1`. For **Open WebUI**, add it under *Admin → Settings → Connections → OpenAI* (or via `OPENAI_API_BASE_URL`); see [`docker-compose.example.yml`](docker-compose.example.yml).

### Clients

The gateway implements the OpenAI **Chat Completions** (`/v1/chat/completions`, streaming + non-streaming) and **Models** (`/v1/models`) endpoints.

**Verified** — a full chat round-trip through the gateway:

- **official OpenAI Python SDK** — streaming, non-streaming, `models.list()` (in CI, via `httpx.ASGITransport`)
- **Aider** — one-shot CLI run returns the agent's reply (gated test `pytest -m client`; see [`scripts/smoke_clients.sh`](scripts/smoke_clients.sh))
- **Open WebUI** — live, end-to-end against goose

**Partially verified:**

- **LibreChat** — recognized as a custom endpoint and successfully fetches the model list from `/v1/models`; a full chat wasn't automated (its version-specific chat API), so it's not yet in the fully-verified list.

**Expected to work** (speak the same two endpoints; not individually tested): Jan, Continue.dev, LlamaIndex/LangChain, the Vercel AI SDK, `curl`, and most OpenAI-compatible tools. Clients that additionally require `/v1/completions` (legacy), `/v1/embeddings`, or capability metadata will see those endpoints 404 (usually harmless). Reports welcome.

### Docker

```bash
# published image (after a release):
docker run --rm -p 8000:8000 -e ACP_URL=http://host.docker.internal:3000 \
  ghcr.io/OWNER/acp-openai-gateway:latest

# or build locally:
docker build -t acp-openai-gateway .
docker run --rm -p 8000:8000 -e ACP_URL=http://host.docker.internal:3000 acp-openai-gateway
```

## Configuration

All via environment variables (or a `.env` file — see [`.env.example`](.env.example)):

| Variable | Default | Description |
|---|---|---|
| `ACP_URL` | `http://localhost:3000` | Agent base URL; the gateway calls `{ACP_URL}/acp`. |
| `ACP_CWD` | `/workspace` | Working dir passed to `session/new`. |
| `ACP_MODE` | `auto` | Session mode (goose: `auto`/`smart_approve`/`approve`/`chat`). `auto` avoids permission stalls. |
| `GATEWAY_HOST` / `GATEWAY_PORT` | `0.0.0.0` / `8000` | Bind address. |
| `GATEWAY_API_KEY` | _(empty)_ | If set, require `Authorization: Bearer <key>`. |
| `MODEL_ID` | _(empty)_ | Model id on `/v1/models`. Empty → derived from the agent's name/version. |
| `EMIT_THOUGHTS` | `false` | Stream reasoning wrapped in `<think></think>`. |
| `EMIT_TOOL_STATUS` | `false` | Emit a status line per tool call. |
| `SESSION_STATE_PATH` | `.acp_sessions.json` | Where the conversation→session map is persisted (empty = memory only). |
| `COLD_SEED_HISTORY` | `true` | Seed a new session with the full transcript when a chat can't be matched. |
| `CONNECT_TIMEOUT` / `READ_TIMEOUT` | `10` / `1800` | Socket timeouts (seconds). |

## How it works

### ACP-over-HTTP transport

ACP is JSON-RPC 2.0. The streamable-HTTP transport routes messages by two headers — pinned empirically against goose (it isn't formally documented):

1. `POST /acp` **initialize** → response header `acp-connection-id`.
2. `GET /acp` (SSE) with `acp-connection-id` → connection channel; `session/new` results arrive here.
3. `POST /acp` **session/new** → `202`, the `sessionId` shows up on the channel from step 2.
4. `GET /acp` (SSE) with `acp-connection-id` **+ `acp-session-id`** → session channel; `session/update` notifications and the terminal prompt result land here.
5. `POST /acp` **session/prompt** (both headers) → streamed as `agent_message_chunk` frames, re-emitted as OpenAI deltas.

`session/request_permission` is auto-answered with *allow* (and `auto` mode usually pre-empts it), so a headless turn can't block.

### Stateful chat over a stateless protocol

OpenAI clients resend the whole transcript each turn; ACP agents keep server-side session state. The gateway bridges them by hashing the transcript: after each turn it stores `hash(messages + reply) → sessionId`, so the next turn's history hash matches and it reuses the session — sending only the new message. Fresh chats create a new session; unmatched ones (restart, edited/branched history) get a new session seeded with the full transcript. The map is persisted (`SESSION_STATE_PATH`), so restarts resume via ACP `session/load`.

## Compatibility

Built and verified against **goose 1.39.0**'s `goose serve`. The gateway targets standard ACP methods (`initialize`, `session/new`, `session/load`, `session/prompt`, `session/set_mode`, `session/request_permission`) and the streamable-HTTP header routing described above. Other ACP agents that implement the same HTTP transport should work; the transport is still stabilizing upstream, so pin your agent version and file an issue if frames differ.

## Development

```bash
pip install -e ".[dev]"
ruff check .
pytest -q --cov=acp_openai_gateway --cov-report=term-missing
```

Tests are split into `tests/unit/` (pure translation + session logic, incl.
property-based checks with Hypothesis) and `tests/integration/` (the real
`AcpClient` driven against an in-memory fake ACP agent via `httpx.MockTransport`,
and the FastAPI app via `TestClient`). No network or live agent is needed — CI
runs the whole suite. To also run against a real agent:

```bash
ACP_LIVE_URL=http://localhost:3000 pytest -m live
```

## Limitations

- ACP HTTP transport is undocumented/evolving — treat agent-version pinning as required.
- Transcript-hash continuity assumes the client echoes prior turns verbatim (Open WebUI does); otherwise it falls back to a freshly-seeded session (context preserved, agent tool-state not).
- Single model per gateway instance (the agent). Selecting among an agent's own sub-models isn't exposed yet.

## License

[MIT](LICENSE)
