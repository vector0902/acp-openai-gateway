"""Runtime configuration, read from environment variables (or a .env file)."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # protected_namespaces=() so a field may be named `model_id` without pydantic
    # warning about its reserved `model_` namespace.
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", protected_namespaces=()
    )

    # ── upstream ACP agent ──────────────────────────────────────────────────
    acp_url: str = "http://localhost:3000"
    """Base URL of the ACP agent. The endpoint used is ``{acp_url}/acp``."""
    acp_cwd: str = "/workspace"
    """Working directory handed to ``session/new``."""
    acp_mode: str = "auto"
    """Session mode when the agent supports it (e.g. goose: auto | smart_approve |
    approve | chat). ``auto`` keeps tools enabled without stalling on permission."""

    # ── HTTP server ─────────────────────────────────────────────────────────
    gateway_host: str = "0.0.0.0"
    gateway_port: int = 8000
    gateway_api_key: str = ""
    """If set, require ``Authorization: Bearer <key>``. Empty = no auth (local use)."""

    # ── model surface ───────────────────────────────────────────────────────
    model_id: str = ""
    """Model id advertised on /v1/models. Empty = derive from the agent's own
    name/version (from the ACP ``initialize`` handshake)."""

    # ── streaming detail ────────────────────────────────────────────────────
    emit_thoughts: bool = False
    """Stream ``agent_thought_chunk`` wrapped in <think></think> (some UIs render
    it as collapsible reasoning; raw OpenAI clients show the tags verbatim)."""
    emit_tool_status: bool = False
    """Emit a short status line per ``tool_call`` (off by default — internal
    bookkeeping tools are noise in a chat)."""

    # ── conversation continuity ─────────────────────────────────────────────
    session_state_path: str = ".acp_sessions.json"
    """File persisting the conversation→ACP-session map across restarts. The map
    lets a stateless OpenAI chat reuse the agent's stateful session. Empty =
    in-memory only."""
    cold_seed_history: bool = True
    """When a conversation can't be matched to a known session (fresh process,
    edited history, branch), seed a new session with the full prior transcript so
    context isn't lost. If false, only the latest user message is sent."""

    # ── timeouts (seconds) ──────────────────────────────────────────────────
    connect_timeout: float = 10.0
    read_timeout: float = 1800.0
    """Per-read timeout for the streaming channel. Remote inference can be slow;
    keep this generous but below the client's own timeout."""


def load_settings() -> Settings:
    return Settings()
