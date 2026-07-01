"""Bridge stateless OpenAI chat requests to stateful ACP sessions.

OpenAI's /v1/chat/completions is stateless: the client resends the full message
history every turn. ACP agents, by contrast, keep a stateful server-side session
(history, tool state). To get the best of both we map a conversation to an ACP
session by hashing its transcript:

  • After each turn we store  hash(messages + [assistant_reply]) → sessionId.
  • On the next turn the client sends those same messages plus a new user
    message; hash(messages[:-1]) matches the stored key → we reuse the session
    and send only the new message.

A fresh conversation (no prior messages) always creates a new session. An
unmatched conversation with history (restart without persistence, edited/branched
history) falls back to a new session seeded with the full transcript.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from typing import Any


def content_text(content: Any) -> str:
    """Flatten an OpenAI message `content` (str, or list of parts) to text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict) and p.get("type") == "text":
                parts.append(p.get("text", ""))
            elif isinstance(p, str):
                parts.append(p)
        return "".join(parts)
    return "" if content is None else str(content)


def _normalize(messages: list[dict]) -> list[dict]:
    return [{"role": m.get("role"), "content": content_text(m.get("content"))} for m in messages]


def transcript_hash(messages: list[dict]) -> str:
    blob = json.dumps(_normalize(messages), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class SessionStore:
    def __init__(self, state_path: str = "") -> None:
        self._path = state_path
        self._map: dict[str, str] = {}
        self._lock = asyncio.Lock()
        self._load()

    def _load(self) -> None:
        if not self._path:
            return
        try:
            with open(self._path) as f:
                data = json.load(f)
            if isinstance(data, dict):
                self._map = {str(k): str(v) for k, v in data.items()}
        except (OSError, ValueError):
            pass

    async def _save(self) -> None:
        if not self._path:
            return
        try:
            snapshot = dict(self._map)
            tmp = f"{self._path}.tmp"
            with open(tmp, "w") as f:
                json.dump(snapshot, f)
            os.replace(tmp, self._path)
        except OSError:
            pass  # persistence is best-effort

    async def lookup(self, prior_messages: list[dict]) -> str | None:
        """Session for a conversation whose transcript-so-far is `prior_messages`."""
        if not prior_messages:
            return None
        async with self._lock:
            return self._map.get(transcript_hash(prior_messages))

    async def remember(self, full_messages: list[dict], session_id: str) -> None:
        """Record that after `full_messages` (incl. the assistant reply) the
        conversation is served by `session_id`."""
        async with self._lock:
            self._map[transcript_hash(full_messages)] = session_id
        await self._save()

    async def forget(self, session_id: str) -> None:
        async with self._lock:
            self._map = {k: v for k, v in self._map.items() if v != session_id}
        await self._save()
