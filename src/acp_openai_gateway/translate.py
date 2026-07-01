"""Shape OpenAI chat-completion payloads. Pure functions, no I/O."""

from __future__ import annotations

import time
import uuid

from .sessions import content_text


def last_user_text(messages: list[dict]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            return content_text(m.get("content"))
    # no explicit user turn (unusual) — fall back to the last message
    return content_text(messages[-1].get("content")) if messages else ""


def flatten_transcript(messages: list[dict]) -> str:
    """Render a conversation as a single prompt, used to seed a cold session."""
    label = {"system": "System", "user": "User", "assistant": "Assistant", "tool": "Tool"}
    lines = []
    for m in messages:
        role = label.get(m.get("role", ""), (m.get("role") or "user").title())
        lines.append(f"{role}: {content_text(m.get('content'))}")
    return "\n\n".join(lines)


def new_completion_id() -> str:
    return "chatcmpl-" + uuid.uuid4().hex


def stream_chunk(cid: str, model: str, *, delta: dict, finish_reason: str | None = None) -> dict:
    return {
        "id": cid,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


def full_completion(cid: str, model: str, text: str) -> dict:
    return {
        "id": cid,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def models_list(model_id: str) -> dict:
    return {
        "object": "list",
        "data": [
            {"id": model_id, "object": "model", "created": int(time.time()), "owned_by": "acp"}
        ],
    }
