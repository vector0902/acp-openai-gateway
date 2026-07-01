"""Opt-in tests against a real ACP agent.

Skipped unless ACP_LIVE_URL points at a running agent, e.g.:

    ACP_LIVE_URL=http://localhost:3000 pytest -m live
"""

import os

import pytest

from acp_openai_gateway.acp import AcpClient

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not os.getenv("ACP_LIVE_URL"), reason="set ACP_LIVE_URL to run"),
]


async def test_live_prompt_roundtrip():
    client = AcpClient(os.environ["ACP_LIVE_URL"])
    try:
        info = await client.agent_info()
        assert info.get("name")

        sid = await client.new_session()
        reply = "".join(
            ev["text"]
            async for ev in client.prompt(sid, "Reply with exactly: LIVE")
            if ev["type"] == "message"
        )
        assert "LIVE" in reply.upper()

        # continuity: same session should recall the prior turn
        assert await client.ensure_attached(sid) is True
    finally:
        await client.aclose()
