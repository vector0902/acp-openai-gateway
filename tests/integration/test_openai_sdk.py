"""Compatibility tests using the *official* OpenAI Python SDK as the client.

The SDK talks to the app in-process via httpx.ASGITransport, so its own request
building and response/stream parsing must accept what the gateway emits. This is
the substantive evidence behind "works with OpenAI-compatible clients": nearly
all such tools are built on, or mirror, this SDK's wire behavior.
"""

import httpx
import pytest
from openai import AsyncOpenAI

from acp_openai_gateway.config import Settings
from acp_openai_gateway.server import create_app
from acp_openai_gateway.sessions import SessionStore

pytestmark = pytest.mark.integration


def _sdk_against(stub, tmp_path) -> tuple[AsyncOpenAI, httpx.AsyncClient]:
    settings = Settings(session_state_path=str(tmp_path / "state.json"))
    app = create_app(settings, acp=stub, store=SessionStore(settings.session_state_path))
    http = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gw")
    client = AsyncOpenAI(api_key="test", base_url="http://gw/v1", http_client=http)
    return client, http


async def test_sdk_non_streaming_completion(stub_acp_cls, tmp_path):
    client, http = _sdk_against(stub_acp_cls(reply="hi from the sdk"), tmp_path)
    try:
        resp = await client.chat.completions.create(
            model="m", messages=[{"role": "user", "content": "hi"}]
        )
        assert resp.object == "chat.completion"
        assert resp.choices[0].message.content == "hi from the sdk"
        assert resp.choices[0].finish_reason == "stop"
    finally:
        await http.aclose()


async def test_sdk_streaming_completion(stub_acp_cls, tmp_path):
    client, http = _sdk_against(stub_acp_cls(reply="streamed via the sdk"), tmp_path)
    try:
        stream = await client.chat.completions.create(
            model="m",
            messages=[{"role": "user", "content": "go"}],
            stream=True,
        )
        parts = [chunk.choices[0].delta.content or "" async for chunk in stream]
        assert "".join(parts) == "streamed via the sdk"
    finally:
        await http.aclose()


async def test_sdk_lists_models(stub_acp_cls, tmp_path):
    client, http = _sdk_against(
        stub_acp_cls(info={"name": "goose", "version": "1.39.0"}), tmp_path
    )
    try:
        models = await client.models.list()
        assert [m.id for m in models.data] == ["goose-1.39.0"]
    finally:
        await http.aclose()
