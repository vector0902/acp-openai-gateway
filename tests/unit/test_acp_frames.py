"""Unit tests for SSE frame parsing (`AcpClient._frames`), including the UTF-8
decoding and byte-boundary reassembly behavior that a naive decoder gets wrong.
"""

from collections.abc import AsyncIterator

import httpx
import pytest

from acp_openai_gateway.acp import AcpClient

pytestmark = pytest.mark.unit


class _Bytes(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for c in self._chunks:
            yield c

    async def aclose(self) -> None:
        pass


def _resp(chunks: list[bytes]) -> httpx.Response:
    return httpx.Response(200, stream=_Bytes(chunks))


async def _collect(chunks: list[bytes]) -> list[dict]:
    return [f async for f in AcpClient._frames(_resp(chunks))]


async def test_parses_data_lines_into_objects():
    frames = await _collect([b'data: {"a": 1}\n\n', b'data: {"b": 2}\n\n'])
    assert frames == [{"a": 1}, {"b": 2}]


async def test_ignores_non_data_and_keepalive_lines():
    frames = await _collect([b": keep-alive\n\n", b"event: x\n", b'data: {"ok": true}\n\n'])
    assert frames == [{"ok": True}]


async def test_handles_crlf_line_endings():
    frames = await _collect([b'data: {"v": 1}\r\n\r\n'])
    assert frames == [{"v": 1}]


async def test_reassembles_frame_split_across_chunks():
    # a single data line delivered in three arbitrary byte-boundary pieces
    frames = await _collect([b'data: {"msg"', b': "hel', b'lo"}\n\n'])
    assert frames == [{"msg": "hello"}]


async def test_decodes_utf8_even_when_multibyte_char_is_split():
    # "I’m" — the ’ is U+2019 (bytes e2 80 99). Cut the stream *inside* those
    # bytes; a per-chunk latin-1 decode would mojibake it, byte-buffering won't.
    full = 'data: {"text": "I’m"}\n\n'.encode()
    cut = full.index(b"\xe2") + 1  # mid multibyte sequence
    frames = await _collect([full[:cut], full[cut:]])
    assert frames == [{"text": "I’m"}]


async def test_skips_malformed_json_without_raising():
    frames = await _collect([b"data: {bad json\n\n", b'data: {"good": 1}\n\n'])
    assert frames == [{"good": 1}]
