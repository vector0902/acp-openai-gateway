"""Opt-in smoke tests that drive a *real* third-party OpenAI client against a
running gateway. Skipped unless the client is installed and the gateway URL is
provided:

    CLIENT_SMOKE_BASE_URL=http://localhost:8000/v1 pytest -m client

These are not hermetic (they need the external binary + a live gateway backed by
a real ACP agent), which is why they're gated rather than part of the default
suite. See scripts/smoke_clients.sh for a one-command harness.
"""

import os
import shutil
import subprocess

import pytest

pytestmark = pytest.mark.client

BASE_URL = os.getenv("CLIENT_SMOKE_BASE_URL")


@pytest.mark.skipif(
    not BASE_URL or not shutil.which("aider"),
    reason="set CLIENT_SMOKE_BASE_URL and install aider (`pip install aider-chat`)",
)
def test_aider_roundtrip(tmp_path):
    """aider (a real OpenAI CLI client) completes a turn through the gateway."""
    subprocess.run(["git", "init", "-q", "."], cwd=tmp_path, check=True)
    env = {
        **os.environ,
        "OPENAI_API_BASE": BASE_URL,
        "OPENAI_API_KEY": os.getenv("CLIENT_SMOKE_API_KEY", "sk-test"),
        "AIDER_ANALYTICS": "false",
    }
    result = subprocess.run(
        [
            "aider",
            "--model", f"openai/{os.getenv('CLIENT_SMOKE_MODEL', 'goose')}",
            "--message", "Reply with exactly: AIDER_OK",
            "--yes-always",
            "--no-auto-commits",
            "--no-show-model-warnings",
            "--no-check-update",
            "--map-tokens", "0",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    combined = result.stdout + result.stderr
    assert "AIDER_OK" in combined, combined[-2000:]
