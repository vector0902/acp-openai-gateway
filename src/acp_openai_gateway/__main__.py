"""Console entrypoint: `acp-openai-gateway` / `python -m acp_openai_gateway`."""

from __future__ import annotations

import uvicorn

from .config import load_settings


def main() -> None:
    settings = load_settings()
    uvicorn.run(
        "acp_openai_gateway.server:app",
        host=settings.gateway_host,
        port=settings.gateway_port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
