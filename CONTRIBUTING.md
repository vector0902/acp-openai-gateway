# Contributing

Thanks for your interest! This project is small and focused: translate ACP ↔ the
OpenAI chat API, correctly and with minimal surface area.

## Dev setup

```bash
pip install -e ".[dev]"
ruff check .          # lint
ruff format .         # format
pytest -q             # unit tests (no network)
```

## Guidelines

- Keep the dependency set small (FastAPI, httpx, pydantic).
- The ACP HTTP transport is undocumented and version-sensitive. If you change
  transport behavior, note **which agent and version** you verified against, and
  prefer capturing real frames over assuming a schema.
- Pure logic (translation, session hashing) belongs in `translate.py` /
  `sessions.py` and must stay unit-testable without a network.
- Add or update tests for behavior changes; CI runs ruff + pytest on 3.10 & 3.12.

## Reporting issues

Include your agent + version (e.g. `goose 1.39.0`), the gateway version, and — if
it's a transport problem — a captured `/acp` frame or two. A minimal `curl` that
reproduces it is ideal.
