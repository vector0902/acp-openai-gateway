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

## Releasing

Releases are cut by pushing a version tag; [`.github/workflows/release.yml`](.github/workflows/release.yml)
publishes to PyPI and GHCR.

1. Bump `version` in `pyproject.toml` (and add a `CHANGELOG.md` entry).
2. Tag and push: `git tag v0.1.0 && git push origin v0.1.0`.
   The workflow verifies the tag equals the `pyproject.toml` version, builds
   sdist+wheel, publishes to PyPI, attaches artifacts to a GitHub Release, and
   pushes `ghcr.io/vadim-vyb/acp-openai-gateway:{version,latest}`.

**One-time setup:** enable PyPI **Trusted Publishing** for this repo (Publishing →
add a pending publisher: workflow `release.yml`, environment `release`) — no
secret needed. Or switch the publish step to an API token (`PYPI_API_TOKEN`
secret); see the commented alternative in the workflow. GHCR uses the built-in
`GITHUB_TOKEN`, no setup required.

## Reporting issues

Include your agent + version (e.g. `goose 1.39.0`), the gateway version, and — if
it's a transport problem — a captured `/acp` frame or two. A minimal `curl` that
reproduces it is ideal.
