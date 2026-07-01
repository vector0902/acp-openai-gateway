# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — Unreleased

### Added
- OpenAI-compatible `POST /v1/chat/completions` (streaming + non-streaming),
  `GET /v1/models`, and `GET /health`.
- Async ACP-over-HTTP client: `initialize`, `session/new`, `session/load`,
  `session/prompt`, `session/set_mode`, with the connection/session header
  routing verified against goose 1.39.0.
- Stateful conversation continuity over the stateless OpenAI protocol via
  transcript hashing, with on-disk persistence and `session/load` resume.
- Auto-approval of `session/request_permission` so headless turns never stall.
- Optional bearer-key auth, configurable model id, thoughts/tool-status
  streaming toggles.
- Docker image, example compose (agent + gateway + Open WebUI), and CI.
- Test suite: unit tests (translation + session logic, incl. Hypothesis
  property tests) and integration tests driving the real `AcpClient` against an
  in-memory fake ACP agent (`httpx.MockTransport`) and the app via FastAPI
  `TestClient`; ~88% coverage, no network needed. Opt-in `live` marker for a
  real agent.
- Release automation: on a `v*` tag, publish to PyPI (Trusted Publishing) and
  push a GHCR image, with a tag/version guard and a GitHub Release.
