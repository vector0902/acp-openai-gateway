#!/usr/bin/env bash
# Drive real third-party OpenAI clients against the gateway.
#
# Prereqs: a running gateway backed by a real ACP agent, reachable at
# $GATEWAY_URL (default http://localhost:8000). Example: point the gateway at a
# `goose serve` instance, then:
#
#   GATEWAY_URL=http://localhost:8000 scripts/smoke_clients.sh
#
# This is what backs the "verified with" claims in the README.
set -euo pipefail

GATEWAY_URL="${GATEWAY_URL:-http://localhost:8000}"
MODEL="${MODEL:-goose}"

echo "== gateway reachable? =="
curl -fsS "${GATEWAY_URL}/v1/models" | head -c 300 ; echo

echo "== aider =="
if ! command -v aider >/dev/null; then
  echo "aider not installed (pip install aider-chat); skipping"
else
  work="$(mktemp -d)"; ( cd "$work" && git init -q . )
  OPENAI_API_BASE="${GATEWAY_URL}/v1" OPENAI_API_KEY="sk-test" AIDER_ANALYTICS=false \
    aider --model "openai/${MODEL}" \
      --message "Reply with exactly: AIDER_OK" \
      --yes-always --no-auto-commits --no-show-model-warnings --no-check-update --map-tokens 0 \
      --subtree-only 2>&1 | tee "$work/out.txt" | tail -20
  grep -q "AIDER_OK" "$work/out.txt" && echo "AIDER: PASS" || { echo "AIDER: FAIL"; exit 1; }
fi

# Run the gated pytest smoke test too, if the dev env is available:
#   CLIENT_SMOKE_BASE_URL="${GATEWAY_URL}/v1" pytest -m client
