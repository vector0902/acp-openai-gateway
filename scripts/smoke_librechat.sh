#!/usr/bin/env bash
# Verify a full chat through a running LibreChat instance that has the gateway
# configured as a custom endpoint (librechat.yaml: endpoints.custom → baseURL of
# this gateway). This drives LibreChat's real API the way its frontend does.
#
#   LIBRECHAT_URL=http://localhost:3080 LC_ENDPOINT=goose-gateway \
#   LC_MODEL=goose-1.39.0 scripts/smoke_librechat.sh
#
# Notes learned the hard way (all LibreChat-side, none require gateway changes):
#   - chat route is POST /api/agents/chat/:endpoint (not the old /api/ask/*)
#   - a browser User-Agent is required (uaParser rejects otherwise)
#   - generation is async: POST returns a streamId; the reply is saved to the
#     conversation, in message.content[].text (message.text may be empty)
set -euo pipefail

LC="${LIBRECHAT_URL:-http://localhost:3080}"
EP="${LC_ENDPOINT:-goose-gateway}"
MODEL="${LC_MODEL:-goose-1.39.0}"
UA="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120 Safari/537.36"

python3 - "$LC" "$EP" "$MODEL" "$UA" <<'PY'
import json, sys, time, uuid, urllib.request, urllib.error
LC, EP, MODEL, UA = sys.argv[1:5]

def call(path, data=None, token=None, stream=False):
    hdrs = {"Content-Type": "application/json", "User-Agent": UA}
    if token: hdrs["Authorization"] = "Bearer " + token
    if stream: hdrs["Accept"] = "text/event-stream"
    req = urllib.request.Request(
        LC + path, data=json.dumps(data).encode() if data is not None else None,
        headers=hdrs, method="POST" if data is not None else "GET")
    try:
        r = urllib.request.urlopen(req, timeout=120)
        return r.status, json.loads(r.read() or b"null")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:200]

u = {"name": "Smoke", "email": "smoke@example.com", "username": "smoke",
     "password": "Test1234!", "confirm_password": "Test1234!"}
call("/api/auth/register", u)
_, login = call("/api/auth/login", {"email": u["email"], "password": u["password"]})
token = login["token"] if isinstance(login, dict) else None
assert token, f"login failed: {login}"

body = {"text": "Reply with exactly: LIBRECHAT_OK", "conversationId": None,
        "parentMessageId": "00000000-0000-0000-0000-000000000000",
        "messageId": str(uuid.uuid4()), "model": MODEL, "endpoint": EP,
        "endpointType": "custom", "ephemeralAgent": {"mcp": [], "execute_code": False,
        "web_search": False}, "isCreatedByUser": True, "sender": "User"}
st, started = call(f"/api/agents/chat/{EP}", body, token=token, stream=True)
assert st == 200 and isinstance(started, dict), f"chat start failed: {st} {started}"
conv = started["conversationId"]
print(f"started conversation {conv}")

for _ in range(30):
    time.sleep(1)
    _, msgs = call(f"/api/messages/{conv}", token=token)
    if not isinstance(msgs, list):
        continue
    for m in msgs:
        if not m.get("isCreatedByUser"):
            text = m.get("text") or "".join(
                b.get("text", "") for b in (m.get("content") or []) if isinstance(b, dict))
            if text and not m.get("unfinished", False):
                assert not m.get("error"), f"assistant error: {m}"
                print(f"assistant replied: {text!r}")
                print("LIBRECHAT: PASS")
                sys.exit(0)
sys.exit("LIBRECHAT: FAIL (no assistant content)")
PY
