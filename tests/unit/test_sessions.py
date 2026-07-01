"""Unit tests for conversation→session continuity, including property-based
invariants for the transcript hash."""

import pytest
from hypothesis import given
from hypothesis import strategies as st

from acp_openai_gateway.sessions import SessionStore, transcript_hash

pytestmark = pytest.mark.unit


def test_transcript_hash_is_form_insensitive():
    # str content and the equivalent list-of-parts content hash identically
    a = [{"role": "user", "content": "hi"}]
    b = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
    assert transcript_hash(a) == transcript_hash(b)


def test_transcript_hash_is_content_sensitive():
    a = [{"role": "user", "content": "hi"}]
    c = [{"role": "user", "content": "bye"}]
    assert transcript_hash(a) != transcript_hash(c)


def test_transcript_hash_is_order_sensitive():
    a = [{"role": "user", "content": "1"}, {"role": "assistant", "content": "2"}]
    b = [{"role": "assistant", "content": "2"}, {"role": "user", "content": "1"}]
    assert transcript_hash(a) != transcript_hash(b)


# ── property-based invariants ────────────────────────────────────────────────
_messages = st.lists(
    st.fixed_dictionaries(
        {
            "role": st.sampled_from(["user", "assistant", "system"]),
            "content": st.text(),
        }
    ),
    min_size=1,
    max_size=6,
)


@given(_messages)
def test_hash_is_deterministic(messages):
    assert transcript_hash(messages) == transcript_hash(list(messages))


@given(_messages)
def test_hash_is_hex_sha256(messages):
    h = transcript_hash(messages)
    assert len(h) == 64
    assert all(c in "0123456789abcdef" for c in h)


@given(_messages, st.text(min_size=1))
def test_appending_a_message_changes_the_hash(messages, extra):
    extended = messages + [{"role": "user", "content": extra}]
    assert transcript_hash(messages) != transcript_hash(extended)


# ── store behavior ───────────────────────────────────────────────────────────
async def test_fresh_conversation_has_no_prior_match():
    store = SessionStore("")
    assert await store.lookup([]) is None


async def test_continuation_roundtrip():
    store = SessionStore("")
    m1 = [{"role": "user", "content": "hello"}]
    await store.remember(m1 + [{"role": "assistant", "content": "hi there"}], "sess-1")
    # next turn resends history + a new user message; prior must resolve
    m2 = m1 + [
        {"role": "assistant", "content": "hi there"},
        {"role": "user", "content": "again"},
    ]
    assert await store.lookup(m2[:-1]) == "sess-1"


async def test_persistence_survives_new_instance(tmp_path):
    path = str(tmp_path / "state.json")
    msgs = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
    await SessionStore(path).remember(msgs, "sess-9")

    resumed = SessionStore(path)  # simulate a restart
    assert await resumed.lookup(msgs) == "sess-9"


async def test_forget_removes_every_key_for_a_session(tmp_path):
    store = SessionStore(str(tmp_path / "s.json"))
    await store.remember([{"role": "user", "content": "x"}], "dead")
    await store.remember([{"role": "user", "content": "y"}], "dead")
    await store.forget("dead")
    assert await store.lookup([{"role": "user", "content": "x"}]) is None
    assert await store.lookup([{"role": "user", "content": "y"}]) is None


async def test_corrupt_state_file_is_ignored(tmp_path):
    path = tmp_path / "state.json"
    path.write_text("{ not json")
    store = SessionStore(str(path))  # must not raise
    assert await store.lookup([{"role": "user", "content": "x"}]) is None
