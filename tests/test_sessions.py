from acp_openai_gateway.sessions import SessionStore, transcript_hash


def test_transcript_hash_stable_and_content_sensitive():
    a = [{"role": "user", "content": "hi"}]
    b = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]  # same text, list form
    c = [{"role": "user", "content": "bye"}]
    assert transcript_hash(a) == transcript_hash(b)  # normalization collapses forms
    assert transcript_hash(a) != transcript_hash(c)


async def test_continuation_roundtrip():
    store = SessionStore("")  # in-memory
    # turn 1: fresh conversation, no prior -> no match
    m1 = [{"role": "user", "content": "hello"}]
    assert await store.lookup(m1[:-1]) is None
    await store.remember(m1 + [{"role": "assistant", "content": "hi there"}], "sess-1")

    # turn 2: client resends history + new user msg; prior must resolve to sess-1
    m2 = m1 + [{"role": "assistant", "content": "hi there"}, {"role": "user", "content": "again"}]
    assert await store.lookup(m2[:-1]) == "sess-1"


async def test_persistence_survives_new_instance(tmp_path):
    path = str(tmp_path / "state.json")
    s1 = SessionStore(path)
    msgs = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
    await s1.remember(msgs, "sess-9")

    s2 = SessionStore(path)  # simulate restart
    prior = msgs + [{"role": "user", "content": "next"}]
    assert await s2.lookup(prior[:-1]) == "sess-9"


async def test_forget_removes_all_keys_for_session(tmp_path):
    store = SessionStore(str(tmp_path / "s.json"))
    await store.remember([{"role": "user", "content": "x"}], "dead")
    await store.forget("dead")
    assert await store.lookup([{"role": "user", "content": "x"}]) is None
