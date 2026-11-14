"""Offline MemoryStore tests: no credentials, network, media or LLM calls."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import pytest

from pipelines.ffmpeg import memory_store as ms


def lesson(n=0, **meta):
    return {"id": hashlib.sha256(str(n).encode()).hexdigest(), "text": "quiet desert opening observation",
            "infer": False, "metadata": {"outcome": "unknown", "J": None, "cost_status": "unknown", **meta}}


def envelope(**data):
    return {"success": True, "data": data, "error": None}


class Client:
    def __init__(self, states=("queued", "completed")):
        self.context = self
        self.states = iter(states)
        self.calls = []
        self.payload = []
        self.rows = []

    def ingest(self, **kwargs):
        self.calls.append(("ingest", kwargs))
        self.payload = json.loads(kwargs["memories"])
        self.rows = self.payload
        return envelope(results=[{"id": f"source-{i}", "status": "queued", "infer": False} for i, _ in enumerate(self.payload)], failed_count=0)

    def status(self, **kwargs):
        self.calls.append(("status", kwargs))
        state = next(self.states, "queued")
        return envelope(statuses=[{"id": i, "indexing_status": state, "success": True} for i in kwargs["ids"]])

    def query(self, **kwargs):
        self.calls.append(("query", kwargs))
        return envelope(chunks=[{"id": f"source-{i}", "chunk_uuid": "not-lesson-id", "chunk_content": row["text"], "metadata": row["metadata"], "collection": kwargs["collection"]} for i, row in enumerate(self.rows[:5])])


@pytest.fixture(autouse=True)
def no_key(monkeypatch):
    monkeypatch.delenv("HYDRADB_API_KEY", raising=False)


def test_default_and_explicit_missing_key(tmp_path):
    for backend in ("auto", "local", "hydradb"):
        store = ms.select_store(tmp_path / "lessons.jsonl", backend)
        assert store.backend == "local"
        assert store.put_many([lesson()])["status"] == "stored"
        assert store.query("quiet") == [lesson()]
    with pytest.raises(ValueError):
        ms.select_store(tmp_path / "memory", "typo")


def test_local_dedupe_unknowns_keyword_scoping_and_limits(tmp_path):
    store = ms.LocalJsonlStore(tmp_path / "episodes/_memory/lessons.jsonl")
    hash_a, hash_b = "a" * 64, "b" * 64
    a, b = lesson(1, footage_hash=hash_a), lesson(2, footage_hash=hash_b)
    b["text"] = "fast city action"
    assert store.put_many([a, a, b])["status"] == "stored"
    store.put_many([a])
    assert len(store.path.read_text().splitlines()) == 2
    assert store.query("city")[0] == b
    assert store.query("city", footage_hash=hash_a) == [a]
    assert store.query("quiet", 0) == []
    assert store.query("quiet", -2) == []
    assert store.query("quiet", footage_hash="c" * 64) == []
    assert store.query("quiet")[0]["metadata"]["J"] is None
    assert store.query("quiet")[0]["metadata"]["outcome"] == "unknown"


def test_concurrent_writes_complete_and_deduplicated(tmp_path):
    path = tmp_path / "lessons.jsonl"
    with ThreadPoolExecutor(max_workers=3) as pool:
        statuses = list(pool.map(lambda i: ms.LocalJsonlStore(path).put_many([lesson(i), lesson(99)]), range(20)))
    assert all(s["status"] == "stored" for s in statuses)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 21
    assert len({r["id"] for r in rows}) == 21
    assert not list(tmp_path.glob(".lessons-*"))


@pytest.mark.parametrize("target", ["file", "parent", "lock"])
def test_symlinks_rejected(tmp_path, target):
    outside = tmp_path / "outside"
    outside.mkdir()
    victim = outside / "victim"
    victim.write_text("unchanged")
    path = tmp_path / "lessons.jsonl"
    if target == "file":
        path.symlink_to(victim)
    elif target == "lock":
        path.with_suffix(".jsonl.lock").symlink_to(victim)
    else:
        parent = tmp_path / "link"
        parent.symlink_to(outside, target_is_directory=True)
        path = parent / "victim"
    store = ms.LocalJsonlStore(path)
    assert store.put_many([lesson()])["status"] == "unavailable"
    assert victim.read_text() == "unchanged"
    if target != "lock":
        assert store.query("quiet") == []


@pytest.mark.parametrize("content", [b"not json\n", b'{}\n', b'{}', b'\n'])
def test_corruption_fail_open_without_overwrite(tmp_path, content):
    path = tmp_path / "lessons.jsonl"
    path.write_bytes(content)
    store = ms.LocalJsonlStore(path)
    assert store.query("quiet") == []
    assert store.put_many([lesson()])["status"] == "unavailable"
    assert path.read_bytes() == content


def test_bounded_rows_file_batch_and_lock(tmp_path, monkeypatch):
    path = tmp_path / "lessons.jsonl"
    store = ms.LocalJsonlStore(path)
    assert store.put_many([lesson(i) for i in range(ms.MAX_BATCH + 1)])["status"] == "no_lessons"
    monkeypatch.setattr(ms, "MAX_ROWS", 1)
    store.put_many([lesson()])
    assert store.put_many([lesson(1)])["status"] == "unavailable"
    monkeypatch.setattr(ms, "MAX_FILE_BYTES", 10)
    assert store.query("quiet") == []
    monkeypatch.setattr(ms, "MAX_FILE_BYTES", 8 * 1024 * 1024)
    lock = os.open(str(path) + ".lock", os.O_RDWR)
    try:
        ms.fcntl.flock(lock, ms.fcntl.LOCK_EX)
        assert store.put_many([lesson()])["status"] == "unavailable"
    finally:
        os.close(lock)


def test_payload_allowlist_exact_text_and_no_inference(tmp_path):
    client = Client()
    store = ms.select_store(tmp_path / "lessons.jsonl", client=client, database="demo", collection="custom")
    row = lesson(episode_id="ep-1", changes={"opening": "shorter"}, deltas={"Q": None})
    row["metadata"].update(raw_transcript="PRIVATE", source_path="/private/video.mp4", api_key="PRIVATE")
    row["unknown"] = "PRIVATE"
    status = store.put_many([row])
    assert status == {"backend": "hydradb", "status": "stored", "limitations": []}
    payload = client.payload[0]
    assert payload["text"] == row["text"]
    assert payload["infer"] is False
    assert payload["metadata"]["lesson_id"] == row["id"]
    assert "PRIVATE" not in json.dumps(payload)
    assert all(k["database"] == "demo" and k["collection"] == "custom" for _, k in client.calls)
    assert all(k["request_options"] == {"timeout_in_seconds": 1, "max_retries": 0} for _, k in client.calls)
    store.put_many([row])
    assert len([c for c in client.calls if c[0] == "ingest"]) == 1
    rows = store.query("quiet")
    assert rows[0]["id"] == row["id"]
    assert "lesson_id" not in rows[0]["metadata"]
    assert client.calls[-1][1]["response_format"] == "structured"
    assert client.calls[-1][1]["query_by"] == "text"


@pytest.mark.parametrize("patch", [
    {"infer": True}, {"id": "bad"}, {"text": "x" * 501}, {"text": "raw /private/video.mp4"},
    {"metadata": {"goal": "sk-secretvalue"}}, {"metadata": {"outcome": "great"}},
    {"metadata": {"deltas": {"secret": "hidden"}}}, {"metadata": {"footage_hash": "bad"}},
    {"metadata": {"start_ms": -1}}, {"metadata": {"J": float("nan")}},
])
def test_unsafe_lessons_never_remote(tmp_path, patch):
    client = Client()
    store = ms.HydraDBStore(tmp_path / "lessons.jsonl", client=client)
    row = {**lesson(), **patch}
    status = store.put_many([row])
    assert status["status"] == "no_lessons"
    assert "invalid_lessons_rejected" in status["limitations"]
    assert not client.calls


@pytest.mark.parametrize("state", ["failed", "errored", "queued", "processing", "garbage"])
def test_index_failure_or_pending_falls_back_bounded(tmp_path, state):
    client = Client((state,) * 10)
    store = ms.HydraDBStore(tmp_path / "lessons.jsonl", client=client)
    assert store.put_many([lesson()])["status"] == "fallback"
    assert len([c for c in client.calls if c[0] == "status"]) <= ms.MAX_POLLS
    assert store.local.query("quiet") == [lesson()]


def test_failed_index_status_not_deprecated_success(tmp_path):
    client = Client()
    client.status = lambda **kw: envelope(statuses=[{"id": "source-0", "indexing_status": "failed", "success": True}])
    store = ms.HydraDBStore(tmp_path / "memory", client=client)
    assert store.put_many([lesson()])["status"] == "fallback"


@pytest.mark.parametrize("response", [None, {}, {"success": False, "data": {}},
    envelope(chunks="bad"), envelope(chunks=[{"chunk_content": "quiet", "metadata": {}}]),
    envelope(chunks=[{"chunk_content": "quiet", "metadata": {"lesson_id": "a" * 64, "api_key": "hidden"}}]),
    envelope(chunks=[{"chunk_content": "quiet", "metadata": {"lesson_id": "a" * 64}, "collection": "wrong"}]),
])
def test_malformed_remote_query_rejected(tmp_path, response):
    client = Client()
    client.query = lambda **kw: response
    store = ms.HydraDBStore(tmp_path / "memory", client=client)
    store.local.put_many([lesson()])
    assert store.query("quiet") == [lesson()]
    assert store.last_status["status"] == "fallback"


def test_remote_dedupe_and_exact_footage_postfilter(tmp_path):
    client = Client()
    a, b = lesson(1, footage_hash="a" * 64), lesson(2, footage_hash="b" * 64)
    client.rows = [{**r, "metadata": {**r["metadata"], "lesson_id": r["id"]}} for r in (a, a, b)]
    store = ms.HydraDBStore(tmp_path / "memory", client=client)
    assert store.query("quiet", footage_hash="a" * 64) == [a]
    assert len(store.query("quiet")) == 2


def test_missing_sdk_lazy_and_no_key_leak(tmp_path, monkeypatch):
    monkeypatch.setenv("HYDRADB_API_KEY", "private-key-value")
    called = []
    def missing(name):
        called.append(name)
        raise ImportError("private-key-value")
    monkeypatch.setattr(ms.importlib, "import_module", missing)
    store = ms.select_store(tmp_path / "memory")
    assert called == []
    status = store.put_many([lesson()])
    assert called == ["hydra_db"]
    assert status["status"] == "fallback"
    assert "private-key-value" not in json.dumps(status)
    assert store.query("quiet") == [lesson()]
    row = lesson(1)
    row["text"] = "private-key-value"
    store.put_many([row])
    assert len(store.local.query("")) == 1


def test_sdk_constructor_verified_options(tmp_path, monkeypatch):
    client = Client()
    calls = []
    monkeypatch.setenv("HYDRADB_API_KEY", "fake-test-key")
    def constructor(**kw):
        calls.append(kw)
        return client
    monkeypatch.setattr(ms.importlib, "import_module", lambda name: SimpleNamespace(HydraDB=constructor))
    store = ms.select_store(tmp_path / "memory")
    assert store.put_many([lesson()])["backend"] == "hydradb"
    assert calls == [{"token": "fake-test-key", "timeout": 1.0, "logging": {"silent": True}}]


def test_hung_sdk_daemon_deadline_and_global_capacity(tmp_path, monkeypatch):
    monkeypatch.setattr(ms, "OP_TIMEOUT", 0.05)
    # Default argument uses original value; replace boundary with short wrapper.
    bounded = ms._bounded
    monkeypatch.setattr(ms, "_bounded", lambda fn: bounded(fn, timeout=0.05))
    monkeypatch.setattr(ms, "_WORKERS", threading.BoundedSemaphore(2))
    release = threading.Event()
    entered = threading.Event()
    workers = []
    def hang(**kw):
        workers.append(threading.current_thread())
        entered.set()
        release.wait(5)
        raise RuntimeError("secret-payload")
    client = Client()
    client.query = hang
    try:
        for n in range(3):
            store = ms.HydraDBStore(tmp_path / str(n), client=client)
            start = time.monotonic()
            assert store.query("quiet") == []
            assert time.monotonic() - start < 0.3
            assert "secret-payload" not in json.dumps(store.last_status)
        assert entered.is_set()
        assert len(workers) == 2
        assert all(w.daemon for w in workers)
    finally:
        release.set()
        for w in workers:
            w.join(1)


def test_malformed_ingestion_and_api_exception_fallback(tmp_path):
    client = Client()
    store = ms.HydraDBStore(tmp_path / "memory", client=client)
    client.ingest = lambda **kw: envelope(results=[{"id": "source", "status": "queued", "infer": True}])
    assert store.put_many([lesson()])["status"] == "fallback"
    def fail(**kw):
        raise RuntimeError("PRIVATE TOKEN /raw/media.mp4")
    client.query = fail
    assert store.query("quiet") == [lesson()]
    assert "PRIVATE" not in json.dumps(store.last_status)

@pytest.mark.parametrize("text", ["(\"/private\")", "ghp_privatecredential", "github_pat_private", "Bearer xyz", "-----BEGIN RSA PRIVATE KEY-----"])
def test_common_credentials_and_quoted_paths_rejected(tmp_path, text):
    client = Client()
    row = {**lesson(), "text": text}
    store = ms.HydraDBStore(tmp_path / "memory", client=client)
    assert store.put_many([row])["status"] == "no_lessons"
    assert client.calls == []


def test_other_configured_secret_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("OTHER_PROVIDER_API_KEY", "some-other-private-value")
    row = lesson(goal="some-other-private-value")
    assert ms.LocalJsonlStore(tmp_path / "memory").put_many([row])["status"] == "no_lessons"


def test_model_response_conversion_and_empty_remote_fallback(tmp_path):
    class Model:
        def __init__(self, value): self.value = value
        def model_dump(self): return self.value
    client = Client()
    row = lesson()
    client.query = lambda **kw: Model(envelope(chunks=[{"chunk_content": row["text"], "metadata": {**row["metadata"], "lesson_id": row["id"]}}]))
    store = ms.HydraDBStore(tmp_path / "memory", client=client)
    assert store.query("quiet") == [row]
    assert store.last_status["backend"] == "hydradb"
    store.local.put_many([row])
    client.query = lambda **kw: envelope(chunks=[])
    assert store.query("quiet") == [row]
    assert store.last_status["limitations"] == ["no_remote_lessons"]


def test_local_nonregular_fifo_never_blocks(tmp_path):
    path = tmp_path / "memory"
    os.mkfifo(path)
    start = time.monotonic()
    store = ms.LocalJsonlStore(path)
    assert store.query("quiet") == []
    assert store.put_many([lesson()])["status"] == "unavailable"
    assert time.monotonic() - start < 0.3


def test_late_sdk_completion_does_not_start_status_calls(tmp_path, monkeypatch):
    monkeypatch.setattr(ms, "OP_TIMEOUT", 0.03)
    client = Client()
    released = threading.Event()
    def late_ingest(**kw):
        released.wait(1)
        return envelope(results=[{"id": "source-0", "status": "queued", "infer": False}])
    client.ingest = late_ingest
    store = ms.HydraDBStore(tmp_path / "memory", client=client)
    try:
        assert store.put_many([lesson()])["status"] == "fallback"
    finally:
        released.set()
    time.sleep(0.03)
    assert not client.calls
