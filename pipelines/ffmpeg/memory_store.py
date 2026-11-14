"""Optional, fail-open lesson persistence. No inference, provisioning or model calls."""
from __future__ import annotations

import fcntl
import importlib
import json
import math
import os
from pathlib import Path
import re
import stat
import threading
import time
from typing import Any, Callable, Protocol
import uuid

MAX_FILE_BYTES = 8 * 1024 * 1024
MAX_ROWS = 5000
MAX_ROW_BYTES = 16384
MAX_BATCH = 128
OP_TIMEOUT = 2.0
LOCK_TIMEOUT = 0.25
MAX_POLLS = 3
_WORKERS = threading.BoundedSemaphore(4)
_FIELDS = frozenset("episode_id goal footage_hash J Q cost cost_status rubric_scores decider outcome kind source_id start_ms end_ms opening_types evaluation_id method_version judge_model judge_provider changes deltas config_hash rationale_status rubric_evidence known_cost selection_plan_hash paired_episode_id paired_evaluation_id".split())
_ID = re.compile(r"[0-9a-f]{64}\Z")
_UNSAFE_KEY = re.compile(r"path|secret|token|password|credential|api.?key|transcript|media", re.I)
_UNSAFE_TEXT = re.compile(r"(?:https?://|file://|(?:^|[\s(\"'])(?:/|~/|[A-Za-z]:\\)|[\w.-]+[/\\][\w.-]+|\b(?:sk-|gh[pousr]_|github_pat_|xox[baprs]-|AKIA[0-9A-Z]{16}|Bearer\s)|-----BEGIN .*PRIVATE KEY|\b(?:api[_ -]?key|password|secret|token)\s*[:=])", re.I)


class MemoryStore(Protocol):
    backend: str
    last_status: dict

    def put_many(self, lessons: list[dict]) -> dict: ...
    def query(self, query: str, limit: int = 5, footage_hash: str | None = None) -> list[dict]: ...


class _Unavailable(Exception):
    pass


def _bounded(fn: Callable[[], Any], timeout: float | None = None) -> Any:
    """A hung SDK cannot delay caller shutdown or spawn unbounded workers."""
    if timeout is None:
        timeout = OP_TIMEOUT
    if not _WORKERS.acquire(blocking=False):
        raise _Unavailable("worker_capacity")
    done = threading.Event()
    result: list[Any] = []

    def run() -> None:
        try:
            result.append((True, fn()))
        except Exception:
            # Never retain/emit exception text: providers may include keys/payloads.
            result.append((False, None))
        finally:
            _WORKERS.release()
            done.set()

    threading.Thread(target=run, daemon=True, name="lesson-store").start()
    if not done.wait(timeout):
        raise _Unavailable("operation_timeout")
    if not result[0][0]:
        raise _Unavailable("operation_failed")
    return result[0][1]


def _safe_string(value: str) -> bool:
    secrets = [v for k, v in os.environ.items() if _UNSAFE_KEY.search(k) and len(v) >= 8]
    return (len(value) <= 500 and not _UNSAFE_TEXT.search(value)
            and not any(ord(c) < 32 and c not in "\n\t" for c in value)
            and not any(secret in value for secret in secrets))


def _safe_value(value: Any, depth: int = 0, budget: list[int] | None = None) -> bool:
    if budget is None:
        budget = [512]
    budget[0] -= 1
    if depth > 4 or budget[0] < 0:
        return False
    if value is None or isinstance(value, bool):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(value) and abs(value) < 1e15
    if isinstance(value, str):
        return _safe_string(value)
    if isinstance(value, list):
        return len(value) <= 32 and all(_safe_value(v, depth + 1, budget) for v in value)
    if isinstance(value, dict):
        return len(value) <= 32 and all(
            isinstance(k, str) and len(k) <= 64 and not _UNSAFE_KEY.search(k)
            and _safe_string(k) and _safe_value(v, depth + 1, budget) for k, v in value.items())
    return False


def _lesson(row: Any, *, strict: bool = False) -> dict:
    if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not _ID.fullmatch(row["id"]):
        raise ValueError("invalid lesson")
    if row.get("infer") is not False or not isinstance(row.get("text"), str) or not row["text"].strip() or not _safe_string(row["text"]):
        raise ValueError("invalid lesson text")
    meta = row.get("metadata")
    if not isinstance(meta, dict) or len(meta) > 64:
        raise ValueError("invalid metadata")
    if strict and (set(row) - {"id", "text", "infer", "metadata"} or set(meta) - _FIELDS):
        raise ValueError("unexpected fields")
    meta = {k: v for k, v in meta.items() if k in _FIELDS}
    if not _safe_value(meta) or meta.get("outcome", "unknown") not in {"helped", "hurt", "unknown"}:
        raise ValueError("unsafe metadata")
    for field in ("footage_hash", "config_hash"):
        if meta.get(field) is not None and (not isinstance(meta[field], str) or not _ID.fullmatch(meta[field])):
            raise ValueError("invalid hash")
    for field in ("start_ms", "end_ms"):
        if meta.get(field) is not None and (type(meta[field]) is not int or meta[field] < 0):
            raise ValueError("invalid interval")
    if meta.get("start_ms") is not None and meta.get("end_ms") is not None and meta["end_ms"] <= meta["start_ms"]:
        raise ValueError("invalid interval")
    result = {"id": row["id"], "text": row["text"], "infer": False, "metadata": meta}
    if len(json.dumps(result, allow_nan=False).encode()) > MAX_ROW_BYTES:
        raise ValueError("oversized lesson")
    return result


def _batch(lessons: list[dict]) -> tuple[list[dict], list[str]]:
    if not isinstance(lessons, list) or len(lessons) > MAX_BATCH:
        return [], ["invalid_or_oversized_batch"]
    rows: dict[str, dict] = {}
    rejected = False
    for row in lessons:
        try:
            valid = _lesson(row)
            rows.setdefault(valid["id"], valid)
        except (ValueError, TypeError, OverflowError):
            rejected = True
    return list(rows.values()), ["invalid_lessons_rejected"] if rejected else []


def _limit(limit: int) -> int:
    return max(0, min(5, limit)) if type(limit) is int else 5


def _status(backend: str, status: str, limitations: list[str] | None = None) -> dict:
    return {"backend": backend, "status": status, "limitations": limitations or []}


class LocalJsonlStore:
    backend = "local"

    def __init__(self, path: Path):
        self.path = Path(path).absolute()
        self.last_status = _status(self.backend, "ready")

    def _directory(self) -> int:
        # Walk using pinned dirfds: reject symlinks in every parent and leaf.
        fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
        try:
            for part in self.path.parent.parts[1:]:
                if part in {".", ".."}:
                    raise ValueError("unsafe path")
                try:
                    os.mkdir(part, mode=0o700, dir_fd=fd)
                except FileExistsError:
                    pass
                child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
                os.close(fd)
                fd = child
            return fd
        except Exception:
            os.close(fd)
            raise

    def _read(self, directory: int) -> list[dict]:
        try:
            fd = os.open(self.path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
        except FileNotFoundError:
            return []
        with os.fdopen(fd, "rb") as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_FILE_BYTES:
                raise ValueError("unsafe memory file")
            data = stream.read(MAX_FILE_BYTES + 1)
        if len(data) > MAX_FILE_BYTES or not data.endswith(b"\n") and data:
            raise ValueError("invalid memory file")
        lines = data.splitlines()
        if len(lines) > MAX_ROWS:
            raise ValueError("too many rows")
        rows: dict[str, dict] = {}
        for line in lines:
            if len(line) > MAX_ROW_BYTES:
                raise ValueError("oversized row")
            row = _lesson(json.loads(line), strict=True)
            rows.setdefault(row["id"], row)
        return list(rows.values())

    def _write(self, rows: list[dict]) -> None:
        directory = self._directory()
        lock = None
        temp = None
        try:
            lock = os.open(self.path.name + ".lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600, dir_fd=directory)
            if not stat.S_ISREG(os.fstat(lock).st_mode):
                raise ValueError("unsafe lock")
            deadline = time.monotonic() + LOCK_TIMEOUT
            while True:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("local lock")
                    time.sleep(0.005)
            existing = {row["id"]: row for row in self._read(directory)}
            for row in rows:
                existing.setdefault(row["id"], row)
            if len(existing) > MAX_ROWS:
                raise ValueError("too many rows")
            data = b"".join((json.dumps(row, sort_keys=True, allow_nan=False) + "\n").encode() for row in existing.values())
            if len(data) > MAX_FILE_BYTES:
                raise ValueError("oversized file")
            temp = ".lessons-" + uuid.uuid4().hex
            fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=directory)
            with os.fdopen(fd, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp, self.path.name, src_dir_fd=directory, dst_dir_fd=directory)
            temp = None
            os.fsync(directory)
        finally:
            if temp:
                os.unlink(temp, dir_fd=directory)
            if lock is not None:
                os.close(lock)
            os.close(directory)

    def put_many(self, lessons: list[dict]) -> dict:
        rows, limitations = _batch(lessons)
        try:
            if rows:
                _bounded(lambda: self._write(rows))
            self.last_status = _status(self.backend, "stored" if rows else "no_lessons", limitations)
        except _Unavailable as exc:
            self.last_status = _status(self.backend, "unavailable", limitations + [str(exc)])
        return self.last_status

    def query(self, query: str, limit: int = 5, footage_hash: str | None = None) -> list[dict]:
        def read() -> list[dict]:
            directory = self._directory()
            try:
                return self._read(directory)
            finally:
                os.close(directory)
        try:
            rows = _bounded(read)
            terms = set(re.findall(r"\w+", query.lower()[:500])) if isinstance(query, str) else set()
            # Unknown outcomes/scores remain observations, not assumed successes.
            rows = [r for r in rows if footage_hash is None or r["metadata"].get("footage_hash") == footage_hash]
            def rank(row: dict) -> tuple:
                words = set(re.findall(r"\w+", (row["text"] + " " + str(row["metadata"].get("goal", ""))).lower()))
                return (-len(terms & words), row["id"])
            self.last_status = _status(self.backend, "queried")
            return sorted(rows, key=rank)[:_limit(limit)]
        except _Unavailable as exc:
            self.last_status = _status(self.backend, "unavailable", [str(exc)])
            return []


def _object(value: Any) -> dict:
    if isinstance(value, dict):
        return value
    if callable(getattr(value, "model_dump", None)):
        return value.model_dump()
    if callable(getattr(value, "dict", None)):
        return value.dict()
    raise ValueError("malformed response")


def _data(response: Any) -> dict:
    envelope = _object(response)
    if envelope.get("success") is not True or envelope.get("error"):
        raise ValueError("provider failure")
    return _object(envelope.get("data"))


class HydraDBStore:
    backend = "hydradb"

    def __init__(self, path: Path, database: str = "framefoundry", collection: str = "edit-lessons", client: Any = None):
        self.local = LocalJsonlStore(path)
        self.database, self.collection = database, collection
        self._client = client
        self._synced: set[str] = set()
        self._guard = threading.Lock()
        self.last_status = _status(self.backend, "ready")

    def _get_client(self) -> Any:
        if self._client is None:
            key = os.environ.get("HYDRADB_API_KEY")
            if not key:
                raise ValueError("missing credentials")
            module = importlib.import_module("hydra_db")
            self._client = module.HydraDB(token=key, timeout=1.0, logging={"silent": True})
        return self._client

    def _scope(self) -> dict:
        return {"database": self.database, "collection": self.collection,
                "request_options": {"timeout_in_seconds": 1, "max_retries": 0}}

    def put_many(self, lessons: list[dict]) -> dict:
        rows, limitations = _batch(lessons)
        local_status = self.local.put_many(rows)
        deadline = time.monotonic() + OP_TIMEOUT
        def ingest() -> str:
            if not self._guard.acquire(blocking=False):
                raise ValueError("provider busy")
            try:
                pending = [r for r in rows if r["id"] not in self._synced]
                if not pending:
                    return "stored" if rows else "no_lessons"
                client = self._get_client()
                payload = [{**r, "metadata": {**r["metadata"], "lesson_id": r["id"]}} for r in pending]
                if time.monotonic() >= deadline:
                    raise TimeoutError("deadline")
                data = _data(client.context.ingest(type="memory", memories=json.dumps(payload, allow_nan=False), **self._scope()))
                items = data.get("results")
                if not isinstance(items, list) or len(items) != len(pending) or data.get("failed_count", 0):
                    raise ValueError("invalid ingestion")
                ids = []
                complete = True
                for item in items:
                    item = _object(item)
                    if item.get("error") or item.get("infer") is not False or item.get("status") not in {"queued", "processing", "completed"} or not isinstance(item.get("id"), str) or not item["id"]:
                        raise ValueError("invalid ingestion item")
                    ids.append(item["id"])
                    complete &= item["status"] == "completed"
                if len(set(ids)) != len(ids):
                    raise ValueError("duplicate source IDs")
                for _ in range(MAX_POLLS):
                    if complete:
                        break
                    if time.monotonic() >= deadline:
                        raise TimeoutError("deadline")
                    statuses = _data(client.context.status(ids=ids, **self._scope())).get("statuses")
                    if not isinstance(statuses, list) or len(statuses) != len(ids):
                        raise ValueError("invalid status")
                    by_id = {_object(s).get("id"): _object(s) for s in statuses}
                    if set(by_id) != set(ids):
                        raise ValueError("status IDs mismatch")
                    states = [by_id[i].get("indexing_status") for i in ids]
                    if any(s not in {"queued", "processing", "completed"} for s in states) or any(s.get("error_code") or s.get("error_message") for s in by_id.values()):
                        raise ValueError("indexing failed")
                    complete = all(s == "completed" for s in states)
                    if not complete:
                        time.sleep(0.05)
                if not complete:
                    raise ValueError("indexing pending")
                self._synced.update(r["id"] for r in pending)
                return "stored"
            finally:
                self._guard.release()
        try:
            status = _bounded(ingest)
            self.last_status = _status(self.backend, status, limitations + local_status["limitations"])
        except _Unavailable as exc:
            self.last_status = _status("local", "fallback", limitations + local_status["limitations"] + [str(exc), "hydradb_unavailable"])
        return self.last_status

    def query(self, query: str, limit: int = 5, footage_hash: str | None = None) -> list[dict]:
        limit = _limit(limit)
        if not limit:
            return []
        def retrieve() -> list[dict]:
            if not isinstance(query, str) or not _safe_string(query):
                raise ValueError("unsafe query")
            if not self._guard.acquire(blocking=False):
                raise ValueError("provider busy")
            try:
                data = _data(self._get_client().query(query=query, type="memory", query_by="text", response_format="structured", max_results=5, graph_context=False, temporal_reasoning=False, query_forceful_relations=False, query_apps=False, **self._scope()))
                chunks = data.get("chunks")
                if not isinstance(chunks, list) or len(chunks) > 5:
                    raise ValueError("invalid chunks")
                results: dict[str, dict] = {}
                for chunk in chunks:
                    chunk = _object(chunk)
                    if chunk.get("infer", False) is not False:
                        raise ValueError("inferred result")
                    meta = _object(chunk.get("metadata"))
                    lesson_id = meta.get("lesson_id")
                    # Do not treat chunk IDs or provider source IDs as lesson IDs.
                    row = _lesson({"id": lesson_id, "text": chunk.get("chunk_content"), "infer": False,
                                   "metadata": {k: v for k, v in meta.items() if k != "lesson_id"}}, strict=True)
                    if chunk.get("collection") not in {None, self.collection}:
                        raise ValueError("wrong collection")
                    if footage_hash is None or row["metadata"].get("footage_hash") == footage_hash:
                        results.setdefault(row["id"], row)
                return list(results.values())[:limit]
            finally:
                self._guard.release()
        try:
            rows = _bounded(retrieve)
            if rows:
                self.last_status = _status(self.backend, "queried")
                return rows
            reason = "no_remote_lessons"
        except _Unavailable as exc:
            reason = str(exc)
        rows = self.local.query(query, limit, footage_hash)
        self.last_status = _status("local", "fallback", [reason] + self.local.last_status["limitations"])
        return rows


def select_store(path: Path, backend: str = "auto", database: str = "framefoundry", collection: str = "edit-lessons", client: Any = None) -> MemoryStore:
    """Path is the JSONL file (normally episodes/_memory/lessons.jsonl)."""
    if backend not in {"auto", "local", "hydradb"}:
        raise ValueError("unknown memory backend")
    if backend == "local":
        return LocalJsonlStore(path)
    if client is not None or os.environ.get("HYDRADB_API_KEY"):
        return HydraDBStore(path, database, collection, client)
    store = LocalJsonlStore(path)
    store.last_status = _status("local", "ready", ["hydradb_credentials_absent"])
    return store
