# Optional edit-lesson memory (WO-15 provider slice)

## Readiness, not live proof

This provider is ready for **offline/mocked sponsor integration**, not a claim of live
HydraDB operation, measured improvement, or production readiness. No authenticated
HydraDB request, provisioning call, paid run, or language-model call was performed.
The parent WO-15 owns lesson extraction, planner/token bounds and experiment wiring.
This slice owns only the provider, its tests, optional dependency extra and this file.

Memory is opt-in in the parent integration. Until the Mac has credentials, use
`LocalJsonlStore`: embedding-free keyword ranking, deterministic ID tie breaks,
exact optional footage-hash filtering, and preservation of `null` scores and
`unknown` outcomes. No historical observation becomes a safety override.

```python
from pathlib import Path
from pipelines.ffmpeg.memory_store import select_store

store = select_store(Path("episodes/_memory/lessons.jsonl"), backend="auto")
# store.put_many(lessons) -> {backend, status, limitations: list[str]}
# store.query(goal, limit=5, footage_hash=footage_hash) -> list[lesson dict]
# store.last_status reports retrieval limitations; inspect it even for [] results.
```

`path` is the **JSONL file**, not an episode directory. `auto` and `hydradb` both
select local when `HYDRADB_API_KEY` is absent; explicit `local` never loads the SDK.
An injected mock `client` selects HydraDB without credentials. Database and collection
are configurable via `select_store(path, database=..., collection=...)`; defaults
are `framefoundry` / `edit-lessons`. There is no implicit database/schema creation.

For a future separately authorized keyed run, install the optional extra:

```sh
python -m pip install '.[memory]'
```

Supply `HYDRADB_API_KEY` externally; never place it in lessons, fixtures or Git.
Declare lesson metadata (including `lesson_id`) in the existing database's schema
according to HydraDB's docs. Incompatible/unprovisioned schemas and SDK errors fall
back locally. Nested observation maps/schema compatibility are **not live-verified**.

## Exact SDK source verification

On 2026-11-14, downloaded to a temporary directory without dependencies or installation:

```sh
python -m pip download --no-deps 'hydradb-sdk>=2.1.7,<3' -d /tmp/wo15-sdk
python -m zipfile -e /tmp/wo15-sdk/hydradb_sdk-2.1.7-py3-none-any.whl /tmp/wo15-sdk/source
```

Resolved version: **2.1.7**. Wheel SHA-256:
`a5c04f8a6f421157f8d2029c7845733b914741b0c876ad5b424ec1dc3005c002`.
Inspected `hydra_db/client.py`, `context/client.py`, `core/request_options.py`,
and generated ingestion/status/search models. Official references:

- <https://docs.hydradb.com/api-reference/v2/sdks>
- <https://docs.hydradb.com/essentials/v2/memories> (redirects to the unified Context page)

Verified call shapes used here:

```python
from hydra_db import HydraDB
client = HydraDB(token=key, timeout=1.0, logging={"silent": True})
opts = {"timeout_in_seconds": 1, "max_retries": 0}
client.context.ingest(
    type="memory", database=database, collection=collection,
    memories=json.dumps([{ "id": lesson_id, "text": text, "infer": False,
                          "metadata": {**safe_metadata, "lesson_id": lesson_id} }]),
    request_options=opts,
)
client.context.status(database=database, collection=collection,
                      ids=provider_source_ids, request_options=opts)
client.query(database=database, collection=collection, type="memory",
             query=query, query_by="text", response_format="structured",
             max_results=5, graph_context=False, temporal_reasoning=False,
             query_forceful_relations=False, query_apps=False,
             request_options=opts)
```

The constructor is keyword-only: despite positional examples in generated docstrings,
**do not pass positional `"2"`**. The default `api_version` is `"2"`.
`memories` is a JSON **string**, not a Python list. `infer:false` stores exact text
without provider-side lesson extraction. This code never invokes a model client;
provider-internal retrieval implementation/billing remains unverified.

All responses use the `{success, error, data}` envelope. Reject a false/missing
`success`, nonempty `error`, unexpected shape or invalid provenance.

- Ingestion: `data.results[]` has `id`, `status`, `infer`, `error`; states are
  `queued`, `processing`, `completed`, `failed`. `id` is the **provider source ID**.
  Queuing is not indexing success. Require one valid result per submitted lesson.
- Poll: `data.statuses[]` has `id`, `indexing_status`, `error_code`,
  `error_message`. Inspect `indexing_status`, **not the deprecated item `success`**,
  which can be true even on failed indexing. Pass returned source IDs to `ids`.
- Query: structured `data.chunks[]` exposes `chunk_content`, `metadata`, `id`,
  `chunk_uuid`, `collection` and other search fields. The source/chunk identifiers
  are **not** deterministic lesson identifiers. Read `metadata.lesson_id`, validate
  hex64, and return the canonical `{id,text,infer:false,metadata}` lesson shape.
  Never inject `llm_prompt`, adjacent chunks, graph facts or unvalidated metadata.

The SDK's arbitrary JSON `memories` argument does **not** establish that a submitted
`id` survives ingestion/retrieval. `metadata.lesson_id` is therefore required for
remote provenance; missing/stripped identity triggers local fallback. No alternate
invented `query_id` or guessed `status(job_id=...)` API is used.

## Payload and local safety

Lessons require a lower-case hex64 ID, nonempty exact text <=500 characters,
`infer:false`, and an allowlisted metadata map. Outcomes, when supplied, must be
`helped`, `hurt` or `unknown`. Missing scores/outcomes are not guessed. Allowed fields:

`episode_id`, `goal`, `footage_hash`, `J`, `Q`, `cost`, `cost_status`,
`rubric_scores`, `decider`, `outcome`, `kind`, `source_id`, `start_ms`, `end_ms`,
`opening_types`, `evaluation_id`, `method_version`, `judge_model`, `judge_provider`,
`changes`, `deltas`, `config_hash`.

Unknown input fields are removed before persistence/dispatch; malformed or unsafe
lessons are rejected. Remote results with unknown metadata fields are rejected,
not silently trusted. Nested maps, strings, lists and numeric values are bounded;
non-finite scores, invalid hashes and inverted/negative intervals are rejected.
Path/URL/credential-looking text and configured credential values are forbidden even inside
allowlisted fields. Source IDs must be opaque identities, not source paths.
The writer must generate compact observations, **never raw transcripts/media**:
syntactic filtering cannot establish whether arbitrary natural-language prose is
private, so it is not a replacement for upstream compact lesson extraction.
Exceptions, provider error strings, keys and debug payloads are never logged or
copied into status. Limitations are fixed machine-readable codes.

Local writes pin each directory with `O_NOFOLLOW` dirfds, reject symlinks in every
path component/file/lock, lock a stable sidecar with bounded nonblocking `flock`,
deduplicate by ID (first observation wins), write/fsync a private temporary file,
then atomically replace and fsync the directory. Queries see complete old or new
files. Corruption is fail-open and is not overwritten. This uses POSIX semantics
(Linux/Mac), not Windows. `episodes/` is already gitignored; never commit lesson data.

## Explicit bounds and limitations

- <=128 submitted lessons per call; <=5 retrieved lessons. The parent's <=400-token
  selection bound remains independent of the provider's retrieval count bound.
- <=5,000 local rows; <=8 MiB file; <=16 KiB row; bounded nested values. At capacity,
  writes fail open with provenance, **not silent eviction**.
- Local-first write: HydraDB failures retain usable local evidence. Returned status
  reports the actual backend for fallback. Local unavailable means no lessons.
- <=3 status polls (50 ms between pending observations); every SDK request has a 1-second transport timeout,
  zero retries, and an outer 2-second deadline for the whole remote operation.
  A put has separate local and remote bounds (up to about 4 seconds); fallback query
  likewise can spend up to about 4 seconds across remote/local boundaries.
- All operations run on daemon workers with a process-wide cap of four outstanding
  workers. A hung SDK cannot block shutdown or allocate infinite worker threads.
  Calls at capacity return immediately with `worker_capacity` rather than queueing.
  Per-store remote concurrency is also fail-fast; pending indexing falls back.
- Python cannot forcibly kill a hung thread. A timed-out operation may finish later;
  its worker retains a cap slot until completion. Do not use this as a transactional
  cancellation API. There is no `ThreadPoolExecutor` shutdown wait on provider calls.
- Remote deduplication is per-store-instance after confirmed indexing. Local IDs are
  durable/idempotent. Cross-process remote idempotency is **not claimed** because
  preserving caller IDs is not guaranteed by the inspected SDK. Retry after a failed
  or timed-out ingestion can create remote duplicates; query deduplicates lesson IDs.
- Remote footage scoping is enforced by exact **client-side** postfiltering; no guessed
  schema/filter syntax. A top-five remote result set with no scoped match falls back
  to local lessons. There is no embedding dependency or inferred lesson generation.
- No production media render was changed or produced. The repository suite exercises
  synthetic media gates; parent integration and real episode verification remain
  separate gates.

Verification (offline):

```sh
PYTHONPATH=$PWD .venv/bin/python -m pytest tests/test_memory_store.py -q
PYTHONPATH=$PWD .venv/bin/python -m pytest -q
```
