# MWJClient — Benchmark vs PeTTa

`MWJClient` is a drop-in HTTP replacement for the local `PeTTa` (Prolog/Janus) backend,
used inside `PeTTaChainer` to run backward chaining and knowledge-base queries over a
remote MWJ server instead of a local Prolog process.

---

## Architecture overview

```
pettachainer.py  (PeTTaChainer)
       │
       ├── USE_MWJ = False  →  PeTTa          (local Prolog/Janus, in-process)
       └── USE_MWJ = True   →  MWJClient      (HTTP POST to localhost:5001/metta)
```

Both backends expose the same two-method API that `PeTTaChainer` calls:

```python
handler.process_metta_string(query: str) -> list
handler.load_metta_file(file_path: str)  -> list
```

---

## How PeTTaChainer uses the backend

`PeTTaChainer` sits on top of the backend and provides the full reasoning pipeline:

| Method | What it does |
|---|---|
| `add_atom(atom)` | Wraps atom as `!(compileadd kb_id ...)` and sends to backend |
| `query(atom, depth)` | Runs `!(query (fromNumber depth) kb_id atom)` — backward chaining |
| `get_all_facts()` | Matches `(: kb_id $prf $type $tv)` against `&kb` |
| `formatter(minedPatterns)` | Converts mined association patterns into MeTTa rules and inserts them via `add_atom` |
| `handle_why_question(question)` | Full pipeline: facts → LLM rewrites question to canonical MeTTa query → backward chaining → LLM explains result in plain language |

Each `PeTTaChainer` instance gets its own unique `kb_id` (`"kb" + uuid4().hex`) so
multiple instances never collide in the shared `&kb` on the MWJ server.

---

## Prerequisites — MWJ server (Docker)

MWJClient requires the MWJ server running locally. Start it with:

```bash
docker run --pull always --rm -d --name mwj -p 127.0.0.1:5001:5001 trueagi/mwj:amd64
```

Wait a few seconds for the container to be ready, then run the benchmark.
To stop it: `docker stop mwj`

---

## Switching backends

In `pettachainer.py`, one flag controls which backend is used:

```python
USE_MWJ = False   # default — uses local PeTTa (Prolog/Janus)
USE_MWJ = True    # uses MWJClient (HTTP to localhost:5001)
```

No other code changes are needed — the rest of `PeTTaChainer` is backend-agnostic.

---

## Test setup

| Item | Value |
|---|---|
| Script | `experiments/chainer/pettachainer.py` |
| Knowledge base | `experiments/atomspace_visualizer/public/data.metta` |
| Facts loaded | 10 facts per article |
| Mined patterns | 4 association rules inserted via `formatter()` |
| Date | 2026-06-29 |
| Machine | Windows 11, localhost |
| MWJ server | `http://localhost:5001/metta` (stateful, `&kb` persists between requests) |

---

## Benchmark results

| Backend | Total execution time | Output correct |
|---|---|---|
| **PeTTa** (local Prolog/Janus) | **~0.46 – 0.56 s** | ✅ |
| **MWJClient** (HTTP to MWJ server) | **~2.9 s** | ✅ |

MWJClient is ~6× slower than PeTTa. The difference is pure HTTP overhead — each
`add-atom`, `match`, and `remove-atom` call is a separate HTTP POST to the MWJ
server, whereas PeTTa does all of this in-process (Prolog memory) with zero
network overhead.

Both backends produce identical fact lists. Only the `kb_id` hash differs between
runs — it is generated fresh each run by `uuid.uuid4().hex`:

```
DEBUG: get_facts output: [
  '(: kbe0aef2a7a233442eaea6b4f9ce2e8bf8 fact1  (audience-expertise A_14219 "Intermediate")  (STV 0.8   0.75))',
  '(: kbe0aef2a7a233442eaea6b4f9ce2e8bf8 fact2  (tone A_14219 "Informative")                 (STV 0.9   0.85))',
  '(: kbe0aef2a7a233442eaea6b4f9ce2e8bf8 fact3  (primary-goal A_14219 "Education")           (STV 0.85  0.8))',
  '(: kbe0aef2a7a233442eaea6b4f9ce2e8bf8 fact4  (audience-sentiment A_14219 "Positive")      (STV 0.6   0.7))',
  '(: kbe0aef2a7a233442eaea6b4f9ce2e8bf8 fact5  (complexity A_14219 "High")                  (STV 0.7   0.65))',
  '(: kbe0aef2a7a233442eaea6b4f9ce2e8bf8 fact6  (actionability A_14219 "Low")                (STV 0.8   0.85))',
  '(: kbe0aef2a7a233442eaea6b4f9ce2e8bf8 fact7  (topic A_14219 "humanoid-robot")             (STV 1.0   1.0))',
  '(: kbe0aef2a7a233442eaea6b4f9ce2e8bf8 fact8  (content-type A_14219 "audio")               (STV 1.0   1.0))',
  '(: kbe0aef2a7a233442eaea6b4f9ce2e8bf8 fact9  (length-bucket A_14219 "Short")              (STV 0.112 0.9))',
  '(: kbe0aef2a7a233442eaea6b4f9ce2e8bf8 fact10 (reading-time A_14219 "Long")                (STV 0.95  0.9))',
]
```

---

## Unit tests

`MWJClient` is covered by unit tests that run without a live MWJ server:

```bash
python -m pytest experiments/tests/test_mwj_client.py -v
```

```
26 passed in 0.76s
```

Tests cover `_parse_response`, `_split_metta_list`, `_handle_get_facts`,
`clear_kb`, and `_deduplicate` using `unittest.mock` to avoid real HTTP calls.

---

## MWJClient design notes

### Stateful KB management

The MWJ server keeps `&kb` alive across requests. `MWJClient` exposes a
`clear_kb(kb_id)` method that removes only atoms belonging to the given
`kb_id` — called from `PeTTaChainer.__init__` after generating `self.kb`.

### Per-instance KB isolation

`PeTTaChainer` generates a unique `kb_id` per instance (`"kb" + uuid4().hex`).
All atoms are stored as `(: kb_id factN ...)` so multiple chainer instances
or parallel runs never read each other's data from the shared `&kb`.

`clear_kb(kb_id)` is scoped to the instance's `kb_id` — it only removes atoms
matching that specific id, so other clients' data on the shared server is
never touched.

### Storage-strategy auto-detection

Different MWJ server versions store atoms in different shapes inside `&kb`:

| Strategy | Stored as | When used |
|---|---|---|
| `direct` | `(: kb_id factN type tv)` | Matches PeTTa's native format |
| `wrapped` | `(mm2compile kb_id (: factN type tv))` | Some MWJ server builds |
| `scan` | anything | Fallback: dump all atoms, filter by `kb_id` prefix |

The strategy is discovered with silent probe requests on the first `get_facts`
call, then cached in `self._strategy_cache` (keyed by `kb_id`) so subsequent
calls skip the probe entirely.

### Deduplication

Because the stateful server can accumulate duplicate atoms across `add-atom`
calls, all results are deduplicated before being returned, preserving insertion
order.

---

## Running the benchmark yourself

```bash
# start MWJ server first (see Prerequisites above), then:
python -m experiments.chainer.pettachainer
```

Set `USE_MWJ` at the top of `pettachainer.py` before running:

```python
USE_MWJ = False  # PeTTa  (~0.5 s)
USE_MWJ = True   # MWJClient  (~2.9 s)
```
