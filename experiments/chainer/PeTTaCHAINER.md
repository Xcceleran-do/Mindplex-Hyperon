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
| `handle_why_question(question)` | Full pipeline: facts → Gemini rewrites question to canonical MeTTa query → backward chaining → Gemini explains result in plain language |

Each `PeTTaChainer` instance gets its own unique `kb_id` (`"kb" + uuid4().hex`) so
multiple instances never collide in the shared `&kb` on the MWJ server.

---

## Switching backends

In `pettachainer.py`, one flag controls which backend is used:

```python
USE_MWJ = False   # default — uses local PeTTa (Prolog/Janus)
USE_MWJ = True    # uses MWJClient (HTTP to localhost:5001)
```

No other code changes are needed — the rest of `PeTTaChainer` is backend-agnostic.

---
## Prerequisites — MWJ server (Docker)

MWJClient requires the MWJ server running locally. Start it with:

```bash
docker run --pull always --rm -d --name mwj -p 127.0.0.1:5001:5001 trueagi/mwj:amd64
```

Wait a few seconds for the container to be ready, then run the benchmark.
To stop it: `docker stop mwj`
## Test setup

| Item | Value |
|---|---|
| Script | `experiments/chainer/pettachainer.py` |
| Knowledge base | `experiments/atomspace_visualizer/public/data.metta` |
| Facts loaded | 10 facts for article `A_16624` |
| Mined patterns | 4 association rules inserted via `formatter()` |
| Date | 2026-06-25 |
| Machine | Windows 11, localhost |
| MWJ server | `http://localhost:5001/metta` (stateful, `&kb` persists between requests) |
| LLM | `gemini-2.5-flash` via `langchain_google_genai` |

---

## Benchmark results

| Backend | Total execution time | Output correct |
|---|---|---|
| **PeTTa** (local Prolog/Janus) | **~0.52 – 0.57 s** | ✅ |
| **MWJClient** (HTTP to MWJ server) | **~8.3 s** | ✅ |

Both backends produce identical fact lists. Only the `kb_id` hash differs between
runs — it is generated fresh each run by `uuid.uuid4().hex`:

```
DEBUG: get_facts output: [
  '(: kb945ad682f0b14729942a988e8007dbee fact1  (partial length (A_16624 "Medium"))      (STV 0.694 0.9))',
  '(: kb945ad682f0b14729942a988e8007dbee fact2  (reading-time A_16624 "Medium")          (STV 0.68  0.9))',
  '(: kb945ad682f0b14729942a988e8007dbee fact3  (tone A_16624 "Instructional")           (STV 0.9   0.95))',
  '(: kb945ad682f0b14729942a988e8007dbee fact4  (audience-expertise A_16624 "Beginner")  (STV 0.8   0.85))',
  '(: kb945ad682f0b14729942a988e8007dbee fact5  (content-type A_16624 "Tutorial")        (STV 0.7   0.8))',
  '(: kb945ad682f0b14729942a988e8007dbee fact6  (date-period A_16624 "Archived")         (STV 1.0   1.0))',
  '(: kb945ad682f0b14729942a988e8007dbee fact7  (primary-goal A_16624 "Inform")          (STV 0.9   0.95))',
  '(: kb945ad682f0b14729942a988e8007dbee fact8  (audience-sentiment A_16624 "Positive")  (STV 0.6   0.7))',
  '(: kb945ad682f0b14729942a988e8007dbee fact9  (popularity A_16624 "Top_10")            (STV 1.0   1.0))',
  '(: kb945ad682f0b14729942a988e8007dbee fact10 (engagement A_16624 "Low")               (STV 0.1   0.9))',
]
```

---

## Where the time goes

MWJClient is ~16× slower than PeTTa. The overhead is entirely network/HTTP,
split across three phases:

| Phase | Triggered by | Approx. cost |
|---|---|---|
| **KB clear** | `MWJClient.__init__` | Fetch all atoms + remove each individually via HTTP — ~0.5 – 2 s |
| **KB load** | `load_metta_file_to_chainer()` → `add_atom()` per line | One HTTP POST per fact — ~6 – 7 s (scales with KB size) |
| **get_facts / query** | `get_all_facts()`, `query()` | Strategy probe (once) + match — ~0.1 – 0.2 s |

PeTTa does all of this in-process (Prolog memory), so there is zero network overhead.

---

## MWJClient design notes

### Stateful KB management

The MWJ server keeps `&kb` alive across requests. `MWJClient.__init__` calls
`_clear_kb()` to drain any leftover atoms before each run — ensuring a clean
slate without restarting the server.

### Per-instance KB isolation

`PeTTaChainer` generates a unique `kb_id` per instance. All atoms are stored as
`(: kb_id factN ...)` so multiple chainer instances (or parallel runs) never
read each other's data from the shared `&kb`.

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
calls (e.g. if a KB is partially loaded more than once), all results are
deduplicated before being returned, preserving insertion order.

---

## Running the benchmark yourself

```bash
# make sure MWJ server is running on localhost:5001 if using MWJClient
python -m experiments.chainer.pettachainer
```

Set `USE_MWJ` at the top of `pettachainer.py` before running:

```python
USE_MWJ = False  # PeTTa  (~0.5 s)
USE_MWJ = True   # MWJClient  (~8.3 s)
```

Requires `GEMINI_API_KEY` in `.env` at the project root for the
`handle_why_question` pipeline (not needed for the basic benchmark).
