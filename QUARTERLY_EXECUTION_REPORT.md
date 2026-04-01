# Quarterly Execution Report — Mindplex-Hyperon

**Period:** Q1 2026 (October 2025 – March 2026)
**Repository:** [Xcceleran-do/Mindplex-Hyperon](https://github.com/Xcceleran-do/Mindplex-Hyperon)
**Report Date:** 2026-03-31

---

## Quarter Goal Summary

Build a transparent, explainable recommendation engine on top of Hyperon/AtomSpace by:
1. Benchmarking and selecting the best symbolic/neuro-symbolic rule miner.
2. Adapting and implementing the chosen miner in MeTTa + AtomSpace.
3. Completing the data ingestion pipeline to feed real Mindplex data into the miner.

---

## Objective 1: Research & Benchmark Neuro-Symbolic Rule Miners

> **Goal:** Evaluate candidate symbolic and neuro-symbolic rule miners and produce a technical decision document with the chosen miner and adaptation plan.

### KR 1 — Review at least 3 symbolic and neuro-symbolic approaches

**Status:** ✅ Complete

Four candidate algorithms were studied, prototyped (where feasible), and documented across two dedicated experimental branches (`SYMBOLIC` and `Neuro-Symbolic`):

| Algorithm | Type | Branch | Key Artefact | Link |
|---|---|---|---|---|
| **EVODA** | Symbolic / Evolutionary (genetic logic programming) | [`SYMBOLIC`](https://github.com/Xcceleran-do/Mindplex-Hyperon/tree/SYMBOLIC) | `Evoda/evoda.metta`; paper summary `Evoda/docs/evoda.md` | [commit](https://github.com/Xcceleran-do/Mindplex-Hyperon/commit/dabdffbec064463a1a284ac1ebd0008a71bc7687) |
| **DFOL** | Neuro-Symbolic (differentiable first-order logic + PyTorch) | [`Neuro-Symbolic`](https://github.com/Xcceleran-do/Mindplex-Hyperon/tree/Neuro-Symbolic) | `DFOL/dfol.metta`, `DFOL/pythonDFOL.py`; paper `paper.pdf` | [commit](https://github.com/Xcceleran-do/Mindplex-Hyperon/commit/8a9a80a4281bbfcadc875293e25686aca989f96b) |
| **RARL** | Neuro-Symbolic (rule-aware reinforcement learning) | [`Neuro-Symbolic`](https://github.com/Xcceleran-do/Mindplex-Hyperon/tree/Neuro-Symbolic) | Paper analysis: `RARL/docs/RARL_Analysis.md` | [PR #2](https://github.com/Xcceleran-do/Mindplex-Hyperon/pull/2) |
| **RuDiK** | Symbolic (SPARQL-based rule discovery for RDF/KBs) | [`Neuro-Symbolic`](https://github.com/Xcceleran-do/Mindplex-Hyperon/tree/Neuro-Symbolic) | Paper analysis: `RuDiK/docs/RuDiK_analysis.md` | [PR #6](https://github.com/Xcceleran-do/Mindplex-Hyperon/pull/6) |

In addition, the built-in **Hyperon-Miner** was evaluated as the fifth candidate in the context of the production MeTTa/AtomSpace environment on the `demo-PeTTa` and predecessor branches.

---

### KR 2 — Run comparative benchmarks on sample AtomSpace data

**Status:** ✅ Complete (with noted constraints)

A full head-to-head benchmark across all five candidates was not achievable because a complete port of DFOL and EVODA to execute natively inside AtomSpace hit fundamental barriers (see KR 3 and [docs/MINER_DECISION.md](https://github.com/Xcceleran-do/Mindplex-Hyperon/blob/copilot/analyze-commits-prs/docs/MINER_DECISION.md) for details). The following benchmarks were conducted:

**EVODA (on small synthetic data):**  
Tested on a minimal dataset. The algorithm ran to completion on toy-scale inputs. For any dataset of realistic size, the AtomSpace ran out of memory even at the smallest population/generation configuration, making it impossible to assess convergence or rule quality.

**DFOL (on toy data):**  
Tested on small synthetic examples. The propositionalization step and PyTorch training loop completed on toy inputs. Scaling to the 180-fact Mindplex dataset caused memory exhaustion. A PeTTa port was not possible because PeTTa had not implemented PyTorch support at the time of evaluation.

**Hyperon-Miner interpreter speed benchmark (January 10, 2026):**  
The Hyperon-Miner was run under both MeTTa interpreters on the 180-fact Mindplex article dataset to determine the preferred execution environment. Note: PeTTa is an alternative MeTTa interpreter (SWI-Prolog backend), not a different mining algorithm — the underlying Hyperon-Miner logic is identical in both cases.

**Dataset:** `experiments/atomspace_visualizer/public/data.metta` — 180 Mindplex article facts, 8 attributes each (length, reading-time, date-period, category, popularity, engagement, authored-by, title)

| Interpreter | Conjunct | Patterns Found | Real Time | Notes |
|---|---|---|---|---|
| Hyperon-Experimental (HE) | 3 | 9 | 78.5 s | Requires predicate filter; crashes at conjunct > 3 |
| PeTTa | 3 | 114 | 1.6 s | No filter needed; stable |
| PeTTa | 4 | 80 | 26.3 s | Stable |
| PeTTa | 5 | 35 | 284.8 s | Stable |

PeTTa is approximately **49× faster** than HE and handles all predicates without filtering. Benchmark results documented in commit [1a40965e](https://github.com/Xcceleran-do/Mindplex-Hyperon/commit/1a40965e62fa84d51ed4dba84c32dd81b45b6a54).

---

### KR 3 — Produce a technical decision document with the chosen miner and adaptation plan

**Status:** ✅ Complete

**→ See [docs/MINER_DECISION.md](https://github.com/Xcceleran-do/Mindplex-Hyperon/blob/copilot/analyze-commits-prs/docs/MINER_DECISION.md) for the full decision document.**

**Chosen miner: Hyperon-Miner, executed under the PeTTa interpreter.**

Summary of the decision:
- EVODA and DFOL were prototyped but both hit hard limits in the AtomSpace environment (memory exhaustion at realistic data sizes), preventing a head-to-head benchmark.
- RARL and RuDiK were unsuitable by design (RL-based graph traversal and SPARQL/RDF respectively) — neither maps to the Mindplex use case of conjunctive attribute co-occurrence mining.
- Hyperon-Miner is the only candidate that ran successfully on real Mindplex data at production scale, without porting barriers.
- After discussion with Dr. Ben Goertzel, using the well-tested internal Hyperon-Miner was confirmed as the right choice for this use case.
- The PeTTa interpreter was chosen over HE for its 49× speed advantage and stability across higher conjunction depths.

---

## Objective 2: Adapt / Implement Chosen Miner to Hyperon/MeTTa

> **Goal:** Implement or adapt the chosen miner (MeTTa interface, or from-scratch implementation), run it on sample AtomSpace, and extract ≥10 interpretable rules.

### KR 1 — Adapt the chosen miner

**Status:** ✅ Complete

The Hyperon-Miner was adapted and extended for the Mindplex use case on the [`demo-PeTTa`](https://github.com/Xcceleran-do/Mindplex-Hyperon/tree/demo-PeTTa) branch. The pipeline is located in `experiments/frequent-pattern-miner/` and `experiments/pattern-miner/`.

The core MeTTa pipeline ([commit 624582f9](https://github.com/Xcceleran-do/Mindplex-Hyperon/commit/624582f915580a3a77738d1132e500477f8ef45d)):
- `abstract-pattern`: extracts unique predicate link shapes from the AtomSpace, filtered by minimum support
- `build-specialization`: generates specialized (ground) forms from abstract patterns
- `candidatePatternMaker`: retains only specializations meeting the minimum support threshold
- `unique_combinations_star` (Python grounded op): generates star-join conjunctions of a chosen depth with a single shared variable — preventing spurious multi-variable cross-joins
- `formatter`: computes support for each conjunction and emits `supportOf` atoms as output

The PeTTa submodule was integrated ([commit f262371f](https://github.com/Xcceleran-do/Mindplex-Hyperon/commit/f262371f53da7a52e67b73e5de68b3c35e29e7e1)), a dedicated Dockerfile was created (`Dockerfile.petta`), and the CI/CD workflow was updated to run the miner under PeTTa.

Alpha-equivalence deduplication (`only_unique`, `giveMeUniqueAcc`, `is-member-custom`) was added to eliminate duplicate conjunctions under variable renaming — [PR #27](https://github.com/Xcceleran-do/Mindplex-Hyperon/pull/27).

---

### KR 2 — Run adapted miner on sample AtomSpace, extracting ≥10 interpretable rules

**Status:** ✅ Complete

The miner was run on the 180-fact Mindplex dataset at conjunction depth 3 with minimum support 3. Results:

- **114 patterns** extracted (far exceeding the ≥10 target)
- All patterns are interpretable conjunctions of article attribute–value pairs

Sample patterns:
```
(supportOf (, (length $X "short") (category $X "Technology")) 5)
(supportOf (, (category $X "Technology") (engagement $X "high")) 4)
(supportOf (, (date-period $X "Q4-2024") (popularity $X "Top_10")) 3)
```

At depth 4: 80 patterns. At depth 5: 35 patterns.

---

### KR 3 — Implementation of the adaptation strategy outlined

**Status:** ✅ Complete

The adaptation strategy described in [docs/MINER_DECISION.md](https://github.com/Xcceleran-do/Mindplex-Hyperon/blob/copilot/analyze-commits-prs/docs/MINER_DECISION.md) was implemented:
- Hyperon-Miner runs natively in MeTTa/AtomSpace — no external porting required.
- PeTTa interpreter integrated via Docker (`Dockerfile.petta`) and submodule ([commit f262371f](https://github.com/Xcceleran-do/Mindplex-Hyperon/commit/f262371f53da7a52e67b73e5de68b3c35e29e7e1)).
- The triple schema `(predicate article-id value)` feeds directly into the miner's `abstract-pattern` step.
- CI/CD pipeline updated to run and verify the miner against the sample dataset on each commit.

---

## Objective 3: Finalize Data Ingestion Pipeline After Miner Adaptation

> **Goal:** Design an ingestion schema consistent with the miner's input/output needs, populate AtomSpace with real Mindplex data, and run the miner on ingested data.

### KR 1 — Design ingestion schema consistent with miner's input/output needs

**Status:** ✅ Complete

All article facts are represented as typed triples in MeTTa format:

```
(predicate article-id value)
```

Example:
```
(category    a42 "Technology")
(authored-by a42 "Alice")
(length      a42 "short")
(popularity  a42 "Top_10")
(engagement  a42 "high")
(date-period a42 "Q4-2024")
```

This schema is directly compatible with the Hyperon-Miner's `abstract-pattern` step, which expects `(predicate subject value)` atoms. The schema design is documented in `experiments/ingestion/README.md` ([commit c1c555116](https://github.com/Xcceleran-do/Mindplex-Hyperon/commit/c1c555116ca7ca3bc3f4607a399cd8e7e0162d5e)).

The ingestion pipeline components built to populate this schema:
- `experiments/ingestion/fetcher.py` — fetches article data from API or local source
- `experiments/ingestion/converter.py` — converts JSON records to MeTTa triple format
- `experiments/ingestion/analyzer.py` — derives discretised attribute values (e.g. engagement buckets)
- `experiments/ingestion/pipeline.py` — orchestrates end-to-end fetching and conversion

Full modular multi-agent ingestion pipeline delivered in [PR #31](https://github.com/Xcceleran-do/Mindplex-Hyperon/pull/31) (135 tests passing).

---

### KR 2 — Populate AtomSpace with real data from Mindplex

**Status:** ✅ Complete

The AtomSpace was populated with real Mindplex production data via the ASI2 API integration. Key commits:

- Working ingestion pipeline with live Mindplex data: [commit 41d5cb20](https://github.com/Xcceleran-do/Mindplex-Hyperon/commit/41d5cb209983069a863f070e0de14f27f7ec5584)
- Production API (ASI2) integration and ingestion UI: [commit 366ad135](https://github.com/Xcceleran-do/Mindplex-Hyperon/commit/366ad135e4b8f4192f466e4dfd8878e6202da04a)

The resulting dataset contains **169,000+ article data points** from the live Mindplex platform.

---

### KR 3 — Run the miner on the ingested data

**Status:** ✅ Complete

The miner was run on the full 169k-datapoint production Mindplex AtomSpace.

**Result:** **924 rules** extracted at minimum support 900 (conjunction depth 3).

Preliminary mining results documented in the [`results` branch](https://github.com/Xcceleran-do/Mindplex-Hyperon/tree/results) — [commit 1a40965e](https://github.com/Xcceleran-do/Mindplex-Hyperon/commit/1a40965e62fa84d51ed4dba84c32dd81b45b6a54).

---

## Overall Progress Summary

| Objective | Key Result | Status |
|---|---|---|
| **Obj 1: Research & Benchmark** | KR 1: ≥3 approaches reviewed | ✅ Done — 4 algorithms reviewed ([EVODA](https://github.com/Xcceleran-do/Mindplex-Hyperon/commit/dabdffbec064463a1a284ac1ebd0008a71bc7687), [DFOL](https://github.com/Xcceleran-do/Mindplex-Hyperon/commit/8a9a80a4281bbfcadc875293e25686aca989f96b), [RARL PR #2](https://github.com/Xcceleran-do/Mindplex-Hyperon/pull/2), [RuDiK PR #6](https://github.com/Xcceleran-do/Mindplex-Hyperon/pull/6)) + Hyperon-Miner evaluated |
| | KR 2: Comparative benchmarks | ✅ Done — EVODA and DFOL tested on toy data (hit scaling limits); interpreter speed benchmark for Hyperon-Miner (HE vs PeTTa) |
| | KR 3: Technical decision document | ✅ Done — [docs/MINER_DECISION.md](https://github.com/Xcceleran-do/Mindplex-Hyperon/blob/copilot/analyze-commits-prs/docs/MINER_DECISION.md) |
| **Obj 2: Adapt/Implement Miner** | KR 1: Adapt chosen miner | ✅ Done — Hyperon-Miner pipeline in `experiments/frequent-pattern-miner/` ([commit 624582f9](https://github.com/Xcceleran-do/Mindplex-Hyperon/commit/624582f915580a3a77738d1132e500477f8ef45d)) |
| | KR 2: ≥10 interpretable rules | ✅ Done — 114 patterns at depth 3 on sample data; 924 rules on production data |
| | KR 3: Adaptation strategy implemented | ✅ Done — PeTTa runtime ([commit f262371f](https://github.com/Xcceleran-do/Mindplex-Hyperon/commit/f262371f53da7a52e67b73e5de68b3c35e29e7e1)), Docker, CI/CD, triple schema |
| **Obj 3: Ingestion Pipeline** | KR 1: Schema design | ✅ Done — triple schema + ingestion pipeline ([PR #31](https://github.com/Xcceleran-do/Mindplex-Hyperon/pull/31)) |
| | KR 2: Populate with Mindplex data | ✅ Done — 169k datapoints from live Mindplex API ([commit 366ad135](https://github.com/Xcceleran-do/Mindplex-Hyperon/commit/366ad135e4b8f4192f466e4dfd8878e6202da04a)) |
| | KR 3: Run miner on ingested data | ✅ Done — 924 rules at min-support 900 on 169k datapoints ([results branch](https://github.com/Xcceleran-do/Mindplex-Hyperon/tree/results)) |

**Legend:** ✅ Complete | 🔄 In Progress

---

## Unplanned Achievements

The following work was completed beyond the planned objectives for this quarter.

### 1. Hyperon-Miner Performance Optimization (32× speedup)

The miner underwent significant performance optimization work, tracked on the [`demo-mork` branch](https://github.com/Xcceleran-do/Mindplex-Hyperon/tree/demo-mork).

**Baseline performance (before optimization):**
```
real    163m38.960s
user    118m57.383s
sys       0m8.689s
```
Running the Hyperon-Miner on 169k datapoints with min-support 900, yielding 924 rules.

**After optimization — static-import approach (without Mork space):**  
([commit a64bda0f](https://github.com/Xcceleran-do/Mindplex-Hyperon/commit/a64bda0f53de344432b2f54023e37a1f259697ac), `demo-mork` branch)
```
real    5m4.606s
user    4m51.371s
sys     0m9.051s
```
Full dataset, same 924-rule output — **~32× faster** than the baseline.

**Mork space integration:**  
([commit 174b06c7](https://github.com/Xcceleran-do/Mindplex-Hyperon/commit/174b06c7aeaec99be95d5729ee5e0a7959738ae3), `demo-mork` branch)
```
real    8m1.965s
user    7m39.289s
sys     0m17.269s
```

**Analysis:**  
For the current dataset size (~169k), native AtomSpace without Mork is faster (5m04s vs 8m01s). However, Mork space provides an important stability advantage: the native space **crashes at min-support 300** on 169k datapoints, while the Mork-backed space continues to work (though it takes hours to finish at that threshold). This demonstrates that Mork space can handle lower support thresholds and potentially much larger datasets than native AtomSpace. For production use at the current data scale, the static-import approach (without Mork) is recommended; Mork becomes advantageous as dataset size grows or when lower support thresholds are needed.

---

### 2. PLN Backward Chainer Ported to MeTTa-Morphism 2 (mm2)

The PLN backward chaining engine was ported to the MeTTa-Morphism 2 (mm2) runtime ([commit 6921356a](https://github.com/Xcceleran-do/Mindplex-Hyperon/commit/6921356a4a10d668ad748c31d34b31d4d75122ae), `demo-mork` branch) and is functioning correctly. This ensures the backward chainer is compatible with the latest MeTTa runtime evolution and is not locked to an older interpreter version.

---

### 3. AtomSpace Visualizer

An interactive, web-based AtomSpace visualization tool was built from scratch — entirely outside the original OKR scope.

The visualizer ([PR #11](https://github.com/Xcceleran-do/Mindplex-Hyperon/pull/11), [commit fd0d083c](https://github.com/Xcceleran-do/Mindplex-Hyperon/commit/fd0d083c90d8bfceb35b6e440bd2e66dad34aa8e)) was built using **SolidJS + TypeScript + Vite** and provides:
- Real-time MeTTa expression parsing into an interactive force-directed / hierarchical / circular graph
- Zoom, pan, node selection, drag-and-drop interaction
- Monaco code editor for live MeTTa input with error reporting
- A "Mine the Gold" pattern mining HUD that triggers the Hyperon-Miner from the UI and displays results in a draggable result card

A subsequent refactor and dark-mode upgrade was delivered in [PR #19](https://github.com/Xcceleran-do/Mindplex-Hyperon/pull/19), modularising the `ColumnarVisualizer` component, introducing design tokens, and adding a full dark/light theme toggle.

---

### 4. Backward Chainer Integrated with Hyperon-Miner

The PLN backward chaining engine was fully integrated with the Hyperon-Miner output, creating an end-to-end pipeline from pattern discovery to symbolic reasoning.

Key deliverables:
- Initial backward chainer prototype wired to mined facts ([commit d9f2688e](https://github.com/Xcceleran-do/Mindplex-Hyperon/commit/d9f2688e11a3d996c78c688c3c4c9b0f8fd97fbe))
- `formatter()` function that converts mined `supportOf` patterns into PLN-compatible logical rules
- `getChainerResult()` backend function that accepts a natural-language why-query, rewrites it into MeTTa, runs the backward chainer, and returns a proof trace ([commit 385494ab](https://github.com/Xcceleran-do/Mindplex-Hyperon/commit/385494abfb7a97ebaa11ab0ca636e383f08b6b0c))
- Full PLN backward chaining engine with depth-controlled recursive inference, rule compilation (supporting `And`, `Or`, `Not`, `LikelierThan`, implication, and inverse implication), and a `PeTTaChainer` Python interface for dynamic rule ingestion and query execution — [PR #32](https://github.com/Xcceleran-do/Mindplex-Hyperon/pull/32) and [PR #34](https://github.com/Xcceleran-do/Mindplex-Hyperon/pull/34)

---

### 5. PeTTa Chainer with STV Value Propagation

The PeTTa-based backward chaining pipeline was extended with full **Simple Truth Value (STV)** propagation throughout the reasoning chain.

Key deliverables:
- **STV & EMPTV Integration for Pattern Mining** ([PR #29](https://github.com/Xcceleran-do/Mindplex-Hyperon/pull/29)): mined patterns are now annotated with Empirical Mean Probability Truth Values (EMPTV) computed from database support counts; all ingested facts carry STV strength/confidence values
- **STV-aware fact ingestion**: deterministic facts (e.g. `authored-by`, `category`) receive STV `(1 1)`; AI-inferred properties (e.g. tone, sentiment) receive computed STV values
- **STV-propagating backward chaining engine**: the `PeTTaChainer` Python interface stores all facts with unique IDs and STV-aware storage, propagates STV through inference steps, and generates explanation prompts that include STV strength/confidence for each step in the proof trace — [PR #32](https://github.com/Xcceleran-do/Mindplex-Hyperon/pull/32) and [PR #34](https://github.com/Xcceleran-do/Mindplex-Hyperon/pull/34)

---

### 6. LLM-Powered AI Chat Interface

An AI chat interface was integrated into the AtomSpace Visualizer, enabling natural-language interaction with both the miner and the backward chainer ([PR #18](https://github.com/Xcceleran-do/Mindplex-Hyperon/pull/18), [commit 385494ab](https://github.com/Xcceleran-do/Mindplex-Hyperon/commit/385494abfb7a97ebaa11ab0ca636e383f08b6b0c)).

Capabilities:
- Users can type queries like *"why is article X high engagement?"* — the system rewrites the query into MeTTa, runs the PLN backward chainer, and returns a natural-language explanation
- LLM function-calling integration (Google Gemini) registered for `mine_pattern()`, `getAllFactsAndRules()`, and `getChainerResult()`, making the assistant a multi-tool AI agent
- Mined pattern results are surfaced in chat as clickable `[Pattern N]` references
- Mining can be triggered either via the "Mine the Gold" button or through the chat interface, with identical effects

---

### 7. STV & EMPTV Truth Value Integration for Mined Patterns

Beyond the core frequency-based mining, mined patterns were enhanced with structured probabilistic truth values ([PR #29](https://github.com/Xcceleran-do/Mindplex-Hyperon/pull/29)):

- **`emp-tv`** function computes the Empirical Mean Probability Truth Value for each mined rule from its support count and total database size
- All factual triples in the AtomSpace carry explicit STV annotations, enabling downstream PLN reasoning to weight facts by confidence
- Continuous numerical attributes (e.g. engagement counts) are discretised into categories (`high` / `medium` / `low`) and assigned appropriate STV values for consistent pattern representation

---

## Risks & Blockers

| Risk | Impact | Mitigation |
|---|---|---|
| Hyperon-Miner (HE / native space) crashes below min-support 300 on 169k datapoints | Limits the minimum support that can be explored for deep patterns | Mork-backed AtomSpace supports lower support thresholds; use Mork for threshold exploration |
| Mork space is slower than native for current dataset scale | Adds ~3 min overhead per run at current data size | Use native static-import approach for production; revisit Mork when data scales further |

---

## Next Steps (Q2 2026)

1. Tune minimum support and conjunction depth on the full 169k-datapoint dataset to identify the most actionable engagement patterns.
2. Evaluate whether Mork space performance advantage materializes as data volume grows.
3. Scale the end-to-end pipeline (mining → PLN backward chaining → LLM explanation → UI) to production Mindplex traffic.
4. Extend the AtomSpace Visualizer to display STV-annotated patterns and proof traces from the backward chainer side-by-side.
