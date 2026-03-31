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

| Algorithm | Type | Branch | Artefact |
|---|---|---|---|
| **EVODA** | Symbolic / Evolutionary (genetic logic programming) | `SYMBOLIC` | Implemented in MeTTa: `Evoda/evoda.metta`; documented in `Evoda/docs/evoda.md` |
| **DFOL** | Neuro-Symbolic (differentiable first-order logic + PyTorch) | `Neuro-Symbolic` | Implemented in MeTTa + Python: `DFOL/dfol.metta`, `DFOL/pythonDFOL.py`; paper in `paper.pdf` |
| **RARL** | Neuro-Symbolic (rule-aware reinforcement learning) | `Neuro-Symbolic` | Paper analysis: `RARL/docs/RARL_Analysis.md` |
| **RuDiK** | Symbolic (SPARQL-based rule discovery for RDF/KBs) | `Neuro-Symbolic` | Paper analysis: `RuDiK/docs/RuDiK_analysis.md` |

In addition, the built-in **Hyperon-Miner** was evaluated as the fifth candidate in the context of the production MeTTa/AtomSpace environment on the `demo-PeTTa` and predecessor branches.

---

### KR 2 — Run comparative benchmarks on sample AtomSpace data

**Status:** ✅ Complete (with noted constraints)

A full head-to-head benchmark across all five candidates was not achievable because a complete port of DFOL and EVODA to execute natively inside AtomSpace hit fundamental barriers (see KR 3 and [`docs/MINER_DECISION.md`](./docs/MINER_DECISION.md) for details). The following benchmarks were conducted:

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

PeTTa is approximately **49× faster** than HE and handles all predicates without filtering. Full benchmark details and reproduction steps are in [`BENCHMARKS.md`](./BENCHMARKS.md).

---

### KR 3 — Produce a technical decision document with the chosen miner and adaptation plan

**Status:** ✅ Complete

**→ See [`docs/MINER_DECISION.md`](./docs/MINER_DECISION.md) for the full decision document.**

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

The Hyperon-Miner was adapted and extended for the Mindplex use case on the `demo-PeTTa` branch. The pipeline is located in `experiments/frequent-pattern-miner/` and `experiments/pattern-miner/`.

The core MeTTa pipeline:
- `abstract-pattern`: extracts unique predicate link shapes from the AtomSpace, filtered by minimum support
- `build-specialization`: generates specialized (ground) forms from abstract patterns
- `candidatePatternMaker`: retains only specializations meeting the minimum support threshold
- `unique_combinations_star` (Python grounded op): generates star-join conjunctions of a chosen depth with a single shared variable — preventing spurious multi-variable cross-joins
- `formatter`: computes support for each conjunction and emits `supportOf` atoms as output

The PeTTa submodule was integrated (`PeTTa/`, commit `f262371`), a dedicated Dockerfile was created (`Dockerfile.petta`), and the CI/CD workflow was updated to run the miner under PeTTa.

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

The adaptation strategy described in `docs/MINER_DECISION.md` was implemented:
- Hyperon-Miner runs natively in MeTTa/AtomSpace — no external porting required.
- PeTTa interpreter integrated via Docker (`Dockerfile.petta`) and submodule (`PeTTa/`).
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

This schema is directly compatible with the Hyperon-Miner's `abstract-pattern` step, which expects `(predicate subject value)` atoms. The schema design is documented in `experiments/ingestion/README.md`.

The ingestion pipeline components built to populate this schema:
- `experiments/ingestion/fetcher.py` — fetches article data from API or local source
- `experiments/ingestion/converter.py` — converts JSON records to MeTTa triple format
- `experiments/ingestion/analyzer.py` — derives discretised attribute values (e.g. engagement buckets)
- `experiments/ingestion/pipeline.py` — orchestrates end-to-end fetching and conversion

---

### KR 2 — Populate AtomSpace with real data from Mindplex

**Status:** 🔄 In Progress

**Completed:**
- 180-fact AtomSpace populated with real Mindplex article data and saved as `experiments/atomspace_visualizer/public/data.metta`
- Ingestion pipeline implemented (`experiments/ingestion/`) and tested

**Pending:**
- Live Mindplex API integration is blocked pending production API credentials. The pipeline code is ready; only the data fetch step requires access.

---

### KR 3 — Run the miner on the ingested data

**Status:** 🔄 In Progress

**Completed:**
- Miner runs successfully on the 180-fact ingested dataset, producing 114 patterns at conjunction depth 3 (detailed above in Objective 2 KR 2).

**Pending:**
- Running the miner on the full production Mindplex dataset is pending completion of KR 2 above.

---

## Overall Progress Summary

| Objective | Key Result | Status |
|---|---|---|
| **Obj 1: Research & Benchmark** | KR 1: ≥3 approaches reviewed | ✅ Done — 4 algorithms reviewed (EVODA, DFOL, RARL, RuDiK) + Hyperon-Miner evaluated |
| | KR 2: Comparative benchmarks | ✅ Done — EVODA and DFOL tested on toy data (hit scaling limits); interpreter speed benchmark for Hyperon-Miner (HE vs PeTTa) |
| | KR 3: Technical decision document | ✅ Done — see [`docs/MINER_DECISION.md`](./docs/MINER_DECISION.md) |
| **Obj 2: Adapt/Implement Miner** | KR 1: Adapt chosen miner | ✅ Done — Hyperon-Miner pipeline in `experiments/frequent-pattern-miner/` |
| | KR 2: ≥10 interpretable rules | ✅ Done — 114 patterns at depth 3 on sample data |
| | KR 3: Adaptation strategy implemented | ✅ Done — PeTTa runtime, Docker, CI/CD, triple schema |
| **Obj 3: Ingestion Pipeline** | KR 1: Schema design | ✅ Done — triple schema + ingestion pipeline |
| | KR 2: Populate with Mindplex data | 🔄 Partial — sample data populated; production blocked pending API access |
| | KR 3: Run miner on ingested data | 🔄 Partial — done on sample; production pending |

**Legend:** ✅ Complete | 🔄 In Progress

---

## Risks & Blockers

| Risk | Impact | Mitigation |
|---|---|---|
| Production Mindplex API access not yet available | Obj 3 KR 2 & 3 incomplete | Pipeline is ready; request credentials as priority for Q2 |
| Hyperon-Miner (HE) crashes at conjunction depth > 3 | Limits depth for HE interpreter | PeTTa interpreter handles depth 3–5 without crashes; use PeTTa for production |

---

## Next Steps (Q2 2026)

1. Obtain Mindplex production API credentials to complete Obj 3 KR 2 and KR 3.
2. Run the miner on the full production Mindplex dataset and report results.
3. Tune minimum support and conjunction depth on real data to identify actionable engagement patterns.
