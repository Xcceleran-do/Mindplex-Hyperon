# Mindplex Hyperon: Progress and Near-Term Plan

## Project Goal
Mindplex Hyperon builds a transparent, explainable recommendation engine that uses MeTTa/Hyperon to model content metadata as a knowledge graph and suggest articles with clear reasoning. Core use case: leverage metadata (topic, tone, length, engagement signals) to recommend content and explain why those recommendations fit audience preferences.

## Delivered (current branch)
- [x] Frequent pattern miner (MeTTa): star-join conjunction mining with deterministic sorting; wrapper `pattern-miner` delegates to frequent-pattern-miner.
- [x] Backward chainer: MeTTa rules + facts loaded into `&res1`; accessible via API for proof-based explanations.
- [x] Unified API server (Flask): mining, chat (Gemini), pattern analysis, backward chaining; CORS enabled; formatter feeds mined rules into the chainer.
- [x] Ingestion pipeline: fetch Mindplex articles, enrich with Gemini 2.0 flash, convert to MeTTa, write `experiments/atomspace_visualizer/public/data.metta`.
- [x] Graph/columnar visualizer (SolidJS + Vite): mining panel, chat auto-open, exact-match visualization of mined patterns; start/stop scripts for backend/frontend.
- [x] Test scaffolding: MeTTa test runner (`run-tests.py`) and module tests for miner components.

## Next Plan
### Phase 1: Forward-Chaining Engagement Simulator (Q2-Q3 2026)
- [ ] Forward chainer: simulate percentage impact of metadata combinations on audience engagement; expose API endpoint and guardrails (depth/iteration caps).
- [ ] PLN confidence propagation: extend rules with confidence derived from pattern support; forward-chain predictions with proof traces.
- [ ] Engagement prediction API: `/api/simulate` endpoint returning predicted engagement + confidence + explanation.

### Phase 2
- [ ] MetaMo/OpenPsi integration: replace manual mining triggers with appraisal-based emotional state (engagement gap → mining intensity).
- [ ] PRIMUS cognitive loop: implement tick-based autonomous mining based on creator goals and system confidence.

### Phase 3: MORK Backend for Scalability (Q2-Q3 2026)
- [ ] PathMap integration: migrate from in-memory spaces to MORK's trie-based backend for 1M+ article scale.
- [ ] Concurrent operations: lock-free reads, versioned writes, Merkle-DAG commits for audit trail.
