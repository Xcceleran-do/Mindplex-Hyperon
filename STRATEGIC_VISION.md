# Mindplex-Hyperon: Strategic Vision & Component Map

## The Big Picture

**Mindplex-Hyperon is Hyperon's canonical use case**: a production-scale application demonstrating how Hyperon's cognitive architecture solves real-world problems while maintaining transparency, composability, and safety.

```
                    HYPERON ECOSYSTEM
     ╔═══════════════════════════════════════════════════════════════╗
     ║                                                               ║
     ║  ┌─────────────────────────────────────────────────────┐     ║
     ║  │ ATOMSPACE (Unified Knowledge Substrate)             │     ║
     ║  │ - Facts, rules, proofs, embeddings coexist          │     ║
     ║  │ - Backed by MORK PathMap (constant-time lookups)    │     ║
     ║  └─────────────────────────────────────────────────────┘     ║
     ║           ↑            ↑            ↑           ↑             ║
     ║       PATTERNS      INFERENCE    NEURAL     MOTIVATION        ║
     ║       (Mining)      (Chaining)   (PC/NN)    (MetaMo)          ║
     ║           ↓            ↓            ↓           ↓             ║
     ║  ┌─────────────────────────────────────────────────────┐     ║
     ║  │ PRIMUS Cognitive Cycle (Goal-Directed + Ambient)    │     ║
     ║  │ - Orchestrates multiple methods per ECAN budget    │     ║
     ║  │ - Weighted Atom Sweeps for selective attention     │     ║
     ║  └─────────────────────────────────────────────────────┘     ║
     ║           ↑                           ↓                       ║
     ║      PERCEPTION                   ACTION                      ║
     ║      (Content)                     (Recommendations)          ║
     ║                                                               ║
     ╚═══════════════════════════════════════════════════════════════╝
                      ↓
           ╔══════════════════════════╗
           │ MINDPLEX-HYPERON         │ (Use Case)
           ├══════════════════════════┤
           │ Phase 1: MetaMo Triggers │
           │ Phase 2: Forward Chaining│
           │ Phase 3: MORK Backend    │
           │ Phase 4: ECAN Scheduling │
           │ Phase 5: MOSSES Modules  │
           │ Phase 6: Decentralization│
           └══════════════════════════┘
```

---

## Evolution Path: From Tool to PRIMUS Agent

### Timeline & Transformation

```
CURRENT (Jan 2026)           PHASE 1 (Q1-Q2)              PHASE 2 (Q2-Q3)
┌─────────────────┐         ┌────────────────┐           ┌─────────────────┐
│ BUTTON CLICK    │    →    │ EMOTION DRIVEN │      →    │ PREDICTIVE      │
│ Manual trigger  │         │ Mining on      │           │ Intelligent     │
│ In-memory store │         │ MetaMo signal  │           │ Simulator       │
│                 │         │ In-memory      │           │ MORK-backed     │
└─────────────────┘         └────────────────┘           └─────────────────┘
     User                        User +                    User + System
     Action                    Motivation                  Interaction

PHASE 3 (Q3)                 PHASE 4 (Q4)                 PHASE 5+ (2027)
┌──────────────────┐         ┌──────────────────┐         ┌────────────────────┐
│ SCALABLE         │    →    │ INTELLIGENT      │    →    │ DECENTRALIZED      │
│ MORK PathMap     │         │ BUDGET-AWARE     │         │ Multi-Creator      │
│ 1M+ articles     │         │ ECAN allocation  │         │ Consensus-driven   │
│ Lock-free reads  │         │ Fair scheduling  │         │ Governance DAO     │
└──────────────────┘         └──────────────────┘         └────────────────────┘
     System                   System + Hyperon             System + Community
     Maturity                 Orchestration                Validation
```

---

## Component Integration Map

### Current System (What Exists)
```
                    MINDPLEX-HYPERON
                    ================

    User Interface (Chat + Graph)
              ↓
    Mining API (Flask)
        ├─ /api/mine          ← Pattern miner
        ├─ /api/chat          ← Gemini integration
        ├─ /api/chainer       ← Backward chaining
        └─ /api/analyze       ← Result analysis
              ↓
    ┌───────────────────────────────────┐
    │     MeTTa + Atomspace             │
    ├───────────────────────────────────┤
    │ &tempo    (articles)              │
    │ &db       (candidates)            │
    │ &res1     (rules)                 │
    │ &metrics  (engagement stats)      │
    └───────────────────────────────────┘
              ↓
    Pattern Mining ←→ Backward Chaining
      (inductive)       (deductive)
```

### Future System (Roadmap Integrated)

```
                    MINDPLEX-HYPERON (FULL PRIMUS)
                    ==============================

    Creator Interaction Layer (Enhanced UI)
         ↓
    ┌──────────────────────────────────────────────────────────┐
    │ Cognitive Cycle Orchestrator (PRIMUS Loop)              │
    │ - MetaMo: Appraisal → Motivation                        │
    │ - ECAN: Budget allocation by STI/LTI                    │
    │ - Forward Chaining: Simulation of engagement outcomes    │
    ├──────────────────────────────────────────────────────────┤
    │ Weighted Atom Sweeps (WAS)                              │
    │ - Selective attention scheduling                        │
    │ - Cognitive resources distribution                      │
    └──────────────────────────────────────────────────────────┘
         ↓ ↓ ↓ ↓ ↓
    ┌─────────────────────────────────────────────────────────┐
    │          ATOMSPACE (Unified Substrate)                   │
    │ ┌──────────┬──────────┬──────────┬──────────────────┐   │
    │ │ Articles │ Patterns │ Rules    │ Proofs/Metadata  │   │
    │ │ &tempo   │ &db      │ &res1    │ &meta            │   │
    │ └──────────┴──────────┴──────────┴──────────────────┘   │
    └─────────────────────────────────────────────────────────┘
         ↓
    ┌─────────────────────────────────────────────────────────┐
    │ MORK Backend (PHASE 3)                                  │
    │ - PathMap trie structure                                │
    │ - Lock-free concurrent reads/writes                     │
    │ - Merkle-DAG versioning                                 │
    │ - Distributed deployment support                        │
    └─────────────────────────────────────────────────────────┘
         ↓
    ┌─────────────────────────────────────────────────────────┐
    │ MOSSES Module System (PHASE 5)                          │
    │ - Pluggable miners (star-join, apriori, sampling)      │
    │ - Pluggable reasoners (backward, forward, abductive)    │
    │ - Runtime algorithm selection                           │
    └─────────────────────────────────────────────────────────┘
         ↓
    ┌─────────────────────────────────────────────────────────┐
    │ Distributed Governance Layer (PHASE 6)                  │
    │ - CID-addressed patterns (immutable)                    │
    │ - Merkle-state versioning                              │
    │ - Multi-creator merging                                │
    │ - Reputation/incentive system                          │
    └─────────────────────────────────────────────────────────┘
```

---

## Dependency Graph: How Components Connect

```
MetaMo (Motivation)
  ↓ (creator's emotion → mining trigger)
Pattern Miner
  ↓ (discovers patterns)
Rules → Backward Chainer
  ↓ (explains current engagement)
  ↓
Forward Chainer (NEW PHASE 2)
  ↓ (simulates engagement if rules applied)
  ↓ (uses confidence from ECAN)
ECAN (Resource Allocation)
  ├─ Mining priority ← STI (engagement gap), LTI (discovery payoff)
  ├─ Chaining priority ← STI (explanation requests), LTI (rule accuracy)
  ├─ Simulation priority ← STI (prediction requests)
  └─ Visualization priority ← STI (user focus)
  ↓
MORK (Storage)
  ├─ Stores articles (&tempo)
  ├─ Stores patterns (&db)
  ├─ Stores rules (&res1)
  └─ Stores metadata (&meta, versioning)
  ↓
MOSSES (Modularity)
  ├─ Modules for different mining algorithms
  ├─ Modules for different inference strategies
  ├─ Runtime selection based on data characteristics
  └─ Version pinning for reproducibility
  ↓
Governance DAO (Decentralization)
  ├─ Multi-creator pattern proposals
  ├─ Reputation tracking
  ├─ Merkle-state consensus
  └─ Incentive alignment
```

---

## Key Transitions & Capabilities Unlocked

### Transition 1: Manual → Motivated (Phase 1)

**Before**: 
```
Creator clicks button
  → System mines
  → Results arrive
```

**After**:
```
Creator's engagement gap detected
  → MetaMo generates motivation signal
  → System mines automatically
  → Explains WHY patterns discovered
```

**New Capability**: **Proactive knowledge discovery** — system learns what creators care about and pursues answers independently.

---

### Transition 2: Reactive → Predictive (Phase 2)

**Before**:
```
Backward chaining answers: "Why is this article engaging?"
  → Backward-looking explanations only
```

**After**:
```
Forward chaining predicts: "IF I publish article X with properties Y, what engagement?"
  → Confidence scores from pattern support
  → Proof traces showing which rules contributed
```

**New Capability**: **Engagement simulation** — creators can A/B test ideas (mentally) before publishing.

---

### Transition 3: Limited Scale → Million-Article Scale (Phase 3)

**Before**:
```
In-memory MeTTa spaces
  → 10k articles: responsive
  → 100k articles: slow
  → 1M articles: infeasible
```

**After**:
```
MORK PathMap backend
  → 1M articles: fast queries
  → 10M articles: distributed shards
  → Concurrent creators: no locks
```

**New Capability**: **Production scale** — move from research prototype to real-world deployment.

---

### Transition 4: Monolithic → Fair Scheduling (Phase 4)

**Before**:
```
All operations get equal CPU
  → Important queries wait for unimportant background tasks
  → No prioritization mechanism
```

**After**:
```
ECAN allocates budget by (STI × LTI) / Cost
  → High-priority operations get more CPU
  → Low-payoff mining paused when accuracy ↓
  → Fair sharing among multiple creators
```

**New Capability**: **Intelligent resource management** — system automatically optimizes where to spend computational effort.

---

### Transition 5: Fixed Algorithms → Pluggable Modules (Phase 5)

**Before**:
```
One mining algorithm hardcoded
  → All creators use same approach
  → Can't specialize per domain
```

**After**:
```
MOSSES provides multiple miners
  → Choose star-join for high accuracy
  → Choose sampling for speed
  → Choose Apriori for domain similarity detection
```

**New Capability**: **Algorithmic composability** — tailor system to each use case at runtime.

---

### Transition 6: Centralized → Decentralized (Phase 6)

**Before**:
```
One mining instance per creator
  → No shared learning
  → Duplicate effort
```

**After**:
```
Shared pattern repository with reputation
  → Creators collaborate on patterns
  → Best miners rewarded
  → Merkle proofs ensure tamper-resistance
```

**New Capability**: **Decentralized intelligence** — community converges on accurate patterns via incentives.

---

## Success Criteria by Phase

### Phase 1: MetaMo Integration ✓
- [ ] Emotion triggers mining (not buttons)
- [ ] System explains motivation for each mining run
- [ ] Appraisal updates as engagement metrics change

### Phase 2: Forward Chaining ✓
- [ ] Engagement predictions within 10% accuracy
- [ ] Confidence scores provided for each prediction
- [ ] Proof traces understandable to non-experts

### Phase 3: MORK Scalability ✓
- [ ] Handle 1M articles in < 5s queries
- [ ] Concurrent creator writes don't block reads
- [ ] Cold-start from disk in < 30s

### Phase 4: ECAN Fairness ✓
- [ ] All creators achieve fair CPU allocation
- [ ] High-STI operations prioritized
- [ ] Overall throughput increases 2-3x under load

### Phase 5: MOSSES Modularity ✓
- [ ] Swap mining algorithms at runtime
- [ ] Load/unload modules without restart
- [ ] Version pinning for reproducibility

### Phase 6: Governance DAO ✓
- [ ] Multi-creator pattern submissions
- [ ] Reputation leaderboard
- [ ] Merkle-proof verification of lineage

---

## Alignment with Hyperon Vision

| Hyperon Principle | Mindplex Implementation |
|---|---|
| **Unified Substrate** | Atomspace holds articles, patterns, rules, proofs |
| **Multiple Methods** | Mining + chaining + neural synthesis |
| **Weakness (Simplicity)** | Min support threshold, star-join constraint |
| **Geodesic Control** | ECAN budget allocation by (STI×LTI)/Cost |
| **First-Class Reflection** | Patterns, rules, proofs queryable as data |
| **Self-Modification** | Rules learned from data, can be audited/refined |
| **Decentralization** | MOSSES modules + DAO governance |
| **Safety** | All decisions traceable to patterns & proofs |

---

## Staffing & Budget Estimate

| Phase | Duration | FTE | Cost (est.) | Hyperon Dependency |
|-------|----------|-----|------------|-------------------|
| 1: MetaMo | 6 wks | 1.0 | $12K | MetaMo API defined |
| 2: Forward Chaining | 8 wks | 1.5 | $18K | PLN confidence calculus |
| 3: MORK | 10 wks | 2.0 | $24K | MORK API docs |
| 4: ECAN | 8 wks | 1.0 | $12K | WAS scheduling algorithm |
| 5: MOSSES | TBD | TBD | $10K+ | MOSSES definition |
| 6: Governance | TBD | TBD | $20K+ | F1R3FLY framework |
| **TOTAL (1-4)** | **6 months** | **5.5** | **$66K** | **High alignment** |

---

## Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|-----------|
| MetaMo API not finalized | P1 blocked | Start with simple motivation model |
| MORK learning curve high | P3 delayed | Partner with Hyperon for integration support |
| ECAN overhead too large | P4 ineffective | Profile critical paths, implement caching |
| MOSSES undefined | P5 stalled | Proceed with P1-4 in parallel, revisit when defined |
| Scale testing reveals bugs | All phases | Invest in load testing early (end of P2) |

---

## Next Steps (Next 30 Days)

1. **Clarify MOSSES** with Hyperon team (or confirm no integration needed)
2. **Prototype Phase 1** (MetaMo appraisal function in MeTTa)
3. **Design Phase 2** (confidence propagation semantics)
4. **Benchmark Phase 3** (profile MORK vs. in-memory performance)
5. **Get community feedback** on roadmap priorities

---

## References

- **Main Roadmap**: [INTEGRATION_ROADMAP.md](INTEGRATION_ROADMAP.md)
- **MOSSES Analysis**: [MOSSES_INTEGRATION_ANALYSIS.md](MOSSES_INTEGRATION_ANALYSIS.md)
- **Logic Summary**: [PROJECT_ACADEMIC_SUMMARY.md](PROJECT_ACADEMIC_SUMMARY.md)
- **Hyperon Whitepaper**: (located in workspace)
- **Hyperon Repository**: https://github.com/opencog/hyperon-experimental

