# Miner Selection Decision Document

**Project:** Mindplex-Hyperon  
**Author:** Sitotaw Ashagre (yotors)  
**Date:** March 2026  
**Status:** Final Decision

---

## 1. Context and Goal

The core mining task in Mindplex-Hyperon is:

> **Find frequently occurring combinations (conjunctions) of article metadata properties that correlate with reader engagement.**

The AtomSpace facts take the form:

```
(predicate article-id value)
```

For example:
```
(category  a42 "Technology")
(length    a42 "short")
(engagement a42 "high")
(authored-by a42 "Alice")
```

The miner must find patterns such as:
```
(category $X "Technology") ∧ (length $X "short") → engagement: "high"  [support ≥ 3]
```

The MeTTa / AtomSpace ecosystem was the required execution environment.

---

## 2. Research and Evaluation Phase

### 2.1 Approach to Evaluation

The research phase (July – October 2025) surveyed both symbolic and neuro-symbolic rule-mining algorithms. Three experimental branches were created to prototype candidate approaches:

| Branch | Purpose |
|---|---|
| `SYMBOLIC` | Symbolic/evolutionary approaches — EVODA |
| `Neuro-Symbolic` | Neuro-symbolic approaches — DFOL, RARL analysis, RuDiK analysis |
| `demo-PeTTa` (and predecessors) | Hyperon-Miner adaptation for production use |

A **full head-to-head benchmark across all candidates was not possible**. The core obstacle was that a complete port of DFOL and EVODA to execute natively *inside AtomSpace* (the required production environment) was not achievable within this quarter. The limitations discovered during porting attempts determined the final decision.

---

### 2.2 Algorithms Reviewed

#### 2.2.1 EVODA — Evolutionary Algorithm for Rule Learning over Knowledge Graphs
*(Implemented: `SYMBOLIC` branch, `Evoda/evoda.metta`)*

**Paper:** "Rule Learning Over Knowledge Graphs With Genetic Logic Programming"  
**Type:** Symbolic / Evolutionary

**Summary:**  
EVODA uses standard genetic operators — selection, mutation, and crossover — to evolve logic rules over knowledge graphs. The algorithm starts from an initial population of candidate rules and iterates across generations, measuring fitness with standard confidence and PCA confidence metrics. A rule-covering step then prunes the KG of covered facts after each generated rule.

**Implementation result:**  
EVODA was successfully prototyped in MeTTa (`Evoda/evoda.metta`). The implementation covers:
- Initial population generation (seeded with data-supported rules)
- Rule fitness calculation (standard confidence / PCA confidence)
- Selection, crossover, mutation operators
- Rule-covering loop

**Limitations encountered:**
- **AtomSpace memory exhaustion**: Even with small population sizes and generation counts, the AtomSpace ran out of memory and crashed. The nature of MeTTa's RAM-based AtomSpace means storing and evaluating a large population of candidate rules causes rapid memory growth.
- **Scalability cap**: Without a sufficiently large population and enough generations, the evolutionary search does not converge to high-quality rules. The algorithm's performance cannot be reliably measured or compared under these constraints.
- **No PeTTa port available**: At the time of evaluation, running EVODA under PeTTa (the SWI-Prolog-backed interpreter) was explored as a potential workaround for the memory limitations of the Hyperon-Experimental interpreter. However, the two interpreters' different memory models did not eliminate the fundamental problem of growing population state in AtomSpace.

---

#### 2.2.2 DFOL — Differentiable First-Order Logic
*(Implemented: `Neuro-Symbolic` branch, `DFOL/dfol.metta` + Python bindings)*

**Paper:** Implemented from the DFOL paper (included as `Neuro-Symbolic/paper.pdf`)  
**Type:** Neuro-Symbolic (combines symbolic first-order logic with neural network differentiable relaxation)

**Summary:**  
DFOL relaxes first-order logic rules into a differentiable form, enabling gradient-based learning over symbolic rule structures. A neural network component learns rule weights while the symbolic layer enforces logical consistency. PyTorch is used for the differentiable training loop.

**Implementation result:**  
DFOL was implemented in MeTTa with PyTorch Python bindings. The implementation covers:
- Propositionalization step (converting KG facts into a tabular / tensor representation)
- PyTorch-based neural network for rule weight learning
- MeTTa interface atoms calling into Python for tensor operations
  (`DFOL/array_like_tools.py`, `DFOL/torchme.py`, `DFOL/pythonDFOL.py`, `DFOL/dfol.metta`)

**Limitations encountered:**
- **Dataset scaling crashes**: DFOL crashed when run on datasets beyond the smallest toy examples. The propositionalization step (which builds a tensor over all entity-predicate-value combinations) grows quadratically with data size, exceeding available memory for the Mindplex article dataset.
- **PeTTa port not possible**: At the time of evaluation, PeTTa (the Prolog-backed MeTTa interpreter) had not yet implemented PyTorch support. Since DFOL's neural component fundamentally requires PyTorch, porting to PeTTa was not an option, preventing any speed improvements via that route.
- **Mismatch with use case**: DFOL is designed to learn weighted rules for link prediction tasks (e.g., predicting missing knowledge graph edges). The Mindplex use case — finding frequent co-occurrence patterns across discrete article attribute values — is a better fit for frequent-pattern mining than for differentiable rule learning.

---

#### 2.2.3 RARL — Rule-Aware Reinforcement Learning
*(Reviewed: `Neuro-Symbolic` branch, `RARL/docs/RARL_Analysis.md`)*

**Type:** Neuro-Symbolic (RL + logic rules)

**Summary:**  
RARL integrates logic rules as constraints or guidance signals into a reinforcement learning agent for knowledge graph reasoning. The RL agent learns a policy for traversing the knowledge graph while being guided by symbolic rules.

**Evaluation:**  
RARL was reviewed through paper analysis and pseudocode documentation. The algorithm was not prototyped. Key concerns for this use case:
- Designed for sequential graph-traversal and link-prediction tasks, not for frequency-based pattern discovery.
- Requires an RL training loop, which adds significant infrastructure overhead.
- No natural mapping from RL-discovered policies to interpretable conjunctive patterns of article properties.

---

#### 2.2.4 RuDiK — Rule Discovery in Knowledge Bases
*(Reviewed: `Neuro-Symbolic` branch, `RuDiK/docs/RuDiK_analysis.md`)*

**Type:** Symbolic (SPARQL-based rule discovery over RDF/OWL knowledge bases)

**Summary:**  
RuDiK discovers non-monotonic rules over RDF knowledge bases using SPARQL queries for candidate generation and support/confidence metrics for filtering. Rules are of the form `body → head` with optional negation-as-failure.

**Evaluation:**  
RuDiK was reviewed through paper analysis and analysis documentation. Key concerns for this use case:
- RuDiK is built for RDF/OWL datastores; adapting it to MeTTa/AtomSpace would require a full reimplementation of its SPARQL candidate-generation engine.
- Focused on relational KG rules (entity–relation–entity triples) rather than attribute co-occurrence mining.
- Does not align with the flat propositional structure of Mindplex article data.

---

#### 2.2.5 Hyperon-Miner (Built-in MeTTa Pattern Miner)
*(Used in production: `demo-PeTTa` branch, `experiments/pattern-miner/` and `experiments/frequent-pattern-miner/`)*

**Type:** Symbolic / Frequent Itemset Mining (built into the Hyperon/MeTTa ecosystem)

**Summary:**  
The Hyperon-Miner is the built-in pattern mining component of the Hyperon/MeTTa framework. It operates directly over AtomSpace data, finding frequent conjunctions of predicates (atoms) with a configurable minimum support threshold.

For the Mindplex use case, it finds patterns of the form:
```
(property-A $X value-1) ∧ (property-B $X value-2) ∧ ...  [support ≥ minsup]
```

**Testing:**  
The Hyperon-Miner was tested on the 180-fact Mindplex article dataset (`experiments/atomspace_visualizer/public/data.metta`) and ran successfully, producing interpretable conjunctive patterns across article attributes.

---

## 3. Interpreter Speed Benchmark

During evaluation it became clear that interpreter performance was a separate concern from algorithm selection. The Hyperon-Miner algorithm was available in two MeTTa runtime environments:

| Runtime | Description |
|---|---|
| **Hyperon-Experimental (HE)** | Python/Rust-backed MeTTa interpreter; RAM-based AtomSpace |
| **PeTTa** | SWI-Prolog-backed MeTTa interpreter; disk-indexed knowledge base |

> **Important:** PeTTa is an interpreter for the MeTTa language, not a different mining algorithm. The underlying Hyperon-Miner logic is the same in both cases.

A speed benchmark was conducted (January 10, 2026) on the same 180-fact dataset and the same Hyperon-Miner algorithm to compare interpreter performance:

| Interpreter | Conjunct | Patterns Found | Real Time |
|---|---|---|---|
| Hyperon-Experimental | 3 | 9* | 78.5 s |
| PeTTa | 3 | 114 | 1.6 s |
| PeTTa | 4 | 80 | 26.3 s |
| PeTTa | 5 | 35 | 284.8 s |

\* *HE required filtering of some predicates to avoid crashes; PeTTa ran without any filtering.*

The ~49× speed advantage of PeTTa and its ability to handle all predicates without requiring filtering confirmed that PeTTa is the preferred execution environment for the Hyperon-Miner. This is an interpreter-level optimization, not a change in mining algorithm.

Full benchmark details and reproduction instructions are in [`BENCHMARKS.md`](../BENCHMARKS.md).

---

## 4. Decision

**Chosen miner: Hyperon-Miner, executed under the PeTTa interpreter.**

### 4.1 Rationale

The decision was made after technical review and a discussion with Dr. Ben Goertzel. The key factors were:

| Factor | Detail |
|---|---|
| **Fit for use case** | The Mindplex task requires frequent co-occurrence mining of article attribute combinations correlated with engagement. This is precisely the problem Hyperon-Miner is designed for. DFOL and RARL are oriented toward link prediction; EVODA targets rule learning with a target predicate. |
| **No porting required** | Hyperon-Miner runs natively in MeTTa/AtomSpace. DFOL and EVODA both required significant adaptation work to run in the target environment and hit fundamental barriers (memory, missing PyTorch in PeTTa). |
| **Well tested** | Hyperon-Miner is a mature component of the Hyperon framework, with existing tests and known behavior. The candidates (DFOL, EVODA) were research prototypes with significant stability concerns on real data sizes. |
| **Internal tooling preference** | Where possible, using well-maintained internal tools reduces integration risk and maintenance burden. This aligns with the recommendation from Dr. Ben Goertzel. |
| **Speed via PeTTa** | The interpreter benchmark confirmed that PeTTa provides the runtime performance needed to mine at conjunction depths relevant to the use case (depth 3–5). |

### 4.2 Algorithm Disqualification Summary

| Algorithm | Reason Not Selected |
|---|---|
| **EVODA** | AtomSpace crashes even at small population/generation sizes; performance cannot be measured without the scale needed for meaningful rule learning |
| **DFOL** | Crashes on real dataset sizes; PyTorch dependency blocks PeTTa port; designed for link prediction, not attribute co-occurrence mining |
| **RARL** | Designed for RL-based graph traversal, not conjunctive attribute pattern mining; no natural mapping to the Mindplex use case |
| **RuDiK** | Built for SPARQL/RDF; would require complete reimplementation for MeTTa/AtomSpace; not aligned with flat propositional attribute data |

---

## 5. Adaptation Plan

1. **Use Hyperon-Miner** with the PeTTa interpreter as the primary mining runtime.
2. **Schema**: Represent each article as a set of `(predicate article-id value)` triples in AtomSpace. The miner's `abstract-pattern` step extracts link shapes from these triples.
3. **Configuration**: Set minimum support (`minsup`) based on dataset size; use conjunction depth 3–5 for meaningful multi-attribute patterns.
4. **Ingestion**: Build a pipeline that fetches article data from the Mindplex API, maps it to the triple schema, and populates the AtomSpace before each mining run.
5. **Output**: The miner produces `supportOf` atoms listing each frequent conjunction with its support count. These can be filtered and ranked for recommendation use.
