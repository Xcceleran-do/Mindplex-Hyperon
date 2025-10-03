# Frequent Pattern Miner

The `frequent-pattern-miner` is a modular pipeline for mining **frequent patterns** from a given atomspace. It extracts abstract patterns, specializes them, filters them by support, and constructs conjunctive patterns, returning only the ones that meet the support threshold.

---


## 🔧 Purpose

To find patterns that **frequently occur** in the atomspace, using a multi-step symbolic mining approach that includes abstraction, specialization, support evaluation, and conjunction generation.

---

## Parameters

| Parameter      | Description                                                                 |
|----------------|-----------------------------------------------------------------------------|
| `$dbspace`     | The atomspace to mine from.                                          |
| `$specspace`   | Space to store specialized patterns.                                        |
| `$cndpspace`   | Space to store candidate patterns.                                          |
| `$aptrnspace`  | Space to store abstract patterns.                                           |
| `$conjspace`   | Space to store pattern conjunctions.                                        |
| `$minsup`      | Minimum support threshold for a pattern to be considered frequent.         |
| `$depth`       | Conjunction size (number of clauses per conjunction). For example: 2 → pairs, 3 → triples. |

---

## How It Works (Pipeline Overview)

### Step 1 — Abstract Pattern Mining (`abstract-pattern`)
- Extract **unique link patterns** from the database.
- Turn them into **abstract patterns** using variables.
- Compute **support** for each, and store only those meeting `$minsup` in `$aptrnspace`.

### Step 2 — Specialization (`build-specialization`)
- Take each abstract pattern and **generate specialized versions** based on how they match in the atomspace.
- Store them in `$specspace`.

### Step 3 — Candidate Generation
- Evaluate the **support of specialized patterns**.
- If the support ≥ `$minsup`, store them in `$cndpspace` as **candidates**.

### Step 4 — Conjunction Generation (`do-conjunct`)
- Build unique combinations of size `$depth` from candidate patterns.
- Evaluate support for each conjunction and keep only those meeting `$minsup`, storing them in `$conjspace`.

### Step 5 — Finalization
- Format and return the valid patterns with support annotations.

---

## Output

- A structured set of **frequent patterns** (including conjunctions) stored in your result space, each annotated with its computed support value.
- These patterns are useful for reasoning, classification, or higher-level symbolic analysis.


