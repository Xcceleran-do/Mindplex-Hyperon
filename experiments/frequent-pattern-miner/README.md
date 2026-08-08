# Frequent Pattern Miner

The frequent-pattern-miner is a MeTTa-based pipeline that mines frequent patterns (including conjunctions) from a database space. It abstracts, specializes, filters by support, and then forms conjunctions of a chosen size using a star-join generator to avoid spurious multi-variable joins.

---

## What’s new in this version
- Conjunctions are generated with the TypeScript `unique_combinations_star` grounded operation in `mining-runner.ts`. It enforces a single hub variable shared by all clauses and prevents secondary shared variables.
- Formatting: conjunctions are normalized with promote_engagement_conj to bring engagement clauses forward, and emitted as supportOf (, ... ) support.
- Simpler API: frequency-pattern-miner returns the final list of annotated patterns directly (no external spaces required by the caller).

---

## Parameters

- $dbspace: database space to mine from
- $minsup: minimum support (integer)
- $depth: conjunction size (number of clauses per conjunction). 2 → pairs, 3 → triples, etc.

---

## Pipeline overview

1) abstract-pattern: extract unique link shapes from $dbspace and keep only those with support ≥ $minsup
2) build-specialization: generate specialized forms from abstract patterns
3) candidatePatternMaker: keep specialized patterns with support ≥ $minsup
4) unique combinations: unique_combinations_star builds size-$depth conjunctions with a single shared variable (hub)
5) formatter: compute support for each conjunction; if support ≥ $minsup, emit supportOf with sorted clauses

The top-level entry in this module is frequency-pattern-miner:

(= (frequency-pattern-miner $dbspace $minsup $depth)
	... returns a list like: ( (supportOf (, A B) 3) (supportOf (, B C) 2) ... ) ...)

---

## Example

Minimal dataset:

(topic 0 "AI")
(length 0 "low")
(topic 1 "AI")
(length 1 "low")
(topic 2 "AI")
(length 2 "high")
(topic 3 "Gardening")
(length 3 "low")

With minsup=2 and depth=2, the frequent conjunction is:

(supportOf (, (length $V0 "low") (topic $V0 "AI")) 2)

Run the focused check through MeTTaScript from the repository root:

```bash
npm exec -- tsx mining-runner.ts experiments/frequent-pattern-miner/tests/frequent-pattern-miner-test.metta
```

---

## Notes
- `unique_combinations_star`, `cut-first-char`, and `promote_engagement_conj` are registered by `mining-runner.ts`.
- Production mining invokes the same runner with `--mine <dataset> <min-support> <conjunction-count>`.
	so results are easier to scan.
- Support is computed with sup-num from experiments/utils/common-utils.metta.


