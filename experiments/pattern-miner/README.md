#  Pattern Miner

This module mines **frequent patterns** from a knowledge base, with **optional surprisingness scoring**.  
It is a **functional implementation** of a pattern mining pipeline written in MeTTa.

##  How It Works

1. **Frequent Pattern Mining**:  
   The miner first extracts frequent patterns from the input corpus using a specified minimum support and conjunction size (`depth`).

2. **Surprisingness Scoring** *(Optional)*:  
   If a **surprisingness mode** is selected, the patterns are evaluated based on how unexpected they are.

3. **Output**:
   If `surp-mode` is `none` → Only frequent patterns are returned:  
     ```
     (supportOf $pattern $support)
     ```
   If `surp-mode` is not `none` → Each pattern is scored for surprisingness:  
     ```
     (surprisingnessOf $pattern $score)
     ```

## Surprisingness Modes

| Mode         | Description                                      |
|--------------|--------------------------------------------------|
| `none`       | No surprisingness calculation. Only support info.|
| `isurp-old`  | Computes surprisingness **without normalization**.|
| `nisurp-old` | Computes surprisingness **with normalization**.   |

## Requirements
- MeTTa v0.2.4
- A corpus (dataset) you want to mine patterns from
- Clone and structure this `hyperon-miner` module

## How to Run
### Example Call
```
;; Register  the hyperon-miner based on your folder structure 
! (register-module! hyperon-miner)

;; import the required modules 
! (import! &self experiments:pattern-miner:pattern-miner)
! (import! &self experiments:frequent-pattern-miner:frequent-pattern-miner)
! (import! &self experiments:utils:common-utils)
! (import! &self experiments:utils:surp-utils)

;; import the corpus (data)
! (import! &db your-path:data)


;; Create a space for storing mining results
!(bind! &res (new-space))

;; Run the pattern miner:
;; (pattern-miner $kb $db $minsup $depth $surp-mode)
;; - $kb: space where results are stored
;; - $db: database space to mine
;; - $minsup: minimum support (integer)
;; - $depth: conjunction size (number of clauses per pattern)
;;           2 -> pairs   (, (link A B) (link C B))
;;           3 -> triples (, (link A B) (link A C) (link A D))
;; - $surp-mode: 'none' | 'isurp-old' | 'nisurp-old' (surprisingness methods)

!(pattern-miner &res &db 3 0 none)
```


## Expected Output Format

### Without Surprisingness (`none`)
```
(supportOf (link A B) 4)
(supportOf (link B C) 3)
```

### With Surprisingness (`isurp-old` or `nisurp-old`)
```
(surprisingnessOf (link A B) 0.71)
(surprisingnessOf (link B C) 0.35)
```