# Pattern Miner

This module is a thin wrapper that runs the frequent pattern mining pipeline in MeTTa.

Current implementation delegates to frequency-pattern-miner and returns a list of
annotated frequent patterns (supportOf …) based on a given minimum support and
conjunction size (depth).

## API

(= (pattern-miner $kb $db $minsup $depth)
   (frequency-pattern-miner $db $minsup $depth))

Parameters:
- $db: database space to mine
- $minsup: minimum support (integer)
- $depth: conjunction size (2 → pairs, 3 → triples, …)

Note: $kb is currently unused in the implementation (kept for compatibility).

## Usage

;; Register according to your folder layout
! (register-module! experiments)

;; Imports
! (import! &self experiments:pattern-miner:pattern-miner)
! (import! &self experiments:frequent-pattern-miner:frequent-pattern-miner)
! (import! &self experiments:utils:common-utils)

;; Prepare DB
!(bind! &db (new-space))
!(add-atom &db (topic 0 "AI"))
!(add-atom &db (length 0 "low"))

;; Run
!(pattern-miner &res &db 2 2)

Output example:

(supportOf (, (length $V0 "low") (topic $V0 "AI")) 2)