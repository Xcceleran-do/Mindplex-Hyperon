
! (import! &self freq-pat)

---------------------------------
;; candidatePattern
;; ---------------------------------
;; Builds the list of candidate patterns from a space of Specialized patterns
;; by folding over them and applying candidatePatternMaker, which retains only
;; those patterns whose support meets or exceeds the minimum threshold.
;; Steps:
;;  1. Iterate through `$specializedPatterns` using foldl-atom.
;;  2. For each item, call `candidatePatternMaker` to check support and update `$acc`.
;;  3. Return the final accumulated list of candidate patterns.
;; Parameters:
;;  - $dbspace: Space containing the database facts/patterns.
;;  - $specializedPatterns: Space of Specialized patterns to evaluate.
;;  - $minsup: Minimum support threshold (Number).
;; Returns:
;;  - Expression (list) of patterns that satisfy the support threshold.
; (:candidatePattern (-> SpaceType SpaceType Number Expression Expression))
(= (candidatePattern  $dbspace $specializedPattern $minsup)
    (foldl-atom $specializedPatterns () $acc $specialPattern (candidatePatternMaker $dbspace $specialPattern $minsup $acc))
)


