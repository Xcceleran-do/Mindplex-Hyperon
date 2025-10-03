
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
(= (candidatePattern  $dbspace $specializedPatterns $minsup)
    (foldl-atom $specializedPatterns () $acc $specialPattern (candidatePatternMaker $dbspace $specialPattern $minsup $acc))
)



;; candidatePatternMaker
;; ---------------------------------
;; Builds the list of candidate patterns by checking the support of a
;; Specialized pattern and adding it to the accumulator if it meets the
;; minimum support threshold.
;; Steps:
;;  1. Evaluate support with `sup-eval` against `$dbspace` and `$minsup`.
;;  2. If support passes, cons the `$pattern` onto `$acc`; otherwise keep `$acc`.
;; Parameters:
;;  - $dbspace: Space containing the database facts/patterns.
;;  - (Specialized $pattern): The pattern under test, wrapped as Specialized.
;;  - $minsup: Minimum support threshold (Number).
;;  - $acc: Accumulator (Expression) of accepted candidate patterns.
;; Returns:
;;  - Updated accumulator Expression with `$pattern` added iff `sup-eval` returns True.
; (:candidatePatternMaker (-> SpaceType (Specialized Pattern) Number Expression Expression))
( = (candidatePatternMaker $dbspace (Specialized $pattern) $minsup $acc)
    (
        let $result (sup-eval $dbspace $pattern $minsup)
            (if $result (cons-atom $pattern $acc) $acc )
    )
)