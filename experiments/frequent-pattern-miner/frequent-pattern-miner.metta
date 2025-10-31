! (import! &self freq-pat)
! (import! &self conj-exp)


;; candidatePatternMaker
;; ---------------------------------
;; Filters specialized link templates using minimum support.
;; Specialized templates are created with placeholders `left` / `right`
;; to indicate the fixed side of a link; the other side becomes a variable.
;;
;; Overloads:
;;  - (candidatePatternMaker $dbspace ($link left  $rValue) $minsup)
;;      checks (sup-eval $dbspace ($link $x $rValue) $minsup) → if True, returns ($link $x $rValue), else Empty
;;  - (candidatePatternMaker $dbspace ($link $lValue right) $minsup)
;;      checks (sup-eval $dbspace ($link $lValue $y) $minsup) → if True, returns ($link $lValue $y), else Empty
;;
;; Inputs:
;;  - $dbspace: database space
;;  - $minsup: minimum support threshold
;;  - specialized pattern with `left`/`right` placeholder
;; Output: the instantiated pattern with a variable on the open side, or Empty
; (:candidatePatternMaker (-> SpaceType (Specialized Pattern) Number Expression Expression))
( = (candidatePatternMaker $dbspace ($link left $rValue) $minsup)
        (if (sup-eval $dbspace ($link $x $rValue) $minsup) ($link $x $rValue) Empty )
)

( = (candidatePatternMaker $dbspace ($link $lValue right) $minsup)
    (if (sup-eval $dbspace ($link $lValue $y) $minsup) ($link $lValue $y) Empty )
)


;---------------------------------------
;; createSpecials
;;---------------------------------------
;; Build specialization templates (with placeholders) from a matched link.
;; For each ($link $shabx $shaby) produce two templates that mark which side is
;; fixed via `left` / `right`. The open side will later be turned into a variable
;; by candidatePatternMaker.


( = (createSpecials ($link $shabx $shaby))
    ( superpose (($link left $shaby) ($link $shabx right)) )
)


;;---------------------------------
;; absSpeCan
;;---------------------------------
;; Create specialization templates directly from DB links and filter to candidates.
;; Steps:
;;  1:- Match all ($link $x $y) from $dbspace and create placeholders via createSpecials
;;  2:- Deduplicate the resulting templates with unique
;;  3:- Use candidatePatternMaker to instantiate open sides and keep only those
;;     meeting $minsup
( = (absSpeCan $dbspace $minsup)
    (let $specializedPattern (unique (createSpecials (match $dbspace ($link $x $y) ($link $x $y))))
        (candidatePatternMaker $dbspace $specializedPattern $minsup))
)

--------------------------------------------
;;  formatter
--------------------------------------------
;; Formats conjunctions with deterministic ordering and support annotation.
;; 1:- Iterate over (conjunct ...) items
;; 2:- Compute support with sup-num; skip if < $minSup
;; 3:- Sort clauses with sort_conj and emit (supportOf (, $sorted) $support)
;(: formatter (-> SpaceType SpaceType Number Atom ))

(= (formatter (conjunct $conjunct) $dbspace $minSup) 
        (function 
            (chain (eval (counter $dbspace $conjunct)) $cnt 
                    (return (if (>= $cnt $minSup)
                        (chain (eval (sort_conj $conjunct)) $sorted 
                                (supportOf (cons-atom , $sorted) $cnt))
                                Empty)))
        )
   
)
; --------------------------------------------------------
;; frequency-pattern-miner
; ---------------------------------------------------------
;; End-to-end frequent pattern mining pipeline.
;; Steps:
;;  1:- Build specialization templates from DB and filter to candidates (absSpeCan)
;;  2:- Generate size-$depth conjunctions with unique_combinations_star (single hub var)
;;  3:- Format passing conjunctions as (supportOf (, ...) n) with sorted clauses
;(: frequency-pattern-miner (-> SpaceType SpaceType SpaceType SpaceType SpaceType Number Number Atoms))
(= (frequency-pattern-miner $dbspace $minsup $depth)
   (let* (
           (() (println! "==== Frequent miner started ====="))
           ($candidatePatterns (collapse (absSpeCan $dbspace $minsup))); get candidate patterns
           (() (println! ("candidatePatterns ended and got" $candidatePatterns)))
           ($conjuncts  (unique_combinations_star $candidatePatterns $depth))
           (() (println! ("conjuncts we have got is "  $conjuncts)))
           ($temp4 (collapse (formatter (superpose $conjuncts) $dbspace $minsup)))      
           (() (println! (" conjuncts we have got is " $temp4)))
        )  $temp4)
)

; !(unique_combinations_star ((length $x "high") (length $x "low") (length $x "medium") (tone $x "Conversational") (tone $x "Humorous") (tone $x "Motivational") (tone $x "Gentle") (tone $x "Educational") (tone $x "Analytical") (writing_style $x "Conversational") (writing_style $x "Descriptive") (writing_style $x "Informal") (writing_style $x "Narrative") (writing_style $x "Formal") ) 3)
