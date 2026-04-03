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
; (: candidatePatternMaker (-> SpaceType Expression Number Expression))
( = (candidatePatternMaker $dbspace ($link left $rValue) $minsup)
    (if (sup-eval $dbspace ($link $x $rValue) $minsup) ($link (cut-first-char l$x) $rValue) (empty))
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
    (let $specializedPattern (only_unique (collapse (match $dbspace ($link $x $y) ($link left $y))))
        (candidatePatternMaker $dbspace (hyperpose $specializedPattern) $minsup))
)
; --------------------------------------------
;;  formatter
; --------------------------------------------
;; Formats conjunctions with deterministic ordering and support annotation.
;; 1:- Iterate over (conjunct ...) items
;; 2:- Compute support with sup-num; skip if < $minSup
;; 3:- Promote engagement clause and emit (supportOf (, $sorted) $support)
;(: formatter (-> SpaceType SpaceType Number Atom ))

(= (formatter (conjunct $conjunct) $dbspace $minSup) 
    (let $cnt (counter $dbspace $conjunct)
        ;; calculate EMPTV
        (let $EMPTV (emp-tv $conjunct $dbspace)
            (if (>= $cnt $minSup)
                (let $organized (promote_engagement_conj $conjunct)
                    (trace! $organized (supportOf ($organized $EMPTV) $cnt)))
                (empty)))))
        
; --------------------------------------------------------
;; End-to-end frequent pattern mining pipeline.
;; Steps:
;;  1:- Build specialization templates from DB and filter to candidates (absSpeCan)
;;  2:- Generate size-$depth conjunctions with unique_combinations_star (single hub var)
;;  3:- Format passing conjunctions as (supportOf (, ...) n) with sorted clauses
;(: frequency-pattern-miner (-> SpaceType SpaceType SpaceType SpaceType SpaceType Number Number Atoms))

(= (frequency-pattern-miner $dbspace $minsup $depth)
   (let $candidatePatterns (collapse (absSpeCan $dbspace $minsup))
        (let $conjuncts (unique_combinations_star $candidatePatterns $depth)
        $conjuncts))
)

; !(unique_combinations_star ((length $x "high") (length $x "low") (length $x "medium") (tone $x "Conversational") (tone $x "Humorous") (tone $x "Motivational") (tone $x "Gentle") (tone $x "Educational") (tone $x "Analytical") (writing_style $x "Conversational") (writing_style $x "Descriptive") (writing_style $x "Informal") (writing_style $x "Narrative") (writing_style $x "Formal") ) 3)
