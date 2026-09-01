(= (sup-eval $db $pattern $minSup) 
    (let $matches (collapse (match $db $pattern $pattern)) 
        (>= (size-atom $matches) $minSup)
    )
)
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
        (if (sup-eval $dbspace ($link $x $rValue) $minsup) ($link (cut-first-char l$x) $rValue) (empty))
)

( = (candidatePatternMaker $dbspace ($link $lValue right) $minsup)
    (if (sup-eval $dbspace ($link $lValue $y) $minsup) ($link $lValue (cut-first-char r$y)) (empty))
)


;---------------------------------------
;; createSpecials
;;---------------------------------------
;; Build specialization templates (with placeholders) from a matched link.
;; For each ($link $shabx $shaby) produce two templates that mark which side is
;; fixed via `left` / `right`. The open side will later be turned into a variable
;; by candidatePatternMaker.


( = (createSpecials ($link $shabx $shaby))
    (superpose (($link left $shaby) ($link $shabx right)))
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
    (let $specializedPattern (only_unique (collapse (createSpecials (match $dbspace ($link $x $y) ($link $x $y)))))
        (candidatePatternMaker $dbspace (superpose $specializedPattern) $minsup))
)

;; --------------------------------------------
;;  formatter
; --------------------------------------------
;; Formats conjunctions with deterministic ordering and CTV annotation.
;; 1:- Iterate over (conjunct ...) items
;; 2:- Compute support with counter; skip if < $minSup
;; 3:- Compute CTV via pattern-stats/compute-ctv against $target-pred/$target-value
;;     (compute-ctv itself guards against nA=0 / N-nA=0 division by zero)
;; 4:- Promote engagement clause and emit (supportOf (, $sorted) $CTV $cnt)
(= (formatter (conjunct $conjunct) $dbspace $minSup $target-pred $target-value)
    (let $cnt (jointSupport $conjunct $dbspace)
        (if (>= $cnt $minSup)
            (let* (
                ($stats     (pattern-stats $target-pred $target-value $conjunct $dbspace))
                ($CTV       (compute-ctv $stats))
                ($organized (promote_engagement_conj $conjunct))
                ($result    (supportOf (push-back $CTV $organized) $cnt))
            )
            $result)
            ())))


; --------------------------------------------------------
;; End-to-end frequent pattern mining pipeline.
;; Steps:
;;  1:- Build specialization templates from DB and filter to candidates (absSpeCan)
;;  2:- Generate size-$depth conjunctions with unique_combinations_star (single hub var)
;;  3:- Format passing conjunctions as (supportOf (, ...) CTV n) with sorted clauses,
;;      CTV computed relative to $target-pred/$target-value
(= (frequency-pattern-miner $dbspace $minsup $depth $target-pred $target-value)
   (let $candidatePatterns (collapse (absSpeCan $dbspace $minsup))
        (let $conjuncts (unique_combinations_star $candidatePatterns $depth)
        (collapse (formatter (superpose $conjuncts) $dbspace $minsup $target-pred $target-value))))
)

;; Backward-compatible overload: defaults to `engagement`/`"High"`.
(= (frequency-pattern-miner $dbspace $minsup $depth)
   (frequency-pattern-miner $dbspace $minsup $depth engagement "High"))