(= (sup-eval $db $pattern $minSup) 
    (let $matches (collapse (match $db $pattern $pattern)) 
        (>= (size-atom $matches) $minSup)
    ))
(= (counter $db $ptrn)
    (let $got (collapse (match $db $ptrn 1)) (size-atom $got))) 
;;---------------------------------
;; candidatePatternMaker - 
;;---------------------------------
;; Filters specialized link templates using minimum support.
;; Only handles 4-argument STV patterns: ($link $lValue $rValue $stv)
;; Specialized templates are created with placeholders `left` / `right`
;; to indicate fixed side of a link; other side becomes a variable.
;;
;; Overloads:
;;  - (candidatePatternMaker $dbspace ($link left $rValue $stv) $minsup) [STV]
;;      checks (sup-eval $dbspace ($link $x $rValue $stv) $minsup) → if True, returns ($link $x $rValue $stv), else Empty
;;  - (candidatePatternMaker $dbspace ($link $lValue right $stv) $minsup) [STV]
;;      checks (sup-eval $dbspace ($link $lValue $y $stv) $minsup) → if True, returns ($link $lValue $y $stv), else Empty
;;
;; Inputs:
;;  - $dbspace: database space
;;  - $minsup: minimum support threshold
;;  - specialized pattern with `left`/`right` placeholder and STV

( = (candidatePatternMaker $dbspace ($link $lValue right) $minsup)
    (if (sup-eval $dbspace ($link $lValue $y) $minsup) ($link $lValue (cut-first-char r$y)) (empty))
)

;; STV-aware overloads
( = (candidatePatternMaker $dbspace ($link left $rValue $stv) $minsup)
        (if (sup-eval $dbspace ($link $x $rValue $stv) $minsup) 
            ($link (cut-first-char l$x) $rValue $stv) (empty))
)

( = (candidatePatternMaker $dbspace ($link $lValue right $stv) $minsup)
    (if (sup-eval $dbspace ($link $lValue $y $stv) $minsup) 
        ($link $lValue (cut-first-char r$y) $stv) (empty))
)


;---------------------------------------
;; createSpecials
;---------------------------------------
;; Build specialization templates (with placeholders) from a matched link.
;; Enhanced to handle both 3-argument and 4-argument STV patterns
;; For each ($link $shabx $shaby) produce two templates that mark which side is
;; fixed via `left` / `right`. The open side will later be turned into a variable
;; by candidatePatternMaker.
;; For STV patterns ($link $shabx $shaby $stv), preserve the STV component

(= (createSpecials ($link $shabx $shaby))
    (superpose (($link left $shaby) ($link $shabx right))))

;; STV-aware overload - Generalize STV to variables
(= (createSpecials ($link $shabx $shaby $stv))
    (superpose (($link left $shaby $stvVar) ($link $shabx right $stvVar))))


;;---------------------------------
;; absSpeCan -
;;---------------------------------
;; Create specialization templates directly from DB links and filter to candidates.
;; Enhanced to handle both 3-argument and 4-argument STV patterns
;; Steps:
;;  1:- Match all ($link $x $y) from $dbspace and create placeholders via createSpecials
;;  2:- Match all ($link $x $y $stv) from $dbspace and create placeholders via createSpecials
;;  3:- Deduplicate the resulting templates with unique
;;  4:- Use candidatePatternMaker to instantiate open sides and keep only those
;;     meeting $minsup
;;  5:- Use STV-aware combination logic for 4-argument patterns

( = (absSpeCan $dbspace $minsup))
    (let $specialized3Arg (unique-atom (collapse (createSpecials (match $dbspace ($link $x $y) ($link $x $y)))))
        (let $specialized4Arg (unique-atom (collapse (createSpecials (match $dbspace ($link $x $y $stv) ($link $x $y $stv)))))
        (let $allSpecialized (superpose $specialized3Arg $specialized4Arg)
            (candidatePatternMaker $dbspace (superpose $allSpecialized) $minsup)))
)

;; STV-aware absSpeCan - detects and uses STV combination logic
;; Generalize STV values to variables during pattern matching
(= (absSpeCan-stv $dbspace $minsup)
    (let $specialized4Arg (only_unique (collapse (createSpecials (match $dbspace ($link $entity $value (STV $strength $confidence)) ($link $entity $value (STV $strength $confidence))))))
    (candidatePatternMaker $dbspace (superpose $specialized4Arg) $minsup)
    ))


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
                    (if (>= $cnt $minSup)
                        (let $organized (promote_engagement_conj $conjunct)
                                (trace! $organized (supportOf $organized $cnt)))
                                (empty))))

; --------------------------------------------------------
;; frequency-pattern-miner - 
;; ---------------------------------------------------------
;; End-to-end frequent pattern mining pipeline for STV patterns only.
;; Steps:
;;  1:- Build STV specialization templates from DB and filter to candidates (absSpeCan-stv)
;;  2:- Generate size-$depth conjunctions with unique_combinations_star (STV-only)
;;  3:- Format passing conjunctions as (supportOf (, ...) n) with sorted clauses
;(: frequency-pattern-miner (-> SpaceType SpaceType SpaceType SpaceType Number Number Atoms))

(= (frequency-pattern-miner $dbspace $minsup $depth)
   (let $candidatePatterns
        (collapse (absSpeCan-stv $dbspace $minsup))
   (let $conjuncts (unique_combinations_star $candidatePatterns $depth)
        (collapse (formatter (superpose $conjuncts) $dbspace $minsup)))))

; !(unique_combinations_star ((length $x "high") (length $x "low") (length $x "medium") (tone $x "Conversational") (tone $x "Humorous") (tone $x "Motivational") (tone $x "Gentle") (tone $x "Educational") (tone $x "Analytical") (writing_style $x "Conversational") (writing_style $x "Descriptive") (writing_style $x "Informal") (writing_style $x "Narrative") (writing_style $x "Formal") ) 3)
