
! (import! &self freq-pat)
! (import! &self conj-exp)

---------------------------------------------
;; abstract-pattern
---------------------------------------------
;; Gets Abstract Patterns with Minimum Support so that we can use them for the specialization.
;; 1: Query the database to retrieve all link nodes (nodes that connect other nodes)
;; 2: Take only the unique links from the queried links
;; 3: Create an Abstract pattern using the links and variables
;; 4: Calculate the support for the Abstract patterns and store the ones satisfy the minimum support in a space
;(: abstract-pattern (-> SpaceType Number SpaceType Atoms))
(= (abstract-pattern $dbspace $minsup)        
        (let $linkUnique (unique (match $dbspace ($link $x $y) $link))
                ( if (sup-eval $dbspace ($linkUnique $z $t) $minsup) ($linkUnique $x $y) Empty ) 
        )
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
( = (candidatePatternMaker $dbspace $pattern $minsup)
    (
        let $result (sup-eval $dbspace $pattern $minsup)
            (if $result $pattern Empty )
    )
)


;---------------------------------------
;; createSpecials
;---------------------------------------
;; Function to create specialized patterns from a list of substitution pairs.
;; For each pair in $shabPairs, generates two specialized atoms by substituting
;; the elements into the pattern, checks if they already exist in the accumulator,
;; and adds them if they do not exist. Returns the updated accumulator containing
;; all unique specialized atoms.

( = (createSpecials ($link $shabx $shaby))
    ( superpose (($link $x $shaby) ($link $shabx $y)) )
)



-------------------------------------------------------------------------------
; build-specialization
-------------------------------------------------------------------------------
; function to build the specialization of a pattern and add it to the specialization space
; - get the valuation of the pattern by destructuring the pattern
; - abstract the pattern and handle patterns with depth
; - build the specialization
; - add the specialization to the specialization space
;(: build-specialization (-> Pattern SpaceType SpaceType Atom))
; (: build-specialization (-> Pattern SpaceType SpaceType Any))
(= (build-specialization ($link $x $y) $dbspace) 
    (let*
        (  
            (() (println! "==== Build specialization started ====="))
            ($shabPair (match $dbspace ($link $x $y) ($link $x $y)))
        )
        (createSpecials $shabPair)
    )
)





--------------------------------------------
;;  formatter
--------------------------------------------
;; Formats conjunctions with their support annotation.
;; 1: get the conjunctions from `$conjspace` and sort them using `sort_conj`
;; 2: build a flattened conjunction using `cons-atom` with `,`
;; 3: annotate with support using `supportOf` and `sup-num`
;(: formatter (-> SpaceType SpaceType Number Atom ))
(= (formatter $conjuncts $dbspace $minSup) 
   (    
    let* (
            (() (println! "==== Formating  started ====="))
            ((conjunct $conjunct) (superpose $conjuncts))
            ($support (sup-num $dbspace $conjunct))
            ($add (>= $support $minSup))
        ) 
        ( if $add
            (let $sorted (sort_conj $conjunct) (supportOf (cons-atom , $sorted) $support))
            Empty            
        )
   )
)


--------------------------------------------------------
;; frequency-pattern-miner
---------------------------------------------------------
;; function to mine the frequent patterns from the atomspace.
;; 1: get all abstract patterns with their support and store them in a space
;; 2: specialize the abstract patterns and store them in a space
;; 3: get all candidate patterns from the specialized patterns and store them in a space
;; 4: expand the conjunctions of the candidate patterns and store them in a space with their support
;; 5: format the patterns and add them to the knowledge base with their support
;(: frequency-pattern-miner (-> SpaceType SpaceType SpaceType SpaceType SpaceType Number Number Atoms))
(= (frequency-pattern-miner $dbspace $minsup $depth)
   (let*(
           (() (println! "==== Frequent miner started ====="))
           ($candidatePatterns (collapse (absSpeCan $dbspace $minsup))); get candidate patterns
           (() (println! ("candidatePatterns ended and got" $candidatePatterns)))
           ($conjuncts  (unique_combinations_star $candidatePatterns $depth))
           (() (println! (" conjucts ended and got" $conjuncts)))
           ($temp4 (collapse (formatter $conjuncts $dbspace $minsup)))      
        )  $temp4)
)


( = (absSpeCan $dbspace $minsup)
        (let* (
           ($aptrn (abstract-pattern $dbspace $minsup))
           ($specializedPattern (unique (build-specialization $aptrn $dbspace)))
        ) (candidatePatternMaker $dbspace $specializedPattern $minsup)
        )
)