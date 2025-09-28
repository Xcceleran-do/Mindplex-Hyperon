
! (import! &self build-specialization)
! (import! &self candidate-patterns)
! (import! &self conjunction-expansion)

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
(= (abstract-pattern $dbspace $minsup $aptrnspace)        
 (let $linkunique (unique (match $dbspace ($link $x $y) $link))
                       ( let True (sup-eval $dbspace ($linkunique $z $t) $minsup)
                         (add-atom $aptrnspace (AbstractPattern ($linkunique Z (S Z)))))))






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
(= (frequency-pattern-miner $dbspace $specspace $cndpspace $aptrnspace $conjspace $minsup $depth)
   (let*(
           (() (println! "==== Frequent miner started ====="))
           (() (abstract-pattern $dbspace $minsup $aptrnspace)) ; prevent patterns that don't have a support from being specialized
           ($var  (collapse (match $aptrnspace (AbstractPattern $pattern) (build-specialization $pattern $dbspace $specspace)))) ; specialized abstract patterns
           ($temp2  (collapse (candidatePattern $dbspace $specspace $minsup $cndpspace))); get candidate patterns
            ($temp3  (collapse (conjunct $dbspace $cndpspace $conjspace $minsup $depth ))) ; store candidate patterns
            ($temp4  (collapse (formatter  $conjspace $dbspace $depth)))      
        )  $temp4)
)

