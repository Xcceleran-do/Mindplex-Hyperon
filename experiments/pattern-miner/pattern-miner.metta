! (bind! &dbspace (new-space))
! (bind! &specspace (new-space))
! (bind! &cndpspace (new-space))
! (bind! &aptrnspace (new-space))
! (bind! &conjspace (new-space))



--------------------------------------------------------------
;; pattern-miner 
--------------------------------------------------------------
;; the main pattern mining pipeline for the functional approach 
;(: pattern-miner (-> $space space Number Number))
(= (pattern-miner $kb $db $minsup $depth) ( 
    let*
      (  (() (println! "==== Pattern Miner Started ====="))
         ($dummy (collapse (let ($link $x $y) (get-atoms $db) (add-atom &dbspace ($link $x $y))) ))
         ($frequent-pattern (frequency-pattern-miner &dbspace &aptrnspace &conjspace $minsup $depth))
         ; ($dummy3 (add-atoms $kb $frequent-pattern))
      )  
      $frequent-pattern
   )
)
