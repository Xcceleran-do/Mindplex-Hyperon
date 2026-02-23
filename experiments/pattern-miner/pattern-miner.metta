; --------------------------------------------------------------
;; pattern-miner 
; --------------------------------------------------------------
;; main pattern mining pipeline for functional approach 
;; Enhanced to handle both 3-argument and 4-argument STV patterns
;(: pattern-miner (-> $space space Number Number))
(= (pattern-miner $db $minsup $depth) ( 
    let $_ (println! ("==== STV Pattern Miner Started ===== depth is" $depth))
      (frequency-pattern-miner $db $minsup $depth)
   )  
)
