; --------------------------------------------------------------
;; pattern-miner 
; --------------------------------------------------------------
;; the main pattern mining pipeline for the functional approach 
;(: pattern-miner (-> $space space Number Number))
(= (pattern-miner $db $minsup $depth) ( 
    let $_ (println! ("==== Pattern Miner Started ===== depth is" $depth))
      (frequency-pattern-miner $db $minsup $depth)
   )  
)
