--------------------------------------------------------------
;; pattern-miner 
--------------------------------------------------------------
;; the main pattern mining pipeline for the functional approach 
;(: pattern-miner (-> $space space Number Number))
(= (pattern-miner $kb $db $minsup $depth) ( 
    let () (println! "==== Pattern Miner Started =====")
      (frequency-pattern-miner $db $minsup $depth)
   )  
)
