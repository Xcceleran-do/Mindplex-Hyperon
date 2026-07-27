; --------------------------------------------------------------
;; pattern-miner - STV-ONLY VERSION  
; --------------------------------------------------------------
;; Main pattern mining pipeline for STV patterns
;(: pattern-miner (-> $space space Number Number))
(= (pattern-miner $db $minsup $depth) 
    (frequency-pattern-miner $db $minsup $depth)
)
