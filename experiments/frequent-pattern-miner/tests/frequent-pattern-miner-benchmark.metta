!(import! &self ../../frequent-pattern-miner/conj_exp)
!(import! &self ../../utils/common-utils)
!((import! &self ../../frequent-pattern-miner/etv-utils))
!(import! &self ../../frequent-pattern-miner/frequent-pattern-miner)

;; 10 entities x 10 attributes = 100 facts. All values are intentionally fixed,
;; producing ten frequent left-open candidates at minsup 10.
(= (seed-benchmark $db $n)
   (if (<= $n 0)
       True
       (let $i (- $n 1)
         (let* (($a (add-atom $db (audience-expertise $i Intermediate)))
            ($b (add-atom $db (tone $i Informative)))
            ($c (add-atom $db (primary-goal $i Education)))
            ($d (add-atom $db (audience-sentiment $i Positive)))
            ($e (add-atom $db (complexity $i Medium)))
            ($f (add-atom $db (actionability $i Low)))
            ($g (add-atom $db (topic $i AI)))
            ($h (add-atom $db (content-type $i article)))
            ($j (add-atom $db (length-bucket $i Short)))
                ($k (add-atom $db (engagement $i Low))))
           (seed-benchmark $db $i)))))

(= (timed-miner-stages $db $minsup $depth)
   (let* (($t0 (current-time))
          ($candidates (collapse (absSpeCan $db $minsup)))
          ($t1 (current-time))
          ($conjunctions (unique-combinations-star $candidates $depth))
          ($t2 (current-time))
          ($formatted (collapse (formatter (superpose $conjunctions) $db $minsup)))
          ($t3 (current-time)))
     (stage-times
        (candidates (size-atom $candidates) (- $t1 $t0))
        (combinations (size-atom $conjunctions) (- $t2 $t1))
        (formatting (size-atom $formatted) (- $t3 $t2)))))

!(seed-benchmark &benchmark-space 10)
!(timed-miner-stages &benchmark-space 5 3)
