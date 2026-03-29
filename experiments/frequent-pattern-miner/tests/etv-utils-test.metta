
!(import! &self ../../utils/common-utils)
!(import! &self ../../frequent-pattern-miner/etv-utils)



(Inheritance Allen human)
(Inheritance Abe human)
(Inheritance Cason human)
(Inheritance Davion human)
(Inheritance Hessley human)
(Inheritance Jadrian human)
(Inheritance Rio human)
(Inheritance Bob human)
(Inheritance Mike human)
(Inheritance Zac human)
(Inheritance Lily human)
(Inheritance Emily human)
(Inheritance Lucy human)
(Inheritance Sophia human)
(Inheritance Alaura human)
(Inheritance Breonna human)
(Inheritance Jaleesa human)
(Inheritance Jassy human)
(Inheritance Jenica human)
(Inheritance Kecia human)

(Inheritance Allen man)
(Inheritance Abe man)
(Inheritance Cason man)
(Inheritance Davion man)
(Inheritance Hessley man)
(Inheritance Jadrian man)
(Inheritance Rio man)
(Inheritance Bob man)
(Inheritance Mike man)
(Inheritance Zac man)

(Inheritance Lily woman)
(Inheritance Emily woman)
(Inheritance Lucy woman)
(Inheritance Sophia woman)
(Inheritance Alaura woman)
(Inheritance Breonna woman)
(Inheritance Jaleesa woman)
(Inheritance Jassy woman)
(Inheritance Jenica woman)
(Inheritance Kecia woman)

(Inheritance Allen ugly)
(Inheritance Abe ugly)
(Inheritance Cason ugly)
(Inheritance Davion ugly)
(Inheritance Hessley ugly)
(Inheritance Lily ugly)
(Inheritance Emily ugly)
(Inheritance Lucy ugly)
(Inheritance Sophia ugly)
(Inheritance Alaura ugly)

(Inheritance Allen sodaDrinker)
(Inheritance Abe sodaDrinker)
(Inheritance Cason sodaDrinker)
(Inheritance Davion sodaDrinker)
(Inheritance Hessley sodaDrinker)
(Inheritance Breonna sodaDrinker)
(Inheritance Jaleesa sodaDrinker)
(Inheritance Jassy sodaDrinker)
(Inheritance Jenica sodaDrinker)
(Inheritance Kecia sodaDrinker)



; !(do_emp_tv (,(Inheritance $x ugly) (Inheritance $x woman)) &kb 0.02)    
; !(let $val (do-ji-tv-est &kb (,(Inheritance $x ugly) (Inheritance $x woman))) (println! $val))
; !(do-ji-tv-est &kb (,(Inheritance $x ugly) (Inheritance $x woman))) 



;; =============================================================================
!(test (universe-count (Inheritance $x) &self) 162)

;; =============================================================================
;; Test: Empirical Truth Value Calculation
;; -----------------------------------------------------------------------------
;; Verifies truth value calculation for specific pattern
;; Expected: (EMPTV 0.125 0.08888888888888889)
;;   - Strength: 12.5% 
;;   - Confidence: ~8.89%
;; =============================================================================
!(test (emp-tv (, (Inheritance $x woman) (Inheritance $x sodaDrinker)) &self) 
              (EMPTV 0.001388888888888889 0.08181818181818183))




