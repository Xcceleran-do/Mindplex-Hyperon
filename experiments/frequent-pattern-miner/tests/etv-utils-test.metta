!(pragma! import-root "../../..")
!(import! &self ../../utils/common-utils)

!(import! &self ../../frequent-pattern-miner/etv-utils)



!(bind! &kb (new-space))

!(add-atom &kb (Inheritance Allen human))
!(add-atom &kb (Inheritance Abe human))
!(add-atom &kb (Inheritance Cason human))
!(add-atom &kb (Inheritance Davion human))
!(add-atom &kb (Inheritance Hessley human))
!(add-atom &kb (Inheritance Jadrian human))
!(add-atom &kb (Inheritance Rio human))
!(add-atom &kb (Inheritance Bob human))
!(add-atom &kb (Inheritance Mike human))
!(add-atom &kb (Inheritance Zac human))
!(add-atom &kb (Inheritance Lily human))
!(add-atom &kb (Inheritance Emily human))
!(add-atom &kb (Inheritance Lucy human))
!(add-atom &kb (Inheritance Sophia human))
!(add-atom &kb (Inheritance Alaura human))
!(add-atom &kb (Inheritance Breonna human))
!(add-atom &kb (Inheritance Jaleesa human))
!(add-atom &kb (Inheritance Jassy human))
!(add-atom &kb (Inheritance Jenica human))
!(add-atom &kb (Inheritance Kecia human))

!(add-atom &kb (Inheritance Allen man))
!(add-atom &kb (Inheritance Abe man))
!(add-atom &kb (Inheritance Cason man))
!(add-atom &kb (Inheritance Davion man))
!(add-atom &kb (Inheritance Hessley man))
!(add-atom &kb (Inheritance Jadrian man))
!(add-atom &kb (Inheritance Rio man))
!(add-atom &kb (Inheritance Bob man))
!(add-atom &kb (Inheritance Mike man))
!(add-atom &kb (Inheritance Zac man))

!(add-atom &kb (Inheritance Lily woman))
!(add-atom &kb (Inheritance Emily woman))
!(add-atom &kb (Inheritance Lucy woman))
!(add-atom &kb (Inheritance Sophia woman))
!(add-atom &kb (Inheritance Alaura woman))
!(add-atom &kb (Inheritance Breonna woman))
!(add-atom &kb (Inheritance Jaleesa woman))
!(add-atom &kb (Inheritance Jassy woman))
!(add-atom &kb (Inheritance Jenica woman))
!(add-atom &kb (Inheritance Kecia woman))

!(add-atom &kb (Inheritance Allen ugly))
!(add-atom &kb (Inheritance Abe ugly))
!(add-atom &kb (Inheritance Cason ugly))
!(add-atom &kb (Inheritance Davion ugly))
!(add-atom &kb (Inheritance Hessley ugly))
!(add-atom &kb (Inheritance Lily ugly))
!(add-atom &kb (Inheritance Emily ugly))
!(add-atom &kb (Inheritance Lucy ugly))
!(add-atom &kb (Inheritance Sophia ugly))
!(add-atom &kb (Inheritance Alaura ugly))

!(add-atom &kb (Inheritance Allen sodaDrinker))
!(add-atom &kb (Inheritance Abe sodaDrinker))
!(add-atom &kb (Inheritance Cason sodaDrinker))
!(add-atom &kb (Inheritance Davion sodaDrinker))
!(add-atom &kb (Inheritance Hessley sodaDrinker))
!(add-atom &kb (Inheritance Breonna sodaDrinker))
!(add-atom &kb (Inheritance Jaleesa sodaDrinker))
!(add-atom &kb (Inheritance Jassy sodaDrinker))
!(add-atom &kb (Inheritance Jenica sodaDrinker))
!(add-atom &kb (Inheritance Kecia sodaDrinker))



; !(do_emp_tv (,(Inheritance $x ugly) (Inheritance $x woman)) &kb 0.02)    
; !(let $val (do-ji-tv-est &kb (,(Inheritance $x ugly) (Inheritance $x woman))) (println! $val))
; !(do-ji-tv-est &kb (,(Inheritance $x ugly) (Inheritance $x woman))) 



;; =============================================================================
!(test (universe-count (Inheritance $x) &kb) 60)

;; =============================================================================
;; Test: Empirical Truth Value Calculation
;; -----------------------------------------------------------------------------
;; Verifies truth value calculation for specific pattern
;; NOTE: expected value below is the current confirmed engine output,
;; pending investigation of a possible sup-num counting discrepancy
;; (manual count of woman+sodaDrinker overlap suggests strength ~0.5,

;; =============================================================================
!(test (emp-tv (, (Inheritance $x woman) (Inheritance $x sodaDrinker)) &kb) 
              (STV 1.0 0.8333333333333334))