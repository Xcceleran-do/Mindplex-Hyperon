! (register-module! ../../../experiments)

! (import! &self experiments:pattern-miner:pattern-miner)
! (import! &self experiments:utils:common-utils)
! (import! &self experiments:frequent-pattern-miner:frequent-pattern-miner)
! (import! &db experiments:data:small-ugly)

! (bind! purifiedDbSpace (new-space)) ; space to hold the database atoms
! (add-reduct purifiedDbSpace (get-atoms &db))

;; space to hold the miner result
!(bind! &res1 (new-space))
; ! (memberChecker (engagement_level $x "high") () purifiedDbSpace 3)
; !(sup-eval purifiedDbSpace (engagement_level $x "high") 3)
; ! (memberChecker (engagement_level $x "low") ( (engagement_level $x "low") (engagement_level $x "med")) purifiedDbSpace 3)
; ! (let $ana (customUnique (match purifiedDbSpace ($link $x $y) (createSpecials ($link $x $y))) purifiedDbSpace 3) (--- :- --- (unique_combinations_star $ana)))

---------------------------------------------------
;; mine frequent patterns without surprissingness 
---------------------------------------------------
;; minimum support 3 -3rd arg
;; number of conjuncts 3 -4th arg

!(pattern-miner purifiedDbSpace 3 3) 
!(println! "============ end ================")
;; demo result structure
    ; ((supportOf (, (Inheritance $y human) (Inheritance $y man)) 3) 
    ; (supportOf (, (Inheritance $j human) (Inheritance $j ugly)) 3) 
    ; (supportOf (, (Inheritance $p human) (Inheritance $p sodaDrinker)) 3) 
    ; (supportOf (, (Inheritance $r man) (Inheritance $r ugly)) 3) 
    ; (supportOf (, (Inheritance $x man) (Inheritance $x sodaDrinker)) 3) 
    ; (supportOf (, (Inheritance $x sodaDrinker) (Inheritance $x ugly)) 3) 
    ; (supportOf (, (Inheritance Abe $r) (Inheritance Cason $r)) 4)
    ; (supportOf (, (Inheritance Abe $l) (Inheritance Allen $l)) 4) 
    ; (supportOf (, (Inheritance Allen $g) (Inheritance Cason $g)) 4))
