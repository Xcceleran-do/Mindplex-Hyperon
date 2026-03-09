
!(import! &self /mnt/c/Users/HP/iCog/Mindplex-Hyperon/PeTTa/lib/lib_import)
!(import_prolog_functions_from_file "/mnt/c/Users/HP/iCog/Mindplex-Hyperon/experiments/frequent-pattern-miner/conj_exp.pl" (unique_combinations_star cut-first-char promote_engagement_conj))
!(import! &self ../../utils/common-utils)
!(import! &self /mnt/c/Users/HP/iCog/Mindplex-Hyperon/experiments/frequent-pattern-miner/frequent-pattern-miner)
!(import! &self ../../pattern-miner/pattern-miner)
!(import! &temp /mnt/c/Users/HP/iCog/Mindplex-Hyperon/experiments/ingestion/outputs/bookcrossing_data)

!(let $atom (match &temp ($fact $stv) (: (fact:- $fact) $fact $stv)) (add-atom &res1 $atom))
!(let $atom (match &temp ($fact $stv) $fact) (add-atom &purifiedDbSpace $atom))


; !(memberChecker (engagement_level $x "high") () purifiedDbSpace 3)
; !(sup-eval purifiedDbSpace (engagement_level $x "high") 3)
; !(memberChecker (engagement_level $x "low") ( (engagement_level $x "low") (engagement_level $x "med")) purifiedDbSpace 3)
; !(let $ana (customUnique (match purifiedDbSpace ($link $x $y) (createSpecials ($link $x $y))) purifiedDbSpace 3) (--- :- --- (unique_combinations_star $ana)))

; ---------------------------------------------------
;; mine frequent patterns without surprissingness 
; ---------------------------------------------------
;; minimum support 3 -3rd arg
;; number of conjuncts 3 -4th arg

!(pattern-miner &purifiedDbSpace 900 5) 
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
