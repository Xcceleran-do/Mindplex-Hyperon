! (register-module! ../../../experiments)

! (import! &self experiments:pattern-miner:pattern-miner)
! (import! &self experiments:utils:common-utils)
! (import! &self experiments:frequent-pattern-miner:frequent-pattern-miner)
! (import! &db experiments:data:small-ugly)

;; create the database 
!(bind! &dbb (new-space))

;; space to hold the miner result
!(bind! &res1 (new-space))

;(: space hyperon::space::DynSpace)

---------------------------------------------------
;; mine frequent patterns without surprissingness 
---------------------------------------------------
;; number of conjucts 2 , so the depth will be 0
;; surprissingness mode none
;; minimum support 3

!(let 
    $answer (pattern-miner &res1 &db 3 0) 
    (that is it we have the answer:- $answer)) 
!(println! "============ end ================")
; !(collapse (match &res1 $pattern $pattern )) ; de-comment the line where add-atom is used and u can de-comment this line to see the result
;; Expected result 
          ; ((supportOf (, (Inheritance $y human) (Inheritance $y man)) 3) 
          ; (supportOf (, (Inheritance $j human) (Inheritance $j ugly)) 3) 
          ; (supportOf (, (Inheritance $p human) (Inheritance $p sodaDrinker)) 3) 
          ; (supportOf (, (Inheritance $r man) (Inheritance $r ugly)) 3) 
          ; (supportOf (, (Inheritance $x man) (Inheritance $x sodaDrinker)) 3) 
          ; (supportOf (, (Inheritance $x sodaDrinker) (Inheritance $x ugly)) 3) 
          ; (supportOf (, (Inheritance Abe $r) (Inheritance Cason $r)) 4) (
          ;   supportOf (, (Inheritance Abe $l) (Inheritance Allen $l)) 4) 
          ;   (supportOf (, (Inheritance Allen $g) (Inheritance Cason $g)) 4))


----------------------------------------------------------------
;; mine frequent patterns without surprissingness (3 - conjunct)
----------------------------------------------------------------
;; number of conjucts 3 , so the depth will be 1
;; surprissingness mode none
;; minimum support 3

!(bind! &res2 (new-space))

; !(let 
;     $answer (pattern-miner &res1 &db 3 1) 
;     (that is it we have the answer:- $answer)) 
; !(println! "============ end ================")

; !(pattern-miner &res2 &db 3 1) 
; !(collapse (match &res2 $pattern $pattern )) ; de-comment the line where add-atom is used and u can de-comment this line to see the result
;; Expected result
; ((supportOf (, (Inheritance $a88#3981498 human) (Inheritance $a88#3981498 man) (Inheritance $a88#3981498 ugly)) 3) 
; (supportOf (, (Inheritance $t39#3981499 human) (Inheritance $t39#3981499 man) (Inheritance $t39#3981499 sodaDrinker)) 3) 
; (supportOf (, (Inheritance $v02#3981500 human) (Inheritance $v02#3981500 sodaDrinker) (Inheritance $v02#3981500 ugly)) 3) 
; (supportOf (, (Inheritance $v54#3981501 man) (Inheritance $v54#3981501 sodaDrinker) (Inheritance $v54#3981501 ugly)) 3) 
; (supportOf (, (Inheritance Abe $v37#3981502) (Inheritance Allen $v37#3981502) (Inheritance Cason $v37#3981502)) 4))

