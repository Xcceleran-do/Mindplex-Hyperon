!(import! &rules rules)
!(import! &facts facts)


(= (removeAnyMetricsAtom $superset $subsets)
     (let ($x $subset) ((superpose $superset) (superpose $subsets))
          (unify 
               $subset
               $x
               (let $a (subtraction-atom $superset ($subset)) (union-atom $a ($subset))) 
               (empty)
          )
     )
)
; !(removeAnyMetricsAtom ((engagement $p41 high) (length $p41 low) (reputation $r45 low)) ((engagement $j $k) (reputation $_ $__) (popularity $___ $____)))

(= (convertIncomingDataHelper2 $organized)
     (if (> (size-atom $organized) 1)
          (let ($head $tail) (decons-atom $organized) (-> $head (convertIncomingDataHelper2 $tail)))
          (car-atom $organized)
     )
)
; !(convertIncomingDataHelper ((engagement $p41 high) (length $p41 low) (reputation $r45 low)))

(= (convertIncomingDataHelper $data)
     (convertIncomingDataHelper2 (removeAnyMetricsAtom $data ((engagement $j $k) (reputation $_ $__) (popularity $___ $____))))
)
; !(convertIncomingDataHelper ((engagement $p41 high) (length $p41 low) (popularity $re45 low)))
     
     
(= (main $rules) 
     (let (supportOf $rule $num) (superpose $rules)
          (let $formattedRule (convertIncomingDataHelper (cdr-atom $rule)) (: (rule:- $formattedRule) (convertIncomingDataHelper (cdr-atom $rule))))
     )
)



;; Define cast functions between Nat and Number
(= (fromNumber $n) (if (<= $n 0) Z (S (fromNumber (- $n 1)))))

(= (fromNumberInterface $n) (trace! (--> (fromNumber (- $n 1)) is what i've get) (fromNumber (- $n 1))))

;; Base case
(= (backward-chain_ True $kb $_ (: $prf $ccln)) (match $kb (: $prf $ccln) (: $prf $ccln)))

;; Recursive step
(= (backward-chain_ True $kb (S $k) (: ($prfabs $prfarg) $ccln))
   (let* 
          (
               ((: $prfabs (-> $prms $ccln)) (backward-chain_ True $kb $k (: $prfabs (-> $prms $ccln))))
               ((: $prfarg $prms) (backward-chain_ True $kb $k (: $prfarg $prms)))
          )
          (: ($prfabs $prfarg) $ccln)
     )
)

(= (backward-chain $kb $depth (: $prf $ccln)) 
     (backward-chain_ True $kb (S (S (S (Z)))) (: $prf $ccln))
)

; !(fromNumber 5)

;! (bind! &kb (new-space))
;! (add-reduct &kb (get-atoms &facts))
;! (add-reduct &kb (main (collapse (get-atoms &rules))))

;!(get-atoms &kb)
;!(backward-chain &kb (fromNumber 5) (: $prf (engagement 1 $what)))
;[(: ((, (engagement 1 high) (topic 1 AI)) (topic 1 AI)) (engagement 1 high)),  (: ((, (engagement 1 high) (length 1 low)) (length 1 low)) (engagement 1 high))]



; ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; ;; Examples of both backward and forward curried chainers.             ;;
; ;;                                                                     ;;
; ;; The curried forward chainer relies on the curried backward chainer  ;;
; ;; to unfold the rules.                                                ;;
; ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

; ;;;;;;;;;
; ;; Nat ;;
; ;;;;;;;;;

; ;; Define Nat
; (: Nat Type)
; (: Z Nat)
; (: S (-> Nat Nat))

; ;; Define cast functions between Nat and Number
; (: fromNumber (-> Number Nat))
; (= (fromNumber $n) (if (<= $n 0) Z (S (fromNumber (- $n 1)))))
; (: fromNat (-> Nat Number))
; (= (fromNat Z) 0)
; (= (fromNat (S $k)) (+ 1 (fromNat $k)))

; ;;;;;;;;;;;;;;;;;;;;
; ;; Knowledge base ;;
; ;;;;;;;;;;;;;;;;;;;;

; ;; Simple knowledge base for expository purpose.  Do not confuse `->`,
; ;; the MeTTa arrow type, and `→`, a regular MeTTa symbol denoting a
; ;; relationship.

; !(bind! &kb (new-space))

; !(add-atom &kb (: a A))
; !(add-atom &kb (: ab (→ A B)))
; !(add-atom &kb (: bc (→ B C)))
; !(add-atom &kb (: ModusPonens
;                   (-> (→ $p $q)           ; Premise 1
;                       (-> $p              ; Premise 2
;                           $q))))          ; Conclusion

; ;;;;;;;;;;;;;;;;;;;;; 
; ;; Backward chainer ;; 
; ;;;;;;;;;;;;;;;;;;;;;

; ;; Tail-recursive backward chainer using an explicit task stack.

; (: List Type)
; (: Nil List)
; (: Cons (-> $a List List))

; (: bc-tail (-> $a Nat $b $b))
; (= (bc-tail $kb $k $query)
;    (bc-loop $kb (Cons (task prove $k $query) Nil) Nil))

; ;; 1. SUCCESS: stack empty, return whatever was proved.
; (= (bc-loop $kb Nil $acc) $acc)

; ;; 2. PROVE: try direct match in knowledge base.
; (= (bc-loop $kb (Cons (task prove $k (: $prf $ccln)) Nil) $acc)
;    (match $kb (: $prf $ccln) (: $prf $ccln)))
; (= (bc-loop $kb (Cons (task prove $k (: $prf $ccln)) (Cons $next $tail)) $acc)
;    (bc-loop $kb (Cons $next $tail) (match $kb (: $prf $ccln) (: $prf $ccln))))

; ;; 3. PROVE: decompose proof term when depth allows.
; (= (bc-loop $kb (Cons (task prove (S $k) (: ($prfabs $prfarg) $ccln)) $rest) $acc)
;    (bc-loop $kb
;       (Cons (task prove $k (: $prfabs (-> $prms $ccln)))
;          (Cons (task seek $k $prfarg $prms $ccln $rest) Nil))
;       $acc))

; ;; 4. SEEK: after rule resolved, prove its premise.
; (= (bc-loop $kb (Cons (task seek $k $prfarg $prms $ccln $saved-rest) $rest) (: $resolved-abs (-> $prms $ccln)))
;    (bc-loop $kb
;       (Cons (task prove $k (: $prfarg $prms))
;          (Cons (task assemble $k $resolved-abs $ccln $saved-rest) Nil))
;       Nil))

; ;; 5. ASSEMBLE: stitch rule and premise, resume prior stack.
; (= (bc-loop $kb (Cons (task assemble $k $resolved-abs $ccln Nil) $rest) (: $resolved-arg $prms))
;    (: ($resolved-abs $resolved-arg) $ccln))
; (= (bc-loop $kb (Cons (task assemble $k $resolved-abs $ccln (Cons $saved-next $saved-tail)) $rest) (: $resolved-arg $prms))
;    (bc-loop $kb (Cons $saved-next $saved-tail) (: ($resolved-abs $resolved-arg) $ccln)))

; ;; Test curried backward chainer
; !(assertEqual
;   (bc-tail &kb (fromNumber 0) (: $prf A))
;   (: a A))
; !(assertEqual
;   (bc-tail &kb (fromNumber 2) (: $prf B))
;   (: ((ModusPonens ab) a) B))
; !(assertEqual
;   (bc-tail &kb (fromNumber 3) (: $prf C))
;   (: ((ModusPonens bc) ((ModusPonens ab) a)) C))