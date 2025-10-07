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
     
(= (convertIncomingData $data)
     (convertIncomingDataHelper (let $x (index-atom $data 1) (cdr-atom $x)))
)
; !(convertIncomingData (supportOf (, (popularity $x medium) (engagement_level $x low) (reputation $x medium) (writing_style $x "Academic")) 6))


(= (main $kb) 
     (let ($rule $rand) ((get-atoms $kb) (random-int 0 1e10))
          (: SomeRule (convertIncomingData $rule))
     )
)

; !(main &rules)


;; Define cast functions between Nat and Number
(= (fromNumber $n) (if (<= $n 0) Z (S (fromNumber (- $n 1)))))

;; Base case
(= (backward-chain $kb $_ (: $prf $ccln)) (match $kb (: $prf $ccln) (: $prf $ccln)))

;; Recursive step
(= (backward-chain $kb (S $k) (: ($prfabs $prfarg) $ccln))
   (let* 
          (
               ((: $prfabs (-> $prms $ccln)) (backward-chain $kb $k (: $prfabs (-> $prms $ccln))))
               ((: $prfarg $prms) (backward-chain $kb $k (: $prfarg $prms)))
          )
          (: ($prfabs $prfarg) $ccln)
     )
)



!(bind! &kb (new-space))
!(add-reduct &kb (get-atoms &facts))
!(add-reduct &kb (main &rules))

; !(get-atoms &kb)
; !(backward-chain &kb (fromNumber 5) (: $prf (reputation 0 high)))
