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
     (convertIncomingDataHelper2 (removeAnyMetricsAtom $data ((engagement_level $j $k) (reputation $_ $__) (popularity $___ $____))))
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
     45356
)



;! (bind! &kb (new-space))
;! (add-reduct &kb (get-atoms &facts))
;! (add-reduct &kb (main (collapse (get-atoms &rules))))

;!(get-atoms &kb)
;!(backward-chain &kb (fromNumber 5) (: $prf (engagement_level 1 $what)))
;[(: ((, (engagement_level 1 high) (topic 1 AI)) (topic 1 AI)) (engagement_level 1 high)),  (: ((, (engagement_level 1 high) (length 1 low)) (length 1 low)) (engagement_level 1 high))]