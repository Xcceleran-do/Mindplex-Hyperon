!(import! &rules rules)
(= (convertIncomingDataHelper $organized)
     (if (> (size-atom $organized) 1)
          (let ($head $tail) (decons-atom $organized) (-> $head (convertIncomingDataHelper $tail)))
          (car-atom $organized)
     )
)

(= (main $rules) 
     (let (supportOf $rule $num) (superpose $rules)
          (let $formattedRule (convertIncomingDataHelper (cdr-atom $rule)) (: (rule:- $formattedRule) $formattedRule))
     )
)
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


!(backward-chain &rules (fromNumber 5) (: ($prfabs $prfarg) (engagement A_N41020 "Medium")))