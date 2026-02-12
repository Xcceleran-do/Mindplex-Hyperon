(= (convertIncomingDataHelper $organized)
     (if (> (size-atom $organized) 1)
          (let ($head $tail) (decons-atom $organized) (-> $head (convertIncomingDataHelper $tail)))
          (car-atom $organized)
     )
)

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