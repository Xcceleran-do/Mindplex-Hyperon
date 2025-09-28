
! (import! &self freq-pat)

---------------------------------
;; candidatePattern
---------------------------------
;get candidate patterns based on theri support
;; 1. get the specialized pattern
;; 2. calculate the support of the pattern
;; 3. store as a candidate if the support is greater than the minimum support
;(: (candidatePattern (-> space space Number space Atom)))
; (:candidatePattern (-> SpaceType SpaceType Number SpaceType Any))
(= (candidatePattern  $dbspace $spezspace $minsup $cndpspace)
    (match $spezspace (SpecializationOf $specialized $pattern)
        (let* ( 
                ( $specializedvar (replacev $specialized))
                ($result (sup-eval $dbspace $specializedvar $minsup))
            )
        (if  $result
            (add-atom $cndpspace (CandidatePattern $specialized))
            ()
        )
)))
