! (import! &self freq-pat)


---------------------------------------
;; dontExist
---------------------------------------
( = (dontExist $selectedAtom $poolOfSpecialized)
    (foldl-atom $poolOfSpecialized True $truthValue $currentAtom  
        (
            if (=alpha $currentAtom $selectedAtom)
                False
                $truthValue
        )
    )
)



---------------------------------------
;; depth-handler
---------------------------------------
;; function to handle the depth of the pattern
;; 1. check if the pattern is an expression
;; 2. if it is not an expression, return the pattern
;; 3. if it is an expression, destructure the pattern and handle the depth of the pattern by recursively getting all 
;; combinations of the variables and symbols of the pattern
;(: depth-handler (-> Pattern Atom))
(= (depth-handler $p)
    (if (not (== (get-metatype $p) Expression))
        $p
        (let* (
                ( ($link $x $y) $p)
                ( $nx (depth-handler $x))
                ( $ny (depth-handler $y)))
        (superpose (
                ($link $nx $w)
                ($link $z $ny)
                ($link $x $u)
                ($link $k $y)
                $d
                ($link $g $o)
                ($link $nx $ny))
    )
)))




;---------------------------------------
;; createSpecials
;---------------------------------------
;; Function to create specialized patterns from a list of substitution pairs.
;; For each pair in $shabPairs, generates two specialized atoms by substituting
;; the elements into the pattern, checks if they already exist in the accumulator,
;; and adds them if they do not exist. Returns the updated accumulator containing
;; all unique specialized atoms.

( = (createSpecials $pair $acc)
        (
            unify ($link $shabx $shaby) $pair
            (
                let* (
                        ($spec1 (Specialized ($link $shabx $y) )) 
                        ($spec2 (Specialized ($link $x $shaby) )) 
                        ($canAddSpec1 (dontExist $spec1 $acc))
                        ($canAddSpec2 (dontExist $spec2 $acc))
                        ($newAcc (if $canAddSpec1 (cons-atom $spec1 $acc) $acc))
                    )   
                    (if $canAddSpec1
                        ( if $canAddSpec2 (cons-atom $spec2 $newAcc) $newAcc )
                        ( if $canAddSpec2  (cons-atom $spec2 $acc) $acc )
                    )
            )
            (println! (================= not  created specials for $pair with =====================))
        )
)



-------------------------------------------------------------------------------
; build-specialization
-------------------------------------------------------------------------------
; function to build the specialization of a pattern and add it to the specialization space
; - get the valuation of the pattern by destructuring the pattern
; - abstract the pattern and handle patterns with depth
; - build the specialization
; - add the specialization to the specialization space
;(: build-specialization (-> Pattern SpaceType SpaceType Atom))
; (: build-specialization (-> Pattern SpaceType SpaceType Any))
(= (build-specialization ($link $x $y) $dbspace) 
    (let*
        (  
            (() (println! "==== Build specialization started ====="))
            ( $shabPairs (collapse (match $dbspace ($link $x $y) ($link (depth-handler $x) (depth-handler $y)))))               
        )
        ( foldl-atom $shabPairs () $acc $pair (createSpecials $pair $acc) )
    )
)


