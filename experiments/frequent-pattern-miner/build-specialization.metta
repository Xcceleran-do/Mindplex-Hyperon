
! (import! &self freq-pat)

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
(= (build-specialization ($link $x $y) $dbspace $specspace) 
            (let*
                (  
                     (() (println! "==== Build specialization started ====="))
                    ( ($link $x1 $y1) (replacev ($link $x $y)))                            ; replacev is a function that replaces index with variables
                    ( ($subx $suby) (match $dbspace ($link $x1 $y1) ($x1 $y1)))            ; getting the substitution /valuation of the pattern
                    ($shabx (depth-handler $subx))                                    ;handle depth in the x node
                    ($shaby (depth-handler $suby))                                    ; handle depth in the y node                               
                    ($spec1 (replace (SpecializationOf ($link $shabx $y1) ($link $x $y)))) ;build the specialization and change the variables to indexs 
                    ($spec2 (replace (SpecializationOf ($link $x1 $shaby) ($link $x $y)))) ;build the specialization and change the variables to indexs
                )
            (
                superpose (
                    (remove-atom $specspace $spec1) ; remove the old specialization to avoid redundancy
                    (add-atom $specspace $spec1) ; add the new specialization
                    (remove-atom $specspace $spec2)
                    (add-atom $specspace $spec2)
                )

        )
)
)


