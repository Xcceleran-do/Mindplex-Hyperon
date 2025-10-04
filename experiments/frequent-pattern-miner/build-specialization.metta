! (import! &self freq-pat)

---------------------------------------
;; dontExist
---------------------------------------
;; Function to check if a specialized atom does not already exist in a pool of specialized atoms.
;; Iterates through the pool and compares each atom with the selected atom using alpha equivalence.
;; Returns False if an equivalent atom is found, otherwise returns True.
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
;(: depth-handler (-> Pattern Atom)) ;TODO: can't we do better? I don't get what it is for but it is slow
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


