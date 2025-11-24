;; Helper functions that are going to be useful for writing any metta code
(= (!= $a $b) (not (== $a $b)))

;; INFO: Uncomment for the use of MeTTaLog.
;; (: >= (-> Number Number Bool))
;; (= (>= $a $b) (or (== $a $b) (> $a $b)))

;Function to cocatinate two tuples (A B) (C D) ==> (A B C D)
(: concatTuple (-> Expression Expression Expression))
(= (concatTuple $x $y) (union-atom $x $y)) 

; a function which checks if an element is a member of a tuple.
; (: isMember (-> Atom Expression Bool)) ;; TODO: This is not working in PeTTa
(= (isMember $x $list)
    (not (== (intersection-atom ($x) $list) ()))
)

;; A function to replace a specific atom
;;    with a new one from a list of atoms.
; (: findAndReplace (-> Atom Atom Expression Expression));; TODO: this is creating unkown error in PeTTa
(= (findAndReplace $old $new $list) 
      (map-atom $list $x (if (== $x $old) $new $x))
)

;; A function that behaves like a do while loop.
;; It executes the functions in the given list and returns the last result.
; (: until -> (-> $a Bool) (-> $a $a) $a $a)
(= (until $predicate $function $x) 
    (let $b ($predicate $x) 
        (if $b $x
            (let $x' ($function $x) 
                (let $1 (until $predicate $function $x') $1)))) 
        )


;; Fold a tuple from right to left
;(: foldr (-> Atom Atom Expression Atom)) ;; TODO: This is creating error in PeTTa
(= (foldr $f $i $xs)
        (let $rev (reverse $xs) (foldl-atom $rev $i $in $acc ($f $in $acc))))

;; Checks if first argument is non empty expression. If so gets tail and head from the first argument and returns forth argument using head and tail values. Returns fifth argument otherwise.
(: if-decons-expr (-> Expression Variable Variable Atom Atom Atom))
(= (if-decons-expr $atom $head $tail $then $else)
  (function (eval (if-equal $atom ()
    (return $else)
    (chain (decons-atom $atom) $list
      (unify $list ($head $tail) (return $then) (return $else)) )))))
    
;; Function: wrapper
;; Description: This function applies a predicate to an atom and returns the atom if the predicate is true, otherwise returns an empty expression.
;; Parameters:
;;   - $predicate: A function that takes an atom and returns a boolean value.
;;   - $x: The atom to which the predicate is applied.
;; Returns:
;;   - Expression: Returns the atom if the predicate evaluates to true, otherwise returns an empty expression.
;(: wrapper (-> (-> $a Bool) $a Atom))
(= (wrapper $predicate $x) 
        (let $bool ($predicate $x)
        (if $bool ($x) ())))

;; Filter function given a predicate
(= (filter $predicate $list) (filter $predicate $list ()))
(= (filter $predicate $list $acc)
    (if (== $list ()) $acc
            (let ($h $t) (decons-atom $list)
                (let $r (wrapper $predicate $h)
                (let $nacc (union-atom $acc $r)
                (filter $predicate $t $nacc))))))

;; (:wrapperB (-> (-> $a $b Bool) $a $b $b))
(= (wrapperB $predicate $x $y) 
    (let $bool ($predicate $x $y) 
    (if $bool ($x) ())))

(= (filterB $predicate $x $list) (filterB $predicate $x $list ()))
(= (filterB $predicate $x $list $acc)
   (if (== $list ()) $acc 
            (let ($h $t) (decons-atom $list)
                (let $r (wrapperB $predicate $h $x)
                (let $nacc (union-atom $acc $r)
                (filterB $predicate $x $t $nacc))))))

;; A function to remove all elements found in $common from a $tuple
(: removeElement (-> Expression Expression Expression))
(= (removeElement $common $tuple)
   (subtraction-atom $tuple $common))

; ;; A function that returns True if any of the input atoms are True and False otherwise
; ;; Example: input ==> (True True False), output ==> True
; ;;          input ==> (False False False), output ==> False
; ;;          input ==> (False True False), output ==> True
;(: any (-> Atom Bool))
(= (any $bools) (isMember True $bools))

(: sum (-> Expression Number))
(= (sum $list) (foldl-atom $list 0 $x $acc (+ $x $acc)))

;; Function to count atom occrence in a list of atoms.
(: count (-> $Atom Expression Number))
(= (count $atom $list)
   (foldl-atom $list 0 $acc $x  (if (== $atom $x) (+ $acc 1) $acc)))

;; A function to check if a given atom is symbol.
;(: isSymbol (-> Atom Bool))
(= (isSymbol $expr) 
    (let $type (get-metatype $expr)
        (== $type Symbol)))

;; A function to check if a given atom is expression.
;(: isExpression (-> Atom Bool))
(= (isExpression $expr)
    (let $type (get-metatype $expr)
        (== $type Expression)))

;; A function to check if a given atom is unit or empty tuple.
;(: isEmpty (-> Atom Bool))
(= (isEmpty $expr) 
    (== $expr ()))

(: isEven (-> Number Bool))
(= (isEven $x) 
    (let $res (% $x 2)
        (== $res 0)))

(: isOdd (-> Number Bool))
(= (isOdd $x) 
    (let $res (% $x 2) (== $res 1)))

;; A function to check if a given atom is an expression 
;;   with has one atom in it.
; (: hasOneAtom (-> Atom Bool))
(= (hasOneAtom $expr) 
    (== (size-atom $expr) 1))

;; A function which check if a given number is postive
(: isPositive (-> Number Bool))
(= (isPositive $x) (> $x 0))

;; A function which subtracts 1 to a given number                                          
(= (-- $x) (- $x 1)) 

;; A function which adds 1 to a given number   
(= (++ $x) (+ $x 1))          

;; Function to replace a given atom from a space
(: update-atom (-> Grounded Atom Atom (->)))
(= (update-atom $space $oatom $natom) 
    (function
    (chain (eval (remove-atom $space $oatom)) $_ 
    (chain (eval (add-atom $space $natom)) $r
    (return $r)))))

;; Function to add a given atom to an expr.
;(: appendAtom (-> Atom Atom Atom))
(= (appendAtom $a $atom) 
    (union-atom $a ($atom)))

;; A function to return a tuple of numbers from natural to a given number.
;; $x is the upperbound
;; $acc is the lowerbound wrapped with as a tuple.
;;    Example: (gen 5 ()) or (gen 5) => (5 4 3 2 1 0)
(= (gen $x) (gen $x ()))
(= (gen $x $acc) 
    (if (== $x 0) 
            (cons-atom 0 $acc)
            (let $nacc (cons-atom $x $acc)
            (gen (- $x 1) $nacc))))

;; Function to create a tuple of tuples given two equal length list of atoms.
(= (zip $a $b) (zip $a $b ()))
(= (zip $a $b $acc)
    (if (== $a ()) $acc
        (let ($ah $at) (decons-atom $a)
            (if (== () $b) ()
            (let ($bh $bt) (decons-atom $b)
                (let $cur ($ah $bh) 
                (let $nacc (union-atom $acc ($cur))
                (zip $at $bt $nacc))))
                ))))

;; Similar to zip but instead of just creating a tuple of atoms,
;;  it performes an operation of $f on them and returns the resulting list of atoms.
(= (zipWith $fn $a $b) (zipWith $fn $a $b ()))
(= (zipWith $fn $a $b $acc)
    (if (== $a ()) $acc
        (let ($ah $at) (decons-atom $a)
            (if (== () $b) ()
                (let ($bh $bt) (decons-atom $b)
                (let $cur ($fn $ah $bh)
                (let $nacc (union-atom $acc ($cur))
                (zipWith $fn $at $bt $nacc))))))))

;; A function to replace an atom at a specific index position with $val -- 0 indexed counting is implied
; (: replaceByIndex (-> Expression Number $t Expression Expression))
(= (replaceByIndex $expr $index $new) (replaceByIndex $expr $index $new ()))
(= (replaceByIndex $expr $index $new $acc)
    (if (== $expr ()) $acc 
        (let ($head $tail) (decons-atom $expr)
        (if (= $index 0)
            (let $f (cons-atom $new $tail)
                (union-atom $acc $f))
            (let $nacc (union-atom $acc ($head))
            (replaceByIndex $tail (- $index 1) $new $nacc))))))
                    
;; Function to empty a space with contents.
(= (clearSpace $space)
   (collapse (let $content (get-atoms $space) (remove-atom $space $content))))

;; function to partially sort the smallest $n Atoms from an expression
;(: selectionSort (-> Expression Number Expression))
(= (selectionSort $list $n) 
    (selectionSort $list $n < ()))

(= (selectionSort $list $n $op $accum)
    (if (== $n 0)
        (union-atom $accum $list)
        (if (== () $list) $accum
            (let ($initial $tail) (decons-atom $list)
            (let $cur (selector $list $initial $op)
            (let $newList (subtraction-atom $list ($cur))
            (let $acc (union-atom $accum ($cur))
            (selectionSort $newList (- $n 1) $op $acc))))))))

(= (selector $expr $i $op) 
    (foldl-atom $expr $i $x $acc (comparator $x $acc $op))
)

(= (comparator $x $y $op) 
    (if ($op $x $y) $x $y))


;; A function to take an atom and repeat it n times.
(= (repeat $a $n) (repeat $a $n ()))
(= (repeat $a $n $acc)
   (if (== $n 0) $acc
        (let $nacc (cons-atom $a $acc) 
            (repeat $a (- $n 1) $nacc))))

;; A function to pick the frist n elements of a tuple 
;;   and returns it.
(= (take $n $a) (take $n $a ()))
(= (take $n $a $acc)
    (if (== () $a) $acc
        (let ($h $t) (decons-atom $a)
        (if (== $n 0) $acc
            (let $nacc (union-atom $acc ($h))
            (take (- $n 1) $t $nacc))))))

;; A function to remove the frist n elements of a tuple 
;;   and returns the rest.
(: drop (-> Number Expression Expression))
(= (drop $n $a)
    (if (== () $a) ()
        (let ($h $t) (decons-atom $a)
        (if (== $n 0) $a
            (drop (- $n 1) $t)))))

;; a function that checks that all of the expressions are true.
(: all (-> Expression Bool))
(= (all $expressions) (if (isMember False $expressions) False True))


(: isInf (-> Number Bool))
(= (isInf $x) (type-cast (isinf-math $x) Bool &self))


;; a function which generates an exper from the $min to $max
(= (range $min $max) (range $min $max ()))
(= (range $min $max $acc) 
    (if (> $min $max) 
            $acc
            (let  $nacc (cons-atom $max $acc)
            (range $min (- $max 1) $nacc))))

(= (range' $max $min) (range' $max $min ()))
(= (range' $max $min $acc) 
    (if (> $min $max)
            $acc
            (let $nacc (cons-atom $min $acc)
            (range' $max (+ $min 1) $nacc))))

; A function which return true if the second atom of the first experation is greater than the second's
(: bySecondAtom (-> ($a Number) ($b Number) Bool))
(= (bySecondAtom $x $y) 
    (let $xSecond (index-atom $x 1)
    (let $ySecond (index-atom $y 1)
    (> $xSecond $ySecond))))

;; A function to randomly pick an atom from a given expression
(: randomAtom (-> Expression Atom)) 
(= (randomAtom $expr)
    (let $size (size-atom $expr)
        (let $randIdx (random-int &rng 0 $size)
        (let $rand (index-atom $expr $randIdx)
        ($rand $randIdx)))))

;; a function to compute the cartesian product
(: cartesian (-> Expression Expression Expression))
(= (cartesian $list-a $list-b)
    (foldl-atom $list-a () $acc-outer $x
        (eval (foldl-atom $list-b $acc-outer $acc-inner $y
            (eval (union-atom $acc-inner (($x $y))))
        ))
    )
)

(= (cartesianAppend $combinations $new-list)
    (foldl-atom $combinations () $acc-outer $combo
        (foldl-atom $new-list $acc-outer $acc-inner $new-element
            (eval (union-atom $acc-inner ((appendAtom $combo $new-element))))
        )
    )
)

;; Computes cartesian for any number of tuples greater than or equal to two. 
;; Input - List of tuples more than or equal to 2
(: multiCartesian Expression Expression)
(= (multiCartesian $list-of-tuples)
   (foldl-atom $list-of-tuples (()) $acc $x (cartesianAppend $acc $x))
)
