!(bind random-int (py-atom random.random))
;; Helper functions that are going to be useful for writing any metta code
(= (!= $a $b) (not (== $a $b)))

;; INFO: Uncomment for the use of MeTTaLog.
;; (: >= (-> Number Number Bool))
;; (= (>= $a $b) (or (== $a $b) (> $a $b)))

;Function to cocatinate two tuples (A B) (C D) ==> (A B C D)
;; (: concatTuple (-> Expression Expression Expression))
(= (concatTuple $x $y) (union-atom $x $y)) 

; a function which checks if an element is a member of a tuple.
(= (isMember $x $list)
    (not (== (intersection-atom ($x) $list) ()))
)

;; A function to replace a specific atom
;;    with a new one from a list of atoms.
(= (findAndReplace $old $new $list) 
      (map-atom $list $x (eval (if-equal $x $old $new $x)))
)

;; A function that behaves like a do while loop.
;; It executes the functions in the given list and returns the last result.
(= (until $predicate $function $x) 
(function
    (chain (eval ($predicate $x)) $b 
    (chain (eval (if-equal $b False 
    (chain (eval ($function $x)) $x' 
    (chain (eval (until $predicate $function $x')) $1 $1)) $x)) $res 
    (return $res)))))


;; Fold a tuple from right to left
(= (foldr $f $i $xs) ;; TODO: I think the minimal metta works with tail recursion only. I couldn't find a work around to make the foldr a tail recursion. So I used this.
    (function 
        (chain (eval (reverse $xs)) $rev
        (chain (eval (foldl-atom $rev $i $in $acc ($f $in $acc))) $res
        (return $res))))
         )

;; Function: wrapper
;; Description: This function applies a predicate to an atom and returns the atom if the predicate is true, otherwise returns an empty expression.
;; Parameters:
;;   - $predicate: A function that takes an atom and returns a boolean value.
;;   - $x: The atom to which the predicate is applied.
;; Returns:
;;   - Expression: Returns the atom if the predicate evaluates to true, otherwise returns an empty expression.
(= (wrapper $predicate $x) 
    (function 
        (chain (eval ($predicate $x)) $bool
        (chain (eval (if-equal True $bool ($x) ())) $r (return $r)))))

;; Filter function given a predicate
(= (filter $predicate $list) (filter $predicate $list ()))
(= (filter $predicate $list $acc)
    (function 
        (chain (eval (if-decons-expr $list $h $t
                (chain (eval (wrapper $predicate $h)) $r
                (chain (eval (union-atom $acc $r)) $nacc
                (chain (eval (filter $predicate $t $nacc)) $r1
                $r1))) $acc)) $res
                (return $res))))

;; (:wrapperB (-> (-> $a $b Bool) $a $b $b))
(= (wrapperB $predicate $x $y) 
    (function 
        (chain (eval ($predicate $x $y)) $bool
        (chain (eval (if-equal True $bool ($y) ())) $r (return $r)))))

(= (filterB $predicate $x $list) (filterB $predicate $x $list ()))
(= (filterB $predicate $x $list $acc)
   (function 
        (chain (eval (if-decons-expr $list $h $t
                (chain (eval (wrapperB $predicate $x $h)) $r
                (chain (eval (union-atom $acc $r)) $nacc
                (chain (eval (filterB $predicate $x $t $nacc)) $r1
                $r1))) $acc)) $res
                (return $res))))

;; A function to remove all elements found in $common from a $tuple
(= (removeElement $common $tuple)
   (subtraction-atom $tuple $common))

; ;; A function that returns True if any of the input atoms are True and False otherwise
; ;; Example: input ==> (True True False), output ==> True
; ;;          input ==> (False False False), output ==> False
; ;;          input ==> (False True False), output ==> True
(: any (-> Atom Bool))
(= (any $bools) (isMember True $bools))

(= (sum $list) (foldl-atom $list 0 $acc $x (+ $x $acc)))

;; Function to count atom occrence in a list of atoms.
(= (count $atom $list)
   (foldl-atom $list 0 $acc $x (eval (if-equal $atom $x (+ $acc 1) $acc))))

;; A function to check if a given atom is symbol.
(= (isSymbol $expr) 
    (function
        (chain (eval (get-metatype $expr)) $type
        (chain (eval (== $type Symbol)) $r
        (return $r)))))

;; A function to check if a given atom is expression.
(: isExpression (-> Atom Bool))
(= (isExpression $expr)
    (function
        (chain (eval (get-metatype $expr)) $type
        (chain (eval (== $type Expression)) $r
        (return $r)))))

;; A function to check if a given atom is unit or empty tuple.
(: isEmpty (-> Atom Bool))
(= (isEmpty $expr) 
    (function
        (chain (eval (== $expr ())) $r
        (return $r))))

(: isEven (-> Number Bool))
(= (isEven $x) 
    (function
        (chain (eval (% $x 2)) $res
        (chain (eval (== $res 0)) $r
        (return $r)))))

(: isOdd (-> Number Bool))
(= (isOdd $x) 
    (function
        (chain (eval (% $x 2)) $res
        (chain (eval (== $res 1)) $r
        (return $r)))))

;; A function to check if a given atom is an expression 
;;   with has one atom in it.
(: hasOneAtom (-> Atom Bool))
(= (hasOneAtom $expr) 
    (function 
        (chain (eval (unify $expr ($x) True False)) $r
        (return $r))))

;; A function which check if a given number is postive
(= (isPositive $x)                                                 
    (function                                                      
        (chain (eval (> $x 0)) $res                                
            (return $res))))         

;; A function which subtracts 1 to a given number                                          
(= (-- $x)                                                                 
    (function 
        (chain (eval (- $x 1)) $res 
            (return $res))))

;; A function which adds 1 to a given number   
(= (++ $x)                                                                 
    (function 
        (chain (eval (+ $x 1)) $res 
            (return $res))))            

;; Function to replace a given atom from a space
(: update-atom (-> Grounded Atom Atom (->)))
(= (update-atom $space $oatom $natom) 
    (function
    (chain (eval (remove-atom $space $oatom)) $_ 
    (chain (eval (add-atom $space $natom)) $r
    (return $r)))))

;; Function to add a given atom to an expr.
(: appendAtom (-> Atom Atom Atom))
(= (appendAtom $a $atom) 
    (function
    (chain (eval (union-atom $a ($atom))) $res
    (return $res))))

;; A function to return a tuple of numbers from natural to a given number.
;; $x is the upperbound
;; $acc is the lowerbound wrapped with as a tuple.
;;    Example: (gen 5 ()) or (gen 5) => (5 4 3 2 1 0)
(= (gen $x) (gen $x ()))
(= (gen $x $acc) 
    (function 
        (chain (eval (if-equal $x 0 (chain (eval (cons-atom 0 $acc)) $f 
                $f)
            (chain (eval (cons-atom $x $acc)) $nacc
            (chain (eval (- $x 1)) $nxt
            (chain (eval (gen $nxt $nacc)) $res $res))))) $final 
                (return $final))))

;; Function to create a tuple of tuples given two equal length list of atoms.
(= (zip $a $b) (zip $a $b ()))
(= (zip $a $b $acc)
    (function 
        (chain (eval (if-decons-expr $a $ah $at 
            (chain (eval (if-decons-expr $b $bh $bt 
                (chain ($ah $bh) $cur
                (chain (eval (union-atom $acc ($cur))) $nacc
                (chain (eval (zip $at $bt $nacc)) $r $r))) ())) $r1 $r1) $acc)) $f 
                (return $f))))

;; Similar to zip but instead of just creating a tuple of atoms,
;;  it performes an operation of $f on them and returns the resulting list of atoms.
(= (zipWith $fn $a $b) (zipWith $fn $a $b ()))
(= (zipWith $fn $a $b $acc)
    (function 
        (chain (eval (if-decons-expr $a $ah $at 
            (chain (eval (if-decons-expr $b $bh $bt 
                (chain (eval ($fn $ah $bh)) $cur
                (chain (eval (union-atom $acc ($cur))) $nacc
                (chain (eval (zipWith $fn $at $bt $nacc)) $r $r))) ())) $r1 $r1) $acc)) $f 
                (return $f))))

;; Finds length of a tuple
(= (len $expr)
    (foldl-atom $expr 0 $acc $x (eval (+ $acc 1))))

;; A function to replace an atom at a specific index position with $val -- 0 indexed counting is implied
; (: replaceByIndex (-> Expression Number $t Expression Expression))
(= (replaceByIndex $expr $index $new) (replaceByIndex $expr $index $new ()))
(= (replaceByIndex $expr $index $new $acc)
    (function
        (chain (eval (if-decons-expr $expr $head $tail
        (chain (eval (if-equal $index 0 
                    (chain (eval (cons-atom $new $tail)) $f 
                    (union-atom $acc $f))
                    (chain (eval (- $index 1)) $nxtInx
                    (chain (eval (union-atom $acc ($head))) $nacc
                    (chain (eval (replaceByIndex $tail $nxtInx $new $nacc)) $r
                    $r))))) $res $res) $acc)) $r1 
                    (return $r1))))
                    
;; Function to empty a space with contents.
(= (clearSpace $space)
   (collapse (let $content (get-atoms $space) (remove-atom $space $content))))

;; function to partially sort the smallest $n Atoms from an expression
(= (selectionSort $list $n) 
    (function
    (chain (eval (selectionSort $list $n < ())) $r
    (return $r))))
(= (selectionSort $list $n $op $accum)
    (function 
    (chain (eval (== $n 0)) $bool
    (chain (unify $bool True
        (eval (union-atom $accum $list))
        (chain (eval (if-decons-expr $list $initial $tail
        (chain (eval (selector $list $initial $op)) $cur
        (chain (eval (subtraction-atom $list ($cur))) $newList
        (chain (eval (- $n 1)) $next
        (chain (eval (union-atom $accum ($cur))) $acc
        (chain (eval (selectionSort $newList $next $op $acc)) $rest $rest)))))
        $accum)) $r2 $r2)
    ) $r1 
    (return $r1)
    )
    )))

(= (selector $expr $i $op) 
    (function 
        (chain (eval (foldl-atom $expr $i $x $acc (chain (eval (comparator $x $acc $op)) $selected $selected))) $res (return $res)))
)

(= (comparator $x $y $op) 
    (function
    (chain (eval ($op $x $y)) $bool 
    (unify $bool True 
        (return $x) 
        (return $y)))))


;; A function to take an atom and repeat it n times.
(= (repeat $a $n) (repeat $a $n ()))
(= (repeat $a $n $acc)
   (function
        (chain (eval (if-equal $n 0 $acc 
            (chain (eval (cons-atom $a $acc)) $nacc
            (chain (eval (- $n 1)) $nxt
            (chain (eval (repeat $a $nxt $nacc)) $r $r))))) $res 
            (return $res)
    )))


;; A function to pick the frist n elements of a tuple 
;;   and returns it.
(= (take $n $a) (take $n $a ()))
(= (take $n $a $acc)
    (function 
    (chain (eval (if-decons-expr $a $h $t 
        (chain (eval (if-equal $n 0 $acc
            (chain (eval (union-atom $acc ($h))) $nacc
            (chain (eval (- $n 1)) $nxt
            (chain (eval (take $nxt $t $nacc)) $r $r))))) $r1 $r1) 
            $acc)) $res 
            (return $res))))

;; A function to remove the frist n elements of a tuple 
;;   and returns the rest.
(= (drop $n $a)
    (function 
    (chain (eval (if-decons-expr $a $h $t 
        (chain (eval (if-equal $n 0 $a
            (chain (eval (- $n 1)) $nxt
            (chain (eval (drop $nxt $t)) $r $r)))) $r1 $r1) 
            ())) $res 
            (return $res))))

;; a function that checks that all of the expressions are true.
(: all (-> Expression Bool))
(= (all $expressions) (if (isMember False $expressions) False True))


(: isInf (-> Number Bool))
(= (isInf $x) (type-cast (isinf-math $x) Bool &self))

;; a function to reverse a given expression
(= (reverse $expr)
    (function 
        (chain (eval (foldl-atom $expr () $acc $i (cons-atom $i $acc))) $rev
        (return $rev))))

;; a function which generates an exper from the $min to $max
(= (range $min $max) (range $min $max ()))
(= (range $min $max $acc) 
    (function 
        (chain (eval (> $min $max)) $bool 
        (chain (eval (if-equal $bool True 
            $acc
            (chain (eval (cons-atom $max $acc)) $nacc
            (chain (eval (- $max 1)) $nxt
            (chain (eval (range $min $nxt $nacc)) $res $res))))) $final 
                (return $final)))))

(= (range' $max $min) (range' $max $min ()))
(= (range' $max $min $acc) 
    (function 
        (chain (eval (> $min $max)) $bool 
        (chain (eval (if-equal $bool True 
            $acc
            (chain (eval (cons-atom $min $acc)) $nacc
            (chain (eval (+ $min 1)) $nxt
            (chain (eval (range' $max $nxt $nacc)) $res $res))))) $final 
                (return $final)))))


; A function which return true if the second atom of the first experation is greater than the second's
(: bySecondAtom (-> ($a Number) ($b Number) Bool))
(= (bySecondAtom $x $y) 
    (function
    (chain (eval (index-atom $x 1)) $xSecond
    (chain (eval (index-atom $y 1)) $ySecond
    (chain (eval (> $xSecond $ySecond)) $bool
    (return $bool))))
    ))

;; A function to randomly pick an atom from a given expression
(: randomAtom (-> Expression Atom)) 
(= (randomAtom $expr)
    (function 
        (chain (eval (size-atom $expr)) $size
        (chain (eval (random-int &rng 0 $size)) $randIdx
        (chain (eval (index-atom $expr $randIdx)) $rand
        (return ($rand $randIdx)))))))

;; This function takes an expression of expressions and returns it flatten
(= (flatten $expr)
    (foldl-atom $expr () $acc $x (eval (union-atom $acc $x))))
