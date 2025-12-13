; Check if a variable is in a list
(= (is-var-in-list $var $list) (if (== $list ()) False (let $head (car-atom $list) (if (== $var $head) True (is-var-in-list $var (cdr-atom $list))))))
; Check if variables from one list exist in another list
(= (vars-exist $list1 $list2) (if (== $list1 ()) False (let $head (car-atom $list1) (if (is-var-in-list $head $list2) True (vars-exist (cdr-atom $list1) $list2)))))
; !(---------)
; !(vars-exist ($tr) ( hi aha a $etr))
; !(---------)


(= (combiner-pat1-pat2 $vars () $accum) $accum)

(= (combiner-pat1-pat2 $vars $pat2 $accum)
  (superpose (
    ;; Option 1: keep the slot
    (combiner-pat1-pat2 $vars (cdr-atom $pat2) (union-atom $accum ((car-atom $pat2))))
    ;; Option 2: replace with each var
    (combiner-pat1-pat2 $vars (cdr-atom $pat2) (union-atom $accum ((superpose $vars))))

  ))
)

; Get filtered combinations of variables
(= (combine_lists $vars1 $vars2) 
    (let* (
        ($combinations (combiner-pat1-pat2 $vars1 $vars2 ()))
        ; (() (println!( "Combinations: " $combinations)))
        ($vars1-check (if (vars-exist $combinations $vars1) $combinations Empty))
        ; (() (println! ("Vars1 check: " $vars1-check)))
        ; ($vars2-check (if (vars-exist $combinations $vars2) $combinations Empty))
        ; (() (println! ("Vars2 check: " $vars2-check)))
    ) $combinations
    )
)


; ; !(experimenting with the second pattern that has 1 variable)
; !(combine_lists ($B) ($C))
; !(combine_lists ($A $B) ($C))

; ; !(experimenting with the second pattern that has 2 variables)
; !(combine_lists ($A $B) ($B $A))

; ; !(experimenting with the second pattern that has 3 variables)
; !(combine_lists ($A $B) ($A $C $D))

; ------------------------------------------------------------------------------------------------------;
; function name:- is_member
; parameters:- accumulator, atom
; return:- Boolean
; method:- compare the atom against each element of the accumulator using alpha-equality
( = (is_member $acc $x) 
    (let $curr (superpose $acc)
        (if (=alpha $curr $x) True (empty))
    )
)
; !(not (== () (collapse (is_member ((hi this is a test) (1 hello world) (foo bar)) (hello world)))))
; !(if (not (== () (collapse (is_member ((name1: $name1) (name2: $name2) (name3: $name3)) (name2: $name2))))) (found) ("not found"))
; ------------------------------------------------------------------------------------------------------;

; is-member-custom
(= (is-member-custom $x $acc) 
    (if (== () $acc) False
        (let ($head $tail) (decons-atom $acc)
            (if (=alpha $head $x) True (is-member-custom $x $tail))
        )

    )
)


(= (fromNumber $n) (if (<= $n 0) Z (S (fromNumber (- $n 1)))))

(= (count $atom $list)
   (foldl-atom $list 0 $acc $x (eval (if (== $atom $x) (+ $acc 1) $acc))))

( = (evalLinkHelper $links $minSup $acc $link) 
    (if (>= (count $link $links) $minSup) (union-atom $acc ($link)) $acc );check support and add if it meets minSup
)

; ------------------------------------------------------------------------------------------------------;
; function name:- evalLink
;; a function that will check whether a given link has a support or not
; parameters:- SpaceType, link, minSupport
; return:- Boolean
; method:- check if the support of the link in the given space is greater than or equal to the minimum support
(= (evalLink $links $minSup)
    (let $uniLinks (unique-atom $links)
        (foldl-atom $uniLinks () $acc $link 
            (evalLinkHelper $links $minSup $acc $link))
    )
)
; !(evalLink (Inheritance1 Inheritance2 Inheritance Inheritance Inheritance2 Inheritance1 Inheritance1 = = = = =) 2)
; ------------------------------------------------------------------------------------------------------;


; ------------------------------------------------------------------------------------------------------;
; function name:- sup-eval
; parameters:- SpaceType, pattern, minSupport
; return:- Boolean
; method:- match the pattern in the given space and ensure its support meets the minimum threshold
(= (sup-eval $db $pattern $minSup) 
    (let $matches (collapse (match $db $pattern $pattern)) 
        (>= (size-atom $matches) $minSup)
    )
)
; ------------------------------------------------------------------------------------------------------;

; ------------------------------------------------------------------------------------------------------;
; function name:- giveMeUniqueAcc
; parameters:- accumulator, atom
; return:- accumulator
; method:- append the atom only when it is not already present in the accumulator
( = (giveMeUniqueAcc $x $acc) 
    (if (is-member-custom $x $acc)
        $acc
        (union-atom $acc ($x))
    )
)
; ------------------------------------------------------------------------------------------------------;

; ------------------------------------------------------------------------------------------------------;
; function name:- only_unique
; parameters:- list of atoms
; return:- list of atoms
; method:- fold the list with giveMeUniqueAcc to remove duplicates
( = (only_unique $specials) 
    (foldl-atom $specials () $acc $x (giveMeUniqueAcc $x $acc))
)
; !(only_unique ((Inheritance $varNum0 woman) (Inheritance $varNum0 $varNum1) (Inheritance Lucy woman) (Inheritance Lucy $varNum1) (Inheritance $varNum0 woman) (Inheritance $varNum0 $varNum1) (Inheritance Emily woman) (Inheritance Emily $varNum1) (Inheritance $varNum0 human) (Inheritance $varNum0 $varNum1) (Inheritance Cason human) (Inheritance Cason $varNum1) (Inheritance $varNum0 man) (Inheritance $varNum0 $varNum1) (Inheritance Cason man) (Inheritance Cason $varNum1) (Inheritance $varNum0 ugly) (Inheritance $varNum0 $varNum1) (Inheritance Cason ugly) (Inheritance Cason $varNum1) (Inheritance $varNum0 sodaDrinker) (Inheritance $varNum0 $varNum1) (Inheritance Cason sodaDrinker) (Inheritance Cason $varNum1) (Inheritance $varNum0 ugly) (Inheritance $varNum0 $varNum1) (Inheritance Lily ugly) (Inheritance Lily $varNum1) (Inheritance $varNum0 sodaDrinker) (Inheritance $varNum0 $varNum1) (Inheritance Lily sodaDrinker) (Inheritance Lily $varNum1) (Inheritance $varNum0 human) (Inheritance $varNum0 $varNum1) (Inheritance Abe human) (Inheritance Abe $varNum1) (Inheritance $varNum0 man) (Inheritance $varNum0 $varNum1) (Inheritance Abe man) (Inheritance Abe $varNum1) (Inheritance $varNum0 ugly) (Inheritance $varNum0 $varNum1) (Inheritance Abe ugly) (Inheritance Abe $varNum1) (Inheritance $varNum0 sodaDrinker) (Inheritance $varNum0 $varNum1) (Inheritance Abe sodaDrinker) (Inheritance Abe $varNum1) (Inheritance $varNum0 human) (Inheritance $varNum0 $varNum1) (Inheritance Allen human) (Inheritance Allen $varNum1) (Inheritance $varNum0 man) (Inheritance $varNum0 $varNum1) (Inheritance Allen man) (Inheritance Allen $varNum1) (Inheritance $varNum0 ugly) (Inheritance $varNum0 $varNum1) (Inheritance Allen ugly) (Inheritance Allen $varNum1) (Inheritance $varNum0 sodaDrinker) (Inheritance $varNum0 $varNum1) (Inheritance Allen sodaDrinker) (Inheritance Allen $varNum1)))

; !(only_unique ((Inheritance Allen sodaDrinker) (Inheritance Allen $_67148) (Inheritance $_67166 sodaDrinker) (Inheritance $_67190 $_67196) (Inheritance Abe sodaDrinker) (Inheritance Abe $_67244) (Inheritance $_67262 sodaDrinker) (Inheritance $_67286 $_67292) (Inheritance Lily sodaDrinker) (Inheritance Lily $_67340) (Inheritance $_67358 sodaDrinker) (Inheritance $_67382 $_67388) (Inheritance Cason sodaDrinker) (Inheritance Cason $_67436) (Inheritance $_67454 sodaDrinker) (Inheritance $_67478 $_67484) (Inheritance Lily ugly) (Inheritance Lily $_67532) (Inheritance $_67550 ugly) (Inheritance $_67574 $_67580) (Inheritance Allen ugly) (Inheritance Allen $_67628) (Inheritance $_67646 ugly) (Inheritance $_67670 $_67676) (Inheritance Abe ugly) (Inheritance Abe $_67724) (Inheritance $_67742 ugly) (Inheritance $_67766 $_67772) (Inheritance Cason ugly) (Inheritance Cason $_67820) (Inheritance $_67838 ugly) (Inheritance $_67862 $_67868) (Inheritance Lily woman) (Inheritance Lily $_67916) (Inheritance $_67934 woman) (Inheritance $_67958 $_67964) (Inheritance Emily woman) (Inheritance Emily $_68012) (Inheritance $_68030 woman) (Inheritance $_68054 $_68060) (Inheritance Lucy woman) (Inheritance Lucy $_68108) (Inheritance $_68126 woman) (Inheritance $_68150 $_68156) (Inheritance Allen man) (Inheritance Allen $_68204) (Inheritance $_68222 man) (Inheritance $_68246 $_68252) (Inheritance Abe man) (Inheritance Abe $_68300) (Inheritance $_68318 man) (Inheritance $_68342 $_68348) (Inheritance Cason man) (Inheritance Cason $_68396) (Inheritance $_68414 man) (Inheritance $_68438 $_68444) (Inheritance Allen human) (Inheritance Allen $_68492) (Inheritance $_68510 human) (Inheritance $_68534 $_68540) (Inheritance Abe human) (Inheritance Abe $_68588) (Inheritance $_68606 human) (Inheritance $_68630 $_68636) (Inheritance Cason human) (Inheritance Cason $_68684) (Inheritance $_68702 human) (Inheritance $_68726 $_68732) (Inheritance Lily human) (Inheritance Lily $_68780) (Inheritance $_68798 human) (Inheritance $_68822 $_68828) (Inheritance Emily human) (Inheritance Emily $_68876) (Inheritance $_68894 human) (Inheritance $_68918 $_68924) (Inheritance Lucy human) (Inheritance Lucy $_68972) (Inheritance $_68990 human) (Inheritance $_69014 $_69020)))
; !(only_unique ((Inheritance $_67166 sodaDrinker) (Inheritance $_67166 sodaDrinker) (Inheritance $_67166 $_67168) ))
; ------------------------------------------------------------------------------------------------------;


; ------------------------------------------------------------------------------------------------------;
; function name:- myFlatten
; parameters:- expression
; return:- expression
; method:- perform a left fold that unions nested elements into a single flat list
(= (myFlatten $expr)
    (foldl-atom $expr () $acc $x (eval (union-atom $acc $x))))
; ------------------------------------------------------------------------------------------------------ ;

; ------------------------------------------------------------------------------------------------------;
;; make_var1
;; Description:
;;   Construct and execute a dynamic expression by formatting the two inputs and performing a double parse/eval cycle.
;;   The function formats $x and $y into the template "${}{}", converts that formatted value to a string representation,
;;   parses and evaluates it, then parses and evaluates the result again and returns the final value.
;; Parameters:
;;   $x  - First value or expression to be inserted into the format template. It will be evaluated as part of the procedure.
;;   $y  - Second value or expression to be inserted into the format template. It will be evaluated as part of the procedure.
;; Returns:
;;   variable :- $(first argument)(second argument)


( = (make_var1) 
    ; (function
    ;     (chain (eval (format-args "${}{}" ($x $y))) $formatted
    ;         (chain (eval (repr $formatted)) $strFormat
    ;             (chain (eval (parse $strFormat)) $onceParsed (return (eval (parse $onceParsed))) )))
    ; ) 
    $doesNotMatter
)
;!(make_var1)  
; ------------------------------------------------------------------------------------------------------;
; 'make-var'(Prefix, Arg, Var) :-
;      term_string(Arg, ArgStr),
;     string_concat("V", ArgStr, VarName),  % e.g. "V1"
;     atom_string(Atom, VarName),
;     atom_to_term(Atom, Var, _).
; ------------------------------------------------------------------------------------------------------;
( = (variableExtractor $input)
    (let $inputMetaType (get-metatype $input)
        (if (== $inputMetaType Expression)
            (let $current (superpose $input) (variableExtractor $current) )
            (if (== $inputMetaType Variable)
                $input
                (empty)            
            )
        )
    )
)
; !(variableExtractor (tru age is $never told (to someone $tru) (($ret $wr sdf (sdf $r)))))
; ------------------------------------------------------------------------------------------------------;

; ------------------------------------------------------------------------------------------------------;
;; variable extractor if the order of the variables matter
( = (getOrderlyVariablesHelper $curr $store)
  (let*
    (
      ($currMetaType (get-metatype $curr))

    )
    (if (== Expression $currMetaType)
      (let $currAnswer (getOrderlyVariables $curr) (union-atom $store $currAnswer))
      (if (== Variable $currMetaType) (union-atom $store ($curr)) $store)
    )

  )
) 

( = (getOrderlyVariables $pattern) 
  (foldl-atom $pattern () $store $curr (getOrderlyVariablesHelper $curr $store))
)
; !(getOrderlyVariables (hi $var1 (((var var ($asdfjkgs ($asfgkb $ask awrjk) $wert $wertr ser) gw ert) weg) wet) wetr $var2) )
; ------------------------------------------------------------------------------------------------------;
; $_6798
; $_6792
; $_6786
; $_6780
; $_6774
; ($_7840 $_7846 $_7852 $_7858 $_7864 $_7870 $_7876)
; ------------------------------------------------------------------------------------------------------;
; Check if a variable is in a list
(= (is-var-in-list $var $list) 
    (not (== () (collapse ( if (== (superpose $list) $var) 1 (empty) ))))
)
; !(is-var-in-list $tr ( hi aha a $etr))
; ------------------------------------------------------------------------------------------------------;

; ------------------------------------------------------------------------------------------------------;
( = (uniteVariables $vars1 $vars2)
    (foldl-atom $vars2 $vars1 $acc $x (if (is-var-in-list $x $acc) $acc (union-atom $acc ($x)) ))
)
; !(uniteVariables ($A $B $C) ($B $D $E))
; ------------------------------------------------------------------------------------------------------;

; ------------------------------------------------------------------------------------------------------;
( = (uniqueVars $variables) 
    (uniteVariables () $variables)
)
; !(uniqueVars ($a $b $c $a $d $e $b))
; ------------------------------------------------------------------------------------------------------;


; Check if a variable is in a list
(= (is-var-in-list-Superpose $var $list) (not (== () (collapse ( if (== (superpose $list) $var) 1 (empty) )))))

; Check if variables from one list exist in another list
(= (vars-exist-Superpose $list1 $list2) 
    (not (== () (collapse (let $equality (== (superpose $list1) (superpose $list2))
        (if $equality True (empty))
    ))))
)
; !(---------)
; !(vars-exist-Superpose ($tr) ( hi aha a $etr))
; !(---------)


(= (combiner-pat1-pat2 $vars () $accum) $accum)
(= (combiner-pat1-pat2 $vars $pat2 $accum)
    (let ($head $tail) (decons-atom $pat2)
        (superpose (
            (combiner-pat1-pat2 $vars $tail (union-atom $accum ($head)))  ;; Option 1: keep the slot
            (let $var (superpose $vars) (if (== $var $head) (empty) (combiner-pat1-pat2 $vars $tail (union-atom $accum ($var))))) ;; Option 2: replace with each var but avoid redundancy that might come
                    )
        )
    )
)

; Get filtered combinations of variables
(= (combine_list $vars1 $vars2)   
    (let* (  
        ; ($debug1 (println! (---- Combining vars1:- $vars1 with vars2:- $vars2 ----)))  
        ($combinations (combiner-pat1-pat2 $vars1 $vars2 ()))  
        ; ($debug2 (println! (from (vars1:- $vars1) (vars2:- $vars2) (combinations:- $combinations))))  
        ($boolGot (vars-exist $combinations $vars1))  
        ; ($debug3 (println! (do $combinations exist in vars1:- $boolGot)))  
        ($vars1-check (if $boolGot $combinations (empty)))  
        ; ($debug4 (println! ($combinations exist in vars1:- $vars1-check)))  
    ) $combinations  
    )  
)
; !(combine_list ($a $b $c) ($1 $2 $3))
; !(combine_list ($varNum0) ($varNum0))
