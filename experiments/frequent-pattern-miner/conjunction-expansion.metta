; /**
 ;  * Given cnjtion and pattern, consider all possible connections
 ;  * (a.k.a linkages) and expand cnjtion accordingly. For instance if
 ;  *
 ;  * cnjtion = (Inheritance X Y)
 ;  * pattern = (Inheritance Z W)
 ;  *
 ;  * return
 ;  *
 ;  *   (, (Inheritance X Y) (Inheritance X W))
 ;  *   (, (Inheritance X Y) (Inheritance Z X))
 ;  *   (, (Inheritance X Y) (Inheritance Y W))
 ;  *   (, (Inheritance X Y) (Inheritance Z Y))
 ;  *
 ;  * It will also only include patterns with minimum support ms
 ;  * according to db, and perform alpha-conversion when necessary.
 ;  *
 ;  * mv is the maximum number of variables allowed in the resulting
 ;  *    patterns.
 ;  *
 ;  */

; ! (register-module! ../../experiments)
; ! (import! &self experiments:utils:common-utils)
! (import! &self conj-exp)
! (import! &self freq-pat)

 ; The starting point for the conjunction expansion. It passes
 ; the Alpha Converted pattern (if necessary) to the expand_conjunction functions.
 ; if it passes the minimum variable check

(= (expand_conjunction $cnjtion $pattern $db $ms $mv $es)
    (if $es
        (expand_conjunctin_es_rec $cnjtion $pattern $db $ms)
        (expand_conjunction_rec $cnjtion $pattern $db $ms)
    )
)

(= (count-atom-element $atom) (if (== $atom ()) 0 (+ 1 (count-atom-element (cdr-atom $atom)))))

(= (get_variables_for_tree $pattern)
    (if (== $pattern ())
        ()
        (let* (
                ($head (car-atom $pattern))
                ($tail (cdr-atom $pattern)))

            (if (== (get-metatype $head) Expression)
                (concatTuple (get_variables_for_tree $head)
                    (get_variables_for_tree $tail))
                (if (== (get-metatype $head) Variable)
                    (concatTuple
                        ($head)
                        (get_variables_for_tree $tail))

                    (get_variables_for_tree $tail))))
    )
)



(= (get_variable_for_pattern $pattern)
    (case ($pattern) (
            ( (()) ())
            (  (($x))
                (if (== (get-metatype $x) Variable)
                    ($x)
                () ;; Not actually a variable
            ))
    ( (  ($link $x $y))
        (let* (
                ( $subx (if ( == ( get-metatype $x ) Expression) (get_variable_for_pattern $x) (get_variable_for_pattern ($x))))
                ( $suby (if (== (get-metatype $y) Expression) (get_variable_for_pattern $y) (get_variable_for_pattern ($y))))
            )
        (concatTuple $subx $suby)
    )
)
( ( ($link $x))
    (let* (
            ( $subx (if ( == ( get-metatype $x ) Expression) (get_variable_for_pattern $x) (get_variable_for_pattern ($x))))
        )
    $subx
)
)
)
)
)


(= (substitute $pattern $variables)
    (case ($pattern $variables) (
            ( ( () $variables) ())
             ; ( ( ($pattern) () ) $pattern)
            ( ( ($x) $variables)
                (if (== (get-metatype $x) Variable)
                    (if (== $variables ())
                        ($x $variables)
                        ( (car-atom $variables) (cdr-atom $variables)) ;; Substitute
                    )
                ($x $variables) ;; Not actually a variable
            ))
    ( (  ($link $x $y) $variables)
        (let* (
                ( ($sublink $remains) (if ( == ( get-metatype $link ) Expression) (substitute $link $variables) (substitute ($link) $variables)))
                ( ($subx $remaining) (if ( == ( get-metatype $x ) Expression) (substitute $x $remains) (substitute ($x) $remains)))

                ( ($suby $final) (if (== (get-metatype $y) Expression) (substitute $y $remaining) (substitute ($y) $remaining)))
            )
        ( ($sublink $subx $suby) $final)
    )
)
( ( ($link $x) $variables)
    (let* (
            ( ($sublink $remains) (if ( == ( get-metatype $link ) Expression) (substitute $link $variables) (substitute ($link) $variables)))
            ( ($subx $remaining) (if ( == ( get-metatype $x ) Expression) (substitute $x $remains) (substitute ($x) $remains)))
        )
    ( ($sublink $subx) $remaining)
)))))

 
 ==============  alpha convert  =======================

(= (alpha_convert  $pattern_vars $cnjtion_vars)
    (extract_values (consolidate_keys (address_conflict $pattern_vars $cnjtion_vars)))
)

(= (address_conflict $pattern_vars $cnjtion_vars)
    (if (== $pattern_vars ())
        ()
        (let* (
                ($head (car-atom $pattern_vars))
                ($tail (cdr-atom $pattern_vars)))
            (if (does_exist $head $cnjtion_vars)
                (concatTuple ( ($head (generateRandomVar))) (address_conflict $tail $cnjtion_vars ))
                (concatTuple ( ($head $head)) (address_conflict $tail $cnjtion_vars))))
    )
)

(= (does_exist $var $list_vars)
    (if (== $list_vars ())
        False
        (if (== $var (car-atom $list_vars))
            True
            (does_exist $var (cdr-atom $list_vars)))
    )
)


(= (consolidate_keys $input_list)
    (let* (
            ($result ())
            ($seen_keys ()))
        (consolidate_helper $input_list $result $seen_keys)
    )
)

(= (consolidate_helper $input_list $result $seen_keys)
    (if (== $input_list ())
        $result
        (let* (
                ($pair (car-atom $input_list))
                ($key (car-atom $pair))
                ($value (cdr-atom $pair)))
            (if (does_exist $key $seen_keys)
                (consolidate_helper (cdr-atom $input_list)
                    (concatTuple ( ($key (let $temp (get_value $key $result) (car-atom $temp)))) $result)
                    $seen_keys)
                (let* (
                        ($new_result (let $temp (cons-atom $key $value) (cons-atom $temp $result)))
                        ($new_seen_keys (cons-atom $key $seen_keys)))
                    (consolidate_helper (cdr-atom $input_list) $new_result $new_seen_keys))))
    )
)

(= (get_value $key $list)
    (if (== $list ())
        ()
        (if (== $key (let $temp (car-atom $list) (car-atom $temp)))
            (let $temp2 (car-atom $list) (cdr-atom $temp2))
            (get_value $key (cdr-atom $list)))
    )
)

(= (extract_values $input_list)
    (extract_helper $input_list ())
)

(= (extract_helper $input_list $values)
    (if (== $input_list ())
        $values
        (let* (
                ($pair (car-atom $input_list))
                ($value (cdr-atom $pair))
                ($tail (cdr-atom $input_list)))
            (extract_helper $tail (cons-atom (car-atom $value) $values)))
    )
)

 ================= alpha convert end ================================


(= (expand_conjunction_rec $cnjtion $apat $db $ms)
    (let $npat (expand_conjunction_connect $cnjtion $apat (get_variable_for_pattern $cnjtion) (get_variable_for_pattern $apat))
         (checker $db $npat $ms)
        ; $npat
    )
)

(= (expand_conjunction_connect $cnjtion $pattern $cv $pv)
    (let $list (replace_pattern $cnjtion $pattern (combine_lists $cv $pv))
        ; (remove_alpha_similar $list)
        $list
    )
)
(= (replace_pattern $cnjtion $pattern ())
    ()
)
(= (replace_pattern $cnjtion $pattern $pv2cv)
    (let* (
            ($sup (superpose $pv2cv))
            ($exp (substitute $pattern $sup))
            ($new_cnjtion (, $cnjtion (car-atom $exp)))
        )
        $new_cnjtion
)
)

(= (checker $db $pattern $ms) 
   (let* (
   ( () (println! (pattern $pattern)))
   ($flat (flatten $pattern))
   ) 
   (if (sup-eval $db $flat $ms)
        $pattern
        ()
    ))
)


(= (remove_alpha_similar $list)
    (remove_alpha_similar_helper $list ())
)

(= (remove_alpha_similar_helper ($listHead $listTail) $seen_list)
    (if (== $listTail ())
        ()
        (let* (
                ($matched (collapse (match &self $listHead $listHead)))
                ($seen_list_new (concatTuple ($matched) $seen_list)))
            (if (or (== $matched ()) (does_exist $matched $seen_list)) ;(or (== $matched ()) )but supp evaluation will check this
                (remove_alpha_similar_helper $listTail $seen_list)
                ( $listHead (remove_alpha_similar_helper $listTail $seen_list_new))))
    )
)

(= (expand_conjunction_es_rec $cnjtion $apat $db $ms)
    (pass)
)

(= (remove_conjuncts_with_redundant_clauses $conjunction) 
    (let* (
        ($conjuncts (get-conjuncts $conjunction))
        ($unique_conjuncts (unique-atom $conjuncts))
        )
    (if (== (size-atom $conjuncts) (size-atom $unique_conjuncts))
        $conjunction
        ()
    )
    )
)


(= ( expand_conjunction_disconnect $cnjtion $pattern ) (
        let* (
            ($cnjtion_vars (get_variables_for_tree $cnjtion))
            ($pat_vars  (get_variables_for_tree $pattern))
            ($acvar (alpha_convert $pat_vars $cnjtion_vars))
            ($acpat (let $pat (substitute $pattern $acvar) (car-atom $pat)))
            ($npat (append_pat $cnjtion $acpat))
            ($nclause (remove_redundant_subclauses $npat))
        ) $acvar
))

(= ( append_pat $pat $conj) ( , $pat $conj))

 
------------------------------------------------
;; do-conjunct
------------------------------------------------
;; function to expand the conjunctions of the patterns and create conjunctions of any amount of clauses

;; parameters:
;; $db: the database that is loaded and cleaned of the stdlib functions.
;; $cndb: the candidate patterns space
;; $conjunct: the conjunctions part of the pattern
;; $ms: minimum support
;; (S $K)/ Z: a debruijin index to check if we are in the last iteration of the recursion.

;steps:
;; 1: get the matches of the candidate patterns in the database
;; 2: expand the conjunctions with the matches and the database
;; 3: replace the variables with the debruijin index
;; 4: remove any redundant clauses from the conjunctions
;; 5: if we are in the last iteration of the recursion, return the conjunctions else call the function again with the new conjunctions and the debruijin index.

; returns:
; a superpose of conjunctions that satisfy the minimum support and are not redundant.
;(: do-conjunct SpaceType SpaceType Pattern Number Nat Atom)
(= (do-conjunct $db $cndb $coming_conj  $ms Z)
    (let* (
        (() (println! "==== Build conjunction with depth 0 ===="))
        ($conjunct (replacev $coming_conj))
        ($matches (match $cndb (CandidatePattern $pattern) (replacev $pattern)))
        ($listconj (expand_conjunction $conjunct $matches $db $ms 2 False))
        ($debconj (replace $listconj))
        ($fitlered (remove_conjuncts_with_redundant_clauses $debconj))
    )
    $fitlered
    )
)

(= (do-conjunct $db $cndb $coming_conj $ms (S $K))
    (let* (
        (() (println! "==== Build conjunction with depth greater than 0 ====="))
        ($conjunct (replacev $coming_conj))
        ($matches (match $cndb (CandidatePattern $pattern) (replacev $pattern)))
        ($listconj (expand_conjunction $conjunct $matches $db $ms 2 False))
        ($debconj (replace $listconj))
        ($fitlered (remove_conjuncts_with_redundant_clauses $debconj))
        ($varconj (replacev $fitlered))
        ($conj (do-conjunct $db $cndb $varconj $ms $K))
    )
    $conj
    )
)

(= (add-atom-local $space $atom)
   (case (match $space $atom $atom)
     (
      ($atom ())
      (Empty (case $atom (
               ((supportOf $x 1) () )
               ($_ (add-atom $space $atom))))))))


(= (format-helper $var-index $dbspace $depth) (
     let $var (replacev $var-index) (supportOf $var (sup-num $dbspace $var))))

--------------------------------------------
;;  formatter
--------------------------------------------
;; function to format the patterns with thier support annotation.
;; 1: get the matches of the conjunctions in the database
;; 2: sort the conjunctions with a custom sorting function and get only the unique conjunctions from the sorte results
;; 3: create a flattened conjunction  using cons-atom with the sorted conjunctions and ,.
;; 4: add the results to the knowledge base with their support
;(: formatter (-> SpaceType SpaceType Number Atom ))
(= (formatter  $conjspace $dbspace $depth) 
   (let* 
        (
            (() (println! "==== Formating  started ====="))
            ($sorted  (match $conjspace $pattern (sort_conj $pattern)))
            ($res  (cons-atom ,  $sorted))
            ($formatted (format-helper $res $dbspace (fromNumber $depth)))
        ) 
        $formatted
    )
)

-----------------------------------------------
;; conjunct 
-----------------------------------------------
;; function to extract candidate patterns and call do-conjunct 
;(: conjunct (-> SpaceType SpaceType SpaceType Number Number Atom))
(= (conjunct $dbspace $cndpspace $conjspace $minsup $depth )   
      (let $pattern (match $cndpspace (CandidatePattern $Cpattern) $Cpattern)
                   (let $conj  (do-conjunct $dbspace $cndpspace  $pattern $minsup (fromNumber $depth)) 
                             (if (== $conj ()) () (add-atom $conjspace  $conj)))))   

   