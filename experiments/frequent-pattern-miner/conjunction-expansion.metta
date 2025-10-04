! (import! &self freq-pat)
! (import! &self conj-exp)

---------------------------------
;; get_variable_for_pattern
;; ---------------------------------
;; Extracts variables from a pattern tree while preserving the original
;; structure (tuple of variables). If a subterm is not a Variable, it is
;; ignored. Handles patterns of the forms (), ($x), ($link $x), and
;; ($link $x $y), recursing into nested Expressions.
;; Parameters:
;;  - $pattern: Pattern tree to analyze.
;; Returns:
;;  - Expression: Tuple of variables found in the pattern (may be () if none).
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


---------------------------------
;; substitute
;; ---------------------------------
;; Performs positional substitution of variables in a pattern using a tuple of
;; replacement atoms (variables list). Walks the pattern structure and returns
;; a pair: (substituted_pattern remaining_variables). Non-variable atoms are
;; preserved as-is. Supports patterns of arity 0, 1, or 2 (link x [y]).
;; Parameters:
;;  - $pattern: Pattern tree with variables to substitute.
;;  - $variables: Tuple of replacement atoms.
;; Returns:
;;  - Pair: (substituted_pattern, remaining_variables).
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

---------------------------------
;; checker
;; ---------------------------------
;; Filters conjunction candidates by support. If the conjunction has support
;; >= $ms in `$db`, append it to the accumulator; otherwise leave the
;; accumulator unchanged.
;; Parameters:
;;  - $db: Database space to evaluate support against.
;;  - (conjunct $truePattern): Conjunction candidate wrapped as conjunct.
;;  - $ms: Minimum support threshold.
;;  - $acc: Accumulator list of accepted conjunctions.
;; Returns:
;;  - Expression: Updated accumulator.
(= (checker $db (conjunct $truePattern) $ms $acc) 
    (
        if (sup-eval $db $truePattern $ms) 
            (cons-atom $truePattern $acc) 
            $acc
    )
)

 
------------------------------------------------
;; do-conjunct
------------------------------------------------
;; Expands patterns into conjunction candidates of a specified size and retains
;; those that satisfy the minimum support.

;; parameters:
;; $db: the database that is loaded and cleaned of the stdlib functions
;; $patterns: the candidate patterns space
;; $ms: minimum support
;; $depth: combination size (number of clauses per conjunction)

;steps:
;; 1: build unique combinations of size `$depth` from `$patterns`
;; 2: filter by support using `checker`

; returns:
; a superpose of conjunctions that satisfy the minimum support and are not redundant.
;(: do-conjunct SpaceType SpaceType Pattern Number Nat Atom)
(= (do-conjunct $db $patterns $ms $depth)
    ( trace! ("==== Build conjunction with depth ====" $depth) (unique_combinations_star $patterns $depth))
)


-----------------------------------------------
;; conjunct 
-----------------------------------------------
;; Extracts candidate conjunctions of a given size from `$candidatePatterns`
;; and adds them to `$conjspace`.
;; 1: call `do-conjunct` to generate supported conjunctions
;; 2: add the resulting atoms into `$conjspace`
;(: conjunct (-> SpaceType SpaceType SpaceType Number Number Atom))
(= (conjunct $candidatePatterns $depth ) (unique_combinations_star $patterns $depth)) 
