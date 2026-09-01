!(import! &self ../../Utilities/helperFunctions)

;; Test cases for until function
!(test (until isPositive -- 100) 100)
!(test (until isPositive ++ -100) 1)
!(test (until isPositive -- 1) 1)
!(test (until isPositive ++ 0) 1)

;; Test cases for != function
!(test (!= 1 2) True)
!(test (!= 1 1) False)
!(test (!= a b) True)
!(test (!= c c) False)
!(test (!= True False) True)
!(test (!= False False) False)

;; Test cases for concaTuple function
!(test (concatTuple (1 2 3) (4 5 6)) (1 2 3 4 5 6))
!(test (concatTuple (a b) ()) (a b))
!(test (concatTuple () ()) ())
!(test (concatTuple () (x y)) (x y))

;; Test cases for isMember
!(test (isMember a (a b c)) True)
!(test (isMember d (a b c)) False)
!(test (isMember 1 (1 2 3)) True)
!(test (isMember 4 (1 2 3)) False)
!(test (isMember b (a b b c)) True)

;; Test cases for findAndReplace
!(test (findAndReplace a b (a b c)) (b b c))
!(test (findAndReplace 1 2 (1 2 3)) (2 2 3))
!(test (findAndReplace x y (a b c)) (a b c))
!(test (findAndReplace b c (a b b)) (a c c))

;; Test cases for foldr
!(test (foldr + 0 (1 2 3)) 6)
!(test (foldr + 1 ()) 1)
!(test (foldr * 1 (2 3 4)) 24)

;; Test cases for wrapper
!(test (wrapper isSymbol Q) (Q))
!(test (wrapper isPositive -2) ())
!(test (wrapper isEven 4) (4))

;; Test cases for filter
!(test (filter isPositive (1 -2 -3)) (1))
!(test (filter isEven (1 2 3)) (2))
!(test (filter isOdd (1 2 3 4)) (1 3))

;; Test cases for wrapperB
!(test (wrapperB == 1 1) (1))
!(test (wrapperB > 1 2) ())
!(test (wrapperB < 2 3) (2))

;; Test cases for filterB
!(test (filterB < 2 (1 2 3 4 5)) (1))
!(test (filterB !=  4 (1 2 3)) (1 2 3))
!(test (filterB == 2 (1 2 3 2)) (2 2))

;; Test cases for removeElement
!(test (removeElement (a b) (a b c d)) (c d))
!(test (removeElement (1 2) (1 2 3 4)) (3 4))
!(test (removeElement (x) (x y z)) (y z))

;; Test cases for any
!(test (any (True False False)) True)
!(test (any (False False False)) False)
!(test (any (False True False)) True)

;; Test cases for sum
!(test (sum (1 2 3 4)) 10)
!(test (sum ()) 0)
!(test (sum (5)) 5)

;; Test cases for count
!(test (count a (a b a c a)) 3)
!(test (count 1 (1 2 3 1 1)) 3)
!(test (count x (x x x)) 3)

;; Test cases for isSymbol
!(test (isSymbol a) True)
!(test (isSymbol (1 2 3)) False)
!(test (isSymbol True) False)

;; Test cases for isExpression
!(test (isExpression (1 2 3)) True)
!(test (isExpression a) False)
!(test (isExpression ()) True)

;; Test cases for isEmpty
!(test (isEmpty ()) True)
!(test (isEmpty (a)) False)
!(test (isEmpty (1 2)) False)

;; Test cases for isEven
!(test (isEven 2) True)
!(test (isEven 3) False)
!(test (isEven 0) True)

;; Test cases for isOdd
!(test (isOdd 2) False)
!(test (isOdd 3) True)
!(test (isOdd 0) False)

;; Test cases for hasOneAtom
!(test (hasOneAtom (a)) True)
!(test (hasOneAtom (a b)) False)
!(test (hasOneAtom ()) False)

;; Test cases for isPositive
!(test (isPositive 1) True)
!(test (isPositive -1) False)
!(test (isPositive 0) False)

;; Test cases for --
!(test (-- 3) 2)
!(test (-- 0) -1)
!(test (-- -2) -3)

;; Test cases for ++
!(test (++ 3) 4)
!(test (++ 0) 1)
!(test (++ -1) 0)

;; Test cases for appendAtom
!(test (appendAtom (a b) c) (a b c))
!(test (appendAtom () x) (x))

;; Test cases for gen
!(test (gen 3 ()) (0 1 2 3))
!(test (gen 0 ()) (0))
!(test (gen 10) (0 1 2 3 4 5 6 7 8 9 10))

;; Test cases for zip
!(test (zip (1 2) (a b) ()) ((1 a) (2 b)))
!(test (zip (x) (y) ()) ((x y)))
!(test (zip (x) (y)) ((x y)))

;; Test cases for zipWith
!(test (zipWith + (1 2) (3 4) ()) (4 6))
!(test (zipWith * (2 3) (4 5) ()) (8 15))
!(test (zipWith > (2 3) (4 5)) (False False))

;; Test cases for replaceByIndex
!(test (replaceByIndex (a b c) 1 x ()) (a x c))
!(test (replaceByIndex (1 2 3) 0 y ()) (y 2 3))
!(test (replaceByIndex (1 2 3) 10 y) (1 2 3))

;; Test cases for selectionSort
!(test (selectionSort (3 1 2 4 15 6 7 8) 2) (1 2 3 4 15 6 7 8))
!(test (selectionSort (5 3 4 1 2) 4) (1 2 3 4 5))

;; Test cases for repeat
!(test (repeat a 3 ()) (a a a))
!(test (repeat b 0 ()) ())
!(test (repeat (1 2 3) 2) ((1 2 3) (1 2 3)))

;; Test cases for take
!(test (take 2 (a b c) ()) (a b))
!(test (take 0 (a b c) ()) ())
!(test (take 10 (a b c) ) (a b c))

;; Test cases for drop
!(test (drop 2 (a b c)) (c))
!(test (drop 0 (a b c)) (a b c))

;; Test cases for all
!(test (all (True True True)) True)
!(test (all (True False True)) False)
!(test (all (False False False)) False)

;; Test cases for reverse
!(test (reverse (1 2 3 4)) (4 3 2 1))
!(test (reverse (r e w q)) (q w e r))
!(test (reverse (gen 3)) (3 2 1 0))

;; Test cases for range
!(test (range 0 4) (0 1 2 3 4))
!(test (range 3 10) (3 4 5 6 7 8 9 10))
!(test (range 2 3) (2 3))
!(test (range 10 1) ())

;; Test cases for range'
!(test (range' 10 4) (10 9 8 7 6 5 4))
!(test (range' 3 10) ())
!(test (range' 12 10) (12 11 10))


;; Test cases for randomAtom ;; TODO: this test is failing sometimes but not the other
; !(test (let ($x $i) (randomAtom (1 2 3)) (isMember $x (1 2 3))) True)
; !(test (let ($x $i) (randomAtom (1 2 3)) (isMember $x (4 5 6))) False)


;; Test cases for cartesian
!(test (cartesian (a b) (c d)) ((a c) (a d) (b c) (b d)))
!(test (cartesian (1 2) (3 4)) ((1 3) (1 4) (2 3) (2 4)))
!(test (cartesian (x y) (1 2 3)) ((x 1) (x 2) (x 3) (y 1) (y 2) (y 3)))

;; Test cases for multiCartesian
!(test (multiCartesian ((1 2) (a b) (x z))) ((1 a x) (1 a z) (1 b x) (1 b z) (2 a x) (2 a z) (2 b x) (2 b z)))
!(test (multiCartesian ((x) (1 2 3) (a b))) ((x 1 a) (x 1 b) (x 2 a) (x 2 b) (x 3 a) (x 3 b)))
!(test (multiCartesian ((1) (a b) (x z) (3 4))) ((1 a x 3) (1 a x 4) (1 a z 3) (1 a z 4) (1 b x 3) (1 b x 4) (1 b z 3) (1 b z 4)))

;; Test cases for clamp01
!(test (clamp01 0.5) 0.5)
!(test (clamp01 0.0) 0.0)
!(test (clamp01 1.0) 1.0)
!(test (clamp01 -0.3) 0.0)
!(test (clamp01 1.7) 1.0)
!(test (clamp01 -5.0) 0.0)
!(test (clamp01 100.0) 1.0)

;; ============================================================
;; Test cases for pick-consequent
;; ============================================================

;; Explicit non-engagement target
!(test
  (pick-consequent Outcome
    ((HasTopic $x agi) (HasFormat $x explainer) (Outcome $x completed)))
  (Outcome $x completed))

;; Another distinct target predicate, to further prove no hardcoding
!(test
  (pick-consequent WillComplete
    ((HasTopic $x news) (WillComplete $x true)))
  (WillComplete $x true))

;; No match for the given target -> empty
!(test
  (pick-consequent Outcome
    ((HasTopic $x agi) (HasFormat $x explainer)))
  empty)

;; Backward-compat 1-arg overload still defaults to engagement
!(test
  (pick-consequent
    ((audience-expertise $x "intermediate") (engagement $x "Low")))
  (engagement $x "Low"))


;; ============================================================
;; Test cases for build-rule
;; ============================================================

;; Non-engagement target: Outcome
!(test
  (build-rule Outcome
    (((HasTopic $x agi) (HasFormat $x explainer) (Outcome $x completed))
     (CTV (STV 0.75 0.857) (STV 0.45 0.9375)))
    rule_1)
  (: rule_1
     (Implication (Premises (HasTopic $x agi) (HasFormat $x explainer))
                   (Conclusions (Outcome $x completed)))
     (CTV (STV 0.75 0.857) (STV 0.45 0.9375))))

;; A THIRD distinct target predicate: WillComplete
!(test
  (build-rule WillComplete
    (((style $x humor) (word-length $x short) (WillComplete $x true))
     (CTV (STV 1.0 1.0) (STV 0.5 0.7)))
    rule_2)
  (: rule_2
     (Implication (Premises (style $x humor) (word-length $x short))
                   (Conclusions (WillComplete $x true)))
     (CTV (STV 1.0 1.0) (STV 0.5 0.7))))

;; Clamping still works under a non-default target
!(test
  (build-rule Outcome
    (((audience-expertise $x "intermediate") (Outcome $x "Low"))
     (CTV (STV 1.5 -0.2) (STV -0.3 1.4)))
    rule_3)
  (: rule_3
     (Implication (Premises (audience-expertise $x "intermediate"))
                   (Conclusions (Outcome $x "Low")))
     (CTV (STV 1.0 0.0) (STV 0.0 1.0))))

;; Old single-STV shape no longer matches, regardless of target
!(test
  (build-rule Outcome
    (((audience-expertise $x "intermediate") (Outcome $x "Low"))
     (STV 0.4 0.9))
    rule_4)
  ())

;; Backward-compat 2-arg overload defaults to engagement specifically
;; -- this is the ONE test that should still use engagement, since
;; its whole point is confirming the default.
!(test
  (build-rule
    (((audience-expertise $x "intermediate") (engagement $x "Low"))
     (CTV (STV 0.4 0.9) (STV 0.2 0.85)))
    rule_5)
  (: rule_5
     (Implication (Premises (audience-expertise $x "intermediate"))
                   (Conclusions (engagement $x "Low")))
     (CTV (STV 0.4 0.9) (STV 0.2 0.85))))


;; ============================================================
;; Test cases for patterns->rules
;; ============================================================

;; Explicit non-engagement target, multiple patterns
!(test
  (patterns->rules Outcome
    ( (((HasTopic $x agi) (HasFormat $x explainer) (Outcome $x completed))
       (CTV (STV 0.75 0.857) (STV 0.45 0.9375)))
      (((HasTopic $x crypto) (Outcome $x completed))
       (CTV (STV 0.5 0.6) (STV 0.4 0.7))) )
    1)
  ( (: rule_1
       (Implication (Premises (HasTopic $x agi) (HasFormat $x explainer))
                     (Conclusions (Outcome $x completed)))
       (CTV (STV 0.75 0.857) (STV 0.45 0.9375)))
    (: rule_2
       (Implication (Premises (HasTopic $x crypto))
                     (Conclusions (Outcome $x completed)))
       (CTV (STV 0.5 0.6) (STV 0.4 0.7))) ))

;; Mixed batch: one pattern matches the target, one doesn't produce
;; a rule because it's still old-style STV -- proves skip-on-failure
;; still works correctly under a non-default target too.
!(test
  (patterns->rules Outcome
    ( (((HasTopic $x agi) (Outcome $x completed))
       (CTV (STV 0.75 0.857) (STV 0.45 0.9375)))
      (((HasTopic $x news) (Outcome $x completed))
       (STV 0.5 0.6)) )
    1)
  ( (: rule_1
       (Implication (Premises (HasTopic $x agi))
                     (Conclusions (Outcome $x completed)))
       (CTV (STV 0.75 0.857) (STV 0.45 0.9375))) ))

;; Backward-compat 1-arg overload (defaults to engagement) -- the
;; ONE test in this block that should use engagement, since that's
;; exactly what it's checking.
!(test
  (patterns->rules
    ( (((audience-expertise $x "intermediate") (engagement $x "Low"))
       (CTV (STV 0.4 0.9) (STV 0.2 0.85))) ))
  ( (: rule_1
       (Implication (Premises (audience-expertise $x "intermediate"))
                     (Conclusions (engagement $x "Low")))
       (CTV (STV 0.4 0.9) (STV 0.2 0.85))) ))

;; Empty pattern list, non-default target
!(test (patterns->rules Outcome () 1) ())