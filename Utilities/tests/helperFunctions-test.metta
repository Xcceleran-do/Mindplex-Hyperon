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

;; ---- find-atom ----
!(assertEqual (find-atom ((audience-expertise $x "intermediate") (STV 0.4 0.9)) STV) (STV 0.4 0.9))
!(assertEqual (find-atom ((audience-expertise $x "intermediate")) STV) ())

;; ---- pick-consequent ----
!(assertEqual (pick-consequent ((audience-expertise $x "intermediate") (engagement $x "Low"))) (engagement $x "Low"))
!(assertEqual (pick-consequent ((audience-expertise $x "intermediate") (style $x humor))) (style $x humor))

;; ---- remove-equal ----
!(assertEqual (remove-equal ((a 1) (b 2) (c 3)) (b 2)) ((a 1) (c 3)))

;; ---- clamp01 ----
!(assertEqual (clamp01 0.5) 0.5)
!(assertEqual (clamp01 (-0.3)) 0.0)
!(assertEqual (clamp01 1.7) 1.0)

;; ---- stv-values ----
!(assertEqual (stv-values ((a 1) (STV 0.4 0.9))) (0.4 0.9))
!(assertEqual (stv-values ((a 1))) (1.0 1.0))

;; ---- strip-stv ----
!(assertEqual (strip-stv ((a 1) (STV 0.4 0.9))) ((a 1)))
!(assertEqual (strip-stv ((a 1))) ((a 1)))

;; ---- build-rule ----
!(assertEqual
  (build-rule ((audience-expertise $x "intermediate") (engagement $x "Low")) (0.4 0.9) rule_1)
  (: rule_1 (Implication (Premises (audience-expertise $x "intermediate")) (Conclusions (engagement $x "Low"))) (STV 0.4 0.9)))

;; ---- pattern->rule ----
!(assertEqual
  (pattern->rule (, (audience-expertise $x "intermediate") (engagement $x "Low")) rule_1)
  (: rule_1 (Implication (Premises (audience-expertise $x "intermediate")) (Conclusions (engagement $x "Low"))) (STV 1.0 1.0)))

;; ---- patterns->rules on your real dataset ----
!(patterns->rules
  ((((audience-expertise $x1 "intermediate") (engagement $x1 "Low")) (STV 0.4117647058823529 0.9333333333333333))
   (((audience-expertise $x2 "intermediate") (engagement $x2 "High")) (STV 0.38235294117647056 0.9285714285714286))
   (((audience-expertise $x3 "mixed") (engagement $x3 "Medium")) (STV 0.6153846153846154 0.8888888888888888))
   (((audience-expertise $x4 "intermediate") (engagement $x4 "Medium")) (STV 0.17647058823529413 0.8571428571428571))))