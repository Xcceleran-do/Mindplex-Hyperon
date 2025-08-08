!(register-module! ../../../Mindplex-Hyperon)
!(import! &self Mindplex-Hyperon:Utilities:helperFunctions)

;; Test cases for until function
!(assertEqual (until isPositive -- 100) 100)
!(assertEqual (until isPositive ++ -1000) 1)
!(assertEqual (until isPositive -- 1) 1)
!(assertEqual (until isPositive ++ 0) 1)

;; Test cases for != function
!(assertEqual (!= 1 2) True)
!(assertEqual (!= 1 1) False)
!(assertEqual (!= a b) True)
!(assertEqual (!= c c) False)
!(assertEqual (!= True False) True)
!(assertEqual (!= False False) False)

;; Test cases for concaTuple function
!(assertEqual (concatTuple (1 2 3) (4 5 6)) (1 2 3 4 5 6))
!(assertEqual (concatTuple (a b) ()) (a b))
!(assertEqual (concatTuple () ()) ())
!(assertEqual (concatTuple () (x y)) (x y))

;; Test cases for isMember
!(assertEqual (isMember a (a b c)) True)
!(assertEqual (isMember d (a b c)) False)
!(assertEqual (isMember 1 (1 2 3)) True)
!(assertEqual (isMember 4 (1 2 3)) False)
!(assertEqual (isMember b (a b b c)) True)

;; Test cases for findAndReplace
!(assertEqual (findAndReplace a b (a b c)) (b b c))
!(assertEqual (findAndReplace 1 2 (1 2 3)) (2 2 3))
!(assertEqual (findAndReplace x y (a b c)) (a b c))
!(assertEqual (findAndReplace b c (a b b)) (a c c))

;; Test cases for foldr
!(assertEqual (foldr + 0 (1 2 3)) 6)
!(assertEqual (foldr + 1 ()) 1)
!(assertEqual (foldr * 1 (2 3 4)) 24)

;; Test cases for wrapper
!(assertEqual (wrapper isSymbol Q) (Q))
!(assertEqual (wrapper isPositive -2) ())
!(assertEqual (wrapper isEven 4) (4))

;; Test cases for filter
!(assertEqual (filter isPositive (1 -2 -3)) (1))
!(assertEqual (filter isEven (1 2 3)) (2))
!(assertEqual (filter isOdd (1 2 3 4)) (1 3))

;; Test cases for wrapperB
!(assertEqual (wrapperB == 1 1) (1))
!(assertEqual (wrapperB > 1 2) ())
!(assertEqual (wrapperB < 2 3) (3))

;; Test cases for filterB
!(assertEqual (filterB < 2 (1 2 3 4 5)) (3 4 5))
!(assertEqual (filterB !=  4 (1 2 3)) ())
!(assertEqual (filterB == 2 (1 2 3 2)) (2 2))

;; Test cases for removeElement
!(assertEqual (removeElement (a b) (a b c d)) (c d))
!(assertEqual (removeElement (1 2) (1 2 3 4)) (3 4))
!(assertEqual (removeElement (x) (x y z)) (y z))

;; Test cases for any
!(assertEqual (any (True False False)) True)
!(assertEqual (any (False False False)) False)
!(assertEqual (any (False True False)) True)

;; Test cases for sum
!(assertEqual (sum (1 2 3 4)) 10)
!(assertEqual (sum ()) 0)
!(assertEqual (sum (5)) 5)

;; Test cases for count
!(assertEqual (count a (a b a c a)) 3)
!(assertEqual (count 1 (1 2 3 1 1)) 3)
!(assertEqual (count x (x x x)) 3)

;; Test cases for isSymbol
!(assertEqual (isSymbol a) True)
!(assertEqual (isSymbol (1 2 3)) False)
!(assertEqual (isSymbol True) False)

;; Test cases for isExpression
!(assertEqual (isExpression (1 2 3)) True)
!(assertEqual (isExpression a) False)
!(assertEqual (isExpression ()) True)

;; Test cases for isEmpty
!(assertEqual (isEmpty ()) True)
!(assertEqual (isEmpty (a)) False)
!(assertEqual (isEmpty (1 2)) False)

;; Test cases for isEven
!(assertEqual (isEven 2) True)
!(assertEqual (isEven 3) False)
!(assertEqual (isEven 0) True)

;; Test cases for isOdd
!(assertEqual (isOdd 2) False)
!(assertEqual (isOdd 3) True)
!(assertEqual (isOdd 0) False)

;; Test cases for hasOneAtom
!(assertEqual (hasOneAtom (a)) True)
!(assertEqual (hasOneAtom (a b)) False)
!(assertEqual (hasOneAtom ()) False)

;; Test cases for isPositive
!(assertEqual (isPositive 1) True)
!(assertEqual (isPositive -1) False)
!(assertEqual (isPositive 0) False)

;; Test cases for --
!(assertEqual (-- 3) 2)
!(assertEqual (-- 0) -1)
!(assertEqual (-- -2) -3)

;; Test cases for ++
!(assertEqual (++ 3) 4)
!(assertEqual (++ 0) 1)
!(assertEqual (++ -1) 0)

;; Test cases for appendAtom
!(assertEqual (appendAtom (a b) c) (a b c))
!(assertEqual (appendAtom () x) (x))

;; Test cases for gen
!(assertEqual (gen 3 ()) (0 1 2 3))
!(assertEqual (gen 0 ()) (0))
!(assertEqual (gen 10) (0 1 2 3 4 5 6 7 8 9 10))

;; Test cases for zip
!(assertEqual (zip (1 2) (a b) ()) ((1 a) (2 b)))
!(assertEqual (zip (x) (y) ()) ((x y)))
!(assertEqual (zip (x) (y)) ((x y)))

;; Test cases for zipWith
!(assertEqual (zipWith + (1 2) (3 4) ()) (4 6))
!(assertEqual (zipWith * (2 3) (4 5) ()) (8 15))
!(assertEqual (zipWith > (2 3) (4 5)) (False False))

;; Test cases for len
!(assertEqual (len (1 2 3 4)) 4)
!(assertEqual (len ()) 0)
!(assertEqual (len (a)) 1)

;; Test cases for replaceByIndex
!(assertEqual (replaceByIndex (a b c) 1 x ()) (a x c))
!(assertEqual (replaceByIndex (1 2 3) 0 y ()) (y 2 3))
!(assertEqual (replaceByIndex (1 2 3) 10 y) (1 2 3))

;; Test cases for clearSpace
; Skipped: requires grounded space

;; Test cases for selectionSort
!(assertEqual (selectionSort (3 1 2 4 15 6 7 8) 2) (1 2 3 4 15 6 7 8))
!(assertEqual (selectionSort (5 3 4 1 2) 4) (1 2 3 4 5))

;; Test cases for repeat
!(assertEqual (repeat a 3 ()) (a a a))
!(assertEqual (repeat b 0 ()) ())
!(assertEqual (repeat (1 2 3) 2) ((1 2 3) (1 2 3)))

;; Test cases for take
!(assertEqual (take 2 (a b c) ()) (a b))
!(assertEqual (take 0 (a b c) ()) ())
!(assertEqual (take 10 (a b c) ) (a b c))

;; Test cases for drop
!(assertEqual (drop 2 (a b c)) (c))
!(assertEqual (drop 0 (a b c)) (a b c))

;; Test cases for all
!(assertEqual (all (True True True)) True)
!(assertEqual (all (True False True)) False)
!(assertEqual (all (False False False)) False)

;; Test cases for isInf
!(assertEqual (isInf 1e1000) True)
!(assertEqual (isInf 1) False)
!(assertEqual (isInf -1e1000) True)

;; Test cases for reverse
!(assertEqual (reverse (1 2 3 4)) (4 3 2 1))
!(assertEqual (reverse (r e w q)) (q w e r))
!(assertEqual (reverse (gen 3)) (3 2 1 0))

;; Test cases for range
!(assertEqual (range 0 4) (0 1 2 3 4))
!(assertEqual (range 3 10) (3 4 5 6 7 8 9 10))
!(assertEqual (range 2 3) (2 3))
!(assertEqual (range 10 1) ())

;; Test cases for range'
!(assertEqual (range' 10 4) (10 9 8 7 6 5 4))
!(assertEqual (range' 3 10) ())
!(assertEqual (range' 12 10) (12 11 10))

;; Test cases for bySecondAtom
!(assertEqual (bySecondAtom (a 1) (b 2)) False)
!(assertEqual (bySecondAtom (a 10) (b 2)) True)

;; Test cases for randomAtom
!(assertEqual (let ($x $i) (randomAtom (1 2 3)) (isMember $x (1 2 3))) True)
!(assertEqual (let ($x $i) (randomAtom (1 2 3)) (isMember $x (4 5 6))) False)

;; Test cases for flatten
!(assertEqual (flatten ((1 2 3) (a b c))) (1 2 3 a b c))
!(assertEqual (flatten ((1 2) ($x $y c) ((1) (2)))) (1 2 $x $y c (1) (2)))