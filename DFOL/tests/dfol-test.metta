
!(import! &self ../../Utilities/helperFunctions)
!(import! &self ../../DFOL/dfol)

;; Test cases for addToVarDom function
!(test 
    (addToVarDom 1 ((mother alice bob) (father bob carol) (mother carol dave) (father alice carol) (mother a d) (mother d c) (father carol d))) 
    ((alice 1) (bob 1) (carol 1) (dave 1) (a 1) (d 1) (c 1))) ; The order is non-determined

!(test 
    (addToVarDom 2 ((mother alice bob) (father bob carol) (mother carol dave) (father alice carol) (mother a d) (mother d c) (father carol d))) 
    ((alice 2) (bob 2) (carol 2) (dave 2) (a 2) (d 2) (c 2))) ; The order is non-determined


;; Test cases for getId function (uses global dict)
!(test (getId X) 1)
!(test (getId Y) 2)
!(test (getId A) 3)
!(test (getId B) 4)
!(test (getId Z) 7)
!(test (getId unknown) (empty))

;; Test cases for getSub function
!(test (getSub 1 ((alice 1) (bob 2) (carol 3))) alice)
!(test (getSub 2 ((alice 1) (bob 2) (carol 3))) bob)
!(test (getSub 3 ((alice 1) (bob 2) (carol 3))) carol)

;; Test cases for ground function with binary predicates 
!(test 
    (ground (mother X Y) ((alice 1) (bob 2)))
    (mother alice bob))

!(test 
    (ground (father X A) ((bob 1) (carol 3)))
    (father bob carol))

;; Test cases for ground function with unary predicates 
!(test 
    (ground (loves X) ((alice 1)))
    (loves alice))

!(test 
    (ground (smart A) ((carol 3)))
    (smart carol))

;; Test cases for basicEmbedding function with unary predicates 
!(test 
    (basicEmbedding (grandparent X) (mother Y))
    0)

!(test 
    (basicEmbedding (loves X) (hate X))
    1)

;; Test cases for basicEmbedding function with binary predicates
!(test 
    (basicEmbedding (grandparent X Y) (mother X Z))
    (1 0))

!(test 
    (basicEmbedding (grandparent X Y) (mother Z Y))
    (0 1))

!(test 
    (basicEmbedding (grandparent X Y) (father X Y))
    (1 1))

!(test 
    (basicEmbedding (loves A B) (hates C D))
    (0 0))

;; Test cases for occurrenceEmbedding function
!(test 
    (occurrenceEmbedding (Z) (mother X Z))
    (1))

!(test 
    (occurrenceEmbedding (A B) (father A B))
    (1 1))

!(test 
    (occurrenceEmbedding (Z) (loves X Y))
    (0))

!(test 
    (occurrenceEmbedding (A B C) (mother A B))
    (1 1 0))

;; Test cases for substitution function
!(bind! &test_vi (0.0 0.0 1.0 0.0))
!(bind! &test_vo (0.0))
!(bind! &test_idx 0)
!(bind! &test_cursub ((alice 1) (bob 2)))
!(bind! &test_cursub_match ((alice 1) (carol 2)))

;; Test substitution with positive example match
!(test 
    (substitution (mother X Y) ((mother alice bob) (father bob carol) (mother carol dave) (father alice carol) (mother a d) (mother d c) (father carol d)) ((grandparent alice carol)) ((0.0 0.0 1.0 0.0) (0.0) 0) ((alice 1) (carol 2)) (grandparent X Y))
    ((0.0 0.0 1.0 0.0) (1.0) 1))

!(test 
    (substitution (mother X Y) ((mother alice bob) (father bob carol) (mother carol dave) (father alice carol) (mother a d) (mother d c) (father carol d)) ((grandparent alice carol)) ((0.0 0.0 1.0 0.0) (0.0) 0) ((alice 1) (bob 2)) (grandparent X Y))
    ((1.0 0.0 1.0 0.0) (0.0) 1))


; ;; Test cases for updateTargetVar function with binary target atom
; (foldl-atom $postiveExamples ($X $Y) $comulative $example 
                        ; (updateTargetVar $targetAtom $example $comulative $traningFact))
!(test 
    (updateTargetVar (grandparent X Y) (grandparent alice carol) (() ()) ((mother alice bob) (father bob carol) (mother carol dave) (father alice carol) (mother a d) (mother d c) (father carol d)))
    (((alice 1)) ((carol 2))))

;; Test cases for updateTargetVar function with unary target atom  
!(test 
    (updateTargetVar (smart X) (smart alice) (() ()) ((mother alice bob) (father bob carol) (mother carol dave) (father alice carol) (mother a d) (mother d c) (father carol d)))
    (((alice 1)) ((alice 2) (bob 2) (carol 2) (dave 2) (a 2) (d 2) (c 2))))

;; Test cases for genInterpretationPair function
!(test 
    (genInterpretationPair ((alice 1) (bob 2)) ((mother X Y) (father X Y)) ((grandparent alice carol)) ((mother alice bob) (father bob carol) (mother carol dave) (father alice carol) (mother a d) (mother d c) (father carol d)) (0.0 0.0 0.0 0.0) 0.0 (grandparent X Y) (()()))
    (((1.0 0.0 0.0 0.0)) (0.0)))

;; Integration test for propositionalization function (simplified)
!(test 
    (propositionalization
        ()
        ()
        (3)
        (grandparent X Y)
        ((mother X A) (father A Y) (mother A Y) (father X A))
        ((grandparent alice carol) (grandparent a b) (grandparent b c) (grandparent f g))
        ((mother alice bob) (father bob carol) (mother x b) (father a x) (mother b y) (mother y c) (father f r) (father r g)))
        ((1.0 1.0 0.0 0.0) (0.0 0.0 1.0 0.0) (1.0 0.0 0.0 0.0) (0.0 0.0 1.0 0.0) (1.0 0.0 0.0 0.0) (0.0 1.0 0.0 0.0) (1.0 0.0 0.0 0.0) (0.0 0.0 0.0 1.0) (0.0 1.0 0.0 0.0) (0.0 0.0 1.0 1.0) (0.0 0.0 1.0 0.0) (0.0 0.0 0.0 1.0) (0.0 1.0 0.0 0.0) (0.0 0.0 0.0 1.0) (1.0 0.0 0.0 0.0) (0.0 1.0 0.0 0.0) (1.0 0.0 0.0 0.0) (0.0 0.0 1.0 0.0) (1.0 0.0 1.0 0.0) (0.0 1.0 0.0 0.0) (1.0 0.0 0.0 0.0) (0.0 0.0 0.0 1.0) (0.0 1.0 0.0 0.0) (0.0 0.0 0.0 1.0) (0.0 0.0 1.0 0.0) (0.0 0.0 0.0 1.0) (0.0 0.0 1.0 0.0) (0.0 1.0 0.0 1.0)) ((1.0) 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 (1.0) 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 (1.0) 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 (1.0)))

