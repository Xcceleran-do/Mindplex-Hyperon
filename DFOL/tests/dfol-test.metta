
!(import! &self ../../PeTTa/lib/lib_he)
!(import! &self ../../Utilities/helperFunctions)
!(import! &self ../../DFOL/dfol)

;; Test cases for addToVarDom function
!(assertEqual 
    (addToVarDom 1 ((mother alice bob) (father bob carol) (mother carol dave) (father alice carol) (mother a d) (mother d c) (father carol d))) 
    ((alice 1) (bob 1) (carol 1) (dave 1) (a 1) (d 1) (c 1))) ; The order is non-determined

!(assertEqual 
    (addToVarDom 2 ((mother alice bob) (father bob carol) (mother carol dave) (father alice carol) (mother a d) (mother d c) (father carol d))) 
    ((alice 2) (bob 2) (carol 2) (dave 2) (a 2) (d 2) (c 2))) ; The order is non-determined


;; Test cases for getId function (uses global dict)
!(assertEqual (getId X) 1)
!(assertEqual (getId Y) 2)
!(assertEqual (getId A) 3)
!(assertEqual (getId B) 4)
!(assertEqual (getId Z) 7)
!(assertEqual (getId unknown) (empty))

;; Test cases for getSub function
!(assertEqual (getSub 1 ((alice 1) (bob 2) (carol 3))) alice)
!(assertEqual (getSub 2 ((alice 1) (bob 2) (carol 3))) bob)
!(assertEqual (getSub 3 ((alice 1) (bob 2) (carol 3))) carol)

;; Test cases for ground function with binary predicates 
!(assertEqual 
    (ground (mother X Y) ((alice 1) (bob 2)))
    (mother alice bob))

!(assertEqual 
    (ground (father X A) ((bob 1) (carol 3)))
    (father bob carol))

;; Test cases for ground function with unary predicates 
!(assertEqual 
    (ground (loves X) ((alice 1)))
    (loves alice))

!(assertEqual 
    (ground (smart A) ((carol 3)))
    (smart carol))

;; Test cases for basicEmbedding function with unary predicates 
!(assertEqual 
    (basicEmbedding (grandparent X) (mother Y))
    0)

!(assertEqual 
    (basicEmbedding (loves X) (hate X))
    1)

;; Test cases for basicEmbedding function with binary predicates
!(assertEqual 
    (basicEmbedding (grandparent X Y) (mother X Z))
    (1 0))

!(assertEqual 
    (basicEmbedding (grandparent X Y) (mother Z Y))
    (0 1))

!(assertEqual 
    (basicEmbedding (grandparent X Y) (father X Y))
    (1 1))

!(assertEqual 
    (basicEmbedding (loves A B) (hates C D))
    (0 0))

;; Test cases for occurrenceEmbedding function
!(assertEqual 
    (occurrenceEmbedding (Z) (mother X Z))
    (1))

!(assertEqual 
    (occurrenceEmbedding (A B) (father A B))
    (1 1))

!(assertEqual 
    (occurrenceEmbedding (Z) (loves X Y))
    (0))

!(assertEqual 
    (occurrenceEmbedding (A B C) (mother A B))
    (1 1 0))

;; Test cases for substitution function
!(bind! &test_vi (0.0 0.0 1.0 0.0))
!(bind! &test_vo (0.0))
!(bind! &test_idx 0)
!(bind! &test_cursub ((alice 1) (bob 2)))
!(bind! &test_cursub_match ((alice 1) (carol 2)))

;; Test substitution with positive example match
!(assertEqual 
    (substitution (mother X Y) ((mother alice bob) (father bob carol) (mother carol dave) (father alice carol) (mother a d) (mother d c) (father carol d)) ((grandparent alice carol)) ((0.0 0.0 1.0 0.0) (0.0) 0) ((alice 1) (carol 2)) (grandparent X Y))
    ((0.0 0.0 1.0 0.0) (1.0) 1))

!(assertEqual 
    (substitution (mother X Y) ((mother alice bob) (father bob carol) (mother carol dave) (father alice carol) (mother a d) (mother d c) (father carol d)) ((grandparent alice carol)) ((0.0 0.0 1.0 0.0) (0.0) 0) ((alice 1) (bob 2)) (grandparent X Y))
    ((1.0 0.0 1.0 0.0) (0.0) 1))


; ;; Test cases for updateTargetVar function with binary target atom
; (foldl-atom $postiveExamples ($X $Y) $comulative $example 
                        ; (updateTargetVar $targetAtom $example $comulative $traningFact))
!(assertEqual 
    (updateTargetVar (grandparent X Y) (grandparent alice carol) (() ()) ((mother alice bob) (father bob carol) (mother carol dave) (father alice carol) (mother a d) (mother d c) (father carol d)))
    (((alice 1)) ((carol 2))))

;; Test cases for updateTargetVar function with unary target atom  
!(assertEqual 
    (updateTargetVar (smart X) (smart alice) (() ()) ((mother alice bob) (father bob carol) (mother carol dave) (father alice carol) (mother a d) (mother d c) (father carol d)))
    (((alice 1)) ((alice 2) (bob 2) (carol 2) (dave 2) (a 2) (d 2) (c 2))))

;; Test cases for genInterpretationPair function
!(bind! &empty_vi (0.0 0.0 0.0 0.0))
!(bind! &empty_vo 0.0)
!(bind! &feature_set_small ((mother X Y) (father X Y)))
!(bind! &test_sub ((alice 1) (bob 2)))
!(assertEqual 
    (genInterpretationPair &test_sub &feature_set_small ((grandparent alice carol)) ((mother alice bob) (father bob carol) (mother carol dave) (father alice carol) (mother a d) (mother d c) (father carol d)) &empty_vi &empty_vo (grandparent X Y) (()()))
    (((1.0 0.0 0.0 0.0)) (0.0)))

;; Integration test for propositionalization function (simplified)
!(assertEqual 
    (propositionalization
        ()
        ()
        (3)
        (grandparent X Y)
        ((mother X A) (father A Y) (mother A Y) (father X A))
        ((grandparent alice carol) (grandparent a b) (grandparent b c) (grandparent f g))
        ((mother alice bob) (father bob carol) (mother x b) (father a x) (mother b y) (mother y c) (father f r) (father r g)))
        ((1.0 1.0 0.0 0.0) (0.0 0.0 1.0 0.0) (1.0 0.0 0.0 0.0) (0.0 0.0 1.0 0.0) (1.0 0.0 0.0 0.0) (0.0 1.0 0.0 0.0) (1.0 0.0 0.0 0.0) (0.0 0.0 0.0 1.0) (0.0 1.0 0.0 0.0) (0.0 0.0 1.0 1.0) (0.0 0.0 1.0 0.0) (0.0 0.0 0.0 1.0) (0.0 1.0 0.0 0.0) (0.0 0.0 0.0 1.0) (1.0 0.0 0.0 0.0) (0.0 1.0 0.0 0.0) (1.0 0.0 0.0 0.0) (0.0 0.0 1.0 0.0) (1.0 0.0 1.0 0.0) (0.0 1.0 0.0 0.0) (1.0 0.0 0.0 0.0) (0.0 0.0 0.0 1.0) (0.0 1.0 0.0 0.0) (0.0 0.0 0.0 1.0) (0.0 0.0 1.0 0.0) (0.0 0.0 0.0 1.0) (0.0 0.0 1.0 0.0) (0.0 1.0 0.0 1.0)) ((1.0) 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 (1.0) 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 (1.0) 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 (1.0)))

