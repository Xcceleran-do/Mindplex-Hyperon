!(register-module! ../../Utilities)
!(register-module! ../../DFOL)
!(import! &self Utilities:helperFunctions)
!(import! &self DFOL:kwargsme)
!(import! &self DFOL:parsing_exceptions)
!(import! &self DFOL:torchme)
!(import! &self DFOL:dfol)
!(import! &self DFOL:pythonDFOL)

; Test data setup - mirrors the example data from dfol.metta
!(bind! &test_dict ((1 alice) (2 bob) (3 carol) (4 dave) (7 eve)))
!(bind! &test_training_facts ((mother alice bob) (father bob carol) (mother carol dave) (father alice carol) (mother a d) (mother d c) (father carol d)))
!(bind! &test_positive_examples ((grandparent alice carol)))
!(bind! &test_feature_set ((mother X A) (mother A Y) (father X Y) (father X Y)))
!(bind! &test_var_list (alice bob carol))


;; Test cases for addToVarDom function
!(assertEqual 
    (addToVarDom 1 &test_training_facts) 
    ((bob 1) (alice 1) (carol 1) (dave 1) (d 1) (a 1) (c 1))) ; The order is non-determined

!(assertEqual 
    (addToVarDom 2 &test_training_facts) 
    ((bob 2) (alice 2) (carol 2) (dave 2) (d 2) (a 2) (c 2))) ; The order is non-determined


;; Test cases for getId function (uses global dict)
!(assertEqual (getId X) 1)
!(assertEqual (getId Y) 2)
!(assertEqual (getId A) 3)
!(assertEqual (getId B) 4)
!(assertEqual (getId Z) 7)
!(assertEqual (getId unknown) Empty)

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
    (substitution (mother X Y) &test_training_facts &test_positive_examples (&test_vi &test_vo &test_idx) &test_cursub_match (grandparent X Y))
    ((0.0 0.0 1.0 0.0) (1.0) 1))

!(assertEqual 
    (substitution (mother X Y) &test_training_facts &test_positive_examples (&test_vi &test_vo &test_idx) &test_cursub (grandparent X Y))
    ((1.0 0.0 1.0 0.0) (0.0) 1))


; ;; Test cases for updateTargetVar function with binary target atom
; (foldl-atom $postiveExamples ($X $Y) $comulative $example 
                        ; (updateTargetVar $targetAtom $example $comulative $traningFact))
!(assertEqual 
    (updateTargetVar (grandparent X Y) (grandparent alice carol) (() ()) &test_training_facts)
    (((alice 1)) ((carol 2))))

;; Test cases for updateTargetVar function with unary target atom  
!(assertEqual 
    (updateTargetVar (smart X) (smart alice) (() ()) &test_training_facts)
    (((alice 1)) ((bob 2) (alice 2) (carol 2) (dave 2) (d 2) (a 2) (c 2))))

;; Test cases for genInterpretationPair function
!(bind! &empty_vi (0.0 0.0 0.0 0.0))
!(bind! &empty_vo 0.0)
!(bind! &feature_set_small ((mother X Y) (father X Y)))
!(bind! &test_sub ((alice 1) (bob 2)))
!(assertEqual 
    (genInterpretationPair &test_sub &feature_set_small &test_positive_examples &test_training_facts &empty_vi &empty_vo (grandparent X Y) (()()))
    (((1.0 0.0 0.0 0.0)) (0.0)))

;; Integration test for propositionalization function (simplified)
!(assertEqual 
    (propositionalization
        ()
        ()
        (3)
        (grandparent X Y)
        ((mother X A) (father A Y))
        ((grandparent alice carol))
        ((mother alice bob) (father bob carol)))
        
        (((1.0 1.0)) ((1.0))))


;; Test case for the whole pipline

!(bind! &C_val 4)

!(bind! &t_val 2)

!(bind! &V_size_val 3)

!(bind! &m1_val 2)

!(bind! &m2_val 1)

!(bind! &na_val 2)

!(let ($vi $vo) (propositionalization
    ()
    ()
    (3) 
    (grandparent X Y)
    ((father X A) (mother A Y) (mother X A) (father A Y))
    ((grandparent "alica" "bob") (grandparent "john" "lol"))
    (
    (mother "alica" "dani")
    (father "dani" "bob")
    (mother "john" "beki")
    (father "beki" "lol")   
    )

 ) (let () (bind! &vi_data_val (torch:tensor $vi)) (bind! &vo_data_val (torch:tensor $vo))))
 

!(let $b (map-atom ((father X A) (mother A Y) (mother X A) (father A Y)) $cur (basicEmbedding $cur (grandparent X Y))) (bind! &b_embeddings_val (torch:tensor $b)))

!(let $o (map-atom ((father X A) (mother A Y) (mother X A) (father A Y)) $cur (occurrenceEmbedding (A) $cur)) (bind! &o_embeddings_val (torch:tensor $o)))


!(bind! &a_val 1.0)

!(bind! &b_val 1.0)

!(bind! &c_val 10.0)

!(bind! &d_val 1.0)

!(println! ("vi after prepostionalization:" &vi_data_val))
!(println! ("vo after prepostionalization:" &vo_data_val))
!(println! ("basic embeddings:" &b_embeddings_val))
!(println! ("occurrence embeddings:" &o_embeddings_val))

!(bind! &trainer (torch:instantiate_module DFOLTrainer pythonDFOL &vi_data_val &vo_data_val &C_val &m1_val &m2_val &na_val &V_size_val &t_val 
                    &b_embeddings_val &o_embeddings_val &a_val &b_val &c_val &d_val))
                    
!(bind! &final (py-dot &trainer train))
!(&final) 