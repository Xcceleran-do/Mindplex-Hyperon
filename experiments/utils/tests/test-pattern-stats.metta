;; =============================================================================
;; test_pattern_stats.metta
;; Integration tests using real data.metta as the source database.
;; Tests non-engagement consequents to prove generalization.
;; common-utils.metta directly.
;; =============================================================================

!(import! &self ../common-utils)
!(import! &tempo ../../atomspace_visualizer/public/data)
!(bind! &test-db (new-space))
!(let $atom (get-atoms &tempo)
    (let $clause (car-atom $atom)
        (add-atom &test-db $clause)))

;; TEST: targetSupport — NON-ENGAGEMENT consequent (topic)
!(test (targetSupport topic "agi" &test-db) 2)
!(test (targetSupport content-type "text" &test-db) 5)
!(test (targetSupport topic "ethics" &test-db) 2)


;; TEST: bodySupport — NON-ENGAGEMENT consequent
!(test
    (bodySupport topic
        (, (content-type $x "text")
           (topic $x "agi"))
        &test-db)
    5)

!(test
    (bodySupport topic
        (, (engagement $x "High")
           (content-type $x "text")
           (topic $x "agi"))
        &test-db)
    2)

;; TEST: jointSupport
;; ethics AND High engagement → only A_3554 → 1
!(test
    (jointSupport
        (, (topic $x "ethics")
           (engagement $x "High"))
        &test-db)
    1)

;; agi AND Medium engagement → A_2786, A_1549 → 2
!(test
    (jointSupport
        (, (topic $x "agi")
           (engagement $x "Medium"))
        &test-db)
    2)

;; agi AND High engagement → no overlap → 0
!(test
    (jointSupport
        (, (topic $x "agi")
           (engagement $x "High"))
        &test-db)
    0)

;; Single clause — no conjunction wrapper
!(test
    (jointSupport
        (engagement $x "High")
        &test-db)
    2)

;; Three-way conjunction: ethics AND text AND High → A_3554 only → 1
!(test
    (jointSupport
        (, (topic $x "ethics")
           (content-type $x "text")
           (engagement $x "High"))
        &test-db)
    1)

;; TEST: pattern-stats — NON-ENGAGEMENT consequent (topic)
;; (nAY nA N nY)

!(test
    (pattern-stats topic "agi"
        (, (content-type $x "text")
           (topic $x "agi"))
        &test-db)
    (2 5 96 2))

!(test
    (pattern-stats topic "ethics"
        (, (engagement $x "High")
           (topic $x "ethics"))
        &test-db)
    (1 2 96 2))


;; TEST: antecedent generalization
!(test
    (antecedent topic
        (, (content-type $x "text")
           (engagement $x "High")
           (topic $x "agi")))
    (, (content-type $x "text") (engagement $x "High")))

!(test
    (antecedent content-type
        (, (topic $x "agi")
           (content-type $x "text")
           (engagement $x "High")))
    (, (topic $x "agi") (engagement $x "High")))

!(test
    (antecedent
        (, (topic $x "agi")
           (content-type $x "text")
           (engagement $x "High")))
    (, (topic $x "agi") (content-type $x "text")))

;; TEST: target-clause? — non-engagement predicates
!(test (target-clause? topic (topic $x "agi")) True)
!(test (target-clause? content-type (content-type $x "text")) True)
!(test (target-clause? topic (engagement $x "High")) False)
!(test (target-clause? content-type (topic $x "agi")) False)


;; =============================================================================
;; TEST: compute-ctv — paper's reference numbers (arithmetic only)
;; nAY=3600, nA=4800, N=16800, nY=9000
;; =============================================================================
!(test
    (compute-ctv (3600 4800 16800 9000))
    (CTV (STV 0.75 0.8571428571428571)
         (STV 0.45 0.9375)))
(= (data-ctv) (let $stats
    (pattern-stats topic "agi"
        (, (engagement $x "High")
           (content-type $x "text")
           (topic $x "agi"))
        &test-db)
    (trace! (pattern-stats-real $stats) $stats)))

!(test
    (compute-ctv (data-ctv))
    (CTV (STV 0.0 0.0024937655860349127)
         (STV 0.02127659574468085 0.10514541387024609)))

!(test
    (compute-ctv (0 0 96 2))
    (CTV (STV 0.0 0.0)
         (STV 0.020833333333333332 0.10714285714285714)))
         
!(test
    (compute-ctv (2 96 96 2))
    (CTV (STV 0.020833333333333332 0.10714285714285714)
         (STV 0.0 0.0)))