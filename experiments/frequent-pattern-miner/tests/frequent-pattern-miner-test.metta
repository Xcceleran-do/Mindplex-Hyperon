!(import! &self ../../../PeTTa/lib/lib_import.metta)
!(import_prolog_functions_from_file
    "../experiments/frequent-pattern-miner/conj_exp.pl"
    (unique_combinations_star cut-first-char promote_engagement_conj))
!(import! &self ../../utils/common-utils)
!(import! &self ../../frequent-pattern-miner/frequent-pattern-miner)

!(bind! &space (new-space))

!(add-atom &space (topic 0 "agi"))
!(add-atom &space (audience-expertise 0 "Intermediate"))
!(add-atom &space (engagement 0 "High"))

!(add-atom &space (topic 1 "agi"))
!(add-atom &space (audience-expertise 1 "Intermediate"))
!(add-atom &space (engagement 1 "High"))

!(add-atom &space (topic 2 "agi"))
!(add-atom &space (audience-expertise 2 "Advanced"))
!(add-atom &space (engagement 2 "Low"))

!(add-atom &space (topic 3 "Gardening"))
!(add-atom &space (audience-expertise 3 "Advanced"))
!(add-atom &space (engagement 3 "Low"))

;; =============================================================================
;; TEST: formatter
;; Output shape: (supportOf (conjunction CTV) support-count)
;; ($organized $CTV) forms one 2-element expression as first arg to supportOf
;; =============================================================================

;; Valid conjunct meeting minSup=2
;; nAY=2, nA=2, N=12, nY=2
;; strength_pos=1.0, strength_neg=0.0
;; conf_pos=2/802≈0.002494, conf_neg=10/810≈0.012346
!(test
    (formatter
        (conjunct (, (topic $x "agi")
                     (audience-expertise $x "Intermediate")
                     (engagement $x "High")))
        &space 2 engagement "High")
    (supportOf
        ((, (topic $x "agi")
            (audience-expertise $x "Intermediate")
            (engagement $x "High"))
         (CTV (STV 1.0 0.0024937655860349127)
              (STV 0.0 0.012345679012345678)))
        2))

;; Below minSup=3 → ()
!(test
    (formatter
        (conjunct (, (topic $x "agi")
                     (audience-expertise $x "Intermediate")
                     (engagement $x "High")))
        &space 3 engagement "High")
    ())

;; Zero overlap → () even with minSup=1
!(test
    (formatter
        (conjunct (, (topic $x "Gardening")
                     (audience-expertise $x "Intermediate")
                     (engagement $x "High")))
        &space 1 engagement "High")
    ())

;; Non-engagement target predicate — topic as target
;; body = (, (audience-expertise $x "Intermediate") (engagement $x "High"))
;; nAY=2, nA=2, N=12, nY=3
;; n¬A=10, n¬A,Y=1
;; strength_pos=1.0, strength_neg=0.1
!(test
    (formatter
        (conjunct (, (topic $x "agi")
                     (audience-expertise $x "Intermediate")
                     (engagement $x "High")))
        &space 2 topic "agi")
    (supportOf
        ((, (topic $x "agi")
            (audience-expertise $x "Intermediate")
            (engagement $x "High"))
         (CTV (STV 1.0 0.0024937655860349127)
              (STV 0.1 0.012345679012345678)))
        2))


;; =============================================================================
;; TEST: frequency-pattern-miner end-to-end
;; Uses is-member-custom (already in common-utils) against superposed results.
;; frequency-pattern-miner returns results via superposition not a list,
;; so is-member-custom handles membership check correctly.
;; =============================================================================

;; Intermediate/High pattern present with explicit target pred
!(test
    (is-member-custom
        (supportOf
            ((, (audience-expertise $x "Intermediate")
                (engagement $x "High"))
             (CTV (STV 1.0 0.0024937655860349127)
                  (STV 0.0 0.012345679012345678)))
            2)
        (frequency-pattern-miner &space 2 2 engagement "High"))
    True)

;; Advanced/Low pattern also present
!(test
    (is-member-custom
        (supportOf
            ((, (audience-expertise $x "Advanced")
                (engagement $x "Low"))
             (CTV (STV 1.0 0.0024937655860349127)
                  (STV 0.0 0.012345679012345678)))
            2)
        (frequency-pattern-miner &space 2 2 engagement "High"))
    True)

;; Backward-compat overload — same results
!(test
    (is-member-custom
        (supportOf
            ((, (audience-expertise $x "Intermediate")
                (engagement $x "High"))
             (CTV (STV 1.0 0.0024937655860349127)
                  (STV 0.0 0.012345679012345678)))
            2)
        (frequency-pattern-miner &space 2 2))
    True)