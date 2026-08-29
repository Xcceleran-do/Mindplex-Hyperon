!(import! &self ../../../PeTTa/lib/lib_import.metta)
!(import_prolog_functions_from_file "../experiments/frequent-pattern-miner/conj_exp.pl"
    (unique_combinations_star cut-first-char promote_engagement_conj))
!(import! &self ../../utils/common-utils)
!(import! &self ../../frequent-pattern-miner/frequent-pattern-miner)

!(bind! &space (new-space))

;; Article 0: agi, Intermediate, High
!(add-atom &space ((topic 0 "agi") (STV 1.0 1.0)))
!(add-atom &space ((audience-expertise 0 "Intermediate") (STV 0.8 0.75)))
!(add-atom &space ((engagement 0 "High") (STV 0.728 0.9)))

;; Article 1: agi, Intermediate, High
!(add-atom &space ((topic 1 "agi") (STV 1.0 1.0)))
!(add-atom &space ((audience-expertise 1 "Intermediate") (STV 0.8 0.75)))
!(add-atom &space ((engagement 1 "High") (STV 0.728 0.9)))

;; Article 2: agi, Advanced, Low
!(add-atom &space ((topic 2 "agi") (STV 1.0 1.0)))
!(add-atom &space ((audience-expertise 2 "Advanced") (STV 0.8 0.75)))
!(add-atom &space ((engagement 2 "Low") (STV 0.14 0.9)))

;; Article 3: Gardening, Advanced, Low
!(add-atom &space ((topic 3 "Gardening") (STV 1.0 1.0)))
!(add-atom &space ((audience-expertise 3 "Advanced") (STV 0.8 0.75)))
!(add-atom &space ((engagement 3 "Low") (STV 0.14 0.9)))

;; =============================================================================
;; TEST: formatter
;; =============================================================================

;; Valid conjunct meeting minSup=2 — produces supportOf with CTV
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

;; Below minSup → empty
!(test
    (formatter
        (conjunct (, (topic $x "agi")
                     (audience-expertise $x "Intermediate")
                     (engagement $x "High")))
        &space 3 engagement "High")
    ())

;; Zero joint support → empty regardless of minSup
!(test
    (formatter
        (conjunct (, (topic $x "Gardening")
                     (audience-expertise $x "Intermediate")
                     (engagement $x "High")))
        &space 1 engagement "High")
    ())

;; Non-engagement target predicate — topic as target
;; body = (, (audience-expertise $x "Intermediate") (engagement $x "High"))
;; nAY=2(agi+Intermediate+High), nA=bodySupport strips topic:
;;   Intermediate+High = {0,1} = 2
;; N=12, nY=targetSupport topic "agi" = 3
;; n¬A=10, n¬A,Y=3-2=1
;; strength_pos=2/2=1.0, strength_neg=1/10=0.1
;; conf_pos=2/802≈0.002494, conf_neg=10/810≈0.012346
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
;; TEST: frequency-pattern-miner
;; With minSup=2, depth=2:
;; Qualifying 2-conjunct patterns (support≥2):
;;   - (, (topic $x "agi") (audience-expertise $x "Intermediate")) → support=2
;;   - (, (audience-expertise $x "Advanced") (engagement $x "Low")) → support=2
;; Non-qualifying (support<2):
;;   - (, (topic $x "agi") (engagement $x "Low")) → support=1
;;   - (, (topic $x "Gardening") ...) → support≤1
;; =============================================================================

;; End-to-end run — confirm it produces results without error
;; (exact output depends on conjunct_has_engagement filter in conj_exp.pl
;;  which requires both engagement AND audience-expertise in every conjunct)
!(frequency-pattern-miner &space 2 2 engagement "High")

;; Backward-compat overload produces same result
!(frequency-pattern-miner &space 2 2)