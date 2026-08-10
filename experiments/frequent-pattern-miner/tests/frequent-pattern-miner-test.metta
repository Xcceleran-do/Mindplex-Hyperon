!(import! &self ../../../PeTTa/lib/lib_import.metta)
!(import! &self ../../utils/common-utils)
!(import! &self ../../frequent-pattern-miner/frequent-pattern-miner)


; Test dataset: simple, deterministic supports
; Updated to include 'audience-expertise' so that combinations pass the REQUIRED_KEYWORDS filter
; in the TS runner (which strictly requires BOTH 'engagement' and 'audience-expertise').

; Entity 0
!(add-atom &space (topic 0 AI (STV 0.3 0.5)))
!(add-atom &space (engagement 0 low (STV 0.3 0.5)))
!(add-atom &space (audience-expertise 0 beginner (STV 0.3 0.5)))

; Entity 1
!(add-atom &space (topic 1 AI (STV 0.3 0.5)))
!(add-atom &space (engagement 1 low (STV 0.3 0.5)))
!(add-atom &space (audience-expertise 1 beginner (STV 0.3 0.5)))

; Entity 3
!(add-atom &space (topic 3 Gardening (STV 0.3 0.5)))
!(add-atom &space (engagement 3 low (STV 0.3 0.5)))
!(add-atom &space (audience-expertise 3 beginner (STV 0.3 0.5)))

; With minsup=2 and depth=2, combinations that include both engagement and audience-expertise
; will now correctly qualify and be returned by the miner.
!(frequency-pattern-miner &space 2 2)
