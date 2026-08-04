!(pragma! import-root "../../..")
!(import! &self ../../../PeTTa/lib/lib_import.metta) 
!(import! &self ../../utils/common-utils)
!(import! &self ../../frequent-pattern-miner/frequent-pattern-miner)


; Test dataset: simple, deterministic supports
!(add-atom &space (topic 0 AI (STV 0.3 0.5)))
!(add-atom &space (engagement 0 low (STV 0.3 0.5)))

!(add-atom &space (topic 1 AI (STV 0.3 0.5)))
!(add-atom &space (engagement 1 low (STV 0.3 0.5)))

!(add-atom &space (topic 3 Gardening (STV 0.3 0.5)))
!(add-atom &space (engagement 3 low (STV 0.3 0.5)))

; With minsup=2 and depth=2, only (topic $V0 "AI") and (length $V0 "low") qualify.
; Their conjunction has support 2 (articles 0 and 1). The formatter sorts atoms lexicographically.
!(frequency-pattern-miner &space 2 2)
