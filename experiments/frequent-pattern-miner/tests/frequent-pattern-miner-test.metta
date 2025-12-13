!(import! &self ../../frequent-pattern-miner/frequent-pattern-miner)
!(import! &self ../../frequent-pattern-miner/conj-exp/main)
!(import! &self ../../frequent-pattern-miner/freq-pat/main)
!(import! &self ../../utils/common-utils)


!(bind! &space (new-space))
; !(add-reduct &self (= (get-space) (new-space)))

; Test dataset: simple, deterministic supports
!(add-atom &space (topic 0 "AI"))
!(add-atom &space (length 0 "low"))

!(add-atom &space (topic 1 "AI"))
!(add-atom &space (length 1 "low"))

!(add-atom &space (topic 2 "AI"))
!(add-atom &space (length 2 "high"))

!(add-atom &space (topic 3 "Gardening"))
!(add-atom &space (length 3 "low"))

; With minsup=2 and depth=2, only (topic $V0 "AI") and (length $V0 "low") qualify.
; Their conjunction has support 2 (articles 0 and 1). The formatter sorts atoms lexicographically.
!(test
  (frequency-pattern-miner &space 2 2)
  ((supportOf (, (length $V0 "low") (topic $V0 "AI")) 2))
)
