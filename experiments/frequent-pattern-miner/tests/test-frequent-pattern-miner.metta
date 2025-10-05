! (register-module! ../../../experiments)
! (import! &self experiments:frequent-pattern-miner:frequent-pattern-miner)
! (import! &self experiments:frequent-pattern-miner:conj-exp)
! (import! &self experiments:frequent-pattern-miner:freq-pat)
! (import! &self experiments:utils:common-utils)

! (add-reduct &self (= (get-space) (new-space)))

; Test dataset: simple, deterministic supports
! (add-atom (get-space) (topic 0 "AI"))
! (add-atom (get-space) (length 0 "low"))

! (add-atom (get-space) (topic 1 "AI"))
! (add-atom (get-space) (length 1 "low"))

! (add-atom (get-space) (topic 2 "AI"))
! (add-atom (get-space) (length 2 "high"))

! (add-atom (get-space) (topic 3 "Gardening"))
! (add-atom (get-space) (length 3 "low"))

; With minsup=2 and depth=2, only (topic $V0 "AI") and (length $V0 "low") qualify.
; Their conjunction has support 2 (articles 0 and 1). The formatter sorts atoms lexicographically.
!(assertEqual
  (frequency-pattern-miner (get-space) 2 2)
  ((supportOf (, (length $V0 "low") (topic $V0 "AI")) 2))
)
